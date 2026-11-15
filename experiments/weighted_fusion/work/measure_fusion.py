#!/usr/bin/env python3
"""Step 2: RRF廃止 → Min-Max正規化＋非対称な重み付き和（別AIレポート提案①）。
- 土俵: fifty 50問・原文TF-IDF top-30プール。
- 言葉数え側: 最良スタック（原文＋単語タグ＋メタデータ連結、countermeasures 対策2(a) と同一作法）の生スコア。
  参照として baseline（原文のみ）の生スコアも併走（旧RRFと同一入力での比較用）。
- 意味側: (a) multilingual-e5-small（vector_hybrid の doc 埋め込み資産を再利用、質問埋め込みは再計算）
         (b) ruri-v3-30m（コミュニティONNX輸出 lc-studio/ruri-v3-30m-onnx、fp32、平均プーリング、
             接頭辞 検索クエリ:/検索文書:、1024トークン分割・重なり48・チャンク平均）
  どちらも条文側の材料は「原文」（vector_hybrid 条件a と同一）。
- 合成: プール内 Min-Max 正規化 → S = α·lex + (1-α)·sem、α ∈ {0.80, 0.85, 0.90, 0.95} 全パターン。
- 測定前に baseline (29/3.88)、連結スタック (31/3.38)、e5単独順位 (results_fifty 条件a) の再現を全問照合する。
"""
import json, math, re, statistics, sys, time
from pathlib import Path
import numpy as np

RK = Path("/home/hatch/workspace/research_kudou")
WF = RK / "weighted_fusion"
FM = RK / "fifty_measurement"
CM = RK / "countermeasures"
CFULL = RK / "prototype/c_full"
LAWAPP = Path("/home/hatch/workspace/lawapp")
sys.path.insert(0, str(CFULL / "work"))
sys.path.insert(0, str(RK / "vector_hybrid/work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
pool_texts = json.load(open(RK / "question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d
gov = {}
for line in open(CFULL / "governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d
ref = json.load(open(FM / "results50.json", encoding="utf-8"))["questions"]
cm_ref = json.load(open(CM / "results_countermeasures.json", encoding="utf-8"))["questions"]
vh_ref = json.load(open(RK / "vector_hybrid/results_fifty.json", encoding="utf-8"))["questions"]

tags = {}
for line in open(RK / "tag_vector/word_tags.jsonl", encoding="utf-8"):
    d = json.loads(line)
    tags[(d["law_id"], d["art"])] = d["tags"]

QIDS = list(pools.keys())
ALL_KEYS = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})


def qtext(qid):
    q = qs[qid]
    return q["stem"] + " " + " ".join(q["choices"])


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


def target_of(qid):
    g = gov[qid]["governing"]
    return (g["law_id"], g["art"])


def score_docs(raw, docs):
    """measure50.py の score_docs の複製。"""
    extra = [c for a, c in E.SYN.items() if a in raw and c not in raw]
    qg = E.gram_counts(E.qnorm(raw + "".join(extra)))
    qterms = [t for t in E.TERMS if t in raw or t in extra]
    out = {}
    for doc in docs:
        sc = 0.0
        for g, c in qg.items():
            w = IDF.get(g)
            if w is None:
                continue
            tf = doc["grams"].get(g, 0)
            if tf:
                sc += w * min(c, 3) * (1 + math.log(tf))
            ctf = doc["cap_grams"].get(g, 0)
            if ctf:
                sc += w * 2.2 * (1 + math.log(ctf))
        for t in qterms:
            if t in doc["full"]:
                w = eng._term_w(t)
                sc += w * 1.6
                if t in doc["cap"]:
                    sc += w * 1.1
        sc = sc / (max(doc["total"], 1) / AVG) ** 0.3
        out[(doc["law"], doc["art"])] = sc
    return out


def rank_from_scores(scores, keys, target):
    ordered = sorted(keys, key=lambda k: (-scores.get(k, 0.0), keys.index(k)))
    return ordered.index(target) + 1


def agg(ranks):
    """ranks: {qid: rank or None(out of pool)} → (top3/50, 平均順位 in-pool)"""
    top3 = sum(1 for r in ranks.values() if r is not None and r <= 3)
    inp = [r for r in ranks.values() if r is not None]
    return top3, round(statistics.mean(inp), 2)


# ---------- 1. 言葉数え側: baseline & 連結スタック（countermeasures と同一作法） ----------
base_scores, stack_scores = {}, {}
for qid in QIDS:
    keys = pool_keys(qid)
    base_scores[qid] = score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys])

mm = []
for qid in QIDS:
    t = target_of(qid)
    if t in pool_keys(qid):
        r = rank_from_scores(base_scores[qid], pool_keys(qid), t)
        if r != ref[qid]["rank_original"]:
            mm.append((qid, r, ref[qid]["rank_original"]))
print(f"baseline照合: 不一致 {len(mm)} 件")
assert not mm

_lawidx_text = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
_law_index = json.loads(_lawidx_text[_lawidx_text.index("["):_lawidx_text.rindex("]") + 1])
chapter_map = {}
for m in _law_index:
    d = json.loads((LAWAPP / f"data/{m['id']}.json").read_text(encoding="utf-8"))
    for a in d["articles"]:
        chapter_map[(m["id"], a["key"].replace("_", "-"))] = {
            "chapter": a.get("chapter") or "", "section": a.get("section") or ""}

PROC_WORDS = ["届出", "許可", "認可", "認定", "承認", "確認申請", "完了検査", "中間検査",
              "定期報告", "検査", "報告", "登録", "免許", "指定", "命令", "公告", "申請", "通知", "同意"]
ACTOR_WORDS = ["建築主", "建築主事", "特定行政庁", "都道府県知事", "市町村長", "国土交通大臣",
               "建築士", "設計者", "工事監理者", "工事施工者", "所有者", "管理者", "占有者",
               "指定確認検査機関", "消防長", "消防署長"]
ACT_WORDS = ["新築", "増築", "改築", "移転", "大規模の修繕", "大規模の模様替", "用途変更", "除却"]
USE_WORDS = ["共同住宅", "長屋", "寄宿舎", "劇場", "映画館", "集会場", "病院", "診療所", "学校",
             "保育所", "老人ホーム", "ホテル", "旅館", "飲食店", "物販店舗", "百貨店", "展示場",
             "事務所", "工場", "倉庫", "自動車車庫", "駐車場", "遊技場", "公衆浴場", "寺院",
             "教会", "幼稚園", "図書館", "博物館", "美術館", "体育館", "葬祭場", "火葬場"]
FINGERPRINT = PROC_WORDS + ACTOR_WORDS + ACT_WORDS + USE_WORDS

UNITS = [("平方メートル", "㎡", 1.0), ("㎡", "㎡", 1.0), ("m²", "㎡", 1.0), ("m2", "㎡", 1.0),
         ("キロメートル", "m", 1000.0), ("センチメートル", "m", 0.01), ("ミリメートル", "m", 0.001),
         ("メートル", "m", 1.0), ("時間", "分", 60.0), ("週間", "日", 7.0), ("週", "日", 7.0),
         ("秒", "分", 1.0 / 60), ("分", "分", 1.0), ("日", "日", 1.0), ("月", "月", 1.0), ("年", "年", 1.0),
         ("階", "階", 1.0), ("人", "人", 1.0), ("台", "台", 1.0), ("本", "本", 1.0), ("室", "室", 1.0),
         ("戸", "戸", 1.0), ("棟", "棟", 1.0), ("％", "%", 1.0), ("パーセント", "%", 1.0), ("m", "m", 1.0)]
NUMSPAN = re.compile(r"([0-9０-９]+(?:\.[0-9０-９]+)?|[一二三四五六七八九十百千万]+)")
KD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
FW = str.maketrans("０１２３４５６７８９", "0123456789")


def _seg_val(s):
    val, num = 0, 0
    for ch in s:
        if ch in KD:
            num = KD[ch]
        elif ch == "十":
            val += (num or 1) * 10; num = 0
        elif ch == "百":
            val += (num or 1) * 100; num = 0
        elif ch == "千":
            val += (num or 1) * 1000; num = 0
    return val + num


def _kanji_val(s):
    if "万" in s:
        head, rest = s.split("万", 1)
        return (_seg_val(head) if head else 1) * 10000 + _seg_val(rest)
    return _seg_val(s)


def extract_numbers(text):
    out = []
    for m in NUMSPAN.finditer(text):
        span = m.group(1)
        rest = text[m.end():m.end() + 6]
        hit = None
        for upat, canon, fac in UNITS:
            if rest.startswith(upat):
                hit = (canon, fac)
                break
        if not hit:
            continue
        if span[0] in "0123456789０１２３４５６７８９":
            v = float(span.translate(FW))
        else:
            v = float(_kanji_val(span))
        out.append((round(v * hit[1], 6), hit[0]))
    return out


def tag_text(key):
    return E.qnorm(" ".join(tags[key]))


fulltags_docs, metafull_docs = {}, {}
meta_strs = {}
for key in ALL_KEYS:
    base = eng.docs[eng.by_key[key]]
    ffull = base["full"] + tag_text(key)
    fg = E.gram_counts(ffull)
    fulltags_docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": ffull,
                          "grams": fg, "total": sum(fg.values()), "cap_grams": base["cap_grams"]}
    ch = chapter_map.get(key, {"chapter": "", "section": ""})
    finger = [w for w in FINGERPRINT if E.qnorm(w) in base["full"]]
    nums = sorted(set(extract_numbers(base["full"])))
    tokens = [f"{v:g}{u}" for v, u in nums]
    meta_strs[key] = (" ".join(x for x in [ch["chapter"], ch["section"]] if x)
                      + " " + " ".join(finger) + " " + " ".join(tokens)).strip()
for key in ALL_KEYS:
    ft = fulltags_docs[key]
    mfull = ft["full"] + E.qnorm(meta_strs[key])
    mg = E.gram_counts(mfull)
    metafull_docs[key] = {"law": key[0], "art": key[1], "cap": ft["cap"], "full": mfull,
                          "grams": mg, "total": sum(mg.values()), "cap_grams": ft["cap_grams"]}

for qid in QIDS:
    keys = pool_keys(qid)
    stack_scores[qid] = score_docs(qtext(qid), [metafull_docs[k] for k in keys])

mm = []
for qid in QIDS:
    t = target_of(qid)
    if t in pool_keys(qid):
        r = rank_from_scores(stack_scores[qid], pool_keys(qid), t)
        if r != cm_ref[qid]["ranks"]["meta_append"]:
            mm.append((qid, r, cm_ref[qid]["ranks"]["meta_append"]))
print(f"連結スタック照合: 不一致 {len(mm)} 件")
assert not mm

# ---------- 2. 意味側 (a) e5-small（doc 埋め込み再利用） ----------
from embedder import E5
doc_emb = np.load(RK / "vector_hybrid/work/doc_emb_fifty_a.npy")
assert doc_emb.shape[0] == len(ALL_KEYS), doc_emb.shape
emb_map = {k: doc_emb[i] for i, k in enumerate(ALL_KEYS)}
e5 = E5()
e5_scores = {}
for qid in QIDS:
    qe = e5.embed_texts([qtext(qid)], "query: ")[0]
    e5_scores[qid] = {k: float(emb_map[k] @ qe) for k in pool_keys(qid)}

mm = []
for qid in QIDS:
    t = target_of(qid)
    if t in pool_keys(qid):
        r = rank_from_scores(e5_scores[qid], pool_keys(qid), t)
        if r != vh_ref[qid]["conditions"]["a"]["rank_vector"]:
            mm.append((qid, r, vh_ref[qid]["conditions"]["a"]["rank_vector"]))
print(f"e5単独順位照合: 不一致 {len(mm)} 件 {mm[:5]}")
assert not mm

# ---------- 3. 意味側 (b) ruri-v3-30m（ONNX） ----------
import onnxruntime as ort
from tokenizers import Tokenizer

RURI_DIR = WF / "work/ruri_model"
RMAXLEN, ROVERLAP = 1024, 48


class Ruri:
    def __init__(self, threads=4):
        self.tok = Tokenizer.from_file(str(RURI_DIR / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(RURI_DIR / "model.onnx"),
                                         sess_options=so, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def _encode_chunk(self, ids_list):
        maxlen = max(len(x) for x in ids_list)
        input_ids = np.zeros((len(ids_list), maxlen), dtype=np.int64)
        attn = np.zeros((len(ids_list), maxlen), dtype=np.int64)
        for i, ids in enumerate(ids_list):
            input_ids[i, :len(ids)] = ids
            attn[i, :len(ids)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        out = self.sess.run(None, feeds)[0]
        mask = attn[:, :, None].astype(np.float32)
        emb = (out * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb.astype(np.float32)

    def embed_texts(self, texts, prefix, batch_chunks=32):
        all_chunks = []
        for ti, t in enumerate(texts):
            ids = self.tok.encode(prefix + t).ids
            if len(ids) <= RMAXLEN:
                all_chunks.append((ti, ids))
            else:
                step = RMAXLEN - ROVERLAP
                start = 0
                while start < len(ids):
                    all_chunks.append((ti, ids[start:start + RMAXLEN]))
                    if start + RMAXLEN >= len(ids):
                        break
                    start += step
        chunk_embs = []
        for i in range(0, len(all_chunks), batch_chunks):
            part = all_chunks[i:i + batch_chunks]
            chunk_embs.extend(self._encode_chunk([c[1] for c in part]))
        agg = {}
        for (ti, _), e in zip(all_chunks, chunk_embs):
            agg.setdefault(ti, []).append(e)
        embs = []
        for ti in range(len(texts)):
            v = np.mean(agg[ti], axis=0)
            embs.append(v / max(np.linalg.norm(v), 1e-12))
        return np.stack(embs).astype(np.float32)


orig_text = {tuple(k.split("|")): v["text"] for k, v in pool_texts.items()}
ruri = Ruri()
t0 = time.time()
ruri_doc_emb = ruri.embed_texts([orig_text[k] for k in ALL_KEYS], "検索文書: ")
np.save(WF / "work/doc_emb_ruri_fifty.npy", ruri_doc_emb)
print(f"ruri doc embedding: {time.time()-t0:.1f}s, shape={ruri_doc_emb.shape}", flush=True)
ruri_map = {k: ruri_doc_emb[i] for i, k in enumerate(ALL_KEYS)}
ruri_scores = {}
for qid in QIDS:
    qe = ruri.embed_texts([qtext(qid)], "検索クエリ: ")[0]
    ruri_scores[qid] = {k: float(ruri_map[k] @ qe) for k in pool_keys(qid)}

# ---------- 4. 合成（Min-Max 正規化＋重み付き和） ----------
ALPHAS = [0.80, 0.85, 0.90, 0.95]


def minmax(scores, keys):
    vals = [scores.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in keys}
    return {k: (scores.get(k, 0.0) - lo) / (hi - lo) for k in keys}


def fuse(lex, sem, alpha):
    return {k: alpha * lex[k] + (1 - alpha) * sem[k] for k in lex}


def ranks_of(score_map):
    out = {}
    for qid in QIDS:
        t = target_of(qid)
        keys = pool_keys(qid)
        out[qid] = rank_from_scores(score_map[qid], keys, t) if t in keys else None
    return out


base_ranks = ranks_of(base_scores)
stack_ranks = ranks_of(stack_scores)
e5_ranks = ranks_of(e5_scores)
ruri_ranks = ranks_of(ruri_scores)

results = {"meta": {
    "task": "Step 2: Min-Max正規化＋非対称重み付き和（RRF廃止）",
    "lexical_main": "連結スタック（原文＋単語タグ＋メタデータ）生スコア",
    "lexical_ref": "baseline（原文のみ）生スコア",
    "semantic_a": "multilingual-e5-small（原文埋め込み、vector_hybrid 資産再利用）",
    "semantic_b": "ruri-v3-30m（lc-studio ONNX fp32輸出、原文埋め込み）",
    "alphas": ALPHAS,
    "references": {"baseline": agg(base_ranks), "stack": agg(stack_ranks),
                   "rrf_original_hybrid": [24, 5.05], "e5_alone": agg(e5_ranks),
                   "ruri_alone": agg(ruri_ranks)},
}, "cells": {}, "questions": {}}

cells = {}
for lex_name, lex_scores, lex_ranks in [("stack", stack_scores, stack_ranks),
                                        ("baseline", base_scores, base_ranks)]:
    for sem_name, sem_scores, sem_ranks in [("e5", e5_scores, e5_ranks),
                                            ("ruri", ruri_scores, ruri_ranks)]:
        for alpha in ALPHAS:
            fused = {}
            for qid in QIDS:
                keys = pool_keys(qid)
                fused[qid] = fuse(minmax(lex_scores[qid], keys),
                                  minmax(sem_scores[qid], keys), alpha)
            fr = ranks_of(fused)
            cname = f"{lex_name}x{sem_name}@{alpha}"
            ent = [q for q in QIDS if fr[q] is not None and fr[q] <= 3
                   and (lex_ranks[q] is None or lex_ranks[q] > 3)]
            ext = [q for q in QIDS if lex_ranks[q] is not None and lex_ranks[q] <= 3
                   and (fr[q] is None or fr[q] > 3)]
            imp = sum(1 for q in QIDS if fr[q] is not None and lex_ranks[q] is not None
                      and fr[q] < lex_ranks[q])
            wor = sum(1 for q in QIDS if fr[q] is not None and lex_ranks[q] is not None
                      and fr[q] > lex_ranks[q])
            cells[cname] = {"top3": agg(fr)[0], "avg": agg(fr)[1],
                            "entries": ent, "exits": ext, "improved": imp, "worsened": wor}
            for qid in QIDS:
                results["questions"].setdefault(qid, {})[cname] = fr[qid]

# 意味側だけが拾った問題（lex で top-3 外、sem 単独で top-3）と合成での救出状況
rescue = {}
for sem_name, sem_ranks in [("e5", e5_ranks), ("ruri", ruri_ranks)]:
    only = [q for q in QIDS if sem_ranks[q] is not None and sem_ranks[q] <= 3
            and (stack_ranks[q] is None or stack_ranks[q] > 3)]
    rescue[sem_name] = {"sem_only_top3_vs_stack": only,
                        "detail": {q: {"sem_rank": sem_ranks[q], "stack_rank": stack_ranks[q],
                                       **{f"stackx{sem_name}@{a}": results["questions"][q][f"stackx{sem_name}@{a}"]
                                          for a in ALPHAS}} for q in only}}
results["rescue"] = rescue
results["cells"] = cells
for qid in QIDS:
    results["questions"][qid].update({
        "governing": [target_of(qid)[0], target_of(qid)[1]],
        "in_pool": target_of(qid) in pool_keys(qid),
        "rank_baseline": base_ranks[qid], "rank_stack": stack_ranks[qid],
        "rank_e5": e5_ranks[qid], "rank_ruri": ruri_ranks[qid]})

json.dump(results, open(WF / "results_weighted_fusion.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== 参照 ===")
print(f"baseline {agg(base_ranks)} / stack {agg(stack_ranks)} / e5単独 {agg(e5_ranks)} / ruri単独 {agg(ruri_ranks)}")
print("=== セル ===")
for name, c in cells.items():
    print(f"{name}: top3={c['top3']} avg={c['avg']} 入{c['entries']} 出{c['exits']} 改善{c['improved']} 悪化{c['worsened']}")
print("=== 意味側のみ top-3（vs stack） ===")
for s, r in rescue.items():
    print(s, r["sem_only_top3_vs_stack"])
