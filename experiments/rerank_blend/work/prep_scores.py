#!/usr/bin/env python3
"""Step 5 前段： 言葉数え側（baseline／連結スタック）生スコアの再現と参照検証、
ruri 参照セル（現行最良）の再現照合、bge 保存スコアの照合。
作法は weighted_fusion/work/measure_fusion.py をそのまま複製し、検証 assert を通す。
出力： work/lex_scores.json（base/stack の生スコア）、work/prep_check.json（照合結果）。
"""
import json, math, re, statistics, sys, time
from pathlib import Path
import numpy as np

RK = Path("/home/hatch/workspace/research_kudou")
RB = RK / "rerank_blend"
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
wf_ref = json.load(open(RK / "weighted_fusion/results_weighted_fusion.json", encoding="utf-8"))
rr_ref = json.load(open(RK / "rerank_ce/results_rerank.json", encoding="utf-8"))["questions"]

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
    top3 = sum(1 for r in ranks.values() if r is not None and r <= 3)
    inp = [r for r in ranks.values() if r is not None]
    return top3, round(statistics.mean(inp), 2)


def ranks_of(score_map):
    out = {}
    for qid in QIDS:
        t = target_of(qid)
        keys = pool_keys(qid)
        out[qid] = rank_from_scores(score_map[qid], keys, t) if t in keys else None
    return out


# ---------- baseline ----------
base_scores = {}
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

# ---------- 連結スタック（countermeasures 対策2(a) と同一作法） ----------
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

stack_scores = {}
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

base_ranks = ranks_of(base_scores)
stack_ranks = ranks_of(stack_scores)
print("baseline:", agg(base_ranks), " stack:", agg(stack_ranks))
assert agg(base_ranks) == (29, 3.88) and agg(stack_ranks) == (31, 3.38)

# ---------- ruri 参照セルの再現（保存済み doc 埋め込み＋質問埋め込み再計算） ----------
import onnxruntime as ort
from tokenizers import Tokenizer

RURI_DIR = RK / "weighted_fusion/work/ruri_model"
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
        aggmap = {}
        for (ti, _), e in zip(all_chunks, chunk_embs):
            aggmap.setdefault(ti, []).append(e)
        embs = []
        for ti in range(len(texts)):
            v = np.mean(aggmap[ti], axis=0)
            embs.append(v / max(np.linalg.norm(v), 1e-12))
        return np.stack(embs).astype(np.float32)


doc_emb = np.load(RK / "weighted_fusion/work/doc_emb_ruri_fifty.npy")
assert doc_emb.shape[0] == len(ALL_KEYS)
ruri_map = {k: doc_emb[i] for i, k in enumerate(ALL_KEYS)}
ruri = Ruri()
ruri_scores = {}
for qid in QIDS:
    qe = ruri.embed_texts([qtext(qid)], "検索クエリ: ")[0]
    ruri_scores[qid] = {k: float(ruri_map[k] @ qe) for k in pool_keys(qid)}


def minmax(scores, keys):
    vals = [scores.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in keys}
    return {k: (scores.get(k, 0.0) - lo) / (hi - lo) for k in keys}


def fuse(lex, sem, alpha):
    return {k: alpha * lex[k] + (1 - alpha) * sem[k] for k in lex}


check = {}
for alpha in [0.80, 0.85, 0.90, 0.95]:
    fused = {qid: fuse(minmax(stack_scores[qid], pool_keys(qid)),
                       minmax(ruri_scores[qid], pool_keys(qid)), alpha) for qid in QIDS}
    fr = ranks_of(fused)
    saved = wf_ref["cells"][f"stackxruri@{alpha}"]
    mine = {"top3": agg(fr)[0], "avg": agg(fr)[1]}
    ok_q = all(fr[q] == wf_ref["questions"][q][f"stackxruri@{alpha}"] for q in QIDS)
    check[f"stackxruri@{alpha}"] = {"mine": mine, "saved": {"top3": saved["top3"], "avg": saved["avg"]},
                                    "per_question_match": ok_q}
    print(f"stack×ruri α={alpha}: 自前 {mine} / 保存 {saved['top3']}/{saved['avg']} / 全問一致 {ok_q}")
    assert mine["top3"] == saved["top3"] and mine["avg"] == saved["avg"] and ok_q

# ---------- bge 保存スコアの照合 ----------
bge_raw = json.load(open(RK / "rerank_ce/work/scores_partial.json", encoding="utf-8"))
bge_scores = {}
for qid in QIDS:
    bge_scores[qid] = {}
    for ks, v in bge_raw[qid]["scores"].items():
        law, art = ks.split("|", 1)
        bge_scores[qid][(law, art)] = v
bge_ranks = ranks_of(bge_scores)
mm = []
for qid in QIDS:
    if bge_ranks[qid] != rr_ref[qid]["rank_rerank"]:
        mm.append((qid, bge_ranks[qid], rr_ref[qid]["rank_rerank"]))
print(f"bge単独照合: 不一致 {len(mm)} 件 {mm[:5]} / 集計 {agg(bge_ranks)}")
assert not mm and agg(bge_ranks) == (13, 9.33)

# ---------- 保存 ----------
def ser(sm):
    return {qid: {f"{k[0]}|{k[1]}": v for k, v in sm[qid].items()} for qid in QIDS}

json.dump({"base": ser(base_scores), "stack": ser(stack_scores),
           "ruri": ser(ruri_scores), "bge": ser(bge_scores)},
          open(RB / "work/lex_scores.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"baseline": agg(base_ranks), "stack": agg(stack_ranks),
           "bge_solo": agg(bge_ranks), "ruri_cells": check},
          open(RB / "work/prep_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("prep OK → work/lex_scores.json, work/prep_check.json")
