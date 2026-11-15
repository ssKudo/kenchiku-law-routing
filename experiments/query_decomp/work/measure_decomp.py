#!/usr/bin/env python3
"""Step 3: 問題文の条件分解（クエリ分解）＋最大スコア集約 MaxP（別AIレポート提案④）。
- 土俵: fifty 50問・原文TF-IDF top-30プール。
- 言葉数え側: 連結スタック（原文＋単語タグ＋メタデータ、countermeasures 対策2(a) と同一作法）の生スコア。
- 分解: クエリ全文（stem＋選択肢）を、条件接続表現と読点で機械的に分割し、最大4チャンクにまとめる。
  分割ルールは progress.md に測定前に固定して明記したもの。
- スコア: S_decomp = w全体 × S(q,d) ＋ wチャンク × max_i S(c_i,d)、重み (0.7,0.3)/(0.6,0.4)/(0.5,0.5)。
- 測定前に baseline (29/3.88) と連結スタック (31/3.38) の再現を全問照合する。
- 生スコア（全体・チャンク別）も結果JSONに保存し、(b) の ruri 合成で再利用する。
"""
import json, math, re, statistics, sys
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
QD = RK / "query_decomp"
FM = RK / "fifty_measurement"
CM = RK / "countermeasures"
CFULL = RK / "prototype/c_full"
LAWAPP = Path("/home/hatch/workspace/lawapp")
sys.path.insert(0, str(CFULL / "work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
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
    top3 = sum(1 for r in ranks.values() if r is not None and r <= 3)
    inp = [r for r in ranks.values() if r is not None]
    return top3, round(statistics.mean(inp), 2)


# ---------- 参照検証1: baseline ----------
base_scores = {}
for qid in QIDS:
    keys = pool_keys(qid)
    base_scores[qid] = score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys])

mm = []
base_ranks = {}
for qid in QIDS:
    t = target_of(qid)
    if t in pool_keys(qid):
        r = rank_from_scores(base_scores[qid], pool_keys(qid), t)
        base_ranks[qid] = r
        if r != ref[qid]["rank_original"]:
            mm.append((qid, r, ref[qid]["rank_original"]))
    else:
        base_ranks[qid] = None
print(f"baseline照合: 不一致 {len(mm)} 件 / 集計 {agg(base_ranks)}")
assert not mm and agg(base_ranks) == (29, 3.88)

# ---------- 連結スタック文書の構築（measure_fusion.py と同一作法） ----------
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
stack_ranks = {}
for qid in QIDS:
    keys = pool_keys(qid)
    stack_scores[qid] = score_docs(qtext(qid), [metafull_docs[k] for k in keys])

mm = []
for qid in QIDS:
    t = target_of(qid)
    if t in pool_keys(qid):
        r = rank_from_scores(stack_scores[qid], pool_keys(qid), t)
        stack_ranks[qid] = r
        if r != cm_ref[qid]["ranks"]["meta_append"]:
            mm.append((qid, r, cm_ref[qid]["ranks"]["meta_append"]))
    else:
        stack_ranks[qid] = None
print(f"連結スタック照合: 不一致 {len(mm)} 件 / 集計 {agg(stack_ranks)}")
assert not mm and agg(stack_ranks) == (31, 3.38)

# ---------- クエリ分解（ルールは測定前に固定： progress.md 参照） ----------
MARKERS = ["場合においては", "にあっては", "ただし", "除く", "において", "とき", "、"]
SPLIT_RE = re.compile("|".join(re.escape(m) for m in MARKERS))
MAX_CHUNKS = 4
MIN_SEG_LEN = 2


def split_query(text):
    segs = [s.strip() for s in SPLIT_RE.split(text)]
    segs = [s for s in segs if len(s) >= MIN_SEG_LEN]
    if not segs:
        return [text]
    while len(segs) > MAX_CHUNKS:
        best_i, best_len = 0, None
        for i in range(len(segs) - 1):
            tot = len(segs[i]) + len(segs[i + 1])
            if best_len is None or tot < best_len:
                best_i, best_len = i, tot
        segs[best_i:best_i + 2] = [segs[best_i] + segs[best_i + 1]]
    return segs


chunks = {qid: split_query(qtext(qid)) for qid in QIDS}
nch = [len(chunks[q]) for q in QIDS]
print(f"チャンク数: min {min(nch)} / max {max(nch)} / 平均 {statistics.mean(nch):.2f} / "
      f"分布 { {n: nch.count(n) for n in sorted(set(nch))} }")

# ---------- チャンク別スコア ----------
chunk_scores = {}   # qid -> [ {key: score}, ... ]
for qid in QIDS:
    keys = pool_keys(qid)
    docs = [metafull_docs[k] for k in keys]
    chunk_scores[qid] = [score_docs(c, docs) for c in chunks[qid]]

WEIGHTS = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
decomp_scores, decomp_ranks = {}, {}
for ww, wc in WEIGHTS:
    name = f"decomp@{ww}/{wc}"
    ds, dr = {}, {}
    for qid in QIDS:
        keys = pool_keys(qid)
        whole = stack_scores[qid]
        ds[qid] = {k: ww * whole.get(k, 0.0)
                   + wc * max(cs.get(k, 0.0) for cs in chunk_scores[qid]) for k in keys}
        t = target_of(qid)
        dr[qid] = rank_from_scores(ds[qid], keys, t) if t in keys else None
    decomp_scores[name], decomp_ranks[name] = ds, dr

# ---------- 動きの内訳（vs 連結スタック） ----------
def movement(new_ranks):
    ent = [q for q in QIDS if new_ranks[q] is not None and new_ranks[q] <= 3
           and (stack_ranks[q] is None or stack_ranks[q] > 3)]
    ext = [q for q in QIDS if stack_ranks[q] is not None and stack_ranks[q] <= 3
           and (new_ranks[q] is None or new_ranks[q] > 3)]
    imp = sum(1 for q in QIDS if new_ranks[q] is not None and stack_ranks[q] is not None
              and new_ranks[q] < stack_ranks[q])
    wor = sum(1 for q in QIDS if new_ranks[q] is not None and stack_ranks[q] is not None
              and new_ranks[q] > stack_ranks[q])
    near_in = [q for q in QIDS if stack_ranks[q] is not None and 4 <= stack_ranks[q] <= 10
               and new_ranks[q] is not None and new_ranks[q] <= 3]
    return {"entries": ent, "exits": ext, "improved": imp, "worsened": wor,
            "near4_10_to_top3": near_in}


results = {"meta": {
    "task": "Step 3: 問題文の条件分解＋MaxP（別AIレポート提案④）",
    "split_rule": {
        "markers": MARKERS,
        "note": "マーカー自体はチャンクに含めない。2文字未満の断片は除外。"
                "4個を超えたら隣接ペアのうち合計文字数最小のペアを左優先で繰り返し結合。",
        "max_chunks": MAX_CHUNKS,
        "target_text": "stem＋選択肢4つを連結した検索クエリ全文",
        "chunk_count_dist": {str(n): nch.count(n) for n in sorted(set(nch))},
    },
    "lexical": "連結スタック（原文＋単語タグ＋メタデータ）の言葉数え生スコア",
    "formula": "S_decomp(q,d) = w全体 × S(q,d) + wチャンク × max_i S(c_i,d)",
    "weights": [f"{ww}/{wc}" for ww, wc in WEIGHTS],
    "references": {"baseline": agg(base_ranks), "stack": agg(stack_ranks),
                   "stack_x_ruri@0.85": [33, 3.33]},
}, "cells": {}, "questions": {}}

for name, dr in decomp_ranks.items():
    results["cells"][name] = {"top3": agg(dr)[0], "avg": agg(dr)[1], **movement(dr)}

for qid in QIDS:
    t = target_of(qid)
    keys = pool_keys(qid)
    argmax_chunk = None
    if t in keys and chunk_scores[qid]:
        vals = [cs.get(t, 0.0) for cs in chunk_scores[qid]]
        argmax_chunk = {"index": vals.index(max(vals)), "score": round(max(vals), 3),
                        "text": chunks[qid][vals.index(max(vals))][:120]}
    results["questions"][qid] = {
        "governing": [t[0], t[1]], "in_pool": t in keys,
        "rank_baseline": base_ranks[qid], "rank_stack": stack_ranks[qid],
        "n_chunks": len(chunks[qid]), "chunks": chunks[qid],
        "governing_argmax_chunk": argmax_chunk,
        **{name: dr[qid] for name, dr in decomp_ranks.items()},
        # (b) 合成用に生スコアを保存（キー: "law|art"）
        "whole_scores": {f"{k[0]}|{k[1]}": stack_scores[qid][k] for k in keys},
        "chunk_scores": [{f"{k[0]}|{k[1]}": cs[k] for k in keys} for cs in chunk_scores[qid]],
        "decomp_scores": {name: {f"{k[0]}|{k[1]}": decomp_scores[name][qid][k] for k in keys}
                          for name in decomp_scores},
    }

json.dump(results, open(QD / "results_query_decomp.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== (a) 分解×連結スタック ===")
print(f"参照 stack: {agg(stack_ranks)}")
for name, dr in decomp_ranks.items():
    mv = movement(dr)
    print(f"{name}: top3={agg(dr)[0]} avg={agg(dr)[1]} 入{mv['entries']} 出{mv['exits']} "
          f"改善{mv['improved']} 悪化{mv['worsened']} 惜しい層→top3 {mv['near4_10_to_top3']}")
print("wrote results_query_decomp.json")
