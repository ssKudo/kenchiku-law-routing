#!/usr/bin/env python3
"""Step 6 (learning-to-rank): 特徴量抽出。

fifty 50問 × プール30条文のペアごとに、これまでの実験資産から特徴量を組む。
- lex_base / lex_stack / ruri / bge_ce : rerank_blend/work/lex_scores.json（保存済みペア別スコア）
- jp_ce                                : rerank_blend/work/scores_jp_partial.json
- bge_ce 照合                           : rerank_ce/work/scores_partial.json と順位一致を確認
- fld_body/meta/tag/exception          : bm25f の F-TFIDF フィールドスコアを同構築で再計算
- e5                                   : vector_hybrid の doc 埋め込み（原文）＋質問埋め込み再計算
- hookA / hookB                        : hook_search のフック集合（A5 / 問題別B）
- fam_freq_loo                         : familiar_freq（299問延べ）から自問の正解分を引いた LOO 頻度

検証ゲート: 保存スコアから再計算した順位が公表値と全問一致すること。
  baseline 29/3.88, stack 31/3.38, bgeCE solo 13/9.33, jpCE solo 13/8.88,
  ruri solo 14/8.31, e5 solo（weighted_fusion の rank_e5 と全問照合）
"""
import json, math, re, statistics, sys
from pathlib import Path
import numpy as np

RK = Path("/home/hatch/workspace/research_kudou")
LTR = RK / "ltr"
FM = RK / "fifty_measurement"
CFULL = RK / "prototype/c_full"
LAWAPP = Path("/home/hatch/workspace/lawapp")
sys.path.insert(0, str(CFULL / "work"))
sys.path.insert(0, str(RK / "vector_hybrid/work"))
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
wf_ref = json.load(open(RK / "weighted_fusion/results_weighted_fusion.json", encoding="utf-8"))["questions"]

QIDS = list(pools.keys())
ALL_KEYS = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})
NPOOL = len(ALL_KEYS)


def qtext(qid):
    q = qs[qid]
    return q["stem"] + " " + " ".join(q["choices"])


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


def target_of(qid):
    g = gov[qid]["governing"]
    return (g["law_id"], g["art"])


def K(key):
    return f"{key[0]}|{key[1]}"


def rank_from_scores(scores, keys, target):
    ordered = sorted(keys, key=lambda k: (-scores.get(K(k), 0.0), keys.index(k)))
    return ordered.index(target) + 1


def agg(ranks):
    return sum(1 for r in ranks.values() if r <= 3), round(statistics.mean(ranks.values()), 2)


# ================= 保存済みペア別スコア =================
lex = json.load(open(RK / "rerank_blend/work/lex_scores.json", encoding="utf-8"))
jp_ce_all = json.load(open(RK / "rerank_blend/work/scores_jp_partial.json", encoding="utf-8"))
bge_ce_all = json.load(open(RK / "rerank_ce/work/scores_partial.json", encoding="utf-8"))

saved = {  # qid -> key文字列 -> スコア
    "lex_base": lex["base"], "lex_stack": lex["stack"], "ruri": lex["ruri"],
    "bge_ce": lex["bge"],
    "jp_ce": {q: v["scores"] for q, v in jp_ce_all.items()},
}

# ================= フィールド別スコア（bm25f の F-TFIDF と同構築） =================
tags = {}
for line in open(RK / "tag_vector/word_tags.jsonl", encoding="utf-8"):
    d = json.loads(line)
    tags[(d["law_id"], d["art"])] = d["tags"]
fp_all = json.load(open(RK / "fingerprint/fingerprints.json", encoding="utf-8"))
exceptions = {tuple(k.split("|")): v["exceptions"] for k, v in fp_all.items()}

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


meta_string = {}
for key in ALL_KEYS:
    doc = eng.docs[eng.by_key[key]]
    full = doc["full"]
    ch = chapter_map.get(key, {"chapter": "", "section": ""})
    finger = [w for w in FINGERPRINT if E.qnorm(w) in full]
    tokens = [f"{v:g}{u}" for v, u in sorted(set(extract_numbers(full)))]
    meta_string[key] = (" ".join(x for x in [ch["chapter"], ch["section"]] if x)
                        + " " + " ".join(finger) + " " + " ".join(tokens)).strip()

FIELDS = ["body", "meta", "tag", "exception"]
field_text, field_grams, field_total = {f: {} for f in FIELDS}, {f: {} for f in FIELDS}, {f: {} for f in FIELDS}
for key in ALL_KEYS:
    base = eng.docs[eng.by_key[key]]
    texts = {"body": base["full"],
             "meta": E.qnorm(meta_string[key]),
             "tag": E.qnorm(" ".join(tags[key])),
             "exception": E.qnorm(" ".join(exceptions.get(key, [])))}
    for f in FIELDS:
        g = base["grams"] if f == "body" else E.gram_counts(texts[f])
        field_text[f][key] = texts[f]
        field_grams[f][key] = g
        field_total[f][key] = sum(g.values())
AVG_F = {f: statistics.mean(field_total[f].values()) for f in FIELDS}


def query_parts(raw):
    extra = [c for a, c in E.SYN.items() if a in raw and c not in raw]
    qg = E.gram_counts(E.qnorm(raw + "".join(extra)))
    qterms = [t for t in E.TERMS if t in raw or t in extra]
    return qg, qterms


def tfidf_field_score(qg, qterms, f, key):
    grams = field_grams[f][key]
    sc = 0.0
    for g, c in qg.items():
        w = IDF.get(g)
        if w is None:
            continue
        tf = grams.get(g, 0)
        if tf:
            sc += w * min(c, 3) * (1 + math.log(tf))
    if f == "body":
        base = eng.docs[eng.by_key[key]]
        for g, c in qg.items():
            w = IDF.get(g)
            if w is None:
                continue
            ctf = base["cap_grams"].get(g, 0)
            if ctf:
                sc += w * 2.2 * (1 + math.log(ctf))
    text = field_text[f][key]
    for t in qterms:
        if t in text:
            sc += eng._term_w(t) * 1.6
            if f == "body" and t in eng.docs[eng.by_key[key]]["cap"]:
                sc += eng._term_w(t) * 1.1
    avg = AVG if f == "body" else AVG_F[f]
    return sc / (max(field_total[f][key], 1) / avg) ** 0.3


field_scores = {f: {} for f in FIELDS}  # f -> qid -> key文字列 -> score
for qid in QIDS:
    qg, qterms = query_parts(qtext(qid))
    for f in FIELDS:
        field_scores[f][qid] = {K(k): tfidf_field_score(qg, qterms, f, k) for k in pool_keys(qid)}

# ================= e5 類似度 =================
from embedder import E5
doc_emb = np.load(RK / "vector_hybrid/work/doc_emb_fifty_a.npy")
assert doc_emb.shape[0] == len(ALL_KEYS)
emb_map = {k: doc_emb[i] for i, k in enumerate(ALL_KEYS)}
e5 = E5()
e5_scores = {}
for qid in QIDS:
    qe = e5.embed_texts([qtext(qid)], "query: ")[0]
    e5_scores[qid] = {K(k): float(emb_map[k] @ qe) for k in pool_keys(qid)}
saved["e5"] = e5_scores
for f in FIELDS:
    saved[f"fld_{f}"] = field_scores[f]

# ================= フック・おなじみ =================
hooksA = json.load(open(RK / "hook_search/hooks_A.json", encoding="utf-8"))
A5 = {f'{d["law_id"]}|{d["art"]}' for d in hooksA["A5"]}
hooksB = json.load(open(RK / "hook_search/hooks_B_per_question.json", encoding="utf-8"))
B_by_q = {q: {f'{d["law_id"]}|{d["art"]}' for d in v} for q, v in hooksB.items()}
freq_d = json.load(open(RK / "hook_search/familiar_freq.json", encoding="utf-8"))
FREQ = {f'{d["law_id"]}|{d["art"]}': d["count"] for d in freq_d["freq"]}

# ================= 検証ゲート =================
inpool = [q for q in QIDS if target_of(q) in pool_keys(q)]
print(f"プール内: {len(inpool)}/50")


def solo_ranks(name):
    return {q: rank_from_scores(saved[name][q], pool_keys(q), target_of(q)) for q in inpool}


checks = {}
r = solo_ranks("lex_base"); checks["baseline 29/3.88"] = (agg(r), agg(r) == (29, 3.88))
r = solo_ranks("lex_stack"); checks["stack 31/3.38"] = (agg(r), agg(r) == (31, 3.38))
r = solo_ranks("bge_ce"); checks["bgeCE solo 13/9.33"] = (agg(r), agg(r) == (13, 9.33))
r = solo_ranks("jp_ce"); checks["jpCE solo 13/8.88"] = (agg(r), agg(r) == (13, 8.88))
r = solo_ranks("ruri"); checks["ruri solo 14/8.31"] = (agg(r), agg(r) == (14, 8.31))
r = solo_ranks("e5")
mm = [q for q in inpool if r[q] != wf_ref[q]["rank_e5"]]
checks["e5 solo vs weighted_fusion rank_e5"] = (agg(r), not mm)
# rerank_ce 側 bge スコアとの照合
mm = []
for q in inpool:
    alt = {kk: vv for kk, vv in bge_ce_all[q]["scores"].items()}
    r_alt = rank_from_scores(alt, pool_keys(q), target_of(q))
    if r_alt != solo_ranks("bge_ce")[q]:
        mm.append(q)
checks["bge lex vs rerank_ce scores"] = (f"不一致{len(mm)}件", not mm)
for name, (val, ok) in checks.items():
    print(("OK  " if ok else "FAIL"), name, val)
assert all(ok for _, ok in checks.values()), "検証ゲート失敗"

# ================= 特徴量テーブル出力 =================
FEATS = ["lex_base", "lex_stack", "fld_body", "fld_meta", "fld_tag", "fld_exception",
         "ruri", "e5", "bge_ce", "jp_ce"]
out = {"features_raw": FEATS + ["hookA", "hookB", "fam_freq_loo"],
       "questions": {}}
for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    rows = []
    for k in keys:
        ks = K(k)
        row = {f: float(saved[f][qid].get(ks, 0.0)) for f in FEATS}
        row["hookA"] = 1 if ks in A5 else 0
        row["hookB"] = 1 if ks in B_by_q.get(qid, set()) else 0
        row["fam_freq_loo"] = FREQ.get(ks, 0) - (1 if k == t else 0)
        row["key"] = ks
        row["label"] = 1 if k == t else 0
        rows.append(row)
    out["questions"][qid] = {
        "year": int(qid.split("-")[0]), "in_pool": t in keys,
        "governing": K(t), "rows": rows}
json.dump(out, open(LTR / "work/features.json", "w", encoding="utf-8"), ensure_ascii=False)
print("features.json 出力完了:", len(out["questions"]), "問")
