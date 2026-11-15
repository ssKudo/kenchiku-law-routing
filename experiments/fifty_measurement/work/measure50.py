#!/usr/bin/env python3
"""50問本測定：質問タグ vs 看板v2（measure_qt.py の50問版）。
- idf / avg_tot は全体コーパス（約4,700条）のエンジンで固定（パイロットと同条件）。
- 各問の baseline 上位30条プール内だけで、文書テキストを
  O=原文 / OQ=原文＋質問タグ / OS=原文＋看板v2 / OSQ=原文＋両方 に差し替えて再スコア。
- 正解条文がプール外の8問は rank=null（並べ替えでは届かない）として記録。
- 条件ごとに results50.json へ逐次保存する。
"""
import json, math, sys, statistics
from pathlib import Path

FM = Path("/home/hatch/workspace/research_kudou/fifty_measurement")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

sb = {}
for line in open(FM / "signboards50.jsonl", encoding="utf-8"):
    d = json.loads(line)
    sb[(d["law_id"], d["art"])] = d
qt = {}
for line in open(FM / "qtags50.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qt[(d["law_id"], d["art"])] = d

pools = json.load(open("/home/hatch/workspace/research_kudou/question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))

qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d

gov = {}
for line in open(CFULL / "governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d

QIDS = list(pools.keys())
VNAMES = [("original", "O"), ("with_qtag", "OQ"), ("with_signboard", "OS"), ("with_both", "OSQ")]

# ---- カバレッジ検証 ----
pool_keys_all = set()
for entries in pools.values():
    for e in entries:
        pool_keys_all.add((e["law"], e["art"]))
missing_qt = [k for k in pool_keys_all if k not in qt]
missing_sb = [k for k in pool_keys_all if k not in sb]
print(f"pool articles: {len(pool_keys_all)}, qtags: {len(qt)}, signboards: {len(sb)}")
assert not missing_qt, f"missing qtags: {missing_qt[:5]}"
assert not missing_sb, f"missing signboards: {missing_sb[:5]}"


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


def sign_text(key):
    s = sb[key]
    return E.qnorm(s["summary"] + " " + " ".join(s["keywords"]))


def qtag_text(key):
    t = qt[key]
    return E.qnorm(" ".join(t["questions"]) + " " + " ".join(t["words"]))


def make_variant(key, vkey):
    doc = eng.docs[eng.by_key[key]]
    if vkey == "O":
        return dict(doc)
    extra_text = ""
    if vkey in ("OQ", "OSQ"):
        extra_text += qtag_text(key)
    if vkey in ("OS", "OSQ"):
        extra_text += sign_text(key)
    d = dict(doc)
    d["full"] = doc["full"] + extra_text
    d["grams"] = E.gram_counts(d["full"])
    d["total"] = sum(d["grams"].values())
    return d


def rank_of(scores, pool_keys, target):
    ordered = sorted(pool_keys, key=lambda k: (-scores.get(k, 0.0), pool_keys.index(k)))
    return ordered.index(target) + 1, ordered


# ---- 文字数統計 ----
def char_stats(texts):
    ls = [len(t) for t in texts]
    return {"mean": round(statistics.mean(ls), 1), "median": statistics.median(ls),
            "min": min(ls), "max": max(ls)}

qt_chars = char_stats([" ".join(qt[k]["questions"]) + " " + " ".join(qt[k]["words"]) for k in pool_keys_all])
sb_chars = char_stats([sb[k]["summary"] + " " + " ".join(sb[k]["keywords"]) for k in pool_keys_all])

results = {"meta": {
    "n_questions": len(QIDS),
    "conditions": [v[0] for v in VNAMES],
    "qtag_chars": qt_chars, "signboard_chars": sb_chars,
}, "questions": {}}

for qid in QIDS:
    g = gov[qid]["governing"]
    target = (g["law_id"], g["art"])
    pool_entries = pools[qid]
    pool_keys = [(e["law"], e["art"]) for e in pool_entries]
    results["questions"][qid] = {
        "governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                      "confidence": gov[qid]["confidence"]},
        "in_pool": target in pool_keys,
    }

# 条件ごとに全問を測って逐次保存
for vname, vkey in VNAMES:
    for qid in QIDS:
        rec = results["questions"][qid]
        if not rec["in_pool"]:
            rec[f"rank_{vname}"] = None
            rec[f"top3_{vname}"] = False
            continue
        q = qs[qid]
        raw = q["stem"] + " " + " ".join(q["choices"])
        pool_keys = [(e["law"], e["art"]) for e in pools[qid]]
        target = (rec["governing"]["law_id"], rec["governing"]["art"])
        docs = [make_variant(k, vkey) for k in pool_keys]
        scores = score_docs(raw, docs)
        r, ordered = rank_of(scores, pool_keys, target)
        rec[f"rank_{vname}"] = r
        rec[f"top3_{vname}"] = r <= 3
        rec[f"top10_{vname}"] = [{"law": k[0], "art": k[1], "score": round(scores[k], 2)} for k in ordered[:10]]
        rec[f"governing_score_{vname}"] = round(scores[target], 2)
    json.dump(results, open(FM / "results50.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = [results["questions"][q][f"rank_{vname}"] for q in QIDS if results["questions"][q]["in_pool"]]
    top3 = sum(1 for q in QIDS if results["questions"][q][f"top3_{vname}"])
    print(f"{vname}: top3 {top3}/50 | mean rank (in-pool) {statistics.mean(done):.2f} | median {statistics.median(done)}")

print("wrote results50.json")
