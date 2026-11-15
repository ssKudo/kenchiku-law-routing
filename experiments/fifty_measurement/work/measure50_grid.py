#!/usr/bin/env python3
"""50問本測定・追加：問題文の言い換え側4条件を results50.json に追記する。
クエリを「言い換えた問題文」に差し替え、文書側の4変種（O/OQ/OS/OSQ）で再スコア。
既存の原文側の結果には触らない。条件ごとに逐次保存。"""
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
    d = json.loads(line); sb[(d["law_id"], d["art"])] = d
qt = {}
for line in open(FM / "qtags50.jsonl", encoding="utf-8"):
    d = json.loads(line); qt[(d["law_id"], d["art"])] = d
para = {}
for line in open(FM / "paraphrased_questions50.jsonl", encoding="utf-8"):
    d = json.loads(line); para[d["qid"]] = d["paraphrase"]

pools = json.load(open("/home/hatch/workspace/research_kudou/question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line); qs[f'{d["year"]}-{d["no"]}'] = d

results = json.load(open(FM / "results50.json", encoding="utf-8"))
QIDS = list(pools.keys())
assert set(para.keys()) == set(QIDS), f"paraphrase coverage mismatch: {set(QIDS) ^ set(para.keys())}"

VNAMES = [("para_original", "O"), ("para_qtag", "OQ"), ("para_signboard", "OS"), ("para_both", "OSQ")]


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


def char_stats(texts):
    ls = [len(t) for t in texts]
    return {"mean": round(statistics.mean(ls), 1), "median": statistics.median(ls),
            "min": min(ls), "max": max(ls)}

results["meta"]["paraphrase_chars"] = char_stats([para[q] for q in QIDS])
results["meta"]["question_chars"] = char_stats(
    [qs[q]["stem"] + " " + " ".join(qs[q]["choices"]) for q in QIDS])
results["meta"]["conditions"] += [v[0] for v in VNAMES]
print("paraphrase chars:", results["meta"]["paraphrase_chars"])
print("question chars:", results["meta"]["question_chars"])

for vname, vkey in VNAMES:
    for qid in QIDS:
        rec = results["questions"][qid]
        if not rec["in_pool"]:
            rec[f"rank_{vname}"] = None
            rec[f"top3_{vname}"] = False
            continue
        pool_keys = [(e["law"], e["art"]) for e in pools[qid]]
        target = (rec["governing"]["law_id"], rec["governing"]["art"])
        docs = [make_variant(k, vkey) for k in pool_keys]
        scores = score_docs(para[qid], docs)
        r, ordered = rank_of(scores, pool_keys, target)
        rec[f"rank_{vname}"] = r
        rec[f"top3_{vname}"] = r <= 3
        rec[f"top10_{vname}"] = [{"law": k[0], "art": k[1], "score": round(scores[k], 2)} for k in ordered[:10]]
        rec[f"governing_score_{vname}"] = round(scores[target], 2)
    json.dump(results, open(FM / "results50.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = [results["questions"][q][f"rank_{vname}"] for q in QIDS if results["questions"][q]["in_pool"]]
    top3 = sum(1 for q in QIDS if results["questions"][q][f"top3_{vname}"])
    print(f"{vname}: top3 {top3}/50 | mean rank (in-pool) {statistics.mean(done):.2f} | median {statistics.median(done)}")

print("updated results50.json")
