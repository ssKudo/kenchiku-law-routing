#!/usr/bin/env python3
"""50問本測定・描写変換条件：results50.json（既存8条件）には触らず、
descriptive_pilot/results_desc50.json に新規3条件を記録する。
- descboard_origq : 原文問題 × (原文＋描写看板)
- orig_descq      : 変換問題 × 原文
- desc_descq      : 変換問題 × (原文＋描写看板)
土俵は fifty_measurement/work/measure50_grid.py と同一（プール内再スコア、idf固定）。"""
import json, math, sys, statistics
from pathlib import Path

DP = Path("/home/hatch/workspace/research_kudou/descriptive_pilot")
FM = Path("/home/hatch/workspace/research_kudou/fifty_measurement")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

boards = {}
for line in open(DP / "boards50.jsonl", encoding="utf-8"):
    d = json.loads(line); boards[(d["law_id"], d["art"])] = d["text"]
conv = {}
for line in open(DP / "converted_questions50.jsonl", encoding="utf-8"):
    d = json.loads(line); conv[d["qid"]] = d["stem"] + " " + " ".join(d["choices"])

pools = json.load(open("/home/hatch/workspace/research_kudou/question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line); qs[f'{d["year"]}-{d["no"]}'] = d
ref = json.load(open(FM / "results50.json", encoding="utf-8"))
QIDS = list(pools.keys())
assert set(conv.keys()) == set(QIDS), f"converted-question coverage mismatch: {set(QIDS) ^ set(conv.keys())}"
for qid in QIDS:
    for e in pools[qid]:
        assert (e["law"], e["art"]) in boards, f"board missing {(e['law'], e['art'])}"

VNAMES = [("descboard_origq", "orig", "OD"), ("orig_descq", "desc", "O"), ("desc_descq", "desc", "OD")]


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


def make_variant(key, vkey):
    doc = eng.docs[eng.by_key[key]]
    if vkey == "O":
        return dict(doc)
    d = dict(doc)
    d["full"] = doc["full"] + E.qnorm(boards[key])
    d["grams"] = E.gram_counts(d["full"])
    d["total"] = sum(d["grams"].values())
    return d


def rank_of(scores, pool_keys, target):
    ordered = sorted(pool_keys, key=lambda k: (-scores.get(k, 0.0), pool_keys.index(k)))
    return ordered.index(target) + 1, ordered


def char_stats(texts):
    ls = [len(t) for t in texts]
    return {"mean": round(statistics.mean(ls), 1), "median": statistics.median(ls),
            "min": min(ls), "max": max(ls), "total": sum(ls)}


out = {"meta": {"n_questions": 50,
                "conditions": [v[0] for v in VNAMES],
                "board_chars": char_stats(list(boards.values())),
                "converted_question_chars": char_stats([conv[q] for q in QIDS]),
                "note": "描写変換条件のみ。既存8条件は fifty_measurement/results50.json を参照。"},
       "questions": {}}
print("board chars:", out["meta"]["board_chars"])
print("converted question chars:", out["meta"]["converted_question_chars"])

for vname, qk, vk in VNAMES:
    for qid in QIDS:
        r0 = ref["questions"][qid]
        rec = out["questions"].setdefault(qid, {"governing": r0["governing"], "in_pool": r0["in_pool"]})
        if not r0["in_pool"]:
            rec[f"rank_{vname}"] = None
            rec[f"top3_{vname}"] = False
            continue
        pool_keys = [(e["law"], e["art"]) for e in pools[qid]]
        target = (r0["governing"]["law_id"], r0["governing"]["art"])
        raw = (qs[qid]["stem"] + " " + " ".join(qs[qid]["choices"])) if qk == "orig" else conv[qid]
        docs = [make_variant(k, vk) for k in pool_keys]
        scores = score_docs(raw, docs)
        r, ordered = rank_of(scores, pool_keys, target)
        rec[f"rank_{vname}"] = r
        rec[f"top3_{vname}"] = r <= 3
        rec[f"top10_{vname}"] = [{"law": k[0], "art": k[1], "score": round(scores[k], 2)} for k in ordered[:10]]
        rec[f"governing_score_{vname}"] = round(scores[target], 2)
    json.dump(out, open(DP / "results_desc50.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = [out["questions"][q][f"rank_{vname}"] for q in QIDS if out["questions"][q]["in_pool"]]
    top3 = sum(1 for q in QIDS if out["questions"][q][f"top3_{vname}"])
    print(f"{vname}: top3 {top3}/50 | mean rank (in-pool) {statistics.mean(done):.2f} | median {statistics.median(done)}")

print("wrote results_desc50.json")
