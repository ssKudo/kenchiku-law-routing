#!/usr/bin/env python3
"""非AI古典手法5チャンネルの測定（fifty_measurement の50問・プール30、原文のみ）。
- baseline は measure50.py の score_docs を複製し、results50.json の rank_original と全問照合する。
- 順位付けは (-score, pool順) で baseline と同一作法。プール外8問は rank=None。
- 合わせ技は RRF(k=60): 1/(60+rank_base) + 1/(60+rank_channel)。
"""
import json, math, re, sys, statistics, hashlib
from pathlib import Path
import numpy as np

BASE = Path("/home/hatch/workspace/research_kudou/classic_channels")
FM = Path("/home/hatch/workspace/research_kudou/fifty_measurement")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

pools = json.load(open("/home/hatch/workspace/research_kudou/question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
pool_texts = json.load(open("/home/hatch/workspace/research_kudou/question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d
gov = {}
for line in open(CFULL / "governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d
ref = json.load(open(FM / "results50.json", encoding="utf-8"))["questions"]
QIDS = list(pools.keys())


def qtext(qid):
    q = qs[qid]
    return q["stem"] + " " + " ".join(q["choices"])


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


def target_of(qid):
    g = gov[qid]["governing"]
    return (g["law_id"], g["art"])


def score_docs(raw, docs):
    """measure50.py の score_docs の複製（原文ドキュメント用）。"""
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


def ranks_dict(scores_by_q):
    return scores_by_q  # helper placeholder


def rank_map(scores, keys):
    """scores dict -> {key: rank}（同点時はプール順）"""
    ordered = sorted(keys, key=lambda k: (-scores.get(k, 0.0), keys.index(k)))
    return {k: i + 1 for i, k in enumerate(ordered)}


def rrf_fuse(base_scores, chan_scores, keys):
    rb = rank_map(base_scores, keys)
    rc = rank_map(chan_scores, keys)
    fused = {k: 1.0 / (60 + rb[k]) + 1.0 / (60 + rc[k]) for k in keys}
    return fused


# ---------- baseline の照合 ----------
base_scores = {}
mismatch = []
for qid in QIDS:
    keys = pool_keys(qid)
    docs = [eng.docs[eng.by_key[k]] for k in keys]
    sc = score_docs(qtext(qid), docs)
    base_scores[qid] = sc
    t = target_of(qid)
    if t in keys:
        r = rank_from_scores(sc, keys, t)
        if r != ref[qid]["rank_original"]:
            mismatch.append((qid, r, ref[qid]["rank_original"]))
print(f"baseline照合: 不一致 {len(mismatch)} 件 {mismatch[:5]}")
assert not mismatch, "baseline が results50 と一致しない"

results = {"meta": {}, "questions": {}}
for qid in QIDS:
    t = target_of(qid)
    g = gov[qid]["governing"]
    results["questions"][qid] = {
        "governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                      "confidence": gov[qid]["confidence"]},
        "in_pool": t in pool_keys(qid),
        "ranks": {"baseline": ref[qid]["rank_original"]},
        "details": {},
    }


def record(qid, name, scores):
    keys = pool_keys(qid)
    t = target_of(qid)
    if t not in keys:
        results["questions"][qid]["ranks"][name] = None
        return None
    r = rank_from_scores(scores, keys, t)
    results["questions"][qid]["ranks"][name] = r
    return r


def record_rrf(qid, name, chan_scores):
    return record(qid, name, rrf_fuse(base_scores[qid], chan_scores, pool_keys(qid)))


def summarize(name):
    rs = [results["questions"][q]["ranks"].get(name) for q in QIDS]
    inp = [r for r in rs if r is not None]
    top3 = sum(1 for r in inp if r <= 3)
    b = [results["questions"][q]["ranks"]["baseline"] for q in QIDS]
    ent = sum(1 for q in QIDS if (results["questions"][q]["ranks"].get(name) or 99) <= 3 and (b[QIDS.index(q)] or 99) > 3)
    ext = sum(1 for q in QIDS if (results["questions"][q]["ranks"].get(name) or 99) > 3 and (b[QIDS.index(q)] or 99) <= 3)
    return {"top3": top3, "mean_rank": round(statistics.mean(inp), 2) if inp else None,
            "median_rank": statistics.median(inp) if inp else None,
            "top3_entries_vs_baseline": ent, "top3_exits_vs_baseline": ext}


if __name__ == "__main__":
    print("baseline OK: top3", summarize("baseline"))
