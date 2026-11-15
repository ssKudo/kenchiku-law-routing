#!/usr/bin/env python3
"""看板パイロット再測定。
- idf / avg_tot は全体コーパス（約4,700条）のエンジンで固定。
- 各問の baseline 上位30条プール内だけで、文書テキストを
  O=原文 / OS=原文＋看板 / S=看板のみ に差し替えて再スコアし、正答条の順位を比較する。
- 採点にだけ governing.jsonl を使う。"""
import json, math, sys
from pathlib import Path

BASE = Path("/home/hatch/workspace/research_kudou/signboard_pilot")
sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

sb = {}
for line in open(BASE / "signboards.jsonl", encoding="utf-8"):
    d = json.loads(line)
    sb[(d["law_id"], d["art"])] = d

pools = json.load(open(BASE / "work/pools.json", encoding="utf-8"))

qs = {}
for line in open("/home/hatch/workspace/research_kudou/small_model_test/blind_questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d

gov = {}
for line in open("/home/hatch/workspace/research_kudou/prototype/c_full/governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d

BASELINE_GIVEN = {"2017-21": 5, "2017-22": 7, "2017-24": 5, "2017-4": 10, "2017-1": 1}


def term_w(t):
    return eng._term_w(t)


def score_docs(raw, docs):
    """engine.Engine.score_all と同一式。docs は dict のリスト。"""
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
                w = term_w(t)
                sc += w * 1.6
                if t in doc["cap"]:
                    sc += w * 1.1
        sc = sc / (max(doc["total"], 1) / AVG) ** 0.3
        out[(doc["law"], doc["art"])] = sc
    return out


def make_variants(key):
    doc = eng.docs[eng.by_key[key]]
    s = sb[key]
    kw = " ".join(s["keywords"])
    sign_text = E.qnorm(s["summary"] + " " + kw)
    o = dict(doc)
    os_full = doc["full"] + sign_text
    os_doc = dict(doc)
    os_doc["full"] = os_full
    os_doc["grams"] = E.gram_counts(os_full)
    os_doc["total"] = sum(os_doc["grams"].values())
    s_doc = dict(doc)
    s_doc["full"] = sign_text
    s_doc["grams"] = E.gram_counts(sign_text)
    s_doc["total"] = sum(s_doc["grams"].values())
    s_doc["cap"] = kw
    s_doc["cap_grams"] = E.gram_counts(E.qnorm(kw))
    return {"O": o, "OS": os_doc, "S": s_doc}


def rank_of(scores, pool_keys, target):
    ordered = sorted(pool_keys, key=lambda k: (-scores.get(k, 0.0), pool_keys.index(k)))
    return ordered.index(target) + 1, ordered


results = {}
for qid in ["2017-21", "2017-22", "2017-24", "2017-4", "2017-1"]:
    q = qs[qid]
    raw = q["stem"] + " " + " ".join(q["choices"])
    pool_entries = pools[qid]
    pool_keys = [(e["law"], e["art"]) for e in pool_entries]
    g = gov[qid]["governing"]
    target = (g["law_id"], g["art"])
    assert target in pool_keys, f"{qid}: governing not in pool!"
    variants = {k: make_variants(k) for k in pool_keys}
    rec = {
        "qid": qid,
        "governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                      "confidence": gov[qid]["confidence"]},
        "baseline_rank_given_full_corpus": BASELINE_GIVEN[qid],
        "pool": [{"law": e["law"], "art": e["art"], "baseline_score": e["score"]} for e in pool_entries],
    }
    for vname, vkey in [("original", "O"), ("with_signboard", "OS"), ("signboard_only", "S")]:
        docs = [variants[k][vkey] for k in pool_keys]
        scores = score_docs(raw, docs)
        r, ordered = rank_of(scores, pool_keys, target)
        rec[f"rank_{vname}"] = r
        rec[f"top3_{vname}"] = r <= 3
        rec[f"top10_{vname}"] = [{"law": k[0], "art": k[1], "score": round(scores[k], 2)} for k in ordered[:10]]
        rec[f"governing_score_{vname}"] = round(scores[target], 2)
    results[qid] = rec
    print(qid, "gov", target, "| baseline(given)", BASELINE_GIVEN[qid],
          "| O", rec["rank_original"], "| OS", rec["rank_with_signboard"], "| S", rec["rank_signboard_only"])

json.dump(results, open(BASE / "pilot_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote pilot_results.json")
