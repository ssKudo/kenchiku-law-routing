"""描写変換パイロット測定: signboard_pilot2/work/measure2.py と同一の土俵。
- idf / avg_tot は全体コーパスで固定。各問の baseline 上位30条プール内で再スコア。
- 文書変種: O=原文 / OD=原文＋描写看板 / D=描写看板のみ
- クエリ変種: orig=原文の問題文 / desc=描写変換した問題文
- 条件: orig(O), orig_desc(OD), desc_orig(O), desc_desc(OD), desconly(D, origクエリ),
        desconly_descq(D, descクエリ)
- 採点にだけ governing.jsonl を使う。"""
import json, math, sys
from pathlib import Path

DP = Path("/home/hatch/workspace/research_kudou/descriptive_pilot")
BASE1 = Path("/home/hatch/workspace/research_kudou/signboard_pilot")
sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

boards = {}
for line in open(DP / "boards_pilot.jsonl", encoding="utf-8"):
    d = json.loads(line)
    boards[(d["law_id"], d["art"])] = d["text"]

pools = json.load(open(BASE1 / "work/pools.json", encoding="utf-8"))

qs = {}
for line in open("/home/hatch/workspace/research_kudou/small_model_test/blind_questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d

conv = {}
for line in open(DP / "converted_questions_pilot.jsonl", encoding="utf-8"):
    d = json.loads(line)
    conv[d["qid"]] = d["stem"] + " " + " ".join(d["choices"])

gov = {}
for line in open("/home/hatch/workspace/research_kudou/prototype/c_full/governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d

BASELINE_GIVEN = {"2017-21": 5, "2017-22": 7, "2017-24": 5, "2017-4": 10, "2017-1": 1}
QORDER = ["2017-21", "2017-22", "2017-24", "2017-4", "2017-1"]


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


def make_variants(key):
    doc = eng.docs[eng.by_key[key]]
    btext = E.qnorm(boards[key])
    out = {"O": doc}
    od = dict(doc)
    od["full"] = doc["full"] + btext
    od["grams"] = E.gram_counts(od["full"])
    od["total"] = sum(od["grams"].values())
    out["OD"] = od
    dd = dict(doc)
    dd["full"] = btext
    dd["grams"] = E.gram_counts(btext)
    dd["total"] = sum(dd["grams"].values())
    dd["cap"] = ""
    dd["cap_grams"] = {}
    out["D"] = dd
    return out


def rank_of(scores, pool_keys, target):
    ordered = sorted(pool_keys, key=lambda k: (-scores.get(k, 0.0), pool_keys.index(k)))
    return ordered.index(target) + 1, ordered


CONDS = [("orig", "orig", "O"), ("orig_desc", "orig", "OD"), ("desc_orig", "desc", "O"),
         ("desc_desc", "desc", "OD"), ("desconly", "orig", "D"), ("desconly_descq", "desc", "D")]
results = {}
for qid in QORDER:
    q = qs[qid]
    raws = {"orig": q["stem"] + " " + " ".join(q["choices"]), "desc": conv[qid]}
    pool_keys = [(e["law"], e["art"]) for e in pools[qid]]
    missing = [k for k in pool_keys if k not in boards]
    assert not missing, f"{qid}: boards missing {missing}"
    g = gov[qid]["governing"]
    target = (g["law_id"], g["art"])
    assert target in pool_keys
    variants = {k: make_variants(k) for k in pool_keys}
    rec = {"qid": qid,
           "governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                         "confidence": gov[qid]["confidence"]},
           "baseline_rank_given_full_corpus": BASELINE_GIVEN[qid]}
    for cname, qk, vk in CONDS:
        docs = [variants[k][vk] for k in pool_keys]
        scores = score_docs(raws[qk], docs)
        r, ordered = rank_of(scores, pool_keys, target)
        rec[f"rank_{cname}"] = r
        rec[f"governing_score_{cname}"] = round(scores[target], 2)
        rec[f"top10_{cname}"] = [{"law": k[0], "art": k[1], "score": round(scores[k], 2)} for k in ordered[:10]]
    results[qid] = rec
    print(qid, "gov", target, "| " + " | ".join(f"{c} {rec[f'rank_{c}']}" for c, _, _ in CONDS))

print()
names = [c for c, _, _ in CONDS]
print(f"{'cond':16s}" + "".join(f"{q:>9s}" for q in ['No.21', 'No.22', 'No.24', 'No.4', 'No.1']) + "  top3  avg")
for c in names:
    rs = [results[q][f"rank_{c}"] for q in QORDER]
    print(f"{c:16s}" + "".join(f"{r:9d}" for r in rs) + f"  {sum(1 for r in rs if r <= 3)}/5  {sum(rs)/5:.1f}")
print("\nbranching survival (board-only):")
for c in ["desconly", "desconly_descq"]:
    rs = [results[q][f"rank_{c}"] for q in QORDER]
    print(f"  {c}: top-10 {sum(1 for r in rs if r <= 10)}/5, top-20 {sum(1 for r in rs if r <= 20)}/5")

board_chars = [len(t) for t in boards.values()]
orig_chars = {}
import statistics
print(f"\nboard chars: mean {statistics.mean(board_chars):.0f} median {statistics.median(board_chars)} min {min(board_chars)} max {max(board_chars)} total {sum(board_chars)}")
qchars_o = [len(qs[q]['stem'] + ' ' + ' '.join(qs[q]['choices'])) for q in QORDER]
qchars_d = [len(conv[q]) for q in QORDER]
print(f"question chars orig: mean {statistics.mean(qchars_o):.0f} | desc: mean {statistics.mean(qchars_d):.0f}")

json.dump(results, open(DP / "pilot_results_desc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote pilot_results_desc.json")
