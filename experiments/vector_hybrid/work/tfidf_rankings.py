#!/usr/bin/env python3
"""TF-IDF 全順位リストの再計算（RRF 合体の材料）と、公表済み順位との照合。
条件: a=O(原文) b=OQ(+質問タグ) c=OS(+看板v2) d=OSQ(+両方) e=OD(+描写看板)×変換問題
照合用に OD×原文問題 / O×変換問題 も計算する。
"""
import json, math, sys
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
sys.path.insert(0, str(RK / "prototype/c_full/work"))
import engine as E

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot


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


def variant(key, extra_texts):
    doc = eng.docs[eng.by_key[key]]
    if not extra_texts:
        return dict(doc)
    d = dict(doc)
    d["full"] = doc["full"] + "".join(E.qnorm(t) for t in extra_texts)
    d["grams"] = E.gram_counts(d["full"])
    d["total"] = sum(d["grams"].values())
    return d


def ordered(scores, pool_keys):
    return sorted(pool_keys, key=lambda k: (-scores.get(k, 0.0), pool_keys.index(k)))


def load_jsonl_map(path, keyf, valf):
    m = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m[keyf(d)] = valf(d)
    return m


def run_dataset(name, pools, questions, conv_questions, qtags, signboards, boards, gov):
    out = {}
    variants_cache = {}
    def get_variant(key, vname):
        ck = (key, vname)
        if ck not in variants_cache:
            if vname == "O":
                variants_cache[ck] = variant(key, [])
            elif vname == "OQ":
                variants_cache[ck] = variant(key, [qtags[key]])
            elif vname == "OS":
                variants_cache[ck] = variant(key, [signboards[key]])
            elif vname == "OSQ":
                variants_cache[ck] = variant(key, [qtags[key], signboards[key]])
            elif vname == "OD":
                variants_cache[ck] = variant(key, [boards[key]])
        return variants_cache[ck]

    COND = {  # cond -> (doc variant, question side)
        "a": ("O", "orig"), "b": ("OQ", "orig"), "c": ("OS", "orig"),
        "d": ("OSQ", "orig"), "e": ("OD", "desc"),
        "e_check_origq": ("OD", "orig"), "check_descq_O": ("O", "desc"),
    }
    for qid, entries in pools.items():
        pool_keys = [(e["law"], e["art"]) for e in entries]
        g = gov[qid]["governing"]
        target = (g["law_id"], g["art"])
        rec = {"governing": {"law": g["law_id"], "art": g["art"]},
               "in_pool": target in pool_keys, "conditions": {}}
        for cond, (vname, qside) in COND.items():
            raw = questions[qid] if qside == "orig" else conv_questions[qid]
            docs = [get_variant(k, vname) for k in pool_keys]
            scores = score_docs(raw, docs)
            order = ordered(scores, pool_keys)
            rank = order.index(target) + 1 if target in pool_keys else None
            rec["conditions"][cond] = {
                "rank": rank,
                "order": [f"{l}|{a}" for l, a in order],
            }
        out[qid] = rec
        print(f"{name} {qid} done", flush=True)
    return out


def qtext(d):
    return d["stem"] + " " + " ".join(d["choices"])


# ---------- パイロット ----------
pools_p = json.load(open(RK / "signboard_pilot/work/pools.json", encoding="utf-8"))
qs_p = {}
for line in open(RK / "small_model_test/blind_questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs_p[f'{d["year"]}-{d["no"]}'] = qtext(d)
conv_p = {}
for line in open(RK / "descriptive_pilot/converted_questions_pilot.jsonl", encoding="utf-8"):
    d = json.loads(line)
    conv_p[d["qid"]] = qtext(d)
gov_all = {}
for line in open(RK / "prototype/c_full/governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov_all[f'{d["year"]}-{d["no"]}'] = d

qt_p = load_jsonl_map(RK / "question_tag_pilot/qtags.jsonl",
                      lambda d: (d["law_id"], d["art"]),
                      lambda d: " ".join(d["questions"]) + " " + " ".join(d["words"]))
sb_p = load_jsonl_map(RK / "signboard_pilot2/signboards_v2.jsonl",
                      lambda d: (d["law_id"], d["art"]),
                      lambda d: d["summary"] + " " + " ".join(d["keywords"]))
bd_p = load_jsonl_map(RK / "descriptive_pilot/boards_pilot.jsonl",
                      lambda d: (d["law_id"], d["art"]), lambda d: d["text"])

pilot = run_dataset("pilot", pools_p, qs_p, conv_p, qt_p, sb_p, bd_p, gov_all)

# ---------- 50問 ----------
pools_5 = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
qs_5 = {}
for line in open(RK / "prototype/c_full/questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs_5[f'{d["year"]}-{d["no"]}'] = qtext(d)
conv_5 = {}
for line in open(RK / "descriptive_pilot/converted_questions50.jsonl", encoding="utf-8"):
    d = json.loads(line)
    conv_5[d["qid"]] = qtext(d)
qt_5 = load_jsonl_map(RK / "fifty_measurement/qtags50.jsonl",
                      lambda d: (d["law_id"], d["art"]),
                      lambda d: " ".join(d["questions"]) + " " + " ".join(d["words"]))
sb_5 = load_jsonl_map(RK / "fifty_measurement/signboards50.jsonl",
                      lambda d: (d["law_id"], d["art"]),
                      lambda d: d["summary"] + " " + " ".join(d["keywords"]))
bd_5 = load_jsonl_map(RK / "descriptive_pilot/boards50.jsonl",
                      lambda d: (d["law_id"], d["art"]), lambda d: d["text"])

fifty = run_dataset("fifty", pools_5, qs_5, conv_5, qt_5, sb_5, bd_5, gov_all)

out = {"pilot": pilot, "fifty": fifty}
json.dump(out, open(RK / "vector_hybrid/work/tfidf_rankings.json", "w", encoding="utf-8"),
          ensure_ascii=False)

# ---------- 照合 ----------
print("\n=== 照合: パイロット ===")
ref_qt = json.load(open(RK / "question_tag_pilot/pilot_results.json", encoding="utf-8"))
ref_v2 = json.load(open(RK / "signboard_pilot2/pilot_results_v2.json", encoding="utf-8"))
ref_desc = json.load(open(RK / "descriptive_pilot/pilot_results_desc.json", encoding="utf-8"))
bad = 0
for qid in pools_p:
    mine = pilot[qid]["conditions"]
    checks = [
        ("a", ref_qt[qid]["rank_original"]),
        ("b", ref_qt[qid]["rank_with_qtag"]),
        ("c", ref_v2[qid]["rank_orig+sb2"]),
        ("e", ref_desc[qid]["rank_desc_desc"]),
        ("e_check_origq", ref_desc[qid]["rank_orig_desc"]),
        ("check_descq_O", ref_desc[qid]["rank_desc_orig"]),
    ]
    for cond, ref in checks:
        got = mine[cond]["rank"]
        ok = got == ref
        if not ok:
            bad += 1
        print(f"{qid} {cond}: mine={got} ref={ref} {'OK' if ok else '** MISMATCH **'}")
print("pilot mismatches:", bad)

print("\n=== 照合: 50問 ===")
ref50 = json.load(open(RK / "fifty_measurement/results50.json", encoding="utf-8"))["questions"]
refd50 = json.load(open(RK / "descriptive_pilot/results_desc50.json", encoding="utf-8"))["questions"]
bad = 0
for qid in pools_5:
    mine = fifty[qid]["conditions"]
    checks = [
        ("a", ref50[qid]["rank_original"]),
        ("b", ref50[qid]["rank_with_qtag"]),
        ("c", ref50[qid]["rank_with_signboard"]),
        ("d", ref50[qid]["rank_with_both"]),
        ("e", refd50[qid]["rank_desc_descq"]),
        ("e_check_origq", refd50[qid]["rank_descboard_origq"]),
        ("check_descq_O", refd50[qid]["rank_orig_descq"]),
    ]
    for cond, ref in checks:
        got = mine[cond]["rank"]
        if got != ref:
            bad += 1
            print(f"{qid} {cond}: mine={got} ref={ref} ** MISMATCH **")
print("fifty mismatches:", bad, "/ ", len(pools_5) * 7)
