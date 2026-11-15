#!/usr/bin/env python3
"""語彙吸い上げ後のフック検索 再測定。

入口: フック条文の検索テキストに child_keywords を追加段落として足した拡張スコアで
上位3フックを種にし、(c) と同一の Graph.expand で到達 (strict/family) を測る。
参照・A5・B は同一ハーネスで再現し、c_full results.json および hook_search の
records_checkpoint.json と全問照合して忠実性を検証する。
拡張スコアは Engine.score_all の文書スコア計算を複製した関数で計算する
(idf・avg_tot は元コーパスのまま固定)。"""
import json, math, statistics, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
from engine import Engine, TERMS, SYN, qnorm, gram_counts
from stage3_measure import Graph

ROOT = Path("/home/hatch/workspace/research_kudou")
CFULL = ROOT / "prototype/c_full"
HOOK = ROOT / "hook_search"
OUT = ROOT / "graph_propagation"


def fam_of(key, gov):
    return key[0] == gov[0] and key[1].split("-")[0] == gov[1].split("-")[0]


def prep_query(engine, raw):
    extra = [c for a, c in SYN.items() if a in raw and c not in raw]
    qg = gram_counts(qnorm(raw + "".join(extra)))
    qterms = [t for t in TERMS if t in raw or t in extra]
    return qg, qterms


def score_doc(engine, qg, qterms, grams, cap_grams, full, cap, total):
    """Engine.score_all の1文書分の計算の複製。"""
    sc = 0.0
    for g, c in qg.items():
        w = engine.idf.get(g)
        if w is None:
            continue
        tf = grams.get(g, 0)
        if tf:
            sc += w * min(c, 3) * (1 + math.log(tf))
        ctf = cap_grams.get(g, 0)
        if ctf:
            sc += w * 2.2 * (1 + math.log(ctf))
    for t in qterms:
        if t in full:
            w = engine._term_w(t)
            sc += w * 1.6
            if t in cap:
                sc += w * 1.1
    sc = sc / (max(total, 1) / engine.avg_tot) ** 0.3
    return sc


def main():
    engine = Engine()
    graph = Graph()
    questions = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    governing = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    ref = {(r["year"], r["no"]): r for r in
           json.loads((CFULL / "results.json").read_text(encoding="utf-8"))["results"]}
    checkpoint = {(r["year"], r["no"]): r for r in
                  json.loads((HOOK / "work/records_checkpoint.json").read_text(encoding="utf-8"))}
    q249 = sorted([k for k, r in ref.items()
                   if r.get("reached_strict") is not None and r.get("confidence") in ("high", "medium")])
    assert len(q249) == 249, len(q249)

    hooksA = json.loads((HOOK / "hooks_A.json").read_text(encoding="utf-8"))
    A5 = {(h["law_id"], h["art"]) for h in hooksA["A5"]}
    freq = Counter()
    for r in governing.values():
        g = r.get("governing")
        if g:
            freq[(g["law_id"], g["art"])] += 1

    kw_data = json.loads((OUT / "child_keywords.json").read_text(encoding="utf-8"))["hooks"]
    kw_by_key = {}
    for rec in kw_data.values():
        kw_by_key[(rec["law_id"], rec["art"])] = rec
    universe = set(kw_by_key)
    assert A5 <= universe

    # --- 拡張文書の事前計算 (フック分のみ) ---
    exp_doc = {}
    for key in universe:
        doc = engine.docs[engine.by_key[key]]
        rec = kw_by_key[key]
        block_q = qnorm(rec["block_text"])
        bg = gram_counts(block_q)
        grams = Counter(doc["grams"])
        grams.update(bg)
        exp_doc[key] = {
            "grams": grams, "cap_grams": doc["cap_grams"],
            "full": doc["full"] + block_q, "cap": doc["cap"],
            "total": doc["total"] + sum(bg.values()),
            "keywords": [k["word"] for k in rec["keywords"]],
        }

    def expand_union(seeds):
        tier = {}
        capped = False
        for (l, a) in seeds:
            ctx, inc_total = graph.expand(l, a)
            if inc_total > 40:
                capped = True
            for k2, t in ctx.items():
                if k2 not in tier or t < tier[k2]:
                    tier[k2] = t
        return tier, capped

    def run_config(scores, govkey, allowed):
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if allowed is not None:
            ranked = [kv for kv in ranked if kv[0] in allowed]
        seeds = [k for k, _ in ranked[:3]]
        tier, capped = expand_union(seeds)
        fam_hit = any(fam_of(k2, govkey) for k2 in tier)
        return {
            "seeds": [{"law_id": l, "art": a, "score": round(s, 2)} for (l, a), s in ranked[:3]],
            "n_seeds": len(seeds),
            "context_size": len(tier),
            "cited_by_capped": capped,
            "reached_strict": govkey in tier,
            "governing_tier": tier.get(govkey),
            "reached_family": fam_hit,
            "gov_in_seeds": govkey in seeds,
        }, tier

    def orig_rank_in_set(scores, allowed, target):
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        filt = [k for k, _ in ranked if k in allowed]
        return (filt.index(target) + 1) if target in filt else None

    records = []
    mism_ref, mism_hook = [], []
    scorer_checked = False
    for key in q249:
        year, no = key
        q = questions[key]
        gov = governing[key]["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])
        scores = engine.score_all(qtext)
        qg, qterms = prep_query(engine, qtext)

        # スコア複製の検証 (最初の1問で全フック分を照合)
        if not scorer_checked:
            for hk in universe:
                d = engine.docs[engine.by_key[hk]]
                mine = score_doc(engine, qg, qterms, d["grams"], d["cap_grams"],
                                 d["full"], d["cap"], d["total"])
                orig = scores.get(hk, 0.0)
                assert abs(mine - orig) < 1e-9, (hk, mine, orig)
            scorer_checked = True
            print("scorer replication verified on", key, flush=True)

        # 拡張スコア (フック分のみ)
        exp_scores = {}
        for hk in universe:
            ed = exp_doc[hk]
            s = score_doc(engine, qg, qterms, ed["grams"], ed["cap_grams"],
                          ed["full"], ed["cap"], ed["total"])
            if s > 0:
                exp_scores[hk] = s

        rec = {"year": year, "no": no, "governing": gov,
               "confidence": governing[key].get("confidence")}
        # --- 参照 (検証) ---
        rref, _ = run_config(scores, govkey, None)
        rr = ref[key]
        ok = (rref["reached_strict"] == rr["reached_strict"]
              and rref["reached_family"] == rr["reached_family"]
              and rref["context_size"] == rr["context_size"]
              and [(s["law_id"], s["art"]) for s in rref["seeds"]]
                  == [(s["law_id"], s["art"]) for s in rr["seeds"]])
        if not ok:
            mism_ref.append({"q": key})
        rec["reference"] = rref
        # --- フック条件 (原文 = 検証, 拡張 = 本測定) ---
        loo = Counter(freq)
        loo[govkey] -= 1
        familiar = {a for a, c in loo.items() if c >= 2}
        Bset = A5 | familiar
        configs = {"A5": (scores, A5), "B": (scores, Bset),
                   "A5_exp": (exp_scores, A5), "B_exp": (exp_scores, Bset)}
        for cname, (sc_map, allowed) in configs.items():
            r, _tier = run_config(sc_map, govkey, allowed)
            r["gov_is_hook"] = govkey in allowed
            if cname.endswith("_exp"):
                base_set = A5 if cname == "A5_exp" else Bset
                seed_info = []
                for s in r["seeds"]:
                    sk = (s["law_id"], s["art"])
                    seed_info.append({
                        "law_id": s["law_id"], "art": s["art"],
                        "score_exp": s["score"],
                        "score_orig": round(scores.get(sk, 0.0), 2),
                        "rank_orig_in_set": orig_rank_in_set(scores, base_set, sk),
                        "matched_keywords": [w for w in exp_doc[sk]["keywords"] if w in qtext],
                        "keywords": exp_doc[sk]["keywords"],
                    })
                r["seed_info"] = seed_info
            rec[cname] = r
        # 原条件の checkpoint 照合
        cp = checkpoint[key]
        for cname in ("A5", "B"):
            mine, theirs = rec[cname], cp[cname]
            ok = (mine["reached_strict"] == theirs["reached_strict"]
                  and mine["reached_family"] == theirs["reached_family"]
                  and mine["context_size"] == theirs["context_size"]
                  and [(s["law_id"], s["art"]) for s in mine["seeds"]]
                      == [(s["law_id"], s["art"]) for s in theirs["seeds"]])
            if not ok:
                mism_hook.append({"q": key, "config": cname})
        rec["familiar_added_B"] = len(familiar - A5)
        records.append(rec)
        print(year, no, "B", rec["B"]["reached_strict"], rec["B"]["reached_family"],
              "| B_exp", rec["B_exp"]["reached_strict"], rec["B_exp"]["reached_family"],
              "| A5_exp", rec["A5_exp"]["reached_strict"], rec["A5_exp"]["reached_family"], flush=True)

    print("VERIFICATION ref mismatches:", len(mism_ref), " hook mismatches:", len(mism_hook), flush=True)

    # --- 未到達の内訳診断 (拡張条件, family 未到達のみ) ---
    for h in universe:
        graph.expand(*h)

    def diagnose(rec, cname, allowed):
        govkey = (rec["governing"]["law_id"], rec["governing"]["art"])
        for h in allowed:
            ctx, _ = graph.expand(*h)
            if any(fam_of(k2, govkey) for k2 in ctx):
                return "entry_miss"
        return "no_bridge"

    for rec in records:
        govkey = (rec["governing"]["law_id"], rec["governing"]["art"])
        loo = Counter(freq)
        loo[govkey] -= 1
        familiar = {a for a, c in loo.items() if c >= 2}
        for cname, allowed in (("A5_exp", A5), ("B_exp", A5 | familiar)):
            rec[cname]["fail_kind"] = (None if rec[cname]["reached_family"]
                                       else diagnose(rec, cname, allowed))

    # --- 勝敗と「吸い上げ語だけ」分析 ---
    def winloss(base, exp):
        wins_f = [r for r in records if r[exp]["reached_family"] and not r[base]["reached_family"]]
        wins_s = [r for r in records if r[exp]["reached_strict"] and not r[base]["reached_strict"]]
        loss_f = [r for r in records if r[base]["reached_family"] and not r[exp]["reached_family"]]
        loss_s = [r for r in records if r[base]["reached_strict"] and not r[exp]["reached_strict"]]

        def kw_driven(win_recs):
            n_any = n_zero = 0
            detail = []
            for r in win_recs:
                seeds = r[exp]["seed_info"]
                driven = [s for s in seeds if s["rank_orig_in_set"] is None or s["rank_orig_in_set"] > 3]
                zero = [s for s in seeds if s["rank_orig_in_set"] is None]
                if driven:
                    n_any += 1
                if zero:
                    n_zero += 1
                detail.append({"q": f'{r["year"]}-{r["no"]}',
                               "gov": f'{r["governing"]["law_id"]}|{r["governing"]["art"]}',
                               "driven_seeds": [{"art": f'{s["law_id"]}|{s["art"]}',
                                                 "rank_orig_in_set": s["rank_orig_in_set"],
                                                 "matched_keywords": s["matched_keywords"]}
                                                for s in driven]})
            return {"n_wins": len(win_recs), "wins_with_keyword_driven_seed": n_any,
                    "wins_with_zero_orig_score_seed": n_zero, "detail": detail}

        return {"family": {"wins": kw_driven(wins_f),
                           "losses": [f'{r["year"]}-{r["no"]}' for r in loss_f]},
                "strict": {"wins": kw_driven(wins_s),
                           "losses": [f'{r["year"]}-{r["no"]}' for r in loss_s]}}

    def agg(rs, cname):
        n = len(rs)
        return {
            "n": n,
            "strict": sum(1 for r in rs if r[cname]["reached_strict"]),
            "family": sum(1 for r in rs if r[cname]["reached_family"]),
            "strict_pct": round(100 * sum(1 for r in rs if r[cname]["reached_strict"]) / n, 1),
            "family_pct": round(100 * sum(1 for r in rs if r[cname]["reached_family"]) / n, 1),
            "gov_is_hook_pct": round(100 * sum(1 for r in rs if r[cname].get("gov_is_hook")) / n, 1),
            "gov_in_seeds": sum(1 for r in rs if r[cname]["gov_in_seeds"]),
            "context_median": statistics.median([r[cname]["context_size"] for r in rs]),
            "context_mean": round(statistics.mean([r[cname]["context_size"] for r in rs]), 1),
            "tier_dist": dict(Counter(str(r[cname]["governing_tier"]) for r in rs if r[cname]["reached_strict"])),
            "fail_kinds": dict(Counter(r[cname].get("fail_kind") for r in rs if r[cname].get("fail_kind"))),
        }

    summary = {
        "verification": {"ref_mismatches": len(mism_ref), "hook_mismatches": len(mism_hook),
                         "scorer_replication": "verified (first question, all hooks, <1e-9)"},
        "reference": agg(records, "reference"),
        "A5": agg(records, "A5"), "B": agg(records, "B"),
        "A5_exp": agg(records, "A5_exp"), "B_exp": agg(records, "B_exp"),
        "B_vs_Bexp": winloss("B", "B_exp"),
        "A5_vs_A5exp": winloss("A5", "A5_exp"),
    }
    (OUT / "results_propagation.json").write_text(json.dumps(
        {"meta": {"questions": "c_full 証拠ベース249問 (high/medium)",
                  "expansion": "stage3_measure.Graph.expand と同一規則",
                  "method": "フック検索テキストに child_keywords (隣接条文群から cover>=2, cover*idf 上位12語) を追加段落として付加し TF-IDF 再検索",
                  "keywords_meta": json.loads((OUT / "child_keywords.json").read_text(encoding="utf-8"))["meta"]},
         "summary": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
