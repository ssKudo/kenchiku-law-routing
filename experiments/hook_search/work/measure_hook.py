#!/usr/bin/env python3
"""フック検索の測定。
入口をフック集合だけに絞った TF-IDF top-3 を種に、(c) と同一の Graph.expand で
展開し、正答条の到達 (strict/family) を測る。
先に参照条件（無制限の種）で results.json と全問照合し、ハーネスの忠実性を検証する。"""
import json, statistics, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
from engine import Engine
from stage3_measure import Graph

CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
OUT = Path("/home/hatch/workspace/research_kudou/hook_search")

def fam_of(key, gov):
    return key[0] == gov[0] and key[1].split("-")[0] == gov[1].split("-")[0]

def main():
    engine = Engine()
    graph = Graph()
    questions = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    governing = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    ref = {(r["year"], r["no"]): r for r in
           json.loads((CFULL / "results.json").read_text(encoding="utf-8"))["results"]}
    q249 = sorted([k for k, r in ref.items()
                   if r.get("reached_strict") is not None and r.get("confidence") in ("high", "medium")])
    assert len(q249) == 249, len(q249)

    hooksA = json.loads((OUT / "hooks_A.json").read_text(encoding="utf-8"))
    A5 = {(h["law_id"], h["art"]) for h in hooksA["A5"]}
    A8 = {(h["law_id"], h["art"]) for h in hooksA["A8"]}
    freq = Counter()
    for r in governing.values():
        g = r.get("governing")
        if g:
            freq[(g["law_id"], g["art"])] += 1

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
        """allowed=None は無制限（参照）。それ以外はフック集合。"""
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

    records = []
    mismatches = []
    for key in q249:
        year, no = key
        q = questions[key]
        gov = governing[key]["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])
        scores = engine.score_all(qtext)
        rec = {"year": year, "no": no, "governing": gov,
               "confidence": governing[key].get("confidence")}
        # --- 参照条件で忠実性検証 ---
        rref, _ = run_config(scores, govkey, None)
        rr = ref[key]
        ok = (rref["reached_strict"] == rr["reached_strict"]
              and rref["reached_family"] == rr["reached_family"]
              and rref["context_size"] == rr["context_size"]
              and [(s["law_id"], s["art"]) for s in rref["seeds"]]
                  == [(s["law_id"], s["art"]) for s in rr["seeds"]])
        if not ok:
            mismatches.append({"q": key, "mine": {k: rref[k] for k in ("reached_strict", "reached_family", "context_size")},
                               "ref": {k: rr[k] for k in ("reached_strict", "reached_family", "context_size")}})
        rec["reference"] = rref
        # --- フック条件 ---
        loo = Counter(freq)
        loo[govkey] -= 1
        familiar = {a for a, c in loo.items() if c >= 2}
        configs = {"A5": A5, "A8": A8, "B": A5 | familiar}
        rec["familiar_added_B"] = len(familiar - A5)
        rec["gov_is_hook"] = {}
        tiers_cache = {}
        for cname, allowed in configs.items():
            r, tier = run_config(scores, govkey, allowed)
            r["gov_is_hook"] = govkey in allowed
            if cname == "B":
                r["gov_via_familiar"] = govkey in familiar and govkey not in A5
            rec[cname] = r
            tiers_cache[cname] = tier
            rec["gov_is_hook"][cname] = govkey in allowed
        records.append(rec)
        print(year, no, "ref", rref["reached_strict"], rref["reached_family"],
              "| A5", rec["A5"]["reached_strict"], rec["A5"]["reached_family"],
              "| B", rec["B"]["reached_strict"], rec["B"]["reached_family"], flush=True)

    (OUT / "work" / "records_checkpoint.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print("VERIFICATION mismatches:", len(mismatches))
    for m in mismatches[:5]:
        print(m)

    # --- 未到達の内訳診断（family 未到達のみ） ---
    # 全フックの展開をキャッシュに揃える（橋フック存在判定用）
    for h in A5:
        graph.expand(*h)
    def diagnose(rec, cname, allowed_extra=None):
        govkey = (rec["governing"]["law_id"], rec["governing"]["art"])
        allowed = {"A5": A5, "A8": A8}.get(cname)
        if cname == "B":
            allowed = A5 | allowed_extra
        bridge_strict = bridge_family = False
        for h in allowed:
            ctx, _ = graph.expand(*h)
            if govkey in ctx:
                bridge_strict = True
            if any(fam_of(k2, govkey) for k2 in ctx):
                bridge_family = True
            if bridge_strict and bridge_family:
                break
        if bridge_family:
            return "entry_miss"       # 橋になるフックはあったが入口 top-3 が外した
        return "no_bridge"            # フック集合のどの文脈にも正解（家族）が無い
    for rec in records:
        year, no = rec["year"], rec["no"]
        govkey = (rec["governing"]["law_id"], rec["governing"]["art"])
        loo = Counter(freq); loo[govkey] -= 1
        familiar = {a for a, c in loo.items() if c >= 2}
        for cname in ("A5", "B"):
            if not rec[cname]["reached_family"]:
                rec[cname]["fail_kind"] = diagnose(rec, cname, familiar)
            else:
                rec[cname]["fail_kind"] = None

    def agg(rs, cname):
        n = len(rs)
        return {
            "n": n,
            "strict": sum(1 for r in rs if r[cname]["reached_strict"]),
            "family": sum(1 for r in rs if r[cname]["reached_family"]),
            "strict_pct": round(100 * sum(1 for r in rs if r[cname]["reached_strict"]) / n, 1),
            "family_pct": round(100 * sum(1 for r in rs if r[cname]["reached_family"]) / n, 1),
            "gov_is_hook_pct": (round(100 * sum(1 for r in rs if r[cname].get("gov_is_hook")) / n, 1) if cname != "reference" else None),
            "gov_in_seeds": sum(1 for r in rs if r[cname]["gov_in_seeds"]),
            "context_median": statistics.median([r[cname]["context_size"] for r in rs]),
            "context_mean": round(statistics.mean([r[cname]["context_size"] for r in rs]), 1),
            "tier_dist": dict(Counter(str(r[cname]["governing_tier"]) for r in rs if r[cname]["reached_strict"])),
            "fail_kinds": dict(Counter(r[cname].get("fail_kind") for r in rs if r[cname].get("fail_kind"))),
        }

    summary = {
        "verification": {"n_checked": len(q249), "mismatches": len(mismatches), "details": mismatches[:10]},
        "hook_sets": {"A5_size": len(A5), "A8_size": len(A8),
                      "B_size_range": None},
        "reference": agg(records, "reference"),
        "A5": agg(records, "A5"),
        "A8": agg(records, "A8"),
        "B": agg(records, "B"),
    }
    bsizes = []
    for rec in records:
        govkey = (rec["governing"]["law_id"], rec["governing"]["art"])
        bsizes.append(len(A5) + rec["familiar_added_B"])
    summary["hook_sets"]["B_size_range"] = [min(bsizes), max(bsizes)]
    summary["B_gov_via_familiar"] = sum(1 for r in records if r["B"].get("gov_via_familiar"))
    (OUT / "results_hook.json").write_text(json.dumps(
        {"meta": {"questions": "c_full 証拠ベース249問 (high/medium)",
                  "expansion": "stage3_measure.Graph.expand と同一規則 (tier0種/tier1逆引き受け手/tier2引用先/tier3引用元40件キャップ)",
                  "A5": "in-degree>=5 の374条文", "A8": "in-degree>=8 の146条文",
                  "B": "A5 + LOOおなじみ (当該問題を除く残りで2回以上正解)"},
         "summary": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
