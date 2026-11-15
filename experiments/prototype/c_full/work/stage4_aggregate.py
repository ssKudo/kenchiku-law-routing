#!/usr/bin/env python3
"""ステージ4: results.json を集計し、EVALUATION_c_full.md 用の数値と失敗分析を作る。"""
import json, statistics
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
res = data["results"]
meas = [r for r in res if r.get("reached_strict") is not None]
special = [r for r in res if r.get("special")]
unmeas = [r for r in res if r.get("measured") is False]

def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "-"

def rank_stats(rs, key):
    vals = [r[key] for r in rs if r.get(key)]
    if not vals:
        return {}
    return {"n": len(vals), "median": statistics.median(vals),
            "within10": sum(1 for v in vals if v <= 10), "within10_pct": pct(sum(1 for v in vals if v <= 10), len(rs)),
            "within20": sum(1 for v in vals if v <= 20)}

summary = {}
summary["total_questions"] = len(res)
summary["special"] = len(special)
summary["measured"] = len(meas)
summary["unmeasured"] = [(r["year"], r["no"], r.get("reason")) for r in unmeas]
summary["strict_reached"] = sum(1 for r in meas if r["reached_strict"])
summary["family_reached"] = sum(1 for r in meas if r["reached_family"])
summary["seed_top1_hit"] = sum(1 for r in meas if r["seed_top1_hit"])
summary["seed_top3_hit"] = sum(1 for r in meas if r["seed_top3_hit"])
summary["rank_hop_strict"] = rank_stats(meas, "rank_hop_strict")
summary["rank_tfidf_strict"] = rank_stats(meas, "rank_tfidf_strict")
summary["rank_hop_family"] = rank_stats(meas, "rank_hop_family")
summary["rank_tfidf_family"] = rank_stats(meas, "rank_tfidf_family")
summary["context_size_median"] = statistics.median([r["context_size"] for r in meas])
summary["capped_share"] = pct(sum(1 for r in meas if r["cited_by_capped"]), len(meas))

# 確信度別
by_conf = {}
for conf in ("high", "medium", "low"):
    rs = [r for r in meas if r.get("confidence") == conf]
    if rs:
        by_conf[conf] = {"n": len(rs),
                         "strict": sum(1 for r in rs if r["reached_strict"]),
                         "family": sum(1 for r in rs if r["reached_family"]),
                         "top1": sum(1 for r in rs if r["seed_top1_hit"]),
                         "top3": sum(1 for r in rs if r["seed_top3_hit"]),
                         "hop_med": statistics.median([r["rank_hop_strict"] for r in rs if r["rank_hop_strict"]] or [0]),
                         "tfidf_med": statistics.median([r["rank_tfidf_strict"] for r in rs if r["rank_tfidf_strict"]] or [0])}
summary["by_confidence"] = by_conf
ev = [r for r in meas if r.get("confidence") in ("high", "medium")]
summary["evidence_subset"] = {"n": len(ev),
                              "strict": sum(1 for r in ev if r["reached_strict"]),
                              "family": sum(1 for r in ev if r["reached_family"]),
                              "top1": sum(1 for r in ev if r["seed_top1_hit"]),
                              "top3": sum(1 for r in ev if r["seed_top3_hit"]),
                              "hop_med": statistics.median([r["rank_hop_strict"] for r in ev if r["rank_hop_strict"]] or [0]),
                              "tfidf_med": statistics.median([r["rank_tfidf_strict"] for r in ev if r["rank_tfidf_strict"]] or [0]),
                              "hop_within10": pct(sum(1 for r in ev if r["rank_hop_strict"] and r["rank_hop_strict"] <= 10), len(ev)),
                              "tfidf_within10": pct(sum(1 for r in ev if r["rank_tfidf_strict"] and r["rank_tfidf_strict"] <= 10), len(ev))}
# 年別
by_year = {}
for y in range(2017, 2027):
    rs = [r for r in meas if r["year"] == y]
    if rs:
        by_year[y] = {"n": len(rs), "strict": sum(1 for r in rs if r["reached_strict"]),
                      "family": sum(1 for r in rs if r["reached_family"])}
summary["by_year"] = by_year
# 正答条の法令別
by_law = {}
for r in meas:
    name = r["governing"]["law_name"]
    d = by_law.setdefault(name, {"n": 0, "strict": 0, "family": 0})
    d["n"] += 1; d["strict"] += 1 if r["reached_strict"] else 0; d["family"] += 1 if r["reached_family"] else 0
summary["by_governing_law"] = by_law
# 到達経路 (tier)
summary["reached_tier"] = Counter(str(r["governing_tier"]) for r in meas if r["reached_strict"])
# 未到達の内訳
fails = [r for r in meas if not r["reached_family"]]
summary["fail_count_family"] = len(fails)
summary["fails"] = [{"year": r["year"], "no": r["no"], "gov": f"{r['governing']['law_name']} {r['governing']['art']}条",
                     "conf": r.get("confidence"), "ctx": r["context_size"],
                     "capped": r["cited_by_capped"], "seed_top3": r["seed_top3_hit"],
                     "seeds": [f"{s['law_id'][:3]}:{s['art']}" for s in r["seeds"]]} for r in fails]
print(json.dumps(summary, ensure_ascii=False, indent=1))
(OUT / "work" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
