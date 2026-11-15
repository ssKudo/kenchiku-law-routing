#!/usr/bin/env python3
"""二択の連続パイロット v2 手順4・5: 採点 (一致率主スコア) + バイアス診断 + oracle 天井。

- ここで初めて governing.jsonl を読む (採点専用)。v1 の結果も比較用に読む。
- プール = 看板パイロットの pools.json (各問 baseline 上位30条、TF-IDFスコア付き)。
- v2 の主スコア = 一致率 = 一致した軸数 ÷ その条文が側を持つ軸数。
  側を持つ軸が0の条文は一致率 0 とする。生の一致数も併記する。
  並び替え: 一致率の降順、同率は baseline TF-IDF スコアの降順。
- 選択が形式崩れ (pick=null) の軸は全条文にとって不一致として扱う (v1 と同一)。
  中立の条文はその軸では一致しない (v1 と同一)。
- 主条件 = 3回の多数決 (v1 と同一の決定規則: 同数は rep 順に最初の有効選択)。
  併記条件 (i) rep1 のみ。
- oracle = 正答条自身の側を選択に使う。正答条の一致率は定義上 1.0
  (側のある軸が1つ以上ある場合)。
- バイアス診断 (v1 と直接比較可能な形):
  * 順序入替一貫性 rep1(AB) vs rep2(BA)、rep3(AB) vs rep2(BA)、
    同順再現性 rep1 vs rep3。
  * 位置バイアス: 有効選択全体と提示順別 (AB提示=rep1+rep3 / BA提示=rep2) の
    「最初に提示されたラベル」選択率。
  * ラベル分布 (多数決・全rep)、形式崩れの数・型 (軸別も)。
  * 正答条について軸ごとの一致/不一致/正答条が中立/選択なしの一覧。
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
SIGN = Path("/home/hatch/workspace/research_kudou/signboard_pilot")
V1 = Path("/home/hatch/workspace/research_kudou/binary_chain_pilot")

axes = json.loads((OUT / "axes.json").read_text(encoding="utf-8"))["axes"]
AX_IDS = [a["id"] for a in axes]
sides = {}
for line in (OUT / "sides.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        sides[(r["law_id"], r["art"])] = r["sides"]
pools = json.loads((SIGN / "work" / "pools.json").read_text(encoding="utf-8"))
picks_doc = json.loads((OUT / "picks.json").read_text(encoding="utf-8"))
governing = {}
for line in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        governing[(r["year"], r["no"])] = r["governing"]
v1_results = json.loads((V1 / "pilot_results.json").read_text(encoding="utf-8"))["results"]
v1_bias = json.loads((V1 / "pilot_results.json").read_text(encoding="utf-8"))["meta"]["bias"]

# --- モデル選択の整理と診断の集計 ---
model = {}
diag_pairs = {"rep1_vs_rep2": [], "rep3_vs_rep2": [], "rep1_vs_rep3": []}
first_pick_flags, first_by_order = [], {"AB": [], "BA": []}
first_by_rep = {}
label_dist_majority, label_dist_all = Counter(), Counter()
fail_kinds, fail_subkinds, fail_by_axis = Counter(), Counter(), Counter()
for q in picks_doc["results"]:
    qm = {}
    for a in q["axes"]:
        reps = {r["rep"]: r for r in a["reps"]}
        valid = [(r["rep"], r["pick"]) for r in a["reps"] if r["pick"] is not None]
        for r in a["reps"]:
            if r["pick"] is not None:
                label_dist_all[(a["axis_id"], r["pick"])] += 1
                first_pick_flags.append(r["picked_presented_first"])
                first_by_order[r["order"]].append(r["picked_presented_first"])
                first_by_rep.setdefault(r["rep"], []).append(r["picked_presented_first"])
            else:
                fail_kinds[r.get("failure_kind", "unknown")] += 1
                fail_subkinds[r.get("failure_subkind") or r.get("failure_kind", "unknown")] += 1
                fail_by_axis[a["axis_id"]] += 1
        if valid:
            cnt = Counter(p for _, p in valid)
            top = max(cnt.values())
            winners = [lab for lab, c in cnt.items() if c == top]
            if len(winners) == 1:
                maj, tie_broken = winners[0], False
            else:
                maj, tie_broken = valid[0][1], True
        else:
            maj, tie_broken = None, False
        if maj is not None:
            label_dist_majority[(a["axis_id"], maj)] += 1
        qm[a["axis_id"]] = {"majority": maj, "majority_tie_broken": tie_broken,
                            "rep1": reps["rep1"]["pick"],
                            "reps_pick": {r["rep"]: r["pick"] for r in a["reps"]}}
        p1, p2, p3 = reps["rep1"]["pick"], reps["rep2"]["pick"], reps["rep3"]["pick"]
        if p1 is not None and p2 is not None:
            diag_pairs["rep1_vs_rep2"].append(p1 == p2)
        if p3 is not None and p2 is not None:
            diag_pairs["rep3_vs_rep2"].append(p3 == p2)
        if p1 is not None and p3 is not None:
            diag_pairs["rep1_vs_rep3"].append(p1 == p3)
    model[(q["year"], q["no"])] = qm

def frac(flags):
    return round(sum(flags) / len(flags), 3) if flags else None

def match_stats(side_map, choice_map):
    sided = sum(1 for aid in AX_IDS if side_map.get(aid) is not None)
    m = 0
    for aid in AX_IDS:
        s, c = side_map.get(aid), choice_map.get(aid)
        if s is not None and c is not None and s == c:
            m += 1
    rate = (m / sided) if sided > 0 else 0.0
    return m, sided, rate

def rank_soft(pool, choice_map, govkey):
    scored = []
    for p in pool:
        key = (p["law"], p["art"])
        m, sided, rate = match_stats(sides[key], choice_map)
        scored.append({"law": p["law"], "art": p["art"], "baseline_score": p["score"],
                       "match": m, "sided": sided, "rate": round(rate, 4),
                       "is_gov": key == govkey})
    scored.sort(key=lambda r: (-r["rate"], -r["baseline_score"]))
    rank = next((i + 1 for i, r in enumerate(scored) if r["is_gov"]), None)
    return scored, rank

results = {}
for qid, pool in pools.items():
    year, no = (int(x) for x in qid.split("-"))
    gov = governing[(year, no)]
    govkey = (gov["law_id"], gov["art"])
    base_rank = next(i + 1 for i, p in enumerate(sorted(pool, key=lambda p: -p["score"]))
                     if (p["law"], p["art"]) == govkey)
    qm = model[(year, no)]
    maj_map = {aid: qm[aid]["majority"] for aid in AX_IDS}
    rep1_map = {aid: qm[aid]["rep1"] for aid in AX_IDS}
    gov_sides = sides[govkey]
    oracle_map = {aid: gov_sides[aid] for aid in AX_IDS}

    scored_maj, rank_maj = rank_soft(pool, maj_map, govkey)
    _, rank_rep1 = rank_soft(pool, rep1_map, govkey)
    _, rank_oracle = rank_soft(pool, oracle_map, govkey)
    gov_row = next(r for r in scored_maj if r["is_gov"])
    pool_max_rate = max(r["rate"] for r in scored_maj)

    axis_detail = {}
    for aid in AX_IDS:
        gs, pick = gov_sides[aid], maj_map[aid]
        if gs is None:
            status = "gov_neutral"
        elif pick is None:
            status = "no_pick"
        elif gs == pick:
            status = "match"
        else:
            status = "mismatch"
        axis_detail[aid] = {"gov_side": gs, "model_majority": pick, "status": status}

    v1q = v1_results[qid]
    results[qid] = {
        "governing": gov,
        "baseline_rank_in_pool": base_rank,
        "v1_majority_rank": v1q["majority"]["rank"],
        "v1_oracle_rank": v1q["oracle"]["rank"],
        "majority_picks": maj_map,
        "majority_tie_broken": {aid: qm[aid]["majority_tie_broken"] for aid in AX_IDS},
        "rep1_picks": rep1_map,
        "reps_pick": {aid: qm[aid]["reps_pick"] for aid in AX_IDS},
        "governing_sides": gov_sides,
        "axis_detail_majority": axis_detail,
        "majority": {"rank": rank_maj, "top3_hit": rank_maj is not None and rank_maj <= 3,
                     "gov_match": gov_row["match"], "gov_sided": gov_row["sided"],
                     "gov_rate": gov_row["rate"], "pool_max_rate": pool_max_rate},
        "rep1_only": {"rank": rank_rep1, "top3_hit": rank_rep1 is not None and rank_rep1 <= 3},
        "oracle": {"rank": rank_oracle, "top3_hit": rank_oracle is not None and rank_oracle <= 3,
                   "gov_sided_axes": gov_row["sided"]},
        "ranking_majority": [{"law": r["law"], "art": r["art"], "match": r["match"],
                              "sided": r["sided"], "rate": r["rate"],
                              "baseline_score": round(r["baseline_score"], 2),
                              "is_gov": r["is_gov"]} for r in scored_maj],
    }
    print("%s base=%s v1=%s v2maj=%s v2rep1=%s v2oracle=%s | gov %d/%d軸一致 rate=%.3f (pool max %.3f)" % (
        qid, base_rank, v1q["majority"]["rank"], rank_maj, rank_rep1, rank_oracle,
        gov_row["match"], gov_row["sided"], gov_row["rate"], pool_max_rate))

bias = {
    "consistency_rep1_vs_rep2_swapped": {"rate": frac(diag_pairs["rep1_vs_rep2"]),
                                          "pairs": len(diag_pairs["rep1_vs_rep2"])},
    "consistency_rep3_vs_rep2_swapped": {"rate": frac(diag_pairs["rep3_vs_rep2"]),
                                          "pairs": len(diag_pairs["rep3_vs_rep2"])},
    "reproducibility_rep1_vs_rep3_same_order": {"rate": frac(diag_pairs["rep1_vs_rep3"]),
                                                 "pairs": len(diag_pairs["rep1_vs_rep3"])},
    "picked_presented_first_rate_all": frac(first_pick_flags),
    "picked_presented_first_by_order": {k: frac(v) for k, v in first_by_order.items()},
    "picked_presented_first_by_rep": {k: frac(v) for k, v in first_by_rep.items()},
    "n_valid_picks": len(first_pick_flags),
    "format_failures_total": int(sum(fail_kinds.values())),
    "format_failure_kinds": dict(fail_kinds),
    "format_failure_subkinds": dict(fail_subkinds),
    "format_failures_by_axis": dict(fail_by_axis),
    "label_distribution_majority": {"%s|%s" % k: v for k, v in sorted(label_dist_majority.items())},
    "label_distribution_all_reps": {"%s|%s" % k: v for k, v in sorted(label_dist_all.items())},
    "v1_reference": {
        "format_failures_total": v1_bias["format_failures_total"],
        "format_failure_kinds": v1_bias["format_failure_kinds"],
        "picked_presented_first_rate_all": v1_bias["picked_presented_first_rate_all"],
        "picked_presented_first_by_rep": v1_bias["picked_presented_first_by_rep"],
        "consistency_rep1_vs_rep2_swapped": v1_bias["consistency_rep1_vs_rep2_swapped"],
        "n_valid_picks": v1_bias["n_valid_picks"],
    },
}
print(json.dumps(bias, ensure_ascii=False, indent=1))

meta = {"picks_meta": picks_doc["meta"],
        "method": "プール内 (baseline上位30条) で、主スコア=一致率 (一致軸数÷条文が側を持つ軸数)、同率は baseline TF-IDF スコア降順で並び替え。生の一致数も併記。主条件は3回の多数決、併記は rep1 のみ。oracle は正答条自身の側を選択に使った天井条件。中立 (側なし) の条文はその軸で一致しない。形式崩れの軸は全条文にとって不一致。",
        "bias": bias}
(OUT / "pilot_results.json").write_text(
    json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved pilot_results.json")
