#!/usr/bin/env python3
"""二択の連続パイロット 手順4・5: 採点 + バイアス診断 + oracle 天井。

- ここで初めて governing.jsonl を読む (採点専用)。
- プール = 看板パイロットの pools.json (各問 baseline 上位30条、TF-IDFスコア付き)。
- 一致数 = 条文の側 (sides.jsonl、ラベル文字列または中立=null) とモデルの選択
  ラベルが一致した軸の数。中立の条文はその軸では一致しない。
  選択が形式崩れ (pick=null) の軸は全条文にとって不一致として扱う。
- 主条件 = 3回の多数決ラベル。多数決の決定規則:
  有効な選択のうち多い方。同数 (有効2回で1-1) の場合は rep1→rep2 の順に
  最初の有効な選択を採用し tie_broken として記録する。
  有効な選択が0回なら「選択なし」。
- 併記条件 (i) rep1 のみ (固定順AB) の選択で採点。
- oracle = 正答条自身の側を選択に使う (正答条の一致数は定義上、
  中立でない軸の数と等しくなる。天井の切り分け用)。
- 並び替え: 一致数の降順、同点は baseline TF-IDF スコアの降順。
- バイアス診断:
  * 順序入替一貫性: rep1(AB) と rep2(BA) で同じラベルを選んだ割合
    (両方有効なペアのみ)。参考に rep3(AB) と rep2(BA)、rep1 と rep3
    (同順の再現性) も出す。
  * 位置バイアス: 有効な選択全体で「最初に提示されたラベル」が
    選ばれた割合 (rep別も)。
  * ラベル分布: 軸ごとの選択ラベル数 (多数決・全rep)。
  * 形式崩れの数と内訳。
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
SIGN = Path("/home/hatch/workspace/research_kudou/signboard_pilot")

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

# --- モデル選択の整理: majority / rep1 ---
model = {}   # (year,no) -> axis_id -> {"majority": {...}, "rep1": label or None}
diag_pairs = {"rep1_vs_rep2": [], "rep3_vs_rep2": [], "rep1_vs_rep3": []}
first_pick_flags = []
first_pick_by_rep = {}
label_dist_majority = Counter()
label_dist_all = Counter()
fail_kinds = Counter()
for q in picks_doc["results"]:
    qm = {}
    for a in q["axes"]:
        reps = {r["rep"]: r for r in a["reps"]}
        valid = [(r["rep"], r["pick"]) for r in a["reps"] if r["pick"] is not None]
        for r in a["reps"]:
            if r["pick"] is not None:
                label_dist_all[(a["axis_id"], r["pick"])] += 1
                first_pick_flags.append(r["picked_presented_first"])
                first_pick_by_rep.setdefault(r["rep"], []).append(r["picked_presented_first"])
            else:
                fail_kinds[r.get("failure_kind", "unknown")] += 1
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

def match_count(side_map, choice_map):
    n = 0
    for aid in AX_IDS:
        s = side_map.get(aid)
        c = choice_map.get(aid)
        if s is not None and c is not None and s == c:
            n += 1
    return n

def rank_soft(pool, choice_map, govkey):
    scored = []
    for p in pool:
        key = (p["law"], p["art"])
        mc = match_count(sides[key], choice_map)
        scored.append({"law": p["law"], "art": p["art"], "baseline_score": p["score"],
                       "match": mc, "is_gov": key == govkey})
    scored.sort(key=lambda r: (-r["match"], -r["baseline_score"]))
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
    gov_match = next(r["match"] for r in scored_maj if r["is_gov"])
    pool_max = max(r["match"] for r in scored_maj)
    gov_sided_axes = sum(1 for aid in AX_IDS if gov_sides[aid] is not None)

    results[qid] = {
        "governing": gov,
        "baseline_rank_in_pool": base_rank,
        "majority_picks": maj_map,
        "majority_tie_broken": {aid: qm[aid]["majority_tie_broken"] for aid in AX_IDS},
        "rep1_picks": rep1_map,
        "reps_pick": {aid: qm[aid]["reps_pick"] for aid in AX_IDS},
        "governing_sides": gov_sides,
        "majority": {"rank": rank_maj, "top3_hit": rank_maj is not None and rank_maj <= 3,
                     "gov_match": gov_match, "pool_max_match": pool_max},
        "rep1_only": {"rank": rank_rep1, "top3_hit": rank_rep1 is not None and rank_rep1 <= 3},
        "oracle": {"rank": rank_oracle, "top3_hit": rank_oracle is not None and rank_oracle <= 3,
                   "gov_sided_axes": gov_sided_axes},
        "ranking_majority": [{"law": r["law"], "art": r["art"], "match": r["match"],
                              "baseline_score": round(r["baseline_score"], 2),
                              "is_gov": r["is_gov"]} for r in scored_maj],
    }
    print("%s base=%s majority=%s rep1=%s oracle=%s | gov_match=%d/%d (gov側あり軸 %d/6)" % (
        qid, base_rank, rank_maj, rank_rep1, rank_oracle, gov_match, pool_max, gov_sided_axes))

bias = {
    "consistency_rep1_vs_rep2_swapped": {"rate": frac(diag_pairs["rep1_vs_rep2"]),
                                          "pairs": len(diag_pairs["rep1_vs_rep2"])},
    "consistency_rep3_vs_rep2_swapped": {"rate": frac(diag_pairs["rep3_vs_rep2"]),
                                          "pairs": len(diag_pairs["rep3_vs_rep2"])},
    "reproducibility_rep1_vs_rep3_same_order": {"rate": frac(diag_pairs["rep1_vs_rep3"]),
                                                 "pairs": len(diag_pairs["rep1_vs_rep3"])},
    "picked_presented_first_rate_all": frac(first_pick_flags),
    "picked_presented_first_by_rep": {k: frac(v) for k, v in first_pick_by_rep.items()},
    "n_valid_picks": len(first_pick_flags),
    "format_failures_total": int(sum(fail_kinds.values())),
    "format_failure_kinds": dict(fail_kinds),
    "label_distribution_majority": {"%s|%s" % k: v for k, v in sorted(label_dist_majority.items())},
    "label_distribution_all_reps": {"%s|%s" % k: v for k, v in sorted(label_dist_all.items())},
}
print(json.dumps(bias, ensure_ascii=False, indent=1))

meta = {"picks_meta": picks_doc["meta"],
        "method": "プール内 (baseline上位30条) で、条文の側とモデルの選択ラベルの一致数の降順、同点は baseline TF-IDF スコア降順で並び替え。主条件は3回の多数決、併記は rep1 のみ。oracle は正答条自身の側を選択に使った天井条件。中立 (側なし) の条文はその軸で一致しない。",
        "bias": bias}
(OUT / "pilot_results.json").write_text(
    json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved pilot_results.json")
