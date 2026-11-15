#!/usr/bin/env python3
"""アキネーター式パイロット 手順4・5: 絞り込みと採点 + oracle 天井診断。

- ここで初めて governing.jsonl を読む (採点専用)。
- プール = 看板パイロットの pools.json (各問 baseline 上位30条、TF-IDFスコア付き)。
- 一致数 = 条文タグ (yes/no) と回答 (○=yes/×=no) が一致した属性の数。
  回答が判定不能 (null) の属性は「全条文不一致」として扱う (記録に残す)。
- 主条件 (柔らかい並び替え): 一致数の降順、同点は baseline TF-IDF スコアの降順。
- oracle 条件: モデル回答の代わりに「正答条自身のタグ」を回答として使う
  (正答条の一致数は定義上 12/12 になる。天井の切り分け用)。
- 診断:
  (i) 正答条の一致数とプール最大一致数の差
  (ii) 一致数しきい値 k ごとの残存条数 (k=12..0) と、その範囲に正答条が残るか
  (iii) ハード消去: 回答とタグが食い違う条文を属性の固定順に1つずつ除外して
        いったとき、正答条が第何問目で消えるか (最後まで残れば 'survives')、
        および各ステップ後の残存条数
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
SIGN = Path("/home/hatch/workspace/research_kudou/signboard_pilot")

attrs = json.loads((OUT / "attributes.json").read_text(encoding="utf-8"))["attributes"]
ATTR_IDS = [a["id"] for a in attrs]
tags = {}
for line in (OUT / "tags.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        tags[(r["law_id"], r["art"])] = r["tags"]
pools = json.loads((SIGN / "work" / "pools.json").read_text(encoding="utf-8"))
answers_doc = json.loads((OUT / "answers.json").read_text(encoding="utf-8"))
model_answers = {}
for q in answers_doc["results"]:
    model_answers[(q["year"], q["no"])] = {r["attr_id"]: r["answer"] for r in q["answers"]}
governing = {}
for line in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        governing[(r["year"], r["no"])] = r["governing"]

def match_count(tag, ans):
    n = 0
    for aid in ATTR_IDS:
        a = ans.get(aid)
        if a is None:
            continue
        if tag[aid] == (a == "○"):
            n += 1
    return n

def rank_soft(pool, ans, govkey):
    scored = []
    for p in pool:
        key = (p["law"], p["art"])
        mc = match_count(tags[key], ans)
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
    ans = model_answers[(year, no)]
    base_rank = next(i + 1 for i, p in enumerate(sorted(pool, key=lambda p: -p["score"]))
                     if (p["law"], p["art"]) == govkey)

    scored, rank_model = rank_soft(pool, ans, govkey)
    gov_match = next(r["match"] for r in scored if r["is_gov"])
    pool_max_match = max(r["match"] for r in scored)
    n_at_max = sum(1 for r in scored if r["match"] == pool_max_match)

    gov_tags = tags[govkey]
    oracle_ans = {aid: ("○" if gov_tags[aid] else "×") for aid in ATTR_IDS}
    _, rank_oracle = rank_soft(pool, oracle_ans, govkey)

    curve = {}
    for k in range(len(ATTR_IDS), -1, -1):
        surv = [r for r in scored if r["match"] >= k]
        curve[str(k)] = {"survivors": len(surv),
                         "gov_survives": any(r["is_gov"] for r in surv)}

    survivors = {(p["law"], p["art"]) for p in pool}
    elim_step, elim_curve = "survives", []
    for i, aid in enumerate(ATTR_IDS, start=1):
        a = ans.get(aid)
        if a is not None:
            want = (a == "○")
            survivors = {k for k in survivors if tags[k][aid] == want}
        elim_curve.append({"step": i, "attr_id": aid, "answer": a, "survivors": len(survivors)})
        if govkey not in survivors and elim_step == "survives":
            elim_step = i

    results[qid] = {
        "governing": gov,
        "baseline_rank_in_pool": base_rank,
        "model_answers": ans,
        "akinator": {"rank": rank_model, "top3_hit": rank_model is not None and rank_model <= 3,
                     "gov_match": gov_match, "pool_max_match": pool_max_match,
                     "gov_match_gap": pool_max_match - gov_match,
                     "n_articles_at_max_match": n_at_max},
        "oracle": {"rank": rank_oracle, "top3_hit": rank_oracle is not None and rank_oracle <= 3},
        "threshold_curve": curve,
        "hard_elimination": {"gov_eliminated_at_step": elim_step, "steps": elim_curve},
        "ranking_model": [{"law": r["law"], "art": r["art"], "match": r["match"],
                           "baseline_score": round(r["baseline_score"], 2),
                           "is_gov": r["is_gov"]} for r in scored],
    }
    print("%s base=%s akinator=%s oracle=%s | gov_match=%d/%d gap=%d (max保持 %d条) | hard消去: %s" % (
        qid, base_rank, rank_model, rank_oracle, gov_match, pool_max_match,
        pool_max_match - gov_match, n_at_max, elim_step))

meta = {"answers_meta": answers_doc["meta"],
        "method": "プール内 (baseline上位30条) で、条文タグと回答の一致数の降順、同点は baseline TF-IDF スコア降順で並び替え。oracle は正答条自身のタグを回答に使った天井条件。",
        "invalid_answer_rule": "判定不能の属性は一致数に数えない (全条文にとって不一致扱い)。"}
(OUT / "pilot_results.json").write_text(
    json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved pilot_results.json")
