#!/usr/bin/env python3
"""Step 5 集計： CE スコア（日本語 hotchpotch／保存済み bge）と連結スタックの
Min-Max 正規化＋重み付き和。全パターンの top-3/平均順位と動きの内訳を出す。
- 混合： S = w_lex·stack_norm + w_ce·CE_norm、(w_lex, w_ce) ∈ {(0.8,0.2),(0.7,0.3),(0.6,0.4)}
- 参照： baseline 29/3.88、連結スタック 31/3.38、現行最良 stack×ruri 33/3.26〜3.33、bge単独 13/9.33
"""
import json, statistics
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
RB = RK / "rerank_blend"

pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
gov = {}
for line in open(RK / "prototype/c_full/governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d.get("governing"):
        gov[f'{d["year"]}-{d["no"]}'] = (d["governing"]["law_id"], d["governing"]["art"])
QIDS = list(pools.keys())


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


lex = json.load(open(RB / "work/lex_scores.json", encoding="utf-8"))


def deser(m):
    return {qid: {tuple(k.split("|", 1)): v for k, v in m[qid].items()} for qid in QIDS}


stack_scores = deser(lex["stack"])
base_scores = deser(lex["base"])
ruri_scores = deser(lex["ruri"])
bge_scores = deser(lex["bge"])

jp_raw = json.load(open(RB / "work/scores_jp_partial.json", encoding="utf-8"))
jp_scores = {}
for qid in QIDS:
    jp_scores[qid] = {}
    for ks, v in jp_raw[qid]["scores"].items():
        law, art = ks.split("|", 1)
        jp_scores[qid][(law, art)] = v


def rank_from_scores(scores, keys, target):
    ordered = sorted(keys, key=lambda k: (-scores.get(k, 0.0), keys.index(k)))
    return ordered.index(target) + 1


def ranks_of(score_map):
    out = {}
    for qid in QIDS:
        t = gov[qid]
        keys = pool_keys(qid)
        out[qid] = rank_from_scores(score_map[qid], keys, t) if t in keys else None
    return out


def agg(ranks):
    top3 = sum(1 for r in ranks.values() if r is not None and r <= 3)
    inp = [r for r in ranks.values() if r is not None]
    return top3, round(statistics.mean(inp), 2)


def minmax(scores, keys):
    vals = [scores.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in keys}
    return {k: (scores.get(k, 0.0) - lo) / (hi - lo) for k in keys}


def movement(new_ranks, ref_ranks):
    ent = [q for q in QIDS if new_ranks[q] is not None and new_ranks[q] <= 3
           and (ref_ranks[q] is None or ref_ranks[q] > 3)]
    ext = [q for q in QIDS if ref_ranks[q] is not None and ref_ranks[q] <= 3
           and (new_ranks[q] is None or new_ranks[q] > 3)]
    imp = [q for q in QIDS if new_ranks[q] is not None and ref_ranks[q] is not None
           and new_ranks[q] < ref_ranks[q]]
    wor = [q for q in QIDS if new_ranks[q] is not None and ref_ranks[q] is not None
           and new_ranks[q] > ref_ranks[q]]
    return {"entries": ent, "exits": ext, "improved": len(imp), "worsened": len(wor),
            "improved_q": imp, "worsened_q": wor}


base_ranks = ranks_of(base_scores)
stack_ranks = ranks_of(stack_scores)
ruri_ranks = ranks_of(ruri_scores)
bge_ranks = ranks_of(bge_scores)
jp_ranks = ranks_of(jp_scores)
assert agg(base_ranks) == (29, 3.88), agg(base_ranks)
assert agg(stack_ranks) == (31, 3.38), agg(stack_ranks)
assert agg(bge_ranks) == (13, 9.33), agg(bge_ranks)

# 現行最良（stack×ruri α=0.85）の順位を再構成（prep で全問一致を検証済み）
def fuse_map2(lex_m, sem_m, alpha):
    out = {}
    for qid in QIDS:
        keys = pool_keys(qid)
        ln = minmax(lex_m[qid], keys)
        sn = minmax(sem_m[qid], keys)
        out[qid] = {k: alpha * ln[k] + (1 - alpha) * sn[k] for k in keys}
    return out


best_ranks = ranks_of(fuse_map2(stack_scores, ruri_scores, 0.85))
assert agg(best_ranks) == (33, 3.33), agg(best_ranks)

WEIGHTS = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
results = {"meta": {
    "task": "Step 5: 日本語リランカー混合版（提案⑤の再戦）",
    "arena": "fifty 50問・原文TF-IDF上位30プール",
    "lexical": "連結スタック（原文＋単語タグ＋メタデータ）生スコア（再検証済み 31/3.38）",
    "ce_jp": "hotchpotch/japanese-reranker-cross-encoder-small-v1（XLMRobertaForSequenceClassification, MiniLM-L12-H384, 約118M）公式ONNX無し→torch読込し自前でONNX変換、ORT実行。ORT vs torch 最大差1e-6",
    "ce_bge_fallback": "rerank_ce 保存済み bge-reranker-v2-m3 ペアスコア（単独 13/9.33 を再現照合済み）",
    "windowing": "rerank_ce と同一： [s] q [/s][/s] chunk [/s]、上限512、窓幅=512-(問題トークン+4)、重なりなし、最大4窓・等間隔サンプリング（先頭含む）、条文スコア=窓最大値",
    "fusion": "プール内 Min-Max 正規化 → S = w_lex·stack + w_ce·CE",
    "references": {"baseline": agg(base_ranks), "stack": agg(stack_ranks),
                   "stack_x_ruri_best": {"alpha0.80": [33, 3.26], "alpha0.85": [33, 3.33], "alpha0.90": [33, 3.31]},
                   "bge_solo": agg(bge_ranks), "ceiling_top3": 42},
}, "cells": {}, "questions": {}}

cells = {}
solo = {"jpCE": jp_ranks, "bgeCE": bge_ranks}
for name, r in solo.items():
    cells[f"{name}_solo"] = {"top3": agg(r)[0], "avg": agg(r)[1],
                             "vs_stack": movement(r, stack_ranks)}
for ce_name, ce_scores in [("jpCE", jp_scores), ("bgeCE", bge_scores)]:
    for w_lex, w_ce in WEIGHTS:
        fr = ranks_of(fuse_map2(stack_scores, ce_scores, w_lex))
        cname = f"stackx{ce_name}@{w_lex}/{w_ce}"
        cells[cname] = {"top3": agg(fr)[0], "avg": agg(fr)[1],
                        "vs_stack": movement(fr, stack_ranks),
                        "vs_current_best": movement(fr, best_ranks)}
        for qid in QIDS:
            results["questions"].setdefault(qid, {})[cname] = fr[qid]

results["cells"] = cells
for qid in QIDS:
    results["questions"][qid].update({
        "governing": [gov[qid][0], gov[qid][1]],
        "in_pool": gov[qid] in pool_keys(qid),
        "rank_baseline": base_ranks[qid], "rank_stack": stack_ranks[qid],
        "rank_ruri_solo": ruri_ranks[qid], "rank_best_stackxruri085": best_ranks[qid],
        "rank_jpCE_solo": jp_ranks[qid], "rank_bgeCE_solo": bge_ranks[qid],
        "jpCE_score_gov": jp_scores[qid].get(gov[qid]),
        "bge_score_gov": bge_scores[qid].get(gov[qid])})

json.dump(results, open(RB / "results_rerank_blend.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("=== 単独 ===")
for n in ["jpCE_solo", "bgeCE_solo"]:
    c = cells[n]
    print(f"{n}: top3={c['top3']} avg={c['avg']} vs_stack 入{c['vs_stack']['entries']} 出{c['vs_stack']['exits']} 改善{c['vs_stack']['improved']} 悪化{c['vs_stack']['worsened']}")
print("=== 混合（vs スタック 31/3.38、現行最良 33/3.26〜3.33） ===")
for w_lex, w_ce in WEIGHTS:
    for ce in ["jpCE", "bgeCE"]:
        c = cells[f"stackx{ce}@{w_lex}/{w_ce}"]
        print(f"stackx{ce} {w_lex}/{w_ce}: top3={c['top3']} avg={c['avg']} "
              f"入{c['vs_stack']['entries']} 出{c['vs_stack']['exits']} 改善{c['vs_stack']['improved']} 悪化{c['vs_stack']['worsened']}")
print("jpCE solo 集計:", agg(jp_ranks))
