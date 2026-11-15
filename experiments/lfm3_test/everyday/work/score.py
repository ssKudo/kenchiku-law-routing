#!/usr/bin/env python3
"""日常文サニティチェック集計: 多数決正解率・形式崩れ・入替一貫性・位置バイアス。

法令実験 (binary_chain_pilot v1/v2) と同じ定義で集計する:
- 多数決: 3回の有効選択の過半数。同数 (1-1+崩れ等) は「多数決なし」。
- 入替一貫性: rep1 (AB) と rep2 (BA) の両方に有効選択があるペアで、同じラベルを選んだ割合。
- 位置バイアス: 有効選択のうち「最初に提示されたラベル」を選んだ割合 (全体 / AB提示 / BA提示)。
"""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent

def majority(q):
    sides = [r["pick_side"] for r in q["reps"] if r["pick_side"] is not None]
    if not sides:
        return None
    c = Counter(sides)
    top, n = c.most_common(1)[0]
    return top if n >= 2 else None

def rate(a, b):
    return round(a / b, 3) if b else None

def main():
    data = json.loads((OUT / "picks.json").read_text(encoding="utf-8"))
    cells = []
    for it in data["results"]:
        for q in it["questions"]:
            maj = majority(q)
            cells.append({
                "qid": q["qid"], "type": q["type"], "text_id": it["id"],
                "label_A": q["label_A"], "label_B": q["label_B"],
                "correct_side": q["correct_side"], "correct_label": q["correct_label"],
                "majority_side": maj,
                "majority_correct": (maj is not None and maj == q["correct_side"]),
                "valid_reps": sum(1 for r in q["reps"] if r["pick_side"] is not None),
                "rep_correct": [r["correct"] for r in q["reps"]],
                "reps": q["reps"],
            })

    def acc(subset):
        decided = [c for c in subset if c["majority_side"] is not None]
        return {"cells": len(subset), "decided": len(decided),
                "correct": sum(1 for c in decided if c["majority_correct"]),
                "accuracy_decided": rate(sum(1 for c in decided if c["majority_correct"]), len(decided)),
                "accuracy_all_cells": rate(sum(1 for c in subset if c["majority_correct"]), len(subset))}

    summary = {
        "overall": acc(cells),
        "concrete": acc([c for c in cells if c["type"] == "concrete"]),
        "abstract": acc([c for c in cells if c["type"] == "abstract"]),
    }

    all_reps = [r for c in cells for r in c["reps"]]
    valid = [r for r in all_reps if r["pick_side"] is not None]
    summary["reps"] = {
        "n_calls": len(all_reps),
        "valid": len(valid),
        "format_failures": sum(1 for r in all_reps if r["format_failure"]),
        "format_failure_rate": rate(sum(1 for r in all_reps if r["format_failure"]), len(all_reps)),
        "failure_kinds": dict(Counter(r.get("failure_kind") for r in all_reps if r["format_failure"])),
        "format_clean_rate": rate(sum(1 for r in valid if r["format_clean"]), len(valid)),
        "dirty_picks": sum(1 for r in valid if not r["format_clean"]),
        "echo_outputs": sum(1 for r in all_reps if r["echo"]),
        "single_rep_accuracy": rate(sum(1 for r in valid if r["correct"]), len(valid)),
    }

    # 入替一貫性 rep1 vs rep2 / 同順再現性 rep1 vs rep3
    def consistency(i, j):
        pairs = []
        for c in cells:
            a, b = c["reps"][i], c["reps"][j]
            if a["pick_side"] is not None and b["pick_side"] is not None:
                pairs.append(a["pick_side"] == b["pick_side"])
        return {"pairs": len(pairs), "same": sum(pairs), "rate": rate(sum(pairs), len(pairs))}
    summary["swap_consistency_rep1_rep2"] = consistency(0, 1)
    summary["same_order_repro_rep1_rep3"] = consistency(0, 2)

    # 位置バイアス
    def first_rate(reps):
        v = [r for r in reps if r["pick_side"] is not None]
        return {"n": len(v), "picked_first": sum(1 for r in v if r["picked_presented_first"]),
                "rate": rate(sum(1 for r in v if r["picked_presented_first"]), len(v))}
    summary["position_bias"] = {
        "overall": first_rate(all_reps),
        "AB_presentations": first_rate([r for r in all_reps if r["order"] == "AB"]),
        "BA_presentations": first_rate([r for r in all_reps if r["order"] == "BA"]),
    }

    # ラベル選択の偏り (正解側との一致を除いた生の偏り): A側/B側の選択回数
    summary["side_counts"] = dict(Counter(r["pick_side"] for r in valid))

    wrong = [c for c in cells if not c["majority_correct"]]
    summary["wrong_or_undecided_cells"] = [
        {"qid": c["qid"], "type": c["type"], "labels": [c["label_A"], c["label_B"]],
         "correct_label": c["correct_label"], "majority_side": c["majority_side"],
         "rep_picks": [r["pick"] for r in c["reps"]],
         "rep_raw": [r["raw_output"] for r in c["reps"]]} for c in wrong]

    (OUT / "scores.json").write_text(json.dumps({"summary": summary,
        "cells": [{k: c[k] for k in ("qid", "type", "label_A", "label_B", "correct_side",
                                     "correct_label", "majority_side", "majority_correct",
                                     "valid_reps", "rep_correct")} for c in cells]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
