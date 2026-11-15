#!/usr/bin/env python3
"""LFM3 思考抑制テスト: 空 think ブロックの prefill で最終回答が直接出るか。
対象: 選択v1の1問目 (2017-21) と スモークの1問目。
条件は本実行スクリプトと同一 (temp 0.2, seed 42)。prefill 以外は無変更。"""
import json, sys, time
from pathlib import Path

SEL = Path("/home/hatch/workspace/research_kudou/lfm3_test/selection")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(SEL))

PREFILL = "<think>\n\n</think>\n"

import selection_test as v1
import smoke_test as smoke

def with_prefill(prompt):
    assert prompt.endswith("<|im_start|>assistant\n"), prompt[-50:]
    return prompt + PREFILL

# --- 選択v1 1問目 ---
blind = [json.loads(l) for l in (SEL / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lists = {r["no"]: r["terms"] for r in json.loads((SEL / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
q = blind[0]
prompt = with_prefill(v1.build_prompt(q["stem"], q["choices"], lists[q["no"]]))
t0 = time.time()
out, elapsed, perf = v1.run_model(prompt)
ap = v1.answer_part(out)
parsed = v1.parse_picks(ap, len(lists[q["no"]]))
print("=== SEL v1 Q1 (%s-%s) ===" % (q["year"], q["no"]))
print("elapsed: %.1fs  perf: %s" % (elapsed, perf))
print("raw: %r" % out[:600])
print("answer_part: %r" % ap[:300])
print("parsed adopted: %s  parse_ok=%s" % (parsed["adopted_numbers"], parsed["parse_ok"]))
print("picked terms: %s" % [lists[q["no"]][n - 1] for n in parsed["adopted_numbers"]])

# --- スモーク 1問目 ---
questions = {(r["year"], r["no"]): r for r in
             (json.loads(l) for l in (CFULL / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
sel = smoke.SELECTED[0]
qq = questions[(sel["year"], sel["no"])]
prompt2 = with_prefill(smoke.build_prompt(qq["stem"], qq["choices"]))
out2, elapsed2, perf2 = smoke.run_model(prompt2)
terms, excluded = smoke.parse_terms(smoke.answer_part(out2))
print("=== SMOKE Q1 (%s-%s) ===" % (sel["year"], sel["no"]))
print("elapsed: %.1fs  perf: %s" % (elapsed2, perf2))
print("raw: %r" % out2[:600])
print("terms: %s  excluded_art: %s" % (terms, excluded))
