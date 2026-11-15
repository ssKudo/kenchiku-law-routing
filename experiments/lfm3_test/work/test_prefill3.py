#!/usr/bin/env python3
"""思考抑制テスト第3弾: (A+B) 強い指示 + <think> logit-bias 禁止 の併用。
対象: 選択v1の1問目 (2017-21) とスモーク1問目。生成上限はテスト用 400。"""
import json, subprocess, sys, time
from pathlib import Path

SEL = Path("/home/hatch/workspace/research_kudou/lfm3_test/selection")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(SEL))
import selection_test as v1
import smoke_test as smoke

THINK_ID = 124901
STRONG = "思考過程（<think>タグや内的な推論）は一切書かず、最終回答だけをそのまま出力してください。"

def run(llama, model, prompt, temp, seed, n=400):
    pf = SEL / "_prompt_tmp_test.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run([llama, "-m", model, "-f", str(pf), "-n", str(n),
                         "--temp", str(temp), "--seed", str(seed), "-t", "4",
                         "--no-display-prompt", "--logit-bias", "%d-100" % THINK_ID],
                        capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    el = time.time() - t0
    out = cp.stdout.split("<|im_end|>")[0].split("> EOF")[0].strip()
    return out, el

blind = [json.loads(l) for l in (SEL / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lists = {r["no"]: r["terms"] for r in json.loads((SEL / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
q = blind[0]
prompt = v1.build_prompt(q["stem"], q["choices"], lists[q["no"]]).replace(v1.SYSTEM, v1.SYSTEM + STRONG)
out, el = run(v1.LLAMA, v1.MODEL, prompt, v1.TEMP, v1.SEED)
pa = v1.parse_picks(v1.answer_part(out), len(lists[q["no"]]))
print("=== (A+B) SEL v1 Q1 ===")
print("elapsed %.1fs | contains_think=%s" % (el, "<think>" in out))
print("raw: %r" % out[:500])
print("adopted: %s" % pa["adopted_numbers"])

questions = {(r["year"], r["no"]): r for r in
             (json.loads(l) for l in (CFULL / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
sel = smoke.SELECTED[0]
qq = questions[(sel["year"], sel["no"])]
prompt2 = smoke.build_prompt(qq["stem"], qq["choices"]).replace(smoke.SYSTEM, smoke.SYSTEM + STRONG)
out2, el2 = run(smoke.LLAMA if hasattr(smoke, "LLAMA") else v1.LLAMA, v1.MODEL, prompt2, smoke.TEMP, smoke.SEED)
terms, excluded = smoke.parse_terms(smoke.answer_part(out2))
print("=== (A+B) SMOKE Q1 (%s-%s) ===" % (sel["year"], sel["no"]))
print("elapsed %.1fs | contains_think=%s" % (el2, "<think>" in out2))
print("raw: %r" % out2[:500])
print("terms: %s | excluded: %s" % (terms, excluded))
