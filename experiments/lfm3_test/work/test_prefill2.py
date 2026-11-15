#!/usr/bin/env python3
"""思考抑制テスト第2弾: (A) 強い指示+prefill / (B) <think> トークンの logit-bias 禁止。
対象は選択v1の1問目 (2017-21)。生成上限はテスト用に 400 トークン。
成功基準: 最終回答 (番号列) が実際に出て、時間が大幅に下がること。"""
import json, re, subprocess, sys, time
from pathlib import Path

SEL = Path("/home/hatch/workspace/research_kudou/lfm3_test/selection")
sys.path.insert(0, str(SEL))
import selection_test as v1

LLAMA = v1.LLAMA; MODEL = v1.MODEL
PREFILL = "<think>\n\n</think>\n"
THINK_ID = 124901

blind = [json.loads(l) for l in (SEL / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lists = {r["no"]: r["terms"] for r in json.loads((SEL / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
q = blind[0]
base_prompt = v1.build_prompt(q["stem"], q["choices"], lists[q["no"]])

def run(prompt, extra_args, n=400):
    pf = SEL / "_prompt_tmp_test.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run([LLAMA, "-m", MODEL, "-f", str(pf), "-n", str(n),
                         "--temp", str(v1.TEMP), "--seed", str(v1.SEED), "-t", "4",
                         "--no-display-prompt"] + extra_args,
                        capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    el = time.time() - t0
    out = cp.stdout.split("<|im_end|>")[0].split("> EOF")[0].strip()
    return out, el

# (A) 強い指示 + prefill: SYSTEM に一文追加した版を build する
strong_system = v1.SYSTEM + "思考過程（<think>タグや内的な推論）は一切書かず、最終回答だけをそのまま出力してください。"
prompt_a = base_prompt.replace(v1.SYSTEM, strong_system) + PREFILL
out_a, el_a = run(prompt_a, [])
ap_a = v1.answer_part(out_a)
pa = v1.parse_picks(ap_a, len(lists[q["no"]]))
print("=== (A) strong instruction + prefill ===")
print("elapsed %.1fs | starts_with_think=%s" % (el_a, out_a.startswith("<think>")))
print("raw: %r" % out_a[:400])
print("answer_part: %r | adopted: %s" % (ap_a[:200], pa["adopted_numbers"]))

# (B) logit-bias で <think> トークンを禁止 (プロンプトは完全無変更)
out_b, el_b = run(base_prompt, ["--logit-bias", "%d-100" % THINK_ID])
ap_b = v1.answer_part(out_b)
pb = v1.parse_picks(ap_b, len(lists[q["no"]]))
print("=== (B) logit-bias ban <think> ===")
print("elapsed %.1fs | contains_think=%s" % (el_b, "<think>" in out_b))
print("raw: %r" % out_b[:400])
print("answer_part: %r | adopted: %s" % (ap_b[:200], pb["adopted_numbers"]))
