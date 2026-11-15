"""Qwen2.5-1.5B 分岐条件（クドウの追加指示, 2026-11-14）。

各問について、プール30条文の看板v2を番号付きで Qwen に提示し、
問題文に合う看板を番号で最大3つ選ばせる。
測るのは「正解条文の看板が選ばれた組に入るか（＝分岐で正解が生き残るか）」。

実行環境は small_model_test/selection_test.py と同一:
llama-completion, ChatML, temp 0.2, seed 42, threads 4。
看板30件でプロンプトが長いため -c 8192 を明示（生成条件は不変）。
パース規則も selection_test.py と同一（全角→半角、範囲外は記録して不採用、
重複除去、3件超は先頭3件を採用し truncated を記録）。
"""
import json, re, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE1 = Path("/home/hatch/workspace/research_kudou/signboard_pilot")
BASE2 = Path("/home/hatch/workspace/research_kudou/signboard_pilot2")

LLAMA = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-completion"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
THREADS = 4
MAX_TOKENS = 128
TEMP = 0.2
SEED = 42
CTX = 8192
MAX_PICKS = 3

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題です。この問題の答えの根拠になる条文を、条文の「看板」（条文をやさしい言葉で説明したもの）の一覧の中から探したいです。

問題文:
{stem}

選択肢:
{choices}

看板の一覧:
{boards}

指示:
- 上の一覧の中から、この問題の根拠になる条文の看板を、番号で最大3個まで選んでください。
- 出力は番号だけをカンマ区切りで書いてください（例: 3, 7, 12）。説明や文章は書かないでください。"""


def build_prompt(stem, choices, boards):
    blist = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(boards))
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), boards=blist)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def run_model(prompt):
    pf = HERE / "_prompt_tmp_branch.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run(
        [LLAMA, "-m", MODEL, "-f", str(pf), "-n", str(MAX_TOKENS), "-c", str(CTX),
         "--temp", str(TEMP), "--seed", str(SEED), "-t", str(THREADS),
         "--no-display-prompt"],
        capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    out = cp.stdout.split("<|im_end|>")[0].split("> EOF")[0].strip()
    perf = {}
    for line in cp.stderr.splitlines():
        mp = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", line)
        if mp:
            perf["prompt_tokens"] = int(mp.group(2)); perf["prompt_tokens_per_sec"] = float(mp.group(3)); continue
        mg = re.search(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second", line)
        if mg:
            perf["gen_tokens"] = int(mg.group(2)); perf["eval_tokens_per_sec"] = float(mg.group(3))
    return out, elapsed, perf


FW = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_picks(raw, n_items):
    nums = [int(x) for x in re.findall(r"\d+", raw.translate(FW))]
    valid, invalid, seen = [], [], set()
    for n in nums:
        if 1 <= n <= n_items:
            if n not in seen:
                seen.add(n); valid.append(n)
        else:
            invalid.append(n)
    truncated = len(valid) > MAX_PICKS
    return {"raw_numbers": nums, "valid_unique": valid, "invalid_numbers": invalid,
            "truncated": truncated, "adopted_numbers": valid[:MAX_PICKS],
            "parse_ok": len(nums) > 0}


def main():
    pools = json.load(open(BASE1 / "work/pools.json", encoding="utf-8"))
    sb2 = {}
    for line in open(BASE2 / "signboards_v2.jsonl", encoding="utf-8"):
        d = json.loads(line)
        sb2[(d["law_id"], d["art"])] = d
    qs = {}
    for line in open("/home/hatch/workspace/research_kudou/small_model_test/blind_questions.jsonl", encoding="utf-8"):
        d = json.loads(line)
        qs[f'{d["year"]}-{d["no"]}'] = d
    gov = {}
    for line in open("/home/hatch/workspace/research_kudou/prototype/c_full/governing.jsonl", encoding="utf-8"):
        d = json.loads(line)
        gov[f'{d["year"]}-{d["no"]}'] = d

    recs = []
    for qid in ["2017-21", "2017-22", "2017-24", "2017-4", "2017-1"]:
        q = qs[qid]
        entries = pools[qid]
        keys = [(e["law"], e["art"]) for e in entries]
        g = gov[qid]["governing"]
        govkey = (g["law_id"], g["art"])
        gov_no = keys.index(govkey) + 1
        boards = []
        for k in keys:
            s = sb2[k]
            boards.append(s["summary"] + "（" + "・".join(s["keywords"]) + "）")
        prompt = build_prompt(q["stem"], q["choices"], boards)
        raw_out, elapsed, perf = run_model(prompt)
        parsed = parse_picks(raw_out, len(keys))
        adopted_keys = [keys[n - 1] for n in parsed["adopted_numbers"]]
        survived = govkey in adopted_keys
        survived_untruncated = gov_no in parsed["valid_unique"]
        rec = {
            "qid": qid,
            "governing": {"law_id": g["law_id"], "law_name": g.get("law_name"), "art": g["art"]},
            "governing_board_no": gov_no,
            "pool_keys_in_order": [{"law_id": k[0], "art": k[1]} for k in keys],
            "model_raw_output": raw_out,
            "parse": parsed,
            "adopted_keys": [{"law_id": k[0], "art": k[1]} for k in adopted_keys],
            "governing_survived_in_adopted": survived,
            "governing_in_valid_unique_before_truncation": survived_untruncated,
            "model_seconds": round(elapsed, 2),
            "perf": perf,
        }
        recs.append(rec)
        print(f"== {qid} gov={g.get('law_name')} {g['art']}条 (看板番号 {gov_no}/30)")
        print(f"   raw: {raw_out!r}")
        print(f"   adopted: {parsed['adopted_numbers']} survived={survived} ({elapsed:.1f}s) perf={perf}", flush=True)

    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS, "temp": TEMP,
            "seed": SEED, "max_tokens": MAX_TOKENS, "ctx": CTX, "max_picks": MAX_PICKS,
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "board_format": "summary + '（' + keywords を '・' 連結 + '）'。提示順は pools.json の baseline 順。",
            "parse_rules": "selection_test.py と同一（全角→半角、範囲外は記録して不採用、重複除去、3件超は先頭3件採用＋truncated記録）。"}
    (BASE2 / "qwen_branch_results.json").write_text(
        json.dumps({"meta": meta, "results": recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(1 for r in recs if r["governing_survived_in_adopted"])
    print(f"\nsurvival: {n}/5")
    print("saved qwen_branch_results.json")


if __name__ == "__main__":
    main()
