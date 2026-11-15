#!/usr/bin/env python3
"""日常文サニティチェック: 1.5B に名前のある二択を1つ選ばせる。

- 法令実験 binary_chain_pilot (v1) と同一プロトコル:
  Qwen2.5-1.5B-Instruct GGUF Q4_K_M、llama-server 常駐+プロンプトキャッシュ、
  temperature 0.2、seed 42、n_predict 32。
- 各セル (文×質問) につき3回聞く。提示順は rep1=AB / rep2=BA / rep3=AB の固定順。
- 問いの形は「次の2つのうち、この文の内容に近いのはどちらですか。
  選んだ方の言葉だけを、そのまま書いてください」。
  数字・記号の例示は入れない。
- 法令版との差分は読み物だけ: system プロンプトの職種語 (建築法規の) を外し、
  問題文+選択肢の提示を「文:」1本に置き換えた。それ以外は同一。
- パース規則も法令版と同一: 生出力中にラベルA・ラベルBのどちらか一方だけが
  現れればその選択。両方/どちらも無しは形式崩れ (pick=null)。
  再試行なし・勝手な補正なし。生出力を全部保存する。
  追加の記録として、出力が文の10文字以上の部分文字列を含む場合に echo フラグを立てる
  (パース結果は変えない)。
"""
import json, re, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PORT = 18099
BASE = "http://127.0.0.1:%d" % PORT
THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 32, 0.2, 42

SYSTEM = "あなたは検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は短い文です。

文:
{text}

質問: 次の2つのうち、この文の内容に近いのはどちらですか。選んだ方の言葉だけを、そのまま書いてください。説明や理由は書かないでください。

{first}
{second}"""

REPS = [("rep1", "AB"), ("rep2", "BA"), ("rep3", "AB")]

def build_prompt(text, first, second):
    user = USER_TMPL.format(text=text, first=first, second=second)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def is_echo(raw, text, n=10):
    t = re.sub(r"[\s　]+", "", text)
    r = re.sub(r"[\s　]+", "", raw)
    return any(t[i:i+n] in r for i in range(len(t) - n + 1)) if len(t) >= n else False

def parse_pick(raw, label_A, label_B, text):
    stripped = re.sub(r"[\s　]+", "", raw)
    in_A, in_B = label_A in raw, label_B in raw
    base = {"echo": is_echo(raw, text)}
    if in_A and not in_B:
        return {"pick": label_A, "pick_side": "A",
                "format_clean": stripped == label_A, "format_failure": False, **base}
    if in_B and not in_A:
        return {"pick": label_B, "pick_side": "B",
                "format_clean": stripped == label_B, "format_failure": False, **base}
    return {"pick": None, "pick_side": None, "format_clean": False,
            "format_failure": True,
            "failure_kind": "both_labels" if (in_A and in_B) else "no_label", **base}

def server_post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def wait_health(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early: %s" % proc.returncode)
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + "/health"), timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("health check timed out")

def ask(prompt):
    t0 = time.time()
    resp = server_post("/completion", {"prompt": prompt, "temperature": TEMP, "seed": SEED,
                                       "n_predict": N_PREDICT, "cache_prompt": True,
                                       "id_slot": 0, "stop": ["<|im_end|>", "<|im_start|>"]})
    return resp.get("content", ""), time.time() - t0, resp.get("timings", {})

def main():
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))["items"]

    print("starting llama-server ...", flush=True)
    proc = subprocess.Popen([SERVER, "-m", MODEL, "-c", str(CTX), "-t", str(THREADS),
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        boot = wait_health(proc)
        print("server ready in %.1fs" % boot, flush=True)
        results, total_sec = [], 0.0
        for item in items:
            irec = {"id": item["id"], "text": item["text"], "questions": []}
            for q in item["questions"]:
                qrec = {"qid": q["qid"], "label_A": q["label_A"], "label_B": q["label_B"],
                        "correct_side": q["correct_side"], "correct_label": q["correct_label"],
                        "type": q["type"], "reps": []}
                for rep_name, order in REPS:
                    first = q["label_A"] if order == "AB" else q["label_B"]
                    second = q["label_B"] if order == "AB" else q["label_A"]
                    raw, elapsed, timings = ask(build_prompt(item["text"], first, second))
                    p = parse_pick(raw, q["label_A"], q["label_B"], item["text"])
                    rec = {"rep": rep_name, "order": order,
                           "presented": [first, second],
                           "raw_output": raw, **p,
                           "correct": (p["pick_side"] is not None and p["pick_side"] == q["correct_side"]),
                           "picked_presented_first": (p["pick"] is not None and p["pick"] == first),
                           "seconds": round(elapsed, 3),
                           "prompt_tokens": timings.get("prompt_n"),
                           "predicted_tokens": timings.get("predicted_n"),
                           "cache_tokens": timings.get("cache_n")}
                    qrec["reps"].append(rec)
                    total_sec += elapsed
                picks = [r["pick_side"] for r in qrec["reps"]]
                print("%s picks(3reps): %s (correct=%s)" % (q["qid"], picks, q["correct_side"]), flush=True)
                irec["questions"].append(qrec)
            results.append(irec)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    all_reps = [r for it in results for q in it["questions"] for r in q["reps"]]
    meta = {"model": MODEL, "server": SERVER, "threads": THREADS, "ctx": CTX,
            "temp": TEMP, "seed": SEED, "n_predict": N_PREDICT,
            "server_boot_seconds": round(boot, 2), "system_prompt": SYSTEM,
            "user_template": USER_TMPL, "rep_orders": REPS,
            "n_calls": len(all_reps), "total_answer_seconds": round(total_sec, 2),
            "format_failures": sum(1 for r in all_reps if r["format_failure"]),
            "format_dirty": sum(1 for r in all_reps if r["pick"] is not None and not r["format_clean"]),
            "echo_outputs": sum(1 for r in all_reps if r["echo"])}
    (OUT / "picks.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved picks.json | failures=%d dirty=%d echo=%d | total %.1fs" % (
        meta["format_failures"], meta["format_dirty"], meta["echo_outputs"], total_sec))

if __name__ == "__main__":
    main()
