#!/usr/bin/env python3
"""二択の連続パイロット 手順3: 1.5B に名前のある二択を1つ選ばせる。

- Qwen2.5-1.5B-Instruct GGUF Q4_K_M、llama-server 常駐+プロンプトキャッシュ、
  temperature 0.2、seed 42 (v1〜アキネーターと同一条件)。n_predict は
  ラベルが語句のため 32。
- 各問×各軸につき3回聞く。提示順は rep1=AB / rep2=BA / rep3=AB の固定順。
- 問いの形は「次の2つのうち、この問題の内容に近いのはどちらですか。
  選んだ方の言葉だけを、そのまま書いてください」。
  数字・記号の例示は入れない。
- パース規則: 生出力中にラベルA・ラベルBのどちらか一方だけが現れれば
  その選択とする。両方現れる/どちらも無い場合は形式崩れ (pick=null)。
  再試行なし・勝手な補正なし。生出力を全部保存する。
"""
import json, re, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SMALL = Path("/home/hatch/workspace/research_kudou/small_model_test")

SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PORT = 18098
BASE = "http://127.0.0.1:%d" % PORT
THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 32, 0.2, 42

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題です。

問題文:
{stem}

選択肢:
{choices}

質問: 次の2つのうち、この問題の内容に近いのはどちらですか。選んだ方の言葉だけを、そのまま書いてください。説明や理由は書かないでください。

{first}
{second}"""

REPS = [("rep1", "AB"), ("rep2", "BA"), ("rep3", "AB")]

def build_prompt(stem, choices, first, second):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices),
                            first=first, second=second)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def parse_pick(raw, label_A, label_B):
    stripped = re.sub(r"[\s　]+", "", raw)
    in_A, in_B = label_A in raw, label_B in raw
    if in_A and not in_B:
        return {"pick": label_A, "pick_side": "A",
                "format_clean": stripped == label_A, "format_failure": False}
    if in_B and not in_A:
        return {"pick": label_B, "pick_side": "B",
                "format_clean": stripped == label_B, "format_failure": False}
    return {"pick": None, "pick_side": None, "format_clean": False,
            "format_failure": True,
            "failure_kind": "both_labels" if (in_A and in_B) else "no_label"}

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
    axes = json.loads((OUT / "axes.json").read_text(encoding="utf-8"))["axes"]
    blind = [json.loads(l) for l in (SMALL / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    print("starting llama-server ...", flush=True)
    proc = subprocess.Popen([SERVER, "-m", MODEL, "-c", str(CTX), "-t", str(THREADS),
                             "--host", "127.0.0.1", "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        boot = wait_health(proc)
        print("server ready in %.1fs" % boot, flush=True)
        results, total_sec = [], 0.0
        for q in blind:
            qrec = {"year": q["year"], "no": q["no"], "axes": []}
            for ax in axes:
                arec = {"axis_id": ax["id"], "label_A": ax["label_A"],
                        "label_B": ax["label_B"], "reps": []}
                for rep_name, order in REPS:
                    first = ax["label_A"] if order == "AB" else ax["label_B"]
                    second = ax["label_B"] if order == "AB" else ax["label_A"]
                    raw, elapsed, timings = ask(build_prompt(q["stem"], q["choices"], first, second))
                    p = parse_pick(raw, ax["label_A"], ax["label_B"])
                    presented_first = first
                    rec = {"rep": rep_name, "order": order,
                           "presented": [first, second],
                           "raw_output": raw, **p,
                           "picked_presented_first": (p["pick"] is not None and p["pick"] == presented_first),
                           "seconds": round(elapsed, 3),
                           "prompt_tokens": timings.get("prompt_n"),
                           "predicted_tokens": timings.get("predicted_n"),
                           "cache_tokens": timings.get("cache_n")}
                    arec["reps"].append(rec)
                    total_sec += elapsed
                picks = [r["pick_side"] for r in arec["reps"]]
                print("2017-%s %s picks(3reps): %s" % (q["no"], ax["id"], picks), flush=True)
                qrec["axes"].append(arec)
            results.append(qrec)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    all_reps = [r for q in results for a in q["axes"] for r in a["reps"]]
    meta = {"model": MODEL, "server": SERVER, "threads": THREADS, "ctx": CTX,
            "temp": TEMP, "seed": SEED, "n_predict": N_PREDICT,
            "server_boot_seconds": round(boot, 2), "system_prompt": SYSTEM,
            "user_template": USER_TMPL, "rep_orders": REPS,
            "n_calls": len(all_reps), "total_answer_seconds": round(total_sec, 2),
            "format_failures": sum(1 for r in all_reps if r["format_failure"]),
            "format_dirty": sum(1 for r in all_reps if r["pick"] is not None and not r["format_clean"])}
    (OUT / "picks.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved picks.json | failures=%d dirty=%d | total %.1fs" % (
        meta["format_failures"], meta["format_dirty"], total_sec))

if __name__ == "__main__":
    main()
