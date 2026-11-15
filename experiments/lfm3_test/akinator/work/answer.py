#!/usr/bin/env python3
"""アキネーター式パイロット 手順3: 1.5B に属性質問へ ○× で答えさせる。

- Qwen2.5-1.5B-Instruct GGUF Q4_K_M、llama-server 常駐+プロンプトキャッシュ、
  temperature 0.2、seed 42、n_predict 16 (v1〜v5 と同一条件)。
- 各問×各属性を1回ずつ独立に聞く (5問×12属性=60回)。
- 問いの形は「問題文と選択肢の中に、その内容の記述が出てくるか」という
  具体的な読み取り (v5 の抽象的な「役立つか」とは変えた点)。
- パース規則は v5 と同じ: 再試行なし、生出力を保存し、最初に現れる
  ○ または ×類を回答とする。空白除去後の全体がその1記号のみなら format_clean。
"""
import json, re, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SMALL = Path("/home/hatch/workspace/research_kudou/small_model_test")

SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
MODEL = "/home/hatch/workspace/models/LFM3-1.7B-Q4_K_M.gguf"
PORT = 18099
BASE = "http://127.0.0.1:%d" % PORT
THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 1024, 0.2, 42

# --- LFM3 差し替えに伴う <think> 処理 (プロンプト・採点規則は Qwen と同一) ---
def answer_part(raw):
    """LFM3 は回答前に <think>...</think> を出すため、パースは </think> の
    後方テキストに対して行う。think が生成上限内に閉じなかった場合は
    回答なし (空文字) として扱い、そのまま各実験のパース規則で無効になる。"""
    if "</think>" in raw:
        return raw.split("</think>", 1)[1]
    if "<think>" in raw:
        return ""
    return raw

def think_stats(raw):
    if "<think>" in raw:
        closed = "</think>" in raw
        think = raw.split("</think>")[0] if closed else raw
        return {"think_present": True, "think_closed": closed, "think_chars": len(think)}
    return {"think_present": False, "think_closed": False, "think_chars": 0}

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題です。

問題文:
{stem}

選択肢:
{choices}

質問: 問題文と選択肢の中に、{topic}についての記述が出てきますか。出てくる場合は「○」、出てこない場合は「×」だけを答えてください。説明や理由は書かないでください。"""

CIRCLE = "○"
CROSS_CHARS = "×✕☓"

def build_prompt(stem, choices, topic):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), topic=topic)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def parse_answer(raw):
    stripped = re.sub(r"[\s　]+", "", raw)
    for ch in raw:
        if ch == CIRCLE:
            return {"answer": "○", "symbol": "○", "format_clean": stripped == "○"}
        if ch in CROSS_CHARS:
            return {"answer": "×", "symbol": ch, "format_clean": stripped == ch}
    return {"answer": None, "symbol": None, "format_clean": False}

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
    attrs = json.loads((OUT / "attributes.json").read_text(encoding="utf-8"))["attributes"]
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
            recs = []
            for at in attrs:
                raw, elapsed, timings = ask(build_prompt(q["stem"], q["choices"], at["topic_phrase"]))
                p = parse_answer(answer_part(raw))
                recs.append({"attr_id": at["id"], "label": at["label"], "topic_phrase": at["topic_phrase"],
                             "raw_output": raw, "think": think_stats(raw), "answer": p["answer"], "symbol_found": p["symbol"],
                             "format_clean": p["format_clean"], "seconds": round(elapsed, 3),
                             "prompt_tokens": timings.get("prompt_n"),
                             "predicted_tokens": timings.get("predicted_n"),
                             "cache_tokens": timings.get("cache_n")})
                total_sec += elapsed
            results.append({"year": q["year"], "no": q["no"], "answers": recs})
            (OUT / "answers_partial.json").write_text(
                json.dumps({"results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
            dist = "".join(r["answer"] or "?" for r in recs)
            print("2017-%s answers: %s" % (q["no"], dist), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    all_a = [r for q in results for r in q["answers"]]
    meta = {"model": MODEL, "server": SERVER, "threads": THREADS, "ctx": CTX,
            "temp": TEMP, "seed": SEED, "n_predict": N_PREDICT,
            "server_boot_seconds": round(boot, 2), "system_prompt": SYSTEM,
            "user_template": USER_TMPL,
            "n_calls": len(all_a), "total_answer_seconds": round(total_sec, 2),
            "distribution": {"circle": sum(1 for r in all_a if r["answer"] == "○"),
                             "cross": sum(1 for r in all_a if r["answer"] == "×"),
                             "invalid": sum(1 for r in all_a if r["answer"] is None),
                             "format_dirty": sum(1 for r in all_a if r["answer"] is not None and not r["format_clean"])}}
    (OUT / "answers.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved answers.json |", meta["distribution"], "| total %.1fs" % total_sec)

if __name__ == "__main__":
    main()
