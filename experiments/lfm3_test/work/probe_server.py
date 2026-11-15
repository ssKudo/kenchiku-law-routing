#!/usr/bin/env python3
"""LFM3 think 挙動プローブ: v5 と同一プロンプト形式で数回 llama-server を叩き、
</think> が閉じるか・思考トークン数・所要秒数を測る。"""
import json, subprocess, time, urllib.request
from pathlib import Path

SMALL = Path("/home/hatch/workspace/research_kudou/small_model_test")
SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
MODEL = "/home/hatch/workspace/models/LFM3-1.7B-Q4_K_M.gguf"
PORT = 18097
BASE = "http://127.0.0.1:%d" % PORT

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題について、根拠となる法令の条文を検索するための作業です。

問題文:
{stem}

選択肢:
{choices}

言葉: 「{term}」

この言葉は、この問題の根拠条文を探すのに役立ちますか。役立つ場合は「○」、役立たない場合は「×」だけを答えてください。説明や理由は書かないでください。"""

def build_prompt(stem, choices, term):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), term=term)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def post(path, payload, timeout=1200):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def wait_health(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited %s" % proc.returncode)
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + "/health"), timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("health timeout")

blind = [json.loads(l) for l in (SMALL / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
lists = {r["no"]: r["terms"] for r in json.loads((SMALL / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
q = blind[0]
terms = lists[q["no"]][:3]

proc = subprocess.Popen([SERVER, "-m", MODEL, "-c", "8192", "-t", "4",
                         "--host", "127.0.0.1", "--port", str(PORT)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
try:
    print("boot %.1fs" % wait_health(proc), flush=True)
    for term in terms:
        prompt = build_prompt(q["stem"], q["choices"], term)
        t0 = time.time()
        resp = post("/completion", {"prompt": prompt, "temperature": 0.2, "seed": 42,
                                    "n_predict": 2048, "cache_prompt": True, "id_slot": 0,
                                    "stop": ["<|im_end|>", "<|im_start|>"]})
        el = time.time() - t0
        content = resp.get("content", "")
        tim = resp.get("timings", {})
        closed = "</think>" in content
        think_part = content.split("</think>")[0] if closed else content
        after = content.split("</think>", 1)[1] if closed else ""
        print("term=%s sec=%.1f pred_n=%s prompt_n=%s cache_n=%s think_closed=%s think_chars=%d after=%r stop_type=%s" % (
            term, el, tim.get("predicted_n"), tim.get("prompt_n"), tim.get("cache_n"),
            closed, len(think_part), after.strip()[:40], resp.get("stop_type")), flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
