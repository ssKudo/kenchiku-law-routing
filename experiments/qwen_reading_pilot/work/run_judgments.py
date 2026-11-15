#!/usr/bin/env python3
"""Qwen 読み判定: 問題文＋看板1件を1ペアずつ「合う／合わない」で判定させる。

条件:
 desc_plain   : 描写変換の問題文 + 描写看板（かっこ無し）
 desc_bracket : 描写変換の問題文 + 描写看板（かっこ有り、意味単位で［］区切り）
 ref_v2       : 原文の問題文 + 看板v2（pilot2 と同じ summary（keywords）形式）

各ペア2回判定（round1 seed=42 / round2 seed=777、temp 0.2、n_predict 16）。
ペアを round1→round2 と連続実行し、llama-server のプロンプトキャッシュで
2回目のプロンプト処理を再利用する。問題ごとにまとめ、接頭辞キャッシュも活かす。
生出力は条件ごとの JSONL に1判定ずつ追記し、再開可能にする。

文量超過（プロンプトが ctx に収まらない）場合は、看板を文字数で切り詰めて
truncated フラグを付けて記録する（対象は施行規則1条の3の描写看板のみの見込み）。
"""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
BASE = Path("/home/hatch/workspace/research_kudou")

SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PORT = 18098
BASEURL = "http://127.0.0.1:%d" % PORT
THREADS = 4
CTX = 32768
N_PREDICT = 16
TEMP = 0.2
SEEDS = {1: 42, 2: 777}

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題と、ある条文の内容をやさしい言葉で説明した文章（看板）です。

問題文:
{stem}

選択肢:
{choices}

看板:
{board}

この看板の条文は、この問題の答えの根拠になる条文ですか。根拠になる場合は「合う」、ならない場合は「合わない」だけを答えてください。説明や理由は書かないでください。"""


def build_prompt(stem, choices, board):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), board=board)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def parse_judgment(raw):
    """合う/合わない を抽出。合わない は 合う を含むので先に判定する。再試行なし。"""
    stripped = re.sub(r"[\s　]+", "", raw)
    m_dis = raw.find("合わない")
    m_fit = raw.find("合う")
    if m_dis != -1 and (m_fit == -1 or m_dis <= m_fit):
        return {"judgment": "合わない", "format_clean": stripped == "合わない"}
    if m_fit != -1:
        return {"judgment": "合う", "format_clean": stripped == "合う"}
    return {"judgment": None, "format_clean": False}


def server_post(path, payload, timeout=1800):
    req = urllib.request.Request(BASEURL + path,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def tokenize(text):
    return len(server_post("/tokenize", {"content": text})["tokens"])


def wait_health(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early: %s" % proc.returncode)
        try:
            req = urllib.request.Request(BASEURL + "/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("health check timeout")


def judge(prompt, seed):
    t0 = time.time()
    resp = server_post("/completion", {
        "prompt": prompt, "temperature": TEMP, "seed": seed,
        "n_predict": N_PREDICT, "cache_prompt": True, "id_slot": 0,
        "stop": ["<|im_end|>", "<|im_start|>"]}, timeout=3600)
    return resp.get("content", ""), time.time() - t0, resp.get("timings", {})


def load_materials():
    mats = json.loads((HERE / "materials.json").read_text(encoding="utf-8"))
    boards_desc = {}
    for l in (BASE / "descriptive_pilot/boards_pilot.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            boards_desc[(r["law_id"], r["art"])] = r["text"]
    boards_v2 = {}
    for l in (BASE / "signboard_pilot2/signboards_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            boards_v2[(r["law_id"], r["art"])] = r["summary"] + "（" + "・".join(r["keywords"]) + "）"
    boards_br = {}
    for l in (OUT / "bracketed_boards.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            boards_br[(r["law_id"], r["art"])] = r["text"]
    q_br = {}
    for l in (OUT / "bracketed_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            q_br[r["qid"]] = {"stem": r["stem"], "choices": r["choices"]}
    return mats, boards_desc, boards_v2, boards_br, q_br


def cond_texts(cond, mats_q, boards_desc, boards_v2, boards_br, q_br, key):
    if cond == "desc_plain":
        q = mats_q["question_desc"]
        return q["stem"], q["choices"], boards_desc[key]
    if cond == "desc_bracket":
        q = q_br[mats_q["qid"]]
        return q["stem"], q["choices"], boards_br[key]
    if cond == "ref_v2":
        q = mats_q["question_orig"]
        return q["stem"], q["choices"], boards_v2[key]
    raise ValueError(cond)


PROMPT_CAP = 12288  # 実用上限。これを超える看板は先頭から切り詰める（対象は施行規則1条の3のみ。
# 当初 24576 で再開したが、単一判定の処理が長時間化し実行セッションの終了に巻き込まれたため、
# 確実に1判定が終わる範囲として 12288 に下げた。全文が文脈に収まらないこと自体は変わらない）


def fit_prompt(stem, choices, board):
    """プロンプトが PROMPT_CAP に収まるよう、必要なら看板を切り詰める。"""
    prompt = build_prompt(stem, choices, board)
    n = tokenize(prompt)
    truncated = False
    if n > PROMPT_CAP:
        board_tokens = tokenize(board)
        budget = board_tokens - (n - PROMPT_CAP)
        ratio = len(board) / max(board_tokens, 1)
        cut = int(budget * ratio * 0.97)
        board = board[:cut]
        prompt = build_prompt(stem, choices, board)
        n = tokenize(prompt)
        truncated = True
        while n > PROMPT_CAP:  # 念のための追加調整
            board = board[:int(len(board) * 0.9)]
            prompt = build_prompt(stem, choices, board)
            n = tokenize(prompt)
    return prompt, board, n, truncated


STOP = False


def run_condition(cond, mats, boards_desc, boards_v2, boards_br, q_br, limit_pairs=None, deadline=None):
    global STOP
    raw_path = OUT / ("raw_%s.jsonl" % cond)
    done = set()
    if raw_path.exists():
        for l in raw_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["qid"], r["law_id"], r["art"], r["round"]))
    fout = open(raw_path, "a", encoding="utf-8")
    n_new = 0
    t_cond = time.time()
    for qid in ["2017-21", "2017-22", "2017-24", "2017-4", "2017-1"]:
        mq = mats[qid]
        gov = (mq["governing"]["law_id"], mq["governing"]["art"])
        pairs = [(p["law_id"], p["art"]) for p in mq["pool"]]
        if limit_pairs:
            pairs = pairs[:limit_pairs]
        for key in pairs:
            if deadline and time.time() > deadline:
                STOP = True
                fout.close()
                print("[%s] deadline reached, stopping (resumable)" % cond, flush=True)
                return
            if (qid, key[0], key[1], 1) in done and (qid, key[0], key[1], 2) in done:
                continue
            stem, choices, board = cond_texts(cond, mq, boards_desc, boards_v2, boards_br, q_br, key)
            orig_chars = len(board)
            prompt, board_used, ntok, truncated = fit_prompt(stem, choices, board)
            for rnd in (1, 2):
                if (qid, key[0], key[1], rnd) in done:
                    continue
                raw, elapsed, timings = judge(prompt, SEEDS[rnd])
                p = parse_judgment(raw)
                rec = {"condition": cond, "qid": qid, "law_id": key[0], "art": key[1],
                       "is_governing": key == gov, "round": rnd, "seed": SEEDS[rnd],
                       "raw_output": raw, "judgment": p["judgment"],
                       "format_clean": p["format_clean"],
                       "board_chars_orig": orig_chars, "board_chars_used": len(board_used),
                       "truncated": truncated, "prompt_tokens": ntok,
                       "seconds": round(elapsed, 3),
                       "cache_tokens": timings.get("cache_n"),
                       "prompt_eval_tokens": timings.get("prompt_n"),
                       "predicted_tokens": timings.get("predicted_n")}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_new += 1
                if n_new % 20 == 0:
                    print("[%s] %d new judgments, %.0fs elapsed" % (cond, n_new, time.time() - t_cond), flush=True)
    fout.close()
    print("[%s] done: %d new judgments in %.0fs" % (cond, n_new, time.time() - t_cond), flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_seconds = None
    for a in sys.argv[1:]:
        if a.startswith("--max-seconds="):
            max_seconds = int(a.split("=", 1)[1])
    conds = args or ["ref_v2", "desc_plain", "desc_bracket"]
    deadline = time.time() + max_seconds if max_seconds else None
    mats, boards_desc, boards_v2, boards_br, q_br = load_materials()
    print("starting llama-server (ctx=%d) ..." % CTX, flush=True)
    proc = subprocess.Popen(
        [SERVER, "-m", MODEL, "-c", str(CTX), "-t", str(THREADS),
         "--parallel", "1", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        boot = wait_health(proc)
        print("server ready in %.1fs" % boot, flush=True)
        for cond in conds:
            run_condition(cond, mats, boards_desc, boards_v2, boards_br, q_br, deadline=deadline)
            if STOP:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
