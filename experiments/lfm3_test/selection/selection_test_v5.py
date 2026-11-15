#!/usr/bin/env python3
"""選択実験 v5: 候補を1語ずつ見せて「関係ある○／ない×」だけ答えさせる二択判定。

経緯:
 - v1 (番号選択・番号例あり): 例の番号をなぞる定型出力。
 - v2 (番号例除去＋順序シャッフル併走): 10回全部が連番出力。
   「番号で選ぶ」形式自体が Qwen2.5-1.5B に合わない、という結論。
 - v3 (言葉を直接書かせる・例なし): 失敗の形が「候補リスト先頭からの写し」に
   変わっただけ (例外は No.4 のみ)。
 - v4 (言葉で選択・few-shot): 「飛ばして8語選ぶ」型は部分的に伝達したが、
   関連度による優先順位付けは付かず、baseline から順位改善した問はゼロ。

v5 の変更点 (仕事の分解):
 - 「20語から最大8個を選ぶ」 (相対比較＋優先順位＋個数管理) をやめ、
   候補リストの全語について1語ずつ独立に ○× 判定させる。
 - 実例なし (ゼロショット、v3 と比較可能に)。数字の例示は入れない。
 - ○と判定された語は上限を設けず全て採用する (8個枠は使わない)。
 - モデル・サンプリング条件は v1〜v4 と同一 (Qwen2.5-1.5B Q4_K_M、
   temperature 0.2、seed 42)。推論は llama-server 常駐＋プロンプトキャッシュで
   高速化する (問題文の共通接頭辞を再利用)。生成上限は回答が短いため 16 トークン。

判定プロンプト (1語ごと):
 system: あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。
 user:
   以下は一級建築士試験（法規）の問題について、根拠となる法令の条文を
   検索するための作業です。

   問題文:
   {stem}

   選択肢:
   {choices}

   言葉: 「{term}」

   この言葉は、この問題の根拠条文を探すのに役立ちますか。
   役立つ場合は「○」、役立たない場合は「×」だけを答えてください。
   説明や理由は書かないでください。

パース規則 (再試行なし・補正なし):
 - 生出力をそのまま保存する。
 - 判定の抽出: 出力に最初に現れる ○ または ×類 (× ✕ ☓) を判定とする。
 - format_clean = 空白を除いた出力全体がその1記号だけの場合に true。
   記号の前後に余分な文がある場合は format_clean=false として数える。
 - ○×類が一切含まれない出力は「判定不能 (invalid)」として数える。
 - 再試行はしない。

採点: binary-words = 問題文 + ○の語全部 (各1回、空白区切り) で TF-IDF 検索。
 比較は baseline / oracle は v1、model-words は v3、fewshot-words は v4 の値を再掲。
診断: v1 の bridge_terms_in_gov_all (候補語のうち正答条の本文に文字列として
 含まれる語 = 橋語) を使い、(i) 橋語の拾い率 = |橋語∩○| / |橋語|、
 (ii) ○の精度 = |橋語∩○| / |○|、(iii) ○の総数を問ごとに記録する。
"""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
from engine import Engine

SERVER = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-server"
COMPLETION_BIN = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-completion"
MODEL = "/home/hatch/workspace/models/LFM3-1.7B-Q4_K_M.gguf"
PORT = 18099
BASE = "http://127.0.0.1:%d" % PORT
THREADS = 4
CTX = 4096
N_PREDICT = 1024
TEMP = 0.2
SEED = 42

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
USER_TMPL = """以下は一級建築士試験（法規）の問題について、根拠となる法令の条文を検索するための作業です。

問題文:
{stem}

選択肢:
{choices}

言葉: 「{term}」

この言葉は、この問題の根拠条文を探すのに役立ちますか。役立つ場合は「○」、役立たない場合は「×」だけを答えてください。説明や理由は書かないでください。"""

CIRCLE = "○"
CROSS_CHARS = "×✕☓"

def build_prompt(stem, choices, term):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), term=term)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def parse_judgment(raw):
    """生出力から判定を抽出する。再試行・補正はしない。"""
    stripped = re.sub(r"[\s　]+", "", raw)
    for i, ch in enumerate(raw):
        if ch == CIRCLE:
            return {"judgment": "○", "symbol": "○", "format_clean": stripped == "○"}
        if ch in CROSS_CHARS:
            return {"judgment": "×", "symbol": ch, "format_clean": stripped == ch}
    return {"judgment": None, "symbol": None, "format_clean": False}

def server_post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def wait_health(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early with code %s" % proc.returncode)
        try:
            req = urllib.request.Request(BASE + "/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("llama-server health check timed out")

def judge(prompt):
    t0 = time.time()
    resp = server_post("/completion", {
        "prompt": prompt, "temperature": TEMP, "seed": SEED,
        "n_predict": N_PREDICT, "cache_prompt": True, "id_slot": 0,
        "stop": ["<|im_end|>", "<|im_start|>"]})
    elapsed = time.time() - t0
    return resp.get("content", ""), elapsed, resp.get("timings", {})

def cond(engine, query, govkey):
    scores = engine.score_all(query)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = [{"law_id": l, "art": a, "score": round(s, 2)} for (l, a), s in ranked[:10]]
    rank = next((i + 1 for i, (k, _) in enumerate(ranked) if k == govkey), None)
    return {"top3": top[:3], "top10": top, "gov_rank": rank,
            "top3_hit": rank is not None and rank <= 3}

def main():
    blind = [json.loads(l) for l in (HERE / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lists = {r["no"]: r["terms"] for r in json.loads((HERE / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
    governing = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    v1 = json.loads((HERE / "selection_results.json").read_text(encoding="utf-8"))
    v1_recs = {(r["year"], r["no"]): r for r in v1["results"]}
    v3 = json.loads((HERE / "selection_results_v3.json").read_text(encoding="utf-8"))
    v3_recs = {(r["year"], r["no"]): r for r in v3["results"]}
    v4 = json.loads((HERE / "selection_results_v4.json").read_text(encoding="utf-8"))
    v4_recs = {(r["year"], r["no"]): r for r in v4["results"]}

    print("starting llama-server ...", flush=True)
    proc = subprocess.Popen(
        [SERVER, "-m", MODEL, "-c", str(CTX), "-t", str(THREADS),
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        boot_sec = wait_health(proc)
        print("server ready in %.1fs" % boot_sec, flush=True)

        engine = Engine()
        out_recs = []
        total_judge_sec = 0.0
        run_t0 = time.time()
        for q in blind:
            key = (q["year"], q["no"])
            terms = lists[q["no"]]
            g = governing[key]; gov = g["governing"]
            govkey = (gov["law_id"], gov["art"])
            qtext = q["stem"] + " " + " ".join(q["choices"])
            bridge_all = v1_recs[key].get("bridge_terms_in_gov_all", [])

            judgments = []
            q_sec = 0.0
            for term in terms:
                raw, elapsed, timings = judge(build_prompt(q["stem"], q["choices"], term))
                p = parse_judgment(answer_part(raw))
                judgments.append({"term": term, "raw_output": raw,
                                  "judgment": p["judgment"], "symbol_found": p["symbol"],
                                  "format_clean": p["format_clean"],
                                  "think": think_stats(raw),
                                  "seconds": round(elapsed, 3),
                                  "prompt_tokens": timings.get("prompt_n"),
                                  "predicted_tokens": timings.get("predicted_n"),
                                  "cache_tokens": timings.get("cache_n")})
                q_sec += elapsed
                total_judge_sec += elapsed
            circle_terms = [j["term"] for j in judgments if j["judgment"] == "○"]
            cross_terms = [j["term"] for j in judgments if j["judgment"] == "×"]
            invalid_terms = [j["term"] for j in judgments if j["judgment"] is None]
            dirty = [j["term"] for j in judgments if j["judgment"] is not None and not j["format_clean"]]
            bw = cond(engine, qtext + " " + " ".join(circle_terms), govkey)

            bridge_set = set(bridge_all)
            circle_set = set(circle_terms)
            diag = {"n_candidates": len(terms),
                    "n_circle": len(circle_terms), "n_cross": len(cross_terms),
                    "n_invalid": len(invalid_terms), "n_format_dirty": len(dirty),
                    "bridge_terms_in_list": bridge_all,
                    "bridge_hit_terms": [t for t in circle_terms if t in bridge_set],
                    "bridge_recall": (len(bridge_set & circle_set) / len(bridge_set)) if bridge_set else None,
                    "circle_precision_bridge": (len(bridge_set & circle_set) / len(circle_set)) if circle_set else None}
            v1r = v1_recs[key]; v3r = v3_recs[key]; v4r = v4_recs[key]
            rec = {"year": q["year"], "no": q["no"],
                   "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"), "art": gov["art"]},
                   "confidence": g.get("confidence"),
                   "candidate_terms": terms,
                   "baseline_from_v1": {"gov_rank": v1r["baseline"]["gov_rank"], "top3_hit": v1r["baseline"]["top3_hit"]},
                   "oracle_from_v1": {"gov_rank": v1r["oracle_all_terms"]["gov_rank"], "top3_hit": v1r["oracle_all_terms"]["top3_hit"]},
                   "model_words_from_v3": {"gov_rank": v3r["model_words"]["gov_rank"], "top3_hit": v3r["model_words"]["top3_hit"]},
                   "fewshot_words_from_v4": {"gov_rank": v4r["fewshot_words"]["gov_rank"], "top3_hit": v4r["fewshot_words"]["top3_hit"]},
                   "judgments": judgments,
                   "circle_terms": circle_terms, "cross_terms": cross_terms,
                   "invalid_terms": invalid_terms, "format_dirty_terms": dirty,
                   "binary_words": bw,
                   "diagnostics": diag,
                   "question_seconds": round(q_sec, 2)}
            out_recs.append(rec)
            (HERE / "selection_results_v5_partial.json").write_text(
                json.dumps({"results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
            print("== %s gov=%s %s条 base=%s oracle=%s v3=%s v4=%s v5=%s | ○%d ×%d invalid%d dirty%d | bridge拾い %s (%.1fs)" % (
                key, gov.get("law_name"), gov["art"], rec["baseline_from_v1"]["gov_rank"],
                rec["oracle_from_v1"]["gov_rank"], v3r["model_words"]["gov_rank"],
                v4r["fewshot_words"]["gov_rank"], bw["gov_rank"],
                len(circle_terms), len(cross_terms), len(invalid_terms), len(dirty),
                diag["bridge_recall"], q_sec), flush=True)
        total_sec = time.time() - run_t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    n_judgments = sum(len(r["judgments"]) for r in out_recs)
    sample_prompt = build_prompt(blind[0]["stem"], blind[0]["choices"], lists[blind[0]["no"]][0])
    meta = {"model": MODEL, "llama_server": SERVER, "llama_completion_ref": COMPLETION_BIN,
            "threads": THREADS, "ctx": CTX, "temp": TEMP, "seed": SEED,
            "n_predict": N_PREDICT, "server_boot_seconds": round(boot_sec, 2),
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "sample_prompt_first_term": sample_prompt,
            "method": "候補リストの全語について、問題文＋言葉1語だけの独立したプロンプトで ○/× 判定 (ゼロショット、実例なし)。llama-server 常駐＋cache_prompt で問題文の共通接頭辞を再利用。○の語は上限なしで全て採用。",
            "parse_rules": "再試行なし。生出力を保存し、最初に現れる ○ または ×類 (× ✕ ☓) を判定として抽出。空白除去後の出力全体がその1記号のみなら format_clean=true、余分な文があれば format_dirty として数える。○×類が無い出力は判定不能 (invalid)。",
            "query_method": "binary-words = stem + choices + ' ' + ○の語を空白区切りで連結 (各1回)。v1〜v4 と同一方式。○が0語の場合は問題文のみのクエリ (= baseline と同一) になる。",
            "baseline_oracle_source": "v1 selection_results.json の値を再掲。v3 は selection_results_v3.json の model_words、v4 は selection_results_v4.json の fewshot_words。",
            "bridge_definition": "v1 の bridge_terms_in_gov_all: 候補語のうち正答条の本文 (見出し＋条文テキスト) に文字列として含まれる語。",
            "n_judgments_total": n_judgments,
            "total_judge_seconds": round(total_judge_sec, 2),
            "total_run_seconds_incl_scoring": round(total_sec, 2),
            "seconds_per_judgment": round(total_judge_sec / n_judgments, 3) if n_judgments else None}
    (HERE / "selection_results_v5.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved selection_results_v5.json | judgments=%d total_judge=%.1fs per_judgment=%.2fs" % (
        n_judgments, total_judge_sec, total_judge_sec / max(n_judgments, 1)))

if __name__ == "__main__":
    main()
