#!/usr/bin/env python3
"""スモークテスト: 小型モデル (Qwen2.5-1.5B-Instruct GGUF) で問題文を
条文用語の検索語リストに変換し、(c) 本格版の TF-IDF エンジン (engine.py) で
「変換前クエリ (問題文のみ)」と「変換後クエリ (問題文+検索語)」の
入口トップ3命中と正答条順位を比較する。

変換プロンプト・条件は SMOKE_REPORT.md / smoke_results.json に記録する。
"""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
from engine import Engine  # (c) 本格版ステージ3と同一エンジン

LLAMA = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-completion"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
THREADS = 4
MAX_TOKENS = 160
TEMP = 0.2
SEED = 42

# 選定: (i) 周辺法・入口ミス・未到達 x3, (ii) 基準法・到達済み(対照) x1, (iii) 基準法・入口ミス x1
SELECTED = [
    {"year": 2022, "no": 24, "kind": "peripheral_miss"},  # 都市計画法53条 high
    {"year": 2020, "no": 25, "kind": "peripheral_miss"},  # 消防法8条 high
    {"year": 2022, "no": 28, "kind": "peripheral_miss"},  # 省エネ法11条 high
    {"year": 2018, "no": 3,  "kind": "control_hit"},      # 建築基準法6条 到達済み
    {"year": 2024, "no": 28, "kind": "kenchiku_miss"},    # 建築基準法12条 high 入口ミス
]

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題です。この問題の根拠となる法令の条文を検索するための検索語を考えてください。

問題文:
{stem}

選択肢:
{choices}

指示:
- 条文で使われる専門用語を最大8個まで挙げてください。
- 問題文の日常的な言い回しは、法令の条文で使われる正式な用語に言い換えてください。
- 出力は検索語を「、」で区切ったリストのみとしてください。説明文、番号付け、条文番号（第○条など）は書かないでください。"""

def build_prompt(stem, choices):
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices))
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def run_model(prompt):
    pf = HERE / "_prompt_tmp.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run(
        [LLAMA, "-m", MODEL, "-f", str(pf), "-n", str(MAX_TOKENS),
         "--temp", str(TEMP), "--seed", str(SEED), "-t", str(THREADS),
         "--no-display-prompt"],
        capture_output=True, text=True, timeout=1200, stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    out = cp.stdout
    out = out.split("<|im_end|>")[0].split("> EOF")[0].strip()
    # perf 行を prompt eval / eval に分けて拾う
    eval_tps = None; gen_tokens = None; prompt_tps = None; prompt_tokens = None
    for line in cp.stderr.splitlines():
        mp = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", line)
        if mp:
            prompt_tokens = int(mp.group(2)); prompt_tps = float(mp.group(3)); continue
        mg = re.search(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second", line)
        if mg:
            gen_tokens = int(mg.group(2)); eval_tps = float(mg.group(3))
    return out, elapsed, {"gen_tokens": gen_tokens, "eval_tokens_per_sec": eval_tps,
                          "prompt_tokens": prompt_tokens, "prompt_tokens_per_sec": prompt_tps}

TERM_SPLIT = re.compile(r"[、,，\n・]")
ART_ONLY = re.compile(r"^第[0-9０-９一二三四五六七八九十百]+条(の[0-9０-９一二三四五六七八九十]+)?(の[0-9０-９一二三四五六七八九十]+)?$")
def parse_terms(raw):
    """検索語リストへ分解。純粋な条文番号だけの語は『言葉の変換』の対象外として
    除外し、別途 excluded_article_terms に記録する（モデルの条番号当て推量を
    混ぜないための設計判断。SMOKE_REPORT.md に明記）。"""
    terms, excluded = [], []
    for t in TERM_SPLIT.split(raw):
        t = t.strip().strip("　 ・:：。.「」")
        t = re.sub(r"^\d+[\.．、]", "", t).strip()
        if not t or len(t) > 30:
            continue
        if ART_ONLY.match(t):
            excluded.append(t)
        else:
            terms.append(t)
    return terms[:12], excluded

def rank_of(engine, query, govkey, limit=10):
    scores = engine.score_all(query)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = [{"law_id": l, "art": a, "score": round(s, 2)} for (l, a), s in ranked[:limit]]
    rank = None
    for i, (k, _) in enumerate(ranked):
        if k == govkey:
            rank = i + 1
            break
    return top, rank

def main():
    questions = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    governing = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    prev = {(r["year"], r["no"]): r for r in
            json.loads((CFULL / "results.json").read_text(encoding="utf-8"))["results"]}

    engine = Engine()
    out_recs = []
    for sel in SELECTED:
        key = (sel["year"], sel["no"])
        q = questions[key]; g = governing[key]; gov = g["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])
        base_top, base_rank = rank_of(engine, qtext, govkey)
        prompt = build_prompt(q["stem"], q["choices"])
        raw_out, elapsed, perf = run_model(prompt)
        terms, excluded_art = parse_terms(raw_out)
        conv_query = qtext + " " + " ".join(terms)
        conv_top, conv_rank = rank_of(engine, conv_query, govkey)
        rec = {
            "year": sel["year"], "no": sel["no"], "kind": sel["kind"],
            "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"),
                          "art": gov["art"]},
            "confidence": g.get("confidence"),
            "prev_result": {"seed_top3_hit": prev[key].get("seed_top3_hit"),
                            "reached_strict": prev[key].get("reached_strict")},
            "stem_head": q["stem"][:80],
            "model_raw_output": raw_out,
            "terms": terms,
            "excluded_article_terms": excluded_art,
            "convert_seconds": round(elapsed, 2),
            "perf": perf,
            "baseline": {"top3": base_top[:3], "top10": base_top, "gov_rank": base_rank,
                         "top3_hit": base_rank is not None and base_rank <= 3},
            "converted": {"top3": conv_top[:3], "top10": conv_top, "gov_rank": conv_rank,
                          "top3_hit": conv_rank is not None and conv_rank <= 3},
        }
        out_recs.append(rec)
        print(f"== {key} {sel['kind']} gov={gov.get('law_name')} {gov['art']}条")
        print(f"   terms: {terms}")
        print(f"   base rank: {base_rank} -> conv rank: {conv_rank}  ({elapsed:.1f}s, {perf})")
    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS,
            "temp": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS,
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "query_method": "baseline = stem + choices (stage3 と同一)。converted = baseline + ' ' + 変換語を空白区切りで連結 (各1回)。"}
    (HERE / "smoke_results.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved smoke_results.json")

if __name__ == "__main__":
    main()
