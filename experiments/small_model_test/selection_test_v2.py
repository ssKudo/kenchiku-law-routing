#!/usr/bin/env python3
"""選択実験 v2: v1 の失敗要因のうち「プロンプトの番号例」と「順序依存」を切り分ける。

v1 (SELECTION_REPORT.md) の問題点:
 - 指示文に具体例「3, 7, 12」を入れたため、モデルが例をなぞって番号を並べた
   (問題もリストも違う2問で出力が完全一致)。
 - リスト順序が固定だったため、番号で選んだのか語で選んだのか区別できない。

v2 の変更点 (リスト・問題・モデル・温度・seed は v1 と同一):
 1. 指示文から番号の具体例を除去。「番号だけをカンマ区切りで」とだけ書く。
 2. 同じリストを2つの順序で出す: A=元の順序, B=シャッフル (random seed 7)。
    語で追う: 両順序で選ばれた「語」が一致すれば内容追随、番号だけ一致すれば順序追随。
 3. リスト自体は作り直さない (正答条を見て直すと oracle が汚染されるため)。
    リストの質の上限 (oracle) は v1 と同じ値になるはずで、ここでは再掲のみ。

判定:
 - 5問の生出力が問ごとに違うか (v1 は4問が「3, 7, 12」で始まった)。
 - A/B で選ばれた語の重なり (共通語数 / 和集合)。語が安定すれば「読んで選んだ」証拠。
 - model-picks の順位が baseline / oracle とどう動くか。
"""
import json, random, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
from engine import Engine

LLAMA = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-completion"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
THREADS = 4
MAX_TOKENS = 128
TEMP = 0.2
SEED = 42
MAX_PICKS = 8
SHUFFLE_SEED = 7

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
# v1 との差分: 例示「(例: 3, 7, 12)」を除去した。
USER_TMPL = """以下は一級建築士試験（法規）の問題です。この問題の根拠となる法令の条文を検索したいです。

問題文:
{stem}

選択肢:
{choices}

検索語の候補リスト:
{candidates}

指示:
- 上の候補リストの中から、この問題の根拠条文を探すのに役立つものを、番号で最大8個まで選んでください。
- 出力は番号だけをカンマ区切りで書いてください。説明や文章は書かないでください。"""

def build_prompt(stem, choices, terms):
    cand = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(terms))
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), candidates=cand)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def run_model(prompt):
    pf = HERE / "_prompt_tmp_sel_v2.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run(
        [LLAMA, "-m", MODEL, "-f", str(pf), "-n", str(MAX_TOKENS),
         "--temp", str(TEMP), "--seed", str(SEED), "-t", str(THREADS),
         "--no-display-prompt"],
        capture_output=True, text=True, timeout=1200, stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    out = cp.stdout.split("<|im_end|>")[0].split("> EOF")[0].strip()
    return out, elapsed

FW = str.maketrans("０１２３４５６７８９", "0123456789")
def parse_picks(raw, n_terms):
    nums = [int(x) for x in re.findall(r"\d+", raw.translate(FW))]
    valid, invalid, seen = [], [], set()
    for n in nums:
        if 1 <= n <= n_terms:
            if n not in seen:
                seen.add(n); valid.append(n)
        else:
            invalid.append(n)
    truncated = len(valid) > MAX_PICKS
    return {"raw_numbers": nums, "valid_unique": valid, "invalid_numbers": invalid,
            "truncated": truncated, "adopted_numbers": valid[:MAX_PICKS],
            "parse_ok": len(nums) > 0}

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
    engine = Engine()
    out_recs = []
    for q in blind:
        key = (q["year"], q["no"])
        terms = lists[q["no"]]
        g = governing[key]; gov = g["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])
        base = cond(engine, qtext, govkey)
        oracle = cond(engine, qtext + " " + " ".join(terms), govkey)

        shuffled = terms[:]
        random.Random(SHUFFLE_SEED).shuffle(shuffled)
        orders = {"A_original": terms, "B_shuffled": shuffled}
        runs = {}
        for name, order in orders.items():
            prompt = build_prompt(q["stem"], q["choices"], order)
            raw_out, elapsed = run_model(prompt)
            parsed = parse_picks(raw_out, len(order))
            picked_terms = [order[n - 1] for n in parsed["adopted_numbers"]]
            picks = cond(engine, qtext + " " + " ".join(picked_terms), govkey) if picked_terms else \
                {"top3": [], "top10": [], "gov_rank": None, "top3_hit": False}
            runs[name] = {"order": order, "model_raw_output": raw_out,
                          "parse": parsed, "picked_terms": picked_terms,
                          "model_picks": picks, "model_seconds": round(elapsed, 2)}
            print(f"== {key} {name} gov={gov.get('law_name')} {gov['art']}条 "
                  f"base={base['gov_rank']} oracle={oracle['gov_rank']} picks={picks['gov_rank']}", flush=True)
            print(f"   raw: {raw_out!r}", flush=True)
            print(f"   picked: {picked_terms} ({elapsed:.1f}s)", flush=True)

        a_terms, b_terms = set(runs["A_original"]["picked_terms"]), set(runs["B_shuffled"]["picked_terms"])
        overlap = sorted(a_terms & b_terms)
        rec = {"year": q["year"], "no": q["no"],
               "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"), "art": gov["art"]},
               "confidence": g.get("confidence"),
               "candidate_terms": terms,
               "baseline": base, "oracle_all_terms": oracle,
               "runs": runs,
               "term_overlap_A_B": overlap,
               "term_overlap_count": len(overlap),
               "term_union_count": len(a_terms | b_terms)}
        out_recs.append(rec)
    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS,
            "temp": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS, "max_picks": MAX_PICKS,
            "shuffle_seed": SHUFFLE_SEED,
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "change_from_v1": "番号の具体例を指示文から除去。順序A(元)と順序B(シャッフル seed 7)の2条件を併走。リスト・問題・モデル設定は v1 と同一。",
            "query_method": "baseline = stem + choices ((c) ステージ3と同一)。追加条件 = baseline + ' ' + 対象語を空白区切りで連結 (各1回)。",
            "parse_rules": "全角数字を半角化してから数値を抽出。範囲外番号は不採用として記録。重複除去 (出現順)。有効番号が8を超えたら先頭8個を採用し truncated を記録。"}
    (HERE / "selection_results_v2.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved selection_results_v2.json")

if __name__ == "__main__":
    main()
