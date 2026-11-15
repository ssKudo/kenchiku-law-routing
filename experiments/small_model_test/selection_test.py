#!/usr/bin/env python3
"""選択実験: 小型モデル (Qwen2.5-1.5B-Instruct GGUF) に検索語を自由生成させる
のではなく、エミリーが作った候補リストから「番号で選ばせるだけ」で務まるかを測る。

条件 (各問):
 0. baseline    : 問題文+選択肢のみで TF-IDF 検索 ((c) ステージ3と同一方式)
 1. oracle      : 問題文に候補リスト全語を足して検索 (リスト自体の質の上限)
 2. model-picks : 問題文にモデルが選んだ語だけを足して検索

TF-IDF エンジンは prototype/c_full/work/engine.py を再利用。
採点データ (governing.jsonl / results.json) はスコア計算と sanity check にのみ使う。
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
MAX_TOKENS = 128
TEMP = 0.2
SEED = 42
MAX_PICKS = 8

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
USER_TMPL = """以下は一級建築士試験（法規）の問題です。この問題の根拠となる法令の条文を検索したいです。

問題文:
{stem}

選択肢:
{choices}

検索語の候補リスト:
{candidates}

指示:
- 上の候補リストの中から、この問題の根拠条文を探すのに役立つものを、番号で最大8個まで選んでください。
- 出力は番号だけをカンマ区切りで書いてください（例: 3, 7, 12）。説明や文章は書かないでください。"""

def build_prompt(stem, choices, terms):
    cand = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(terms))
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), candidates=cand)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def run_model(prompt):
    pf = HERE / "_prompt_tmp_sel.txt"
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

FW = str.maketrans("０１２３４５６７８９", "0123456789")
def parse_picks(raw, n_terms):
    """番号を抽出する。後処理で勝手に直さない方針:
    - 全角数字は半角化してから抽出 (記録に明記)
    - 範囲外の番号は invalid_numbers として記録し不採用
    - 重複は出現順を保って除去
    - 有効番号が MAX_PICKS を超えたら先頭 MAX_PICKS 個を採用し truncated=True と記録
    """
    nums = [int(x) for x in re.findall(r"\d+", raw.translate(FW))]
    valid, invalid, seen = [], [], set()
    for n in nums:
        if 1 <= n <= n_terms:
            if n not in seen:
                seen.add(n); valid.append(n)
        else:
            invalid.append(n)
    truncated = len(valid) > MAX_PICKS
    adopted = valid[:MAX_PICKS]
    return {"raw_numbers": nums, "valid_unique": valid, "invalid_numbers": invalid,
            "truncated": truncated, "adopted_numbers": adopted,
            "parse_ok": len(nums) > 0}

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

def cond(engine, query, govkey):
    top, rank = rank_of(engine, query, govkey)
    return {"top3": top[:3], "top10": top, "gov_rank": rank,
            "top3_hit": rank is not None and rank <= 3}

def main():
    blind = [json.loads(l) for l in (HERE / "blind_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lists = {r["no"]: r["terms"] for r in json.loads((HERE / "candidate_lists.json").read_text(encoding="utf-8"))["lists"]}
    governing = {(r["year"], r["no"]): r for r in
                 (json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    prev = {(r["year"], r["no"]): r for r in
            json.loads((CFULL / "results.json").read_text(encoding="utf-8"))["results"]}

    engine = Engine()
    out_recs = []
    for q in blind:
        key = (q["year"], q["no"])
        terms = lists[q["no"]]
        g = governing[key]; gov = g["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])

        base = cond(engine, qtext, govkey)
        # sanity check: baseline top3 が results.json の seeds と一致するか
        prev_seeds = [(s["law_id"], s["art"]) for s in prev[key]["seeds"]]
        base_top3_keys = [(t["law_id"], t["art"]) for t in base["top3"]]
        sanity = (base_top3_keys == prev_seeds)

        oracle = cond(engine, qtext + " " + " ".join(terms), govkey)

        prompt = build_prompt(q["stem"], q["choices"], terms)
        raw_out, elapsed, perf = run_model(prompt)
        parsed = parse_picks(raw_out, len(terms))
        picked_terms = [terms[n - 1] for n in parsed["adopted_numbers"]]
        picks = cond(engine, qtext + " " + " ".join(picked_terms), govkey) if picked_terms else \
            {"top3": [], "top10": [], "gov_rank": None, "top3_hit": False}

        # 診断用: 候補語のうち正答条の条文テキストに含まれる語 (橋の有無)
        govdoc = engine.docs[engine.by_key[govkey]]
        bridge_all = [t for t in terms if t in govdoc["full"]]
        bridge_picked = [t for t in picked_terms if t in govdoc["full"]]

        rec = {
            "year": q["year"], "no": q["no"],
            "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"), "art": gov["art"]},
            "confidence": g.get("confidence"),
            "candidate_terms": terms,
            "baseline": base, "baseline_sanity_matches_stage3": sanity,
            "oracle_all_terms": oracle,
            "model_raw_output": raw_out,
            "parse": parsed,
            "picked_terms": picked_terms,
            "model_picks": picks,
            "bridge_terms_in_gov_all": bridge_all,
            "bridge_terms_in_gov_picked": bridge_picked,
            "model_seconds": round(elapsed, 2),
            "perf": perf,
        }
        out_recs.append(rec)
        print(f"== {key} gov={gov.get('law_name')} {gov['art']}条 sanity={sanity}")
        print(f"   base rank: {base['gov_rank']} | oracle rank: {oracle['gov_rank']} | picks rank: {picks['gov_rank']}")
        print(f"   raw: {raw_out!r}")
        print(f"   picked: {picked_terms} ({elapsed:.1f}s)")
    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS,
            "temp": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS, "max_picks": MAX_PICKS,
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "query_method": "baseline = stem + choices (stage3 と同一)。追加条件 = baseline + ' ' + 対象語を空白区切りで連結 (各1回)。",
            "parse_rules": "全角数字を半角化してから数値を抽出。範囲外番号は不採用として記録。重複除去 (出現順)。有効番号が8を超えたら先頭8個を採用し truncated を記録。"}
    (HERE / "selection_results.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved selection_results.json")

if __name__ == "__main__":
    main()
