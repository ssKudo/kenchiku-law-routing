#!/usr/bin/env python3
"""選択実験 v3: 「番号で選ばせる」をやめ、選んだ『言葉そのもの』を書かせる。

経緯:
 - v1 (番号選択・番号例あり): モデルは例「3, 7, 12」をなぞり、異なる2問で出力が完全一致。
 - v2 (番号例除去＋順序シャッフル併走): 10回全部が連番出力。例の真似ではなく
   「番号で選ぶ」形式自体が Qwen2.5-1.5B に合わない、という結論。

v3 の変更点 (問題・リスト・モデル・temp 0.2・seed 42 は v1/v2 と同一):
 - 候補リストを「番号なし・言葉だけ」で提示し、選んだ言葉をそのまま
   カンマ区切りで書かせる。番号例・数字の例示は入れない。
 - 並び順は元の順 (A) のみ。
 - max_tokens だけ 128 → 256 に変更 (日本語8語が 128 トークンを超えて
   語の途中で切れるのを防ぐため。サンプリング条件は不変)。

記録:
 - 生出力をそのまま保存。
 - パース結果を (a) リスト内から選ばれた語 と (b) リスト外の語 に分離。
   空白・読点・括弧の差は吸収するが、語そのものの改変は (b) として扱い補正しない。
 - 診断: 選択が「リスト先頭8語の写し」になっていないか (先頭8語との一致度)、
   問題ごとに選択が食い違うか。

採点: baseline / oracle は v1 (selection_results.json) の値を再掲。
新条件 model-words = 問題文 + (a) の採用語のみで TF-IDF 検索。
"""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")
sys.path.insert(0, str(CFULL / "work"))
from engine import Engine

LLAMA = "/home/hatch/workspace/tools/llama.cpp/llama-b12314/llama-completion"
MODEL = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
THREADS = 4
MAX_TOKENS = 256
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
- 上の候補リストの中から、この問題の根拠条文を探すのに役立つ言葉を、最大8個まで選んでください。
- 選んだ言葉は、リストに書かれているとおりの言葉で、カンマ区切りで書いてください。
- リストに無い言葉は書かないでください。
- 説明や文章は書かないでください。選んだ言葉だけを書いてください。"""

def build_prompt(stem, choices, terms):
    cand = "\n".join(terms)  # 番号なし・言葉だけ
    user = USER_TMPL.format(stem=stem, choices="\n".join(choices), candidates=cand)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def run_model(prompt):
    pf = HERE / "_prompt_tmp_sel_v3.txt"
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

def _norm(s):
    """表記ゆれ吸収用: 空白類を除去した形。"""
    return re.sub(r"[\s　]+", "", s)

def parse_words(raw, terms):
    """生出力を (リスト内選択語, リスト外の語) に分ける。
    区切りは読点 (、, ，) と改行。候補語自体に読点を含むものがあるため
    (例: '敷地、構造又は建築設備')、単独で一致しない断片は後続断片と
    読点で再結合して一致を試みる。語そのものの改変は一致とみなさない。"""
    term_by_norm = {}
    for t in terms:
        term_by_norm.setdefault(_norm(t), t)
    text = raw.replace("，", "、").replace(",", "、").replace("\n", "、")
    frags = [f.strip(" \t　「」『』\"'“”‘’・:：。") for f in text.split("、")]
    frags = [f for f in frags if f != ""]
    picked, outside = [], []
    i = 0
    while i < len(frags):
        hit, consumed = None, 0
        joined = ""
        for j in range(i, len(frags)):
            joined = frags[j] if j == i else joined + "、" + frags[j]
            t = term_by_norm.get(_norm(joined))
            if t is not None:
                hit, consumed = t, j - i + 1
                break
        if hit is not None:
            if hit not in picked:
                picked.append(hit)
            i += consumed
        else:
            outside.append(frags[i])
            i += 1
    truncated = len(picked) > MAX_PICKS
    return {"picked_in_list_all": picked, "adopted_terms": picked[:MAX_PICKS],
            "truncated": truncated, "outside_words": outside}

def head8_diag(picked, terms):
    head = terms[:MAX_PICKS]
    prefix = 0
    for a, b in zip(picked, terms):
        if a == b:
            prefix += 1
        else:
            break
    return {"head8_exact_copy": picked == head,
            "head8_set_overlap": len(set(picked) & set(head)),
            "common_prefix_len": prefix}

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
    engine = Engine()
    out_recs = []
    for q in blind:
        key = (q["year"], q["no"])
        terms = lists[q["no"]]
        g = governing[key]; gov = g["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])

        prompt = build_prompt(q["stem"], q["choices"], terms)
        raw_out, elapsed = run_model(prompt)
        parsed = parse_words(raw_out, terms)
        adopted = parsed["adopted_terms"]
        mw = cond(engine, qtext + " " + " ".join(adopted), govkey) if adopted else \
            {"top3": [], "top10": [], "gov_rank": None, "top3_hit": False}
        diag = head8_diag(adopted, terms)
        v1r = v1_recs[key]
        rec = {"year": q["year"], "no": q["no"],
               "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"), "art": gov["art"]},
               "confidence": g.get("confidence"),
               "candidate_terms": terms,
               "baseline_from_v1": {"gov_rank": v1r["baseline"]["gov_rank"], "top3_hit": v1r["baseline"]["top3_hit"]},
               "oracle_from_v1": {"gov_rank": v1r["oracle_all_terms"]["gov_rank"], "top3_hit": v1r["oracle_all_terms"]["top3_hit"]},
               "model_raw_output": raw_out,
               "picked_in_list": parsed["picked_in_list_all"],
               "adopted_terms": adopted,
               "outside_words": parsed["outside_words"],
               "truncated": parsed["truncated"],
               "head8_diag": diag,
               "model_words": mw,
               "model_seconds": round(elapsed, 2)}
        out_recs.append(rec)
        print(f"== {key} gov={gov.get('law_name')} {gov['art']}条 "
              f"base={rec['baseline_from_v1']['gov_rank']} oracle={rec['oracle_from_v1']['gov_rank']} "
              f"words={mw['gov_rank']}", flush=True)
        print(f"   raw: {raw_out!r}", flush=True)
        print(f"   picked({len(adopted)}): {adopted} ({elapsed:.1f}s)", flush=True)
        print(f"   outside: {parsed['outside_words']} | head8: {diag}", flush=True)

    # 問題間の選択の食い違い (同型パターンの検出)
    sets = [set(r["adopted_terms"]) for r in out_recs]
    pairwise = []
    for i in range(len(out_recs)):
        for j in range(i + 1, len(out_recs)):
            pairwise.append({"pair": [f"{out_recs[i]['year']}-{out_recs[i]['no']}",
                                      f"{out_recs[j]['year']}-{out_recs[j]['no']}"],
                             "identical": sets[i] == sets[j],
                             "overlap": len(sets[i] & sets[j])})
    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS,
            "temp": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS, "max_picks": MAX_PICKS,
            "system_prompt": SYSTEM, "user_template": USER_TMPL,
            "change_from_v2": "候補を番号なし・言葉だけで提示し、選んだ言葉そのものをカンマ区切りで書かせる。並び順は元の順のみ。max_tokens は 128→256 (語の途中切断防止のため; サンプリング条件は同一)。",
            "baseline_oracle_source": "v1 selection_results.json の値を再掲 (baseline の sanity check は v1 で通過済み)。",
            "query_method": "model-words = stem + choices + ' ' + 採用語を空白区切りで連結 (各1回)。baseline/oracle と同一方式。",
            "parse_rules": "区切りは読点 (、, ，) と改行。空白・括弧類の差は吸収。候補語自体に読点を含む場合は断片の再結合で照合。語そのものが改変された断片はリスト外として記録し補正しない。リスト内選択が8語を超えたら先頭8語を採用し truncated を記録。",
            "pairwise_pick_overlap": pairwise}
    (HERE / "selection_results_v3.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved selection_results_v3.json")

if __name__ == "__main__":
    main()
