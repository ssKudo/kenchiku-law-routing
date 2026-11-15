#!/usr/bin/env python3
"""選択実験 v4: few-shot 版 — 実例2件を見せてから、言葉を直接書かせる。

経緯:
 - v1 (番号選択・番号例あり): 例の番号をなぞる定型出力。
 - v2 (番号例除去＋順序シャッフル併走): 10回全部が連番出力。
   「番号で選ぶ」形式自体が Qwen2.5-1.5B に合わない、という結論。
 - v3 (言葉を直接書かせる・例なし): 失敗の形が「候補リスト先頭からの写し」に
   変わっただけ。5問中3問が17〜19語を並び順に出力し上限8個を超過。
   例外は No.4 のみ (中後半から順不同で7語、順位も 10→9 と微改善)。

v4 の変更点 (問題・本番リスト・モデル・temp 0.2・seed 42・max_tokens 256 は v3 と同一):
 - プロンプトに実例2件を先行させる。構成: 指示 → 実例1 → 実例2 → 本番の問題。
 - 実例は fewshot_examples.json (候補リストと選択はファイルのまま、並べ替え・改変なし)、
   問題文は blind_examples.jsonl の 2018年 No.2 / No.1 を year/no で引く。
 - 実例の選択はリスト先頭8語と意図的にずらしてある (「内容で選ぶ」見本)。
 - 指示文に数字の例示は入れない (v1 の轍を踏まない)。
 - llama-completion のコンテキスト長だけ -c 8192 を明示 (実例2件でプロンプトが
   長くなるため。サンプリング条件には影響しない)。

記録 (v3 と同じ物差し):
 - 生出力をそのまま保存。
 - パースは v3 と同じ規則: (a) リスト内選択語 と (b) リスト外の語 に分割。
   空白・読点・括弧の差は吸収、語そのものの改変は (b) 扱いで補正しない。
   リスト内選択が8語を超えたら先頭8語を採用し truncated を記録。
 - 診断: 先頭8語との一致度、問題ごとの選択の食い違い、上限遵守、
   実例の丸写し (実例の選択語のうち本番リストに無い語が本番出力に混入していないか)。

採点: baseline / oracle は v1、model-words は v3 の値を再掲。
新条件 fewshot-words = 問題文 + (a) の採用語のみで TF-IDF 検索。
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
CTX = 8192
MAX_TOKENS = 256
TEMP = 0.2
SEED = 42
MAX_PICKS = 8

SYSTEM = "あなたは建築法規の検索アシスタントです。指示に厳密に従ってください。"
INSTRUCTION = """以下は一級建築士試験（法規）の問題について、根拠となる法令の条文を検索するための作業です。

指示:
- 候補リストの中から、その問題の根拠条文を探すのに役立つ言葉を、最大8個まで選んでください。
- 選んだ言葉は、リストに書かれているとおりの言葉で、カンマ区切りで書いてください。
- リストに無い言葉は書かないでください。
- 説明や文章は書かないでください。選んだ言葉だけを書いてください。

まず実例を2件示します。"""
BLOCK_TMPL = """問題文:
{stem}

選択肢:
{choices}

検索語の候補リスト:
{candidates}

選んだ言葉:
{selected}"""
TARGET_TMPL = """問題文:
{stem}

選択肢:
{choices}

検索語の候補リスト:
{candidates}

選んだ言葉:"""

def block(stem, choices, terms, selected=None):
    cand = "\n".join(terms)
    if selected is None:
        return TARGET_TMPL.format(stem=stem, choices="\n".join(choices), candidates=cand)
    return BLOCK_TMPL.format(stem=stem, choices="\n".join(choices), candidates=cand,
                             selected="、".join(selected))

def build_prompt(examples, stem, choices, terms):
    """examples: [(stem, choices, candidates, selected), ...]"""
    parts = [INSTRUCTION]
    for i, (es, ec, et, esel) in enumerate(examples, 1):
        parts.append("【実例%d】\n" % i + block(es, ec, et, esel))
    parts.append("【本番の問題】\n" + block(stem, choices, terms))
    user = "\n\n".join(parts)
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def run_model(prompt):
    pf = HERE / "_prompt_tmp_sel_v4.txt"
    pf.write_text(prompt, encoding="utf-8")
    t0 = time.time()
    cp = subprocess.run(
        [LLAMA, "-m", MODEL, "-f", str(pf), "-n", str(MAX_TOKENS), "-c", str(CTX),
         "--temp", str(TEMP), "--seed", str(SEED), "-t", str(THREADS),
         "--no-display-prompt"],
        capture_output=True, text=True, timeout=1200, stdin=subprocess.DEVNULL)
    elapsed = time.time() - t0
    out = cp.stdout.split("<|im_end|>")[0].split("> EOF")[0].strip()
    return out, elapsed

def _norm(s):
    """表記ゆれ吸収用: 空白類を除去した形。"""
    return re.sub(r"[\s　]+", "", s)

def split_frags(raw):
    text = raw.replace("，", "、").replace(",", "、").replace("\n", "、")
    frags = [f.strip(" \t　「」『』\"'“”‘’・:：。") for f in text.split("、")]
    return [f for f in frags if f != ""]

def parse_words(raw, terms):
    """v3 と同一のパース規則。生出力を (リスト内選択語, リスト外の語) に分ける。"""
    term_by_norm = {}
    for t in terms:
        term_by_norm.setdefault(_norm(t), t)
    frags = split_frags(raw)
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

def example_leak(raw, examples_selected, target_terms):
    """実例の選択語のうち本番候補リストに無い語が、本番の生出力に現れたら丸写し混入。
    照合は v3 パースと同じ断片分割＋表記ゆれ吸収で行う。"""
    target_norm = {_norm(t) for t in target_terms}
    frag_norms = {_norm(f) for f in split_frags(raw)}
    raw_norm = _norm(raw)
    leaks = []
    for sel in examples_selected:
        for w in sel:
            if _norm(w) in target_norm:
                continue  # 本番リストにもある語は混入と判定できない
            if _norm(w) in frag_norms or _norm(w) in raw_norm:
                leaks.append(w)
    return leaks

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

    # 実例: 問題文は blind_examples.jsonl を year/no で引き、候補と選択は fewshot_examples.json のまま
    ex_q = {(r["year"], r["no"]): r for r in
            (json.loads(l) for l in (HERE / "blind_examples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    ex_data = json.loads((HERE / "fewshot_examples.json").read_text(encoding="utf-8"))["examples"]
    examples = []
    for e in ex_data:
        q = ex_q[(e["year"], e["no"])]
        missing = [w for w in e["selected"] if w not in e["candidates"]]
        assert not missing, "example %s-%s: selected not in candidates: %s" % (e["year"], e["no"], missing)
        examples.append((q["stem"], q["choices"], e["candidates"], e["selected"]))
    examples_selected = [e["selected"] for e in ex_data]

    engine = Engine()
    out_recs = []
    for q in blind:
        key = (q["year"], q["no"])
        terms = lists[q["no"]]
        g = governing[key]; gov = g["governing"]
        govkey = (gov["law_id"], gov["art"])
        qtext = q["stem"] + " " + " ".join(q["choices"])

        prompt = build_prompt(examples, q["stem"], q["choices"], terms)
        raw_out, elapsed = run_model(prompt)
        parsed = parse_words(raw_out, terms)
        adopted = parsed["adopted_terms"]
        fw = cond(engine, qtext + " " + " ".join(adopted), govkey) if adopted else \
            {"top3": [], "top10": [], "gov_rank": None, "top3_hit": False}
        diag = head8_diag(adopted, terms)
        leaks = example_leak(raw_out, examples_selected, terms)
        v1r = v1_recs[key]; v3r = v3_recs[key]
        rec = {"year": q["year"], "no": q["no"],
               "governing": {"law_id": gov["law_id"], "law_name": gov.get("law_name"), "art": gov["art"]},
               "confidence": g.get("confidence"),
               "candidate_terms": terms,
               "baseline_from_v1": {"gov_rank": v1r["baseline"]["gov_rank"], "top3_hit": v1r["baseline"]["top3_hit"]},
               "oracle_from_v1": {"gov_rank": v1r["oracle_all_terms"]["gov_rank"], "top3_hit": v1r["oracle_all_terms"]["top3_hit"]},
               "model_words_from_v3": {"gov_rank": v3r["model_words"]["gov_rank"], "top3_hit": v3r["model_words"]["top3_hit"],
                                       "adopted_terms": v3r["adopted_terms"]},
               "model_raw_output": raw_out,
               "picked_in_list": parsed["picked_in_list_all"],
               "adopted_terms": adopted,
               "outside_words": parsed["outside_words"],
               "truncated": parsed["truncated"],
               "head8_diag": diag,
               "example_leak_words": leaks,
               "fewshot_words": fw,
               "model_seconds": round(elapsed, 2)}
        out_recs.append(rec)
        print("== %s gov=%s %s条 base=%s oracle=%s v3=%s v4=%s" % (
            key, gov.get("law_name"), gov["art"], rec["baseline_from_v1"]["gov_rank"],
            rec["oracle_from_v1"]["gov_rank"], v3r["model_words"]["gov_rank"], fw["gov_rank"]), flush=True)
        print("   raw: %r" % raw_out, flush=True)
        print("   picked(%d): %s (%.1fs)" % (len(adopted), adopted, elapsed), flush=True)
        print("   outside: %s | head8: %s | leak: %s" % (parsed["outside_words"], diag, leaks), flush=True)

    sets = [set(r["adopted_terms"]) for r in out_recs]
    pairwise = []
    for i in range(len(out_recs)):
        for j in range(i + 1, len(out_recs)):
            pairwise.append({"pair": ["%s-%s" % (out_recs[i]["year"], out_recs[i]["no"]),
                                      "%s-%s" % (out_recs[j]["year"], out_recs[j]["no"])],
                             "identical": sets[i] == sets[j],
                             "overlap": len(sets[i] & sets[j])})
    sample_prompt = build_prompt(examples, blind[0]["stem"], blind[0]["choices"], lists[blind[0]["no"]])
    meta = {"model": MODEL, "llama_completion": LLAMA, "threads": THREADS, "ctx": CTX,
            "temp": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS, "max_picks": MAX_PICKS,
            "system_prompt": SYSTEM,
            "prompt_structure": "指示 → 【実例1】(問題文・選択肢・候補リスト・お手本の選択) → 【実例2】 → 【本番の問題】(問題文・選択肢・候補リスト・『選んだ言葉:』で終了し、続きを生成させる)",
            "sample_prompt_first_question": sample_prompt,
            "fewshot_examples": [{"year": e["year"], "no": e["no"], "candidates": e["candidates"],
                                  "selected": e["selected"]} for e in ex_data],
            "change_from_v3": "プロンプト冒頭に実例2件 (fewshot_examples.json、問題文は blind_examples.jsonl) を追加したのみ。本番の問題・候補リスト・並び順・モデル・サンプリング条件は v3 と同一。コンテキスト長 -c 8192 を明示 (実例でプロンプトが長くなるため)。",
            "baseline_oracle_source": "v1 selection_results.json の値を再掲。v3 比較値は selection_results_v3.json の model_words。",
            "query_method": "fewshot-words = stem + choices + ' ' + 採用語を空白区切りで連結 (各1回)。v1〜v3 と同一方式。",
            "parse_rules": "v3 と同一。区切りは読点 (、, ，) と改行。空白・括弧類の差は吸収。候補語自体に読点を含む場合は断片の再結合で照合。語そのものが改変された断片はリスト外として記録し補正しない。リスト内選択が8語を超えたら先頭8語を採用し truncated を記録。",
            "example_leak_rule": "実例の選択語のうち本番候補リストに無い語が、本番の生出力 (断片単位または全体の表記ゆれ吸収形) に現れた場合に丸写し混入として記録。",
            "pairwise_pick_overlap": pairwise}
    (HERE / "selection_results_v4.json").write_text(
        json.dumps({"meta": meta, "results": out_recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved selection_results_v4.json")

if __name__ == "__main__":
    main()
