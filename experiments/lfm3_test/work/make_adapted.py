#!/usr/bin/env python3
"""Qwen 用スクリプト群を LFM3 用に複製する。
変更点 (プロンプト・候補・採点方法は一切変更しない):
 1. モデルパスを LFM3-1.7B-Q4_K_M.gguf に差し替え
 2. 生成上限を引き上げ (LFM3 の <think> が閉じて最終回答が出るまで確保するため)
 3. パースを </think> の後方テキストに対して実施し、生出力 (think 込み全文) と
    think 統計 (有無・閉じたか・文字数) を記録に追加
 4. サーバ系は質問ごとに部分結果ファイルを書き出す (途中停止対策)
"""
import shutil
from pathlib import Path

ROOT = Path("/home/hatch/workspace/research_kudou")
LFM = ROOT / "lfm3_test"
SMALL = ROOT / "small_model_test"
QWEN = "/home/hatch/workspace/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LFM3 = "/home/hatch/workspace/models/LFM3-1.7B-Q4_K_M.gguf"

HELPERS = '''
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

'''

def adapt(src: Path, dst: Path, edits: list):
    text = src.read_text(encoding="utf-8")
    assert QWEN in text, f"model path not found in {src}"
    text = text.replace(QWEN, LFM3)
    anchor = 'SYSTEM = "'
    i = text.index(anchor)
    line_start = text.rindex("\n", 0, i) + 1
    text = text[:line_start] + HELPERS.lstrip("\n") + text[line_start:]
    for old, new in edits:
        assert old in text, f"anchor not found in {src}: {old[:70]!r}"
        assert text.count(old) == 1, f"anchor not unique in {src}: {old[:70]!r} x{text.count(old)}"
        text = text.replace(old, new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print("wrote", dst)

SEL = LFM / "selection"
SEL.mkdir(parents=True, exist_ok=True)
for f in ["blind_questions.jsonl", "blind_examples.jsonl", "candidate_lists.json", "fewshot_examples.json"]:
    shutil.copy(SMALL / f, SEL / f)

# --- スモーク ---
adapt(SMALL / "smoke_test.py", SEL / "smoke_test.py", [
    ("MAX_TOKENS = 160", "MAX_TOKENS = 1024"),
    ("terms, excluded_art = parse_terms(raw_out)", "terms, excluded_art = parse_terms(answer_part(raw_out))"),
    ('"model_raw_output": raw_out,', '"model_raw_output": raw_out,\n            "think": think_stats(raw_out),'),
])

# --- 選択v1 ---
adapt(SMALL / "selection_test.py", SEL / "selection_test.py", [
    ("MAX_TOKENS = 128", "MAX_TOKENS = 1024"),
    ("parsed = parse_picks(raw_out, len(terms))", "parsed = parse_picks(answer_part(raw_out), len(terms))"),
    ('"model_raw_output": raw_out,\n            "parse": parsed,',
     '"model_raw_output": raw_out,\n            "think": think_stats(raw_out),\n            "parse": parsed,'),
])

# --- 選択v2 ---
adapt(SMALL / "selection_test_v2.py", SEL / "selection_test_v2.py", [
    ("MAX_TOKENS = 128", "MAX_TOKENS = 1024"),
    ("parsed = parse_picks(raw_out, len(order))", "parsed = parse_picks(answer_part(raw_out), len(order))"),
    ('runs[name] = {"order": order, "model_raw_output": raw_out,',
     'runs[name] = {"order": order, "model_raw_output": raw_out, "think": think_stats(raw_out),'),
])

# --- 選択v3 ---
adapt(SMALL / "selection_test_v3.py", SEL / "selection_test_v3.py", [
    ("MAX_TOKENS = 256", "MAX_TOKENS = 1024"),
    ("parsed = parse_words(raw_out, terms)", "parsed = parse_words(answer_part(raw_out), terms)"),
    ('"model_raw_output": raw_out,\n               "picked_in_list": parsed["picked_in_list_all"],',
     '"model_raw_output": raw_out,\n               "think": think_stats(raw_out),\n               "picked_in_list": parsed["picked_in_list_all"],'),
])

# --- 選択v4 ---
adapt(SMALL / "selection_test_v4.py", SEL / "selection_test_v4.py", [
    ("MAX_TOKENS = 256", "MAX_TOKENS = 1024"),
    ("parsed = parse_words(raw_out, terms)", "parsed = parse_words(answer_part(raw_out), terms)"),
    ("leaks = example_leak(raw_out, examples_selected, terms)",
     "leaks = example_leak(answer_part(raw_out), examples_selected, terms)"),
    ('"model_raw_output": raw_out,\n               "picked_in_list": parsed["picked_in_list_all"],',
     '"model_raw_output": raw_out,\n               "think": think_stats(raw_out),\n               "picked_in_list": parsed["picked_in_list_all"],'),
])

# --- 選択v5 (llama-server) ---
adapt(SMALL / "selection_test_v5.py", SEL / "selection_test_v5.py", [
    ("N_PREDICT = 16", "N_PREDICT = 1024"),
    ("p = parse_judgment(raw)", "p = parse_judgment(answer_part(raw))"),
    ('"format_clean": p["format_clean"],\n                                  "seconds": round(elapsed, 3),',
     '"format_clean": p["format_clean"],\n                                  "think": think_stats(raw),\n                                  "seconds": round(elapsed, 3),'),
    ("            out_recs.append(rec)\n",
     "            out_recs.append(rec)\n"
     "            (HERE / \"selection_results_v5_partial.json\").write_text(\n"
     "                json.dumps({\"results\": out_recs}, ensure_ascii=False, indent=1), encoding=\"utf-8\")\n"),
])

# --- アキネーター ---
AK = LFM / "akinator"
AK.mkdir(parents=True, exist_ok=True)
for f in ["attributes.json", "tags.jsonl"]:
    shutil.copy(ROOT / "akinator_pilot" / f, AK / f)
adapt(ROOT / "akinator_pilot" / "work" / "answer.py", AK / "work" / "answer.py", [
    ("THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 16, 0.2, 42",
     "THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 1024, 0.2, 42"),
    ("p = parse_answer(raw)", "p = parse_answer(answer_part(raw))"),
    ('"raw_output": raw, "answer": p["answer"],',
     '"raw_output": raw, "think": think_stats(raw), "answer": p["answer"],'),
    ('            results.append({"year": q["year"], "no": q["no"], "answers": recs})\n',
     '            results.append({"year": q["year"], "no": q["no"], "answers": recs})\n'
     '            (OUT / "answers_partial.json").write_text(\n'
     '                json.dumps({"results": results}, ensure_ascii=False, indent=1), encoding="utf-8")\n'),
])
shutil.copy(ROOT / "akinator_pilot" / "work" / "score.py", AK / "work" / "score.py")

# --- 二択 v1 / v2 ---
for name, srcdir in [("binary_v1", "binary_chain_pilot"), ("binary_v2", "binary_chain_pilot2")]:
    BD = LFM / name
    BD.mkdir(parents=True, exist_ok=True)
    for f in ["axes.json", "sides.jsonl"]:
        shutil.copy(ROOT / srcdir / f, BD / f)
    adapt(ROOT / srcdir / "work" / "ask.py", BD / "work" / "ask.py", [
        ("THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 32, 0.2, 42",
         "THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 1024, 0.2, 42"),
        (("p = parse_pick(raw, ax[\"label_A\"], ax[\"label_B\"])" if name == "binary_v1"
          else "p = parse_pick(raw, ax[\"label_A\"], ax[\"label_B\"], qtext)"),
         ("p = parse_pick(answer_part(raw), ax[\"label_A\"], ax[\"label_B\"])" if name == "binary_v1"
          else "p = parse_pick(answer_part(raw), ax[\"label_A\"], ax[\"label_B\"], qtext)")),
        ('"raw_output": raw, **p,', '"raw_output": raw, "think": think_stats(raw), **p,'),
        ("            results.append(qrec)\n",
         "            results.append(qrec)\n"
         "            (OUT / \"picks_partial.json\").write_text(\n"
         "                json.dumps({\"results\": results}, ensure_ascii=False, indent=1), encoding=\"utf-8\")\n"),
    ])
    shutil.copy(ROOT / srcdir / "work" / "score.py", BD / "work" / "score.py")

# --- 日常文 ---
EV = LFM / "everyday"
EV.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "everyday_check" / "items.json", EV / "items.json")
adapt(ROOT / "everyday_check" / "work" / "ask.py", EV / "work" / "ask.py", [
    ("THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 32, 0.2, 42",
     "THREADS, CTX, N_PREDICT, TEMP, SEED = 4, 4096, 1024, 0.2, 42"),
    ("p = parse_pick(raw, q[\"label_A\"], q[\"label_B\"], item[\"text\"])",
     "p = parse_pick(answer_part(raw), q[\"label_A\"], q[\"label_B\"], item[\"text\"])"),
    ('"raw_output": raw, **p,', '"raw_output": raw, "think": think_stats(raw), **p,'),
    ("            results.append(irec)\n",
     "            results.append(irec)\n"
     "            (OUT / \"picks_partial.json\").write_text(\n"
     "                json.dumps({\"results\": results}, ensure_ascii=False, indent=1), encoding=\"utf-8\")\n"),
])
shutil.copy(ROOT / "everyday_check" / "work" / "score.py", EV / "work" / "score.py")
print("ALL OK")
