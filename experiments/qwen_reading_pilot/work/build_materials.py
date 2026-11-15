#!/usr/bin/env python3
"""材料構築: プール定義、3条件分の問題文・看板、かっこ版の生成。

かっこ付けルール（問題側・看板側で同一の機械的ルール）:
 janome で形態素解析し、文節相当の意味単位で ［ ］ を付ける。
 - 単位は自立語（名詞・動詞・形容詞・副詞・連体詞・接続詞・感動詞）から始める。
 - 単位を閉じる契機（そのトークンを単位に含めてから閉じる）:
   格助詞・係助詞のうち は・を・へ・と・も・から・まで・より・で・に、
   接続助詞、文末の助動詞列の終わり、読点・句点（句読点はかっこの外に出す）。
   ただし連体化の「の」では閉じない（修飾の連なりを切らない）。
 - 2文字以下の単位は次の単位に結合する（断片化の防止）。
 - 句読点・括弧類はかっこの外にそのまま置く。
"""
import json
from pathlib import Path
from janome.tokenizer import Tokenizer

BASE = Path("/home/hatch/workspace/research_kudou")
OUT = BASE / "qwen_reading_pilot"
WORK = OUT / "work"
_tok = Tokenizer()

CLOSE_PARTICLES = {"は", "を", "へ", "と", "も", "から", "まで", "より", "で", "に"}
CONTENT_POS = ("名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭詞")


def bracket_text(text: str) -> str:
    units = []      # list of (unit_str, punct_after)
    cur = ""
    pending_punct = ""
    tokens = list(_tok.tokenize(text))
    for i, t in enumerate(tokens):
        pos = t.part_of_speech.split(",")
        major, sub = pos[0], pos[1] if len(pos) > 1 else ""
        surf = t.surface
        if major == "記号":
            # 句読点・括弧は現在の単位を閉じて外に出す
            if cur:
                units.append(cur)
                cur = ""
            units.append(surf)  # 記号はそのまま（後で結合処理）
            continue
        cur += surf
        close = False
        if major == "助詞" and sub in ("格助詞", "係助詞") and surf in CLOSE_PARTICLES:
            close = True
        elif major == "助詞" and sub == "接続助詞":
            close = True
        elif major == "助動詞":
            # 次が助詞・記号・文末なら閉じる
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or nxt.part_of_speech.split(",")[0] in ("助詞", "記号"):
                close = True
        if close and cur:
            units.append(cur)
            cur = ""
    if cur:
        units.append(cur)
    # 記号の処理: 開き括弧・鍵括弧は次の単位の頭へ、それ以外は直前の単位の後ろへ
    merged = []
    pending_open = ""
    for u in units:
        if len(u) == 1 and not u.isalnum() and u not in "［］":
            if u in ("（", "「", "『", "【"):
                pending_open += u
            elif merged:
                merged[-1] += u
            else:
                merged.append(u)
            continue
        merged.append(pending_open + u)
        pending_open = ""
    if pending_open:
        if merged:
            merged[-1] += pending_open
        else:
            merged.append(pending_open)
    final = []
    buf = ""
    for u in merged:
        buf += u
        if len(buf) >= 3 or u.endswith(("。", "、")):
            final.append(buf)
            buf = ""
    if buf:
        if final:
            final[-1] += buf
        else:
            final.append(buf)
    out = ""
    for u in final:
        if u in ("。", "、"):
            out += u
        else:
            out += "［" + u + "］"
    return out


def main():
    p2 = json.loads((BASE / "signboard_pilot2/pilot_results_v2.json").read_text(encoding="utf-8"))
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
    conv_q = {}
    for l in (BASE / "descriptive_pilot/converted_questions_pilot.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            conv_q[r["qid"]] = {"stem": r["stem"], "choices": r["choices"]}
    orig_q = {}
    for l in (BASE / "small_model_test/blind_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            orig_q["%d-%d" % (r["year"], r["no"])] = {"stem": r["stem"], "choices": r["choices"]}

    materials = {}
    missing = []
    for qid, v in p2.items():
        pool = []
        for i, p in enumerate(v["pool"]):
            key = (p["law"], p["art"])
            if key not in boards_desc:
                missing.append(("desc", qid, key))
            if key not in boards_v2:
                missing.append(("v2", qid, key))
            pool.append({"law_id": p["law"], "art": p["art"],
                         "baseline_pool_rank": i + 1,
                         "baseline_score": p["baseline_score"]})
        materials[qid] = {
            "qid": qid,
            "governing": {"law_id": v["governing"]["law_id"], "art": v["governing"]["art"],
                          "law_name": v["governing"]["law_name"], "confidence": v["governing"].get("confidence")},
            "baseline_rank_full_corpus": v["baseline_rank_given_full_corpus"],
            "pool": pool,
            "question_desc": conv_q[qid],
            "question_orig": orig_q[qid],
        }
    print("missing:", missing)
    (WORK / "materials.json").write_text(json.dumps(materials, ensure_ascii=False, indent=1), encoding="utf-8")

    # かっこ版（描写の問題文と看板だけに付ける）
    bq = {}
    for qid, m in materials.items():
        bq[qid] = {"stem": bracket_text(m["question_desc"]["stem"]),
                   "choices": [bracket_text(c) for c in m["question_desc"]["choices"]]}
    (OUT / "bracketed_questions.jsonl").write_text(
        "".join(json.dumps({"qid": q, **bq[q]}, ensure_ascii=False) + "\n" for q in bq), encoding="utf-8")
    bb = {}
    for (law, art), text in boards_desc.items():
        bb[(law, art)] = bracket_text(text)
    (OUT / "bracketed_boards.jsonl").write_text(
        "".join(json.dumps({"law_id": l, "art": a, "text": bb[(l, a)]}, ensure_ascii=False) + "\n"
                for (l, a) in boards_desc), encoding="utf-8")
    print("materials + bracketed files written")
    # サンプル表示（様式の目視確認用）
    gov24 = ("343AC0000000100", "58-2")
    print("SAMPLE board 58-2 bracketed head:", bb[gov24][:300])


if __name__ == "__main__":
    main()
