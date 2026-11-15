#!/usr/bin/env python3
"""タグ作成用ブリーフ生成: 条文ごとに種材料（qtags単語・看板KW・見出し・本文抽出語）を集める。
使い方: python make_briefs.py pilot|fifty  → work/briefs_<ds>.json
"""
import json, re, sys
from collections import Counter
from pathlib import Path
from janome.tokenizer import Tokenizer

RK = Path("/home/hatch/workspace/research_kudou")
TV = RK / "tag_vector"
JT = Tokenizer()

UNIT = r"(?:平方メートル|メートル|センチメートル|ミリメートル|キロメートル|日|時間|分|秒|年|月|階|人|戸|台|％|パーセント|ルクス|デシベル|リットル|キログラム|トン|件|回|以上|以下|未満|超)"
NUMRE = re.compile(r"[0-9０-９一二三四五六七八九十百千]+(?=\s*" + UNIT + r")")
STOP = set("もの こと ため 場合 者 人 規定 法令 条 項 号 規定 規定 政令 省令 勅令 規則 法律 建築 建築物 当該 前項 次項 本項 同項 各号 一 二 三 四 五 六 七 八 九 十 及び 又は 関 対 基づ 含 行 為 為 規定".split())


def nouns(text):
    c = Counter()
    for t in JT.tokenize(text):
        p = t.part_of_speech.split(",")
        if p[0] == "名詞" and p[1] not in ("代名詞", "非自立", "数", "接尾") and len(t.surface) >= 2:
            if t.surface not in STOP:
                c[t.surface] += 1
    return [w for w, _ in c.most_common(40)]


def numbers(text):
    seen, out = set(), []
    for m in re.finditer(r"[0-9０-９一二三四五六七八九十百千]+\s*" + UNIT, text):
        s = re.sub(r"\s+", "", m.group(0))
        if s not in seen:
            seen.add(s); out.append(s)
        if len(out) >= 14:
            break
    return out


def load_jsonl_map(path, valf):
    m = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m[(d["law_id"], d["art"])] = valf(d)
    return m


def main(ds):
    if ds == "pilot":
        pools = json.load(open(RK / "signboard_pilot/work/pools.json", encoding="utf-8"))
        texts = json.load(open(RK / "question_tag_pilot/work/pool_texts.json", encoding="utf-8"))
        qt = load_jsonl_map(RK / "question_tag_pilot/qtags.jsonl", lambda d: d)
        sb = load_jsonl_map(RK / "signboard_pilot2/signboards_v2.jsonl", lambda d: d)
    else:
        pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
        texts = json.load(open(RK / "question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
        qt = load_jsonl_map(RK / "fifty_measurement/qtags50.jsonl", lambda d: d)
        sb = load_jsonl_map(RK / "fifty_measurement/signboards50.jsonl", lambda d: d)
    keys = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})
    briefs = {}
    for k in keys:
        t = texts[f"{k[0]}|{k[1]}"]
        txt = t["text"]
        quoted = list(dict.fromkeys(re.findall(r"「([^」]{2,20})」", txt)))[:10]
        briefs[f"{k[0]}|{k[1]}"] = {
            "law_id": k[0], "art": k[1],
            "law_title": t["law_title"], "label": t["label"], "caption": t["caption"],
            "text_chars": len(txt),
            "qtag_words": qt[k]["words"], "qtag_questions": qt[k]["questions"],
            "sb_keywords": sb[k]["keywords"], "sb_summary": sb[k]["summary"][:220],
            "quoted": quoted, "nouns": nouns(txt), "numbers": numbers(txt),
        }
    out = TV / f"work/briefs_{ds}.json"
    json.dump(briefs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out, "articles:", len(briefs))


if __name__ == "__main__":
    main(sys.argv[1])
