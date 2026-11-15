#!/usr/bin/env python3
"""ステージ1 仕上げ: 公式PDFから15件の問題文・選択肢を補完する (年度差を吸収)。"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "/home/hatch/workspace/research_kudou/venv/lib/python3.12/site-packages")
import pymupdf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
TARGETS = {2017: [12, 16, 23, 26], 2018: [28, 29], 2019: [13, 30],
           2021: [12], 2022: [22], 2023: [3, 4, 22, 27], 2026: [29]}
FW = "０１２３４５６７８９"
def fw2int(s):
    return int("".join(str(FW.index(c)) if c in FW else c for c in s))

HEADER = re.compile(r"〔\s*[ＮN]\s*[ｏo]．\s*([０-９0-9]+)\s*〕")
MARK = {1: r"[１₁1]", 2: r"[２₂2]", 3: r"[３₃3]", 4: r"[４₄4]"}

def clean(t):
    t = re.sub(r"[\ue000-\uf8ff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*学科Ⅲ\s*―?\s*\d*\s*―?\s*$", "", t)
    return t

def split_choices(body):
    # 1,2,3,4 のマーカーを順に探す
    pos = []
    start = 0
    for k in (1, 2, 3, 4):
        m = re.search(r"(?:^|\n)\s*" + MARK[k] + r"\s*[．.]\s*", body[start:])
        if not m:
            return body, []
        pos.append(start + m.start())
        start = start + m.end()
    stem = body[:pos[0]]
    choices = []
    ends = pos[1:] + [len(body)]
    for i, p in enumerate(pos):
        seg = body[p:ends[i]]
        seg = re.sub(r"^\s*[１２３４₁₂₃₄1234]\s*[．.]\s*", "", seg.strip(), count=1)
        choices.append(seg)
    return stem, choices

def extract(year):
    doc = pymupdf.open(HERE / f"pdf/gakka3_{year}.pdf")
    txt = "\n".join(p.get_text() for p in doc)
    parts = HEADER.split(txt)
    qs = {}
    for i in range(1, len(parts) - 1, 2):
        no = fw2int(parts[i])
        stem, ch = split_choices(parts[i + 1])
        qs[no] = (clean(stem), [clean(c) for c in ch])
    return qs

def main():
    rows = [json.loads(l) for l in (OUT / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["year"], r["no"]): r for r in rows}
    for year, nos in TARGETS.items():
        qs = extract(year)
        print(year, "extracted questions:", len(qs))
        for no in nos:
            stem, ch = qs.get(no, ("", []))
            r = by[(year, no)]
            r["stem"] = stem; r["choices"] = ch; r["explanations"] = []
            r["has_figure"] = ("図" in stem)
            r["text_source"] = "official_pdf"
            r["status"] = "ok" if (stem and len(ch) == 4) else "parse_incomplete"
            print(" ", no, "choices:", len(ch), "stem len:", len(stem))
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r.get("status") for r in rows))

if __name__ == "__main__":
    main()
