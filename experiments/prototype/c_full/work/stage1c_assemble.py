#!/usr/bin/env python3
"""ステージ1 最終組立て: キャッシュ済みHTMLを修正パーサで再パースし、
欠けはID連番の内挿URLを付けて browser 取得待ちとして記録する。"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_fetch as s1

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
HTML_DIR = s1.HTML_DIR

def main():
    official = s1.parse_official()
    old = {}
    for l in (OUT / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); old[(r["year"], r["no"])] = r
    idmap = {}
    for (y, n), r in old.items():
        if r.get("source_url"):
            idmap.setdefault(y, {})[n] = int(r["source_url"].split("/")[-1])
    urls = {}
    for y, d in idmap.items():
        pairs = sorted(d.items())
        base = pairs[0][1] - pairs[0][0]
        contig = all(i - n == base for n, i in pairs)
        print(y, "contiguous:", contig, "base:", base, "known:", len(pairs))
        for n, i in pairs:
            urls[(y, n)] = f"https://1kenchikushi.kakomonn.com/questions/{i}"
        if contig:
            for n in range(1, 31):
                if n not in d:
                    urls[(y, n)] = f"https://1kenchikushi.kakomonn.com/questions/{base + n}"
    rows = []
    mismatches = []
    derived_checked = 0
    for year in range(2017, 2027):
        for no in range(1, 31):
            url = urls.get((year, no))
            p = HTML_DIR / f"q_{year}_{no:02d}.html"
            if not url:
                rows.append({"year": year, "no": no, "source_url": None, "status": "url_not_found"})
                continue
            if not p.exists():
                rows.append({"year": year, "no": no, "source_url": url, "status": "needs_browser"})
                continue
            html = p.read_text(encoding="utf-8")
            stem, choices, explanations = s1.parse_question(html)
            derived, method = s1.derive_answer(stem, explanations, choices)
            rec = {"year": year, "no": no, "source_url": url, "stem": stem, "choices": choices,
                   "explanations": explanations,
                   "has_figure": ("図" in stem and ("示す" in stem or "図に" in stem)),
                   "kakomonn_derived_answer": derived, "derive_method": method}
            off = official.get((year, no), "NA")
            if (year, no) == (2026, 29):
                rec["answer"] = None; rec["answer_source"] = "special_all_scored"
            elif off != "NA":
                rec["answer"] = off; rec["answer_source"] = "official_index"
                if derived is not None:
                    derived_checked += 1
                    if derived != off:
                        mismatches.append((year, no, off, derived))
            else:
                rec["answer"] = derived
                rec["answer_source"] = "kakomonn_derived" if derived else "unconfirmed"
            rec["status"] = "ok" if (stem and len(choices) == 4) else "parse_incomplete"
            rows.append(rec)
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r.get("status") for r in rows))
    print("derived checked (non-null, official years):", derived_checked, "mismatches:", mismatches)
    print("2020:", [(r["no"], r["answer"], r.get("derive_method")) for r in rows if r["year"] == 2020])
    print("needs_browser:", [(r["year"], r["no"], r["source_url"]) for r in rows if r.get("status") == "needs_browser"])
    print("other non-ok:", [(r["year"], r["no"], r.get("status")) for r in rows if r.get("status") not in ("ok", "needs_browser")])

if __name__ == "__main__":
    main()
