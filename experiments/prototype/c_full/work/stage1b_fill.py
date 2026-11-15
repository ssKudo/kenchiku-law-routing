#!/usr/bin/env python3
"""ステージ1 補完: 修正済みパーサでキャッシュHTMLを再パースして questions.jsonl を
再構築し、未解決URLは ID の広域スキャン (年+ラベル照合) で埋める。"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_fetch as s1

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
HTML_DIR = s1.HTML_DIR

def page_year(html):
    m = re.search(r"（(\d{4})年）\s*問\d+（学科3", html)
    return int(m.group(1)) if m else None

def build_row(year, no, url, html, official):
    stem, choices, explanations = s1.parse_question(html)
    derived, method = s1.derive_answer(stem, explanations)
    rec = {"year": year, "no": no, "source_url": url, "stem": stem, "choices": choices,
           "explanations": explanations,
           "has_figure": ("図" in stem and ("示す" in stem or "図に" in stem)),
           "kakomonn_derived_answer": derived, "derive_method": method}
    off = official.get((year, no), "NA")
    if (year, no) == (2026, 29):
        rec["answer"] = None; rec["answer_source"] = "special_all_scored"
    elif off != "NA":
        rec["answer"] = off; rec["answer_source"] = "official_index"
    else:
        rec["answer"] = derived
        rec["answer_source"] = "kakomonn_derived" if derived else "unconfirmed"
    rec["status"] = "ok" if (stem and len(choices) == 4) else "parse_incomplete"
    return rec

def main():
    official = s1.parse_official()
    # 既存 questions.jsonl から URL を引き継ぐ
    old = {}
    if (OUT / "questions.jsonl").exists():
        for l in (OUT / "questions.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); old[(r["year"], r["no"])] = r
    urls = {}
    for (y, n), r in old.items():
        if r.get("source_url"):
            urls[(y, n)] = r["source_url"]
    # 一覧からも再構築 (キャッシュ)
    for year, lid in s1.LIST_IDS.items():
        found = {}
        for page in (1, 2, 3):
            html = s1.fetch(f"https://1kenchikushi.kakomonn.com/list1/{lid}?page={page}",
                            HTML_DIR / f"list_{year}_{page}.html")
            for m in re.finditer(r'href="(https://1kenchikushi\.kakomonn\.com/questions/(\d+))"[^>]*>(.*?)</a>', html, re.S):
                txt = s1.clean(m.group(3))
                mm = re.search(r"学科3（法規）\s*問(\d+)", txt)
                if mm:
                    n = int(mm.group(1)); n = n - 40 if n > 30 else n
                    found[n] = (int(m.group(2)), m.group(1))
        for n, (_, u) in found.items():
            urls.setdefault((year, n), u)
        # 広域スキャンで欠けを埋める
        known = sorted(found)
        for no in range(1, 31):
            if (year, no) in urls:
                continue
            lo = max([k for k in known if k < no], default=None)
            hi = min([k for k in known if k > no], default=None)
            guess = found[lo][0] + (no - lo) if lo is not None else found[hi][0] - (hi - no)
            got = None
            for d in range(0, 21):
                for cand in dict.fromkeys([guess + d, guess - d]):
                    u = f"https://1kenchikushi.kakomonn.com/questions/{cand}"
                    html = s1.fetch(u, HTML_DIR / f"scan_{year}_{no}_{cand}.html")
                    if html and s1.page_label(html) == no and page_year(html) == year:
                        got = u
                        (HTML_DIR / f"scan_{year}_{no}_{cand}.html").rename(HTML_DIR / f"q_{year}_{no:02d}.html")
                        break
                    else:
                        (HTML_DIR / f"scan_{year}_{no}_{cand}.html").unlink(missing_ok=True)
                if got:
                    break
            if got:
                urls[(year, no)] = got
                print("filled by scan:", year, no, got)
            else:
                print("STILL unresolved:", year, no)
    rows = []
    mismatches = []
    for year in range(2017, 2027):
        for no in range(1, 31):
            url = urls.get((year, no))
            if not url:
                rows.append({"year": year, "no": no, "source_url": None, "status": "url_not_found"})
                continue
            html = s1.fetch(url, HTML_DIR / f"q_{year}_{no:02d}.html")
            if html is None:
                rows.append({"year": year, "no": no, "source_url": url, "status": "fetch_failed"})
                continue
            if s1.page_label(html) != no or (page_year(html) not in (None, year)):
                print("LABEL MISMATCH:", year, no, url)
                rows.append({"year": year, "no": no, "source_url": url, "status": "label_mismatch"})
                continue
            rec = build_row(year, no, url, html, official)
            rows.append(rec)
            off = official.get((year, no))
            if off and rec["kakomonn_derived_answer"] and rec["kakomonn_derived_answer"] != off:
                mismatches.append((year, no, off, rec["kakomonn_derived_answer"]))
        print(year, "done")
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r.get("status") for r in rows))
    print("derived-vs-official mismatches:", mismatches)
    d2020 = [(r["no"], r["answer"]) for r in rows if r["year"] == 2020]
    print("2020 answers:", d2020)
    print("no-answer:", [(r["year"], r["no"], r.get("status")) for r in rows
                         if r.get("answer") is None and r.get("answer_source") != "special_all_scored"])

if __name__ == "__main__":
    main()
