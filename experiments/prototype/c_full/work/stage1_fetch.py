#!/usr/bin/env python3
"""ステージ1: 過去問ドットコムから 学科III(法規) 2017-2026 の問題を取得し
questions.jsonl を作る。公式正答は answer_index.txt から読み、2020年分は
解説の肢別判定から導出する(9年分で検証)。"""
import json, re, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
HTML_DIR = HERE / "html"
HTML_DIR.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (personal research; polite fetch)"}
LIST_IDS = {2017: 69003, 2018: 69004, 2019: 69005, 2020: 69006, 2021: 69007,
            2022: 69008, 2023: 69009, 2024: 69010, 2025: 69011, 2026: 69012}

def fetch(url, path=None):
    if path and path.exists() and path.stat().st_size > 1000:
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers=UA)
    try:
        html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
    except Exception as e:
        print("fetch failed:", url, e)
        return None
    if path:
        path.write_text(html, encoding="utf-8")
    time.sleep(0.7)
    return html

def clean(t):
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t).strip()

# ---------- 公式正答 (学科III) ----------
def parse_official():
    txt = (HERE / "answer_index.txt").read_text(encoding="utf-8")
    ans = {}
    cur_year = None
    in_hoki = False
    for line in txt.splitlines():
        m = re.match(r"=== (\d{4})年 ===", line)
        if m:
            cur_year = int(m.group(1)); in_hoki = False; continue
        if line.startswith("学科Ⅲ（法規）"):
            in_hoki = True; continue
        if line.startswith("学科") and not line.startswith("学科Ⅲ"):
            in_hoki = False
        if in_hoki and cur_year:
            for mm in re.finditer(r"No\.(\d+):\s*(\d|全員得点)", line):
                ans[(cur_year, int(mm.group(1)))] = (
                    None if mm.group(2) == "全員得点" else int(mm.group(2)))
    return ans

# ---------- 一覧ページから (年, 法規No) -> 質問URL ----------
def _norm_no(n):
    return n - 40 if n > 30 else n

def page_label(html):
    m = re.search(r"（学科3（法規）\s*問(\d+)）", html)
    return _norm_no(int(m.group(1))) if m else None

def question_urls():
    urls = {}
    for year, lid in LIST_IDS.items():
        found = {}
        for page in (1, 2, 3):
            p = HTML_DIR / f"list_{year}_{page}.html"
            html = fetch(f"https://1kenchikushi.kakomonn.com/list1/{lid}?page={page}", p)
            for m in re.finditer(r'href="(https://1kenchikushi\.kakomonn\.com/questions/(\d+))"[^>]*>(.*?)</a>', html, re.S):
                txt = clean(m.group(3))
                mm = re.search(r"学科3（法規）\s*問(\d+)", txt)
                if mm:
                    found[_norm_no(int(mm.group(1)))] = int(m.group(2))
        # 欠けは ID 連番の内挿で補い、詳細ページのラベルで照合する
        known = sorted(found)
        for no in range(1, 31):
            if no in found:
                urls[(year, no)] = f"https://1kenchikushi.kakomonn.com/questions/{found[no]}"
                continue
            lo = max([k for k in known if k < no], default=None)
            hi = min([k for k in known if k > no], default=None)
            guess = None
            if lo is not None:
                guess = found[lo] + (no - lo)
            elif hi is not None:
                guess = found[hi] - (hi - no)
            ok = False
            if guess:
                for cand in (guess, guess - 1, guess + 1, guess - 2, guess + 2):
                    u = f"https://1kenchikushi.kakomonn.com/questions/{cand}"
                    html = fetch(u, HTML_DIR / f"q_{year}_{no:02d}.html")
                    if html and page_label(html) == no:
                        urls[(year, no)] = u; ok = True; break
                    else:
                        (HTML_DIR / f"q_{year}_{no:02d}.html").unlink(missing_ok=True)
            if not ok:
                print("URL unresolved:", year, no)
    return urls

# ---------- 詳細ページのパース ----------
def parse_question(html):
    # 題幹
    m = re.search(r'<div class="problem_detail[^"]*">.*?<div class="ttl"[^>]*>(.*?)</div>', html, re.S)
    stem = clean(m.group(1)) if m else ""
    # 選択肢
    choices = []
    m = re.search(r'<div class="problem_detail.*?<ul class="list"[^>]*>(.*?)</ul>', html, re.S)
    if m:
        for li in re.finditer(r"<li>\s*<div>(.*?)</div>\s*</li>", m.group(1), re.S):
            choices.append(clean(li.group(1)))
    # 解説 (item ごとに top + 肢別)
    explanations = []
    si = html.find('class="sect_commentary"')
    block = html[si:] if si >= 0 else ""
    starts = [m.start() for m in re.finditer(r'<div class="item\s*"\s*>', block)]
    for si2, st in enumerate(starts):
        en = starts[si2 + 1] if si2 + 1 < len(starts) else st + 60000
        body = block[st:en]
        exp = {"top": "", "choices": {}}
        tm = re.search(r'<div class="expound-top">(.*?)</div>\s*<p></p>', body, re.S)
        if tm:
            exp["top"] = clean(tm.group(1))
        for k in (1, 2, 3, 4):
            em = re.search(rf'<div class="expound-{k}">(.*?)</div>\s*<p></p>', body, re.S)
            if em:
                exp["choices"][str(k)] = clean(em.group(1))
        if exp["top"] or exp["choices"]:
            explanations.append(exp)
    return stem, choices, explanations

WRONG = ["誤っている", "誤りです", "不適当", "誤った記述", "×"]
RIGHT = ["適当です", "正しい", "適切です", "〇", "○"]

def derive_answer(stem, explanations, choices=None):
    """解説の肢別判定から正解を導出。戻り: (answer or None, method)"""
    if not explanations:
        return None, "no_explanation"
    ask_wrong = any(s in stem for s in ["誤っている", "不適当", "誤ったもの", "間違っている"])
    ask_right = any(s in stem for s in ["適当なもの", "正しいもの", "正しい記述", "適したもの", "適切なもの"])
    FW = "１２３４"
    def num(g):
        return int(g) if g.isdigit() else FW.index(g) + 1
    # 1) 明示: 正解はNです
    for exp in explanations:
        m = re.search(r"正解は\s*([1-4１２３４])\s*です", exp["top"])
        if m:
            return num(m.group(1)), "derived_explicit_number"
    # 2) 明示: 正解は「選択肢文」です -> 選択肢と照合
    if choices:
        for exp in explanations:
            m = re.search(r"正解は\s*[「『](.+?)[」』]\s*です", exp["top"], re.S)
            if m:
                qt = re.sub(r"\s+", "", m.group(1))
                for i, c in enumerate(choices):
                    ct = re.sub(r"\s+", "", c)
                    if qt[:18] and (qt[:18] in ct or ct[:18] in qt):
                        return i + 1, "derived_explicit_quote"
    # 3) compact 記法: "1 [正]" / "３．×" / "1:〇" / "1:設問通り" (全解説の top を走査)
    for exp in explanations:
        top = exp["top"]
        marks = {}
        for m in re.finditer(r"([1-4])\s*\[(正|誤)\]", top):
            marks[int(m.group(1))] = (m.group(2) == "誤")
        for m in re.finditer(r"([１２３４])\s*[．.:：]\s*([〇○×✕])", top):
            marks[FW.index(m.group(1)) + 1] = (m.group(2) in "×✕")
        for m in re.finditer(r"(?<!\d)([1-4])\s*[:：]\s*(設問通り|誤りです|誤っています)", top):
            marks[int(m.group(1))] = (m.group(2) != "設問通り")
        if len(marks) >= 3:
            wrongs = [k for k, w in marks.items() if w]
            rights = [k for k, w in marks.items() if not w]
            if ask_wrong and len(wrongs) == 1:
                return wrongs[0], "derived_compact_wrong"
            if ask_right and len(rights) == 1:
                return rights[0], "derived_compact_right"
    # 4) 解説1件目の肢別ブロックの判定語
    exp = explanations[0]
    verdict = {}
    for k, txt in exp["choices"].items():
        head = txt[:400]
        is_wrong = any(w in head for w in WRONG)
        is_right = any(w in head for w in RIGHT)
        verdict[int(k)] = (is_wrong, is_right)
    wrongs = [k for k, (w, r) in verdict.items() if w and not r]
    rights = [k for k, (w, r) in verdict.items() if r and not w]
    if ask_wrong and len(wrongs) == 1:
        return wrongs[0], "derived_wrong_choice"
    if ask_right and len(rights) == 1:
        return rights[0], "derived_right_choice"
    return None, f"derive_failed wrongs={wrongs} rights={rights}"

def main():
    official = parse_official()
    urls = question_urls()
    print("urls found:", len(urls))
    rows = []
    mismatches = []
    for year in range(2017, 2027):
        for no in range(1, 31):
            url = urls.get((year, no))
            rec = {"year": year, "no": no, "source_url": url}
            if not url:
                rec["status"] = "url_not_found"; rows.append(rec); continue
            html = fetch(url, HTML_DIR / f"q_{year}_{no:02d}.html")
            if html is None:
                rec["status"] = "fetch_failed"; rows.append(rec); continue
            lab = page_label(html)
            if lab != no:
                print("LABEL MISMATCH:", year, no, "label:", lab, url)
                rec["status"] = "label_mismatch"; rows.append(rec); continue
            stem, choices, explanations = parse_question(html)
            rec.update({"stem": stem, "choices": choices,
                        "explanations": explanations,
                        "has_figure": ("図" in stem and ("示す" in stem or "図に" in stem))})
            derived, method = derive_answer(stem, explanations, choices)
            rec["kakomonn_derived_answer"] = derived
            rec["derive_method"] = method
            off = official.get((year, no), "NA")
            if (year, no) == (2026, 29):
                rec["answer"] = None
                rec["answer_source"] = "special_all_scored"
            elif off != "NA":
                rec["answer"] = off
                rec["answer_source"] = "official_index"
                if derived is not None and off is not None and derived != off:
                    mismatches.append((year, no, off, derived))
            else:  # 2020
                rec["answer"] = derived
                rec["answer_source"] = "kakomonn_derived" if derived else "unconfirmed"
            rec["status"] = "ok" if (stem and len(choices) == 4) else "parse_incomplete"
            rows.append(rec)
        print(year, "done")
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print("rows:", len(rows), "ok:", ok)
    print("official-vs-derived mismatches:", mismatches)
    noans = [(r["year"], r["no"]) for r in rows if r.get("answer") is None and r.get("answer_source") != "special_all_scored"]
    print("no-answer rows:", noans)

if __name__ == "__main__":
    main()
