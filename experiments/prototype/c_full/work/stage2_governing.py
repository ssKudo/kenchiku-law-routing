#!/usr/bin/env python3
"""ステージ2: 解説文から条文引用を抽出し、正答条 (根拠条文) を特定する。
- 正答肢の解説が引く条文を主値とし、全肢分の引用も併記する。
- 引用は (a) 正式名つき引用、(b) 略称引用 (法/令/規則+第○条; 設問の支配的法令群で解決)、
  (c) 引用が無い場合は TF-IDF フォールバック、の順で証拠の強さを付ける。
- 各候補は条文テキストとの TF-IDF 類似度で裏づけスコアを持つ。"""
import json, re, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
DB = "file:/home/hatch/.cache/jlawcite/jp_search.sqlite?mode=ro"

LAW_ALIASES = [
    ("建築基準法施行規則", "325M50004000040"),
    ("建築基準法施行令", "325CO0000000338"),
    ("建築基準法", "325AC0000000201"),
    ("建築士法施行規則", None), ("建築士法施行令", None), ("建築士法", "325AC1000000202"),
    ("都市計画法施行規則", None), ("都市計画法施行令", None), ("都市計画法", "343AC0000000100"),
    ("消防法施行規則", None), ("消防法施行令", None), ("消防法", "323AC1000000186"),
    ("建築物のエネルギー消費性能の向上等に関する法律", "427AC0000000053"),
    ("建築物のエネルギー消費性能の向上に関する法律", "427AC0000000053"),
    ("省エネ法", "427AC0000000053"), ("建築物省エネ法", "427AC0000000053"),
    ("高齢者、障害者等の移動等の円滑化の促進に関する法律", "418AC0000000091"),
    ("バリアフリー法", "418AC0000000091"),
    ("建築物の耐震改修の促進に関する法律", "407AC0000000123"), ("耐震改修促進法", "407AC0000000123"),
    ("長期優良住宅の普及の促進に関する法律", "420AC0000000087"), ("長期優良住宅法", "420AC0000000087"),
    ("建設業法", "324AC0000000100"),
    ("住宅の品質確保の促進等に関する法律", None), ("品確法", None),
    ("景観法", None), ("宅地造成等規制法", None), ("水道法", None), ("下水道法", None),
    ("文化財保護法", None), ("駐車場法", None), ("屋外広告物法", None),
    ("高齢者の居住の安定確保に関する法律", None),
    ("マンションの建替え等の円滑化に関する法律", None),
    ("密集市街地における防災街区の整備の促進に関する法律", None),
]
# 略称解決のための法令群: 群名 -> {法: id, 令: id or title pattern, 規則: ...}
FAMILIES = {
    "建築基準法": {"法": "325AC0000000201", "令": "325CO0000000338", "規則": "325M50004000040"},
    "建築士法": {"法": "325AC1000000202", "令": "建築士法施行令", "規則": "建築士法施行規則"},
    "都市計画法": {"法": "343AC0000000100", "令": "都市計画法施行令", "規則": "都市計画法施行規則"},
    "消防法": {"法": "323AC1000000186", "令": "消防法施行令", "規則": "消防法施行規則"},
}

KAN = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
def kanji_num(s):
    if not s:
        return None
    total, cur = 0, 0
    for ch in s:
        if ch in KAN:
            cur = KAN[ch]
        elif ch == "十":
            total += (cur or 1) * 10; cur = 0
        elif ch == "百":
            total += (cur or 1) * 100; cur = 0
        else:
            return None
    return total + cur

def parse_article_after(text, i):
    """text[i:] の先頭から 第N条のNのN を読む。戻り: (art_key, consumed) or (None, 0)"""
    m = re.match(r"第?\s*(\d+)\s*条(?:\s*の\s*(\d+))?(?:\s*の\s*(\d+))?", text[i:])
    if m:
        parts = [g for g in m.groups() if g]
        return "-".join(parts), m.end()
    m = re.match(r"第?\s*([〇一二三四五六七八九十百]+)\s*条(?:\s*の\s*([〇一二三四五六七八九十百]+))?(?:\s*の\s*([〇一二三四五六七八九十百]+))?", text[i:])
    if m:
        nums = [kanji_num(g) for g in m.groups() if g]
        if all(n is not None for n in nums):
            return "-".join(str(n) for n in nums), m.end()
    return None, 0

class Resolver:
    def __init__(self):
        self.db = sqlite3.connect(DB, uri=True)
        self.title2id = dict(self.db.execute("SELECT title, law_id FROM laws"))
        self.alias2id = {}
        for name, lid in LAW_ALIASES:
            if lid:
                self.alias2id[name] = lid
            elif name in self.title2id:
                self.alias2id[name] = self.title2id[name]
        # chunk 存在チェック用: (law_id, article_id) 集合は大きいので都度問い合わせる
    def law_id_by_title(self, title):
        return self.title2id.get(title)
    def article_exists(self, law_id, art):
        if not law_id:
            return False
        r = self.db.execute("SELECT 1 FROM chunks WHERE law_id=? AND article_id=? LIMIT 1",
                            (law_id, f"{law_id}_a{art}")).fetchone()
        return r is not None

def extract_citations(text, family, resolver):
    """テキストから (law_id, art, kind, raw) のリストを返す。kind: formal/abbrev"""
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    cites = []
    spans = []
    # 正式名つき: 法令名の直後に 第N条... が続く (第は省略形も可)
    for name, lid in resolver.alias2id.items():
        start = 0
        while True:
            i = text.find(name, start)
            if i < 0:
                break
            j = i + len(name)
            k = j
            while k < len(text) and text[k] in " 　":
                k += 1
            art, used = parse_article_after(text, k)
            if art:
                cites.append({"law_id": lid, "law_name": name, "art": art,
                              "kind": "formal", "raw": text[i:k + used]})
                spans.append((i, k + used))
            start = j
    # 文単位: 文中に法令名が1つと 第N条 があれば結びつける
    for sent in re.split(r"[。\n]", text):
        named = [(name, lid) for name, lid in resolver.alias2id.items() if name in sent]
        if len(named) == 1:
            name, lid = named[0]
            for m in re.finditer(r"第\s*(\d+)\s*条(?:\s*の\s*(\d+))?(?:\s*の\s*(\d+))?", sent):
                art = "-".join(g for g in m.groups() if g)
                cites.append({"law_id": lid, "law_name": name, "art": art,
                              "kind": "formal_sentence", "raw": m.group(0)})
    # 略称: 法/令/規則 + (第)?N条 (正式名スパンの内側は除く)
    for m in re.finditer(r"(法|令|規則)\s*第?\s*(\d+|[〇一二三四五六七八九十百]+)\s*条(?:\s*の\s*(\d+|[〇一二三四五六七八九十百]+))?(?:\s*の\s*(\d+|[〇一二三四五六七八九十百]+))?", text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        # 直前が法令名の一部 (例: 港湾法の法) の場合は法令名として扱わない
        prev = text[m.start() - 1] if m.start() > 0 else ""
        slot = FAMILIES.get(family, FAMILIES["建築基準法"])[m.group(1)]
        lid = slot if slot and slot.startswith("3") else resolver.law_id_by_title(slot or "")
        nums = []
        for g in m.groups()[1:]:
            if g is None:
                continue
            nums.append(str(int(g)) if g.isdigit() else str(kanji_num(g)))
        cites.append({"law_id": lid, "law_name": family + ("令" if m.group(1) == "令" else "規則" if m.group(1) == "規則" else ""),
                      "art": "-".join(nums), "kind": "abbrev", "raw": m.group(0)})
    return cites

def main():
    resolver = Resolver()
    engine = Engine()
    rows = [json.loads(l) for l in (OUT / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        rec = {"year": r["year"], "no": r["no"]}
        if r.get("answer_source") == "special_all_scored":
            rec["special"] = "all_scored"; rec["governing"] = None
            out.append(rec); continue
        if r.get("status") not in ("ok", "ok_image_choices") or r.get("answer") is None:
            rec["governing"] = None
            rec["note"] = r.get("status") if r.get("status") not in ("ok", "ok_image_choices") else "no_answer"
            out.append(rec); continue
        ans = r["answer"]
        # 設問全体の支配的法令群
        all_text = " ".join([r["stem"]] + r["choices"] +
                            [e["top"] for e in r["explanations"]] +
                            [t for e in r["explanations"] for t in e["choices"].values()])
        fam_votes = {f: all_text.count(f) for f in FAMILIES}
        family = max(fam_votes, key=fam_votes.get) if max(fam_votes.values()) > 0 else "建築基準法"
        per_choice = {}
        # 旧形式 (2020年等): 解説 top 内の肢別セグメントからも引用を拾う
        top_segments = {str(ci): [] for ci in (1, 2, 3, 4)}
        for e in r["explanations"]:
            top = e["top"]
            marks = [(m.start(), "１２３４".index(m.group(1)) + 1 if m.group(1) in "１２３４" else int(m.group(1)))
                     for m in re.finditer(r"(?<![0-9０-９])([1-4１２３４])(?:\s*\[(?:正|誤)\]|\s*[:：]\s*(?:設問|誤り|正しい|〇|○|×|✕)|\s*[．.]\s*(?:設問|誤り|正しい|適当|不適当|〇|○|×|✕))", top)]
            marks = [(p, n) for p, n in marks if 1 <= n <= 4]
            if len(marks) >= 2:
                for i, (p, n) in enumerate(marks):
                    en = marks[i + 1][0] if i + 1 < len(marks) else len(top)
                    top_segments[str(n)].append(top[p:en])
        for ci in (1, 2, 3, 4):
            cites = []
            for e in r["explanations"]:
                txt = e["choices"].get(str(ci), "")
                if txt:
                    cites += extract_citations(txt, family, resolver)
            for seg in top_segments[str(ci)]:
                cites += extract_citations(seg, family, resolver)
            # 重複除去 (law_id, art)
            seen = {}
            for c in cites:
                seen[(c["law_id"], c["art"])] = c
            per_choice[str(ci)] = list(seen.values())
        rec["choice_citations"] = per_choice
        cand = per_choice.get(str(ans), [])
        # 候補の裏づけ: 選択肢文との TF-IDF スコア
        choice_text = r["choices"][ans - 1] if len(r["choices"]) >= ans else ""
        scores = engine.score_all(r["stem"] + " " + choice_text)
        for c in cand:
            c["exists"] = resolver.article_exists(c["law_id"], c["art"])
            c["tfidf"] = round(scores.get((c["law_id"], c["art"]), 0.0), 3)
        formal = [c for c in cand if c["kind"] in ("formal", "formal_sentence") and c["exists"]]
        abbrev = [c for c in cand if c["kind"] == "abbrev" and c["exists"]]
        pool = formal or abbrev
        if len(pool) == 1:
            g = pool[0]
            rec["governing"] = {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"]}
            rec["confidence"] = "high"
            rec["basis"] = f"解説の明示引用 ({g['kind']}): {g['raw']}"
        elif len(pool) > 1:
            pool.sort(key=lambda c: -c["tfidf"])
            g = pool[0]
            rec["governing"] = {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"]}
            rec["confidence"] = "medium"
            rec["basis"] = "複数の引用候補から選択肢文とのTF-IDF類似度で主値を選定: " + \
                          ", ".join(f"{c['law_name']}{c['art']}条({c['tfidf']})" for c in pool)
            rec["alternatives"] = [{"law_id": c["law_id"], "law_name": c["law_name"], "art": c["art"]} for c in pool[1:]]
        else:
            top = engine.rank(r["stem"] + " " + choice_text, 3)
            if top:
                (lid, art), sc = top[0]
                name = next((m["title"] for m in engine.index if m["id"] == lid), lid)
                rec["governing"] = {"law_id": lid, "law_name": name, "art": art}
                rec["confidence"] = "low"
                rec["basis"] = f"解説に条文引用なし。TF-IDFフォールバック (score {sc:.2f})"
                rec["alternatives"] = [{"law_id": l2, "art": a2} for (l2, a2), _ in top[1:]]
            else:
                rec["governing"] = None; rec["confidence"] = "low"; rec["basis"] = "特定不能"
        out.append(rec)
        print(r["year"], r["no"], "ans", ans, "->", rec.get("governing"), rec.get("confidence"))
    with open(OUT / "governing.jsonl", "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r.get("confidence") for r in out))

if __name__ == "__main__":
    main()
