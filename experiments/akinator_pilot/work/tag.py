#!/usr/bin/env python3
"""アキネーター式パイロット 手順2: プール123条への属性タグ付け。

- 対象は看板パイロットの pools.json に現れる全 (law, art) = 123条。
- タグ付けは attributes.json のキーワード規則による機械判定のみ
  (条文の見出し+本文+看板の要約・キーワードを対象テキストとする)。
- この段階では governing.jsonl は読まない。
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # work/
OUT = HERE.parent                                # akinator_pilot/
LAWAPP = Path("/home/hatch/workspace/lawapp")
SIGN = Path("/home/hatch/workspace/research_kudou/signboard_pilot")

attrs = json.loads((OUT / "attributes.json").read_text(encoding="utf-8"))["attributes"]
pools = json.loads((SIGN / "work" / "pools.json").read_text(encoding="utf-8"))
uniq = sorted({(p["law"], p["art"]) for ps in pools.values() for p in ps})
signboards = {}
for line in (SIGN / "signboards.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        signboards[(r["law_id"], r["art"])] = r

cache = {}
def get_article(law, art):
    if law not in cache:
        cache[law] = json.loads((LAWAPP / f"data/{law}.json").read_text(encoding="utf-8"))
    for a in cache[law]["articles"]:
        if a["key"].replace("_", "-") == art:
            return a
    return None

DEF_RE = re.compile(r"とは.{0,60}?をいう")

recs = []
missing = []
for law, art in uniq:
    a = get_article(law, art)
    if a is None:
        missing.append((law, art))
        continue
    caption = a.get("caption") or ""
    body = "\n".join(p["text"] for p in a["paras"])
    sb = signboards.get((law, art), {})
    aux = (sb.get("summary") or "") + " " + " ".join(sb.get("keywords") or [])
    text = caption + "\n" + body + "\n" + aux
    tags, evidence = {}, {}
    for at in attrs:
        if at["id"] == "definition":
            hit = ("定義" in caption) or bool(DEF_RE.search(body))
            tags["definition"] = bool(hit)
            evidence["definition"] = "caption/pattern" if hit else ""
        else:
            hits = [kw for kw in at["tag_keywords"] if kw in text]
            tags[at["id"]] = len(hits) > 0
            evidence[at["id"]] = hits[:3]
    recs.append({"law_id": law, "art": art, "tags": tags, "evidence": evidence})

with open(OUT / "tags.jsonl", "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("tagged:", len(recs), "missing:", missing)
from collections import Counter
c = Counter()
for r in recs:
    for k, v in r["tags"].items():
        if v:
            c[k] += 1
for at in attrs:
    print("%-12s yes=%3d / %d" % (at["id"], c[at["id"]], len(recs)))
