#!/usr/bin/env python3
"""二択の連続パイロット 手順2: プール123条への「側」付け。

- 対象は看板パイロットの pools.json に現れる全 (law, art) = 123条。
- 側付けは axes.json の side_rule による機械判定のみ:
  側スコア = 5×(その側の属性タグ true 数) + キーワード出現回数の合計
  (対象テキストはアキネーター式と同じ 見出し+本文+看板補助)。
  同点タイブレーク: タグ数 → 出現回数 → 側A。
- 属性タグとキーワードはアキネーター式 attributes.json / tags.jsonl を再利用。
- この段階では governing.jsonl は読まない。
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # work/
OUT = HERE.parent                                # binary_chain_pilot/
LAWAPP = Path("/home/hatch/workspace/lawapp")
SIGN = Path("/home/hatch/workspace/research_kudou/signboard_pilot")
AKI = Path("/home/hatch/workspace/research_kudou/akinator_pilot")

axes = json.loads((OUT / "axes.json").read_text(encoding="utf-8"))["axes"]
attrs = json.loads((AKI / "attributes.json").read_text(encoding="utf-8"))["attributes"]
ATTR_KW = {a["id"]: a.get("tag_keywords", []) for a in attrs}
aki_tags = {}
for line in (AKI / "tags.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        aki_tags[(r["law_id"], r["art"])] = r["tags"]

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

def side_score(side_attrs, tags, text, body):
    n_tags = sum(1 for aid in side_attrs if tags.get(aid))
    occ = 0
    for aid in side_attrs:
        if aid == "definition":
            occ += len(DEF_RE.findall(body))
        else:
            for kw in ATTR_KW[aid]:
                occ += text.count(kw)
    return 5 * n_tags + occ, n_tags, occ

recs, missing = [], []
split_summary = {}
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
    tags = aki_tags[(law, art)]
    sides, detail = {}, {}
    for ax in axes:
        sA = side_score(ax["attrs_A"], tags, text, body)
        sB = side_score(ax["attrs_B"], tags, text, body)
        tie_broken = False
        if sA[0] != sB[0]:
            side = "A" if sA[0] > sB[0] else "B"
        elif sA[1] != sB[1]:
            side = "A" if sA[1] > sB[1] else "B"; tie_broken = True
        elif sA[2] != sB[2]:
            side = "A" if sA[2] > sB[2] else "B"; tie_broken = True
        else:
            side = None  # 完全同点は中立 (どちらの選択にも一致しない)
        sides[ax["id"]] = (ax["label_A"] if side == "A"
                           else ax["label_B"] if side == "B" else None)
        detail[ax["id"]] = {"side": side, "score_A": sA[0], "score_B": sB[0],
                            "tags_A": sA[1], "tags_B": sB[1],
                            "occ_A": sA[2], "occ_B": sB[2], "tie_broken": tie_broken,
                            "neutral": side is None}
        split_summary.setdefault(ax["id"], {"A": 0, "B": 0, "neutral": 0})
        split_summary[ax["id"]][side if side else "neutral"] += 1
    recs.append({"law_id": law, "art": art, "sides": sides, "detail": detail})

with (OUT / "sides.jsonl").open("w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("sides saved:", len(recs), "missing:", missing)
print("pool split (123条):", json.dumps(split_summary, ensure_ascii=False))

# 問ごとプール (30条) の割れ方も参考に表示
for qid, pool in pools.items():
    line = [qid]
    bykey = {(r["law_id"], r["art"]): r for r in recs}
    for ax in axes:
        nA = sum(1 for p in pool if bykey[(p["law"], p["art"])]["detail"][ax["id"]]["side"] == "A")
        nB = sum(1 for p in pool if bykey[(p["law"], p["art"])]["detail"][ax["id"]]["side"] == "B")
        line.append("%s A%d/B%d/N%d" % (ax["id"], nA, nB, len(pool) - nA - nB))
    print(" | ".join(line))
