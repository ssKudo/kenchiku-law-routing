#!/usr/bin/env python3
"""フック集合の構築。
A: 11法クラスタ内 CITES の条文レベル in-degree（異なる引用元条文数）。
   主定義 A5 = in-degree >= 5、感度用 A8 = in-degree >= 8。
B: A5 + leave-one-out おなじみ（governing.jsonl 全299件の正解頻度から、
   当該問題自身を除いた残りで2回以上正解になった条文）。頻度表を保存し、
   LOO の適用は測定側で行う。"""
import json, re, sqlite3, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
from engine import Engine

OUT = Path("/home/hatch/workspace/research_kudou/hook_search")
CFULL = Path("/home/hatch/workspace/research_kudou/prototype/c_full")

eng = Engine()
cluster = sorted({d["law"] for d in eng.docs})
docset = {(d["law"], d["art"]) for d in eng.docs}
names = {d["law"]: d["law_title"] for d in eng.docs}

db = sqlite3.connect("file:/home/hatch/.cache/jlawcite/jp_search.sqlite?mode=ro", uri=True)
def art_of(node):
    m = re.match(r".*?_a([0-9]+(?:-[0-9]+)*)", node)
    return m.group(1) if m else None
def law_of(node):
    return node.split("_")[0]

incl = set(cluster)
srcs = {}
rawc = Counter()
for s, t in db.execute("SELECT source, target FROM edges WHERE rel='CITES'"):
    ls, lt = law_of(s), law_of(t)
    if ls not in incl or lt not in incl:
        continue
    as_, at = art_of(s), art_of(t)
    if not as_ or not at:
        continue
    srcs.setdefault((lt, at), set()).add((ls, as_))
    rawc[(lt, at)] += 1

indeg = {k: len(v) for k, v in srcs.items()}
assert all(k in docset for k in indeg)
A5 = sorted([k for k, v in indeg.items() if v >= 5], key=lambda k: (-indeg[k], k))
A8 = sorted([k for k, v in indeg.items() if v >= 8], key=lambda k: (-indeg[k], k))

def fmt(keys):
    return [{"law_id": l, "law_name": names.get(l, l), "art": a, "indeg": indeg[(l, a)]}
            for (l, a) in keys]

(OUT / "hooks_A.json").write_text(json.dumps({
    "definition": {
        "universe": "法アプリ11法令コーパス全2,726条文",
        "indeg": "CiteGraph CITES エッジで、11法クラスタ内の異なる引用元条文からの引用数（条文レベル、項・号ノードは条に集約）",
        "A5": "in-degree >= 5（主定義）",
        "A8": "in-degree >= 8（感度確認用：扉をさらに絞った版）",
    },
    "n_articles_with_indeg_ge1": len(indeg),
    "A5": fmt(A5), "A8": fmt(A8),
    "indeg_table": [{"law_id": l, "art": a, "indeg": v, "raw_edges": rawc[(l, a)]}
                    for (l, a), v in sorted(indeg.items(), key=lambda kv: -kv[1])],
}, ensure_ascii=False, indent=1), encoding="utf-8")

# おなじみ頻度（300問の正解データ; governing のある299件）
gov = [json.loads(l) for l in (CFULL / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
freq = Counter()
gname = {}
for r in gov:
    g = r.get("governing")
    if g:
        freq[(g["law_id"], g["art"])] += 1
        gname[(g["law_id"], g["art"])] = g["law_name"]
(OUT / "familiar_freq.json").write_text(json.dumps({
    "definition": "governing.jsonl の governing 延べ頻度（299件）。LOO: 問題qでは total - (q自身の正解なら1) >= 2 の条文を q の『おなじみ』とする",
    "freq": [{"law_id": l, "law_name": gname.get((l, a), names.get(l, l)), "art": a, "count": c}
             for (l, a), c in freq.most_common()],
}, ensure_ascii=False, indent=1), encoding="utf-8")
print("A5:", len(A5), "A8:", len(A8), "indeg>=1:", len(indeg))
print("familiar freq>=2 articles:", sum(1 for v in freq.values() if v >= 2))
print("A5 per law:", Counter(l for l, _ in A5))
