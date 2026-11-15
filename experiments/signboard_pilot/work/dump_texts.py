import json
from pathlib import Path
LAWAPP = Path("/home/hatch/workspace/lawapp")
pools = json.load(open("pools.json", encoding="utf-8"))
uniq = sorted({(p["law"], p["art"]) for ps in pools.values() for p in ps})
idx_js = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
index = json.loads(idx_js[idx_js.index("["):idx_js.rindex("]") + 1])
titles = {m["id"]: m["title"] for m in index}
cache = {}
def get_article(law, art):
    if law not in cache:
        cache[law] = json.loads((LAWAPP / f"data/{law}.json").read_text(encoding="utf-8"))
    for a in cache[law]["articles"]:
        if a["key"].replace("_", "-") == art:
            return a
    return None
out = []
for law, art in uniq:
    a = get_article(law, art)
    if a is None:
        out.append(f"@@@ {law} | {art} | NOT FOUND")
        continue
    disp = a.get("label") or a.get("title") or ("第" + str(a.get("num")) + "条")
    body = "\n".join(p["text"] for p in a["paras"])
    out.append(f"@@@ {law} | {art} | {titles.get(law,'')} {disp} | 見出し: {a.get('caption') or ''}\n{body}")
open("pool_texts.txt", "w", encoding="utf-8").write("\n\n".join(out))
print("articles dumped:", len(uniq))
import os
print("chars:", os.path.getsize("pool_texts.txt"))
