import json, sys
sys.path.insert(0, '/home/hatch/workspace/research_kudou/prototype/c_full/work')
from engine import gram_counts as grams
rows = [json.loads(l) for l in open('/home/hatch/workspace/research_kudou/signboard_pilot2/signboards_v2.jsonl')]
texts = {}
for r in rows:
    texts[f"{r['law_id']}|{r['art']}"] = r['summary'] + ' ' + ' '.join(r['keywords'])
keys = sorted(texts)
g = {k: set(grams(texts[k])) for k in keys}
print('--- top similar pairs (char 2/3-gram Jaccard) ---')
pairs = []
for i, a in enumerate(keys):
    for b in keys[i+1:]:
        inter = len(g[a] & g[b]); union = len(g[a] | g[b])
        if union: pairs.append((inter/union, a, b))
pairs.sort(reverse=True)
for j, a, b in pairs[:25]:
    print(f"{j:.3f}  {a}  <->  {b}")
# keyword collisions: same keyword string in >=3 signboards
from collections import Counter, defaultdict
kw = defaultdict(list)
for r in rows:
    for k in r['keywords']:
        kw[k].append(f"{r['law_id']}|{r['art']}")
print('--- keywords used by >=3 signboards ---')
for k, v in sorted(kw.items(), key=lambda x: -len(x[1])):
    if len(v) >= 3: print(len(v), k, v)
