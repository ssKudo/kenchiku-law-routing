import json, sys
pt = json.load(open('/home/hatch/workspace/research_kudou/question_tag_pilot/work/pool_texts.json'))
keys = sorted(pt.keys(), key=lambda k: (pt[k]['law_title'], k))
a, b = int(sys.argv[1]), int(sys.argv[2])
lim = int(sys.argv[3]) if len(sys.argv) > 3 else 5500
for i in range(a, b):
    k = keys[i]; v = pt[k]
    t = v['text']
    print(f"\n{'='*60}\nIDX {i} KEY: {k} | {v['law_title']} | {v.get('caption','')} | {len(t)} chars")
    print(t[:lim])
    if len(t) > lim: print(f"……（以下 {len(t)-lim} 字省略）")
