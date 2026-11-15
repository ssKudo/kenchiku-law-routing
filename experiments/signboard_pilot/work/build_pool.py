import json, sys
sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
from engine import Engine

eng = Engine()
qs = [json.loads(l) for l in open("/home/hatch/workspace/research_kudou/small_model_test/blind_questions.jsonl", encoding="utf-8") if l.strip()]
pools = {}
for q in qs:
    raw = q["stem"] + " " + " ".join(q["choices"])
    ranked = eng.rank(raw, limit=30)
    pools[f"{q['year']}-{q['no']}"] = [{"law": k[0], "art": k[1], "score": round(v, 3)} for k, v in ranked]
    print(q['year'], q['no'], "top1:", ranked[0][0], ranked[0][1])
json.dump(pools, open("pools.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
uniq = {(p["law"], p["art"]) for ps in pools.values() for p in ps}
print("unique articles in pools:", len(uniq))
