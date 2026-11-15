#!/usr/bin/env python3
"""tag_parts の各モジュールを結合し word_tags.jsonl を作る（既存分は保持し上書き更新）。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "tag_parts"))
TV = Path("/home/hatch/workspace/research_kudou/tag_vector")

merged = {}
import importlib
for name in ["p01","p01b","p02","p02b","p03","p03b","p04","p04b","p05","p05b","p06","p06b","f01","f01b","f02","f02b","f03","f03b","f04","f04b","f05","f05b","f06","f06b","f07","f07b","f08","f08b","f09","f09b","f10","f10b","f11","f11b","f12","f12b","f13","f13b","f14","f14b","f15","f15b","f16","f16b","f17","f17b","f18","f18b","f19","f19b"]:
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError:
        continue
    src = getattr(m, "TAGS", None) or getattr(m, "ADD", None)
    for k, words in src.items():
        cur = merged.setdefault(k, [])
        for w in words:
            if w not in cur:
                cur.append(w)

out = TV / "word_tags.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for k in sorted(merged):
        law, art = k.split("|")
        f.write(json.dumps({"law_id": law, "art": art, "tags": merged[k]}, ensure_ascii=False) + "\n")
counts = [len(v) for v in merged.values()]
print(f"articles={len(merged)} words: min={min(counts)} mean={sum(counts)/len(counts):.1f} max={max(counts)}")
