#!/usr/bin/env python3
"""単語タグ×ベクトル検索の測定。
条文側: word_tags.jsonl のタグ列を1テキストとして passage: 埋め込み。
条件A: 原文の質問を query: 埋め込み。条件B: 辞書拡張した質問を query: 埋め込み。
プール内コサイン順位付け（同点はプール順）。
使い方: python measure.py pilot|fifty
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/research_kudou/vector_hybrid/work")
sys.path.insert(0, str(Path(__file__).parent))
from embedder import E5
from expand import expand

RK = Path("/home/hatch/workspace/research_kudou")
TV = RK / "tag_vector"


def qtext(d):
    return d["stem"] + " " + " ".join(d["choices"])


def main(ds):
    tfidf = json.load(open(RK / "vector_hybrid/work/tfidf_rankings.json", encoding="utf-8"))[ds]
    if ds == "pilot":
        pools = json.load(open(RK / "signboard_pilot/work/pools.json", encoding="utf-8"))
        qsrc = RK / "small_model_test/blind_questions.jsonl"
    else:
        pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
        qsrc = RK / "prototype/c_full/questions.jsonl"
    questions = {}
    for line in open(qsrc, encoding="utf-8"):
        d = json.loads(line)
        questions[f'{d["year"]}-{d["no"]}'] = qtext(d)
    tags = {}
    for line in open(TV / "word_tags.jsonl", encoding="utf-8"):
        d = json.loads(line)
        tags[(d["law_id"], d["art"])] = d["tags"]

    all_keys = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})
    missing = [k for k in all_keys if k not in tags]
    assert not missing, f"tags missing: {missing[:5]}"
    tag_counts = [len(tags[k]) for k in all_keys]

    model = E5()
    t0 = time.time()
    doc_emb = model.embed_texts([" ".join(tags[k]) for k in all_keys], "passage: ")
    emb_map = {k: doc_emb[i] for i, k in enumerate(all_keys)}
    print(f"{ds}: {len(all_keys)} tag docs embedded in {time.time()-t0:.1f}s; "
          f"tag words min={min(tag_counts)} mean={np.mean(tag_counts):.1f} max={max(tag_counts)}", flush=True)

    results = {"dataset": ds, "questions": {}}
    added_stats = []
    for qid, entries in pools.items():
        pool_keys = [(e["law"], e["art"]) for e in entries]
        q_orig = questions[qid]
        q_exp, added = expand(q_orig)
        added_stats.append(len(added))
        rec = {"governing": tfidf[qid]["governing"], "in_pool": tfidf[qid]["in_pool"],
               "n_added": len(added), "added_words": added}
        g = tfidf[qid]["governing"]
        target = (g["law"], g["art"])
        for cond, qq in (("A", q_orig), ("B", q_exp)):
            qe = model.embed_texts([qq], "query: ")[0]
            sims = {k: float(emb_map[k] @ qe) for k in pool_keys}
            order = sorted(pool_keys, key=lambda k: (-sims[k], pool_keys.index(k)))
            rec[f"rank_{cond}"] = order.index(target) + 1 if target in order else None
            rec[f"cos_{cond}"] = sims.get(target)
            rec[f"top5_{cond}"] = [f"{l}|{a}" for l, a in order[:5]]
        results["questions"][qid] = rec
    results["tag_stats"] = {"n_articles": len(all_keys), "min": int(min(tag_counts)),
                            "mean": float(np.mean(tag_counts)), "max": int(max(tag_counts))}
    results["expansion_stats"] = {"mean_added": float(np.mean(added_stats)),
                                  "min_added": int(min(added_stats)), "max_added": int(max(added_stats))}
    out = TV / f"results_{ds}.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out)
    for cond in ("A", "B"):
        ranks = [q[f"rank_{cond}"] for q in results["questions"].values()]
        inpool = [r for r in ranks if r is not None]
        top3 = sum(1 for r in ranks if r is not None and r <= 3)
        print(f"{cond}: top3={top3}/{len(ranks)} mean_rank(in-pool n={len(inpool)})={np.mean(inpool):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
