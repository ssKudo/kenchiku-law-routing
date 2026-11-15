#!/usr/bin/env python3
"""ベクトル検索＋RRF 合わせ技の測定。
使い方: python measure_vector.py pilot|fifty
条件: a=原文 b=原文+質問タグ c=原文+看板v2 d=原文+両方 e=原文+描写看板×変換問題
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embedder import E5

RK = Path("/home/hatch/workspace/research_kudou")
VH = RK / "vector_hybrid"
CONDS = ["a", "b", "c", "d", "e"]


def load_jsonl_map(path, keyf, valf):
    m = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m[keyf(d)] = valf(d)
    return m


def qtext(d):
    return d["stem"] + " " + " ".join(d["choices"])


def rrf_order(tfidf_order, vec_order, pool_keys, k=60):
    t_rank = {key: i + 1 for i, key in enumerate(tfidf_order)}
    v_rank = {key: i + 1 for i, key in enumerate(vec_order)}
    return sorted(pool_keys,
                  key=lambda key: (-(1.0 / (k + t_rank[key]) + 1.0 / (k + v_rank[key])),
                                   pool_keys.index(key)))


def main(dataset):
    tfidf = json.load(open(VH / "work/tfidf_rankings.json", encoding="utf-8"))[dataset]
    if dataset == "pilot":
        pools = json.load(open(RK / "signboard_pilot/work/pools.json", encoding="utf-8"))
        pool_texts = json.load(open(RK / "question_tag_pilot/work/pool_texts.json", encoding="utf-8"))
        questions, conv = {}, {}
        for line in open(RK / "small_model_test/blind_questions.jsonl", encoding="utf-8"):
            d = json.loads(line)
            questions[f'{d["year"]}-{d["no"]}'] = qtext(d)
        for line in open(RK / "descriptive_pilot/converted_questions_pilot.jsonl", encoding="utf-8"):
            d = json.loads(line)
            conv[d["qid"]] = qtext(d)
        qt = load_jsonl_map(RK / "question_tag_pilot/qtags.jsonl",
                            lambda d: (d["law_id"], d["art"]),
                            lambda d: " ".join(d["questions"]) + " " + " ".join(d["words"]))
        sb = load_jsonl_map(RK / "signboard_pilot2/signboards_v2.jsonl",
                            lambda d: (d["law_id"], d["art"]),
                            lambda d: d["summary"] + " " + " ".join(d["keywords"]))
        bd = load_jsonl_map(RK / "descriptive_pilot/boards_pilot.jsonl",
                            lambda d: (d["law_id"], d["art"]), lambda d: d["text"])
    else:
        pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
        pool_texts = json.load(open(RK / "question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
        questions, conv = {}, {}
        for line in open(RK / "prototype/c_full/questions.jsonl", encoding="utf-8"):
            d = json.loads(line)
            questions[f'{d["year"]}-{d["no"]}'] = qtext(d)
        for line in open(RK / "descriptive_pilot/converted_questions50.jsonl", encoding="utf-8"):
            d = json.loads(line)
            conv[d["qid"]] = qtext(d)
        qt = load_jsonl_map(RK / "fifty_measurement/qtags50.jsonl",
                            lambda d: (d["law_id"], d["art"]),
                            lambda d: " ".join(d["questions"]) + " " + " ".join(d["words"]))
        sb = load_jsonl_map(RK / "fifty_measurement/signboards50.jsonl",
                            lambda d: (d["law_id"], d["art"]),
                            lambda d: d["summary"] + " " + " ".join(d["keywords"]))
        bd = load_jsonl_map(RK / "descriptive_pilot/boards50.jsonl",
                            lambda d: (d["law_id"], d["art"]), lambda d: d["text"])

    orig_text = {tuple(k.split("|")): v["text"] for k, v in pool_texts.items()}

    def doc_text(key, cond):
        base = orig_text[key]
        if cond == "a":
            return base
        if cond == "b":
            return base + "\n" + qt[key]
        if cond == "c":
            return base + "\n" + sb[key]
        if cond == "d":
            return base + "\n" + sb[key] + "\n" + qt[key]
        if cond == "e":
            return base + "\n" + bd[key]
        raise ValueError(cond)

    all_keys = sorted({(e["law"], e["art"]) for entries in pools.values() for e in entries})
    print(f"{dataset}: questions={len(pools)} unique articles={len(all_keys)}", flush=True)

    model = E5()
    results = {"dataset": dataset, "questions": {}}
    char_stats = {}
    for cond in CONDS:
        t0 = time.time()
        texts = [doc_text(k, cond) for k in all_keys]
        char_stats[cond] = {"mean_chars": round(float(np.mean([len(t) for t in texts])), 1),
                            "total_chars": int(sum(len(t) for t in texts))}
        doc_emb = model.embed_texts(texts, "passage: ")
        emb_map = {k: doc_emb[i] for i, k in enumerate(all_keys)}
        np.save(VH / f"work/doc_emb_{dataset}_{cond}.npy", doc_emb)
        print(f"cond {cond}: docs embedded in {time.time()-t0:.1f}s "
              f"(mean {char_stats[cond]['mean_chars']} chars)", flush=True)

        qside = "desc" if cond == "e" else "orig"
        for qid, entries in pools.items():
            pool_keys = [(e["law"], e["art"]) for e in entries]
            q = (conv if qside == "desc" else questions)[qid]
            qe = model.embed_texts([q], "query: ")[0]
            sims = {k: float(emb_map[k] @ qe) for k in pool_keys}
            vec_order_t = sorted(pool_keys, key=lambda k: (-sims[k], pool_keys.index(k)))
            vec_order = [f"{l}|{a}" for l, a in vec_order_t]
            tf_order = tfidf[qid]["conditions"][cond]["order"]
            hyb_order_t = rrf_order([tuple(x.split("|")) for x in tf_order],
                                    vec_order_t, pool_keys)
            hyb_order = [f"{l}|{a}" for l, a in hyb_order_t]
            g = tfidf[qid]["governing"]
            target = f'{g["law"]}|{g["art"]}'
            rec = results["questions"].setdefault(qid, {
                "governing": g, "in_pool": tfidf[qid]["in_pool"], "conditions": {}})
            def rank_of(order):
                return order.index(target) + 1 if target in order else None
            rec["conditions"][cond] = {
                "rank_tfidf": rank_of(tf_order),
                "rank_vector": rank_of(vec_order),
                "rank_hybrid": rank_of(hyb_order),
                "governing_cos": sims.get((g["law"], g["art"])),
                "vector_top5": vec_order[:5],
                "hybrid_top5": hyb_order[:5],
            }
        print(f"cond {cond}: ranking done", flush=True)

    results["char_stats"] = char_stats
    out_path = VH / f"results_{dataset}.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out_path)

    # サマリ
    for cond in CONDS:
        for method in ["tfidf", "vector", "hybrid"]:
            ranks = [q["conditions"][cond][f"rank_{method}"]
                     for q in results["questions"].values()]
            inpool = [r for r in ranks if r is not None]
            top3 = sum(1 for r in ranks if r is not None and r <= 3)
            print(f"{cond} {method}: top3={top3}/{len(ranks)} "
                  f"mean_rank(in-pool n={len(inpool)})={np.mean(inpool):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
