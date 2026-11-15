#!/usr/bin/env python3
"""クドウの統合案「単語タグ（条文側）× 辞書による自動タグ付け（質問側）× 言葉数え」の測定。
- 物差しは measure50.py の score_docs と同一方式（TF-IDF・文字2/3gram・TERMS/SYN・長さ正規化）。
- 条文側ドキュメント変形:
    tags     = 単語タグのみ（word_tags.jsonl を空白連結→qnorm、見出しブーストなし）
    fulltags = 原文ドキュメントの full にタグ文字列を連結（measure50 の OQ/OS と同じ作法、cap は原文のまま）
- 質問側:
    raw      = stem + choices（measure50 と同一）
    expanded = raw + tag_vector で記録済みの追加語（expansion_dict.json の辞書引き・追加のみ）
               ※ expand() を再実行して記録と全問照合する
- 順位付けは (-score, プール順)。プール外8問は rank=None。
"""
import json, math, sys, statistics
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
TC = RK / "tag_counting"
FM = RK / "fifty_measurement"
CFULL = RK / "prototype/c_full"
sys.path.insert(0, str(CFULL / "work"))
sys.path.insert(0, str(RK / "tag_vector/work"))
import engine as E
from expand import expand as dict_expand

eng = E.Engine()
IDF, AVG = eng.idf, eng.avg_tot

pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
qs = {}
for line in open(CFULL / "questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d
gov = {}
for line in open(CFULL / "governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    gov[f'{d["year"]}-{d["no"]}'] = d
ref = json.load(open(FM / "results50.json", encoding="utf-8"))["questions"]
tv = json.load(open(RK / "tag_vector/results_fifty.json", encoding="utf-8"))["questions"]

tags = {}
for line in open(RK / "tag_vector/word_tags.jsonl", encoding="utf-8"):
    d = json.loads(line)
    tags[(d["law_id"], d["art"])] = d["tags"]

QIDS = list(pools.keys())


def qtext(qid):
    q = qs[qid]
    return q["stem"] + " " + " ".join(q["choices"])


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


def target_of(qid):
    g = gov[qid]["governing"]
    return (g["law_id"], g["art"])


def score_docs(raw, docs):
    """measure50.py の score_docs の複製。"""
    extra = [c for a, c in E.SYN.items() if a in raw and c not in raw]
    qg = E.gram_counts(E.qnorm(raw + "".join(extra)))
    qterms = [t for t in E.TERMS if t in raw or t in extra]
    out = {}
    for doc in docs:
        sc = 0.0
        for g, c in qg.items():
            w = IDF.get(g)
            if w is None:
                continue
            tf = doc["grams"].get(g, 0)
            if tf:
                sc += w * min(c, 3) * (1 + math.log(tf))
            ctf = doc["cap_grams"].get(g, 0)
            if ctf:
                sc += w * 2.2 * (1 + math.log(ctf))
        for t in qterms:
            if t in doc["full"]:
                w = eng._term_w(t)
                sc += w * 1.6
                if t in doc["cap"]:
                    sc += w * 1.1
        sc = sc / (max(doc["total"], 1) / AVG) ** 0.3
        out[(doc["law"], doc["art"])] = sc
    return out


def rank_from_scores(scores, keys, target):
    ordered = sorted(keys, key=lambda k: (-scores.get(k, 0.0), keys.index(k)))
    return ordered.index(target) + 1, ordered


# ---------- ドキュメント変形の構築（条文単位で1回だけ） ----------
def tag_text(key):
    return E.qnorm(" ".join(tags[key]))


tags_docs, fulltags_docs = {}, {}
for key in {(e["law"], e["art"]) for v in pools.values() for e in v}:
    base = eng.docs[eng.by_key[key]]
    tt = tag_text(key)
    tg = E.gram_counts(tt)
    tags_docs[key] = {"law": key[0], "art": key[1], "cap": "", "full": tt,
                      "grams": tg, "total": sum(tg.values()), "cap_grams": {}}
    ffull = base["full"] + tt
    fg = E.gram_counts(ffull)
    fulltags_docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": ffull,
                          "grams": fg, "total": sum(fg.values()), "cap_grams": base["cap_grams"]}

# ---------- 拡張の照合（記録済み追加語 vs expand() 再実行） ----------
expanded_text, expand_mismatch = {}, []
for qid in QIDS:
    raw = qtext(qid)
    recorded = tv[qid]["added_words"]
    _, rerun_added = dict_expand(raw)
    if rerun_added != recorded:
        expand_mismatch.append(qid)
    expanded_text[qid] = raw + " " + " ".join(recorded)
print(f"拡張照合: 記録と再実行の不一致 {len(expand_mismatch)} 件 {expand_mismatch[:5]}")
assert not expand_mismatch, "追加語の記録が expand() 再実行と一致しない"

# ---------- baseline 照合 ----------
mismatch = []
for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    if t in keys:
        sc = score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys])
        r, _ = rank_from_scores(sc, keys, t)
        if r != ref[qid]["rank_original"]:
            mismatch.append((qid, r, ref[qid]["rank_original"]))
print(f"baseline照合: 不一致 {len(mismatch)} 件 {mismatch[:5]}")
assert not mismatch, "baseline が results50 と一致しない"

# ---------- 4条件の測定 ----------
CONDS = {
    "pure_exp_tags": ("expanded", "tags"),      # 1 純粋形（本命）
    "safe_exp_fulltags": ("expanded", "fulltags"),  # 2 安全版
    "raw_tags": ("raw", "tags"),                # 3 切り分け
    "raw_fulltags": ("raw", "fulltags"),        # 4 切り分け
}
DOCMAP = {"tags": tags_docs, "fulltags": fulltags_docs}

results = {"meta": {
    "n_questions": len(QIDS),
    "ruler": "TF-IDF (measure50 score_docs replication; engine idf/avg over full corpus)",
    "question_expansion": "expansion_dict.json lookup, additions only; added words reused from tag_vector/results_fifty.json and verified against expand() re-run",
    "tag_source": "tag_vector/word_tags.jsonl",
    "conditions": list(CONDS.keys()),
}, "questions": {}}

for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    g = gov[qid]["governing"]
    rec = {"governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                         "confidence": gov[qid]["confidence"]},
           "in_pool": t in keys,
           "n_added": tv[qid]["n_added"], "added_words": tv[qid]["added_words"],
           "ranks": {"baseline": ref[qid]["rank_original"]}}
    results["questions"][qid] = rec
    if t not in keys:
        for cname in CONDS:
            rec["ranks"][cname] = None
        continue
    qmap = {"raw": qtext(qid), "expanded": expanded_text[qid]}
    for cname, (qkind, dkind) in CONDS.items():
        docs = [DOCMAP[dkind][k] for k in keys]
        sc = score_docs(qmap[qkind], docs)
        r, ordered = rank_from_scores(sc, keys, t)
        rec["ranks"][cname] = r
        rec[f"governing_score_{cname}"] = round(sc[t], 2)
        rec[f"top5_{cname}"] = [{"law": k[0], "art": k[1], "score": round(sc[k], 2)} for k in ordered[:5]]

    # ---- 純粋形の診断: 正解タグと拡張後質問の重なり ----
    exp_q = expanded_text[qid]
    raw_q = qtext(qid)
    def tag_hits(key, text):
        return [tg for tg in tags[key] if tg in text]
    gh_exp = tag_hits(t, exp_q)
    gh_raw = tag_hits(t, raw_q)
    sc_pure = score_docs(exp_q, [tags_docs[k] for k in keys])
    _, ordered_pure = rank_from_scores(sc_pure, keys, t)
    top1 = ordered_pure[0]
    rec["pure_diag"] = {
        "gov_n_tags": len(tags[t]),
        "gov_tag_hits_expanded": len(gh_exp),
        "gov_tag_hits_raw": len(gh_raw),
        "gov_hit_tags_expanded": gh_exp,
        "top1_key": {"law": top1[0], "art": top1[1]},
        "top1_tag_hits_expanded": len(tag_hits(top1, exp_q)),
        "top1_n_tags": len(tags[top1]),
        "is_top1_governing": top1 == t,
    }

json.dump(results, open(TC / "results_tag_counting.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)


def summarize(name):
    rs = [results["questions"][q]["ranks"].get(name) for q in QIDS]
    inp = [r for r in rs if r is not None]
    top3 = sum(1 for r in inp if r <= 3)
    b = [results["questions"][q]["ranks"]["baseline"] for q in QIDS]
    ent = sum(1 for q, r in zip(QIDS, rs) if r is not None and r <= 3 and (b[QIDS.index(q)] or 99) > 3)
    ext = sum(1 for q, r in zip(QIDS, rs) if r is None or (r > 3 and (b[QIDS.index(q)] or 99) <= 3))
    return top3, round(statistics.mean(inp), 2), statistics.median(inp), ent, ext


print("\n=== 結果 (top-3 / 平均順位 / 中央値 / baseline比 新規top3 / 喪失top3) ===")
t3, mn, md, _, _ = summarize("baseline")
print(f"baseline (参照・再現): top3 {t3}/50 mean {mn} median {md}")
for cname in CONDS:
    t3, mn, md, ent, ext = summarize(cname)
    print(f"{cname}: top3 {t3}/50 mean {mn} median {md} entries {ent} exits {ext}")
print("\nwrote results_tag_counting.json")
