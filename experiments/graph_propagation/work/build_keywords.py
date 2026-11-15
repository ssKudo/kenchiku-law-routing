#!/usr/bin/env python3
"""フック条文への語彙吸い上げ: child_keywords 構築。

抽出ルール（測定前に固定・過去問データは不使用）:
1. 隣接 N(h) = Graph.expand(h) の文脈キー − h 自身
   (tier1 逆引き委任受け手 / tier2 引用先 / tier3 引用元・先頭40件キャップ)
2. 隣接本文: エンジンコーパス内なら engine doc の full (qnorm済)、外なら CiteGraph chunks 連結を qnorm
3. 候補語: (a) janome 名詞 (数詞・接尾・代名詞・非自立を除く、2文字以上)
           (b) エンジン TERMS (+SYN値) のうち隣接本文に現れるもの
4. cover(w) = 隣接条文のうち w を含む (部分文字列) 条文数。cover >= 2 必須
5. idf(w): df = エンジンコーパス 2,726 条文での部分文字列文書頻度、エンジンと同じ式
6. score = cover × idf。フック自身の本文に含まれる語は除外。上位12語
"""
import json, math, sqlite3, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/research_kudou/prototype/c_full/work")
from engine import Engine, TERMS, SYN, qnorm
from stage3_measure import Graph

from janome.tokenizer import Tokenizer

ROOT = Path("/home/hatch/workspace/research_kudou")
OUT = ROOT / "graph_propagation"
DB = "file:/home/hatch/.cache/jlawcite/jp_search.sqlite?mode=ro"
TOPK = 12
MIN_COVER = 2

_tok = Tokenizer()
EXCLUDE_SUB = {"数", "接尾", "代名詞", "非自立"}


def nouns_of(text):
    out = set()
    for t in _tok.tokenize(text):
        pos = t.part_of_speech.split(",")
        if pos[0] != "名詞":
            continue
        if len(pos) > 1 and pos[1] in EXCLUDE_SUB:
            continue
        s = t.surface
        if len(s) >= 2:
            out.add(s)
    return out


def main():
    engine = Engine()
    graph = Graph()
    db = sqlite3.connect(DB, uri=True)
    docs = engine.docs
    N = len(docs)
    print("corpus docs:", N, flush=True)

    hooksA = json.loads((ROOT / "hook_search/hooks_A.json").read_text(encoding="utf-8"))
    A5 = {(h["law_id"], h["art"]) for h in hooksA["A5"]}
    governing = [json.loads(l) for l in (ROOT / "prototype/c_full/governing.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
    freq = Counter()
    for r in governing:
        g = r.get("governing")
        if g:
            freq[(g["law_id"], g["art"])] += 1
    familiar = {a for a, c in freq.items() if c >= 2}
    universe = sorted(A5 | familiar)
    print("hook universe:", len(universe), "(A5", len(A5), "+ familiar", len(familiar), ")", flush=True)

    # --- 条文テキスト取得 (qnorm済 full) ---
    text_cache = {}

    def text_of(key):
        if key in text_cache:
            return text_cache[key]
        law, art = key
        idx = engine.by_key.get(key)
        if idx is not None:
            t = docs[idx]["full"]
        else:
            rows = db.execute(
                "SELECT body FROM chunks WHERE article_id=? ORDER BY rowid",
                (f"{law}_a{art}",)).fetchall()
            t = qnorm(" ".join(b or "" for (b,) in rows))
        text_cache[key] = t
        return t

    noun_cache = {}

    def nouns(key):
        if key not in noun_cache:
            noun_cache[key] = nouns_of(text_of(key))
        return noun_cache[key]

    term_vocab = list(dict.fromkeys(TERMS + list(SYN.values())))

    # --- フックごとに候補と cover を集める ---
    per_hook_cands = {}   # hook -> {word: cover}
    per_hook_meta = {}
    for i, h in enumerate(universe):
        ctx, _inc = graph.expand(*h)
        neigh = [k for k in ctx if k != h]
        n_in = sum(1 for k in neigh if k in engine.by_key)
        cand = Counter()
        for k in neigh:
            t = text_of(k)
            if not t:
                continue
            for w in nouns(k):
                cand[w] += 1
            for w in term_vocab:
                if w in t:
                    cand[w] += 1
        cand = {w: c for w, c in cand.items() if c >= MIN_COVER}
        per_hook_cands[h] = cand
        per_hook_meta[h] = {"n_neighbors": len(neigh), "n_neighbors_in_corpus": n_in,
                            "tiers": dict(Counter(str(ctx[k]) for k in neigh))}
        if (i + 1) % 50 == 0:
            print(f"hooks {i+1}/{len(universe)}", flush=True)

    # --- df をまとめて計算 (生き残った候補のみ) ---
    all_words = set()
    for cand in per_hook_cands.values():
        all_words.update(cand)
    print("distinct candidate words (cover>=2):", len(all_words), flush=True)
    df = {}
    fulls = [d["full"] for d in docs]
    for j, w in enumerate(sorted(all_words)):
        df[w] = sum(1 for f in fulls if w in f)
        if (j + 1) % 2000 == 0:
            print(f"df {j+1}/{len(all_words)}", flush=True)

    def idf_of(w):
        c = df[w]
        return math.log(1 + (N - c + 0.5) / (c + 0.5))

    term_set = set(term_vocab)
    result = {}
    n_empty = 0
    kw_counts = []
    for h in universe:
        own = text_of(h)
        scored = []
        for w, cov in per_hook_cands[h].items():
            if w in own:
                continue
            s = cov * idf_of(w)
            scored.append({"word": w, "cover": cov, "df": df[w],
                           "idf": round(idf_of(w), 4), "score": round(s, 4),
                           "source": "term" if w in term_set else "noun"})
        scored.sort(key=lambda r: (-r["score"], -r["cover"], r["word"]))
        top = scored[:TOPK]
        if not top:
            n_empty += 1
        kw_counts.append(len(top))
        result[f"{h[0]}|{h[1]}"] = {
            "law_id": h[0], "art": h[1],
            "in_A5": h in A5, "in_familiar_candidates": h in familiar,
            **per_hook_meta[h],
            "keywords": top,
            "block_text": " ".join(r["word"] for r in top),
        }
    (OUT / "child_keywords.json").write_text(
        json.dumps({"meta": {"rule": "cover>=2, score=cover*idf(engine式), own-text除外, top12",
                             "n_hooks": len(universe), "n_hooks_no_keywords": n_empty,
                             "avg_keywords": round(sum(kw_counts) / len(kw_counts), 2)},
                    "hooks": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done. hooks:", len(universe), "no-keyword hooks:", n_empty,
          "avg keywords:", round(sum(kw_counts) / len(kw_counts), 2), flush=True)


if __name__ == "__main__":
    main()
