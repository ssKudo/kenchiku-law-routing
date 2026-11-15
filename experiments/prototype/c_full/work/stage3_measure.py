#!/usr/bin/env python3
"""ステージ3: v0.3 規則で文脈を展開し、正答条の到達と順位を測る。
種 = 法アプリ v1.4 TF-IDF エンジンで問題文から選んだ上位3条 (正解は見ない)。
文脈 = 種(0) + 逆引き受け手(1) + 引用先(2) + 引用元・先頭40件(3)。
v0.3 との差: ノード集合を LIKE 前方一致でなく条境界の正確一致で取る
(a6 が a60-a69 を巻き込まない)。規則の意味は同一。"""
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
DB = "file:/home/hatch/.cache/jlawcite/jp_search.sqlite?mode=ro"
ACT, ORD, RULE = "325AC0000000201", "325CO0000000338", "325M50004000040"
PHRASE = re.compile(r"(政令|省令|国土交通省令|命令)で定める")
KAN = "〇一二三四五六七八九"

def kanji(n):
    n = int(n); out = ""
    for dig, unit in ((100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, dig)
        if d:
            out += (KAN[d] if d > 1 or unit == "" else "") + unit
    return out or "〇"

def art_kanji(a):
    return "の".join(kanji(p) for p in a.split("-")) + "条" if "-" in a else kanji(a) + "条"

def art_of(node):
    m = re.match(r".*?_a([0-9]+(?:-[0-9]+)*)", node)
    return m.group(1) if m else None

def law_of(node):
    return node.split("_")[0]

def artkey(art):
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in art.split("-"))

class Graph:
    def __init__(self):
        self.db = sqlite3.connect(DB, uri=True)
        self.names = dict(self.db.execute("SELECT law_id, title FROM laws"))
        self._exp_cache = {}
        self._chunk_cache = {}

    def _nodes_edges(self, aid, rel, direction):
        if direction == "out":
            return self.db.execute(
                "SELECT target FROM edges WHERE rel=? AND (source=? OR source LIKE ? ESCAPE '\\')",
                (rel, aid, aid + "\\_%")).fetchall()
        return self.db.execute(
            "SELECT source FROM edges WHERE rel=? AND (target=? OR target LIKE ? ESCAPE '\\')",
            (rel, aid, aid + "\\_%")).fetchall()

    def _own_ref(self, body, seed_law, seed_art):
        k = art_kanji(seed_art)
        if seed_law == ACT:
            pat = "法第" + k
            for m in re.finditer(re.escape(pat), body):
                before = body[:m.start()]
                if before.endswith("建築基準"):
                    return True
                if not before or not ("\u4e00" <= before[-1] <= "\u9fff"):
                    return True
            return ("建築基準法第" + k) in body
        if seed_law == ORD:
            pats = ["建築基準法施行令第" + k, "施行令第" + k, "令第" + k]
        elif seed_law == RULE:
            pats = ["建築基準法施行規則第" + k, "施行規則第" + k, "規則第" + k]
        else:
            pats = [self.names.get(seed_law, "") + "第" + k]
        return any(p in body for p in pats if p)

    def _chunks_with_phrase(self, law_id):
        if law_id not in self._chunk_cache:
            rows = self.db.execute(
                "SELECT article_id, body FROM chunks WHERE law_id=? AND body LIKE '%で定める%'",
                (law_id,)).fetchall()
            self._chunk_cache[law_id] = rows
        return self._chunk_cache[law_id]

    def expand(self, law_id, art):
        """種条文の文脈を { (law_id, art): tier } と引用元総数で返す。"""
        key = (law_id, art)
        if key in self._exp_cache:
            return self._exp_cache[key]
        aid = f"{law_id}_a{art}"
        ctx = {key: 0}
        # 逆引き受け手
        dlaws = {t for (t,) in self._nodes_edges(aid, "DELEGATES_TO", "out")}
        for dl in dlaws:
            for article_id, body in self._chunks_with_phrase(dl):
                if body and self._own_ref(body, law_id, art) and PHRASE.search(body):
                    a2 = art_of(article_id)
                    if a2:
                        ctx.setdefault((dl, a2), 1)
        # 引用先
        out = {}
        for (t,) in self._nodes_edges(aid, "CITES", "out"):
            l2, a2 = law_of(t), art_of(t)
            if a2:
                out[(l2, a2)] = True
        for k2 in out:
            ctx.setdefault(k2, 2)
        # 引用元 (v0.3 と同じ並び: 法令名, 第○条ラベル文字列) の先頭40
        inc = {}
        for (s,) in self._nodes_edges(aid, "CITES", "in"):
            l2, a2 = law_of(s), art_of(s)
            if a2:
                inc[(l2, a2)] = {"law": self.names.get(l2, l2), "article": f"第{a2}条"}
        ordered = sorted(inc.keys(), key=lambda k2: (inc[k2]["law"], inc[k2]["article"]))
        for k2 in ordered[:40]:
            ctx.setdefault(k2, 3)
        res = (ctx, len(inc))
        self._exp_cache[key] = res
        return res

def main():
    engine = Engine()
    graph = Graph()
    questions = { (r["year"], r["no"]): r for r in
                  (json.loads(l) for l in (OUT / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) }
    governing = { (r["year"], r["no"]): r for r in
                  (json.loads(l) for l in (OUT / "governing.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) }
    results = []
    for (year, no), q in sorted(questions.items()):
        g = governing.get((year, no), {})
        rec = {"year": year, "no": no}
        if q.get("answer_source") == "special_all_scored":
            rec["special"] = "all_scored"; results.append(rec); continue
        gov = g.get("governing")
        if not gov or q.get("status") not in ("ok", "ok_image_choices"):
            rec["measured"] = False
            rec["reason"] = "no_governing" if not gov else q.get("status")
            results.append(rec); continue
        qtext = q["stem"] + " " + " ".join(q["choices"])
        scores = engine.score_all(qtext)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        seeds = [k for k, _ in ranked[:3]]
        rec["seeds"] = [{"law_id": l, "art": a, "score": round(s, 2)} for (l, a), s in ranked[:3]]
        govkey = (gov["law_id"], gov["art"])
        rec["governing"] = gov
        rec["confidence"] = g.get("confidence")
        rec["seed_top1_hit"] = bool(seeds and seeds[0] == govkey)
        rec["seed_top3_hit"] = govkey in seeds
        # 文脈の和集合
        tier = {}
        capped = False
        for (l, a) in seeds:
            ctx, inc_total = graph.expand(l, a)
            if inc_total > 40:
                capped = True
            for k2, t in ctx.items():
                if k2 not in tier or t < tier[k2]:
                    tier[k2] = t
        rec["context_size"] = len(tier)
        rec["cited_by_capped"] = capped
        # ホップ距離順
        hop_order = sorted(tier.keys(), key=lambda k2: (tier[k2], k2[0], artkey(k2[1])))
        # TF-IDF 類似度順
        tf_order = sorted(tier.keys(), key=lambda k2: (-scores.get(k2, 0.0), k2[0], artkey(k2[1])))
        def rank_of(order, target):
            for i, k2 in enumerate(order):
                if k2 == target:
                    return i + 1
            return None
        def family_rank(order, target):
            base = target[1].split("-")[0]
            best = None
            for i, k2 in enumerate(order):
                if k2[0] == target[0] and k2[1].split("-")[0] == base:
                    best = i + 1; break
            return best
        rec["reached_strict"] = govkey in tier
        rec["governing_tier"] = tier.get(govkey)
        rec["rank_hop_strict"] = rank_of(hop_order, govkey)
        rec["rank_tfidf_strict"] = rank_of(tf_order, govkey)
        rec["rank_hop_family"] = family_rank(hop_order, govkey)
        rec["rank_tfidf_family"] = family_rank(tf_order, govkey)
        rec["reached_family"] = rec["rank_hop_family"] is not None
        results.append(rec)
        print(year, no, "gov", govkey, "strict", rec["reached_strict"],
              "hop", rec["rank_hop_strict"], "tfidf", rec["rank_tfidf_strict"],
              "ctx", rec["context_size"])
    (OUT / "results.json").write_text(json.dumps(
        {"meta": {"seeds_per_question": 3, "cited_by_cap": 40,
                  "note": "rank は文脈内順位 (1始まり)。到達しない場合は null。"},
         "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    meas = [r for r in results if r.get("reached_strict") is not None]
    print("measured:", len(meas),
          "strict reached:", sum(1 for r in meas if r["reached_strict"]),
          "family reached:", sum(1 for r in meas if r["reached_family"]))

if __name__ == "__main__":
    main()
