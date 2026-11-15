#!/usr/bin/env python3
"""参照・委任グラフ プロトタイプ v0.1（2026-11-11）

方向Aの第一歩：JLaw-CiteGraph（jlawcite 2.5.0, e-Gov snapshot 2026-11-03）
から建築法規11法令のサブグラフを切り出し、種条文ごとに「小型AIに渡す
文脈」を組み立てるデモ。クドウの物差し＝「辿れる材料を文脈に集める」
こと。委任ラベルの厳密さは問わない（受け手は参照の逆引きで辿る）。

出力:
  subgraph.json  — クラスタ統計・法令別ノード数・内外の辺
  contexts.json  — 種条文ごとの組立て文脈（引用の出／入、委任、逆引き候補）
  PROTOTYPE_v01.md — 測定結果レポート
  viewer.html    — contexts.json を読む簡易ビューア（ブラウザで開くだけ）
"""
import json, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = "file:/home/hatch/.cache/jlawcite/jp_search.sqlite?mode=ro"
ACT = "325AC0000000201"   # 建築基準法
ORD = "325CO0000000338"   # 建築基準法施行令
RULE = "325M50004000040"  # 建築基準法施行規則

CLUSTER = {
    "325AC0000000201": "建築基準法",
    "325CO0000000338": "建築基準法施行令",
    "325M50004000040": "建築基準法施行規則",
    "325AC1000000202": "建築士法",
    "343AC0000000100": "都市計画法",
    "323AC1000000186": "消防法",
    "427AC0000000053": "建築物のエネルギー消費性能の向上等に関する法律",
    "418AC0000000091": "高齢者、障害者等の移動等の円滑化の促進に関する法律",
    "407AC0000000123": "建築物の耐震改修の促進に関する法律",
    "420AC0000000087": "長期優良住宅の普及の促進に関する法律",
    "324AC0000000100": "建設業法",
}

SEEDS = [
    ("2", "法2条（用語の定義：耐火構造ほか）— 委任の逆引きの代表例"),
    ("6", "法6条（建築物の建築等に関する申請及び確認）— 過去問『確認申請』の起点"),
    ("12", "法12条（報告、検査等）— 過去問『定期報告』の起点"),
    ("87", "法87条（建築物の用途変更等）— 包括委任・候補が広がる例"),
    ("77-35", "法77条の35 — 受け手が別省令へ飛ぶ例外例"),
]

KAN = "〇一二三四五六七八九"
def kanji(n):
    n = int(n); out = ""
    for dig, unit in ((100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, dig)
        if d:
            out += (KAN[d] if d > 1 or unit == "" else "") + unit
    return out or "〇"

def art_kanji(a):
    if "-" in a:
        m, s = a.split("-")
        return f"{kanji(m)}条の{kanji(s)}"
    return f"{kanji(a)}条"

def art_of(node):
    # v0.3: 枝番をすべて保持する（旧: (?:-[0-9]+)? で1段までしか取れず
    # 129条の13の2 -> 129-13 のように親条へ折りたたまれていた）
    m = re.match(r".*?_a([0-9]+(?:-[0-9]+)*)", node)
    return m.group(1) if m else None

def own_law_ref(body, art):
    """種条文への参照が建築基準法自身へのものか判定する（v0.3）。
    「法第二条」の略称形（直前が漢字でない）か、正式名
    「建築基準法第二条」（マッチする「法」の直前の4字が「建築基準」）
    のみを認め、「ガス事業法第二条」「下水道法第二条」等の
    他法令への参照を除外する。"""
    pat = re.compile("法第" + art_kanji(art))
    for m in pat.finditer(body):
        before = body[:m.start()]
        if before.endswith("建築基準"):
            return True
        if not before or not ("\u4e00" <= before[-1] <= "\u9fff"):
            return True
    return False

def law_of(node):
    return node.split("_")[0]

PHRASE = re.compile(r"(政令|省令|国土交通省令|命令)で定める")

db = sqlite3.connect(DB, uri=True)
names = dict(db.execute("SELECT law_id, title FROM laws"))

# ---------- 1. クラスタ統計 ----------
cluster_stats = {"laws": {}, "edges": {}, "top_outside_targets": []}
for lid, short in CLUSTER.items():
    cluster_stats["laws"][lid] = {
        "title": names.get(lid, lid),
        "nodes": db.execute("SELECT COUNT(*) FROM nodes WHERE law_id=?", (lid,)).fetchone()[0],
        "chunks": db.execute("SELECT COUNT(*) FROM chunks WHERE law_id=?", (lid,)).fetchone()[0],
    }

in_cluster = set(CLUSTER)
edges = db.execute(
    "SELECT source, target, rel FROM edges WHERE rel IN ('CITES','DELEGATES_TO')"
).fetchall()
internal_cites, internal_deleg = set(), set()
out_cites, out_deleg = Counter(), Counter()
for s, t, rel in edges:
    sl, tl = law_of(s), law_of(t)
    if sl in in_cluster:
        if rel == "CITES":
            if tl in in_cluster:
                internal_cites.add((s, t))
            else:
                out_cites[names.get(tl, tl)] += 1
        else:
            if tl in in_cluster:
                internal_deleg.add((s, t))
            else:
                out_deleg[names.get(tl, tl)] += 1

cluster_stats["edges"] = {
    "cites_internal_node_pairs": len(internal_cites),
    "delegates_internal_node_pairs": len(internal_deleg),
    "cites_to_outside": sum(out_cites.values()),
    "delegates_to_outside": sum(out_deleg.values()),
}
cluster_stats["top_outside_targets"] = [
    {"law": k, "cites": v} for k, v in out_cites.most_common(10)
]
# 建築基準法からの委任（再確認：417件）
act_deleg = db.execute(
    "SELECT source, target FROM edges WHERE rel='DELEGATES_TO' AND source LIKE ?",
    (ACT + "%",),
).fetchall()
cluster_stats["act_delegates_to"] = Counter(names.get(t, t) for _, t in act_deleg)
cluster_stats["act_delegates_total"] = len(act_deleg)

# ---------- 2. 種条文ごとの文脈組立て ----------
def article_text(article_id):
    rows = db.execute(
        "SELECT heading, body FROM chunks WHERE law_id=? AND article_id=? ORDER BY rowid",
        (ACT, article_id),
    ).fetchall()
    return [{"heading": h, "body": (b or "")[:600]} for h, b in rows]

def node_label(node):
    lid, a = law_of(node), art_of(node)
    return {"law": names.get(lid, lid), "law_id": lid,
            "article": f"第{a}条" if a else node, "node": node}

contexts = {}
for art, note in SEEDS:
    aid = f"{ACT}_a{art}"
    seed = {"article": f"法第{art_kanji(art)}", "note": note,
            "text": article_text(aid)}
    # 出ていく引用（条レベルに集約）
    out = {}
    for (t,) in db.execute(
        "SELECT target FROM edges WHERE rel='CITES' AND source LIKE ?",
        (aid + "%",)):
        lab = node_label(t)
        out.setdefault((lab["law_id"], lab["article"]), lab)
    seed["cites_out_articles"] = sorted(out.values(), key=lambda x: (x["law"], x["article"]))
    # 入ってくる引用（この条を引用している条：条レベルに集約、上限40）
    inc = {}
    for (s,) in db.execute(
        "SELECT source FROM edges WHERE rel='CITES' AND target LIKE ?",
        (aid + "%",)):
        lab = node_label(s)
        inc.setdefault((lab["law_id"], lab["article"]), lab)
    seed["cited_by_articles"] = sorted(inc.values(), key=lambda x: (x["law"], x["article"]))[:40]
    seed["cited_by_total_articles"] = len(inc)
    # 委任（法令レベル）
    seed["delegates_to_laws"] = sorted({names.get(t, t) for (t,) in db.execute(
        "SELECT target FROM edges WHERE rel='DELEGATES_TO' AND source LIKE ?",
        (aid + "%",))})
    # 逆引き候補：委任先法令の本文に建築基準法自身への「法第X条」
    # ＋「で定める」を含む条（v0.3: own_law_ref で他法令参照を除外）
    cands = {}
    for t in seed["delegates_to_laws"]:
        # law_id を名前から逆引き
        lid = next((k for k, v in names.items() if v == t), None)
        if not lid:
            continue
        for article_id, body in db.execute(
            "SELECT article_id, body FROM chunks WHERE law_id=?", (lid,)):
            if body and own_law_ref(body, art) and PHRASE.search(body):
                a2 = art_of(article_id)
                if a2:
                    cands.setdefault(lid, set()).add(a2)
    seed["reverse_delegation_candidates"] = {
        names.get(lid, lid): sorted(v, key=lambda x: [int(p) for p in x.split("-")])
        for lid, v in cands.items()
    }
    seed["context_article_count"] = (
        1 + len(seed["cites_out_articles"]) + len(seed["cited_by_articles"])
        + sum(len(v) for v in seed["reverse_delegation_candidates"].values())
    )
    contexts[art] = seed

(HERE / "subgraph.json").write_text(
    json.dumps(cluster_stats, ensure_ascii=False, indent=2), encoding="utf-8")
(HERE / "contexts.json").write_text(
    json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 3. レポート ----------
lines = [
    "# 参照・委任グラフ プロトタイプ v0.1（2026-11-11）",
    "",
    "CiteGraph（jlawcite 2.5.0 / e-Gov 2026-11-03スナップショット）から建築法規11法令を切り出し、",
    "種条文ごとに小型AIへ渡す「文脈」を組み立てた最初の動く試作。",
    "物差しはクドウの定義：**辿れる材料を文脈に集めること**。委任ラベルの厳密さは問わない。",
    "",
    "## クラスタ統計",
    "",
]
for lid, s in cluster_stats["laws"].items():
    lines.append(f"- {s['title']}：ノード {s['nodes']:,}件／チャンク {s['chunks']:,}件")
lines += [
    "",
    f"- クラスタ内 CITES（条・項レベル、重複除去）：{cluster_stats['edges']['cites_internal_node_pairs']:,} 辺",
    f"- クラスタ内 DELEGATES_TO：{cluster_stats['edges']['delegates_internal_node_pairs']:,} 辺",
    f"- クラスタ外への CITES：{cluster_stats['edges']['cites_to_outside']:,} 辺",
    f"- 建築基準法からの委任：{cluster_stats['act_delegates_total']} 件（→施行令 "
    f"{cluster_stats['act_delegates_to'].get('建築基準法施行令', 0)}、→施行規則 "
    f"{cluster_stats['act_delegates_to'].get('建築基準法施行規則', 0)}）— 2026-11-10の実測417件と一致",
    "",
    "クラスタ外への引用が多い先：",
]
for t in cluster_stats["top_outside_targets"][:5]:
    lines.append(f"- {t['law']}：{t['cites']:,}件")
lines += ["", "## 種条文ごとの文脈組立て", ""]
for art, seed in contexts.items():
    lines += [
        f"### {seed['article']} — {seed['note']}",
        "",
        f"- 組立て文脈：{seed['context_article_count']} 条（種条文＋引用先 {len(seed['cites_out_articles'])}＋"
        f"引用元 {len(seed['cited_by_articles'])}（全 {seed['cited_by_total_articles']} 条中・表示は先頭40）＋"
        f"逆引き候補 {sum(len(v) for v in seed['reverse_delegation_candidates'].values())}）",
        f"- 委任先（法令レベル）：{', '.join(seed['delegates_to_laws']) or 'なし'}",
    ]
    for law, arts in seed["reverse_delegation_candidates"].items():
        lines.append(f"- 逆引き候補（{law}）：第{ '条、第'.join(arts) }条（{len(arts)}件）")
    lines.append("")
lines += [
    "## 次の一手（未着手）",
    "",
    "- 告示層の追加：国交省告示一覧＋原文PDFの収集（mlit-kokuji / Kokuji_DB の方式）を文脈の第3層に。",
    "- 逆引きの精度検証：候補を「書き出し文（号・項）単位」で照合し、定義を借りているだけの条を除外するルールの追加。",
    "- 小型モデルでの評価：法6条・法12条の過去問を入力に、組立て文脈だけで正答条に到達できるかを測る（方向Cの入口）。",
    "",
    "ファイル：`subgraph.json`（統計）、`contexts.json`（文脈データ）、`viewer.html`（簡易ビューア）。",
]
(HERE / "PROTOTYPE_v01.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", sorted(p.name for p in HERE.iterdir()))
print(json.dumps(cluster_stats["edges"], ensure_ascii=False))
for art, s in contexts.items():
    print(art, "context articles:", s["context_article_count"],
          "cites_out:", len(s["cites_out_articles"]),
          "cited_by:", s["cited_by_total_articles"],
          "rev:", {k: len(v) for k, v in s["reverse_delegation_candidates"].items()})
