#!/usr/bin/env python3
"""指紋測定： 最良の土台（原文＋単語タグ＋メタデータ）に条文固有の「指紋」を足す。
- 土俵は fifty_measurement の50問・プール30。物差しは measure50.py の score_docs と同一方式。
- 土台は countermeasures の (a) 付加形 metafull_docs を同一作法で再構築し、
  baseline (results50 rank_original) と土台 (results_countermeasures meta_append) の
  両方を全問照合してから測る。
- 指紋は条文側のデータだけから機械的に作る:
    1. 数字の指紋: 本文から (値, 単位, 向き) の組を抽出。値は漢数字→算用数字・単位は正準形
       (㎡/m/日/…)・向きは 以上/以下/未満/超 に正規化し「13m超」「100㎡以上」の形のトークンにする。
    2. 例外の指紋: 原文を文に切り、「ただし」で始まる部分・「を除く」「適用しない」
       「準用しない」「適用除外」「この限りでない」を含む文を抜き出して付加する。
    3. 組み合わせの指紋: 条文の単語タグから2語組トークン「A×B」(語順は辞書順で正規化) を作り付加。
       質問側も、質問文に現れる語 (エンジン TERMS ∪ 単語タグ語彙) から同じ形式の2語組を生成して足し、
       一致が起きるようにする。
- 条件: fp_num (土台+数字) / fp_exc (土台+例外) / fp_combo (土台+組み合わせ) / fp_all (土台+3種全部)。
- 指紋ヒット診断: 各問について、正解条文の指紋が質問側と当たった件数と、
  ライバル (プール内他条文) の最大・平均を記録する。
"""
import itertools, json, math, re, statistics, sys, unicodedata
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
FP = RK / "fingerprint"
FM = RK / "fifty_measurement"
CM = RK / "countermeasures"
CFULL = RK / "prototype/c_full"
LAWAPP = Path("/home/hatch/workspace/lawapp")
sys.path.insert(0, str(CFULL / "work"))
import engine as E

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
cm_ref = json.load(open(CM / "results_countermeasures.json", encoding="utf-8"))["questions"]

tags = {}
for line in open(RK / "tag_vector/word_tags.jsonl", encoding="utf-8"):
    d = json.loads(line)
    tags[(d["law_id"], d["art"])] = d["tags"]

QIDS = list(pools.keys())
ALL_KEYS = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})


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


# ---------- 数値抽出（countermeasures / classic_channels と同一の単位表） ----------
UNITS = [("平方メートル", "㎡", 1.0), ("㎡", "㎡", 1.0), ("m²", "㎡", 1.0), ("m2", "㎡", 1.0),
         ("キロメートル", "m", 1000.0), ("センチメートル", "m", 0.01), ("ミリメートル", "m", 0.001),
         ("メートル", "m", 1.0), ("時間", "分", 60.0), ("週間", "日", 7.0), ("週", "日", 7.0),
         ("秒", "分", 1.0 / 60), ("分", "分", 1.0), ("日", "日", 1.0), ("月", "月", 1.0), ("年", "年", 1.0),
         ("階", "階", 1.0), ("人", "人", 1.0), ("台", "台", 1.0), ("本", "本", 1.0), ("室", "室", 1.0),
         ("戸", "戸", 1.0), ("棟", "棟", 1.0), ("％", "%", 1.0), ("パーセント", "%", 1.0), ("m", "m", 1.0)]
NUMSPAN = re.compile(r"([0-9０-９]+(?:\.[0-9０-９]+)?|[一二三四五六七八九十百千万]+)")
KD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
FW = str.maketrans("０１２３４５６７８９", "0123456789")
DIR_PATTERNS = [("を超えない", "以下"), ("を超える", "超"), ("以上", "以上"), ("以下", "以下"),
                ("未満", "未満"), ("以内", "以下"), ("超える", "超"), ("超", "超")]


def _seg_val(s):
    val, num = 0, 0
    for ch in s:
        if ch in KD:
            num = KD[ch]
        elif ch == "十":
            val += (num or 1) * 10; num = 0
        elif ch == "百":
            val += (num or 1) * 100; num = 0
        elif ch == "千":
            val += (num or 1) * 1000; num = 0
    return val + num


def _kanji_val(s):
    if "万" in s:
        head, rest = s.split("万", 1)
        return (_seg_val(head) if head else 1) * 10000 + _seg_val(rest)
    return _seg_val(s)


def extract_number_triples(text):
    """(値, 単位, 向き) の組を抽出。向きは 以上/以下/未満/超/'' に正規化。"""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for m in NUMSPAN.finditer(text):
        span = m.group(1)
        rest = text[m.end():m.end() + 10]
        hit, unit_len = None, 0
        for upat, canon, fac in UNITS:
            if rest.startswith(upat):
                hit, unit_len = (canon, fac), len(upat)
                break
        if not hit:
            continue
        if span[0] in "0123456789０１２３４５６７８９":
            v = float(span.translate(FW))
        else:
            v = float(_kanji_val(span))
        v = round(v * hit[1], 6)
        window = text[m.end() + unit_len: m.end() + unit_len + 7]
        direction = ""
        for i in range(len(window)):
            for pat, canon in DIR_PATTERNS:
                if window.startswith(pat, i):
                    direction = canon
                    break
            if direction:
                break
        out.append((v, hit[0], direction))
    return out


def triple_token(v, u, d):
    return f"{v:g}{u}{d}"


# ---------- 土台の再構築（countermeasures と同一作法） ----------
_lawidx_text = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
_law_index = json.loads(_lawidx_text[_lawidx_text.index("["):_lawidx_text.rindex("]") + 1])
chapter_map, raw_map = {}, {}
for m in _law_index:
    d = json.loads((LAWAPP / f"data/{m['id']}.json").read_text(encoding="utf-8"))
    for a in d["articles"]:
        k = (m["id"], a["key"].replace("_", "-"))
        chapter_map[k] = {"chapter": a.get("chapter") or "", "section": a.get("section") or ""}
        raw_map[k] = (a.get("caption") or "") + " " + " ".join(p["text"] for p in a["paras"])

PROC_WORDS = ["届出", "許可", "認可", "認定", "承認", "確認申請", "完了検査", "中間検査",
              "定期報告", "検査", "報告", "登録", "免許", "指定", "命令", "公告", "申請", "通知", "同意"]
ACTOR_WORDS = ["建築主", "建築主事", "特定行政庁", "都道府県知事", "市町村長", "国土交通大臣",
               "建築士", "設計者", "工事監理者", "工事施工者", "所有者", "管理者", "占有者",
               "指定確認検査機関", "消防長", "消防署長"]
ACT_WORDS = ["新築", "増築", "改築", "移転", "大規模の修繕", "大規模の模様替", "用途変更", "除却"]
USE_WORDS = ["共同住宅", "長屋", "寄宿舎", "劇場", "映画館", "集会場", "病院", "診療所", "学校",
             "保育所", "老人ホーム", "ホテル", "旅館", "飲食店", "物販店舗", "百貨店", "展示場",
             "事務所", "工場", "倉庫", "自動車車庫", "駐車場", "遊技場", "公衆浴場", "寺院",
             "教会", "幼稚園", "図書館", "博物館", "美術館", "体育館", "葬祭場", "火葬場"]
FINGERPRINT_WORDS = PROC_WORDS + ACTOR_WORDS + ACT_WORDS + USE_WORDS


def num_token_plain(v, u):
    return f"{v:g}{u}"


def build_docs(extra_fn=None):
    """extra_fn(key) -> 追加文字列 (qnorm 前)。None なら土台のみ。"""
    docs = {}
    for key in ALL_KEYS:
        base = eng.docs[eng.by_key[key]]
        full = base["full"]
        ch = chapter_map.get(key, {"chapter": "", "section": ""})
        finger = [w for w in FINGERPRINT_WORDS if E.qnorm(w) in full]
        nums = sorted(set(extract_number_triples(full)))
        meta_str = " ".join(x for x in [ch["chapter"], ch["section"]] if x) \
            + " " + " ".join(finger) + " " + " ".join(num_token_plain(v, u) for v, u, _ in nums)
        f = base["full"] + E.qnorm(" ".join(tags[key])) + E.qnorm(meta_str.strip())
        if extra_fn is not None:
            f = f + E.qnorm(extra_fn(key))
        g = E.gram_counts(f)
        docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": f,
                     "grams": g, "total": sum(g.values()), "cap_grams": base["cap_grams"]}
    return docs


base_docs = build_docs()

# ---------- baseline / 土台の再現照合 ----------
mm_base, mm_tudai = [], []
for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    if t not in keys:
        continue
    sc = score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys])
    r, _ = rank_from_scores(sc, keys, t)
    if r != ref[qid]["rank_original"]:
        mm_base.append((qid, r, ref[qid]["rank_original"]))
    sc2 = score_docs(qtext(qid), [base_docs[k] for k in keys])
    r2, _ = rank_from_scores(sc2, keys, t)
    if r2 != cm_ref[qid]["ranks"]["meta_append"]:
        mm_tudai.append((qid, r2, cm_ref[qid]["ranks"]["meta_append"]))
print(f"baseline照合: 不一致 {len(mm_base)} 件 {mm_base[:5]}")
print(f"土台(meta_append)照合: 不一致 {len(mm_tudai)} 件 {mm_tudai[:5]}")
assert not mm_base and not mm_tudai, "再現できない"

# ================= 指紋の構築 =================
# 1. 数字の指紋
num_fp = {}
for key in ALL_KEYS:
    triples = sorted(set(extract_number_triples(eng.docs[eng.by_key[key]]["full"])))
    num_fp[key] = {"triples": [[v, u, d] for v, u, d in triples],
                   "tokens": [triple_token(v, u, d) for v, u, d in triples]}

# 2. 例外の指紋
EXC_PATTERNS = ["を除く", "除く", "適用しない", "準用しない", "適用除外", "この限りでない", "対象としない"]
exc_fp = {}
for key in ALL_KEYS:
    sents = [s.strip() for s in raw_map[key].split("。") if s.strip()]
    out = []
    for s in sents:
        if "ただし" in s:
            out.append(s[s.index("ただし"):])
        elif any(p in s for p in EXC_PATTERNS):
            out.append(s)
    exc_fp[key] = out

# 3. 組み合わせの指紋
TAG_VOCAB = sorted({E.qnorm(w) for k in ALL_KEYS for w in tags[k] if len(E.qnorm(w)) >= 2})
TERM_SET = {E.qnorm(t) for t in E.TERMS if len(E.qnorm(t)) >= 2}


def pair_token(a, b):
    x, y = sorted([a, b])
    return f"{x}×{y}"


combo_fp = {}
for key in ALL_KEYS:
    words = sorted({E.qnorm(w) for w in tags[key] if len(E.qnorm(w)) >= 2})
    combo_fp[key] = [pair_token(a, b) for a, b in itertools.combinations(words, 2)]


def question_pairs(raw):
    extra = [c for a, c in E.SYN.items() if a in raw and c not in raw]
    qn = E.qnorm(raw + "".join(extra))
    detected = sorted({w for w in TERM_SET | set(TAG_VOCAB) if w in qn})
    return detected, [pair_token(a, b) for a, b in itertools.combinations(detected, 2)]


print(f"数字指紋: 平均 {statistics.mean(len(v['tokens']) for v in num_fp.values()):.1f} トークン/条文")
print(f"例外指紋: 平均 {statistics.mean(len(v) for v in exc_fp.values()):.1f} 件/条文 "
      f"(0件 {sum(1 for v in exc_fp.values() if not v)}/{len(exc_fp)})")
print(f"組手指紋: 平均 {statistics.mean(len(v) for v in combo_fp.values()):.0f} トークン/条文")

# 材料ファイル
materials = {}
for key in ALL_KEYS:
    materials[f"{key[0]}|{key[1]}"] = {
        "numeric_triples": num_fp[key]["triples"],
        "numeric_tokens": num_fp[key]["tokens"],
        "exceptions": exc_fp[key],
        "combo_pair_count": len(combo_fp[key]),
        "combo_pairs_sample": combo_fp[key][:30],
    }
json.dump(materials, open(FP / "fingerprints.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ================= 測定 =================
docs_num = build_docs(lambda k: " ".join(num_fp[k]["tokens"]))
docs_exc = build_docs(lambda k: " ".join(exc_fp[k]))
docs_combo = build_docs(lambda k: " ".join(combo_fp[k]))
docs_all = build_docs(lambda k: " ".join(num_fp[k]["tokens"]) + " " + " ".join(exc_fp[k])
                      + " " + " ".join(combo_fp[k]))
COND_DOCS = {"base": base_docs, "fp_num": docs_num, "fp_exc": docs_exc,
             "fp_combo": docs_combo, "fp_all": docs_all}

results = {"meta": {
    "n_questions": len(QIDS),
    "ruler": "TF-IDF (measure50 score_docs replication; engine idf/avg over full corpus)",
    "base": "article text + word tags (tag_vector) + metadata string (countermeasures (a) append form), "
            "reproduced and verified against results_countermeasures meta_append (31/3.38)",
    "fingerprints": {
        "numeric": "(value, unit, direction) triples from article text; value kanji->arabic, "
                   "unit canonical (㎡/m/日/...), direction in {以上,以下,未満,超,''}; tokens like '13m超'",
        "exception": "sentences cut from raw article text: from 'ただし' to sentence end, and sentences "
                     "containing を除く/適用しない/準用しない/適用除外/この限りでない/対象としない",
        "combo": "all 2-word pairs of the article's word tags, token 'A×B' (sorted); question side "
                 "generates the same-form pairs from words detected in the question (TERMS + tag vocabulary)",
    },
    "conditions": ["base", "fp_num", "fp_exc", "fp_combo", "fp_all"],
    "references": {"baseline_original": "29/3.88 (results50)"},
}, "questions": {}}

for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    g = gov[qid]["governing"]
    raw = qtext(qid)
    det_words, qpairs = question_pairs(raw)
    q_combo = raw + " " + " ".join(qpairs)
    rec = {"governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                         "confidence": gov[qid]["confidence"]},
           "in_pool": t in keys,
           "ranks": {"baseline": ref[qid]["rank_original"]},
           "question_detected_words": len(det_words),
           "question_pairs": len(qpairs)}
    results["questions"][qid] = rec
    if t not in keys:
        for c in COND_DOCS:
            rec["ranks"][c] = None
        rec["hits"] = None
        continue

    for cname, docs in COND_DOCS.items():
        qq = q_combo if cname in ("fp_combo", "fp_all") else raw
        sc = score_docs(qq, [docs[k] for k in keys])
        r, _ = rank_from_scores(sc, keys, t)
        rec["ranks"][cname] = r
        rec[f"governing_score_{cname}"] = round(sc[t], 2)

    # ---- 指紋ヒット診断 ----
    qtrip = {(v, u, d) for v, u, d in extract_number_triples(raw)}
    qtrip_vu = {(v, u) for v, u, d in qtrip}
    qpair_set = set(qpairs)
    qg = E.gram_counts(E.qnorm(raw))

    def rival_stats(vals):
        rv = [vals[k] for k in keys if k != t]
        return {"max": max(rv) if rv else 0, "mean": round(statistics.mean(rv), 2) if rv else 0}

    num_hits, num_hits_vu, exc_ratio, combo_hits = {}, {}, {}, {}
    for k in keys:
        tr = {(v, u, d) for v, u, d in (tuple(x) for x in num_fp[k]["triples"])}
        num_hits[k] = len(tr & qtrip)
        num_hits_vu[k] = len({(v, u) for v, u, d in tr} & qtrip_vu)
        eg = E.gram_counts(E.qnorm(" ".join(exc_fp[k])))
        exc_ratio[k] = round(sum(1 for gr in eg if gr in qg) / len(eg), 3) if eg else 0.0
        combo_hits[k] = len(set(combo_fp[k]) & qpair_set)
    rec["hits"] = {
        "numeric_exact": {"governing": num_hits[t], "governing_total": len(num_fp[t]["tokens"]),
                          **{f"rival_{s}": v for s, v in rival_stats(num_hits).items()}},
        "numeric_value_unit": {"governing": num_hits_vu[t],
                               **{f"rival_{s}": v for s, v in rival_stats(num_hits_vu).items()}},
        "exception_gram_overlap": {"governing": exc_ratio[t], "governing_clauses": len(exc_fp[t]),
                                   **{f"rival_{s}": v for s, v in rival_stats(exc_ratio).items()}},
        "combo": {"governing": combo_hits[t],
                  **{f"rival_{s}": v for s, v in rival_stats(combo_hits).items()}},
    }

json.dump(results, open(FP / "results_fingerprint.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)


def summarize(name, refname="base"):
    rs = {q: results["questions"][q]["ranks"].get(name) for q in QIDS}
    inp = [r for r in rs.values() if r is not None]
    top3 = sum(1 for r in inp if r <= 3)
    ent = sum(1 for q in QIDS if rs[q] is not None and rs[q] <= 3
              and (results["questions"][q]["ranks"][refname] or 99) > 3)
    ext = sum(1 for q in QIDS if rs[q] is not None and rs[q] > 3
              and (results["questions"][q]["ranks"][refname] or 99) <= 3)
    imp = sum(1 for q in QIDS if rs[q] is not None
              and rs[q] < (results["questions"][q]["ranks"][refname] or 99))
    wors = sum(1 for q in QIDS if rs[q] is not None
               and rs[q] > (results["questions"][q]["ranks"][refname] or 99))
    return {"top3": top3, "mean_rank": round(statistics.mean(inp), 2),
            "median_rank": statistics.median(inp),
            "entries_vs_base": ent, "exits_vs_base": ext, "improved": imp, "worsened": wors}


summary = {n: summarize(n) for n in COND_DOCS}
summary["baseline"] = summarize("baseline", "baseline")
results["summary"] = summary
json.dump(results, open(FP / "results_fingerprint.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== 結果 (top-3 / 平均 / 中央値 / 土台比 入・出 / 改善・悪化） ===")
print("baseline(原文のみ, 参考)", summary["baseline"])
for n in COND_DOCS:
    print(n, summary[n])

hits_in = [results["questions"][q]["hits"] for q in QIDS if results["questions"][q]["hits"]]
print("\n=== ヒット診断 (プール内42問の平均） ===")
for kind in ("numeric_exact", "numeric_value_unit", "combo"):
    print(kind, "正解", round(statistics.mean(h[kind]["governing"] for h in hits_in), 2),
          "/ ライバル最大", round(statistics.mean(h[kind]["rival_max"] for h in hits_in), 2),
          "/ ライバル平均", round(statistics.mean(h[kind]["rival_mean"] for h in hits_in), 2))
print("exception_gram_overlap 正解",
      round(statistics.mean(h["exception_gram_overlap"]["governing"] for h in hits_in), 3),
      "/ ライバル最大",
      round(statistics.mean(h["exception_gram_overlap"]["rival_max"] for h in hits_in), 3))
print("wrote results_fingerprint.json / fingerprints.json")
