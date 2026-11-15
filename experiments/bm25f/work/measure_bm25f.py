#!/usr/bin/env python3
"""Step 1: BM25F（フィールド別重み付きスコア）の測定。
- 別AIレポート提案③。現行最良スタック（原文＋単語タグ＋メタデータ連結 = countermeasures
  対策2(a) meta_append, 31/3.38）を、フィールド別に分けて数え直す。
- 土俵は fifty_measurement の50問・プール30。測定前に baseline (29/3.88) と
  連結スタック meta_append (31/3.38) を全問再現してから測る。
- フィールド:
    body      = エンジンの条文 full（見出し込み原文）
    meta      = 章・節＋指紋語＋数値トークン（countermeasures と同一構築のメタデータ文字列）
    tag       = 単語タグ（tag_vector/word_tags.jsonl）
    exception = 例外節（fingerprint/fingerprints.json の exceptions、そのまま再利用）
- 変種:
    F-TFIDF: measure50 の score_docs と同じ式をフィールドごとに適用。
             長さ正規化はフィールドごとの平均長で実施。body は見出しブースト
             （cap_grams ×2.2、用語の cap 一致 +1.1）を含む（連結版と同一作法）。
    F-BM25 : 本家 BM25 をフィールドごとに適用。IDF はフィールド別に
             プール全463条文から log(1+(N-df+0.5)/(df+0.5)) で計算。
             トークンは文字2/3gram（エンジンと同一）＋ TERMS（出現回数を tf とする）。
             クエリ側は gram は distinct、TERMS は出現有無のみ。
             パラメータ: body k1=1.2 b=0.75 / meta k1=1.2 b=0.3（中心は b=0.0 も追加）
                         / tag k1=0.2 b=0.75 / exception k1=1.2 b=0.75
- 重み: body=1.0、meta∈{0.4,0.5,0.6}、tag∈{0.2,0.25,0.3}、exc∈{0.2,0.3,0.4} の
  27パターンを全て測る。診断用に body のみ / 中心・例外なし も測る。全パターン報告。
"""
import json, math, re, statistics, sys
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
B = RK / "bm25f"
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

fp_all = json.load(open(RK / "fingerprint/fingerprints.json", encoding="utf-8"))
exceptions = {tuple(k.split("|")): v["exceptions"] for k, v in fp_all.items()}

QIDS = list(pools.keys())
ALL_KEYS = sorted({(e["law"], e["art"]) for v in pools.values() for e in v})
NPOOL = len(ALL_KEYS)


def qtext(qid):
    q = qs[qid]
    return q["stem"] + " " + " ".join(q["choices"])


def pool_keys(qid):
    return [(e["law"], e["art"]) for e in pools[qid]]


def target_of(qid):
    g = gov[qid]["governing"]
    return (g["law_id"], g["art"])


def score_docs(raw, docs):
    """measure50.py の score_docs の複製（連結版の再現用）。"""
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


# ================= メタデータ構築（countermeasures と同一） =================
_lawidx_text = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
_law_index = json.loads(_lawidx_text[_lawidx_text.index("["):_lawidx_text.rindex("]") + 1])
chapter_map = {}
for m in _law_index:
    d = json.loads((LAWAPP / f"data/{m['id']}.json").read_text(encoding="utf-8"))
    for a in d["articles"]:
        chapter_map[(m["id"], a["key"].replace("_", "-"))] = {
            "chapter": a.get("chapter") or "", "section": a.get("section") or ""}

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
FINGERPRINT = PROC_WORDS + ACTOR_WORDS + ACT_WORDS + USE_WORDS

UNITS = [("平方メートル", "㎡", 1.0), ("㎡", "㎡", 1.0), ("m²", "㎡", 1.0), ("m2", "㎡", 1.0),
         ("キロメートル", "m", 1000.0), ("センチメートル", "m", 0.01), ("ミリメートル", "m", 0.001),
         ("メートル", "m", 1.0), ("時間", "分", 60.0), ("週間", "日", 7.0), ("週", "日", 7.0),
         ("秒", "分", 1.0 / 60), ("分", "分", 1.0), ("日", "日", 1.0), ("月", "月", 1.0), ("年", "年", 1.0),
         ("階", "階", 1.0), ("人", "人", 1.0), ("台", "台", 1.0), ("本", "本", 1.0), ("室", "室", 1.0),
         ("戸", "戸", 1.0), ("棟", "棟", 1.0), ("％", "%", 1.0), ("パーセント", "%", 1.0), ("m", "m", 1.0)]
NUMSPAN = re.compile(r"([0-9０-９]+(?:\.[0-9０-９]+)?|[一二三四五六七八九十百千万]+)")
KD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
FW = str.maketrans("０１２３４５６７８９", "0123456789")


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


def extract_numbers(text):
    out = []
    for m in NUMSPAN.finditer(text):
        span = m.group(1)
        rest = text[m.end():m.end() + 6]
        hit = None
        for upat, canon, fac in UNITS:
            if rest.startswith(upat):
                hit = (canon, fac)
                break
        if not hit:
            continue
        if span[0] in "0123456789０１２３４５６７８９":
            v = float(span.translate(FW))
        else:
            v = float(_kanji_val(span))
        out.append((round(v * hit[1], 6), hit[0]))
    return out


def num_token(v, u):
    return f"{v:g}{u}"


meta_string = {}
for key in ALL_KEYS:
    doc = eng.docs[eng.by_key[key]]
    full = doc["full"]
    ch = chapter_map.get(key, {"chapter": "", "section": ""})
    finger = [w for w in FINGERPRINT if E.qnorm(w) in full]
    tokens = [num_token(v, u) for v, u in sorted(set(extract_numbers(full)))]
    meta_string[key] = (" ".join(x for x in [ch["chapter"], ch["section"]] if x)
                        + " " + " ".join(finger) + " " + " ".join(tokens)).strip()

# ================= 検証ゲート: baseline と連結スタックの再現 =================
fulltags_docs, metafull_docs = {}, {}
for key in ALL_KEYS:
    base = eng.docs[eng.by_key[key]]
    ffull = base["full"] + E.qnorm(" ".join(tags[key]))
    fg = E.gram_counts(ffull)
    fulltags_docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": ffull,
                          "grams": fg, "total": sum(fg.values()), "cap_grams": base["cap_grams"]}
    mfull = ffull + E.qnorm(meta_string[key])
    mg = E.gram_counts(mfull)
    metafull_docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": mfull,
                          "grams": mg, "total": sum(mg.values()), "cap_grams": base["cap_grams"]}

mm_base, mm_stack = [], []
for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    if t not in keys:
        continue
    r, _ = rank_from_scores(score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys]), keys, t)
    if r != ref[qid]["rank_original"]:
        mm_base.append((qid, r, ref[qid]["rank_original"]))
    r2, _ = rank_from_scores(score_docs(qtext(qid), [metafull_docs[k] for k in keys]), keys, t)
    if r2 != cm_ref[qid]["ranks"]["meta_append"]:
        mm_stack.append((qid, r2, cm_ref[qid]["ranks"]["meta_append"]))
print(f"baseline照合: 不一致 {len(mm_base)} 件 {mm_base[:5]}")
print(f"連結スタック(meta_append)照合: 不一致 {len(mm_stack)} 件 {mm_stack[:5]}")
assert not mm_base and not mm_stack, "再現できない"

# ================= フィールド構築 =================
FIELDS = ["body", "meta", "tag", "exception"]
field_text = {f: {} for f in FIELDS}     # qnorm 済みテキスト
field_grams = {f: {} for f in FIELDS}
field_total = {f: {} for f in FIELDS}
for key in ALL_KEYS:
    base = eng.docs[eng.by_key[key]]
    texts = {"body": base["full"],
             "meta": E.qnorm(meta_string[key]),
             "tag": E.qnorm(" ".join(tags[key])),
             "exception": E.qnorm(" ".join(exceptions.get(key, [])))}
    for f in FIELDS:
        g = base["grams"] if f == "body" else E.gram_counts(texts[f])
        field_text[f][key] = texts[f]
        field_grams[f][key] = g
        field_total[f][key] = sum(g.values())

AVG_F = {f: statistics.mean(field_total[f].values()) for f in FIELDS}
print("フィールド平均長(gram数):", {f: round(AVG_F[f], 1) for f in FIELDS})
print("例外フィールド: 空の条文",
      sum(1 for k in ALL_KEYS if field_total["exception"][k] == 0), "/", NPOOL)

# BM25 用: フィールド別 IDF（gram と TERMS の両方）
bm25_idf_g, bm25_idf_t = {}, {}
for f in FIELDS:
    df = {}
    for key in ALL_KEYS:
        for g in field_grams[f][key]:
            df[g] = df.get(g, 0) + 1
    bm25_idf_g[f] = {g: math.log(1 + (NPOOL - c + 0.5) / (c + 0.5)) for g, c in df.items()}
    dft = {}
    for t in E.TERMS:
        nt = E.qnorm(t)
        c = sum(1 for key in ALL_KEYS if nt in field_text[f][key])
        if c:
            dft[t] = math.log(1 + (NPOOL - c + 0.5) / (c + 0.5))
    bm25_idf_t[f] = dft

BM25_PARAM = {"body": (1.2, 0.75), "meta": (1.2, 0.3), "tag": (0.2, 0.75),
              "exception": (1.2, 0.75)}


def query_parts(raw):
    extra = [c for a, c in E.SYN.items() if a in raw and c not in raw]
    qg = E.gram_counts(E.qnorm(raw + "".join(extra)))
    qterms = [t for t in E.TERMS if t in raw or t in extra]
    return qg, qterms


def tfidf_field_score(qg, qterms, f, key):
    """score_docs の式を1フィールドに適用。body は見出しブースト込み。"""
    grams = field_grams[f][key]
    sc = 0.0
    for g, c in qg.items():
        w = IDF.get(g)
        if w is None:
            continue
        tf = grams.get(g, 0)
        if tf:
            sc += w * min(c, 3) * (1 + math.log(tf))
    if f == "body":
        base = eng.docs[eng.by_key[key]]
        for g, c in qg.items():
            w = IDF.get(g)
            if w is None:
                continue
            ctf = base["cap_grams"].get(g, 0)
            if ctf:
                sc += w * 2.2 * (1 + math.log(ctf))
    text = field_text[f][key]
    for t in qterms:
        if t in text:
            sc += eng._term_w(t) * 1.6
            if f == "body" and t in eng.docs[eng.by_key[key]]["cap"]:
                sc += eng._term_w(t) * 1.1
    avg = AVG if f == "body" else AVG_F[f]
    return sc / (max(field_total[f][key], 1) / avg) ** 0.3


def bm25_field_score(qg, qterms, f, key, b_override=None):
    k1, b = BM25_PARAM[f]
    if b_override is not None:
        b = b_override
    dl = field_total[f][key]
    avgdl = AVG_F[f]
    if dl == 0:
        return 0.0
    denom_base = k1 * (1 - b + b * dl / avgdl)
    grams = field_grams[f][key]
    idfg = bm25_idf_g[f]
    sc = 0.0
    for g in qg:
        idf = idfg.get(g)
        if idf is None:
            continue
        tf = grams.get(g, 0)
        if tf:
            sc += idf * tf * (k1 + 1) / (tf + denom_base)
    text = field_text[f][key]
    idft = bm25_idf_t[f]
    for t in qterms:
        idf = idft.get(t)
        if idf is None:
            continue
        tf = text.count(E.qnorm(t))
        if tf:
            sc += idf * tf * (k1 + 1) / (tf + denom_base)
    return sc


# ================= フィールドスコアの事前計算 =================
# field_scores[variant][qid][field][key] = score （BM25 meta は b=0.3 と b=0.0 の2通り）
FS = {"tfidf": {}, "bm25": {}, "bm25_bmeta0": {}}
for qid in QIDS:
    keys = pool_keys(qid)
    qg, qterms = query_parts(qtext(qid))
    for variant in FS:
        FS[variant][qid] = {f: {} for f in FIELDS}
        for key in keys:
            FS[variant][qid]["body"][key] = (
                tfidf_field_score(qg, qterms, "body", key) if variant == "tfidf"
                else bm25_field_score(qg, qterms, "body", key))
            FS[variant][qid]["tag"][key] = (
                tfidf_field_score(qg, qterms, "tag", key) if variant == "tfidf"
                else bm25_field_score(qg, qterms, "tag", key))
            FS[variant][qid]["exception"][key] = (
                tfidf_field_score(qg, qterms, "exception", key) if variant == "tfidf"
                else bm25_field_score(qg, qterms, "exception", key))
            if variant == "tfidf":
                FS[variant][qid]["meta"][key] = tfidf_field_score(qg, qterms, "meta", key)
            elif variant == "bm25":
                FS[variant][qid]["meta"][key] = bm25_field_score(qg, qterms, "meta", key)
            else:
                FS[variant][qid]["meta"][key] = bm25_field_score(qg, qterms, "meta", key,
                                                                 b_override=0.0)
print("フィールドスコア計算完了")

# ================= 重みパターン測定 =================
PATTERNS = []
for wm in (0.4, 0.5, 0.6):
    for wt in (0.2, 0.25, 0.3):
        for we in (0.2, 0.3, 0.4):
            PATTERNS.append((wm, wt, we))
DIAG = {"body_only": (0.0, 0.0, 0.0), "center_noexc": (0.5, 0.25, 0.0)}
CENTER = (0.5, 0.25, 0.3)


def combine(variant, qid, w):
    wm, wt, we = w
    fs = FS[variant][qid]
    keys = pool_keys(qid)
    return {k: fs["body"][k] + wm * fs["meta"][k] + wt * fs["tag"][k] + we * fs["exception"][k]
            for k in keys}


def summarize(ranks, ref_ranks):
    inp = {q: r for q, r in ranks.items() if r is not None}
    top3 = sum(1 for r in inp.values() if r <= 3)
    ent = sum(1 for q, r in inp.items() if r <= 3 and (ref_ranks[q] or 99) > 3)
    ext = sum(1 for q, r in inp.items() if r > 3 and (ref_ranks[q] or 99) <= 3)
    imp = sum(1 for q, r in inp.items() if r < (ref_ranks[q] or 99))
    wors = sum(1 for q, r in inp.items() if r > (ref_ranks[q] or 99))
    return {"top3": top3, "mean_rank": round(statistics.mean(inp.values()), 2),
            "median_rank": statistics.median(inp.values()),
            "entries": ent, "exits": ext, "improved": imp, "worsened": wors}


base_ranks = {q: ref[q]["rank_original"] for q in QIDS}
stack_ranks = {q: cm_ref[q]["ranks"]["meta_append"] for q in QIDS}

results = {"meta": {
    "n_questions": len(QIDS),
    "task": "Step 1 (other-AI report proposal 3): BM25F field-weighted scoring vs concatenated stack",
    "fields": {"body": "article full text (engine full, caption included)",
               "meta": "chapter/section + fingerprint words + canonical numeric tokens "
                       "(countermeasures meta_string, identical construction)",
               "tag": "word tags (tag_vector/word_tags.jsonl)",
               "exception": "exception clauses (fingerprint/fingerprints.json exceptions, reused as-is)"},
    "variants": {
        "tfidf": "measure50 score_docs formula applied per field; per-field length "
                 "normalization (body uses engine avg_tot, others use pool-field means); "
                 "body includes caption boost (cap_grams x2.2, term-in-cap +1.1)",
        "bm25": "textbook BM25 per field; per-field IDF over the 463 pool articles "
                "log(1+(N-df+0.5)/(df+0.5)); tokens = char 2/3-grams + TERMS (occurrence tf); "
                "query grams distinct; params body k1=1.2 b=0.75, meta k1=1.2 b=0.3, "
                "tag k1=0.2 b=0.75, exception k1=1.2 b=0.75",
        "bm25_bmeta0": "same as bm25 but meta b=0.0 (proposal range 0.0-0.3 lower end)"},
    "weights": "body fixed 1.0; grid meta{0.4,0.5,0.6} x tag{0.2,0.25,0.3} x exc{0.2,0.3,0.4} "
               "= 27 patterns, all reported; diagnostics body_only and center_noexc",
    "center_pattern": {"meta": CENTER[0], "tag": CENTER[1], "exc": CENTER[2]},
    "verification": "baseline and concatenated stack (meta_append) reproduced for all "
                    "in-pool questions before measuring",
    "references": {"baseline": "29/3.88", "concat_stack": "31/3.38"},
    "field_avg_grams": {f: round(AVG_F[f], 1) for f in FIELDS},
}, "patterns": {}, "questions": {}}

for qid in QIDS:
    g = gov[qid]["governing"]
    results["questions"][qid] = {
        "governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                      "confidence": gov[qid]["confidence"]},
        "in_pool": target_of(qid) in pool_keys(qid),
        "ranks": {"baseline": base_ranks[qid], "concat_stack": stack_ranks[qid]}}

for variant in ("tfidf", "bm25", "bm25_bmeta0"):
    pats = list(PATTERNS)
    if variant != "bm25_bmeta0":
        pats += [DIAG["body_only"], DIAG["center_noexc"]]
    for w in pats:
        name = f"{variant}|m{w[0]}_t{w[1]}_e{w[2]}"
        ranks = {}
        for qid in QIDS:
            keys = pool_keys(qid)
            t = target_of(qid)
            if t not in keys:
                ranks[qid] = None
                continue
            sc = combine(variant, qid, w)
            r, _ = rank_from_scores(sc, keys, t)
            ranks[qid] = r
        s_stack = summarize(ranks, stack_ranks)
        s_base = summarize(ranks, base_ranks)
        results["patterns"][name] = {"weights": {"body": 1.0, "meta": w[0], "tag": w[1],
                                                 "exc": w[2]},
                                     "vs_stack": s_stack, "vs_baseline": s_base}
        if w == CENTER or w in DIAG.values():
            for qid in QIDS:
                results["questions"][qid]["ranks"][name] = ranks[qid]

json.dump(results, open(B / "results_bm25f.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== 参照 ===")
print("baseline", summarize(base_ranks, base_ranks))
print("concat_stack", summarize(stack_ranks, base_ranks))
print("\n=== パターン結果（vs 連結スタック: top3 / 平均 / 中央値 / 入・出 / 改善・悪化） ===")
for name, p in results["patterns"].items():
    s = p["vs_stack"]
    print(f"{name:28s} top3 {s['top3']:2d}  mean {s['mean_rank']:.2f}  med {s['median_rank']}  "
          f"入{s['entries']} 出{s['exits']}  改善{s['improved']} 悪化{s['worsened']}")
print("wrote results_bm25f.json")
