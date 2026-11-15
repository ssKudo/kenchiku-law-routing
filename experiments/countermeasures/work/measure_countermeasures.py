#!/usr/bin/env python3
"""対策1（PRF：質問を条文言葉に寄せる）・対策2（メタデータ）の測定。
- 土俵は fifty_measurement の50問・プール30。物差しは measure50.py の score_docs と同一方式。
- baseline は results50.json の rank_original と全問照合してから測る。
- 対策1 PRF: 言葉数えで一度引き、プール内 top-3 条文から特徴語を抽出して質問に「追加だけ」し再検索（1ラウンド）。
    候補語彙 V = エンジン TERMS ∪ 単語タグ語彙（どちらも条文側の語彙）。
    (a) prf_text: 候補を top-3 条文の「本文」に現れる語に限定。重み = _term_w(w) × (1+log(本文出現回数合計))。
    (b) prf_tag : 候補を top-3 条文の「単語タグ」に現れる語に限定。重み = _term_w(w) × (タグに含む top-3 条文数)。
    どちらも質問に既出の語は除き、上位20語まで追加。追加語は問題ごとに記録。
- 対策2 メタデータ: 条文側データだけから作る文字列
    = 章・節（法令集アプリ data の chapter/section）＋ 本文に現れる対象・手続・行為・用途語（固定語彙の出現ベース）
      ＋ 数値の境目の正規形トークン（classic_channels の extract_numbers を複製し「100㎡」等の形に文字列化）。
    (a) meta_append: 条文 = 原文＋単語タグ＋メタデータ文字列（tag_counting 条件4 = 原文＋単語タグ と比較）。
    (b) meta_rerank: baseline top-10 内で、質問特徴（classic_channels rules.json の42特徴トリガー＋数値）と
        条文側マッチテキスト（章・節・見出し・指紋語・数値トークン・単語タグ）の一致数で並べ替え（同数は baseline スコア順）。
        top-10 の外は baseline 順のまま後ろに置く。
"""
import json, math, re, sys, statistics
from pathlib import Path

RK = Path("/home/hatch/workspace/research_kudou")
CM = RK / "countermeasures"
FM = RK / "fifty_measurement"
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
tc_ref = json.load(open(RK / "tag_counting/results_tag_counting.json", encoding="utf-8"))["questions"]

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


# ---------- baseline 照合 & ベーススコア ----------
base_scores = {}
mismatch = []
for qid in QIDS:
    keys = pool_keys(qid)
    sc = score_docs(qtext(qid), [eng.docs[eng.by_key[k]] for k in keys])
    base_scores[qid] = sc
    t = target_of(qid)
    if t in keys:
        r, _ = rank_from_scores(sc, keys, t)
        if r != ref[qid]["rank_original"]:
            mismatch.append((qid, r, ref[qid]["rank_original"]))
print(f"baseline照合: 不一致 {len(mismatch)} 件 {mismatch[:5]}")
assert not mismatch, "baseline が results50 と一致しない"

# ---------- ドキュメント変形: 原文＋単語タグ（tag_counting と同一作法） ----------
def tag_text(key):
    return E.qnorm(" ".join(tags[key]))


fulltags_docs = {}
for key in ALL_KEYS:
    base = eng.docs[eng.by_key[key]]
    ffull = base["full"] + tag_text(key)
    fg = E.gram_counts(ffull)
    fulltags_docs[key] = {"law": key[0], "art": key[1], "cap": base["cap"], "full": ffull,
                          "grams": fg, "total": sum(fg.values()), "cap_grams": base["cap_grams"]}

# 条件4（素の質問×全文＋タグ）の再現チェック
mm4 = []
for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    if t in keys:
        sc = score_docs(qtext(qid), [fulltags_docs[k] for k in keys])
        r, _ = rank_from_scores(sc, keys, t)
        if r != tc_ref[qid]["ranks"]["raw_fulltags"]:
            mm4.append((qid, r, tc_ref[qid]["ranks"]["raw_fulltags"]))
print(f"条件4再現照合: 不一致 {len(mm4)} 件 {mm4[:5]}")
assert not mm4, "tag_counting 条件4 が再現できない"


# ================= 対策1: PRF（疑似適合フィードバック） =================
VOCAB = sorted({E.qnorm(w) for w in E.TERMS} |
               {E.qnorm(w) for k in ALL_KEYS for w in tags[k]})
VOCAB = [w for w in VOCAB if len(w) >= 2]
PRF_K = 20


def prf_words(qid, mode):
    """mode='text': top-3 条文の本文に現れる候補語 / mode='tag': top-3 条文のタグに現れる候補語。
    戻り値: (追加語リスト, top3キー)"""
    keys = pool_keys(qid)
    sc = base_scores[qid]
    top3 = sorted(keys, key=lambda k: (-sc.get(k, 0.0), keys.index(k)))[:3]
    raw_q = E.qnorm(qtext(qid))
    cand = {}
    if mode == "text":
        for k in top3:
            full = eng.docs[eng.by_key[k]]["full"]
            for w in VOCAB:
                c = full.count(w)
                if c:
                    cand[w] = cand.get(w, 0.0) + eng._term_w(w) * (1 + math.log(c))
    else:
        for k in top3:
            for w in {E.qnorm(t) for t in tags[k]}:
                if len(w) >= 2:
                    cand[w] = cand.get(w, 0.0) + eng._term_w(w)
    ranked = sorted(((w, s) for w, s in cand.items() if w not in raw_q),
                    key=lambda ws: (-ws[1], ws[0]))
    return [w for w, _ in ranked[:PRF_K]], top3


prf_added = {}
for qid in QIDS:
    a_text, top3 = prf_words(qid, "text")
    a_tag, _ = prf_words(qid, "tag")
    prf_added[qid] = {"text": a_text, "tag": a_tag,
                      "top3_keys": [[k[0], k[1]] for k in top3]}
print("PRF追加語数: text 平均", round(statistics.mean(len(v["text"]) for v in prf_added.values()), 1),
      "/ tag 平均", round(statistics.mean(len(v["tag"]) for v in prf_added.values()), 1))


# ================= 対策2: メタデータ構築 =================
# 章・節（法令集アプリ data から。engine の art 形式に合わせる）
_lawidx_text = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
_law_index = json.loads(_lawidx_text[_lawidx_text.index("["):_lawidx_text.rindex("]") + 1])
chapter_map = {}
for m in _law_index:
    d = json.loads((LAWAPP / f"data/{m['id']}.json").read_text(encoding="utf-8"))
    for a in d["articles"]:
        chapter_map[(m["id"], a["key"].replace("_", "-"))] = {
            "chapter": a.get("chapter") or "", "section": a.get("section") or ""}

# 指紋語彙（本文に現れればメタデータに載せる。条文側の出現ベース）
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

# 数値抽出（classic_channels/work/channels.py の extract_numbers を複製）
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


# 条文ごとのメタデータ
meta_info = {}
for key in ALL_KEYS:
    doc = eng.docs[eng.by_key[key]]
    full = doc["full"]
    ch = chapter_map.get(key, {"chapter": "", "section": ""})
    finger = [w for w in FINGERPRINT if E.qnorm(w) in full]
    nums = sorted(set(extract_numbers(full)))
    tokens = [num_token(v, u) for v, u in nums]
    meta_str = " ".join(x for x in [ch["chapter"], ch["section"]] if x) \
        + " " + " ".join(finger) + " " + " ".join(tokens)
    match_text = E.qnorm(" ".join(x for x in [ch["chapter"], ch["section"], doc["cap"]] if x)
                          + " " + " ".join(finger) + " " + " ".join(tokens)
                          + " " + " ".join(tags[key]))
    meta_info[key] = {"chapter": ch["chapter"], "section": ch["section"],
                      "fingerprint": finger, "numbers": [[v, u] for v, u in nums],
                      "meta_string": meta_str.strip(), "match_text": match_text}

meta_lens = [len(v["meta_string"]) for v in meta_info.values()]
print(f"メタデータ: 平均{statistics.mean(meta_lens):.0f}字 / 章あり "
      f"{sum(1 for v in meta_info.values() if v['chapter'])}/{len(meta_info)} / 節あり "
      f"{sum(1 for v in meta_info.values() if v['section'])}/{len(meta_info)} / 指紋語 平均"
      f"{statistics.mean(len(v['fingerprint']) for v in meta_info.values()):.1f}語 / 数値トークン 平均"
      f"{statistics.mean(len(v['numbers']) for v in meta_info.values()):.1f}個")

# (a) 付加形ドキュメント: 原文＋単語タグ＋メタデータ
metafull_docs = {}
for key in ALL_KEYS:
    ft = fulltags_docs[key]
    mfull = ft["full"] + E.qnorm(meta_info[key]["meta_string"])
    mg = E.gram_counts(mfull)
    metafull_docs[key] = {"law": key[0], "art": key[1], "cap": ft["cap"], "full": mfull,
                          "grams": mg, "total": sum(mg.values()), "cap_grams": ft["cap_grams"]}

# (b) 二段形: 質問特徴（rules 42特徴のトリガー＋数値）
rules = json.load(open(RK / "classic_channels/materials/rules.json", encoding="utf-8"))
FEATS = rules["features"]


def question_features(qid):
    qt = qtext(qid)
    fired = [f for f in FEATS if any(tr in qt for tr in f["triggers"])]
    return fired, extract_numbers(qt)


def meta_matches(key, fired, qnums):
    mt = meta_info[key]["match_text"]
    n = 0
    hit_names = []
    for f in fired:
        if any(E.qnorm(kw) in mt for kw in f["boost_keywords"]):
            n += 1
            hit_names.append(f["name"])
    if qnums:
        anums = meta_info[key]["numbers"]
        by_unit = {}
        for v, u in anums:
            by_unit.setdefault(u, []).append(v)
        num_hit = False
        for v, u in sorted(set(qnums)):
            cand = by_unit.get(u)
            if not cand:
                continue
            if any(abs(t - v) < 1e-9 for t in cand) or \
               (any(t < v for t in cand) and any(t > v for t in cand)):
                num_hit = True
                break
        if num_hit:
            n += 1
            hit_names.append("数値")
    return n, hit_names


# ================= 測定 =================
results = {"meta": {
    "n_questions": len(QIDS),
    "ruler": "TF-IDF (measure50 score_docs replication; engine idf/avg over full corpus)",
    "prf": f"1 round; top-3 pool articles as feedback; up to {PRF_K} added words; additions only",
    "prf_vocab": "engine TERMS + word-tag vocabulary (article-side words only)",
    "metadata": "chapter/section (lawapp data) + fingerprint words present in article text "
                "(procedure/actor/act/use lists) + canonical numeric tokens (extract_numbers)",
    "conditions": ["prf_text_orig", "prf_tag_orig", "prf_tag_fulltags",
                   "meta_append", "meta_rerank"],
    "references": {"baseline": "29/3.88 (results50)", "raw_fulltags": "31/3.48 (tag_counting cond.4)",
                   "safe_exp_fulltags": "31/3.74 (tag_counting cond.2)"},
}, "questions": {}}

for qid in QIDS:
    keys = pool_keys(qid)
    t = target_of(qid)
    g = gov[qid]["governing"]
    raw = qtext(qid)
    rec = {"governing": {"law_id": g["law_id"], "law_name": g["law_name"], "art": g["art"],
                         "confidence": gov[qid]["confidence"]},
           "in_pool": t in keys,
           "prf_added_text": prf_added[qid]["text"],
           "prf_added_tag": prf_added[qid]["tag"],
           "prf_top3": prf_added[qid]["top3_keys"],
           "ranks": {"baseline": ref[qid]["rank_original"],
                     "raw_fulltags": tc_ref[qid]["ranks"]["raw_fulltags"],
                     "safe_exp_fulltags": tc_ref[qid]["ranks"]["safe_exp_fulltags"]}}
    results["questions"][qid] = rec
    if t not in keys:
        for c in results["meta"]["conditions"]:
            rec["ranks"][c] = None
        continue

    # 対策1 (a)(b) と組み合わせ
    q_prf_text = raw + " " + " ".join(prf_added[qid]["text"])
    q_prf_tag = raw + " " + " ".join(prf_added[qid]["tag"])
    orig_docs = [eng.docs[eng.by_key[k]] for k in keys]
    ft_docs = [fulltags_docs[k] for k in keys]
    for cname, qq, docs in (("prf_text_orig", q_prf_text, orig_docs),
                            ("prf_tag_orig", q_prf_tag, orig_docs),
                            ("prf_tag_fulltags", q_prf_tag, ft_docs)):
        sc = score_docs(qq, docs)
        r, _ = rank_from_scores(sc, keys, t)
        rec["ranks"][cname] = r
        rec[f"governing_score_{cname}"] = round(sc[t], 2)

    # 対策2 (a) 付加形
    sc = score_docs(raw, [metafull_docs[k] for k in keys])
    r, _ = rank_from_scores(sc, keys, t)
    rec["ranks"]["meta_append"] = r
    rec["governing_score_meta_append"] = round(sc[t], 2)

    # 対策2 (b) 二段形: baseline top-10 をメタデータ一致数で並べ替え
    fired, qnums = question_features(qid)
    base_order = sorted(keys, key=lambda k: (-base_scores[qid].get(k, 0.0), keys.index(k)))
    top10, rest = base_order[:10], base_order[10:]
    mcount = {k: meta_matches(k, fired, qnums) for k in top10}
    reranked = sorted(top10, key=lambda k: (-mcount[k][0], -base_scores[qid].get(k, 0.0),
                                            keys.index(k)))
    full_order = reranked + rest
    rec["ranks"]["meta_rerank"] = full_order.index(t) + 1
    rec["rerank_diag"] = {
        "n_features_fired": len(fired),
        "fired": [f["name"] for f in fired],
        "governing_in_base_top10": t in top10,
        "governing_matches": mcount[t][0] if t in top10 else None,
        "governing_match_names": mcount[t][1] if t in top10 else None,
        "top1_key": [reranked[0][0], reranked[0][1]],
        "top1_matches": mcount[reranked[0]][0],
    }

json.dump(results, open(CM / "results_countermeasures.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)


def summarize(name, refname="baseline"):
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
            "entries_vs_baseline": ent, "exits_vs_baseline": ext,
            "improved": imp, "worsened": wors}


print("\n=== 結果 (top-3 / 平均 / 中央値 / baseline比 入・出 / 改善・悪化） ===")
for n in ("baseline", "raw_fulltags", "safe_exp_fulltags"):
    print(n, summarize(n))
for n in results["meta"]["conditions"]:
    print(n, summarize(n))
gov10 = sum(1 for q in QIDS if results["questions"][q].get("rerank_diag", {}).get("governing_in_base_top10"))
print(f"二段形： 正解が baseline top-10 内 {gov10}/42問")
print("wrote results_countermeasures.json")
