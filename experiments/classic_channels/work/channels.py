#!/usr/bin/env python3
"""チャンネル1〜5の測定。common.py の土俵（照合済み baseline）の上で測る。"""
import json, re, sys, hashlib
from pathlib import Path
import numpy as np
import ahocorasick
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).parent))
from common import (BASE, QIDS, base_scores, eng, pool_keys, pool_texts, qtext,
                    record, record_rrf, results, score_docs, summarize, target_of)

ALL_KEYS = sorted({k for qid in QIDS for k in pool_keys(qid)})


def raw_text(key):
    t = pool_texts[f"{key[0]}|{key[1]}"]
    return (t.get("caption") or "") + t.get("text", "")


# ================= チャンネル1: ルールベース =================
rules = json.load(open(BASE / "materials/rules.json", encoding="utf-8"))
FEATS = rules["features"]
W_KW, W_LAW = rules["meta"]["weights"]["weight_kw"], rules["meta"]["weights"]["weight_law"]
auto = ahocorasick.Automaton()
for fi, f in enumerate(FEATS):
    for tr in f["triggers"]:
        auto.add_word(tr, fi)
auto.make_automaton()

rule_cov = {"fired_questions": 0, "governing_boosted": 0, "fired_total": 0}
for qid in QIDS:
    qt = qtext(qid)
    fired = sorted({fi for _, fi in auto.iter(qt)})
    rule_cov["fired_total"] += len(fired)
    if fired:
        rule_cov["fired_questions"] += 1
    keys = pool_keys(qid)
    scores = {}
    for k in keys:
        txt = raw_text(k)
        sc = 0.0
        for fi in fired:
            f = FEATS[fi]
            if any(kw in txt for kw in f["boost_keywords"]):
                sc += W_KW
            if f["laws"] and k[0] in f["laws"]:
                sc += W_LAW
        scores[k] = sc
    t = target_of(qid)
    if t in keys and scores.get(t, 0) > 0:
        rule_cov["governing_boosted"] += 1
    results["questions"][qid]["details"]["rule_fired"] = [FEATS[fi]["name"] for fi in fired]
    record(qid, "rule", scores)
    record_rrf(qid, "rule_rrf", scores)
results["meta"]["rule_coverage"] = {
    "questions_with_any_rule": rule_cov["fired_questions"],
    "mean_fired_per_question": round(rule_cov["fired_total"] / len(QIDS), 2),
    "governing_boosted_in_pool": rule_cov["governing_boosted"],
}
print("ch1 rule:", summarize("rule"), "| rrf:", summarize("rule_rrf"), "| cov:", results["meta"]["rule_coverage"])

# ================= チャンネル2: シソーラス正規化 =================
thes = json.load(open(BASE / "materials/thesaurus.json", encoding="utf-8"))
PAIRS = dict(thes["seed_from_engine_syn"])
PAIRS.update(thes["synonyms_added"])
KAN = "一二三四五六七八九"
FW = str.maketrans("０１２３４５６７８９", "0123456789")


def int_to_kanji(n):
    if n == 0:
        return "零"
    out = ""
    for unit, u in ((1000, "千"), (100, "百"), (10, "十")):
        d, n = divmod(n, unit)
        if d:
            out += (KAN[d - 1] if d > 1 else "") + u
    if n:
        out += KAN[n - 1]
    return out


def normalize_numerals(text):
    text = text.translate(FW)
    text = re.sub(r"(?<=\d)[,，](?=\d)", "", text)

    def repl(m):
        s, e = m.start(), m.end()
        if s > 0 and text[s - 1] in "第.．":
            return m.group(0)
        if e < len(text) and text[e] in ".．":
            return m.group(0)
        n = int(m.group(0))
        if n >= 10000:
            return m.group(0)
        return int_to_kanji(n)

    return re.sub(r"\d+", repl, text)


def expand_syn(text):
    extra = [t for s, ts in PAIRS.items() if s in text for t in ts if t not in text]
    return text + "".join(extra)


thes_changed = 0
for qid in QIDS:
    raw = qtext(qid)
    q2 = normalize_numerals(expand_syn(raw))
    if q2 != raw:
        thes_changed += 1
    keys = pool_keys(qid)
    docs = [eng.docs[eng.by_key[k]] for k in keys]
    sc = score_docs(q2, docs)
    results["questions"][qid]["details"]["thesaurus_changed"] = (q2 != raw)
    record(qid, "thesaurus", sc)
    record_rrf(qid, "thesaurus_rrf", sc)
results["meta"]["thesaurus_changed_questions"] = thes_changed
print("ch2 thesaurus:", summarize("thesaurus"), "| rrf:", summarize("thesaurus_rrf"), "| changed:", thes_changed)

# ================= チャンネル3: 数値 =================
UNITS = [("平方メートル", "㎡", 1.0), ("㎡", "㎡", 1.0), ("m²", "㎡", 1.0), ("m2", "㎡", 1.0),
         ("キロメートル", "m", 1000.0), ("センチメートル", "m", 0.01), ("ミリメートル", "m", 0.001),
         ("メートル", "m", 1.0), ("時間", "分", 60.0), ("週間", "日", 7.0), ("週", "日", 7.0),
         ("秒", "分", 1.0 / 60), ("分", "分", 1.0), ("日", "日", 1.0), ("月", "月", 1.0), ("年", "年", 1.0),
         ("階", "階", 1.0), ("人", "人", 1.0), ("台", "台", 1.0), ("本", "本", 1.0), ("室", "室", 1.0),
         ("戸", "戸", 1.0), ("棟", "棟", 1.0), ("％", "%", 1.0), ("パーセント", "%", 1.0), ("m", "m", 1.0)]
NUMSPAN = re.compile(r"([0-9０-９]+(?:\.[0-9０-９]+)?|[一二三四五六七八九十百千万]+)")
KD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def seg_val(s):
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


def kanji_val(s):
    if "万" in s:
        head, rest = s.split("万", 1)
        return (seg_val(head) if head else 1) * 10000 + seg_val(rest)
    return seg_val(s)


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
            v = float(kanji_val(span))
        out.append((round(v * hit[1], 6), hit[0]))
    return out


def numeric_score(qnums, anums):
    by_unit = {}
    for v, u in anums:
        by_unit.setdefault(u, []).append(v)
    sc = 0.0
    for v, u in sorted(set(qnums)):
        cand = by_unit.get(u)
        if not cand:
            continue
        if any(abs(t - v) < 1e-9 for t in cand):
            sc += 3.0
        elif any(v / 1.5 <= t <= v * 1.5 for t in cand):
            sc += 1.0
        if any(t < v for t in cand) and any(t > v for t in cand):
            sc += 1.5
    return sc


art_numbers = {k: extract_numbers(raw_text(k)) for k in ALL_KEYS}
num_cov = {"questions_with_numbers": 0, "governing_scored": 0}
for qid in QIDS:
    qn = extract_numbers(qtext(qid))
    if qn:
        num_cov["questions_with_numbers"] += 1
    keys = pool_keys(qid)
    scores = {k: numeric_score(qn, art_numbers[k]) for k in keys}
    t = target_of(qid)
    if t in keys and scores.get(t, 0) > 0:
        num_cov["governing_scored"] += 1
    results["questions"][qid]["details"]["n_question_numbers"] = len(set(qn))
    record(qid, "numeric", scores)
    record_rrf(qid, "numeric_rrf", scores)
results["meta"]["numeric_coverage"] = num_cov
print("ch3 numeric:", summarize("numeric"), "| rrf:", summarize("numeric_rrf"), "| cov:", num_cov)

# ================= チャンネル4: 編集距離 =================
import engine as E2
WORD_RE = re.compile(r"[一-鿿ァ-ヶー]{2,}|[A-Za-z]{2,}")
art_words = {}
art_by_first = {}
for k in ALL_KEYS:
    doc = eng.docs[eng.by_key[k]]
    ws = {w for w in WORD_RE.findall(doc["full"]) if len(w) <= 10}
    art_words[k] = ws
    idx = {}
    for w in ws:
        idx.setdefault(w[0], []).append(w)
    art_by_first[k] = idx

edit_matches_total = 0
for qid in QIDS:
    raw = qtext(qid)
    qws = {w for w in WORD_RE.findall(E2.qnorm(raw)) if 3 <= len(w) <= 8 or w in E2.TERMS}
    keys = pool_keys(qid)
    scores = {}
    for k in keys:
        doc = eng.docs[eng.by_key[k]]
        base = base_scores[qid][k]
        bonus_ws = []
        for w in qws:
            if w in doc["full"]:
                continue
            thr = 1 if len(w) <= 4 else 2
            found = False
            for cand in art_by_first[k].get(w[0], ()):
                if abs(len(cand) - len(w)) > thr:
                    continue
                if Levenshtein.distance(w, cand, score_cutoff=thr) <= thr:
                    found = True
                    break
            if found:
                bonus_ws.append(w)
        bonus = sum(sorted((0.5 * eng._term_w(w) for w in bonus_ws), reverse=True)[:3])
        edit_matches_total += len(bonus_ws)
        scores[k] = base + bonus
    results["questions"][qid]["details"]["edit_qwords"] = len(qws)
    record(qid, "editdist", scores)
    record_rrf(qid, "editdist_rrf", scores)
results["meta"]["editdist_weak_matches"] = edit_matches_total
print("ch4 editdist:", summarize("editdist"), "| rrf:", summarize("editdist_rrf"), "| weak matches:", edit_matches_total)

# ================= チャンネル5: MinHash / SimHash =================
def grams3(s):
    s = E2.qnorm(s)
    return [s[i:i + 3] for i in range(len(s) - 2)]


HASH = {}
def gram_hash(g):
    h = HASH.get(g)
    if h is None:
        h = int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "little")
        HASH[g] = h
    return h


rng = np.random.RandomState(42)
NP = 128
A = rng.randint(1, 2**63, size=NP, dtype=np.uint64)
B = rng.randint(0, 2**63, size=NP, dtype=np.uint64)


def minhash_sig(gramset):
    if not gramset:
        return np.full(NP, np.iinfo(np.uint64).max, dtype=np.uint64)
    H = np.array([gram_hash(g) for g in gramset], dtype=np.uint64)
    return ((H[:, None] * A[None, :] + B[None, :]).min(axis=0))


def simhash_fp(gram_counts):
    acc = np.zeros(64, dtype=np.int64)
    for g, c in gram_counts.items():
        h = gram_hash(g)
        bits = np.array([(h >> i) & 1 for i in range(64)], dtype=np.int64)
        acc += np.where(bits == 1, c, -c)
    fp = 0
    for i in range(64):
        if acc[i] > 0:
            fp |= (1 << i)
    return fp


art_mh, art_sh = {}, {}
for k in ALL_KEYS:
    gl = grams3(raw_text(k))
    art_mh[k] = minhash_sig(set(gl))
    from collections import Counter
    art_sh[k] = simhash_fp(Counter(gl))

for qid in QIDS:
    gl = grams3(qtext(qid))
    from collections import Counter
    qmh = minhash_sig(set(gl))
    qsh = simhash_fp(Counter(gl))
    keys = pool_keys(qid)
    mh_scores = {k: float(np.mean(qmh == art_mh[k])) for k in keys}
    sh_scores = {k: 1.0 - ((qsh ^ art_sh[k]).bit_count() / 64.0) for k in keys}
    record(qid, "minhash", mh_scores)
    record_rrf(qid, "minhash_rrf", mh_scores)
    record(qid, "simhash", sh_scores)
    record_rrf(qid, "simhash_rrf", sh_scores)
print("ch5 minhash:", summarize("minhash"), "| rrf:", summarize("minhash_rrf"))
print("ch5 simhash:", summarize("simhash"), "| rrf:", summarize("simhash_rrf"))

# ================= 保存 =================
names = ["baseline", "rule", "rule_rrf", "thesaurus", "thesaurus_rrf", "numeric", "numeric_rrf",
         "editdist", "editdist_rrf", "minhash", "minhash_rrf", "simhash", "simhash_rrf"]
results["meta"]["summaries"] = {n: summarize(n) for n in names}
results["meta"]["method"] = (
    "土俵: fifty_measurement 50問・プール30条文・原文のみ。baseline は measure50 の score_docs を複製し "
    "results50.json の rank_original と42問全照合一致を確認。順位は (-score, プール順)。"
    "合わせ技は RRF k=60。プール外8問は全条件 rank=null、top-3 は分母50・平均順位はプール内42問。")
json.dump(results, open(BASE / "results_classic.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote results_classic.json")
for n in names:
    print(n, results["meta"]["summaries"][n])
