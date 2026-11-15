#!/usr/bin/env python3
"""Step 6: learning-to-rank 学習・検証。

- 特徴量: extract_features.py の features.json（10スコア＋hookA/hookB＋fam_freq_loo）。
  スコア特徴量は問題内で Min-Max 正規化し、学習 fold の統計で全特徴量を標準化する。
  fam_freq_loo は log1p 変換。
- モデル（torch、乱数種固定、ハイパラは事前固定で test 年に合わせた調整なし）:
  1. pointwise ロジスティック回帰（正例重み付き BCE、L2 λ=1.0）
  2. pairwise RankNet 型ロジスティック（同一問題内の 正解 vs 他条文 ペア）
- 検証: leave-one-year-out（10年、各年5問中プール内のみ評価。全年集計42問）。
- 診断: 全データ学習→同一データ評価（in-sample、天井診断として明示）、
  グループ別 ablation（drop-one / only-one、pairwise）。
"""
import json, math, statistics
from pathlib import Path
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
RK = Path("/home/hatch/workspace/research_kudou")
LTR = RK / "ltr"

data = json.load(open(LTR / "work/features.json", encoding="utf-8"))
SCORE_FEATS = data["features_raw"][:10]
ALL_FEATS = SCORE_FEATS + ["hookA", "hookB", "fam_freq_loo"]
GROUPS = {
    "lexical(言葉数え+フィールド)": ["lex_base", "lex_stack", "fld_body", "fld_meta", "fld_tag", "fld_exception"],
    "semantic(ruri/e5)": ["ruri", "e5"],
    "CE(bge/日本語)": ["bge_ce", "jp_ce"],
    "hook/おなじみ": ["hookA", "hookB", "fam_freq_loo"],
}

QS = {q: d for q, d in data["questions"].items() if d["in_pool"]}
YEARS = sorted({d["year"] for d in QS.values()})
print(f"対象: プール内 {len(QS)} 問、年 {YEARS}")


def build_xy(qids, feat_names):
    """特徴量行列（問題内 Min-Max → fold 標準化は呼び出し側）。生の変換済み値を返す。"""
    X, y, meta = [], [], []
    for qid in qids:
        rows = QS[qid]["rows"]
        raw = {f: np.array([r[f] for r in rows], dtype=np.float64) for f in feat_names}
        for f in feat_names:
            if f in SCORE_FEATS:
                v = raw[f]
                lo, hi = v.min(), v.max()
                raw[f] = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
            elif f == "fam_freq_loo":
                raw[f] = np.log1p(raw[f])
        for i, r in enumerate(rows):
            X.append([raw[f][i] for f in feat_names])
            y.append(r["label"])
            meta.append((qid, i))
    return np.array(X), np.array(y), meta


def train_model(Xtr, ytr, qtr, mode, l2=1.0, epochs=400, lr=0.05):
    """標準化して学習し、(重み, バイアス, 平均, 標準偏差) を返す。"""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    y = torch.tensor(ytr, dtype=torch.float32)
    # 問題ごとにインデックスを引けるようにする
    q_index = {}
    for i, (qid, _) in enumerate(qtr):
        q_index.setdefault(qid, []).append(i)
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    for ep in range(epochs):
        opt.zero_grad()
        s = Z @ w + b
        if mode == "pointwise":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s, y, pos_weight=torch.tensor(pos_w))
        else:  # pairwise: 正解 vs 同一問題の他条文
            losses = []
            for qid, idxs in q_index.items():
                pos = [i for i in idxs if ytr[i] == 1]
                if not pos:
                    continue
                sp = s[pos[0]]
                neg = torch.tensor([i for i in idxs if ytr[i] == 0])
                losses.append(torch.nn.functional.softplus(-(sp - s[neg])).mean())
            loss = torch.stack(losses).mean()
        loss = loss + l2 * (w * w).sum() / len(w)
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach()), mu, sd


def predict_ranks(model, qids, feat_names):
    w, b, mu, sd = model
    X, y, meta = build_xy(qids, feat_names)
    s = (X - mu) / sd @ w + b
    ranks = {}
    per_q_scores = {}
    for qid in qids:
        idxs = [i for i, (q, _) in enumerate(meta) if q == qid]
        order = sorted(idxs, key=lambda i: (-s[i], meta[i][1]))
        pos = [i for i in idxs if y[i] == 1][0]
        ranks[qid] = order.index(pos) + 1
        per_q_scores[qid] = {QS[qid]["rows"][meta[i][1]]["key"]: float(s[i]) for i in idxs}
    return ranks, per_q_scores


def agg(ranks):
    return {"top3": sum(1 for r in ranks.values() if r <= 3),
            "avg": round(statistics.mean(ranks.values()), 2),
            "n": len(ranks)}


def loyo(feat_names, mode):
    all_ranks, coefs = {}, []
    per_year = {}
    for yr in YEARS:
        tr = [q for q in QS if QS[q]["year"] != yr]
        te = [q for q in QS if QS[q]["year"] == yr]
        if not te:
            continue
        Xtr, ytr, mtr = build_xy(tr, feat_names)
        model = train_model(Xtr, ytr, mtr, mode)
        coefs.append(model[0])
        r, _ = predict_ranks(model, te, feat_names)
        all_ranks.update(r)
        per_year[yr] = agg(r)
    return all_ranks, per_year, np.mean(coefs, axis=0)


results = {"arena": "fifty 50問・原文TF-IDF上位30プール（プール内42問）",
           "references": {"baseline": {"top3": 29, "avg": 3.88},
                          "concat_stack": {"top3": 31, "avg": 3.38},
                          "stack_x_ruri_best": {"top3": 33, "avg": 3.26},
                          "stack_x_bgeCE_0.8/0.2": {"top3": 34, "avg": 3.14}},
           "features": ALL_FEATS, "models": {}}

for mode in ["pointwise", "pairwise"]:
    ranks, per_year, coef = loyo(ALL_FEATS, mode)
    results["models"][mode] = {
        "loyo": agg(ranks), "per_year": per_year, "ranks": ranks,
        "coef_mean_over_folds": {f: round(float(c), 3) for f, c in zip(ALL_FEATS, coef)}}
    print(mode, "LOYO:", agg(ranks))

# in-sample 診断（全42問で学習→同じ42問を評価。一般化の主張には使わない）
for mode in ["pointwise", "pairwise"]:
    X, y, m = build_xy(list(QS), ALL_FEATS)
    model = train_model(X, y, m, mode)
    r, _ = predict_ranks(model, list(QS), ALL_FEATS)
    results["models"][mode]["in_sample_diagnostic"] = agg(r)
    print(mode, "in-sample診断:", agg(r))

# ablation（pairwise）: グループを落とす / グループだけ
abl = {}
for gname, gfeats in GROUPS.items():
    rest = [f for f in ALL_FEATS if f not in gfeats]
    ranks, _, _ = loyo(rest, "pairwise")
    abl[f"drop {gname}"] = agg(ranks)
    ranks, _, _ = loyo(gfeats, "pairwise")
    abl[f"only {gname}"] = agg(ranks)
    print("ablation", gname, "drop:", abl[f"drop {gname}"], "only:", abl[f"only {gname}"])
results["ablation_pairwise"] = abl

json.dump(results, open(LTR / "results_ltr.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("results_ltr.json 出力完了")
