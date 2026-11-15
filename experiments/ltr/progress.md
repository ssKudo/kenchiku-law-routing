# Step 6: learning-to-rank 進捗

- 2026-11-15 開始。土俵: fifty 50問・原文TF-IDF上位30プール（プール内42問）。
- 資産確認済み:
  - ペア別スコア: rerank_blend/work/lex_scores.json（base/stack/ruri/bge の30件/問）、
    rerank_blend/work/scores_jp_partial.json（日本語CE）、rerank_ce/work/scores_partial.json（bge CE、照合用）。
  - フィールド別スコア: bm25f/work/measure_bm25f.py の F-TFIDF 構築を再利用して再計算する。
  - e5: vector_hybrid/work/doc_emb_fifty_a.npy（463条文・原文）＋ embedder.py で質問埋め込みを再計算。
  - フック: hook_search/hooks_A.json（A5集合）、hooks_B_per_question.json（問題別B集合）。
  - おなじみ頻度: hook_search/familiar_freq.json（299問の延べ頻度）から LOO で引く。
  - 学習器: sklearn なし → torch 2.15 CPU でロジスティック回帰（pointwise / pairwise）を実装。
- 検証方式: leave-one-year-out（10年×5問、年単位で抜く）。
- 特徴量抽出完了（work/features.json、13特徴量）。検証ゲート全通過：保存スコアから baseline 29/3.88、stack 31/3.38、bgeCE 13/9.33、jpCE 13/8.88、ruri 14/8.31、e5（weighted_fusion順位と一致）を完全再現。
- 学習完了（torch ロジスティック回帰 pointwise / pairwise RankNet、LOYO）:
  - pointwise: **34 / 2.93**（top-3 は現行最良セルと同数、平均は全測定中最良）
  - pairwise: 33 / **2.88**（平均は最良）
  - in-sample 診断と LOYO がほぼ同水準 → 年を跨いで効く重み。
  - 重み最大: fld_tag (+0.50)、fam_freq_loo (+0.38)、lex_stack、bge_ce。ruri/e5/jpCE/hookB はほぼゼロ。fld_exception は負。
  - ablation: 言葉数え系を落とすと 21/5.55 に崩壊（背骨）。フック/おなじみは平均の底上げ役。
- LTR_REPORT.md 作成済み。キュー Step 6（最終）完了。
