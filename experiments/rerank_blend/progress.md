# Step 5：日本語リランカー混合版（rerank_blend）進捗

開始： 2026-11-15（クドウ承認済み、別AIレポート提案キューの Step 5）

## 測る仕組み
- 土俵： fifty 50問・原文 TF-IDF 上位30プール（従来と同一）。
- 言葉数え側： 連結スタック（原文＋単語タグ＋メタデータ）の生スコア（weighted_fusion と同一作法、再検証する）。
- CE側（主測定）： hotchpotch/japanese-reranker-cross-encoder-small-v1（XLM-RoBERTa系 MiniLM-L12-H384、約118M、生成しない採点専用）。公式ONNX配布なし（model.safetensors のみ）→ 変換して ONNX Runtime で実行する。
- 混合： プール内 Min-Max 正規化 → S = w言葉数え·スタック ＋ wCE·CE、(w言葉数え, wCE) ∈ {(0.8,0.2), (0.7,0.3), (0.6,0.4)} 全パターン報告。
- フォールバック（日本語モデルが動かない場合に必須）： rerank_ce 保存済み bge ペアスコアで同じ混合を計算。
- 参照値： 連結スタック 31/3.38； 現行最良 スタック×ruri α=0.80〜0.90 = 33/3.26〜3.33； bge単独 13/9.33； 天井 top-3 42。

## 記録
- 2026-11-15 開始。資産確認： rerank_ce/work/scores_partial.json に bge 全ペアスコアあり。weighted_fusion/work/measure_fusion.py にスタック生スコアの再現手順（検証assert付き）あり。
- 候補モデル確認： hotchpotch small = XLMRobertaForSequenceClassification（hidden 384, 12層, vocab 250002, max 512）、tokenizer.json あり。ONNX 配布は両候補とも無し。
- torch 導入を試行中（変換の正攻法＋ numpy/ONNX 自作実装の照合用）。ダメなら safetensors から ONNX グラフを自作する。
- 参照再検証（prep_scores.py、全て assert 通過）： baseline 29/3.88（results50 と全問一致）、連結スタック 31/3.38（countermeasures と全問一致）、現行最良 stack×ruri は保存済み doc 埋め込みから再構成し α=0.80→33/3.26、0.85→33/3.33、0.90→33/3.31、0.95→32/3.33 で保存結果と全問一致。bge 保存スコアから単独 13/9.33 を完全再現。
- torch 2.15.0+cpu と transformers 5.23.0 の導入に成功（過去の失敗は配布サーバ側の一過性とみられる）。hotchpotch モデルを DL 済み。
- ONNX 変換完了（convert_and_check.py）： ORT vs torch の最大差 1e-6 で一致を検証。健全性チェック： 2019-3 では正解条文が 0.67 で他（≤0.23）を大きく上回る等、モデルは正常に反応している。
- 全ペア採点を実行中（score_jp.py、rerank_ce と同一の窓方式、約0.43秒/ペア）。
- 採点は 41/50問時点でサービス再起動によりプロセスが一度死んだが、問題ごとの逐次保存から再開し、欠けなく完走（全 4,109ペア、計 1,708秒）。
- 集計完了（blend.py）。結果：
  - 日本語CE単独： 13/8.88（bge単独 13/9.33 と同水準の負け。日本語特化でも単独は勝てない）
  - スタック × 日本語CE： 0.8/0.2 → 33/3.40、0.7/0.3 → 33/3.33、0.6/0.4 → 28/3.79
  - スタック × bge CE（フォールバック）： **0.8/0.2 → 34/3.14（新最良、出0問）**、0.7/0.3 → 34/3.33、0.6/0.4 → 32/3.36
- RERANK_BLEND_REPORT.md と results_rerank_blend.json を作成済み。Step 5 完了。
