# rerank_ce 進捗

## 2026-11-15 開始
- 承認： クドウ 2026-11-15（「おお、やってみて」）。
- 測る仕組み： 採点専用 cross-encoder **bge-reranker-v2-m3**（約568M、生成なし）で、fifty_measurement の50問・原文 TF-IDF 上位30条文プールを並べ替える。
- 参照値の再検証：
  - ベースライン（results50.json rank_original から再計算）： プール内42問、top-3 **29/50**、平均順位 **3.88**（プール内42問の平均）。✅ 一致
  - 最良スタック（原文＋単語タグ＋メタデータ、countermeasures の meta_append = 31/3.38）： results_countermeasures.json から再計算予定。
  - 完璧な並べ替えの天井： top-3 42（プール内全問）。
- 素材の同一性： 問題文 = c_full/questions.jsonl の stem + 選択肢（measure50 と同一の組み立て）。条文 = qtag50_pool_texts.json の原文 text（vector_hybrid と同一）。
- モデル入手： 公式 BAAI リポジトリに ONNX が無いため、コミュニティの ONNX 量子化エクスポート `arnauld-a/bge-reranker-v2-m3-onnx-quant` の model_quant.onnx（int8系、880MB）を使用。onnxruntime 1.31.1（venv）で実行（vector_hybrid の e5 と同じ方式）。
- 長い条文の扱い： モデル上限512トークン。問題文トークンを除いた残り枠で条文を重なり付きチャンクに分割し、ペアごとに採点、**最大値を条文の点数**とする（max pooling）。方法はレポートに明記する。

## モデル入手の試行錯誤（正直に記録）
1. `arnauld-a/bge-reranker-v2-m3-onnx-quant` の model_quant.onnx（880MB）： DequantizeLinear の scale が float16 で、onnxruntime CPU が INVALID_GRAPH で拒否。読み込み不可。
2. `sudhanshu746/bge-reranker-v2-m3-quant-onnx`（1.14GB）： 読み込みはできたが全ペアのスコアが **NaN**。かつ 4.0秒/ペアと遅い。使い物にならない。
3. 対策： `EmbeddedLLM/bge-reranker-v2-m3-onnx-o3-cpu` の fp32 版（2.27GB）を落とし、onnxruntime 標準の **動的量子化（int8）を自分で適用**して使う。fp32 と int8 のスコア相関も少数ペアで検証する。

## モデル確定（2026-11-15）
- 採用： `EmbeddedLLM/bge-reranker-v2-m3-onnx-o3-cpu` の fp32 ONNX を自前で動的量子化（MatMul 72個の重みを出力チャネル単位 int8 対称量子化、DynamicQuantizeLinear＋MatMulInteger に置換。Attention 融合op・Gemm・埋め込みは fp32 のまま）。ORT 標準量子化ツールは /tmp 容量とメモリで落ちたため手作業化（重みは外部データからストリーミング読み）。
- 検証： 同一18ペアで fp32 と int8 のスコアが近接（例 -0.67/-0.59、-4.25/-4.19）。NaN なし。速度 int8 = **1.65秒/ペア**（fp32 4.48秒の約2.7倍速）。相関係数は集計時に18ペア分を算出する。
- 長い条文の扱い（確定）： 条文トークン列を「512 − 問題文 − 特殊4」の窓幅で重なりなしに区切り、**最大4窓**まで。4窓を超える条文は全体から等間隔に4窓をサンプリング（先頭窓は必ず含む）。条文の点数 = 窓スコアの最大値。全ペア数 4,109（うち上限サンプリング 540スロット、完全な複数窓 608、単窓 352）。
  - 理由： 条文は中央値605トークン・最大48,557トークンと非常に長く、全窓の最大値方式だと15,000ペア超で計算が回らない。

## 状態
- [x] 参照値の再検証（ベースライン 29/3.88・最良スタック 31/3.38、集計時に再計算し完全一致）
- [x] モデルDL・動作確認（速度計測）→ int8 自前量子化で確定
- [x] 50問×30条文の採点 → 全4,109ペア完了（45問時点でプロセスが一度死に、逐次保存から残り5問を再開して完走）
- [x] 集計・RERANK_REPORT.md

## 結果（確定）
- **cross-encoder リランク： top-3 13/50・平均順位 9.33 —— ベースライン（29/3.88）にも最良スタック（31/3.38）にも大きく負け。**
- 動き： 上がった9問 / 下がった29問、top-3 入り3問 vs 落ち19問。1位は8問（ベースライン18問）。
- 量子化検証： fp32 との相関 Pearson 0.991 / Spearman 0.963（18ペア）——量子化は敗因ではない。
- 詳細は RERANK_REPORT.md。
- [ ] 50問×30条文の採点（問題ごとに逐次保存・再開可能）
- [ ] 集計・RERANK_REPORT.md
