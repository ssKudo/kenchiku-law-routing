# セットアップ記録（小型モデル試し運転）

日時: 2026-11-12

## マシンスペック（この VM）
- CPU: 4 vCPU 表示（共有約1.5コア相当の割り当て）
- メモリ: 7.7GB（空き約6.0GB）、スワップ 1.0GB
- ディスク: 150GB 中 147GB 空き
- アーキテクチャ: x86_64、Python 3.12.3

## 推論環境
- llama.cpp プリビルドバイナリ b12314（`llama-b12314-bin-ubuntu-x64.tar.gz`、CPU版 22MB）
  - 取得元: https://github.com/ggml-org/llama.cpp/releases/download/b12314/llama-b12314-bin-ubuntu-x64.tar.gz
  - 展開先: `~/workspace/tools/llama.cpp/llama-b12314/`（`llama-cli --version` = 0.9.0-dev build 12314 で動作確認）
  - 注意点: tar 展開時に「Cannot change ownership to uid 1001」という警告が多数出て tar の終了コードは 2 になったが、ファイル自体は全て展開されており実行に問題なし（所有権変更だけの失敗）。
  - 使用バイナリ: `llama-completion`（ChatML 整形済みプロンプトを渡す素の補完モード。出力パースが単純なため採用）
- GitHub API の `releases/latest` は v0.9.0（nightly-tag のみ）でバイナリが無かったため、タグ b12314 のリリースを直接指定した。

## モデル
- Qwen/Qwen2.5-1.5B-Instruct-GGUF、量子化 Q4_K_M
  - ファイル: `qwen2.5-1.5b-instruct-q4_k_m.gguf`（1,117,320,736 バイト = 約1.12GB）
  - 取得元: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
  - 保存先: `~/workspace/models/`
  - 取得は curl の通常ペースで約2〜3MB/s、所要およそ7分。HF への接続は問題なし（ミラー不要だった）。

## つまずいた点
- 上記 tar の所有権警告のみ。モデル取得・推論環境とも大きな障害なし。
