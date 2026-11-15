# data/ ― 公開可能な派生データ

いずれも条文・過去問から機械的またはモデル支援で作った派生データです。**過去問の問題文・選択肢の全文は、著作権上の理由で本リポジトリのどこにも含めていません**（問題は「実施年＋問題番号」で参照します）。

| ファイル | 内容 | 出どころ |
|---|---|---|
| `governing_map.jsonl` | 問題ID（year, no）↔ 正解条文（law_id, art）の対応表と、同定の確度（confidence: high/medium/low）。300問分。正解条文は解説の引用など証拠に基づいて同定したもの | `experiments/prototype/c_full/` の評価資産から、該当フィールドのみを抽出 |
| `word_tags.jsonl` | 条文ごとの単語タグ（485条文、平均50.5語）。条文を読んで「その条文を言い表す単語」を貼った下ごしらえ | `experiments/tag_vector/` |
| `signboards50.jsonl` | 50問測定で使った「看板v2」（条文の平易な言い換え、463条文） | `experiments/fifty_measurement/` |
| `qtags50.jsonl` | 50問測定で使った質問タグ（想定質問文＋単語、463条文） | `experiments/fifty_measurement/` |
| `hooks_A.json` | フック集合A： JLaw-CiteGraph の11法クラスタ内で被参照数（in-degree）≥5 の374条文（A8: ≥8 の146条文も収録） | `experiments/hook_search/` |
| `hooks_B.json` | フック集合Bの定義と「おなじみ」候補（過去問で繰り返し正解になった条文） | `experiments/hook_search/` |
| `familiar_freq.json` | 過去問300問における正解条文の頻度表（おなじみ信号の材料。測定時は leave-one-out で使用） | `experiments/hook_search/` |
| `expansion_dict.json` | 質問側の展開辞書（1,092エントリ。実行時は辞書引きだけで拡張し、AIは使わない設計のもの） | `experiments/tag_vector/` |
| `child_keywords.json` | フック条文ごとに、グラフで1ホップ隣接する子条文から吸い上げた固有語（約400フック） | `experiments/graph_propagation/` |

## 帰属

- `hooks_A.json` など JLaw-CiteGraph のグラフ構造から導いたデータは、JLaw-CiteGraph（yabooung、CC BY 4.0、https://github.com/yabooung/jp-law-citation-graph ）の派生物です。
- 法令ID（law_id）は e-Gov 法令検索の法令IDです。条文番号（art）は「112」「126-2」のような本則の条番号表記です。
