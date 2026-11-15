#!/usr/bin/env python3
"""hotchpotch/japanese-reranker-cross-encoder-small-v1（ONNX 変換版）で
fifty 50問の TF-IDF 上位30プールを採点する。作法は rerank_ce/work/rerank.py と同一：
- 問題文 = c_full/questions.jsonl の stem + 選択肢
- 条文 = qtag50_pool_texts.json の原文 text
- ペア: [s] q [/s][/s] chunk [/s]、上限512。窓幅 = 512 - (問題トークン + 4)。
  重なりなし、窓は最大4つ（超える場合は等間隔サンプリング、先頭含む）。条文の点数 = 窓の最大値。
- 問題ごとに work/scores_jp_partial.json へ逐次保存（再開可能）。"""
import json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

RK = Path("/home/hatch/workspace/research_kudou")
WORK = RK / "rerank_blend/work"
MAXLEN = 512
BATCH = 16
MAXWIN = 4

tok = Tokenizer.from_file(str(WORK / "hotchpotch_model/tokenizer.json"))
BOS = tok.token_to_id("<s>")
EOS = tok.token_to_id("</s>")
PAD = tok.token_to_id("<pad>")
print(f"BOS={BOS} EOS={EOS} PAD={PAD}", flush=True)

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(WORK / "hotchpotch.onnx"), sess_options=so,
                            providers=["CPUExecutionProvider"])
INAMES = [i.name for i in sess.get_inputs()]
print("inputs:", INAMES, flush=True)


def ids_of(text):
    return tok.encode(text, add_special_tokens=False).ids


def build_pairs(q_ids, p_ids):
    budget = MAXLEN - (len(q_ids) + 4)
    if budget < 32:
        q_ids = q_ids[: MAXLEN - 36]
        budget = MAXLEN - (len(q_ids) + 4)
    if len(p_ids) <= budget:
        chunks = [p_ids]
    else:
        n_win = -(-len(p_ids) // budget)
        if n_win <= MAXWIN:
            starts = [i * budget for i in range(n_win)]
        else:
            span = len(p_ids) - budget
            starts = sorted({round(span * i / (MAXWIN - 1)) for i in range(MAXWIN)})
        chunks = [p_ids[s : s + budget] for s in starts]
    return [[BOS] + q_ids + [EOS, EOS] + c + [EOS] for c in chunks]


def score_batch(seqs):
    maxlen = max(len(s) for s in seqs)
    input_ids = np.full((len(seqs), maxlen), PAD, dtype=np.int64)
    attn = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = s
        attn[i, : len(s)] = 1
    feeds = {"input_ids": input_ids, "attention_mask": attn}
    if "token_type_ids" in INAMES:
        feeds["token_type_ids"] = np.zeros_like(input_ids)
    out = sess.run(None, feeds)[0]
    return np.asarray(out).reshape(len(seqs), -1)[:, 0]


def main():
    pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
    pool_texts = json.load(open(RK / "question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
    texts = {tuple(k.split("|")): v["text"] for k, v in pool_texts.items()}
    qs = {}
    for line in open(RK / "prototype/c_full/questions.jsonl", encoding="utf-8"):
        d = json.loads(line)
        qs[f'{d["year"]}-{d["no"]}'] = d["stem"] + " " + " ".join(d["choices"])

    partial_path = WORK / "scores_jp_partial.json"
    done = json.load(open(partial_path, encoding="utf-8")) if partial_path.exists() else {}

    for qi, qid in enumerate(pools.keys()):
        if qid in done:
            print(f"[{qi+1}/50] {qid} skip (done)", flush=True)
            continue
        t0 = time.time()
        q_ids = ids_of(qs[qid])
        jobs = []
        for e in pools[qid]:
            key = (e["law"], e["art"])
            p_ids = ids_of(texts[key])
            for seq in build_pairs(q_ids, p_ids):
                jobs.append((key, seq))
        jobs.sort(key=lambda x: len(x[1]))
        scores = {}
        nchunks = {}
        for i in range(0, len(jobs), BATCH):
            batch = jobs[i : i + BATCH]
            vals = score_batch([s for _, s in batch])
            for (key, _), v in zip(batch, vals):
                ks = f"{key[0]}|{key[1]}"
                nchunks[ks] = nchunks.get(ks, 0) + 1
                if ks not in scores or v > scores[ks]:
                    scores[ks] = float(v)
        done[qid] = {"scores": scores, "n_chunks": nchunks,
                     "n_pairs": len(jobs), "sec": round(time.time() - t0, 1)}
        json.dump(done, open(partial_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{qi+1}/50] {qid}: {len(jobs)} pairs in {time.time()-t0:.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
