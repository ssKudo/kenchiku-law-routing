#!/usr/bin/env python3
"""bge-reranker-v2-m3 (ONNX int8) で fifty 50問の TF-IDF 上位30プールを並べ替える。
- 問題文 = c_full/questions.jsonl の stem + 選択肢（measure50 と同一）
- 条文 = qtag50_pool_texts.json の原文 text
- ペア採点: [s] q [/s][/s] passage_chunk [/s]、上限512トークン。
  長い条文は重なり64でチャンク分割し、最大値を条文の点数とする。
- 問題ごとに work/scores_partial.json へ逐次保存（再開可能）。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

RK = Path("/home/hatch/workspace/research_kudou")
WORK = RK / "rerank_ce/work"
import os
MODEL = Path(os.environ.get("RERANK_MODEL", WORK / "model3/model_int8.onnx"))
MAXLEN = 512
OVERLAP = 64
BATCH = 16

tok = Tokenizer.from_file(str(WORK / "model3/tokenizer.json"))
BOS = tok.token_to_id("<s>")
EOS = tok.token_to_id("</s>")
PAD = tok.token_to_id("<pad>")
print(f"BOS={BOS} EOS={EOS} PAD={PAD}", flush=True)

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])
INAMES = [i.name for i in sess.get_inputs()]
print("inputs:", INAMES, "outputs:", [o.name for o in sess.get_outputs()], flush=True)


def ids_of(text):
    return tok.encode(text, add_special_tokens=False).ids


MAXWIN = 4


def build_pairs(q_ids, p_ids):
    """(q_ids, p_ids) → 窓ごとの完全な input_ids 列のリスト。
    窓幅 = 512 - (問題文トークン + 特殊4)。重なりなし。
    窓数が MAXWIN を超える条文は、条文全体から等間隔に MAXWIN 窓をサンプリングする
    （先頭の窓は必ず含む）。条文の点数は窓の最大値。"""
    budget = MAXLEN - (len(q_ids) + 4)  # BOS + EOS EOS + EOS
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

    partial_path = WORK / "scores_partial.json"
    done = json.load(open(partial_path, encoding="utf-8")) if partial_path.exists() else {}

    for qi, qid in enumerate(pools.keys()):
        if qid in done:
            print(f"[{qi+1}/50] {qid} skip (done)", flush=True)
            continue
        t0 = time.time()
        q_ids = ids_of(qs[qid])
        jobs = []  # (key, seq)
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
