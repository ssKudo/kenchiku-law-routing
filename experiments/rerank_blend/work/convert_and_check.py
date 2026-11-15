#!/usr/bin/env python3
"""hotchpotch/japanese-reranker-cross-encoder-small-v1 を torch で読み、
少数ペアで健全性チェック → ONNX 変換 → ORT と torch の出力一致を検証する。
ペア構築は rerank_ce/work/rerank.py と同一作法（[s] q [/s][/s] chunk [/s]、上限512）。"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

RK = Path("/home/hatch/workspace/research_kudou")
WORK = RK / "rerank_blend/work"
MDIR = WORK / "hotchpotch_model"

torch.set_num_threads(4)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(str(MDIR))
model.eval()
print("arch:", model.config.architectures, "num_labels:", model.config.num_labels)

tok = Tokenizer.from_file(str(MDIR / "tokenizer.json"))
BOS = tok.token_to_id("<s>"); EOS = tok.token_to_id("</s>"); PAD = tok.token_to_id("<pad>")
print(f"BOS={BOS} EOS={EOS} PAD={PAD}")


def ids_of(text):
    return tok.encode(text, add_special_tokens=False).ids


def build_seq(q_ids, p_ids):
    budget = 512 - (len(q_ids) + 4)
    if budget < 32:
        q_ids = q_ids[: 512 - 36]
        budget = 512 - (len(q_ids) + 4)
    return [BOS] + q_ids + [EOS, EOS] + p_ids[:budget] + [EOS]


def torch_scores(seqs):
    maxlen = max(len(s) for s in seqs)
    input_ids = np.full((len(seqs), maxlen), PAD, dtype=np.int64)
    attn = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = s
        attn[i, : len(s)] = 1
    with torch.no_grad():
        out = model(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attn))
    return out.logits.numpy().reshape(len(seqs), -1)[:, 0]


# ---- 健全性チェック： 実問題で正解条文 vs プール中央あたりの条文 ----
pools = json.load(open(RK / "question_tag_pilot/work/qtag50_pools.json", encoding="utf-8"))
pool_texts = json.load(open(RK / "question_tag_pilot/work/qtag50_pool_texts.json", encoding="utf-8"))
texts = {tuple(k.split("|")): v["text"] for k, v in pool_texts.items()}
qs = {}
for line in open(RK / "prototype/c_full/questions.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qs[f'{d["year"]}-{d["no"]}'] = d["stem"] + " " + " ".join(d["choices"])
gov = {}
for line in open(RK / "prototype/c_full/governing.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d.get("governing"):
        gov[f'{d["year"]}-{d["no"]}'] = (d["governing"]["law_id"], d["governing"]["art"])

test_seqs, labels = [], []
for qid in ["2017-11", "2019-3", "2020-3"]:
    q_ids = ids_of(qs[qid])
    t = gov[qid]
    others = [ (e["law"], e["art"]) for e in pools[qid] if (e["law"], e["art"]) != t ][:4]
    for key, lab in [(t, "GOV")] + [(k, "other") for k in others]:
        test_seqs.append(build_seq(q_ids, ids_of(texts[key])))
        labels.append(f"{qid} {lab} {key[0][-4:]}|{key[1]}")
t0 = time.time()
vals = torch_scores(test_seqs)
print(f"torch sanity ({time.time()-t0:.1f}s for {len(test_seqs)} seqs):")
for lab, v in zip(labels, vals):
    print(f"  {lab}: {v:.4f}")

# ---- ONNX 変換 ----
class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


onnx_path = WORK / "hotchpotch.onnx"
dummy_ids = torch.tensor([[BOS] + [100] * 20 + [EOS]], dtype=torch.int64)
dummy_mask = torch.ones_like(dummy_ids)
torch.onnx.export(Wrap(model), (dummy_ids, dummy_mask), str(onnx_path),
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                "attention_mask": {0: "batch", 1: "seq"},
                                "logits": {0: "batch"}},
                  opset_version=14, dynamo=False)
print("exported:", onnx_path, onnx_path.stat().st_size / 1e6, "MB")

# ---- ORT 照合 ----
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
maxlen = max(len(s) for s in test_seqs)
input_ids = np.full((len(test_seqs), maxlen), PAD, dtype=np.int64)
attn = np.zeros((len(test_seqs), maxlen), dtype=np.int64)
for i, s in enumerate(test_seqs):
    input_ids[i, : len(s)] = s
    attn[i, : len(s)] = 1
ovals = sess.run(None, {"input_ids": input_ids, "attention_mask": attn})[0].reshape(len(test_seqs), -1)[:, 0]
diff = np.abs(ovals - vals)
print(f"ORT vs torch: max|diff|={diff.max():.6f} mean|diff|={diff.mean():.6f}")
assert diff.max() < 1e-3, "ONNX/torch mismatch"
print("ONNX CHECK OK")
