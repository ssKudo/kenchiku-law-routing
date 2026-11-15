#!/usr/bin/env python3
"""multilingual-e5-small (ONNX int8) 埋め込み。
- 接頭辞: query: / passage:
- チャンク: 最大480 tokens、重なり48。チャンク埋め込み（L2正規化済）を平均し再正規化。
- プーリング: attention_mask による平均（e5 公式手順）。
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
MAXLEN = 480
OVERLAP = 48


class E5:
    def __init__(self, threads=4):
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(MODEL_DIR / "model_qint8_avx512_vnni.onnx"),
            sess_options=so, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def _encode_chunk(self, ids_list):
        """ids のリスト（各要素は special token 込みの完全な列）をバッチ推論し正規化済み埋め込みを返す。"""
        maxlen = max(len(x) for x in ids_list)
        input_ids = np.zeros((len(ids_list), maxlen), dtype=np.int64)
        attn = np.zeros((len(ids_list), maxlen), dtype=np.int64)
        for i, ids in enumerate(ids_list):
            input_ids[i, :len(ids)] = ids
            attn[i, :len(ids)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        out = self.sess.run(None, feeds)[0]  # (B, L, H)
        mask = attn[:, :, None].astype(np.float32)
        emb = (out * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb.astype(np.float32)

    def embed_texts(self, texts, prefix, batch_chunks=48):
        """テキスト群 → (N, H) 正規化済み埋め込み。長いテキストはチャンク平均。"""
        # まず全テキストをトークン化し、チャンク列に分解
        all_chunks = []       # (text_idx, ids)
        for ti, t in enumerate(texts):
            enc = self.tok.encode(prefix + t)
            ids = enc.ids
            if len(ids) <= MAXLEN:
                all_chunks.append((ti, ids))
            else:
                step = MAXLEN - OVERLAP
                # 先頭の [CLS] を保ち、本文をずらしながら切る簡易方式:
                # ids 全体を MAXLEN ごとに重なり付きで切る（境界の special token 重複は許容）
                start = 0
                while start < len(ids):
                    all_chunks.append((ti, ids[start:start + MAXLEN]))
                    if start + MAXLEN >= len(ids):
                        break
                    start += step
        embs = [None] * len(texts)
        chunk_embs = []
        for i in range(0, len(all_chunks), batch_chunks):
            part = all_chunks[i:i + batch_chunks]
            chunk_embs.extend(self._encode_chunk([c[1] for c in part]))
        agg = {}
        for (ti, _), e in zip(all_chunks, chunk_embs):
            agg.setdefault(ti, []).append(e)
        for ti in range(len(texts)):
            v = np.mean(agg[ti], axis=0)
            embs[ti] = v / max(np.linalg.norm(v), 1e-12)
        return np.stack(embs).astype(np.float32)


if __name__ == "__main__":
    import time
    m = E5()
    print("inputs:", m.input_names)
    q = "建築物に設ける非常用の照明装置について、建築基準法で定める技術的基準はどれか。"
    docs = [
        "非常用の照明装置は、停電時においても床面において1ルクス以上の照度を確保できるものとして、国土交通大臣が定めた構造のものとしなければならない。",
        "建築物の敷地は、接する道の境より高くなければならない。ただし、敷地の排水に支障がない場合はこの限りでない。",
        "定期報告の対象となる建築物については、所有者は一級建築士等に調査させ、その結果を特定行政庁に報告しなければならない。",
    ]
    t0 = time.time()
    qe = m.embed_texts([q], "query: ")
    de = m.embed_texts(docs, "passage: ")
    print("elapsed", round(time.time() - t0, 2), "s")
    print("cos:", (de @ qe[0]).round(4))
    # 長文チャンクの速度確認（約3000字）
    long_text = docs[0] * 30
    t0 = time.time()
    le = m.embed_texts([long_text], "passage: ")
    print("long text elapsed", round(time.time() - t0, 2), "s, norm", float(np.linalg.norm(le[0])))
