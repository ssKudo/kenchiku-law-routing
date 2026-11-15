#!/usr/bin/env python3
"""bge-reranker-v2-m3 (ORT最適化済み fp32 グラフ) の MatMul 重みだけを
ストリーミングで読み、動的量子化パターンに手作業で置換する。
- MatMul(A, W) → DynamicQuantizeLinear(A) → MatMulInteger → Cast → スケール Mul
- W は出力チャネルごとに int8 対称量子化（scale は fp32 で保持）
- 重みは外部データファイルから1テンソルずつ読むのでメモリを食わない。
- Attention 融合op・Gemm・埋め込みは fp32 のまま。
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path

WORK = Path("/home/hatch/workspace/research_kudou/rerank_ce/work")
SRC = WORK / "model3/model.onnx"
DATA = WORK / "model3/model.onnx.data"
DST = WORK / "model3/model_int8.onnx"
DST_DATA = WORK / "model3/model_int8.onnx.data"

m = onnx.load(str(SRC), load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}
fdata = open(DATA, "rb")

DTYPE = {TensorProto.FLOAT: np.float32, TensorProto.INT8: np.int8,
         TensorProto.INT32: np.int32, TensorProto.INT64: np.int64}


def read_tensor(t):
    loc = {e.key: e.value for e in t.external_data}
    off, ln = int(loc["offset"]), int(loc["length"])
    fdata.seek(off)
    raw = fdata.read(ln)
    arr = np.frombuffer(raw, dtype=DTYPE[t.data_type]).reshape(list(t.dims))
    return arr


new_nodes = []
new_inits = []
drop_inits = set()
int8_buf = bytearray()
stats = []

for node in g.node:
    if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in inits:
        a_name, w_name = node.input[0], node.input[1]
        out_name = node.output[0]
        W = read_tensor(inits[w_name]).astype(np.float32)  # [K, N]
        K, N = W.shape
        scale = np.maximum(np.abs(W).max(axis=0) / 127.0, 1e-12).astype(np.float32)  # [N]
        Wq = np.round(W / scale).clip(-128, 127).astype(np.int8)
        # 外部データへ追記
        off = len(int8_buf)
        int8_buf += Wq.tobytes()
        tag = w_name.replace("/", "_")
        wq_name, ws_name, wzp_name = f"{tag}_q", f"{tag}_scale", f"{tag}_zp"
        t = TensorProto()
        t.name = wq_name
        t.data_type = TensorProto.INT8
        t.dims.extend([K, N])
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", DST_DATA.name), ("offset", str(off)),
                     ("length", str(Wq.size))):
            e = t.external_data.add(); e.key = k; e.value = v
        new_inits.append(t)
        new_inits.append(numpy_helper.from_array(scale, name=ws_name))
        new_inits.append(numpy_helper.from_array(
            np.zeros([N], dtype=np.int8), name=wzp_name))
        drop_inits.add(w_name)
        aq, asc, azp = f"{tag}_aq", f"{tag}_asc", f"{tag}_azp"
        yi, yf = f"{tag}_yi", f"{tag}_yf"
        s1, sfull = f"{tag}_s1", f"{tag}_sfull"
        new_nodes.append(helper.make_node(
            "DynamicQuantizeLinear", [a_name], [aq, asc, azp], name=f"{tag}_dql"))
        new_nodes.append(helper.make_node(
            "MatMulInteger", [aq, wq_name, azp, wzp_name], [yi], name=f"{tag}_mmi"))
        new_nodes.append(helper.make_node(
            "Cast", [yi], [yf], name=f"{tag}_cast", to=TensorProto.FLOAT))
        new_nodes.append(helper.make_node(
            "Unsqueeze", [asc], [s1], name=f"{tag}_unsq", axes=[-1]))
        new_nodes.append(helper.make_node(
            "Mul", [s1, ws_name], [sfull], name=f"{tag}_smul"))
        new_nodes.append(helper.make_node(
            "Mul", [yf, sfull], [out_name], name=f"{tag}_omul"))
        stats.append((w_name, W.shape))
    else:
        new_nodes.append(node)

del g.node[:]
g.node.extend(new_nodes)
kept = [i for i in g.initializer if i.name not in drop_inits]
del g.initializer[:]
g.initializer.extend(kept + new_inits)

DST_DATA.write_bytes(bytes(int8_buf))
onnx.save_model(m, str(DST), save_as_external_data=False)
print(f"quantized {len(stats)} MatMul weights; int8 data {len(int8_buf)/1e6:.0f} MB")
print("saved", DST)
