#!/usr/bin/env python3
"""fp32 ONNX (model3) を onnxruntime 標準の動的量子化で int8 化する。"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

WORK = Path("/home/hatch/workspace/research_kudou/rerank_ce/work")
src = WORK / "model3/model.onnx"
dst = WORK / "model3/model_int8.onnx"
quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
print("written", dst, dst.stat().st_size)
