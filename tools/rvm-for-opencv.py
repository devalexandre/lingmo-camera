#!/usr/bin/env python3
# Makes data/rvm_mobilenetv3_640x360.onnx from Robust Video Matting's
# rvm_mobilenetv3_fp32.onnx (https://github.com/PeterL1n/RobustVideoMatting, GPL-3.0),
# so OpenCV's dnn module runs it:
#   - fixed shapes: a 640x360 picture, downsample_ratio 0.25 built in
#   - simplified (onnxsim), so the dynamic Resize/Shape parts become constants
#   - AveragePool with ceil_mode on odd sizes -> edge Pad + plain AveragePool (same result;
#     OpenCV divides the partial windows differently)
#   - Tanh inputs clamped to [-10, 10] (OpenCV's Tanh gives NaN for large inputs)
#
# pip install onnx onnxsim numpy
# python tools/rvm-for-opencv.py rvm_mobilenetv3_fp32.onnx data/rvm_mobilenetv3_640x360.onnx
import math
import sys

import numpy as np
import onnx
import onnxsim
from onnx import helper, numpy_helper, shape_inference

W, H, RATIO = 640, 360, 0.25

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
graph = model.graph

inputs = [i for i in graph.input if i.name != "downsample_ratio"]
del graph.input[:]
graph.input.extend(inputs)
graph.initializer.append(numpy_helper.from_array(np.array([RATIO], np.float32), "downsample_ratio"))


def halve(x, times):
    for _ in range(times):
        x = math.ceil(x / 2)
    return x


h, w = round(H * RATIO), round(W * RATIO)
shapes = {"src": [1, 3, H, W]}
for name, channels, level in [("r1i", 16, 1), ("r2i", 20, 2), ("r3i", 40, 3), ("r4i", 64, 4)]:
    shapes[name] = [1, channels, halve(h, level), halve(w, level)]
model, ok = onnxsim.simplify(model, overwrite_input_shapes=shapes)
assert ok

model = shape_inference.infer_shapes(model)
graph = model.graph
dims = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(graph.value_info) + list(graph.input)}
graph.initializer.extend([numpy_helper.from_array(np.array(-10, np.float32), "clamp_lo"),
                          numpy_helper.from_array(np.array(10, np.float32), "clamp_hi")])
nodes = []
for i, node in enumerate(graph.node):
    if node.op_type == "AveragePool" and any(a.name == "ceil_mode" and a.i for a in node.attribute):
        shape = dims[node.input[0]]
        for a in node.attribute:
            if a.name == "ceil_mode":
                a.i = 0
        pads = [0, 0, 0, 0, 0, 0, shape[2] % 2, shape[3] % 2]
        if any(pads):
            graph.initializer.append(numpy_helper.from_array(np.array(pads, np.int64), f"pad{i}"))
            nodes.append(helper.make_node("Pad", [node.input[0], f"pad{i}"], [f"pad{i}_out"],
                                          mode="edge", name=f"Pad{i}"))
            node.input[0] = f"pad{i}_out"
    elif node.op_type == "Tanh":
        nodes.append(helper.make_node("Clip", [node.input[0], "clamp_lo", "clamp_hi"], [f"clamp{i}_out"],
                                      name=f"Clamp{i}"))
        node.input[0] = f"clamp{i}_out"
    nodes.append(node)
del graph.node[:]
graph.node.extend(nodes)
del graph.value_info[:]
onnx.checker.check_model(model)
onnx.save(model, dst)
print("recurrent state shapes:", {k: v for k, v in shapes.items() if k != "src"})
