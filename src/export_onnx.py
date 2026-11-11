import copy
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from src.models import TwoViewCNN

ONNX_PATH = Path("data") / "model.onnx"


class SlicePool(nn.Module):
    """Same result as AdaptiveAvgPool1d, written with plain slices so ONNX can export it."""

    def __init__(self, out=4):
        super().__init__()
        self.out = out

    def forward(self, x):
        L = int(x.shape[-1])
        pieces = []
        for i in range(self.out):
            start = (i * L) // self.out
            end = -((-(i + 1) * L) // self.out)
            pieces.append(x[..., start:end].mean(dim=-1, keepdim=True))
        return torch.cat(pieces, dim=-1)


def swap_pools(model):
    for branch in (model.global_branch, model.local_branch):
        for i, m in enumerate(branch):
            if isinstance(m, nn.AdaptiveAvgPool1d):
                branch[i] = SlicePool(4)


def median_ms(fn, n=100, warm=10):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main():
    torch.manual_seed(0)
    model = TwoViewCNN().eval()

    export_model = copy.deepcopy(model).eval()
    swap_pools(export_model)
    ONNX_PATH.parent.mkdir(exist_ok=True)
    torch.onnx.export(
        export_model, (torch.randn(1, 1, 2001), torch.randn(1, 1, 201)), str(ONNX_PATH),
        input_names=["g", "l"], output_names=["logit"],
        dynamic_axes={"g": {0: "b"}, "l": {0: "b"}, "logit": {0: "b"}},
        opset_version=17, dynamo=False,
    )
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # 1) do both give the same answers?
    g, l = torch.randn(64, 1, 2001), torch.randn(64, 1, 201)
    with torch.no_grad():
        ref = model(g, l).numpy()
    out = sess.run(None, {"g": g.numpy(), "l": l.numpy()})[0]
    print("Max abs difference:", float(np.abs(ref - out).max()))
    print("Same planet/not-planet call:", float(((ref > 0) == (out > 0)).mean()))

    # 2) speed
    rows = []
    for batch in (1, 256):
        g, l = torch.randn(batch, 1, 2001), torch.randn(batch, 1, 201)
        gn, ln = g.numpy(), l.numpy()

        def run_torch():
            with torch.no_grad():
                model(g, l)

        def run_onnx():
            sess.run(None, {"g": gn, "l": ln})

        for name, fn in (("pytorch_cpu", run_torch), ("onnxruntime_cpu", run_onnx)):
            ms = median_ms(fn)
            rows.append({"engine": name, "batch": batch, "latency_ms": round(ms, 3),
                         "samples_per_sec": round(batch / (ms / 1000), 1)})

    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/inference.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()