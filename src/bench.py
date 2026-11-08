import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.models import TwoViewCNN
from src.train import load_processed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--amp", action="store_true")      # mixed precision
    p.add_argument("--preload", action="store_true")  # keep all data on the device
    p.add_argument("--compile", action="store_true")  # torch.compile
    args = p.parse_args()
    dev = args.device

    G, L, y, _ = load_processed()
    y = torch.tensor(y).float()
    if args.preload:
        G, L, y = G.to(dev), L.to(dev), y.to(dev)
    n = len(y)

    model = TwoViewCNN().to(dev)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = args.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_epoch():
        perm = torch.randperm(n, device=G.device)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            g = G[idx].to(dev, non_blocking=True)
            l = L[idx].to(dev, non_blocking=True)
            t = y[idx].to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(g, l), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    run_epoch()  # warm-up, not timed
    if dev == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.epochs):
        run_epoch()
    if dev == "cuda":
        torch.cuda.synchronize()
    per_epoch = (time.perf_counter() - start) / args.epochs

    row = {
        "name": args.name, "device": dev, "batch": args.batch, "amp": use_amp,
        "preload": args.preload, "compile": args.compile,
        "sec_per_epoch": round(per_epoch, 3), "samples_per_sec": round(n / per_epoch, 1),
    }
    print(row)
    path = Path("results/benchmarks.csv")
    out = pd.DataFrame([row])
    if path.exists():
        out = pd.concat([pd.read_csv(path), out])
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()