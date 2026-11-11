import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/benchmarks_batch_sweep.csv")
sweep = df[df["name"].str.match(r"^b\d+$")].sort_values("batch")

if sweep.empty:
    print("No batch-sweep rows yet. This file still has the old benchmark rows.")
else:
    plt.figure(figsize=(6, 4))
    plt.plot(sweep["batch"], sweep["samples_per_sec"], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Training samples per second (T4 GPU, AMP)")
    plt.title("Throughput vs batch size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/throughput_vs_batch.png", dpi=150)
    print(sweep[["name", "batch", "samples_per_sec"]].to_string(index=False))