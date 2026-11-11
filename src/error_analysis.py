from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_koi

PROCESSED = Path("data") / "processed"
INFO_COLS = [
    "koi_period", "koi_prad", "koi_model_snr",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
]


def plot_group(rows, title, path):
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 2 * len(rows)))
    for i, (_, r) in enumerate(rows.iterrows()):
        d = np.load(PROCESSED / f"{int(r['kepid'])}.npz")
        axes[i, 0].plot(d["local_view"], ".", markersize=3)
        axes[i, 0].set_ylabel(f"{int(r['kepid'])}\nscore {r['score']:.2f}", fontsize=7)
        axes[i, 1].plot(d["global_view"], ".", markersize=1.5)
        if i == 0:
            axes[i, 0].set_title("Local view")
            axes[i, 1].set_title("Global view")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main(name="final", n=10):
    preds = pd.read_csv(f"results/cnn_predictions_{name}.csv")
    stars = preds.groupby("kepid").agg(label=("label", "first"), score=("score", "mean")).reset_index()

    koi = load_koi()
    koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"]).drop_duplicates("kepid")
    stars = stars.merge(koi[["kepid"] + INFO_COLS], on="kepid", how="left")

    fps = stars[stars["label"] == 0].sort_values("score", ascending=False).head(n)
    fns = stars[stars["label"] == 1].sort_values("score").head(n)

    Path("results").mkdir(exist_ok=True)
    plot_group(fps, "Worst false positives (not planets, high CNN score)", "results/errors_false_positives.png")
    plot_group(fns, "Worst false negatives (real planets, low CNN score)", "results/errors_false_negatives.png")

    cols = ["kepid", "score"] + INFO_COLS
    print("\nWorst false positives:\n", fps[cols].round(2).to_string(index=False))
    print("\nWorst false negatives:\n", fns[cols].round(2).to_string(index=False))
    fps[cols].to_csv("results/errors_false_positives.csv", index=False)
    fns[cols].to_csv("results/errors_false_negatives.csv", index=False)


if __name__ == "__main__":
    main()