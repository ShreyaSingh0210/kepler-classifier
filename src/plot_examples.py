from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DIR = Path("data") / "processed"


def load_all():
    items = []
    for f in sorted(PROCESSED_DIR.glob("*.npz")):
        d = np.load(f)
        items.append((f.stem, int(d["label"]), d["global_view"], d["local_view"]))
    return items


def plot_examples(n_each=3):
    items = load_all()
    planets = [i for i in items if i[1] == 1][:n_each]
    fakes = [i for i in items if i[1] == 0][:n_each]
    chosen = planets + fakes
    print(f"Stars saved: {len(items)} (using {len(planets)} planets, {len(fakes)} false positives)")

    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 2.2 * len(chosen)))
    for row, (kepid, label, g, l) in enumerate(chosen):
        name = "planet" if label == 1 else "false positive"
        axes[row, 0].plot(g, ".", markersize=1.5)
        axes[row, 0].set_ylabel(f"{kepid}\n{name}", fontsize=8)
        axes[row, 1].plot(l, ".", markersize=3)
        if row == 0:
            axes[row, 0].set_title("Global view (full orbit)")
            axes[row, 1].set_title("Local view (zoom on the dip)")

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/example_curves.png", dpi=150)
    plt.close()
    print("Saved results/example_curves.png")


if __name__ == "__main__":
    plot_examples()