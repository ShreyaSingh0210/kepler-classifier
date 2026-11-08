from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.features import FEATURES, fetch_processed_dataset
from src.splits import get_folds

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
N_REPEATS = 10
SPLIT_SEED = 0  # one fixed set of 5 star-grouped folds


def run_learning_curve():
    df = fetch_processed_dataset().reset_index(drop=True)
    X = df[FEATURES].to_numpy()
    y = df["label"].to_numpy()
    folds = list(get_folds(df, SPLIT_SEED))

    rows = []
    for frac in FRACTIONS:
        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(repeat)
            fold_scores = []
            for fold, train_idx, test_idx in folds:
                n = max(2, int(len(train_idx) * frac))
                sub_idx = rng.choice(train_idx, size=n, replace=False)
                model = XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=4,
                    random_state=repeat, eval_metric="logloss",
                )
                model.fit(X[sub_idx], y[sub_idx])
                preds = model.predict(X[test_idx])
                fold_scores.append(f1_score(y[test_idx], preds))
            rows.append({"fraction": frac, "repeat": repeat, "f1": np.mean(fold_scores)})
        print(f"fraction {frac} done")

    runs = pd.DataFrame(rows)
    table = runs.groupby("fraction")["f1"].agg(["mean", "std"]).round(3)

    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/learning_curve.csv")
    print(table)

    plt.figure(figsize=(6, 4))
    plt.errorbar(table.index * 100, table["mean"], yerr=table["std"], marker="o", capsize=4)
    plt.xlabel("Training data used (%)")
    plt.ylabel("F1 (mean ± std)")
    plt.title("Learning curve (XGBoost)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150)
    plt.close()
    return table


if __name__ == "__main__":
    run_learning_curve()