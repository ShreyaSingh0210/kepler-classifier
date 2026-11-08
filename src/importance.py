from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from src.features import FEATURES, fetch_processed_dataset
from src.splits import get_folds

SEED = 0


def run_importance():
    df = fetch_processed_dataset().reset_index(drop=True)
    X = df[FEATURES].to_numpy()
    y = df["label"].to_numpy()

    results = []
    for fold, train_idx, test_idx in get_folds(df, SEED):
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=SEED, eval_metric="logloss",
        )
        model.fit(X[train_idx], y[train_idx])
        r = permutation_importance(
            model, X[test_idx], y[test_idx],
            scoring="average_precision", n_repeats=10, random_state=SEED,
        )
        results.append(pd.Series(r.importances_mean, index=FEATURES))

    all_folds = pd.concat(results, axis=1)
    table = pd.DataFrame({"mean": all_folds.mean(axis=1), "std": all_folds.std(axis=1)})
    table = table.sort_values("mean", ascending=False).round(4)

    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/importance.csv")
    print(table)

    plt.figure(figsize=(6, 4))
    table_sorted = table.sort_values("mean")
    plt.barh(table_sorted.index, table_sorted["mean"], xerr=table_sorted["std"])
    plt.xlabel("Drop in PR-AUC when shuffled")
    plt.title("Permutation importance (XGBoost)")
    plt.tight_layout()
    plt.savefig("results/importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_importance()