from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.baseline import evaluate_metrics, initialize_classifiers
from src.data import load_koi
from src.features import FEATURES
from src.splits import get_folds

PROCESSED_DIR = Path("data") / "processed"
METRICS = ["precision", "recall", "f1", "pr_auc", "precision_at_k"]


def load_same_stars():
    """Same stars, same order as the CNN training uses."""
    files = sorted(PROCESSED_DIR.glob("*.npz"))
    kepids = np.array([int(f.stem) for f in files])
    labels = np.array([int(np.load(f)["label"]) for f in files])

    koi = load_koi()
    koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    # same row choice as lightcurves.py: first KOI row per star
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    koi = koi.drop_duplicates(subset="kepid").set_index("kepid")
    X = koi.loc[kepids, FEATURES].to_numpy()
    return pd.DataFrame({"kepid": kepids, "label": labels}), X, labels


def main(n_seeds=1):
    df, X, y = load_same_stars()
    print(f"Stars: {len(df)} | planets: {y.sum()} | missing feature cells: {int(np.isnan(X).sum())}")

    rows = []
    for seed in range(n_seeds):
        for fold, train_idx, test_idx in get_folds(df, seed):
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X[train_idx])
            X_test = imputer.transform(X[test_idx])
            for name, model in initialize_classifiers(seed).items():
                model.fit(X_train, y[train_idx])
                scores = model.predict_proba(X_test)[:, 1]
                r = evaluate_metrics(y[test_idx], scores)
                r.update({"model": name + "_1k", "seed": seed, "fold": fold})
                rows.append(r)
    tab = pd.DataFrame(rows)

    # add the CNN runs if the file is there
    cnn_path = Path("results") / "cnn_runs.csv"
    if cnn_path.exists():
        cnn = pd.read_csv(cnn_path).rename(columns={"run": "model"})
        tab = pd.concat([tab, cnn[["model", "seed", "fold"] + METRICS]])

    summary = tab.groupby("model")[METRICS].agg(["mean", "std"]).round(3)
    Path("results").mkdir(exist_ok=True)
    tab.to_csv("results/compare_runs.csv", index=False)
    summary.to_csv("results/compare_table.csv")
    print(summary.T)


if __name__ == "__main__":
    main()