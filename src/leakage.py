from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier

from src.data import load_koi
from src.features import FEATURES
from src.splits import SEEDS, get_folds

LEAK_COLS = ["koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]


def load_for_leakage():
    df = load_koi()
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    # same rows for both runs, so the comparison is fair
    df = df[["kepid", "label"] + FEATURES + LEAK_COLS].dropna()
    print("Rows used:", len(df))
    return df.reset_index(drop=True)


def run_setting(df, columns, name):
    X = df[columns].to_numpy()
    y = df["label"].to_numpy()
    rows = []
    for seed in SEEDS:
        for fold, train_idx, test_idx in get_folds(df, seed):
            model = XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                random_state=seed, eval_metric="logloss",
            )
            model.fit(X[train_idx], y[train_idx])
            scores = model.predict_proba(X[test_idx])[:, 1]
            rows.append({
                "setting": name,
                "f1": f1_score(y[test_idx], (scores >= 0.5).astype(int)),
                "pr_auc": average_precision_score(y[test_idx], scores),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_for_leakage()
    without = run_setting(df, FEATURES, "without leak columns")
    with_leak = run_setting(df, FEATURES + LEAK_COLS, "with leak columns")

    runs = pd.concat([without, with_leak])
    table = runs.groupby("setting")[["f1", "pr_auc"]].agg(["mean", "std"]).round(3)

    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/leakage_table.csv")
    print(table.T)