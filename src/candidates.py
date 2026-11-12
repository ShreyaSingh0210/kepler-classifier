from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data import load_koi
from src.features import FEATURES

N_SEEDS = 5
TOP_N = 100


def main():
    koi = load_koi()
    labeled = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].dropna(subset=FEATURES)
    cands = koi[koi["koi_disposition"] == "CANDIDATE"].dropna(subset=FEATURES).copy()
    X, y = labeled[FEATURES].to_numpy(), (labeled["koi_disposition"] == "CONFIRMED").astype(int).to_numpy()
    print(f"Train rows: {len(labeled)} | candidates to rank: {len(cands)}")

    scores = np.zeros(len(cands))
    for seed in range(N_SEEDS):
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=seed, eval_metric="logloss",
        )
        model.fit(X, y)
        scores += model.predict_proba(cands[FEATURES].to_numpy())[:, 1]
    cands["model_score"] = scores / N_SEEDS
    cands["snapshot_date"] = str(date.today())

    cols = ["kepid", "kepoi_name", "model_score", "koi_disposition", "koi_pdisposition", "koi_score", "snapshot_date"]
    ranked = cands.sort_values("model_score", ascending=False)[cols].reset_index(drop=True)
    Path("results").mkdir(exist_ok=True)
    ranked.to_csv("results/candidates_ranked.csv", index=False)
    print(ranked.head(20).round(3).to_string(index=False))

    # sanity check only: koi_score comes from NASA's own vetting, and was NOT used in training
    top = ranked.head(TOP_N)
    print(f"\nTop {TOP_N}: mean koi_score {top['koi_score'].mean():.3f} | all candidates: {ranked['koi_score'].mean():.3f}")
    print("Rank correlation (model_score vs koi_score):",
          round(ranked["model_score"].corr(ranked["koi_score"], method="spearman"), 3))


if __name__ == "__main__":
    main()
