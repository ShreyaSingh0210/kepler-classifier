import numpy as np
import pandas as pd

from src.baseline import compute_precision_at_k, evaluate_metrics
from src.splits import get_folds


def make_df(n_stars=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for kepid in range(n_stars):
        label = int(rng.random() < 0.4)
        for _ in range(rng.integers(1, 4)):
            rows.append({"kepid": kepid, "label": label})
    return pd.DataFrame(rows)


def test_no_star_in_train_and_test():
    df = make_df()
    for seed in [0, 1, 2]:
        for fold, train_idx, test_idx in get_folds(df, seed):
            train_stars = set(df.loc[train_idx, "kepid"])
            test_stars = set(df.loc[test_idx, "kepid"])
            assert not (train_stars & test_stars)
            assert len(train_idx) + len(test_idx) == len(df)


def test_precision_at_k():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_precision_at_k(y, scores, target_k=2) == 1.0
    assert compute_precision_at_k(y, scores, target_k=4) == 0.5


def test_perfect_scores_give_perfect_metrics():
    y = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    m = evaluate_metrics(y, scores)
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0