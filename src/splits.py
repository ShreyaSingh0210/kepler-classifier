from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]


def get_folds(df, seed):
    """Yield (fold_number, train_index, test_index) with no star in both."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df, df["label"], groups=df["kepid"])
    ):
        yield fold, train_idx, test_idx


if __name__ == "__main__":
    from src.features import fetch_processed_dataset

    df = fetch_processed_dataset()
    df = df.reset_index(drop=True)

    for seed in SEEDS:
        for fold, train_idx, test_idx in get_folds(df, seed):
            train_stars = set(df.loc[train_idx, "kepid"])
            test_stars = set(df.loc[test_idx, "kepid"])
            overlap = train_stars & test_stars
            assert len(overlap) == 0, "A star is in both train and test!"
            print(
                f"seed {seed} fold {fold}: train {len(train_idx)}, test {len(test_idx)}, "
                f"planet share in test {df.loc[test_idx, 'label'].mean():.2f}"
            )
    print("OK: no star appears in both train and test.")