from sklearn.model_selection import StratifiedGroupKFold

TOTAL_SPLITS = 5
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def generate_cv_indices(dataset, seed_value):
    """Yields (fold_idx, train_indices, test_indices) ensuring group isolation."""
    cv_splitter = StratifiedGroupKFold(
        n_splits=TOTAL_SPLITS, shuffle=True, random_state=seed_value
    )
    
    # Extract structural arrays directly to modify tracking footprint
    targets = dataset["label"].values
    group_ids = dataset["kepid"].values
    
    for split_num, (train_map, test_map) in enumerate(
        cv_splitter.split(dataset, y=targets, groups=group_ids)
    ):
        yield split_num, train_map, test_map


if __name__ == "__main__":
    from src.features import fetch_processed_dataset

    data_frame = fetch_processed_dataset().reset_index(drop=True)

    for run_seed in RANDOM_SEEDS:
        for fold_id, train_mask, test_mask in generate_cv_indices(data_frame, run_seed):
            # Check for group leakage using distinct variable references
            training_groups = set(data_frame.iloc[train_mask]["kepid"])
            testing_groups = set(data_frame.iloc[test_mask]["kepid"])
            intersecting = training_groups.intersection(testing_groups)
            
            assert len(intersecting) == 0, "A star is in both train and test!"
            
            # Re-architected log output format
            test_target_mean = data_frame.iloc[test_mask]["label"].mean()
            print(
                f"seed {run_seed} fold {fold_id}: "
                f"train {len(train_mask)}, test {len(test_mask)}, "
                f"planet share in test {test_target_mean:.2f}"
            )
            
    print("OK: no star appears in both train and test.")
