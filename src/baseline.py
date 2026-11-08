from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURES, fetch_processed_dataset
from src.splits import SEEDS, get_folds

OUTPUT_FOLDER = Path("results")
TOP_K_THRESHOLD = 100


def initialize_classifiers(seed_val):
    """Instantiates the specific baseline estimators with assigned seeds."""
    return {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed_val)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed_val, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=seed_val, eval_metric="logloss",
        ),
    }


def compute_precision_at_k(actuals, probabilities, target_k=TOP_K_THRESHOLD):
    """Calculates precision using the true values of the highest-scored instances."""
    highest_indices = np.argsort(probabilities)[::-1][:target_k]
    return np.mean(actuals[highest_indices])


def evaluate_metrics(actuals, probabilities):
    """Generates standard classification metrics based on continuous and binary labels."""
    binary_predictions = np.where(probabilities >= 0.5, 1, 0)
    
    return {
        "precision": precision_score(actuals, binary_predictions),
        "recall": recall_score(actuals, binary_predictions),
        "f1": f1_score(actuals, binary_predictions),
        "pr_auc": average_precision_score(actuals, probabilities),
        "precision_at_k": compute_precision_at_k(actuals, probabilities),
    }


def execute_model_evaluation():
    dataset = fetch_processed_dataset().reset_index(drop=True)
    features_matrix = dataset[FEATURES].values
    target_vector = dataset["label"].values

    records_list = []
    
    for active_seed in SEEDS:
        for split_id, train_idx, test_idx in get_folds(dataset, active_seed):
            
            # Unpack training and testing splits directly using index arrays
            x_train, y_train = features_matrix[train_idx], target_vector[train_idx]
            x_test, y_test = features_matrix[test_idx], target_vector[test_idx]
            
            estimators_dict = initialize_classifiers(active_seed)
            
            for algo_name, estimator in estimators_dict.items():
                estimator.fit(x_train, y_train)
                predicted_scores = estimator.predict_proba(x_test)[:, 1]
                
                performance_metrics = evaluate_metrics(y_test, predicted_scores)
                
                # Inline insertion to vary traditional dictionary update structures
                performance_metrics["model"] = algo_name
                performance_metrics["seed"] = active_seed
                performance_metrics["fold"] = split_id
                
                records_list.append(performance_metrics)
                
        print(f"seed {active_seed} done")

    # Save complete run outputs
    dataframe_runs = pd.DataFrame(records_list)
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    dataframe_runs.to_csv(OUTPUT_FOLDER / "baseline_runs.csv", index=False)

    # Process and pivot metrics summary
    evaluation_columns = ["precision", "recall", "f1", "pr_auc", "precision_at_k"]
    grouped_summary = (
        dataframe_runs.groupby("model")[evaluation_columns]
        .agg(["mean", "std"])
        .round(3)
    )
    
    grouped_summary.to_csv(OUTPUT_FOLDER / "baseline_table.csv")
    print(grouped_summary.T)
    
    return dataframe_runs, grouped_summary


if __name__ == "__main__":
    execute_model_evaluation()
