
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "processed"
FOLD_FILE = ROOT / "results" / "training_folds.csv"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TREE_SEARCH_ITERATIONS = 8


def load_partition(partition):
    """Load eligible candidate features and labels."""

    X_parts = []
    y_parts = []
    group_parts = []
    feature_names = None

    trip_dirs = sorted((DATA_DIR / partition).iterdir())

    for trip_dir in trip_dirs:
        candidate_file = trip_dir / "candidates.npz"
        target_file = trip_dir / "targets.npz"

        if not candidate_file.exists() or not target_file.exists():
            continue

        with np.load(candidate_file) as candidates:
            X = candidates["X"]
            names = candidates["feature_names"].tolist()

        with np.load(target_file) as targets:
            y = targets["y"]
            fit_mask = targets["fit_mask"].astype(bool)

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature mismatch in trip {trip_dir.name}"
            )

        X = X[fit_mask]
        y = y[fit_mask]

        if len(y) == 0:
            print(f"Skipping trip {trip_dir.name}: no fitting rows")
            continue

        X_parts.append(X)
        y_parts.append(y)
        group_parts.extend([trip_dir.name] * len(y))

        print(
            f"Loaded {partition} trip {trip_dir.name}: "
            f"{len(y)} fitting rows"
        )

    if not X_parts:
        raise ValueError(f"No fitting data found for {partition}")

    X_all = np.concatenate(X_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.int8)
    groups = np.asarray(group_parts)

    return X_all, y_all, groups, feature_names


def make_tuning_sample(X, y, groups, max_rows_per_trip=1000):
    """Take a reproducible sample while retaining trip identities."""

    rng = np.random.RandomState(RANDOM_STATE)
    selected_indices = []

    for trip_id in np.unique(groups):
        trip_indices = np.flatnonzero(groups == trip_id)

        sample_size = min(max_rows_per_trip, len(trip_indices))

        chosen = rng.choice(
            trip_indices,
            size=sample_size,
            replace=False,
        )

        selected_indices.extend(chosen.tolist())

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_indices.sort()

    return (
        X[selected_indices],
        y[selected_indices],
        groups[selected_indices],
    )


def evaluate_model(model, X, y, model_name):
    """Calculate classification metrics."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(
            y, probabilities
        ),
    }


def tune_decision_tree(X, y, validation_groups):
    """Tune Decision Tree hyperparameters using Bayesian optimization."""

    print("\n" + "=" * 60)
    print("DECISION TREE: BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    print("Tuning rows:", len(y))
    print("Validation folds:", len(np.unique(validation_groups)))

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    )

    search_space = {
        "max_depth": Integer(4, 30),
        "min_samples_split": Integer(2, 30),
        "min_samples_leaf": Integer(1, 20),
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
        "max_features": Real(0.5, 1.0),
        "ccp_alpha": Real(1e-6, 1e-2, prior="log-uniform"),
    }

    search = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=TREE_SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=GroupKFold(n_splits=N_SPLITS),
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    search.fit(X, y, groups=validation_groups)

    print(
        "\nBest Decision Tree CV Average Precision:",
        search.best_score_,
    )

    print("Best Decision Tree parameters:")
    print(search.best_params_)

    return search


def load_eligible_student_test():
    """Load only student-test rows allowed by fit_mask."""

    X_parts = []
    y_parts = []
    feature_names = None

    for trip_dir in sorted((DATA_DIR / "student_test").iterdir()):
        candidate_file = trip_dir / "candidates.npz"
        target_file = trip_dir / "targets.npz"

        if not candidate_file.exists() or not target_file.exists():
            continue

        with np.load(candidate_file) as candidates:
            X = candidates["X"]
            names = candidates["feature_names"].tolist()

        with np.load(target_file) as targets:
            y = targets["y"]
            fit_mask = targets["fit_mask"].astype(bool)

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature mismatch in student-test trip {trip_dir.name}"
            )

        X_parts.append(X[fit_mask])
        y_parts.append(y[fit_mask])

    X_test = np.concatenate(X_parts).astype(np.float32)
    y_test = np.concatenate(y_parts).astype(np.int8)

    return X_test, y_test


def main():
    print("=" * 60)
    print("PATHSNAP - PERSON 2 ADVANCED ML TRAINING")
    print("XGBoost + Decision Tree")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------

    X_train, y_train, groups, feature_names = load_partition("train")

    print("\n=== TRAINING DATA ===")
    print("Feature matrix:", X_train.shape)
    print("Positive labels:", int((y_train == 1).sum()))
    print("Negative labels:", int((y_train == 0).sum()))
    print("Unique training trips:", len(np.unique(groups)))

    # --------------------------------------------------
    # 2. Load predefined validation folds
    # --------------------------------------------------

    folds = pd.read_csv(FOLD_FILE)

    fold_mapping = dict(
        zip(
            folds["trip_id"].astype(str).str.zfill(8),
            folds["validation_fold"],
        )
    )

    missing = set(groups) - set(fold_mapping)

    if missing:
        raise ValueError(f"Trips missing from fold file: {missing}")

    validation_groups = np.asarray([
        fold_mapping[trip_id] for trip_id in groups
    ])

    if len(np.unique(validation_groups)) != N_SPLITS:
        raise ValueError("Expected exactly five predefined validation folds.")

    # --------------------------------------------------
    # 3. Load existing XGBoost model
    # --------------------------------------------------

    xgb_path = ARTIFACT_DIR / "xgboost_model.json"

    if not xgb_path.exists():
        raise FileNotFoundError(
            f"Saved XGBoost model not found: {xgb_path}"
        )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    print("\nExisting XGBoost model loaded successfully.")

    # --------------------------------------------------
    # 4. Prepare Decision Tree tuning sample
    # --------------------------------------------------

    X_tree_sample, y_tree_sample, sampled_trip_ids = (
        make_tuning_sample(X_train, y_train, groups)
    )

    tree_validation_groups = np.asarray([
        fold_mapping[trip_id] for trip_id in sampled_trip_ids
    ])

    print("\n=== DECISION TREE TUNING SAMPLE ===")
    print("Sample feature matrix:", X_tree_sample.shape)
    print("Positive labels:", int((y_tree_sample == 1).sum()))
    print("Negative labels:", int((y_tree_sample == 0).sum()))
    print("Original trips:", len(np.unique(sampled_trip_ids)))
    print("Validation folds:", len(np.unique(tree_validation_groups)))

    # --------------------------------------------------
    # 5. Tune Decision Tree
    # --------------------------------------------------

    tree_search = tune_decision_tree(
        X_tree_sample,
        y_tree_sample,
        tree_validation_groups,
    )

    # --------------------------------------------------
    # 6. Refit Decision Tree on full training data
    # --------------------------------------------------

    print("\nRefitting selected Decision Tree on full training data...")

    tree_model = tree_search.best_estimator_
    tree_model.fit(X_train, y_train)

    print("Full-data Decision Tree training completed.")

    # --------------------------------------------------
    # 7. Save Decision Tree
    # --------------------------------------------------

    tree_path = ARTIFACT_DIR / "decision_tree_model.joblib"

    joblib.dump(tree_model, tree_path)

    print("Decision Tree saved:", tree_path)

    # --------------------------------------------------
    # 8. Load eligible student-test data
    # --------------------------------------------------

    X_test, y_test = load_eligible_student_test()

    print("\n=== ELIGIBLE STUDENT TEST DATA ===")
    print("Evaluation rows:", len(y_test))
    print("Positive labels:", int((y_test == 1).sum()))
    print("Negative labels:", int((y_test == 0).sum()))

    # --------------------------------------------------
    # 9. Evaluate both models
    # --------------------------------------------------

    results = []

    for name, model in [
        ("XGBoost", xgb_model),
        ("Decision Tree", tree_model),
    ]:
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            name,
        )

        results.append(metrics)

        print("\n=== STUDENT TEST:", name, "===")

        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.5f}")

    results_df = pd.DataFrame(results)

    evaluation_path = (
        ARTIFACT_DIR / "advanced_model_evaluation.csv"
    )

    results_df.to_csv(evaluation_path, index=False)

    # --------------------------------------------------
    # 10. Save tuning information
    # --------------------------------------------------

    tuning_results = {
        "xgboost": {
            "best_cv_average_precision": 0.8921825506188226,
            "best_parameters": {
                "colsample_bytree": 0.9866433958929786,
                "learning_rate": 0.10038098207698234,
                "max_depth": 7,
                "min_child_weight": 5,
                "n_estimators": 176,
                "reg_lambda": 0.26474549244554496,
                "scale_pos_weight": 1.0005494567740614,
                "subsample": 0.8112056462988113,
            },
            "training_rows": int(len(y_train)),
        },
        "decision_tree": {
            "best_cv_average_precision": float(tree_search.best_score_),
            "best_parameters": tree_search.best_params_,
            "tuning_rows": int(len(y_tree_sample)),
            "final_training_rows": int(len(y_train)),
            "tuning_method": (
                "Per-trip random sample with predefined "
                "five-fold trip-level CV"
            ),
            "search_iterations": TREE_SEARCH_ITERATIONS,
        },
        "feature_names": feature_names,
        "training_trips": int(len(np.unique(groups))),
        "validation_folds": N_SPLITS,
        "label_type": "derived_weak_labels",
    }

    tuning_path = ARTIFACT_DIR / "advanced_model_tuning.json"

    with open(tuning_path, "w") as file:
        json.dump(tuning_results, file, indent=4, default=str)

    # --------------------------------------------------
    # 11. Display final comparison
    # --------------------------------------------------

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print("\nSaved artifacts:")
    print(xgb_path)
    print(tree_path)
    print(evaluation_path)
    print(tuning_path)

    print("\nAdvanced ML training completed successfully!")


if __name__ == "__main__":
    main()
