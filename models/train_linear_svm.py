
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
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
from skopt.space import Real

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "processed"
FOLD_FILE = ROOT / "results" / "training_folds.csv"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


def load_partition(partition):
    X_parts = []
    y_parts = []
    group_parts = []
    feature_names = None

    for trip_dir in sorted((DATA_DIR / partition).iterdir()):
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
            raise ValueError(f"Feature mismatch: {trip_dir.name}")

        X = X[fit_mask]
        y = y[fit_mask]

        if len(y) == 0:
            print(f"Skipping trip {trip_dir.name}")
            continue

        X_parts.append(X)
        y_parts.append(y)
        group_parts.extend([trip_dir.name] * len(y))

        print(f"Loaded {trip_dir.name}: {len(y)} rows")

    X_all = np.concatenate(X_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.int8)
    groups = np.asarray(group_parts)

    return X_all, y_all, groups, feature_names


def make_tuning_sample(X, y, groups, max_rows_per_trip=1000):
    rng = np.random.RandomState(RANDOM_STATE)
    selected = []

    for trip_id in np.unique(groups):
        indices = np.flatnonzero(groups == trip_id)
        count = min(max_rows_per_trip, len(indices))

        chosen = rng.choice(indices, count, replace=False)
        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=int)
    selected.sort()

    return X[selected], y[selected], groups[selected]


def tune_linear_svm(X, y, validation_groups):
    print("\n" + "=" * 60)
    print("LINEAR SVM: BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            dual="auto",
        )),
    ])

    search_space = {
        "svm__C": Real(0.001, 10.0, prior="log-uniform")
    }

    search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=5,
        scoring="average_precision",
        cv=GroupKFold(n_splits=N_SPLITS),
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )

    search.fit(X, y, groups=validation_groups)

    print("\nBest Linear SVM CV Average Precision:")
    print(search.best_score_)

    print("Best parameters:", search.best_params_)

    return search


def evaluate_model(model, X, y, name, use_decision_function=False):
    if use_decision_function:
        scores = model.decision_function(X)
        predictions = model.predict(X)
    else:
        scores = model.predict_proba(X)[:, 1]
        predictions = (scores >= 0.5).astype(np.int8)

    return {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
    }


def load_eligible_test():
    X_parts = []
    y_parts = []

    for trip_dir in sorted((DATA_DIR / "student_test").iterdir()):
        with np.load(trip_dir / "candidates.npz") as candidates:
            X = candidates["X"]

        with np.load(trip_dir / "targets.npz") as targets:
            y = targets["y"]
            fit_mask = targets["fit_mask"].astype(bool)

        X_parts.append(X[fit_mask])
        y_parts.append(y[fit_mask])

    return (
        np.concatenate(X_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int8),
    )


def main():
    print("=" * 60)
    print("PATHSNAP - PERSON 2 ADVANCED ML TRAINING")
    print("XGBoost + Linear SVM")
    print("=" * 60)

    # Load training data
    X_train, y_train, groups, feature_names = load_partition("train")

    print("\nTraining shape:", X_train.shape)
    print("Positive labels:", int((y_train == 1).sum()))
    print("Negative labels:", int((y_train == 0).sum()))
    print("Training trips:", len(np.unique(groups)))

    # Load predefined folds
    folds = pd.read_csv(FOLD_FILE)

    fold_mapping = dict(zip(
        folds["trip_id"].astype(str).str.zfill(8),
        folds["validation_fold"]
    ))

    missing = set(groups) - set(fold_mapping)
    if missing:
        raise ValueError(f"Missing fold assignments: {missing}")

    # Load existing XGBoost
    xgb_path = ARTIFACT_DIR / "xgboost_model.json"

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    print("\nExisting XGBoost model loaded.")

    # Create tuning sample
    X_sample, y_sample, sampled_trips = make_tuning_sample(
        X_train, y_train, groups
    )

    validation_groups = np.asarray([
        fold_mapping[trip_id] for trip_id in sampled_trips
    ])

    print("\nSVM tuning sample:", X_sample.shape)
    print("Sample trips:", len(np.unique(sampled_trips)))
    print("Validation folds:", len(np.unique(validation_groups)))

    # Tune SVM
    svm_search = tune_linear_svm(
        X_sample,
        y_sample,
        validation_groups
    )

    # Refit on full training dataset
    print("\nTraining final Linear SVM on full dataset...")

    svm_model = svm_search.best_estimator_
    svm_model.fit(X_train, y_train)

    print("Full-data Linear SVM training completed.")

    # Save model
    svm_path = ARTIFACT_DIR / "linear_svm_model.joblib"
    joblib.dump(svm_model, svm_path)

    # Load eligible student-test data
    X_test, y_test = load_eligible_test()

    print("\n=== STUDENT TEST DATA ===")
    print("Evaluation rows:", len(y_test))
    print("Positive labels:", int((y_test == 1).sum()))
    print("Negative labels:", int((y_test == 0).sum()))

    # Evaluate both models
    results = []

    xgb_metrics = evaluate_model(
        xgb_model, X_test, y_test, "XGBoost"
    )
    results.append(xgb_metrics)

    svm_metrics = evaluate_model(
        svm_model,
        X_test,
        y_test,
        "Linear SVM",
        use_decision_function=True,
    )
    results.append(svm_metrics)

    for metrics in results:
        print("\n===", metrics["model"], "===")
        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.5f}")

    # Save evaluation
    results_df = pd.DataFrame(results)

    evaluation_path = (
        ARTIFACT_DIR / "advanced_model_evaluation.csv"
    )
    results_df.to_csv(evaluation_path, index=False)

    # Save tuning information
    tuning_results = {
        "linear_svm": {
            "best_cv_average_precision": float(svm_search.best_score_),
            "best_parameters": svm_search.best_params_,
            "tuning_rows": int(len(y_sample)),
            "final_training_rows": int(len(y_train)),
            "validation_folds": N_SPLITS,
            "class_weight": "balanced",
            "scoring": "average_precision",
        },
        "xgboost": {
            "best_cv_average_precision": 0.8921825506188226,
            "training_rows": int(len(y_train)),
        },
        "feature_names": feature_names,
        "training_trips": int(len(np.unique(groups))),
        "label_type": "derived_weak_labels",
    }

    tuning_path = ARTIFACT_DIR / "advanced_model_tuning.json"

    with open(tuning_path, "w") as file:
        json.dump(tuning_results, file, indent=4, default=str)

    print("\n=== FINAL MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print("\nSaved artifacts:")
    print(xgb_path)
    print(svm_path)
    print(evaluation_path)
    print(tuning_path)

    print("\nLinear SVM training completed successfully!")


if __name__ == "__main__":
    main()
