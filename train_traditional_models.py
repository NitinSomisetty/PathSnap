from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "processed" / "train"
FOLD_FILE = ROOT / "results" / "training_folds.csv"
OUTPUT_DIR = ROOT / "artifacts" / "traditional_ml"

SEED = 42
MAX_TRAIN_ROWS = 150_000
KNN_TRAIN_ROWS = 5_000
VALIDATION_ROWS = 15_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stratified_sample(X, y, max_rows, seed):
    """Sample rows while approximately preserving class proportions."""
    if len(y) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    selected = []
    for cls, count in zip(classes, counts):
        proportion = count / len(y)
        n = max(1, int(round(max_rows * proportion)))
        indices = np.flatnonzero(y == cls)
        selected.extend(rng.choice(indices, size=min(n, len(indices)),
                                   replace=False).tolist())

    selected = np.array(selected, dtype=np.int64)
    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)

    rng.shuffle(selected)
    return X[selected], y[selected]


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int8)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def load_training_data():
    folds = pd.read_csv(FOLD_FILE)
    fold_map = {
        str(int(row.trip_id)).zfill(8): int(row.validation_fold)
        for row in folds.itertuples()
    }

    data_by_fold = {
        fold: {"X": [], "y": []}
        for fold in range(5)
    }

    trip_dirs = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Loading {len(trip_dirs)} training trips...")

    for index, trip_dir in enumerate(trip_dirs, start=1):
        trip_id = trip_dir.name

        if trip_id not in fold_map:
            raise ValueError(f"No fold assignment for trip {trip_id}")

        with np.load(trip_dir / "candidates.npz") as candidate_file:
            X = candidate_file["X"]

        with np.load(trip_dir / "targets.npz") as target_file:
            y = target_file["y"]
            mask = target_file["fit_mask"]

        X = X[mask]
        y = y[mask]

        fold = fold_map[trip_id]
        if len(y):
            data_by_fold[fold]["X"].append(X)
            data_by_fold[fold]["y"].append(y)

        if index % 20 == 0 or index == len(trip_dirs):
            print(f"  Loaded {index}/{len(trip_dirs)} trips")

    for fold in range(5):
        data_by_fold[fold]["X"] = np.concatenate(
            data_by_fold[fold]["X"], axis=0
        )
        data_by_fold[fold]["y"] = np.concatenate(
            data_by_fold[fold]["y"], axis=0
        )

    return data_by_fold


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("========== PATHSNAP TRADITIONAL ML ==========")
    print("Using five-fold trip-level cross-validation.")
    print("Student-test data will not be loaded.")

    data = load_training_data()
    results = []

    for fold in range(5):
        print(f"\n========== FOLD {fold} ==========")

        X_val, y_val = data[fold]["X"], data[fold]["y"]

        X_train = np.concatenate([
            data[f]["X"] for f in range(5) if f != fold
        ])
        y_train = np.concatenate([
            data[f]["y"] for f in range(5) if f != fold
        ])

        X_val, y_val = stratified_sample(
            X_val, y_val, VALIDATION_ROWS, SEED + fold
        )

        print("Training candidates available:", len(y_train))
        print("Validation candidates evaluated:", len(y_val))

        model_configs = [
            (
                "Logistic Regression",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
                MAX_TRAIN_ROWS,
            ),
            (
                "Distance-Weighted KNN",
                make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        algorithm="brute",
                        n_jobs=-1,
                    ),
                ),
                KNN_TRAIN_ROWS,
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
                MAX_TRAIN_ROWS,
            ),
        ]

        for model_name, model, train_limit in model_configs:
            print(f"\nTraining {model_name}...")

            X_fit, y_fit = stratified_sample(
                X_train, y_train, train_limit,
                SEED + fold
            )

            start = time.perf_counter()
            model.fit(X_fit, y_fit)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            probabilities = model.predict_proba(X_val)[:, 1]
            predict_seconds = time.perf_counter() - start

            metrics = calculate_metrics(y_val, probabilities)

            record = {
                "model": model_name,
                "fold": fold,
                "training_rows": len(y_fit),
                "validation_rows": len(y_val),
                "training_seconds": train_seconds,
                "prediction_seconds": predict_seconds,
                **metrics,
            }
            results.append(record)

            print(
                f"F1={metrics['f1']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )
            print(
                f"Training time={train_seconds:.2f}s | "
                f"Prediction time={predict_seconds:.2f}s"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "training_seconds",
        "prediction_seconds",
    ]

    summary = results_df.groupby("model")[metric_columns].agg(
        ["mean", "std"]
    )
    summary.to_csv(OUTPUT_DIR / "model_summary.csv")

    print("\n========== FINAL CROSS-VALIDATION SUMMARY ==========")
    print(summary.round(4).to_string())

    print("\nResults saved to:")
    print(OUTPUT_DIR / "fold_metrics.csv")
    print(OUTPUT_DIR / "model_summary.csv")
    print("\nTraditional ML experiment completed.")


if __name__ == "__main__":
    main()
