from pathlib import Path
import time
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "processed" / "train"
FOLD_FILE = ROOT / "results" / "training_folds.csv"
OUTPUT_DIR = ROOT / "artifacts" / "five_model_cv"

SEED = 42
N_FOLDS = 5

MAX_TRAIN_ROWS = 150_000
KNN_TRAIN_ROWS = 5_000
SVM_ROWS_PER_TRIP = 1_000

SVM_C = 0.060163078295899286

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stratified_sample(X, y, max_rows, seed):
    if len(y) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    selected = []

    for cls, count in zip(classes, counts):
        n = max(1, int(round(max_rows * count / len(y))))
        indices = np.flatnonzero(y == cls)

        selected.extend(
            rng.choice(
                indices,
                size=min(n, len(indices)),
                replace=False
            ).tolist()
        )

    selected = np.asarray(selected, dtype=np.int64)

    if len(selected) > max_rows:
        selected = rng.choice(
            selected,
            size=max_rows,
            replace=False
        )

    rng.shuffle(selected)

    return X[selected], y[selected]


def sample_svm_per_trip(X, y, trip_ids, max_per_trip, seed):
    rng = np.random.default_rng(seed)
    selected = []

    for trip_id in np.unique(trip_ids):
        indices = np.flatnonzero(trip_ids == trip_id)
        count = min(max_per_trip, len(indices))

        chosen = rng.choice(indices, count, replace=False)
        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    selected.sort()

    return X[selected], y[selected]


def load_all_training_trips():
    folds_df = pd.read_csv(FOLD_FILE)

    required_columns = {"trip_id", "validation_fold"}
    if not required_columns.issubset(folds_df.columns):
        raise ValueError(
            f"Fold file must contain {required_columns}"
        )

    fold_map = {
        str(int(row.trip_id)).zfill(8): int(row.validation_fold)
        for row in folds_df.itertuples()
    }

    data = {
        fold: {
            "X": [],
            "y": [],
            "trip": [],
            "observation": [],
            "arc": [],
        }
        for fold in range(N_FOLDS)
    }

    feature_names = None
    usable_trip_count = 0

    trip_dirs = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
    )

    print("Training trip folders:", len(trip_dirs))

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name

        if trip_id not in fold_map:
            raise ValueError(
                f"Missing fold assignment for {trip_id}"
            )

        candidate_file = trip_dir / "candidates.npz"
        target_file = trip_dir / "targets.npz"

        if not candidate_file.exists() or not target_file.exists():
            raise FileNotFoundError(
                f"Missing NPZ files for {trip_id}"
            )

        with np.load(candidate_file, allow_pickle=False) as c:
            X = c["X"]
            observations = c["observation_index"]
            arcs = c["arc_id"]
            names = c["feature_names"].tolist()

        with np.load(target_file, allow_pickle=False) as t:
            y = t["y"]
            mask = t["fit_mask"].astype(bool)

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature mismatch in trip {trip_id}"
            )

        if not (
            len(X) == len(y)
            == len(mask)
            == len(observations)
            == len(arcs)
        ):
            raise ValueError(
                f"Array length mismatch in trip {trip_id}"
            )

        X = X[mask].astype(np.float32)
        y = y[mask].astype(np.int8)
        observations = observations[mask]
        arcs = arcs[mask]

        if len(y) == 0:
            continue

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite feature values in trip {trip_id}"
            )

        fold = fold_map[trip_id]

        data[fold]["X"].append(X)
        data[fold]["y"].append(y)
        data[fold]["trip"].append(
            np.full(len(y), trip_id, dtype="U8")
        )
        data[fold]["observation"].append(observations)
        data[fold]["arc"].append(arcs)

        usable_trip_count += 1

    for fold in range(N_FOLDS):
        if not data[fold]["y"]:
            raise ValueError(f"Fold {fold} has no usable data")

        for key in data[fold]:
            data[fold][key] = np.concatenate(
                data[fold][key],
                axis=0
            )

    print("Usable training trips:", usable_trip_count)
    print("Features:", len(feature_names))
    print("Feature names:", feature_names)

    for fold in range(N_FOLDS):
        print(
            f"Fold {fold}: "
            f"{len(data[fold]['y']):,} validation rows | "
            f"{len(np.unique(data[fold]['trip']))} trips"
        )

    return data, feature_names


def make_models():
    return [
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
            "KNN",
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
        (
            "XGBoost",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=2,
                colsample_bytree=0.9866433958929786,
                learning_rate=0.10038098207698234,
                max_depth=7,
                min_child_weight=5,
                n_estimators=176,
                reg_lambda=0.26474549244554496,
                scale_pos_weight=1.0005494562989786,
                subsample=0.8112056462988113,
            ),
            MAX_TRAIN_ROWS,
        ),
        (
            "Linear SVM",
            make_pipeline(
                StandardScaler(),
                LinearSVC(
                    C=SVM_C,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=SEED,
                    dual="auto",
                ),
            ),
            None,
        ),
    ]


def calculate_metrics(y_true, scores, predictions):
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(
            y_true, scores
        ),
    }

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    result.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

    return result


def calculate_observation_metrics(
    y_true,
    scores,
    trip_ids,
    observation_ids,
):
    frame = pd.DataFrame({
        "trip_id": trip_ids,
        "observation_index": observation_ids,
        "y": y_true,
        "score": scores,
    })

    group_columns = ["trip_id", "observation_index"]

    best_indices = (
        frame.groupby(group_columns, sort=False)["score"].idxmax()
    )

    selected = frame.loc[best_indices]

    top1 = float(selected["y"].mean())

    group_labels = frame.groupby(
        group_columns, sort=False
    )["y"].max()

    no_positive_rate = float((group_labels == 0).mean())

    return {
        "observation_count": int(len(group_labels)),
        "observation_top1_accuracy": top1,
        "no_positive_candidate_rate": no_positive_rate,
    }


def main():
    print("\n========== PATHSNAP FIVE-MODEL CROSS-VALIDATION ==========")
    print("Five-fold trip-level CV")
    print("Decision Tree excluded")
    print("Student-test data will not be loaded")
    print("Evaluator-test data will not be loaded")

    data, feature_names = load_all_training_trips()

    results = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 65)
        print(f"VALIDATION FOLD {fold}")
        print("=" * 65)

        validation = data[fold]

        X_val = validation["X"]
        y_val = validation["y"]

        val_trip = validation["trip"]
        val_observation = validation["observation"]

        train_folds = [
            f for f in range(N_FOLDS) if f != fold
        ]

        X_train = np.concatenate([
            data[f]["X"] for f in train_folds
        ])

        y_train = np.concatenate([
            data[f]["y"] for f in train_folds
        ])

        train_trip = np.concatenate([
            data[f]["trip"] for f in train_folds
        ])

        print("Training rows available:", f"{len(y_train):,}")
        print("Validation rows:", f"{len(y_val):,}")
        print("Validation trips:", len(np.unique(val_trip)))

        for model_name, model, train_limit in make_models():
            print(f"\n--- {model_name} ---")

            if model_name == "Linear SVM":
                X_fit, y_fit = sample_svm_per_trip(
                    X_train,
                    y_train,
                    train_trip,
                    SVM_ROWS_PER_TRIP,
                    SEED + fold,
                )
            else:
                X_fit, y_fit = stratified_sample(
                    X_train,
                    y_train,
                    train_limit,
                    SEED + fold,
                )

            print("Rows used for fitting:", f"{len(y_fit):,}")

            start = time.perf_counter()
            model.fit(X_fit, y_fit)
            training_seconds = time.perf_counter() - start

            start = time.perf_counter()

            if model_name == "Linear SVM":
                scores = model.decision_function(X_val)
                predictions = model.predict(X_val)
            else:
                scores = model.predict_proba(X_val)[:, 1]
                predictions = (scores >= 0.5).astype(np.int8)

            prediction_seconds = time.perf_counter() - start

            metrics = calculate_metrics(
                y_val,
                scores,
                predictions,
            )

            observation_metrics = calculate_observation_metrics(
                y_val,
                scores,
                val_trip,
                val_observation,
            )

            record = {
                "model": model_name,
                "fold": fold,
                "training_rows": len(y_fit),
                "validation_rows": len(y_val),
                "training_trips": len(np.unique(train_trip)),
                "validation_trips": len(np.unique(val_trip)),
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                **metrics,
                **observation_metrics,
            }

            results.append(record)

            print(
                f"F1: {metrics['f1']:.4f} | "
                f"AP: {metrics['average_precision']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            print(
                f"Top-1: "
                f"{observation_metrics['observation_top1_accuracy']:.4f} | "
                f"No-positive rate: "
                f"{observation_metrics['no_positive_candidate_rate']:.4f}"
            )

            print(
                f"Training: {training_seconds:.2f}s | "
                f"Inference: {prediction_seconds:.2f}s"
            )

            del model, X_fit, y_fit, scores, predictions

        print(f"\nCompleted fold {fold}")

    results_df = pd.DataFrame(results)

    fold_file = OUTPUT_DIR / "fold_metrics.csv"
    results_df.to_csv(fold_file, index=False)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "observation_top1_accuracy",
        "no_positive_candidate_rate",
        "training_seconds",
        "prediction_seconds",
    ]

    summary = results_df.groupby("model")[metric_columns].agg(
        ["mean", "std"]
    )

    summary_file = OUTPUT_DIR / "model_summary.csv"
    summary.to_csv(summary_file)

    metadata = {
        "experiment": "PathSnap five-model five-fold CV",
        "seed": SEED,
        "folds": N_FOLDS,
        "usable_training_trips": 70,
        "models": [
            "Logistic Regression",
            "KNN",
            "Random Forest",
            "XGBoost",
            "Linear SVM",
        ],
        "excluded_model": "Decision Tree",
        "feature_names": feature_names,
        "training_limits": {
            "logistic_regression": MAX_TRAIN_ROWS,
            "knn": KNN_TRAIN_ROWS,
            "random_forest": MAX_TRAIN_ROWS,
            "xgboost": MAX_TRAIN_ROWS,
            "linear_svm_per_trip": SVM_ROWS_PER_TRIP,
        },
        "linear_svm_C": SVM_C,
        "validation": "Complete fit_mask-eligible candidate rows",
        "test_partitions_loaded": [],
    }

    with (OUTPUT_DIR / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 65)
    print("FIVE-MODEL CROSS-VALIDATION COMPLETE")
    print("=" * 65)

    print("\nFOLD RESULTS:")
    print(results_df.round(4).to_string(index=False))

    print("\nMEAN AND STANDARD DEVIATION:")
    print(summary.round(4).to_string())

    print("\nSaved results:")
    print(fold_file)
    print(summary_file)
    print(OUTPUT_DIR / "run_metadata.json")


if __name__ == "__main__":
    main()
