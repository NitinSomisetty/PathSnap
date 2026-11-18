import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import run_five_model_cv as cv
from run_final_evaluation import calculate_metrics, decode_viterbi


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "results" / "processed"
RAW_TEST = ROOT / "Student_folder" / "Student folder" / "student_test"

MODEL_DIR = ROOT / "artifacts" / "xgboost_final_model"
OUTPUT = ROOT / "artifacts" / "xgboost_final_evaluation"

SEED = 42
MAX_TRAIN_ROWS = 150_000


def main():
    started = time.perf_counter()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\n========== PATHSNAP XGBOOST FINAL EVALUATION ==========")
    print("Training partition: train")
    print("Evaluation partition: student_test")
    print("Student-test data will NOT be used for fitting or tuning.")

    # 1. Load the same training data used by the RF evaluation.
    print("\n[1/6] Loading training data...")

    training_data, feature_names = cv.load_all_training_trips()

    X_all = np.concatenate(
        [training_data[f]["X"] for f in range(cv.N_FOLDS)],
        axis=0,
    )
    y_all = np.concatenate(
        [training_data[f]["y"] for f in range(cv.N_FOLDS)],
        axis=0,
    )

    if len(np.unique(y_all)) != 2:
        raise ValueError("Training data must contain both classes.")

    print("Eligible training rows:", f"{len(y_all):,}")
    print("Training positives:", f"{int(y_all.sum()):,}")
    print("Feature count:", len(feature_names))

    # 2. Reproduce the same deterministic sample as final RF.
    print("\n[2/6] Preparing identical training sample...")

    X_fit, y_fit = cv.stratified_sample(
        X_all,
        y_all,
        MAX_TRAIN_ROWS,
        SEED,
    )

    print("Training sample rows:", f"{len(y_fit):,}")
    print("Sample positives:", f"{int(y_fit.sum()):,}")

    # 3. Reproduce the exact original five-model CV XGBoost config.
    print("\n[3/6] Fitting original CV XGBoost configuration...")

    model = XGBClassifier(
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
    )

    fit_started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - fit_started

    model_path = MODEL_DIR / "xgboost_baseline.joblib"
    joblib.dump(model, model_path)

    with (MODEL_DIR / "feature_names.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(feature_names, f, indent=4)

    parameters = model.get_params()

    training_metadata = {
        "model": "XGBoost baseline from five-model CV",
        "xgboost_version": __import__("xgboost").__version__,
        "training_partition": "train",
        "student_test_used_for_fitting": False,
        "eligible_training_rows": int(len(y_all)),
        "final_sample_rows": int(len(y_fit)),
        "final_sample_positive_rows": int(y_fit.sum()),
        "sampling_method": "Existing stratified_sample function",
        "sampling_seed": SEED,
        "max_training_rows": MAX_TRAIN_ROWS,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "parameters": parameters,
        "fit_seconds": fit_seconds,
    }

    with (MODEL_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(training_metadata, f, indent=4, default=str)

    print("Saved model:", model_path)
    print(f"Training time: {fit_seconds:.2f} seconds")

    del X_all, y_all, X_fit, y_fit, training_data
    gc.collect()

    # 4. Evaluate the same ten student-test trips.
    print("\n[4/6] Evaluating student-test trips...")

    test_root = PROCESSED / "student_test"
    trip_metrics = []
    prediction_frames = []
    route_frames = []

    total_test_rows = 0
    total_eligible_rows = 0
    total_inference_seconds = 0.0
    skipped_trips = []

    for folder in sorted(p for p in test_root.iterdir() if p.is_dir()):
        trip_id = folder.name

        with np.load(folder / "candidates.npz") as c:
            X = c["X"].astype(np.float32)
            observation_index = c["observation_index"].astype(np.int64)
            arc_id = c["arc_id"].astype(np.int64)
            names = c["feature_names"].tolist()

        with np.load(folder / "targets.npz") as t:
            y = t["y"].astype(np.int8)
            fit_mask = t["fit_mask"].astype(bool)

        if names != feature_names:
            raise ValueError(f"Feature order mismatch in trip {trip_id}")

        if not (
            len(X) == len(y) == len(fit_mask)
            == len(observation_index) == len(arc_id)
        ):
            raise ValueError(f"Array length mismatch in trip {trip_id}")

        observations = pd.read_csv(
            folder / "observations.csv.gz"
        )

        raw_trip = RAW_TEST / trip_id

        arcs = np.loadtxt(
            raw_trip / f"{trip_id}.arcs",
            dtype=np.int64,
            ndmin=2,
        )

        if len(arc_id) and (
            arc_id.min() < 0 or arc_id.max() >= len(arcs)
        ):
            raise ValueError(f"Candidate arc outside graph in {trip_id}")

        inference_started = time.perf_counter()
        probabilities = model.predict_proba(X)[:, 1]
        inference_seconds = time.perf_counter() - inference_started

        total_inference_seconds += inference_seconds
        total_test_rows += len(y)
        total_eligible_rows += int(fit_mask.sum())

        prediction_frames.append(pd.DataFrame({
            "trip_id": trip_id,
            "observation_index": observation_index,
            "arc_id": arc_id,
            "weak_label": y,
            "fit_mask": fit_mask,
            "probability": probabilities,
            "predicted_label": (probabilities >= 0.5).astype(np.int8),
        }))

        eligible_y = y[fit_mask]
        eligible_probabilities = probabilities[fit_mask]

        if len(eligible_y) == 0:
            skipped_trips.append(trip_id)
            print(
                f"{trip_id}: no eligible labels; "
                "excluded from classification metrics"
            )
            continue

        metrics = calculate_metrics(
            eligible_y,
            eligible_probabilities,
        )

        eligible_frame = pd.DataFrame({
            "observation_index": observation_index[fit_mask],
            "y": eligible_y,
            "score": eligible_probabilities,
        })

        selected = eligible_frame.loc[
            eligible_frame.groupby(
                "observation_index", sort=False
            )["score"].idxmax()
        ]

        group_labels = eligible_frame.groupby(
            "observation_index", sort=False
        )["y"].max()

        metrics["observation_count"] = int(len(group_labels))
        metrics["observation_top1_accuracy"] = float(selected["y"].mean())
        metrics["no_positive_candidate_rate"] = float(
            (group_labels == 0).mean()
        )
        metrics["trip_id"] = trip_id
        metrics["inference_seconds"] = inference_seconds
        metrics["inference_candidates_per_second"] = (
            float(len(X) / inference_seconds)
            if inference_seconds > 0 else None
        )

        # Same graph-aware decoder and fixed parameters as RF.
        decoded = decode_viterbi(
            observation_index,
            arc_id,
            probabilities,
            observations,
            arcs,
        )

        target_by_observation = {}
        positive_indices = np.flatnonzero((y == 1) & fit_mask)

        for idx in positive_indices:
            obs = int(observation_index[idx])
            target_by_observation.setdefault(obs, set()).add(
                int(arc_id[idx])
            )

        evaluated = []
        correct = []

        for obs, target_arcs in target_by_observation.items():
            if obs < 0 or obs >= len(decoded):
                continue

            predicted_arc = int(decoded[obs])
            if predicted_arc < 0:
                continue

            evaluated.append(obs)
            correct.append(predicted_arc in target_arcs)

        route_accuracy = (
            float(np.mean(correct)) if correct else None
        )

        metrics["viterbi_weak_label_accuracy"] = route_accuracy
        metrics["viterbi_evaluated_observations"] = len(evaluated)

        route_frames.append(pd.DataFrame({
            "trip_id": trip_id,
            "observation_index": np.arange(len(observations)),
            "decoded_arc_id": decoded,
            "has_weak_reference_candidate": [
                i in target_by_observation
                for i in range(len(observations))
            ],
        }))

        trip_metrics.append(metrics)

        print(
            f"{trip_id}: eligible={metrics['candidate_count']:,}, "
            f"F1={metrics['f1']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"Top-1={metrics['observation_top1_accuracy']:.4f}, "
            f"Viterbi={route_accuracy if route_accuracy is not None else 'N/A'}, "
            f"inference={inference_seconds:.3f}s"
        )

    # 5. Save all outputs.
    print("\n[5/6] Saving XGBoost evaluation outputs...")

    if not trip_metrics:
        raise ValueError("No student-test trips had eligible labels.")

    per_trip = pd.DataFrame(trip_metrics)
    per_trip.to_csv(OUTPUT / "per_trip_metrics.csv", index=False)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(
        OUTPUT / "student_test_candidate_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    if route_frames:
        pd.concat(route_frames, ignore_index=True).to_csv(
            OUTPUT / "student_test_viterbi_predictions.csv",
            index=False,
        )

    eligible_predictions = all_predictions[
        all_predictions["fit_mask"]
    ].copy()

    aggregate = calculate_metrics(
        eligible_predictions["weak_label"].to_numpy(),
        eligible_predictions["probability"].to_numpy(),
    )

    grouped = eligible_predictions.groupby(
        ["trip_id", "observation_index"],
        sort=False,
    )

    best_indices = grouped["probability"].idxmax()
    selected = eligible_predictions.loc[best_indices]
    group_labels = grouped["weak_label"].max()

    aggregate["observation_count"] = int(len(group_labels))
    aggregate["observation_top1_accuracy"] = float(
        selected["weak_label"].mean()
    )
    aggregate["no_positive_candidate_rate"] = float(
        (group_labels == 0).mean()
    )

    viterbi_values = per_trip[
        "viterbi_weak_label_accuracy"
    ].dropna()

    aggregate["mean_trip_viterbi_weak_label_accuracy"] = (
        float(viterbi_values.mean())
        if len(viterbi_values) else None
    )

    aggregate["evaluated_trip_count"] = int(len(per_trip))
    aggregate["student_test_trip_count"] = int(
        len(list(p for p in test_root.iterdir() if p.is_dir()))
    )
    aggregate["total_candidate_rows"] = int(total_test_rows)
    aggregate["eligible_candidate_rows"] = int(total_eligible_rows)
    aggregate["excluded_trips_without_eligible_labels"] = skipped_trips
    aggregate["total_inference_seconds"] = total_inference_seconds
    aggregate["inference_candidates_per_second"] = (
        float(total_test_rows / total_inference_seconds)
        if total_inference_seconds > 0 else None
    )

    aggregate["label_caveat"] = (
        "Evaluation targets are derived weak labels aligned from "
        "reference routes, not independently supplied GPS ground truth."
    )

    with (OUTPUT / "aggregate_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate, f, indent=4)

    run_metadata = {
        "model_file": str(model_path.relative_to(ROOT)),
        "training_metadata_file": str(
            (MODEL_DIR / "training_metadata.json").relative_to(ROOT)
        ),
        "model_selection_status": (
            "Baseline XGBoost evaluated for development-stage comparison"
        ),
        "student_test_used_for_model_fitting": False,
        "student_test_used_for_hyperparameter_tuning": False,
        "candidate_classification_threshold": 0.5,
        "viterbi_gap_seconds": 30,
        "viterbi_switch_penalty": 2.0,
        "viterbi_disconnected_penalty": 8.0,
        "viterbi_note": (
            "Same fixed graph-aware heuristic decoder as the RF evaluation."
        ),
        "excluded_trips": skipped_trips,
        "fit_seconds": fit_seconds,
        "total_inference_seconds": total_inference_seconds,
        "runtime_seconds": time.perf_counter() - started,
    }

    with (OUTPUT / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_metadata, f, indent=4)

    # 6. Print summary.
    print("\n[6/6] XGBOOST EVALUATION COMPLETE")
    print("=" * 65)

    print("\nAggregate candidate metrics:")
    for key, value in aggregate.items():
        print(f"{key}: {value}")

    print("\nPer-trip metrics:")
    print(per_trip.round(4).to_string(index=False))

    print("\nSaved outputs:")
    print("Model:", model_path)
    print("Aggregate:", OUTPUT / "aggregate_metrics.json")
    print("Per trip:", OUTPUT / "per_trip_metrics.csv")
    print("Candidate predictions:", OUTPUT / "student_test_candidate_predictions.csv.gz")
    print("Viterbi predictions:", OUTPUT / "student_test_viterbi_predictions.csv")
    print("Metadata:", OUTPUT / "run_metadata.json")


if __name__ == "__main__":
    main()
