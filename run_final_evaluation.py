import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import run_five_model_cv as cv


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "results" / "processed"
RAW_TEST = ROOT / "Student_folder" / "Student folder" / "student_test"
BAYESIAN = ROOT / "artifacts" / "bayesian_optimization"
OUTPUT = ROOT / "artifacts" / "final_evaluation"
MODEL_DIR = ROOT / "artifacts" / "random_forest_comparison"

SEED = 42
MAX_TRAIN_ROWS = 150_000
GAP_SECONDS = 30
SWITCH_PENALTY = 2.0
DISCONNECTED_PENALTY = 8.0


def calculate_metrics(y, probability):
    prediction = (probability >= 0.5).astype(np.int8)

    result = {
        "candidate_count": int(len(y)),
        "positive_count": int(np.sum(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)),
    }

    result["roc_auc"] = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2 else None
    )
    return result


def decode_viterbi(observation_index, arc_id, probability,
                   observations, arcs):
    """
    Graph-aware Viterbi decoding.
    Candidate probabilities are emissions.
    Same-arc, connected-arc and disconnected-arc transitions
    receive fixed penalties.
    """
    n_observations = len(observations)
    output = np.full(n_observations, -1, dtype=np.int64)

    groups = [
        np.flatnonzero(observation_index == i)
        for i in range(n_observations)
    ]

    start = 0

    while start < n_observations:
        if len(groups[start]) == 0:
            start += 1
            continue

        end = start + 1

        while end < n_observations:
            if len(groups[end]) == 0:
                break

            dt = observations.iloc[end]["dt_s"]
            if not np.isfinite(dt) or float(dt) > GAP_SECONDS:
                break

            end += 1

        scores = []
        backpointers = []

        for i in range(start, end):
            ids = groups[i]
            emission = np.log(np.clip(probability[ids], 1e-8, 1.0))

            if i == start:
                scores.append(emission)
                backpointers.append(None)
                continue

            previous_arcs = arc_id[groups[i - 1]]
            current_arcs = arc_id[ids]

            same = (
                previous_arcs[:, None] == current_arcs[None, :]
            )

            connected = (
                arcs[previous_arcs, 1][:, None]
                == arcs[current_arcs, 0][None, :]
            )

            transition = np.where(
                same,
                0.0,
                np.where(
                    connected,
                    -SWITCH_PENALTY,
                    -DISCONNECTED_PENALTY,
                ),
            )

            table = scores[-1][:, None] + transition
            best_previous = np.argmax(table, axis=0)

            scores.append(
                emission + table[
                    best_previous,
                    np.arange(len(ids))
                ]
            )
            backpointers.append(best_previous)

        state = int(np.argmax(scores[-1]))

        for i in range(end - 1, start - 1, -1):
            output[i] = int(arc_id[groups[i]][state])

            if i > start:
                state = int(backpointers[i - start][state])

        start = end

    return output


def main():
    started = time.perf_counter()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== PATHSNAP FINAL MODEL EVALUATION ==========")
    print("Training partition: train only")
    print("Evaluation partition: student_test")
    print("Student-test data will NOT be used for fitting or tuning.")

    # ---------------------------------------------------------
    # 1. Load eligible training data using the established CV loader
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Reproduce the established stratified sampling approach
    # ---------------------------------------------------------
    print("\n[2/6] Preparing final training sample...")

    X_fit, y_fit = cv.stratified_sample(
        X_all,
        y_all,
        MAX_TRAIN_ROWS,
        SEED,
    )

    print("Final model training rows:", f"{len(y_fit):,}")
    print("Sample positives:", f"{int(y_fit.sum()):,}")

    # ---------------------------------------------------------
    # 3. Fit the Bayesian-optimised Random Forest
    # ---------------------------------------------------------
    print("\n[3/6] Fitting tuned Random Forest...")

    parameter_file = BAYESIAN / "best_params.json"
    if not parameter_file.exists():
        raise FileNotFoundError(parameter_file)

    with parameter_file.open("r", encoding="utf-8") as f:
        bayesian_result = json.load(f)

    params = bayesian_result["best_parameters"]

    model = RandomForestClassifier(
        **params,
        bootstrap=True,
        n_jobs=-1,
        random_state=SEED,
    )

    fit_started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - fit_started

    model_path = MODEL_DIR / "random_forest_tuned.joblib"
    joblib.dump(model, model_path)

    with (MODEL_DIR / "feature_names.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(feature_names, f, indent=4)

    training_metadata = {
        "model": "Bayesian-optimised Random Forest",
        "training_partition": "train",
        "student_test_used_for_fitting": False,
        "training_trip_count": int(
            sum(len(training_data[f]["y"]) > 0 for f in range(cv.N_FOLDS))
        ),
        "eligible_training_rows": int(len(y_all)),
        "final_sample_rows": int(len(y_fit)),
        "final_sample_positive_rows": int(y_fit.sum()),
        "sampling_method": "Existing stratified_sample function",
        "sampling_seed": SEED,
        "max_training_rows": MAX_TRAIN_ROWS,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "bayesian_best_trial": bayesian_result["best_trial"],
        "bayesian_objective": bayesian_result["objective"],
        "bayesian_best_parameters": params,
        "fit_seconds": fit_seconds,
    }

    with (MODEL_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(training_metadata, f, indent=4)

    print("Saved model:", model_path)
    print(f"Fit time: {fit_seconds:.2f} seconds")

    del X_all, y_all, X_fit, y_fit, training_data
    gc.collect()

    # ---------------------------------------------------------
    # 4. Predict student-test candidates and calculate metrics
    # ---------------------------------------------------------
    print("\n[4/6] Evaluating student-test trips...")

    test_root = PROCESSED / "student_test"
    trip_metrics = []
    prediction_frames = []
    route_frames = []

    total_test_rows = 0
    total_eligible_rows = 0
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

        # Predict all candidates so the sequence decoder can use
        # the complete candidate set. Labels are evaluated only
        # where fit_mask is true.
        probabilities = model.predict_proba(X)[:, 1]

        eligible_y = y[fit_mask]
        eligible_probabilities = probabilities[fit_mask]

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

        trip_metrics.append(metrics)

        # -----------------------------------------------------
        # 5. Graph-aware decoding against positive weak labels
        # -----------------------------------------------------
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

        route_frames.append(pd.DataFrame({
            "trip_id": trip_id,
            "observation_index": np.arange(len(observations)),
            "decoded_arc_id": decoded,
            "has_weak_reference_candidate": [
                i in target_by_observation
                for i in range(len(observations))
            ],
        }))

        metrics["viterbi_weak_label_accuracy"] = route_accuracy
        metrics["viterbi_evaluated_observations"] = len(evaluated)

        print(
            f"{trip_id}: eligible={metrics['candidate_count']:,}, "
            f"F1={metrics['f1']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"Top-1={metrics['observation_top1_accuracy']:.4f}, "
            f"Viterbi={route_accuracy if route_accuracy is not None else 'N/A'}"
        )

    # ---------------------------------------------------------
    # 6. Save aggregate and per-trip evaluation artifacts
    # ---------------------------------------------------------
    print("\n[5/6] Saving evaluation outputs...")

    if not trip_metrics:
        raise ValueError("No student-test trips had eligible labels.")

    per_trip = pd.DataFrame(trip_metrics)
    per_trip.to_csv(OUTPUT / "per_trip_metrics.csv", index=False)

    all_predictions = pd.concat(
        prediction_frames, ignore_index=True
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
        "student_test_used_for_model_selection": False,
        "student_test_used_for_hyperparameter_tuning": False,
        "candidate_classification_threshold": 0.5,
        "viterbi_gap_seconds": GAP_SECONDS,
        "viterbi_switch_penalty": SWITCH_PENALTY,
        "viterbi_disconnected_penalty": DISCONNECTED_PENALTY,
        "viterbi_note": (
            "Fixed graph-aware heuristic decoder; route metric compares "
            "decoded arcs with positive weak-label candidate arcs."
        ),
        "excluded_trips": skipped_trips,
        "runtime_seconds": time.perf_counter() - started,
    }

    with (OUTPUT / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_metadata, f, indent=4)

    print("\n[6/6] FINAL EVALUATION COMPLETE")
    print("=" * 65)
    print("Aggregate candidate metrics:")
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
