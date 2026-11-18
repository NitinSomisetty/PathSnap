import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, log_loss

from run_final_evaluation import decode_viterbi


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "results" / "processed" / "student_test"
RAW_TEST = ROOT / "Student_folder" / "Student folder" / "student_test"

MODELS = {
    "Random Forest": ROOT / "artifacts" / "final_evaluation",
    "XGBoost": ROOT / "artifacts" / "xgboost_final_evaluation",
}

OUTPUT = ROOT / "artifacts" / "route_comparison"
GAP_SECONDS = 30.0


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate_continuity(predicted, observations, arcs):
    connected = 0
    disconnected = 0
    skipped = 0

    for i in range(1, len(predicted)):
        dt = observations.iloc[i].get("dt_s", np.nan)

        if not np.isfinite(dt) or dt <= 0 or dt > GAP_SECONDS:
            continue

        previous_arc = int(predicted[i - 1])
        current_arc = int(predicted[i])

        if previous_arc < 0 or current_arc < 0:
            skipped += 1
            continue

        if previous_arc == current_arc:
            connected += 1
        elif arcs[previous_arc, 1] == arcs[current_arc, 0]:
            connected += 1
        else:
            disconnected += 1

    total = connected + disconnected

    return {
        "continuous_transitions": connected,
        "disconnected_transitions": disconnected,
        "transitions_missing_prediction": skipped,
        "evaluated_transitions": total,
        "route_continuity_rate": safe_rate(connected, total),
        "disconnected_transition_rate": safe_rate(disconnected, total),
    }


def evaluate_sequences(predicted, reference, usable, observations):
    exact = 0
    total = 0

    start = 0

    while start < len(predicted):
        end = start + 1

        while end < len(predicted):
            dt = observations.iloc[end].get("dt_s", np.nan)

            if not np.isfinite(dt) or dt <= 0 or dt > GAP_SECONDS:
                break

            end += 1

        indices = np.arange(start, end)
        indices = indices[usable[start:end]]

        if len(indices):
            total += 1

            if np.all(predicted[indices] == reference[indices]):
                exact += 1

        start = end

    return {
        "exactly_matched_sequences": exact,
        "evaluated_sequences": total,
        "sequence_exact_match_rate": safe_rate(exact, total),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_trip_results = []
    aggregate_results = {}

    print("\n========== PATHSNAP COMPLETE ROUTE COMPARISON ==========")

    for model_name, model_dir in MODELS.items():
        print(f"\n===== {model_name} =====")

        prediction_file = (
            model_dir / "student_test_candidate_predictions.csv.gz"
        )

        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)

        predictions = pd.read_csv(prediction_file, dtype={"trip_id": str})

        candidate_y = []
        candidate_prob = []

        total_usable = 0
        total_misses = 0
        classifier_correct = 0
        viterbi_correct = 0
        classifier_evaluated = 0
        viterbi_evaluated = 0

        total_continuous = 0
        total_disconnected = 0
        total_missing_transitions = 0

        total_exact_sequences = 0
        total_sequences = 0

        for folder in sorted(p for p in PROCESSED.iterdir() if p.is_dir()):
            trip_id = folder.name

            trip_predictions = predictions[
                predictions["trip_id"].astype(str) == trip_id
            ].copy()

            labels = pd.read_csv(folder / "labels.csv.gz")
            observations = pd.read_csv(folder / "observations.csv.gz")

            raw_trip = RAW_TEST / trip_id
            arcs = np.loadtxt(
                raw_trip / f"{trip_id}.arcs",
                dtype=np.int64,
                ndmin=2,
            )

            required = {
                "reference_arc_id",
                "label_usable",
                "reference_in_candidates",
            }

            if not required.issubset(labels.columns):
                raise ValueError(
                    f"Missing label columns in {trip_id}: "
                    f"{required - set(labels.columns)}"
                )

            usable = labels["label_usable"].to_numpy(dtype=bool)
            reference = labels["reference_arc_id"].to_numpy(dtype=np.int64)
            reference_in_candidates = (
                labels["reference_in_candidates"].to_numpy(dtype=bool)
            )

            if len(labels) != len(observations):
                raise ValueError(
                    f"Label/observation count mismatch in {trip_id}"
                )

            n_observations = len(observations)

            classifier_prediction = np.full(
                n_observations, -1, dtype=np.int64
            )

            for obs, group in trip_predictions.groupby(
                "observation_index", sort=False
            ):
                obs = int(obs)

                if 0 <= obs < n_observations:
                    best = group["probability"].idxmax()
                    classifier_prediction[obs] = int(
                        predictions.loc[best, "arc_id"]
                    )

            # Re-run the saved-probability Viterbi decoder for every trip.
            # This includes trips without eligible classification labels.
            with np.load(folder / "candidates.npz") as candidates:
                observation_index = candidates[
                    "observation_index"
                ].astype(np.int64)

                arc_id = candidates["arc_id"].astype(np.int64)

            probability_by_index = (
                trip_predictions.set_index(
                    trip_predictions.index
                )["probability"]
            )

            # Align probabilities to candidate order using the stored
            # prediction rows, which retain the original candidate order.
            if len(trip_predictions) != len(observation_index):
                raise ValueError(
                    f"Candidate prediction count mismatch in {trip_id}"
                )

            trip_predictions = trip_predictions.sort_index()

            probabilities = trip_predictions[
                "probability"
            ].to_numpy(dtype=float)

            viterbi_prediction = decode_viterbi(
                observation_index,
                arc_id,
                probabilities,
                observations,
                arcs,
            )

            # Candidate-level metrics use only eligible fit-mask rows.
            eligible = trip_predictions["fit_mask"].to_numpy(dtype=bool)

            if eligible.any():
                candidate_y.extend(
                    trip_predictions.loc[eligible, "weak_label"].astype(int)
                )
                candidate_prob.extend(
                    trip_predictions.loc[eligible, "probability"].astype(float)
                )

            # End-to-end route metrics use every usable reference alignment.
            usable_count = int(usable.sum())
            miss_count = int((usable & ~reference_in_candidates).sum())

            classifier_mask = usable & (classifier_prediction >= 0)
            viterbi_mask = usable & (viterbi_prediction >= 0)

            classifier_matches = (
                classifier_prediction[classifier_mask]
                == reference[classifier_mask]
            )

            viterbi_matches = (
                viterbi_prediction[viterbi_mask]
                == reference[viterbi_mask]
            )

            classifier_correct += int(classifier_matches.sum())
            classifier_evaluated += usable_count

            viterbi_correct += int(viterbi_matches.sum())
            viterbi_evaluated += usable_count

            classifier_accuracy = safe_rate(
                int(classifier_matches.sum()), usable_count
            )
            viterbi_accuracy = safe_rate(
                int(viterbi_matches.sum()), usable_count
            )

            classifier_continuity = evaluate_continuity(
                classifier_prediction, observations, arcs
            )

            viterbi_continuity = evaluate_continuity(
                viterbi_prediction, observations, arcs
            )

            classifier_sequences = evaluate_sequences(
                classifier_prediction, reference, usable, observations
            )

            viterbi_sequences = evaluate_sequences(
                viterbi_prediction, reference, usable, observations
            )

            total_usable += usable_count
            total_misses += miss_count

            total_continuous += viterbi_continuity[
                "continuous_transitions"
            ]
            total_disconnected += viterbi_continuity[
                "disconnected_transitions"
            ]
            total_missing_transitions += viterbi_continuity[
                "transitions_missing_prediction"
            ]

            total_exact_sequences += viterbi_sequences[
                "exactly_matched_sequences"
            ]
            total_sequences += viterbi_sequences[
                "evaluated_sequences"
            ]

            result = {
                "model": model_name,
                "trip_id": trip_id,
                "usable_reference_observations": usable_count,
                "candidate_misses": miss_count,
                "candidate_miss_rate": safe_rate(
                    miss_count, usable_count
                ),
                "classifier_only_correct": int(
                    classifier_matches.sum()
                ),
                "classifier_only_end_to_end_accuracy": classifier_accuracy,
                "viterbi_correct": int(viterbi_matches.sum()),
                "viterbi_end_to_end_accuracy": viterbi_accuracy,
                "classifier_route_continuity": (
                    classifier_continuity["route_continuity_rate"]
                ),
                "classifier_disconnected_transitions": (
                    classifier_continuity["disconnected_transitions"]
                ),
                "viterbi_route_continuity": (
                    viterbi_continuity["route_continuity_rate"]
                ),
                "viterbi_disconnected_transitions": (
                    viterbi_continuity["disconnected_transitions"]
                ),
                "viterbi_missing_transitions": (
                    viterbi_continuity["transitions_missing_prediction"]
                ),
                "viterbi_sequence_exact_match_rate": (
                    viterbi_sequences["sequence_exact_match_rate"]
                ),
                "viterbi_evaluated_sequences": (
                    viterbi_sequences["evaluated_sequences"]
                ),
            }

            all_trip_results.append(result)

            print(
                f"{trip_id}: usable={usable_count}, "
                f"misses={miss_count}, "
                f"classifier={classifier_accuracy}, "
                f"Viterbi={viterbi_accuracy}, "
                f"continuity={viterbi_continuity['route_continuity_rate']}"
            )

        # Aggregate candidate-level classification metrics.
        y_array = np.asarray(candidate_y, dtype=np.int8)
        p_array = np.asarray(candidate_prob, dtype=float)
        predicted_binary = (p_array >= 0.5).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_array, predicted_binary, labels=[0, 1]
        ).ravel()

        loss = log_loss(y_array, p_array, labels=[0, 1])

        confusion = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        with (OUTPUT / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(confusion, f, indent=4)

        result = {
            "model": model_name,
            "eligible_candidate_count": int(len(y_array)),
            "confusion_matrix": confusion,
            "log_loss": float(loss),
            "usable_reference_observations": total_usable,
            "candidate_misses": total_misses,
            "candidate_miss_rate": safe_rate(
                total_misses, total_usable
            ),
            "classifier_only_correct": classifier_correct,
            "classifier_only_end_to_end_accuracy": safe_rate(
                classifier_correct, classifier_evaluated
            ),
            "viterbi_correct": viterbi_correct,
            "viterbi_end_to_end_accuracy": safe_rate(
                viterbi_correct, viterbi_evaluated
            ),
            "viterbi_continuous_transitions": total_continuous,
            "viterbi_disconnected_transitions": total_disconnected,
            "viterbi_missing_transitions": total_missing_transitions,
            "viterbi_route_continuity_rate": safe_rate(
                total_continuous,
                total_continuous + total_disconnected,
            ),
            "viterbi_sequence_exact_matches": total_exact_sequences,
            "viterbi_evaluated_sequences": total_sequences,
            "viterbi_sequence_exact_match_rate": safe_rate(
                total_exact_sequences, total_sequences
            ),
            "label_caveat": (
                "Reference targets are derived monotone-alignment weak labels, "
                "not independently supplied GPS ground truth."
            ),
        }

        aggregate_results[model_name] = result

    pd.DataFrame(all_trip_results).to_csv(
        OUTPUT / "per_trip_route_comparison.csv", index=False
    )

    with (OUTPUT / "aggregate_route_comparison.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate_results, f, indent=4)

    print("\n========== AGGREGATE COMPARISON ==========")

    for model_name, result in aggregate_results.items():
        print(f"\n{model_name}")
        for key, value in result.items():
            print(f"{key}: {value}")

    print("\nSaved:")
    print(OUTPUT / "aggregate_route_comparison.json")
    print(OUTPUT / "per_trip_route_comparison.csv")


if __name__ == "__main__":
    main()
