from pathlib import Path
import json
import time
import gc
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from run_five_model_cv import (
    ROOT,
    SEED,
    N_FOLDS,
    MAX_TRAIN_ROWS,
    load_all_training_trips,
    stratified_sample,
    calculate_observation_metrics,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = ROOT / "artifacts" / "bayesian_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = 8

BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced_subsample",
    "max_samples": 1.0,
}


def evaluate_fold(model, validation):
    X_val = validation["X"]
    y_val = validation["y"]

    scores = model.predict_proba(X_val)[:, 1]
    predictions = (scores >= 0.5).astype(np.int8)

    result = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, scores),
        "average_precision": average_precision_score(
            y_val, scores
        ),
    }

    observation = calculate_observation_metrics(
        y_val,
        scores,
        validation["trip"],
        validation["observation"],
    )

    result.update(observation)
    return result


def make_params(trial):
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 350, step=50
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", [None, 10, 15, 20, 25, 30, 40]
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 20
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 8
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.75]
        ),
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy"]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight",
            ["balanced_subsample", "balanced", None]
        ),
        "max_samples": trial.suggest_float(
            "max_samples", 0.65, 1.0
        ),
    }


def main():
    start_all = time.perf_counter()

    print("\n========== PATHSNAP BAYESIAN OPTIMISATION ==========")
    print("Model: Random Forest")
    print("Optimiser: Optuna TPE")
    print("Trials:", TRIALS)
    print("Validation: Existing five-fold trip-level CV")
    print("Student-test data: NOT LOADED")

    data, feature_names = load_all_training_trips()

    print("\nPreparing fixed training samples for all trials...")

    fold_data = {}

    for fold in range(N_FOLDS):
        validation = data[fold]
        train_folds = [f for f in range(N_FOLDS) if f != fold]

        X_train = np.concatenate(
            [data[f]["X"] for f in train_folds]
        )
        y_train = np.concatenate(
            [data[f]["y"] for f in train_folds]
        )

        X_fit, y_fit = stratified_sample(
            X_train,
            y_train,
            MAX_TRAIN_ROWS,
            SEED + fold,
        )

        fold_data[fold] = {
            "X_fit": X_fit,
            "y_fit": y_fit,
            "validation": validation,
        }

        print(
            f"Fold {fold}: {len(y_fit):,} training rows, "
            f"{len(validation['y']):,} validation rows"
        )

    records = []

    def objective(trial):
        params = make_params(trial)

        fold_results = []

        print(f"\nTrial {trial.number + 1}/{TRIALS}")
        print("Parameters:", params)

        for fold in range(N_FOLDS):
            current = fold_data[fold]

            model = RandomForestClassifier(
                **params,
                bootstrap=True,
                n_jobs=-1,
                random_state=SEED + fold,
            )

            model.fit(
                current["X_fit"],
                current["y_fit"],
            )

            metrics = evaluate_fold(
                model,
                current["validation"],
            )

            metrics["fold"] = fold
            metrics["trial"] = trial.number
            metrics["training_rows"] = len(current["y_fit"])
            fold_results.append(metrics)

            print(
                f"  Fold {fold}: "
                f"AP={metrics['average_precision']:.5f} | "
                f"F1={metrics['f1']:.5f} | "
                f"Top-1={metrics['observation_top1_accuracy']:.5f}"
            )

            del model
            gc.collect()

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "observation_top1_accuracy",
        ]

        means = {
            name: float(np.mean([
                row[name] for row in fold_results
            ]))
            for name in metric_names
        }

        trial.set_user_attr("fold_results", fold_results)
        trial.set_user_attr("mean_metrics", means)

        records.extend(fold_results)

        print(
            f"  MEAN AP={means['average_precision']:.5f} | "
            f"F1={means['f1']:.5f} | "
            f"Top-1={means['observation_top1_accuracy']:.5f}"
        )

        return means["average_precision"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="pathsnap_random_forest_bayesian_optimization",
    )

    # Evaluate the existing baseline configuration as trial zero.
    study.enqueue_trial(BASELINE_PARAMS)

    def save_checkpoint(study, trial):
        completed = [
            item for item in study.trials
            if item.value is not None
        ]

        fold_rows = []
        trial_rows = []

        for item in completed:
            fold_rows.extend(item.user_attrs.get("fold_results", []))
            trial_rows.append({
                "trial": item.number,
                "state": item.state.name,
                "objective_average_precision": item.value,
                **item.params,
                **item.user_attrs.get("mean_metrics", {}),
            })

        pd.DataFrame(fold_rows).to_csv(
            OUTPUT_DIR / "tuned_fold_metrics.csv",
            index=False,
        )
        pd.DataFrame(trial_rows).to_csv(
            OUTPUT_DIR / "trials.csv",
            index=False,
        )

        print(f"\nCheckpoint saved: {len(completed)} completed trials.")

    study.optimize(
        objective,
        n_trials=TRIALS,
        show_progress_bar=True,
        callbacks=[save_checkpoint],
    )

    # Save every fold-level result.
    fold_results = pd.DataFrame(records)
    fold_results.to_csv(
        OUTPUT_DIR / "tuned_fold_metrics.csv",
        index=False,
    )

    # Save trial-level parameters and mean metrics.
    trial_records = []

    for trial in study.trials:
        if trial.value is None:
            continue

        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "objective_average_precision": trial.value,
            **trial.params,
            **trial.user_attrs.get("mean_metrics", {}),
        }
        trial_records.append(row)

    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(
        OUTPUT_DIR / "trials.csv",
        index=False,
    )

    # Summarise the best trial.
    best = study.best_trial
    best_metrics = best.user_attrs["mean_metrics"]

    with (OUTPUT_DIR / "best_params.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({
            "model": "Random Forest",
            "optimizer": "Optuna TPE",
            "best_trial": best.number,
            "objective": "Mean five-fold average precision",
            "best_value": best.value,
            "best_parameters": best.params,
            "mean_validation_metrics": best_metrics,
            "seed": SEED,
            "folds": N_FOLDS,
        }, f, indent=4)

    # Compare the baseline and tuned model using the same folds.
    baseline_file = ROOT / "artifacts" / "five_model_cv" / "fold_metrics.csv"
    baseline_df = pd.read_csv(baseline_file)
    baseline_df = baseline_df[
        baseline_df["model"] == "Random Forest"
    ]

    baseline_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "observation_top1_accuracy",
    ]

    baseline_mean = baseline_df[baseline_metrics].mean()

    comparison = []

    comparison.append({
        "model": "Random Forest Baseline",
        **{metric: float(baseline_mean[metric])
           for metric in baseline_metrics},
    })

    comparison.append({
        "model": "Random Forest Bayesian Optimised",
        **best_metrics,
    })

    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / "baseline_vs_tuned.csv",
        index=False,
    )

    metadata = {
        "experiment": "PathSnap Random Forest Bayesian Optimisation",
        "optimizer": "Optuna TPESampler",
        "trials": TRIALS,
        "seed": SEED,
        "folds": N_FOLDS,
        "objective": "Mean five-fold average precision",
        "training_row_limit_per_fold": MAX_TRAIN_ROWS,
        "sampling": "Fixed stratified training sample per fold",
        "validation": "Existing trip-level training folds",
        "usable_training_trips": 70,
        "student_test_loaded": False,
        "feature_names": feature_names,
        "baseline_parameters": BASELINE_PARAMS,
        "search_space": {
            "n_estimators": "100 to 350, step 50",
            "max_depth": [None, 10, 15, 20, 25, 30, 40],
            "min_samples_split": "2 to 20",
            "min_samples_leaf": "1 to 8",
            "max_features": ["sqrt", "log2", 0.5, 0.75],
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced_subsample", "balanced", None],
            "max_samples": "0.65 to 1.0",
        },
        "best_trial": best.number,
        "best_average_precision": best.value,
        "elapsed_seconds": time.perf_counter() - start_all,
    }

    with (OUTPUT_DIR / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n========== OPTIMISATION COMPLETE ==========")
    print("Best trial:", best.number)
    print("Best mean AP:", round(best.value, 6))
    print("Best parameters:")
    print(json.dumps(best.params, indent=4))

    print("\nBest mean validation metrics:")
    print(json.dumps(best_metrics, indent=4))

    print("\nBaseline versus tuned:")
    print(pd.DataFrame(comparison).round(5).to_string(index=False))

    print("\nSaved artifacts:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print(
        "\nTotal elapsed minutes:",
        round((time.perf_counter() - start_all) / 60, 2),
    )


if __name__ == "__main__":
    main()
