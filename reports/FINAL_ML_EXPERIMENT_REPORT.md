# PathSnap — Final Machine Learning Experiment Report

## 1. Project overview

PathSnap performs GPS-to-road map matching. The machine-learning component scores candidate road segments for each GPS observation. A graph-aware Viterbi decoder then uses candidate scores and road-network connectivity to reconstruct a continuous route.

The candidate classifier uses 19 features:

- distance_m
- arc_length_m
- projection_fraction
- distance_to_start_m
- distance_to_end_m
- heading_cos
- heading_sin
- heading_valid
- speed_mps
- dt_s
- step_distance_m
- gap_flag
- speed_outlier_flag
- previous_point_distance_m
- candidate_rank
- candidate_count
- distance_margin_m
- out_degree
- in_degree

The final integrated/demo classifier is **XGBoost**, consistent with the model used in the live demonstration. Random Forest was also evaluated and Bayesian-optimised as an experimental comparison.

## 2. Dataset and evaluation protocol

| Item | Details |
|---|---|
| Training partition | 80 trips |
| Usable training trips | 70 |
| Development/student-test partition | 10 trips |
| Cross-validation | 5-fold trip-level CV |
| Features | 19 |
| Final fitting sample | 150,000 candidate rows |
| Sampling seed | 42 |
| Final fitting sample positives | 22,420 |
| Student-test candidate rows | 130,923 |
| Student-test eligible candidate rows | 118,594 |
| Student-test trips with eligible labels | 9 of 10 |

The final fitting sample was drawn from the training partition. The student-test partition was not used for model fitting or Bayesian hyperparameter tuning.

Trip 00000076 had no eligible labels and was excluded from classification metric calculations. It remains part of the ten-trip evaluation inventory.

## 3. Five-model cross-validation comparison

The following are mean results across five folds. Values in parentheses are standard deviations.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision | Observation Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 92.53% (1.47) | 77.38% (4.36) | 69.89% (4.61) | 73.44% (4.43) | 93.56% (1.67) | 78.21% (5.00) | 76.93% (3.99) |
| Linear SVM | 89.35% (1.50) | 59.11% (2.55) | 91.12% (2.64) | 71.68% (2.44) | 96.49% (1.08) | 82.29% (5.33) | 80.96% (4.27) |
| Logistic Regression | 89.06% (1.55) | 58.23% (2.57) | 92.47% (2.20) | 71.44% (2.29) | 96.62% (1.06) | 82.47% (5.52) | 81.04% (4.37) |
| Random Forest | 94.10% (1.49) | 77.37% (4.72) | 85.31% (3.23) | 81.12% (3.89) | 97.55% (0.91) | 88.34% (3.62) | 81.76% (4.23) |
| XGBoost | 94.40% (1.46) | 82.48% (4.15) | 79.00% (4.44) | 80.70% (4.25) | 97.69% (0.86) | 88.82% (3.98) | 81.85% (4.19) |

### Interpretation

- XGBoost achieved the strongest mean accuracy, precision, ROC-AUC, average precision and observation Top-1 accuracy.
- Random Forest achieved the strongest mean recall and F1.
- Linear SVM and Logistic Regression showed high recall but lower precision.
- KNN provided a useful distance-based baseline.
- XGBoost was retained as the integrated/demo model. The student-test results below are reported as evaluation evidence, not as a hyperparameter-selection procedure.

## 4. Bayesian optimisation

Bayesian optimisation was conducted on Random Forest using Optuna's TPE sampler.

| Setting | Value |
|---|---|
| Optimiser | Optuna TPE |
| Trials | 8 |
| Cross-validation | 5 folds |
| Objective | Mean validation average precision |
| Best trial | 4 |
| Best objective value | 0.890408 |
| Random seed | 42 |
| Approximate optimisation runtime | 29.54 minutes |

### Best hyperparameters

| Parameter | Value |
|---|---:|
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_split | 3 |
| min_samples_leaf | 8 |
| max_features | 0.75 |
| criterion | entropy |
| class_weight | balanced_subsample |
| max_samples | 0.690554 |

### Baseline versus tuned Random Forest

| Metric | Baseline RF | Tuned RF |
|---|---:|---:|
| Accuracy | 94.10% | 92.49% |
| Precision | 77.37% | 68.23% |
| Recall | 85.31% | 92.87% |
| F1 | 81.12% | 78.62% |
| ROC-AUC | 97.55% | 97.74% |
| Average precision | 88.34% | 89.04% |
| Observation Top-1 | 81.76% | 81.71% |

Tuning improved recall, ROC-AUC and average precision. It reduced accuracy, precision, F1 and observation Top-1 accuracy. Therefore, the optimisation produced a trade-off rather than a universal improvement.

The tuned Random Forest was refitted on a stratified 150,000-row sample from the training partition and saved as a separate experimental model.

## 5. Final XGBoost model and student-test evaluation

The XGBoost baseline from the five-model comparison was fitted using the same 150,000-row stratified training sample and seed 42.

| Metric | Student-test result |
|---|---:|
| Candidate accuracy | 94.71% |
| Candidate precision | 83.11% |
| Candidate recall | 80.80% |
| Candidate F1 | 81.94% |
| Average precision | 90.79% |
| ROC-AUC | 97.99% |
| Observation Top-1 accuracy | 82.79% |
| Total inference time | 0.157 seconds |
| Inference throughput | Approximately 834,227 candidates/second |

The evaluation covered 118,594 eligible candidate rows across nine trips. Trip 00000076 had no eligible labels.

The saved XGBoost model is:

`artifacts/final_model/xgboost_baseline.joblib`

## 6. Random Forest student-test evaluation

The Bayesian-optimised Random Forest was evaluated on the same eligible student-test candidate rows.

| Metric | Student-test result |
|---|---:|
| Candidate accuracy | 92.45% |
| Candidate precision | 67.84% |
| Candidate recall | 93.50% |
| Candidate F1 | 78.63% |
| Average precision | 90.71% |
| ROC-AUC | 98.02% |
| Observation Top-1 accuracy | 82.54% |

The saved tuned Random Forest model is:

`artifacts/random_forest_comparison/random_forest_tuned.joblib`

## 7. Candidate coverage and route reconstruction

There were 17,646 usable reference observations across the ten student-test trips.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Candidate misses | 36 | 36 |
| Candidate miss rate | 0.204% | 0.204% |
| Classifier-only end-to-end accuracy | 82.37% | 82.62% |
| Viterbi end-to-end accuracy | 89.96% | 88.75% |
| Continuous Viterbi transitions | 19,032 | 18,979 |
| Disconnected Viterbi transitions | 116 | 169 |
| Missing Viterbi transitions | 0 | 0 |
| Viterbi route continuity | 99.39% | 99.12% |
| Log loss | 0.1793 | 0.1259 |

The Viterbi decoder uses the same fixed graph-aware transition heuristic for both models. It considers candidate scores, road-segment continuity, disconnected transitions and GPS gaps.

The comparison shows that Viterbi decoding improves end-to-end route accuracy over classifier-only predictions for both models. Random Forest has higher reported Viterbi route accuracy and continuity in this comparison, while XGBoost has stronger candidate classification F1 and lower log loss.

The sequence-exact-match rate was zero for both models under the strict full-sequence criterion. It is not treated as a headline result.

## 8. Confusion matrices

### Random Forest

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 93,179 | 7,805 |
| Actual positive | 1,145 | 16,465 |

### XGBoost

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 98,093 | 2,891 |
| Actual positive | 3,381 | 14,229 |

## 9. Limitations and interpretation caveats

1. Reference targets are generated through monotone alignment to reference routes. They are weak labels, not independently supplied GPS ground truth.
2. Trip 00000076 has no eligible labels and is excluded from classification metric calculations.
3. Candidate miss rate measures usable reference observations whose aligned reference segment was absent from the candidate list.
4. Route accuracy and continuity are calculated against the available weak-label reference alignments.
5. A strict full-sequence exact match requires every evaluated observation in a sequence to match; neither model achieved this criterion.
6. The reported Viterbi decoder uses fixed transition penalties. These are not learned parameters.
7. Student-test results are reported for evaluation and comparison. They were not used for fitting or Bayesian hyperparameter optimisation.

## 10. Final model status

**XGBoost is the final integrated/demo model**, consistent with the model used in the live demonstration.

Random Forest is retained as the five-model comparison winner on cross-validation F1 and as the Bayesian-optimisation experiment. Its stronger recall and route-reconstruction results are reported alongside XGBoost rather than used to silently replace the integrated model.

## 11. Reproducibility and saved evidence

Key experiment scripts:

- `run_bayesian_optimization.py`
- `run_final_evaluation.py`
- `run_xgboost_final_evaluation.py`
- `run_route_comparison.py`

Key existing experiment evidence:

- `artifacts/five_model_cv/model_summary.csv`
- `artifacts/five_model_cv/fold_metrics.csv`
- `artifacts/bayesian_optimization/best_params.json`
- `artifacts/bayesian_optimization/trials.csv`
- `artifacts/bayesian_optimization/baseline_vs_tuned.csv`
- `artifacts/final_evaluation/aggregate_metrics.json`
- `artifacts/xgboost_final_evaluation/aggregate_metrics.json`
- `artifacts/route_comparison/aggregate_route_comparison.json`
- `artifacts/route_comparison/per_trip_route_comparison.csv`

The model files and detailed prediction outputs are stored under `artifacts/`. The repository's `.gitignore` currently excludes these generated artifacts, so they must be handled deliberately if the submission requires them to be version-controlled.
