# PathSnap

PathSnap is a map-matching and route-reconstruction project that processes GPS observations, generates candidate road segments and features, and evaluates machine-learning approaches.

## Data processing

Run `01_data_exploration.ipynb` from the repository root after installing `requirements.txt`. It validates training and student-test data, generates candidate features and separate weak labels, exports per-trip data under `results/processed`, creates exploratory Folium maps, and writes trip-level folds. Arc IDs are trip-local.

## Five-model evaluation

`run_five_model_cv.py` performs five-fold trip-level cross-validation using KNN, Linear SVM, Logistic Regression, Random Forest, and XGBoost. Decision Tree was intentionally excluded. The experiment uses 19 features and 70 usable training trips. Student-test data was not loaded.

Results: `artifacts/five_model_cv/`

XGBoost achieved the strongest mean accuracy, precision, ROC-AUC, average precision, and observation Top-1 accuracy. Random Forest achieved the strongest mean recall and F1.

## Random Forest Bayesian optimisation

`run_bayesian_optimization.py` uses Optuna TPE with 8 trials and five-fold trip-level cross-validation. Mean average precision is the optimisation objective. The student-test partition was not loaded.

Results: `artifacts/bayesian_optimization/`

Optimising average precision does not imply improvement in every evaluation metric. The tuned Random Forest improved recall and average precision but reduced accuracy, precision, F1, and observation Top-1 accuracy relative to its baseline.

## Final model evaluation

The final evaluation scripts are:

- `run_final_evaluation.py` — evaluates the Bayesian-optimised Random Forest.
- `run_xgboost_final_evaluation.py` — evaluates the XGBoost baseline from the five-model comparison.
- `run_route_comparison.py` — compares candidate classification and graph-aware Viterbi route reconstruction.

Both final classifiers were fitted using a stratified 150,000-row sample from the training partition. The ten-trip student-test partition was not used for fitting or hyperparameter tuning. One student-test trip had no eligible labels and was excluded from classification metrics.

The live demonstration used XGBoost as the integrated classifier. Its saved model is `artifacts/final_model/xgboost_baseline.joblib`. The tuned Random Forest comparison model is saved under `artifacts/random_forest_comparison/`.

Final results and the methodology are documented in `reports/FINAL_ML_EXPERIMENT_REPORT.md`.

## Evaluation caveat

Reference targets are generated through monotone alignment to reference routes. They are weak labels, not independently supplied GPS ground truth. Candidate classification, candidate coverage, and route-reconstruction metrics should be interpreted with this limitation in mind.

## Dependencies

Install dependencies using:

`python -m pip install -r requirements.txt`

## Experiment scope

This branch records the traditional machine-learning experiments, evaluation scripts, and results. The live application integration and demo environment are maintained separately.
