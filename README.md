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
The existing `pipeline_updated.py` builds trip-local candidate features; training scripts and saved model artifacts are under `models/` and `artifacts/`. `backend.py` runs raw-trip inference with the saved XGBoost classifier and a graph-aware Viterbi decoder. `frontend.py` is the evaluator interface.

## Setup

On both laptops, install Python dependencies with `pip install -r requirements.txt`. For local development, the API automatically uses the included `Student_folder/Student folder/student_test` trips. Restart the API after changing configuration. For evaluation, on the API laptop place the reserved trip directories (each containing matching `.track`, `.nodes`, and `.arcs` files, with optional `.route`) under `evaluation_trips/`, or set `PATHSNAP_EVALUATION_DIR` to their parent directory. Keep this directory on the API laptop. The trained model defaults to `artifacts/xgboost_model.json`; override with `PATHSNAP_MODEL_PATH` if needed.

Start the API on laptop 1:

```powershell
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Start the UI on laptop 2:

```powershell
$env:PATHSNAP_API_URL = 'http://<API-LAPTOP-IP>:8000'
streamlit run frontend.py --server.address 0.0.0.0
```

The UI lists trips from `GET /trips` and requests prediction from `POST /trips/{trip_id}/predict`. Check `GET /health` for setup status. If both laptops are on different origins, set `PATHSNAP_CORS_ORIGINS` on laptop 1 to the UI origin. The UI can also enter the API URL in its sidebar.

The API computes candidate features directly from the raw trip, then applies XGBoost and Viterbi. A supplied `.route` file is used only after prediction to compute a weak alignment accuracy. It is never supplied to the feature builder or decoder. Missing candidates return null arc IDs and reduce coverage. GPS gaps above 30 seconds restart decoding. The green map line shows selected road segments in observation order; it does not insert unobserved connecting arcs.

Training comparison, tuning, and selection must be reviewed from recorded experiments before claiming a final result. The reserved evaluator data must never be used for training or model selection.
