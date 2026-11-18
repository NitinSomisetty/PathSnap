# PathSnap

PathSnap is a map-matching and route-reconstruction project that processes GPS observations, generates candidate road segments and features, and evaluates machine-learning approaches.

## Data processing

Run 01_data_exploration.ipynb from the repository root after installing requirements.txt. It validates training and student-test data, generates candidate features and separate weak labels, exports per-trip data under results/processed, creates exploratory Folium maps, and writes trip-level folds. Arc IDs are trip-local.

## Five-model evaluation

run_five_model_cv.py performs five-fold trip-level cross-validation using KNN, Linear SVM, Logistic Regression, Random Forest, and XGBoost. Decision Tree was intentionally excluded. The experiment uses 19 features and 70 usable training trips. Student-test data was not loaded.

Results: artifacts/five_model_cv/

## Random Forest Bayesian optimisation

run_bayesian_optimization.py uses Optuna TPE with 8 trials and five-fold trip-level cross-validation. Mean average precision is the optimisation objective. The student-test partition was not loaded.

Results: artifacts/bayesian_optimization/

Optimising average precision does not imply improvement in every evaluation metric.

## Dependencies

Install dependencies using python -m pip install -r requirements.txt.

## Experiment scope

The experiment artifacts document model evaluation and hyperparameter optimisation. They do not independently establish which model is integrated into the live application or verify the live demo implementation.
