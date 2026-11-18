
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "processed" / "student_test"
MODEL_PATH = ROOT / "artifacts" / "xgboost_model.json"


def main():
    print("=" * 60)
    print("PATHSNAP - XGBOOST MODEL PREDICTION")
    print("=" * 60)

    # 1. Load trained model
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print("\nXGBoost model loaded successfully!")

    # 2. Load candidate features from all student-test trips
    X_parts = []
    trip_ids = []
    feature_names = None

    for trip_dir in sorted(DATA_DIR.iterdir()):
        if not trip_dir.is_dir():
            continue

        candidate_file = trip_dir / "candidates.npz"

        if not candidate_file.exists():
            continue

        with np.load(candidate_file) as data:
            X = data["X"].astype(np.float32)
            names = data["feature_names"].tolist()

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature mismatch in trip {trip_dir.name}"
            )

        X_parts.append(X)
        trip_ids.extend([trip_dir.name] * len(X))

        print(f"Loaded trip {trip_dir.name}: {len(X)} candidates")

    if not X_parts:
        raise FileNotFoundError("No candidate data found.")

    X_test = np.concatenate(X_parts, axis=0)

    # 3. Validate feature dimensions
    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Expected {model.n_features_in_} features, "
            f"but received {X_test.shape[1]}"
        )

    # 4. Generate predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    # 5. Display results
    print("\n" + "=" * 60)
    print("XGBOOST PREDICTION RESULTS")
    print("=" * 60)

    print("Total candidate rows:", len(X_test))
    print("Number of features:", X_test.shape[1])
    print("Positive predictions:", int(np.sum(predictions == 1)))
    print("Negative predictions:", int(np.sum(predictions == 0)))

    print("\nFirst 10 predictions:")

    for i in range(min(10, len(predictions))):
        print(
            f"Trip: {trip_ids[i]} | "
            f"Candidate: {i + 1} | "
            f"Prediction: {predictions[i]} | "
            f"Probability: {probabilities[i]:.4f}"
        )

    # 6. Save predictions
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "xgboost_predictions.csv"

    import pandas as pd

    results = pd.DataFrame({
        "trip_id": trip_ids,
        "prediction": predictions,
        "positive_probability": probabilities
    })

    results.to_csv(output_file, index=False)

    print("\nPredictions saved to:", output_file)
    print("\nXGBoost prediction completed successfully!")


if __name__ == "__main__":
    main()
