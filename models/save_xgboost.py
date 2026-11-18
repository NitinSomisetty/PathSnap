
from pathlib import Path
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "processed"
FOLD_FILE = ROOT / "results" / "training_folds.csv"
OUTPUT_DIR = ROOT / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_parts = []
y_parts = []

for trip_dir in sorted((DATA_DIR / "train").iterdir()):
    if not trip_dir.is_dir():
        continue

    with np.load(trip_dir / "candidates.npz") as data:
        X = data["X"]

    with np.load(trip_dir / "targets.npz") as data:
        y = data["y"]
        mask = data["fit_mask"].astype(bool)

    if mask.sum() == 0:
        continue

    X_parts.append(X[mask])
    y_parts.append(y[mask])

X_train = np.concatenate(X_parts).astype(np.float32)
y_train = np.concatenate(y_parts).astype(np.int8)

print("Training XGBoost using previously discovered best parameters")
print("Training rows:", len(y_train))

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
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

model.fit(X_train, y_train)

output = OUTPUT_DIR / "xgboost_model.json"
model.save_model(str(output))

print("XGBoost model saved:", output)
