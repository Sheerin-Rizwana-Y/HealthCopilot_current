
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA = Path("datasets/vitals/sepsis_processed")

# Load temporal datasets
train = pd.read_csv(DATA / "temporal_train.csv")
val = pd.read_csv(DATA / "temporal_val.csv")

# Separate features from identifiers and targets
EXCLUDE = ["patient", "hour", "SepsisLabel"]

FEATURES = [
    col for col in train.columns
    if col not in EXCLUDE
]

X_train = train[FEATURES]
y_train = train["SepsisLabel"].astype(int)

X_val = val[FEATURES]
y_val = val["SepsisLabel"].astype(int)

# Verify patient separation
assert set(train["patient"]).isdisjoint(
    set(val["patient"])
)
assert list(X_train.columns) == list(X_val.columns)
assert set(y_train.unique()) == {0, 1}
assert set(y_val.unique()) == {0, 1}

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print("\n--- XGBOOST V3.1 TRAINING ---")
print("Training patients:", train["patient"].nunique())
print("Validation patients:", val["patient"].nunique())
print("Training observations:", len(train))
print("Validation observations:", len(val))
print("Features:", len(FEATURES))
print("Positive training rows:", positive)
print("Positive validation rows:", int(y_val.sum()))
print("scale_pos_weight:", round(scale_pos_weight, 2))

# Keep the main settings consistent with V3
# to initially examine the effect of temporal features.
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Validation predictions
probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n--- XGBOOST V3.1 VALIDATION ---")

print(
    classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=[
            "No sepsis label",
            "Sepsis label"
        ],
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_val, predictions, labels=[0, 1]
)

ap = average_precision_score(
    y_val, probabilities
)

auc = roc_auc_score(
    y_val, probabilities
)

precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

f1 = f1_score(
    y_val, predictions, zero_division=0
)

print("Confusion matrix [0, 1]:")
print(cm)

print("Average precision:", round(ap, 4))
print("ROC-AUC:", round(auc, 4))
print("Positive precision:", round(precision, 4))
print("Positive recall:", round(recall, 4))
print("Positive F1:", round(f1, 4))

# Store probabilities for later error analysis
results = val[
    ["patient", "hour", "SepsisLabel"]
].copy()

results["probability"] = probabilities
results["prediction"] = predictions

results.to_csv(
    DATA / "xgboost_v31_predictions.csv",
    index=False
)

# Save the trained model
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "threshold": 0.5,
        "version": "xgboost_v3.1_temporal",
        "target": "SepsisLabel",
        "warning": (
            "Research-only sepsis-label classifier. "
            "Not validated for medical decisions."
        ),
    },
    DATA / "xgboost_v31.joblib"
)

metrics = {
    "model": "XGBoost V3.1 Temporal",
    "train_patients": int(train["patient"].nunique()),
    "validation_patients": int(val["patient"].nunique()),
    "feature_count": len(FEATURES),
    "scale_pos_weight": float(scale_pos_weight),
    "threshold": 0.5,
    "average_precision": float(ap),
    "roc_auc": float(auc),
    "positive_precision": float(precision),
    "positive_recall": float(recall),
    "positive_f1": float(f1),
    "confusion_matrix": cm.tolist(),
}

(DATA / "xgboost_v31_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print("\n--- FILES SAVED ---")
print("xgboost_v31.joblib")
print("xgboost_v31_metrics.json")
print("xgboost_v31_predictions.csv")
