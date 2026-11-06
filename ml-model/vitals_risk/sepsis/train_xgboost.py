
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

DATA = Path("datasets/vitals/sepsis_processed")

df = pd.read_csv(DATA / "train.csv")

FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "HR_missing",
    "O2Sat_missing",
    "Temp_missing",
]

# Split by patient, not by observation.
patients = (
    df.groupby("patient")["SepsisLabel"]
    .max()
    .reset_index()
)

train_ids, val_ids = train_test_split(
    patients["patient"],
    test_size=0.20,
    random_state=42,
    stratify=patients["SepsisLabel"],
)

train = df[df["patient"].isin(train_ids)].copy()
val = df[df["patient"].isin(val_ids)].copy()

assert set(train["patient"]).isdisjoint(set(val["patient"]))

X_train = train[FEATURES]
y_train = train["SepsisLabel"].astype(int)

X_val = val[FEATURES]
y_val = val["SepsisLabel"].astype(int)

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print("\n--- V3 XGBOOST DATA ---")
print("Train patients:", len(train_ids))
print("Validation patients:", len(val_ids))
print("Train observations:", len(train))
print("Validation observations:", len(val))
print("Train positive observations:", positive)
print("Validation positive observations:", int(y_val.sum()))
print("scale_pos_weight:", round(scale_pos_weight, 2))

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

probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n--- V3 XGBOOST VALIDATION RESULTS ---")

print(
    classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["No sepsis label", "Sepsis label"],
        zero_division=0,
    )
)

cm = confusion_matrix(y_val, predictions, labels=[0, 1])

print("Confusion matrix [0, 1]:")
print(cm)

ap = average_precision_score(y_val, probabilities)
auc = roc_auc_score(y_val, probabilities)

print("Average precision:", round(ap, 4))
print("ROC-AUC:", round(auc, 4))

OUTPUT = DATA / "xgboost_v3.joblib"

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "version": "xgboost_v3",
        "target": "SepsisLabel",
        "threshold": 0.5,
        "warning": (
            "Research-only sepsis-label model. "
            "Not a validated general physiological-risk classifier."
        ),
    },
    OUTPUT,
)

metrics = {
    "train_patients": len(train_ids),
    "validation_patients": len(val_ids),
    "train_positive_observations": positive,
    "validation_positive_observations": int(y_val.sum()),
    "scale_pos_weight": float(scale_pos_weight),
    "average_precision": float(ap),
    "roc_auc": float(auc),
    "confusion_matrix": cm.tolist(),
}

(DATA / "xgboost_validation_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\nModel saved:", OUTPUT)
print("Metrics saved:", DATA / "xgboost_validation_metrics.json")
