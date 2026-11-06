
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA = Path("datasets/vitals/sepsis_processed")

train = pd.read_csv(DATA / "temporal_train.csv")
val = pd.read_csv(DATA / "temporal_val.csv")

bundle = joblib.load(DATA / "xgboost_v31.joblib")
xgb = bundle["model"]
features = bundle["features"]

assert set(train["patient"]).isdisjoint(set(val["patient"]))

X_train = train[features]
y_train = train["SepsisLabel"].astype(int)

X_val = val[features]
y_val = val["SepsisLabel"].astype(int)

# Imputer fitted only on inner training patients.
rf = make_pipeline(
    SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True
    ),
    RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
)

rf.fit(X_train, y_train)

models = {
    "Random Forest": rf,
    "XGBoost V3.1": xgb,
}

print("\n--- CONTROLLED MODEL COMPARISON ---")
print("Validation patients:", val["patient"].nunique())
print("Validation rows:", len(val))
print("Positive rows:", int(y_val.sum()))
print("Positive prevalence:", round(y_val.mean(), 4))
print()

for name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"--- {name} ---")
    print(
        "Average precision:",
        round(average_precision_score(y_val, probabilities), 4)
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_val, probabilities), 4)
    )
    print(
        "Precision:",
        round(precision_score(
            y_val, predictions, zero_division=0
        ), 4)
    )
    print(
        "Recall:",
        round(recall_score(
            y_val, predictions, zero_division=0
        ), 4)
    )
    print(
        "F1:",
        round(f1_score(
            y_val, predictions, zero_division=0
        ), 4)
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))
    print()
