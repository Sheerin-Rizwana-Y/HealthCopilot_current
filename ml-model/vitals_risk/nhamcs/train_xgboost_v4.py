
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path("datasets/vitals/nhamcs/2022")
OUTPUT = ROOT / "models"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "processed/nhamcs_2022_clean.csv"
)

FEATURES = ["hr", "spo2", "temperature"]
LABELS = ["Low", "Moderate", "High"]

# Check input data
assert df[FEATURES].notna().all().all()
assert df["risk_label"].isin(LABELS).all()

df = df.copy()
df["target"] = df["risk_label"].map({
    "Low": 0,
    "Moderate": 1,
    "High": 2,
})

# Hold out 15% for final testing.
development, test = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["target"],
)

# Validation is 15% of the full dataset.
train, val = train_test_split(
    development,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=development["target"],
)

print("\n--- NHAMCS XGBOOST V4 DATA ---")
print("Training records:", len(train))
print("Validation records:", len(val))
print("Held-out test records:", len(test))

print("\nTraining label distribution:")
print(train["risk_label"].value_counts())

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=250,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=1.0,
    min_child_weight=8,
    reg_lambda=5,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    train[FEATURES],
    train["target"],
)

# Evaluate only the validation set.
pred = model.predict(val[FEATURES])

print("\n--- XGBOOST V4 VALIDATION ---")

print(
    classification_report(
        val["target"],
        pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        zero_division=0,
    )
)

cm = confusion_matrix(
    val["target"],
    pred,
    labels=[0, 1, 2],
)

print("Confusion matrix order:", LABELS)
print(cm)

metrics = {
    "validation_accuracy": float(
        accuracy_score(val["target"], pred)
    ),
    "validation_balanced_accuracy": float(
        balanced_accuracy_score(val["target"], pred)
    ),
    "validation_macro_f1": float(
        f1_score(
            val["target"],
            pred,
            average="macro",
        )
    ),
    "confusion_matrix": cm.tolist(),
}

print("\nAccuracy:", round(
    metrics["validation_accuracy"], 4
))
print("Balanced accuracy:", round(
    metrics["validation_balanced_accuracy"], 4
))
print("Macro F1:", round(
    metrics["validation_macro_f1"], 4
))

# Majority-class baseline for comparison
majority_class = int(
    train["target"].mode().iloc[0]
)
baseline_pred = np.full(
    len(val), majority_class
)

print("\nMajority baseline accuracy:",
      round(accuracy_score(
          val["target"], baseline_pred
      ), 4))

print("Majority baseline macro F1:",
      round(f1_score(
          val["target"],
          baseline_pred,
          average="macro",
          zero_division=0
      ), 4))

# Save all splits, including untouched test data.
train.to_csv(OUTPUT / "train.csv", index=False)
val.to_csv(OUTPUT / "validation.csv", index=False)
test.to_csv(OUTPUT / "test_untouched.csv", index=False)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "labels": LABELS,
        "version": "nhamcs_xgboost_v4",
        "target": "NHAMCS ED triage urgency proxy",
        "warning": (
            "Research-only triage-urgency model. "
            "Not a validated wearable medical risk model."
        ),
    },
    OUTPUT / "xgboost_v4.joblib"
)

(OUTPUT / "validation_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n--- SAVED ---")
print("Model: xgboost_v4.joblib")
print("Metrics: validation_metrics.json")
print("Held-out test set preserved.")
