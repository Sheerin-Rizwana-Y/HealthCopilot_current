
from pathlib import Path
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path("datasets/vitals/nhamcs/2022/models")

# Reuse exactly the same V4 train/validation split.
train = pd.read_csv(ROOT / "train.csv")
val = pd.read_csv(ROOT / "validation.csv")

FEATURES = ["hr", "spo2", "temperature"]
LABELS = ["Low", "Moderate", "High"]

X_train = train[FEATURES]
y_train = train["target"].astype(int)

X_val = val[FEATURES]
y_val = val["target"].astype(int)

# Give minority classes more importance.
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("\n--- XGBOOST V4.1 TRAINING ---")
print("Training rows:", len(train))
print("Validation rows:", len(val))

print("\nAverage weight by class:")
for label_id, label in enumerate(LABELS):
    print(
        label,
        round(weights[y_train == label_id].mean(), 3)
    )

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
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

pred = model.predict(X_val)

print("\n--- XGBOOST V4.1 VALIDATION ---")

print(
    classification_report(
        y_val,
        pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        zero_division=0
    )
)

print("Confusion matrix order:", LABELS)
print(
    confusion_matrix(
        y_val, pred, labels=[0, 1, 2]
    )
)

print(
    "Accuracy:",
    round(accuracy_score(y_val, pred), 4)
)

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_val, pred), 4)
)

print(
    "Macro F1:",
    round(f1_score(y_val, pred, average="macro"), 4)
)

print("\nPredicted class distribution:")
print(
    pd.Series(pred)
    .map(dict(enumerate(LABELS)))
    .value_counts()
)

# Save separately. Do not overwrite the V4 model.
model.save_model(
    ROOT / "xgboost_v41_balanced.json"
)

print("\nSaved: xgboost_v41_balanced.json")
print("Original V4 model and test set preserved.")
