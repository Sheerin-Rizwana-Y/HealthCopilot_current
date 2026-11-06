
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "datasets/sisfall/sisfall_features.csv"
)

OUTPUT_DIR = Path(
    "ml-model/fall_detection/results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "acc_mean",
    "acc_std",
    "acc_min",
    "acc_max",
    "acc_range",
    "gyro_mean",
    "gyro_std",
    "gyro_max"
]


# ============================================================
# LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS]
y = df["label"]
groups = df["subject"]

print("=" * 55)
print("SISFALL RANDOM FOREST EVALUATION")
print("=" * 55)

print("Dataset shape:", df.shape)
print("Subjects:", groups.nunique())
print("Fall windows:", (y == 1).sum())
print("Non-fall windows:", (y == 0).sum())


# ============================================================
# 5-FOLD GROUP CROSS VALIDATION
# ============================================================

group_kfold = GroupKFold(n_splits=5)

fold_results = []

# Store predictions from all validation folds
oof_predictions = np.full(len(df), -1, dtype=int)

for fold, (train_idx, val_idx) in enumerate(
    group_kfold.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    train_groups = groups.iloc[train_idx]
    val_groups = groups.iloc[val_idx]

    # Subject leakage check
    overlap = set(train_groups).intersection(
        set(val_groups)
    )

    assert len(overlap) == 0, "Subject leakage detected!"

    # Original Random Forest configuration
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Training
    model.fit(X_train, y_train)

    # Validation predictions
    predictions = model.predict(X_val)

    # Store out-of-fold predictions
    oof_predictions[val_idx] = predictions

    # Calculate fold metrics
    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"\nFOLD {fold}")
    print("-" * 40)
    print("Training subjects:", train_groups.nunique())
    print("Validation subjects:", val_groups.nunique())
    print("Validation windows:", len(y_val))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


# ============================================================
# VERIFY ALL WINDOWS HAVE OOF PREDICTIONS
# ============================================================

assert np.all(oof_predictions != -1)

results_df = pd.DataFrame(fold_results)

# Save individual fold results
results_df.to_csv(
    OUTPUT_DIR / "fold_results.csv",
    index=False
)


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

means = results_df[metric_columns].mean()
stds = results_df[metric_columns].std()

print("\n" + "=" * 55)
print("CROSS-VALIDATION SUMMARY")
print("=" * 55)

for metric in metric_columns:
    print(
        f"{metric}: "
        f"{means[metric] * 100:.2f}% "
        f"+/- {stds[metric] * 100:.2f}%"
    )


# ============================================================
# SAVE FOUR-BAR PERFORMANCE CHART
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

scores = means[metric_columns].values * 100
errors = stds[metric_columns].values * 100

plt.figure(figsize=(9, 5))

bars = plt.bar(
    metric_names,
    scores,
    yerr=errors,
    capsize=5,
    width=0.55
)

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.title("Random Forest Model Performance")
plt.ylabel("Performance (%)")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "performance_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMBINED OUT-OF-FOLD CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    oof_predictions,
    labels=[0, 1]
)

print("\nCombined Out-of-Fold Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Fall", "Fall"]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y,
    oof_predictions,
    labels=[0, 1],
    target_names=["Non-Fall", "Fall"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:
    f.write(report)


# ============================================================
# SAVE ALL OOF PREDICTIONS
# ============================================================

prediction_df = df[
    ["subject", "recording", "label"]
].copy()

prediction_df["predicted_label"] = oof_predictions

prediction_df.to_csv(
    OUTPUT_DIR / "evaluation_results.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 55)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Subject leakage check: PASSED")

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():
    print("-", file.name)

print("\nResults saved to:", OUTPUT_DIR)
