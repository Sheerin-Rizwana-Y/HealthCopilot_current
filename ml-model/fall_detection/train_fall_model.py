from pathlib import Path
import pandas as pd


# ============================================================
# LOAD PROCESSED SISFALL FEATURES
# ============================================================

DATA_PATH = Path(
    "datasets/sisfall/sisfall_features.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)


# ============================================================
# VERIFY DATASET
# ============================================================

print("=" * 50)
print("SISFALL TRAINING DATA CHECK")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nClass counts:")
print(df["label"].value_counts().sort_index())

print("\nUnique subjects:")
print(df["subject"].nunique())

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# SUBJECT-WISE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split


# Get all unique subject IDs
subjects = df["subject"].unique()


# ------------------------------------------------------------
# First split:
# 26 subjects for training
# 12 subjects temporarily for validation + testing
# ------------------------------------------------------------

train_subjects, temp_subjects = train_test_split(
    subjects,
    test_size=12,
    random_state=42
)


# ------------------------------------------------------------
# Second split:
# 6 validation subjects
# 6 test subjects
# ------------------------------------------------------------

val_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=6,
    random_state=42
)


# ------------------------------------------------------------
# Create datasets using subject IDs
# ------------------------------------------------------------

train_df = df[df["subject"].isin(train_subjects)].copy()

val_df = df[df["subject"].isin(val_subjects)].copy()

test_df = df[df["subject"].isin(test_subjects)].copy()


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 50)
print("SUBJECT-WISE DATA SPLIT")
print("=" * 50)

print("\nTraining subjects:")
print(sorted(train_subjects))
print("Number of subjects:", len(train_subjects))

print("\nValidation subjects:")
print(sorted(val_subjects))
print("Number of subjects:", len(val_subjects))

print("\nTest subjects:")
print(sorted(test_subjects))
print("Number of subjects:", len(test_subjects))


print("\nDataset sizes:")
print("Training windows:", len(train_df))
print("Validation windows:", len(val_df))
print("Test windows:", len(test_df))


print("\nTraining class counts:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation class counts:")
print(val_df["label"].value_counts().sort_index())

print("\nTest class counts:")
print(test_df["label"].value_counts().sort_index())


# ============================================================
# LEAKAGE CHECK
# ============================================================

assert set(train_subjects).isdisjoint(val_subjects)
assert set(train_subjects).isdisjoint(test_subjects)
assert set(val_subjects).isdisjoint(test_subjects)

print("\nSubject leakage check: PASSED")

# ============================================================
# RANDOM FOREST BASELINE
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Features used by the model
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


# Prepare training data
X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]

# Prepare validation data
X_val = val_df[FEATURE_COLUMNS]
y_val = val_df["label"]


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

y_val_pred = model.predict(X_val)


# ============================================================
# VALIDATION METRICS
# ============================================================

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)
f1 = f1_score(y_val, y_val_pred, zero_division=0)

cm = confusion_matrix(y_val, y_val_pred)


print("\n" + "=" * 50)
print("RANDOM FOREST - VALIDATION RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nMatrix format:")
print("[[TN FP]")
print(" [FN TP]]")

# ============================================================
# FINAL TEST SET EVALUATION
# ============================================================

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["label"]

# Predict previously unseen test subjects
y_test_pred = model.predict(X_test)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\n" + "=" * 50)
print("RANDOM FOREST - TEST RESULTS")
print("=" * 50)

print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")

print("\nConfusion Matrix:")
print(test_cm)

print("\nMatrix format:")
print("[[TN FP]")
print(" [FN TP]]")

# ============================================================
# OVERFITTING CHECK
# Compare Training vs Validation vs Test performance
# ============================================================

def evaluate_split(name, X, y):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print(f"\n{name}")
    print("-" * 35)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


print("\n" + "=" * 50)
print("OVERFITTING CHECK")
print("=" * 50)

evaluate_split(
    "TRAINING SET",
    X_train,
    y_train
)

evaluate_split(
    "VALIDATION SET",
    X_val,
    y_val
)

evaluate_split(
    "TEST SET",
    X_test,
    y_test
)

# ============================================================
# SAVE TRAINED FALL DETECTION MODEL
# ============================================================

import joblib

MODEL_PATH = Path(
    "ml-model/fall_detection/fall_detection_rf.joblib"
)

model_package = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "sampling_rate": 200,
    "window_size": 400,
    "window_seconds": 2
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\n" + "=" * 50)
print("MODEL SAVED")
print("=" * 50)
print("Model path:", MODEL_PATH)