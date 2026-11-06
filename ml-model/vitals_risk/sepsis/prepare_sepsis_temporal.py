
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW = Path("datasets/vitals/sepsis_training")
OUT = Path("datasets/vitals/sepsis_processed")
OUT.mkdir(parents=True, exist_ok=True)

SENSORS = ["HR", "O2Sat", "Temp"]

# Use the existing 400 training patients only.
patients = pd.read_csv(RAW / "train_patients.csv")
held_out = pd.read_csv(RAW / "test_patients.csv")

assert set(patients["patient"]).isdisjoint(
    set(held_out["patient"])
)

# Create an inner patient-wise train/validation split.
train_ids, val_ids = train_test_split(
    patients["patient"],
    test_size=0.20,
    random_state=42,
    stratify=patients["patient_positive"]
)

def prepare_patient(patient_id):
    path = RAW / f"{patient_id}.psv"
    df = pd.read_csv(path, sep="|")

    df = df[SENSORS + ["SepsisLabel"]].copy()
    df["patient"] = patient_id
    df["hour"] = np.arange(len(df))

    # All features use the current or earlier hours.
    for sensor in SENSORS:
        df[sensor] = pd.to_numeric(
            df[sensor], errors="coerce"
        )

        df[f"{sensor}_missing"] = (
            df[sensor].isna().astype(int)
        )

        # Previous-hour measurement.
        df[f"{sensor}_lag1"] = (
            df[sensor].shift(1)
        )

        # Change from previous hour.
        df[f"{sensor}_delta1"] = (
            df[sensor] - df[f"{sensor}_lag1"]
        )

        # Current hour + up to 5 preceding hours.
        df[f"{sensor}_rolling6_mean"] = (
            df[sensor]
            .rolling(window=6, min_periods=1)
            .mean()
        )

        # Number of available readings in the window.
        df[f"{sensor}_rolling6_count"] = (
            df[sensor]
            .rolling(window=6, min_periods=1)
            .count()
        )

    return df

def prepare_split(patient_ids):
    frames = [
        prepare_patient(patient_id)
        for patient_id in patient_ids
    ]
    return pd.concat(frames, ignore_index=True)

train = prepare_split(train_ids)
val = prepare_split(val_ids)

assert set(train["patient"]).isdisjoint(
    set(val["patient"])
)

assert not set(train["patient"]).intersection(
    set(held_out["patient"])
)

assert not set(val["patient"]).intersection(
    set(held_out["patient"])
)

train.to_csv(
    OUT / "temporal_train.csv",
    index=False
)

val.to_csv(
    OUT / "temporal_val.csv",
    index=False
)

features = [
    col for col in train.columns
    if col not in ["patient", "hour", "SepsisLabel"]
]

print("\n--- V3.1 TEMPORAL PREPROCESSING ---")
print("Train patients:", train["patient"].nunique())
print("Validation patients:", val["patient"].nunique())
print("Train observations:", len(train))
print("Validation observations:", len(val))
print("Train positive observations:",
      int(train["SepsisLabel"].sum()))
print("Validation positive observations:",
      int(val["SepsisLabel"].sum()))
print("Feature count:", len(features))
print("Expected missing feature values:",
      int(train[features].isna().sum().sum()))
print("Patient overlap:",
      len(set(train["patient"]) & set(val["patient"])))
print("Held-out test patients used: 0")
print("\nFiles saved:")
print(OUT / "temporal_train.csv")
print(OUT / "temporal_val.csv")
