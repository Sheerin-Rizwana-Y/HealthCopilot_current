
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

DATA = Path("datasets/vitals/sepsis_training")
OUTPUT = Path("datasets/vitals/sepsis_processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

SENSORS = ["HR", "O2Sat", "Temp"]
FEATURES = SENSORS + [f"{c}_missing" for c in SENSORS]

def load_patients(split_file):
    patients = pd.read_csv(DATA / split_file)["patient"]
    frames = []

    for patient in patients:
        path = DATA / f"{patient}.psv"
        df = pd.read_csv(path, sep="|")

        frame = df[SENSORS + ["SepsisLabel"]].copy()
        frame["patient"] = patient
        frame["hour"] = np.arange(len(frame))

        for sensor in SENSORS:
            frame[f"{sensor}_missing"] = (
                frame[sensor].isna().astype(int)
            )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)

train = load_patients("train_patients.csv")
test = load_patients("test_patients.csv")

assert set(train["patient"]).isdisjoint(set(test["patient"]))

imputer = SimpleImputer(strategy="median")

train[SENSORS] = imputer.fit_transform(train[SENSORS])
test[SENSORS] = imputer.transform(test[SENSORS])

train.to_csv(OUTPUT / "train.csv", index=False)
test.to_csv(OUTPUT / "test.csv", index=False)

joblib.dump(
    {
        "imputer": imputer,
        "features": FEATURES,
        "sensors": SENSORS,
    },
    OUTPUT / "preprocessor.joblib"
)

print("\n--- V3 PREPROCESSING ---")
print("Train observations:", len(train))
print("Test observations:", len(test))
print("Train positive observations:", int(train["SepsisLabel"].sum()))
print("Test positive observations:", int(test["SepsisLabel"].sum()))
print("Missing feature values:",
      int(train[FEATURES].isna().sum().sum()
          + test[FEATURES].isna().sum().sum()))
print("Feature columns:", FEATURES)
print("Patient overlap:",
      len(set(train["patient"]) & set(test["patient"])))
