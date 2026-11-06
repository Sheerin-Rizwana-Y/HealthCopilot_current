
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path("datasets/vitals/sepsis_training")
summary = pd.read_csv(DATA / "validation_summary.csv")

# Patient-level target: whether any hourly observation
# for this patient has a positive SepsisLabel.
summary["patient_positive"] = (
    summary["positive_rows"] > 0
).astype(int)

train, test = train_test_split(
    summary,
    test_size=0.20,
    random_state=42,
    stratify=summary["patient_positive"]
)

assert set(train["patient"]).isdisjoint(set(test["patient"]))

train.to_csv(DATA / "train_patients.csv", index=False)
test.to_csv(DATA / "test_patients.csv", index=False)

print("\n--- PATIENT-WISE SPLIT ---")
print("Train patients:", len(train))
print("Test patients:", len(test))
print("Train positive patients:",
      train["patient_positive"].sum())
print("Test positive patients:",
      test["patient_positive"].sum())
print("Patient overlap:",
      len(set(train["patient"]) & set(test["patient"])))
