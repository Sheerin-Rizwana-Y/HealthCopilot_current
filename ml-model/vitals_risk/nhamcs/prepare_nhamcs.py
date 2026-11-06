
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("datasets/vitals/nhamcs/2022")
files = list((ROOT / "extracted").rglob("*.dta"))

if len(files) != 1:
    raise RuntimeError(
        f"Expected one .dta file, found {len(files)}"
    )

df = pd.read_stata(
    files[0],
    convert_categoricals=False,
    columns=["PULSE", "POPCT", "TEMPF", "IMMEDR"]
)

# Apply documented missing/special codes.
df["PULSE"] = df["PULSE"].replace(
    {-9: np.nan, 998: np.nan}
)
df["POPCT"] = df["POPCT"].replace(
    {-9: np.nan}
)
df["TEMPF"] = df["TEMPF"].replace(
    {-9: np.nan}
)

# NHAMCS stores Fahrenheit * 10.
fahrenheit = df["TEMPF"] / 10
df["temperature_c"] = (fahrenheit - 32) * 5 / 9

# Keep only actual recorded triage levels.
df = df[df["IMMEDR"].isin([1, 2, 3, 4, 5])].copy()

# Experimental ED urgency proxy, not clinical
# ground truth for wearable physiological risk.
mapping = {
    1: "High",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "Low",
}

df["risk_label"] = df["IMMEDR"].map(mapping)

# Require all three sensor measurements for this
# first dataset-quality audit.
features = ["PULSE", "POPCT", "temperature_c"]
complete = df.dropna(subset=features).copy()

complete = complete.rename(columns={
    "PULSE": "hr",
    "POPCT": "spo2",
    "temperature_c": "temperature"
})

output = ROOT / "processed"
output.mkdir(parents=True, exist_ok=True)

complete[
    ["hr", "spo2", "temperature", "IMMEDR", "risk_label"]
].to_csv(output / "nhamcs_2022_clean.csv", index=False)

print("\n--- NHAMCS V4 CLEANING REPORT ---")
print("Original records:", 16025)
print("Records with valid triage level:", len(df))
print("Complete three-sensor records:", len(complete))

print("\nRisk-label distribution:")
print(complete["risk_label"].value_counts())

print("\nSensor ranges after code cleaning:")
print(complete[["hr", "spo2", "temperature"]].describe())

print("\nSaved:", output / "nhamcs_2022_clean.csv")
