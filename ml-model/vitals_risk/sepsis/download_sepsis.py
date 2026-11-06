
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import pandas as pd
import time

OUTPUT = Path("datasets/vitals/sepsis_training")
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://physionet.org/files/"
    "challenge-2019/1.0.0/training/training_setA"
)

records = []

for patient_id in range(1, 501):
    filename = f"p{patient_id:06d}.psv"
    destination = OUTPUT / filename

    if not destination.exists():
        try:
            urlretrieve(f"{BASE_URL}/{filename}", destination)
        except (HTTPError, URLError) as error:
            print(f"Download failed: {filename}: {error}")
            continue

        time.sleep(0.2)

    try:
        df = pd.read_csv(destination, sep="|")

        required = {"HR", "O2Sat", "Temp", "SepsisLabel"}
        if not required.issubset(df.columns):
            print(f"Missing columns: {filename}")
            continue

        records.append({
            "patient": filename,
            "rows": len(df),
            "positive_rows": int(
                (df["SepsisLabel"] == 1).sum()
            ),
            "complete_rows": int(
                df[["HR", "O2Sat", "Temp"]]
                .notna()
                .all(axis=1)
                .sum()
            )
        })

        print(f"Processed {filename}")

    except Exception as error:
        print(f"Invalid file {filename}: {error}")

summary = pd.DataFrame(records)

if summary.empty:
    raise RuntimeError("No valid patient files found.")

summary.to_csv(OUTPUT / "sample_summary.csv", index=False)

print("\n--- DATASET SUMMARY ---")
print("Patients:", len(summary))
print("Total observations:", summary["rows"].sum())
print("Sepsis-positive patients:",
      (summary["positive_rows"] > 0).sum())
print("Sepsis-positive observations:",
      summary["positive_rows"].sum())
print("Complete three-sensor observations:",
      summary["complete_rows"].sum())
