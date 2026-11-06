
import pandas as pd
from pathlib import Path

DATA_DIR = Path("datasets/vitals/raw")

signals = ["hr", "spo2", "temperature"]

for signal in signals:
    file_path = DATA_DIR / f"case_1_{signal}.csv"

    print(f"\n{'=' * 50}")
    print(f"SIGNAL: {signal.upper()}")
    print(f"FILE: {file_path}")
    print("=" * 50)

    df = pd.read_csv(file_path,compression="gzip")

    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumeric summary:")
    print(df.describe(include="all").to_string())
