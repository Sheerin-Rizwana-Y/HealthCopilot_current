from pathlib import Path
import pandas as pd

file_path = Path("datasets/sisfall/SisFall_dataset/SA01/F01_SA01_R01.txt")

columns = [
    "adxl_x", "adxl_y", "adxl_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mma_x", "mma_y", "mma_z"
]
df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    sep=",",
    engine="python"
)

df["mma_z"] = (
    df["mma_z"]
    .astype(str)
    .str.replace(";", "", regex=False)
    .astype(float)
)

print(df.head())
print("Shape:", df.shape)

print("\nData types:")
print(df.dtypes)

#Convertion of raw sensor values
# Convert raw sensor values to physical units

# ADXL345: 13-bit, ±16g
adxl_scale = (2 * 16) / (2 ** 13)

df["adxl_x"] = df["adxl_x"] * adxl_scale
df["adxl_y"] = df["adxl_y"] * adxl_scale
df["adxl_z"] = df["adxl_z"] * adxl_scale

# ITG3200: 16-bit, ±2000 deg/s
gyro_scale = (2 * 2000) / (2 ** 16)

df["gyro_x"] = df["gyro_x"] * gyro_scale
df["gyro_y"] = df["gyro_y"] * gyro_scale
df["gyro_z"] = df["gyro_z"] * gyro_scale

# MMA8451Q: 14-bit, ±8g
mma_scale = (2 * 8) / (2 ** 14)

df["mma_x"] = df["mma_x"] * mma_scale
df["mma_y"] = df["mma_y"] * mma_scale
df["mma_z"] = df["mma_z"] * mma_scale

print(df.head())

#CRERATION OF ACCLERATION MAGNITUDE FEATURE
import numpy as np

df["acc_mag"] = np.sqrt(
    df["adxl_x"]**2 +
    df["adxl_y"]**2 +
    df["adxl_z"]**2
)

# Gyroscope magnitude
df["gyro_mag"] = np.sqrt(
    df["gyro_x"]**2 +
    df["gyro_y"]**2 +
    df["gyro_z"]**2
)

print(df[["adxl_x", "adxl_y", "adxl_z", "acc_mag"]].head())

#MAXIMUM ACCELERATION MAGNITUDE
max_index = df["acc_mag"].idxmax()
max_acc = df.loc[max_index, "acc_mag"]

print("\nMaximum acceleration magnitude:")
print("Sample:", max_index)
print("Acceleration:", round(max_acc, 3), "g")

#WINDOWING OF SENSOR DATA
WINDOW_SIZE = 400

windows = []

for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
    end = start + WINDOW_SIZE
    window = df.iloc[start:end]
    windows.append(window)

print("\nWindowing:")
print("Window size:", WINDOW_SIZE, "samples")
print("Number of windows:", len(windows))
feature_rows = []

for i, window in enumerate(windows):
    features = {
        "window": i + 1,
        "acc_mean": window["acc_mag"].mean(),
        "acc_std": window["acc_mag"].std(),
        "acc_min": window["acc_mag"].min(),
        "acc_max": window["acc_mag"].max(),
        "acc_range": window["acc_mag"].max() - window["acc_mag"].min(),

        # Gyroscope features
        "gyro_mean": window["gyro_mag"].mean(),
        "gyro_std": window["gyro_mag"].std(),
        "gyro_max": window["gyro_mag"].max()
    }

    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

print("\nExtracted features:")
print(features_df.to_string(index=False))