from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("datasets/sisfall/SisFall_dataset")

SAMPLING_RATE = 200
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 400 samples


# ============================================================
# 1. LOAD AND PREPROCESS ONE SISFALL RECORDING
# ============================================================

def load_sisfall_file(file_path):
    """Load one SisFall recording and convert raw sensor values."""

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

    # Remove the semicolon at the end of each row
    df["mma_z"] = (
        df["mma_z"]
        .astype(str)
        .str.replace(";", "", regex=False)
        .astype(float)
    )

    # --------------------------------------------------------
    # Convert raw sensor values to physical units
    # --------------------------------------------------------

    # ADXL345: 13-bit, +/-16 g
    adxl_scale = (2 * 16) / (2 ** 13)

    df["adxl_x"] *= adxl_scale
    df["adxl_y"] *= adxl_scale
    df["adxl_z"] *= adxl_scale

    # ITG3200: 16-bit, +/-2000 degrees/second
    gyro_scale = (2 * 2000) / (2 ** 16)

    df["gyro_x"] *= gyro_scale
    df["gyro_y"] *= gyro_scale
    df["gyro_z"] *= gyro_scale

    # MMA8451Q: 14-bit, +/-8 g
    mma_scale = (2 * 8) / (2 ** 14)

    df["mma_x"] *= mma_scale
    df["mma_y"] *= mma_scale
    df["mma_z"] *= mma_scale

    # --------------------------------------------------------
    # Magnitude features
    # --------------------------------------------------------

    df["acc_mag"] = np.sqrt(
        df["adxl_x"] ** 2
        + df["adxl_y"] ** 2
        + df["adxl_z"] ** 2
    )

    df["gyro_mag"] = np.sqrt(
        df["gyro_x"] ** 2
        + df["gyro_y"] ** 2
        + df["gyro_z"] ** 2
    )

    return df


# ============================================================
# 2. EXTRACT 2-SECOND WINDOW FEATURES
# ============================================================

def extract_window_features(df, is_fall_recording=False):
    """
    Extract statistical features from 2-second windows.

    ADL recording:
        Keep every complete window as label 0.

    Fall recording:
        Estimate the impact using maximum acceleration magnitude.
        Keep only the 2-second window containing that impact
        and assign label 1.

    Important:
        SisFall does not provide sample-level fall boundaries,
        so the fall window is an estimated event window.
    """

    feature_rows = []

    fall_window_index = None

    if is_fall_recording:
        peak_sample = df["acc_mag"].idxmax()
        fall_window_index = peak_sample // WINDOW_SIZE

    window_index = 0

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        WINDOW_SIZE
    ):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        # ----------------------------------------------------
        # Determine label
        # ----------------------------------------------------

        if is_fall_recording:

            # Ignore uncertain pre-fall/post-fall windows
            if window_index != fall_window_index:
                window_index += 1
                continue

            label = 1

        else:
            label = 0

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = {
            "acc_mean": window["acc_mag"].mean(),
            "acc_std": window["acc_mag"].std(),
            "acc_min": window["acc_mag"].min(),
            "acc_max": window["acc_mag"].max(),
            "acc_range": (
                window["acc_mag"].max()
                - window["acc_mag"].min()
            ),

            "gyro_mean": window["gyro_mag"].mean(),
            "gyro_std": window["gyro_mag"].std(),
            "gyro_max": window["gyro_mag"].max(),

            "label": label
        }

        feature_rows.append(features)

        window_index += 1

    return pd.DataFrame(feature_rows)


# ============================================================
# 3. VERIFY DATASET LOCATION
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"SisFall dataset not found at: {DATASET_PATH}"
    )

print("SisFall dataset found.")


# ============================================================
# 4. FIND SUBJECT FOLDERS
# ============================================================

subject_folders = sorted(
    folder
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
    and (
        folder.name.startswith("SA")
        or folder.name.startswith("SE")
    )
)

print("Number of subject folders:", len(subject_folders))


# ============================================================
# 5. FIND ALL RECORDINGS
# ============================================================

recording_files = []

for subject_folder in subject_folders:
    recording_files.extend(
        sorted(subject_folder.glob("*.txt"))
    )

print("Total recording files:", len(recording_files))


# ============================================================
# 6. RECORDING-LEVEL CLASS SUMMARY
# ============================================================

fall_files = [
    file_path
    for file_path in recording_files
    if file_path.name.startswith("F")
]

non_fall_files = [
    file_path
    for file_path in recording_files
    if file_path.name.startswith("D")
]

print("\nRecording summary:")
print("Non-fall recordings:", len(non_fall_files))
print("Fall recordings:", len(fall_files))
print("Total:", len(recording_files))


# ============================================================
# 7. PROCESS COMPLETE DATASET
# ============================================================

print("\nProcessing complete SisFall dataset...")
print("-" * 50)

all_features = []

processed_files = 0
failed_files = 0

for file_path in recording_files:

    try:
        # Load recording
        df = load_sisfall_file(file_path)

        # Determine recording type
        is_fall_recording = file_path.name.startswith("F")

        # Extract features
        file_features = extract_window_features(
            df,
            is_fall_recording=is_fall_recording
        )

        # Safety check
        if file_features.empty:
            raise ValueError(
                "No complete windows were extracted."
            )

        # Add metadata
        file_features["subject"] = file_path.parent.name
        file_features["recording"] = file_path.name

        # Store result
        all_features.append(file_features)

        processed_files += 1

        if processed_files % 250 == 0:
            print(
                f"Processed "
                f"{processed_files}/{len(recording_files)} files..."
            )

    except Exception as error:

        failed_files += 1

        print(
            f"ERROR processing {file_path.name}: "
            f"{repr(error)}"
        )


# ============================================================
# 8. COMBINE ALL FEATURES
# ============================================================

if not all_features:
    raise RuntimeError(
        "No features were extracted from the dataset."
    )

full_features_df = pd.concat(
    all_features,
    ignore_index=True
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("PREPROCESSING SUMMARY")
print("=" * 50)

print("Successfully processed files:", processed_files)
print("Failed files:", failed_files)

print("\nFeature dataset shape:")
print(full_features_df.shape)

print("\nWindow label counts:")
print(
    full_features_df["label"]
    .value_counts()
    .sort_index()
)

print("\nWindow label percentages:")
print(
    full_features_df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nUnique subjects:")
print(full_features_df["subject"].nunique())

print("\nFirst 5 rows:")
print(
    full_features_df
    .head()
    .to_string(index=False)
)

# ============================================================
# 10. SAVE PROCESSED FEATURE DATASET
# ============================================================

OUTPUT_PATH = Path(
    "datasets/sisfall/sisfall_features.csv"
)

full_features_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nProcessed feature dataset saved to:")
print(OUTPUT_PATH)