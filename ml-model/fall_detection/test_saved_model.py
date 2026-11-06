from pathlib import Path
import joblib


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = Path(
    "ml-model/fall_detection/fall_detection_rf.joblib"
)


# ============================================================
# CHECK MODEL EXISTS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
feature_columns = model_package["feature_columns"]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 50)
print("SAVED FALL MODEL CHECK")
print("=" * 50)

print("\nModel loaded successfully.")

print("\nModel type:")
print(type(model).__name__)

print("\nExpected features:")
for feature in feature_columns:
    print("-", feature)

print("\nSampling rate:")
print(model_package["sampling_rate"], "Hz")

print("\nWindow size:")
print(model_package["window_size"], "samples")

print("\nWindow duration:")
print(model_package["window_seconds"], "seconds")

print("\nNumber of trees:")
print(model.n_estimators)

print("\nSaved model verification: PASSED")

# ============================================================
# TEST REAL PREDICTIONS
# ============================================================

import pandas as pd

FEATURE_DATA_PATH = Path(
    "datasets/sisfall/sisfall_features.csv"
)

df = pd.read_csv(FEATURE_DATA_PATH)


# Select one known non-fall example
non_fall_sample = df[df["label"] == 0].iloc[[0]]

# Select one known fall example
fall_sample = df[df["label"] == 1].iloc[[0]]


def test_prediction(sample, sample_name):

    X = sample[feature_columns]

    actual_label = int(sample["label"].iloc[0])

    predicted_label = int(
        model.predict(X)[0]
    )

    probabilities = model.predict_proba(X)[0]

    non_fall_probability = probabilities[0]
    fall_probability = probabilities[1]

    print("\n" + "=" * 50)
    print(sample_name)
    print("=" * 50)

    print("Recording:",
          sample["recording"].iloc[0])

    print("Subject:",
          sample["subject"].iloc[0])

    print("Actual label:",
          actual_label)

    print("Predicted label:",
          predicted_label)

    print(
        f"Non-fall probability: "
        f"{non_fall_probability:.4f}"
    )

    print(
        f"Fall probability: "
        f"{fall_probability:.4f}"
    )

    if predicted_label == actual_label:
        print("Prediction: CORRECT")
    else:
        print("Prediction: INCORRECT")


test_prediction(
    non_fall_sample,
    "NON-FALL SAMPLE TEST"
)

test_prediction(
    fall_sample,
    "FALL SAMPLE TEST"
)

# ============================================================
# INFERENCE LATENCY TEST
# ============================================================

import time

X_latency = fall_sample[feature_columns]

NUM_RUNS = 1000

# Warm-up
for _ in range(10):
    model.predict(X_latency)

latencies_ms = []

for _ in range(NUM_RUNS):

    start_time = time.perf_counter()

    model.predict(X_latency)

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    latencies_ms.append(latency_ms)


print("\n" + "=" * 50)
print("MODEL INFERENCE LATENCY")
print("=" * 50)

print("Number of predictions:", NUM_RUNS)

print(
    f"Average latency: "
    f"{sum(latencies_ms) / len(latencies_ms):.3f} ms"
)

print(
    f"Minimum latency: "
    f"{min(latencies_ms):.3f} ms"
)

print(
    f"Maximum latency: "
    f"{max(latencies_ms):.3f} ms"
)