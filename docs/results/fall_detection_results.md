# CareCopilot Fall Detection Results

## Dataset

Dataset: SisFall

Sampling frequency: 200 Hz

Window duration: 2 seconds

Window size: 400 samples

Total processed recordings: 4505

Successfully processed recordings: 4505

Failed recordings: 0

Total feature windows: 27540

Non-fall windows: 25742

Fall-event windows: 1798

Number of subjects: 38


## Features

The Random Forest uses the following features:

- acc_mean
- acc_std
- acc_min
- acc_max
- acc_range
- gyro_mean
- gyro_std
- gyro_max


## Model

Algorithm: Random Forest Classifier

Number of trees: 200

Class weighting: Balanced


## Subject-Grouped Cross-Validation

5-fold GroupKFold was used so that data from the same subject
did not appear in both training and validation within a fold.

Mean Accuracy: 98.65% ± 0.59%
Mean Precision: 86.37% ± 6.13%
Mean Recall: 94.69% ± 2.10%
Mean F1 Score: 90.24% ± 3.71%

Subject leakage check: PASSED for all folds.


## Inference Latency

Number of predictions tested: 1000

Average model inference latency: 51.506 ms

Minimum latency: 29.057 ms

Maximum latency: 100.312 ms

These measurements represent model inference latency only and
do not include sensor acquisition, network transmission, feature
extraction, backend processing, or alert delivery.


## Important Limitation

SisFall fall recordings were converted into fall-event windows
using the window containing the maximum acceleration magnitude.

Therefore, the positive window labels are derived using a
peak-impact heuristic rather than independent sample-level
fall-event boundary annotations.

The reported results represent prototype fall-detection
performance on the processed SisFall dataset and should not be
interpreted as clinical validation.