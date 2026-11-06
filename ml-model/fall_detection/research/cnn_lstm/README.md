# CNN-LSTM fall detection (research model)

An **additional, experimental** fall detector for Lifora, trained on SisFall. It lives beside the
existing Random Forest (RF) and does not replace, wrap, or ensemble with it.

| | Random Forest (existing baseline) | CNN-LSTM (this folder) |
|---|---|---|
| Input | 8 statistical features per window | raw sequence, 6 channels x 400 samples |
| Code | `ml-model/fall_detection/*.py` (untouched) | `ml-model/fall_detection/research/cnn_lstm/` |
| Artifact | `fall_detection_rf.joblib` (untouched) | `research/cnn_lstm/artifacts/cnn_lstm_fall_detector.pt` |
| Backend / alerts | not connected (`ml_service.py` reports `not_connected`) | **not connected; predictions carry `alert_eligible: false`** |

## Status - read this first

* **Trained on real SisFall: NO.** The SisFall dataset was not in the supplied project, so only the code,
  tests and a *synthetic* end-to-end smoke test have been run. No real accuracy figure exists yet.
* Synthetic runs are labelled `SYNTHETIC` in every console output, plot, JSON, report and in the checkpoint itself.
* Do not claim the architecture is effective until `compare_models.py` has been run on the real dataset.

## Architecture

```
SisFall .txt (9 integer columns, 200 Hz)        adxl_x,y,z | gyro_x,y,z | mma_x,y,z
  -> physical units (g, deg/s)                  same scale factors as the RF
  -> 6 channels (adxl + gyro; configurable)     the sensors the RF features come from
  -> 400-sample (2 s) windows                   same tiling/labels as the RF
  -> per-channel z-score (TRAIN statistics only)
  -> 3 x [Conv1d -> BatchNorm -> ReLU -> MaxPool(2)]   k = 7, 5, 3 ; 32/64/64 filters  (local movement patterns)
  -> Dropout(0.3)  -> sequence [50 steps x 64 features]
  -> LSTM(64)                                    temporal dependencies across the window
  -> last hidden state -> Dropout -> Linear(32) -> ReLU -> Linear(1)
  -> sigmoid -> P(fall)   (1 = fall, 0 = ADL, as in the RF)   ~59.7 k parameters
```

Training: AdamW, BCE-with-logits with `pos_weight = n_ADL/n_fall`, LR-on-plateau, gradient clipping, early stopping
on validation PR-AUC, fixed seeds, CPU. The decision threshold is the F1-optimal one **on the validation subjects**.

## Data, labels, leakage

* Windows and labels replicate `preprocess_sisfall.py`: ADL files -> every complete non-overlapping tile (label 0);
  fall files -> only the tile containing the peak ADXL magnitude (label 1). SisFall has no sample-level fall labels, so
  fall windows are *estimated* event windows (label noise affects both models).
* Split by **subject** with the RF's exact `train_test_split` calls (seed 42): 26 train / 6 val / 6 test of 38 subjects.
  If `datasets/sisfall/sisfall_features.csv` exists its subject order is copied (read-only) to guarantee the identical split.
* Val/test contain only the RF-identical windows. Extra impact-centred fall crops (default 8 per fall file) and optional
  ADL overlap exist **only for training subjects**, so nothing overlaps across splits. Normalisation uses train windows only.
* The RF script cannot be imported (it runs its pipeline at import), so its window/feature logic is re-implemented here.
  `tests/test_data_pipeline.py::test_parity_with_existing_rf_preprocessing` executes the RF's own functions
  (parsed out of the script, not imported) and checks that the windows, labels and features match.

## Install

From the repository root (CPU PyTorch is enough):

```powershell
pip install -r ml-model/fall_detection/research/cnn_lstm/requirements.txt
```

## Run (from the repository root; SisFall expected at `datasets/sisfall/SisFall_dataset`)

```powershell
$C = "ml-model/fall_detection/research/cnn_lstm"
python $C/prepare_data.py                     # -> datasets/sisfall/cnn_lstm_windows/  (~1 GB of float32 windows)
python $C/train.py                            # -> $C/artifacts/cnn_lstm_fall_detector.pt, history, curves
python $C/evaluate.py --split test            # CNN-LSTM metrics, confusion matrix, ROC/PR
python $C/compare_models.py                   # RF vs CNN-LSTM -> $C/artifacts/comparison/evaluation_report.md
python $C/predict.py --file datasets/sisfall/SisFall_dataset/SA01/F05_SA01_R01.txt
python $C/sanity_checks.py --all              # sanity checks on the result (several trainings; see docstring)
python -m pytest -q $C/tests                  # unit tests + synthetic smoke tests (no dataset needed)
```

Memory: windows are memory-mapped and read lazily. To shrink further set `adl_max_windows_per_recording` or
`fall_aug_crops` in `config.yaml`. Use `--dataset-root/--out/--artifacts-dir` to redirect any output
(e.g. when experimenting, so a good checkpoint is not overwritten by a re-train).

### Prediction API (for later integration, not wired in)

```python
from predict import FallDetectorCNNLSTM          # add this folder to sys.path
det = FallDetectorCNNLSTM.load(".../artifacts/cnn_lstm_fall_detector.pt")
det.predict_window(window)   # window: [400, 6] adxl g + gyro deg/s (or [400, 9] SisFall order)
# -> {"model": "cnn_lstm", "experimental": True, "alert_eligible": False, "label": "fall"|"adl",
#     "fall_probability": 0.87, "threshold": 0.61, "training_data_source": "sisfall"}
```

## Comparison protocol (`compare_models.py`)

Both models score the identical held-out test windows (the RF's own 6 test subjects). The RF `.joblib` is loaded
read-only and its eight features recomputed from the raw windows. Reported for each model at the default 0.5 cut-off and at
the validation-tuned cut-off: accuracy, precision, recall, specificity, F1, FPR, ROC-AUC, PR-AUC, confusion matrix, false alarms
per hour of lab ADL recording, single-window CPU latency, and model size. If the split is not the RF's own, the report
warns that the RF may have seen the test subjects. **Limits:** 6 test subjects, estimated fall windows, one seed, no confidence
intervals - repeat with several seeds / grouped cross-validation before drawing conclusions. Lab ADL trials do not represent
real-life false-alarm rates.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | all paths and hyper-parameters |
| `data_pipeline.py` | loader, windowing, RF-identical labels, subject split, normalisation |
| `prepare_data.py` | builds train/val/test window files + `prepared_meta.json` |
| `model.py` | CNN-LSTM network, checkpoint save/load |
| `train.py` | training loop, early stopping, history, curves |
| `evaluate.py` | metrics, plots, held-out evaluation |
| `predict.py` | independent, research-only prediction interface |
| `compare_models.py` | RF vs CNN-LSTM table, plots, report |
| `sanity_checks.py` | per-activity errors, non-impact windows, multi-seed, shuffled-label and sensor-ablation checks |
| `synthetic.py` | SisFall-format fake data for software tests only |
| `tests/` | 25 tests |
| `artifacts/` | checkpoint, history, plots, reports (empty until you train) |

## Promotion checklist (future, needs your explicit decision)

Real-data comparison reviewed -> multi-seed / grouped-CV confirmation -> tested on the target wearable's sensor and
sampling rate (SisFall is a waist-worn lab dataset) -> then, and only then, discuss wiring. No ensemble or automatic replacement is implemented.
