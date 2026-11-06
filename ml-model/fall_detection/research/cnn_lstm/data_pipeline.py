"""SisFall loading, windowing, participant-independent splitting and normalisation
for the CNN-LSTM research model.

This module is a SEPARATE implementation from the Random Forest pipeline
(ml-model/fall_detection/preprocess_sisfall.py). It does not import or modify it.
(That script executes its whole pipeline at import time, so it cannot be imported.)
Instead it reproduces, line for line, the RF's conventions so that both models can be
evaluated on identical windows:

* file format     : 9 comma-separated integers per line, last value ends in ';'
* columns         : adxl_x,y,z | gyro_x,y,z | mma_x,y,z
* unit conversion : ADXL345 +-16 g/13 bit, ITG3200 +-2000 deg/s/16 bit, MMA8451Q +-8 g/14 bit
* window          : 400 samples (2 s @ 200 Hz), non-overlapping tiles starting at sample 0
* labels          : 0 = ADL ("D*" files, every complete tile)
                    1 = fall ("F*" files, ONLY the tile containing the peak ADXL
                        acceleration magnitude; SisFall has no sample-level fall bounds)
* subject split   : sklearn train_test_split on the subject list, random_state=42,
                    test_size=12 then 6  ->  26 train / 6 val / 6 test subjects
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------- constants
ALL_COLUMNS = [
    "adxl_x", "adxl_y", "adxl_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mma_x", "mma_y", "mma_z",
]
LABEL_ADL = 0
LABEL_FALL = 1
LABEL_NAMES = {LABEL_ADL: "adl", LABEL_FALL: "fall"}

ADXL_SCALE = (2 * 16) / (2 ** 13)      # g per raw count
GYRO_SCALE = (2 * 2000) / (2 ** 16)    # deg/s per raw count
MMA_SCALE = (2 * 8) / (2 ** 14)        # g per raw count
_SCALES = np.array([ADXL_SCALE] * 3 + [GYRO_SCALE] * 3 + [MMA_SCALE] * 3, dtype=np.float64)

RECORDING_RE = re.compile(r"^(?P<activity>[FD]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")
SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- config / paths
def find_project_root(start: Path | None = None) -> Path:
    """Walk up from this file until the directory that contains `ml-model/` is found."""
    here = Path(start or __file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "ml-model").is_dir():
            return parent
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    d = cfg["data"]
    if d["window_size"] != d["sampling_rate"] * d["window_seconds"]:
        raise ValueError("data.window_size must equal sampling_rate * window_seconds")
    unknown = [c for c in d["channels"] if c not in ALL_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown channels {unknown}; valid: {ALL_COLUMNS}")
    return cfg


def resolve(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root or find_project_root()) / p


def channel_indices(channels: Iterable[str]) -> list[int]:
    return [ALL_COLUMNS.index(c) for c in channels]


# --------------------------------------------------------------------------- recordings
@dataclass(frozen=True)
class RecordingInfo:
    path: Path
    subject: str
    activity: str
    trial: str

    @property
    def is_fall(self) -> bool:
        return self.activity.startswith("F")

    @property
    def label(self) -> int:
        return LABEL_FALL if self.is_fall else LABEL_ADL

    @property
    def name(self) -> str:
        return self.path.name


def parse_recording_name(path: Path) -> RecordingInfo | None:
    m = RECORDING_RE.match(path.name)
    if not m:
        return None
    return RecordingInfo(path, m["subject"], m["activity"], m["trial"])


def load_sisfall_recording(path: str | Path) -> np.ndarray:
    """Read one SisFall .txt file -> float64 array (n_samples, 9) in physical units
    (g for accelerometers, deg/s for the gyroscope). Row order is preserved."""
    text = Path(path).read_text(encoding="utf-8", errors="strict").replace(";", "")
    df = pd.read_csv(io.StringIO(text), header=None, sep=",", skipinitialspace=True)
    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(f"{path}: expected {len(ALL_COLUMNS)} columns, found {df.shape[1]}")
    arr = df.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: contains non-finite values")
    return arr * _SCALES


def accel_magnitude(sig: np.ndarray) -> np.ndarray:
    """ADXL345 magnitude (same quantity the RF's `acc_mag` uses)."""
    return np.sqrt((sig[:, 0:3] ** 2).sum(axis=1))


def discover_recordings(dataset_root: Path) -> tuple[list[RecordingInfo], list[Path]]:
    """All SA*/SE* subject recordings (sorted, like the RF script) + ignored files."""
    recs: list[RecordingInfo] = []
    ignored: list[Path] = []
    folders = sorted(f for f in Path(dataset_root).iterdir()
                     if f.is_dir() and (f.name.startswith("SA") or f.name.startswith("SE")))
    for folder in folders:
        for f in sorted(folder.glob("*.txt")):
            info = parse_recording_name(f)
            (recs if info else ignored).append(info or f)
    return recs, ignored


# --------------------------------------------------------------------------- windowing
def tile_starts(n_samples: int, window: int, stride: int | None = None) -> list[int]:
    stride = stride or window
    return list(range(0, n_samples - window + 1, stride))


def eval_windows(sig: np.ndarray, is_fall: bool, window: int) -> tuple[np.ndarray, np.ndarray]:
    """RF-identical windows.

    ADL  : every complete non-overlapping tile, label 0.
    Fall : only the tile containing the first maximum of ADXL magnitude, label 1
           (empty if that tile is incomplete - the RF treats that file as failed).
    Returns (windows [k, window, 9], starts [k]).
    """
    if is_fall:
        peak = int(np.argmax(accel_magnitude(sig)))
        start = (peak // window) * window
        starts = [start] if start + window <= len(sig) else []
    else:
        starts = tile_starts(len(sig), window)
    wins = np.stack([sig[s:s + window] for s in starts]) if starts else np.empty((0, window, sig.shape[1]))
    return wins, np.asarray(starts, dtype=np.int64)


def fall_augment_windows(sig: np.ndarray, window: int, n_crops: int, margin: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """TRAIN-ONLY extra fall windows: crops that all contain the impact sample, with the
    impact at different positions in the window (position drawn from [margin, window-margin])."""
    if n_crops <= 0 or len(sig) < window:
        return np.empty((0, window, sig.shape[1])), np.empty(0, dtype=np.int64)
    peak = int(np.argmax(accel_magnitude(sig)))
    starts = []
    for pos in rng.integers(margin, window - margin, size=n_crops):
        starts.append(int(np.clip(peak - pos, 0, len(sig) - window)))
    return np.stack([sig[s:s + window] for s in starts]), np.asarray(starts, dtype=np.int64)


# --------------------------------------------------------------------------- split
def subject_order(dataset_subjects: list[str], rf_csv: Path | None) -> tuple[list[str], str]:
    """Subject list in the SAME ORDER the RF script used (`df['subject'].unique()`).
    If the RF's feature CSV is available its order is copied (read-only); otherwise the
    sorted subject-folder order is used, which is what the RF CSV order is by construction."""
    if rf_csv is not None and Path(rf_csv).exists():
        order = pd.read_csv(rf_csv, usecols=["subject"])["subject"].unique().tolist()
        if set(order) == set(dataset_subjects):
            return order, "rf_features_csv"
    return sorted(dataset_subjects), "sorted_folder_names"


def split_subjects(subjects: list[str], n_val: int = 6, n_test: int = 6,
                   seed: int = 42) -> dict[str, list[str]]:
    """Participant-independent split using the identical sklearn calls as the RF script
    (train_test_split(subjects, test_size=n_val+n_test), then (temp, test_size=n_test))."""
    from sklearn.model_selection import train_test_split
    arr = np.array(subjects, dtype=object)
    train, temp = train_test_split(arr, test_size=n_val + n_test, random_state=seed)
    val, test = train_test_split(temp, test_size=n_test, random_state=seed)
    out = {"train": list(train), "val": list(val), "test": list(test)}
    assert set(out["train"]).isdisjoint(out["val"]) and set(out["train"]).isdisjoint(out["test"]) \
        and set(out["val"]).isdisjoint(out["test"]), "subject leakage"
    return out


def scaled_split_sizes(n_subjects: int, n_val: int, n_test: int) -> tuple[int, int, bool]:
    """With the full 38-subject SisFall the configured 6/6 is used (RF-identical). With a
    smaller (e.g. synthetic) subject set the held-out sizes are scaled proportionally."""
    if n_subjects == 38:
        return n_val, n_test, True
    nv = max(1, round(n_subjects * n_val / 38))
    nt = max(1, round(n_subjects * n_test / 38))
    while n_subjects - nv - nt < 1:
        nv, nt = max(1, nv - 1), max(1, nt - 1)
    return nv, nt, False


# --------------------------------------------------------------------------- prepared data
def load_prepared_meta(prepared_dir: Path) -> dict:
    with open(Path(prepared_dir) / "prepared_meta.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_split(prepared_dir: Path, split: str) -> tuple[np.memmap, pd.DataFrame]:
    """Memory-mapped windows [N, C, T] float32 plus the per-window metadata table."""
    prepared_dir = Path(prepared_dir)
    meta = load_prepared_meta(prepared_dir)
    n = meta["splits"][split]["n_windows"]
    c, t = len(meta["channels"]), meta["window_size"]
    info = pd.read_csv(prepared_dir / f"{split}_meta.csv")
    if n == 0:
        return np.empty((0, c, t), dtype=np.float32), info
    X = np.memmap(prepared_dir / f"{split}_X.f32", dtype=np.float32, mode="r", shape=(n, c, t))
    if len(info) != n:
        raise ValueError(f"{split}: {n} windows but {len(info)} metadata rows")
    return X, info


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score. x is [..., C, T]; mean/std have shape [C]."""
    m = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(-1, 1)
    return (x - m) / s
