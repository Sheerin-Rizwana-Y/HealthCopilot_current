"""Independent prediction interface for the EXPERIMENTAL CNN-LSTM fall detector.

    from predict import FallDetectorCNNLSTM
    det = FallDetectorCNNLSTM.load("ml-model/fall_detection/research/cnn_lstm/artifacts/cnn_lstm_fall_detector.pt")
    result = det.predict_window(window)          # window: [400, 6] (adxl xyz g, gyro xyz deg/s)

    python predict.py --file datasets/sisfall/SisFall_dataset/SA01/F05_SA01_R01.txt

RESEARCH ONLY. Every result carries `alert_eligible: False`. This module is not imported by
the backend and must not be connected to emergency alerts or caregiver notifications until
the model has been validated and explicitly approved. The Random Forest remains the baseline.

Input contract: physical units, columns in the order of the checkpoint's `channels`
(default adxl_x, adxl_y, adxl_z [g], gyro_x, gyro_y, gyro_z [deg/s]); a full 9-column SisFall
window (adxl, gyro, mma) is also accepted and the needed channels are selected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402
from model import CHECKPOINT_NAME, load_checkpoint  # noqa: E402


class FallDetectorCNNLSTM:
    def __init__(self, model: torch.nn.Module, ckpt: dict):
        self.model = model.eval()
        self.channels: list[str] = ckpt["channels"]
        self.window_size: int = ckpt["window_size"]
        self.sampling_rate: int = ckpt["sampling_rate"]
        self.threshold: float = ckpt["threshold"]
        self.mean = np.asarray(ckpt["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["normalization"]["std"], dtype=np.float32)
        self.data_source: str = ckpt["data_source"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FallDetectorCNNLSTM":
        if path is None:
            path = Path(dp.load_config()["paths"]["artifacts_dir"])
            path = dp.resolve(path) / CHECKPOINT_NAME
        model, ckpt = load_checkpoint(path)
        return cls(model, ckpt)

    # ------------------------------------------------------------------ helpers
    def _to_channels_first(self, window: np.ndarray) -> np.ndarray:
        w = np.asarray(window, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != self.window_size:
            raise ValueError(f"window must have shape [{self.window_size}, n_channels]; got {w.shape}")
        if not np.isfinite(w).all():
            raise ValueError("window contains NaN/inf")
        if w.shape[1] == len(dp.ALL_COLUMNS):
            w = w[:, dp.channel_indices(self.channels)]
        elif w.shape[1] != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} channels {self.channels} (or 9 SisFall columns); "
                             f"got {w.shape[1]}")
        return w.T.astype(np.float32)

    def _result(self, p: float) -> dict:
        is_fall = bool(p >= self.threshold)
        return {
            "model": "cnn_lstm", "experimental": True, "alert_eligible": False,
            "label": "fall" if is_fall else "adl", "is_fall": is_fall,
            "fall_probability": float(p), "threshold": self.threshold,
            "training_data_source": self.data_source,
        }

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def predict_windows_proba(self, windows: np.ndarray) -> np.ndarray:
        """windows [N, T, C] -> P(fall) [N]."""
        x = np.stack([dp.normalize(self._to_channels_first(w), self.mean, self.std) for w in windows])
        return torch.sigmoid(self.model(torch.from_numpy(x))).numpy()

    def predict_window(self, window: np.ndarray) -> dict:
        return self._result(float(self.predict_windows_proba(np.asarray(window)[None])[0]))

    def predict_file(self, path: str | Path) -> list[dict]:
        """Tile a SisFall recording into 2 s windows and classify each (no alerts are raised)."""
        sig = dp.load_sisfall_recording(path)
        starts = dp.tile_starts(len(sig), self.window_size)
        if not starts:
            raise ValueError(f"{path}: shorter than one {self.window_size}-sample window")
        probs = self.predict_windows_proba(np.stack([sig[s:s + self.window_size] for s in starts]))
        return [dict(self._result(float(p)), window_start=int(s)) for s, p in zip(starts, probs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="a SisFall recording (.txt)")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    det = FallDetectorCNNLSTM.load(args.checkpoint)
    res = det.predict_file(args.file)
    falls = [r for r in res if r["is_fall"]]
    print(f"{args.file}: {len(res)} windows, {len(falls)} flagged as fall (EXPERIMENTAL, no alert raised)")
    for r in res:
        print(f"  start {r['window_start']:6d}  P(fall) {r['fall_probability']:.3f}  -> {r['label']}")
    if det.data_source == "synthetic":
        print("\n!! This checkpoint was trained on SYNTHETIC data - outputs are meaningless !!")


if __name__ == "__main__":
    main()
