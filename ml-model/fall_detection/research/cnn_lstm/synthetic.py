"""Synthetic SisFall-FORMAT data generator - SOFTWARE TESTING ONLY.

Writes files with the same layout as the real dataset (9 integer columns per line,
trailing ';', names like F05_SA01_R01.txt in SA##/SE## folders) so the real loader and
the whole pipeline can be exercised when the real SisFall download is not available.

The signals are invented. Any metric computed from them says NOTHING about fall-detection
quality. A marker file (SYNTHETIC_DATA.txt) is written so downstream tools label every
result as synthetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data_pipeline import ADXL_SCALE, GYRO_SCALE, MMA_SCALE, SYNTHETIC_MARKER

FS = 200


def _adl(rng: np.random.Generator, seconds: float, kind: int) -> np.ndarray:
    """kind 0 = standing/sitting, 1 = walking, 2 = vigorous (jumping-like) ADL."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    acc = np.zeros((n, 3))
    acc[:, 2] = 1.0                                  # gravity on z
    gyro = rng.normal(0, 3, (n, 3))
    if kind >= 1:
        f = rng.uniform(1.6, 2.2)
        acc[:, 2] += 0.35 * np.sin(2 * np.pi * f * t)
        acc[:, 0] += 0.25 * np.sin(2 * np.pi * f * t + 1.0)
        gyro += 40 * np.sin(2 * np.pi * f * t)[:, None] * rng.uniform(0.5, 1, 3)
    if kind == 2:                                    # sharp but non-fall peaks
        for c in rng.integers(100, max(101, n - 100), size=max(1, n // 600)):
            acc[c:c + 6, 2] += rng.uniform(1.5, 2.8)
    acc += rng.normal(0, 0.03, acc.shape)
    return np.hstack([acc, gyro])


def _fall(rng: np.random.Generator, seconds: float = 15.0) -> np.ndarray:
    base = _adl(rng, seconds, 1)
    n = len(base)
    impact = int(rng.integers(4 * FS, n - 4 * FS))
    base[impact - 40:impact - 10, :3] *= 0.3                       # free-fall dip
    base[impact:impact + 8, :3] += rng.uniform(3.0, 6.0, 3) * np.array([0.6, 0.5, 1.0])
    base[impact:impact + 40, 3:] += rng.normal(0, 250, (40, 3))    # rotation burst
    # lying posture afterwards: gravity moves from z to x
    base[impact + 40:, 0] += 0.9
    base[impact + 40:, 2] -= 0.9
    return base


def _to_text(sig6: np.ndarray, rng: np.random.Generator) -> str:
    n = len(sig6)
    mma = sig6[:, :3] + rng.normal(0, 0.02, (n, 3))
    raw = np.hstack([
        np.rint(sig6[:, :3] / ADXL_SCALE),
        np.rint(sig6[:, 3:6] / GYRO_SCALE),
        np.rint(mma / MMA_SCALE),
    ]).astype(int)
    lines = [",".join(f"{v:6d}" for v in row) + ";" for row in raw]
    return "\n".join(lines) + "\n"


def write_synthetic_sisfall(root: str | Path, n_subjects: int = 12, adl_per_subject: int = 4,
                            falls_per_subject: int = 3, adl_seconds: float = 12.0,
                            seed: int = 0) -> Path:
    """Create `root/SA01..` (and SE.. for the last ~third) with synthetic recordings."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / SYNTHETIC_MARKER).write_text(
        "Synthetic SisFall-format data generated for software tests. NOT real data.\n")
    rng = np.random.default_rng(seed)
    n_se = n_subjects // 3
    names = [f"SA{i:02d}" for i in range(1, n_subjects - n_se + 1)] + \
            [f"SE{i:02d}" for i in range(1, n_se + 1)]
    for subj in names:
        d = root / subj
        d.mkdir(exist_ok=True)
        for a in range(adl_per_subject):
            kind = a % 3
            (d / f"D{a + 1:02d}_{subj}_R01.txt").write_text(_to_text(_adl(rng, adl_seconds, kind), rng))
        for f in range(falls_per_subject):
            (d / f"F{f + 1:02d}_{subj}_R01.txt").write_text(_to_text(_fall(rng), rng))
    return root
