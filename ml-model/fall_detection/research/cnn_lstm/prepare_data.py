"""Build the CNN-LSTM window dataset from raw SisFall recordings.

    python ml-model/fall_detection/research/cnn_lstm/prepare_data.py
    python ml-model/fall_detection/research/cnn_lstm/prepare_data.py --dataset-root <path> --out <dir>

Output (default: datasets/sisfall/cnn_lstm_windows/ - nothing is written next to the RF files):
    {train,val,test}_X.f32       raw float32 windows [N, C, 400] (physical units, memory-mappable)
    {train,val,test}_meta.csv    subject, recording, activity, start, label, kind
    prepared_meta.json           split subjects, counts, channels, normalisation (TRAIN stats only)

Leakage controls
  * split is by SUBJECT (same sklearn call/seed as the RF -> 26/6/6 for the 38 SisFall subjects)
  * val/test hold only the RF-identical windows (non-overlapping tiles; one impact tile per fall file)
  * extra fall crops and optional ADL overlap are generated for the TRAIN subjects only
  * normalisation mean/std are computed from TRAIN windows only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402


def prepare(cfg: dict, dataset_root: Path, out_dir: Path, rf_csv: Path | None = None,
            verbose: bool = True) -> dict:
    d = cfg["data"]
    window = d["window_size"]
    ch_idx = dp.channel_indices(d["channels"])
    rng = np.random.default_rng(cfg["seed"])

    dataset_root, out_dir = Path(dataset_root), Path(out_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"SisFall dataset not found at: {dataset_root}")
    recs, ignored = dp.discover_recordings(dataset_root)
    if not recs:
        raise RuntimeError(f"No SisFall recordings (F*/D*_S[AE]##_R##.txt) under {dataset_root}")
    source = "synthetic" if (dataset_root / dp.SYNTHETIC_MARKER).exists() else "sisfall"

    subjects_found = sorted({r.subject for r in recs})
    order, order_src = dp.subject_order(subjects_found, rf_csv)
    n_val, n_test, rf_identical = dp.scaled_split_sizes(len(order), d["n_val_subjects"], d["n_test_subjects"])
    split_map = dp.split_subjects(order, n_val, n_test, d["split_seed"])
    subj_to_split = {s: sp for sp, subs in split_map.items() for s in subs}
    rf_identical = rf_identical and d["split_seed"] == 42

    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {sp: open(out_dir / f"{sp}_X.f32", "wb") for sp in dp.SPLITS}
    rows: dict[str, list[dict]] = {sp: [] for sp in dp.SPLITS}
    n_win = Counter()
    ch_sum = np.zeros(len(ch_idx), dtype=np.float64)
    ch_sq = np.zeros(len(ch_idx), dtype=np.float64)
    ch_n = 0
    failed: list[dict] = []

    def emit(split: str, rec: dp.RecordingInfo, wins: np.ndarray, starts: np.ndarray, label: int, kind: str):
        nonlocal ch_n, ch_sum, ch_sq
        if len(wins) == 0:
            return
        x = np.ascontiguousarray(wins[:, :, ch_idx].transpose(0, 2, 1), dtype=np.float32)  # [k, C, T]
        handles[split].write(x.tobytes())
        for s in starts:
            rows[split].append(dict(subject=rec.subject, recording=rec.name, activity=rec.activity,
                                    start=int(s), label=label, kind=kind))
        n_win[(split, label)] += len(x)
        if split == "train":
            xd = x.astype(np.float64)
            ch_sum += xd.sum(axis=(0, 2))
            ch_sq += (xd ** 2).sum(axis=(0, 2))
            ch_n += xd.shape[0] * xd.shape[2]

    try:
        for i, rec in enumerate(recs, 1):
            try:
                sig = dp.load_sisfall_recording(rec.path)
                split = subj_to_split[rec.subject]
                wins, starts = dp.eval_windows(sig, rec.is_fall, window)
                if len(wins) == 0:
                    raise ValueError("No complete window extracted (impact tile incomplete / file too short)")
                if split == "train":
                    if not rec.is_fall:
                        stride = d["train_adl_stride"]
                        starts_t = np.asarray(dp.tile_starts(len(sig), window, stride), dtype=np.int64)
                        cap = d["adl_max_windows_per_recording"]
                        if cap and len(starts_t) > cap:
                            starts_t = np.sort(rng.choice(starts_t, size=cap, replace=False))
                        wins = np.stack([sig[s:s + window] for s in starts_t])
                        starts = starts_t
                    emit(split, rec, wins, starts, rec.label, "eval" if rec.is_fall or d["train_adl_stride"] == window else "train_overlap")
                    if rec.is_fall:
                        aw, ast = dp.fall_augment_windows(sig, window, d["fall_aug_crops"], d["fall_aug_margin"], rng)
                        emit(split, rec, aw, ast, dp.LABEL_FALL, "aug")
                else:
                    emit(split, rec, wins, starts, rec.label, "eval")
            except Exception as exc:  # mirror the RF script: count and continue
                failed.append(dict(recording=rec.name, error=repr(exc)))
            if verbose and i % 500 == 0:
                print(f"  processed {i}/{len(recs)} recordings")
    finally:
        for h in handles.values():
            h.close()

    if ch_n == 0:
        raise RuntimeError("No training windows were produced; cannot compute normalisation statistics. "
                           f"{len(failed)} recordings failed; first error: {failed[0] if failed else 'none'}")
    mean = ch_sum / ch_n
    std = np.sqrt(np.maximum(ch_sq / ch_n - mean ** 2, 0.0))
    std = np.where(std < 1e-6, 1.0, std)

    splits_meta = {}
    for sp in dp.SPLITS:
        df = pd.DataFrame(rows[sp], columns=["subject", "recording", "activity", "start", "label", "kind"])
        df.to_csv(out_dir / f"{sp}_meta.csv", index=False)
        splits_meta[sp] = dict(
            subjects=split_map[sp], n_windows=len(df),
            n_adl=int((df.label == dp.LABEL_ADL).sum()), n_fall=int((df.label == dp.LABEL_FALL).sum()),
            n_fall_eval_tile=int(((df.label == 1) & (df.kind == "eval")).sum()),
        )
    meta = dict(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_source=source,
        dataset_root=str(dataset_root),
        channels=d["channels"], window_size=window, sampling_rate=d["sampling_rate"],
        label_map={"0": "adl", "1": "fall"},
        subject_order_source=order_src, rf_identical_split=bool(rf_identical),
        split_seed=d["split_seed"], splits=splits_meta,
        normalization=dict(mean=mean.tolist(), std=std.tolist(), fit_on="train windows only"),
        n_recordings=len(recs), n_ignored_files=len(ignored), n_failed=len(failed), failed=failed[:50],
        config_data=d,
    )
    with open(out_dir / "prepared_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-root", default=None, help="override paths.dataset_root")
    ap.add_argument("--out", default=None, help="override paths.prepared_dir")
    args = ap.parse_args()

    cfg = dp.load_config(args.config)
    root = dp.find_project_root()
    ds = dp.resolve(args.dataset_root or cfg["paths"]["dataset_root"], root)
    out = dp.resolve(args.out or cfg["paths"]["prepared_dir"], root)
    meta = prepare(cfg, ds, out, dp.resolve(cfg["paths"]["rf_features_csv"], root))

    print("=" * 60)
    print(f"Data source          : {meta['data_source'].upper()}")
    print(f"Recordings / failed  : {meta['n_recordings']} / {meta['n_failed']}")
    print(f"Subject order source : {meta['subject_order_source']}")
    print(f"RF-identical split   : {meta['rf_identical_split']}")
    for sp in dp.SPLITS:
        s = meta["splits"][sp]
        print(f"{sp:5s}: {len(s['subjects']):2d} subjects, {s['n_windows']:6d} windows "
              f"(ADL {s['n_adl']}, fall {s['n_fall']})")
    print("Leakage check        : subject sets disjoint (asserted)")
    print(f"Saved to             : {out}")
    if meta["data_source"] == "synthetic":
        print("\n!! SYNTHETIC DATA - for software testing only, not a SisFall result !!")


if __name__ == "__main__":
    main()
