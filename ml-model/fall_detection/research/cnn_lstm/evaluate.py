"""Metrics, scoring and held-out evaluation for the CNN-LSTM research model.

    python ml-model/fall_detection/research/cnn_lstm/evaluate.py --split test

Reports accuracy, precision, recall (sensitivity), specificity, F1, false-positive rate,
ROC-AUC, PR-AUC, confusion matrix and false alarms per hour of ADL recording, and writes
confusion-matrix / ROC / PR plots. Results are labelled with the data source stored in the
checkpoint ("sisfall" vs "synthetic") so synthetic smoke-test numbers can never be
mistaken for real ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402
from model import CHECKPOINT_NAME, load_checkpoint  # noqa: E402

# colour-blind-safe pair (blue / orange); direct text labels always accompany colour
C_BLUE, C_ORANGE, C_GRID, C_INK = "#1f6fb4", "#d95f02", "#d9d9d9", "#222222"


# --------------------------------------------------------------------------- metrics
def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Decision threshold maximising F1 (choose on VALIDATION data only)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def false_alarms_per_hour(n_false_positive: int, n_adl_windows: int, window_seconds: float) -> float | None:
    """FP windows per hour of ADL recording. Valid because ADL windows are contiguous 2 s tiles
    of continuous recordings. Laboratory ADL trials are NOT a model of real-life wear time."""
    hours = n_adl_windows * window_seconds / 3600.0
    return float(n_false_positive / hours) if hours > 0 else None


def compute_metrics(y_true, y_score, threshold: float = 0.5, window_seconds: float = 2.0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: float(a / b) if b else None  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    both = len(np.unique(y_true)) == 2
    return dict(
        threshold=float(threshold), n=int(len(y_true)), n_fall=int((y_true == 1).sum()), n_adl=int((y_true == 0).sum()),
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=div(tp + tn, len(y_true)), precision=precision, recall=recall, sensitivity=recall,
        specificity=div(tn, tn + fp), false_positive_rate=div(fp, fp + tn),
        f1=(2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None),
        roc_auc=float(roc_auc_score(y_true, y_score)) if both else None,
        pr_auc=float(average_precision_score(y_true, y_score)) if both else None,
        false_alarms_per_hour=false_alarms_per_hour(fp, int((y_true == 0).sum()), window_seconds),
        confusion_matrix=[[tn, fp], [fn, tp]],   # [[TN FP],[FN TP]] - same layout as the RF scripts
    )


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_windows(model: torch.nn.Module, X, mean, std, batch_size: int = 256, device: str = "cpu",
                  ch_idx=None) -> np.ndarray:
    """P(fall) for raw windows X [N, C, T] (physical units); z-scoring uses the checkpoint stats."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = np.asarray(X[i:i + batch_size], dtype=np.float32)
        if ch_idx is not None:
            xb = xb[:, ch_idx, :]
        xb = dp.normalize(xb, mean, std)
        out.append(torch.sigmoid(model(torch.from_numpy(xb).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def benchmark_latency(fn, runs: int = 200, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return dict(mean_ms=float(np.mean(ts)), p95_ms=float(np.percentile(ts, 95)), runs=runs)


# --------------------------------------------------------------------------- plots
def plot_confusion(ax, cm, title: str) -> None:
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else C_INK)
    ax.set_xticks([0, 1], ["ADL", "Fall"])
    ax.set_yticks([0, 1], ["ADL", "Fall"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, banner: str = "") -> None:
    """curves: name -> (y_true, y_score). One y-axis per panel."""
    colors = [C_BLUE, C_ORANGE, "#555555"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for (name, (yt, ys)), col in zip(curves.items(), colors):
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        p, r, _ = precision_recall_curve(yt, ys)
        a1.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {roc_auc_score(yt, ys):.3f})")
        a2.plot(r, p, color=col, lw=2, label=f"{name} (AP {average_precision_score(yt, ys):.3f})")
    a1.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    for a in (a1, a2):
        a.grid(color=C_GRID, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=8, frameon=False, loc="lower right" if a is a1 else "lower left")
    if banner:
        fig.suptitle(banner, color="#b00020", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- evaluation CLI
def evaluate_checkpoint(ckpt_path: Path, prepared_dir: Path, split: str, out_dir: Path,
                        threshold: float | None = None) -> dict:
    model, ckpt = load_checkpoint(ckpt_path)
    meta = dp.load_prepared_meta(prepared_dir)
    missing = [c for c in ckpt["channels"] if c not in meta["channels"]]
    if missing:
        raise ValueError(f"Checkpoint channels {missing} are not in the prepared data {meta['channels']}")
    ch_idx = None if ckpt["channels"] == meta["channels"] else [meta["channels"].index(c) for c in ckpt["channels"]]
    X, info = dp.load_split(prepared_dir, split)
    if split == "train":
        info_mask = (info["kind"] == "eval").to_numpy()   # evaluate on the RF-identical windows only
    else:
        info_mask = np.ones(len(info), dtype=bool)
    idx = np.flatnonzero(info_mask)
    y = info["label"].to_numpy()[idx]
    scores = score_windows(model, X[idx] if split == "train" else X, ckpt["normalization"]["mean"],
                           ckpt["normalization"]["std"], ch_idx=ch_idx)
    thr = ckpt["threshold"] if threshold is None else threshold
    wsec = ckpt["window_size"] / ckpt["sampling_rate"]
    res = compute_metrics(y, scores, thr, wsec)
    res.update(split=split, model="cnn_lstm", checkpoint_data_source=ckpt["data_source"],
               prepared_data_source=meta["data_source"], rf_identical_split=meta["rf_identical_split"],
               n_subjects=len(meta["splits"][split]["subjects"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"evaluation_{split}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    banner = "SYNTHETIC DATA - software test only, not a SisFall result" if meta["data_source"] == "synthetic" else ""
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    plot_confusion(ax, res["confusion_matrix"], f"CNN-LSTM - {split} (thr {thr:.3f})")
    if banner:
        fig.suptitle(banner, color="#b00020", fontsize=8, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"confusion_matrix_{split}.png", dpi=140)
    plt.close(fig)
    plot_roc_pr({"CNN-LSTM": (y, scores)}, out_dir / f"roc_pr_{split}.png", banner)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=list(dp.SPLITS))
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="override the validation-selected threshold")
    args = ap.parse_args()
    cfg = dp.load_config(args.config)
    root = dp.find_project_root()
    prepared = dp.resolve(args.prepared_dir or cfg["paths"]["prepared_dir"], root)
    art = dp.resolve(args.artifacts_dir or cfg["paths"]["artifacts_dir"], root)
    res = evaluate_checkpoint(art / CHECKPOINT_NAME, prepared, args.split, art / "evaluation", args.threshold)
    print(f"CNN-LSTM on '{args.split}' ({res['prepared_data_source'].upper()} data, "
          f"{res['n_subjects']} subjects, {res['n']} windows: {res['n_fall']} fall / {res['n_adl']} ADL)")
    for k in ("accuracy", "precision", "recall", "specificity", "f1", "false_positive_rate",
              "roc_auc", "pr_auc", "false_alarms_per_hour"):
        v = res[k]
        print(f"  {k:22s}: {'n/a' if v is None else f'{v:.4f}'}")
    print("  confusion [[TN FP],[FN TP]]:", res["confusion_matrix"])
    if res["prepared_data_source"] == "synthetic":
        print("\n!! SYNTHETIC DATA - software test only, not a SisFall result !!")


if __name__ == "__main__":
    main()
