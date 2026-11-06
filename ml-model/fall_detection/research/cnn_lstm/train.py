"""Train the CNN-LSTM fall detector on prepared SisFall windows.

    python ml-model/fall_detection/research/cnn_lstm/train.py

Writes ONLY into artifacts_dir (default ml-model/fall_detection/research/cnn_lstm/artifacts):
    cnn_lstm_fall_detector.pt   best checkpoint (weights + normalisation + threshold + data source)
    training_history.json/csv   per-epoch metrics
    training_curves.png         loss and accuracy curves
It never touches ml-model/fall_detection/fall_detection_rf.joblib.

Class imbalance: BCE pos_weight = n_adl/n_fall (train windows) + impact-centred fall crops (train only).
Model selection: early stopping on validation PR-AUC (or loss); the decision threshold is then
chosen on the VALIDATION set (max F1) - the test subjects are never used for any choice.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402
from evaluate import C_BLUE, C_GRID, C_ORANGE, best_f1_threshold, compute_metrics, score_windows  # noqa: E402
from model import CHECKPOINT_NAME, build_model, count_parameters, save_checkpoint  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class WindowDataset(Dataset):
    """Lazy, memory-mapped windows; z-scored per channel with TRAIN statistics."""

    def __init__(self, X, y, mean, std, noise_std: float = 0.0, ch_idx=None):
        self.X, self.y, self.mean, self.std, self.noise = X, np.asarray(y, dtype=np.float32), mean, std, noise_std
        self.ch_idx = ch_idx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)
        if self.ch_idx is not None:
            x = x[self.ch_idx]
        x = dp.normalize(x, self.mean, self.std)
        if self.noise > 0:
            x = x + np.random.normal(0, self.noise, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[i])


def _run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct += ((logits > 0).float() == yb).sum().item()
            n += len(yb)
    return tot_loss / max(n, 1), correct / max(n, 1)


def plot_history(history: list[dict], path: Path, banner: str = "") -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, key, title in ((a1, "loss", "Loss"), (a2, "acc", "Accuracy")):
        a.plot(ep, [h[f"train_{key}"] for h in history], color=C_BLUE, lw=2, label="train")
        a.plot(ep, [h[f"val_{key}"] for h in history], color=C_ORANGE, lw=2, label="validation")
        a.set(xlabel="Epoch", title=title)
        a.grid(color=C_GRID, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False)
    if banner:
        fig.suptitle(banner, color="#b00020", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(cfg: dict, prepared_dir: Path, artifacts_dir: Path, verbose: bool = True,
          channels: list[str] | None = None, shuffle_labels: bool = False) -> dict:
    """`channels` (subset of the prepared channels) and `shuffle_labels` exist for sanity_checks.py only."""
    t = cfg["training"]
    set_seed(cfg["seed"])
    if t.get("num_threads"):
        torch.set_num_threads(t["num_threads"])
    device = pick_device(t["device"])
    meta = dp.load_prepared_meta(prepared_dir)
    use = list(channels) if channels else list(meta["channels"])
    ch_idx = None if use == meta["channels"] else [meta["channels"].index(c) for c in use]
    sel = ch_idx if ch_idx is not None else slice(None)
    mean = np.asarray(meta["normalization"]["mean"], dtype=np.float32)[sel]
    std = np.asarray(meta["normalization"]["std"], dtype=np.float32)[sel]

    Xtr, mtr = dp.load_split(prepared_dir, "train")
    Xva, mva = dp.load_split(prepared_dir, "val")
    ytr, yva = mtr["label"].to_numpy(), mva["label"].to_numpy()
    if shuffle_labels:        # sanity control: train AND validation labels randomised, so neither the
        rng_s = np.random.default_rng(cfg["seed"])        # weights nor early stopping / the threshold can
        ytr, yva = rng_s.permutation(ytr), rng_s.permutation(yva)   # use any real label information
    if len(ytr) == 0 or len(yva) == 0:
        raise RuntimeError("Empty train or validation split")
    n_fall, n_adl = int((ytr == 1).sum()), int((ytr == 0).sum())
    if n_fall == 0 or n_adl == 0:
        raise RuntimeError("Training split must contain both classes")

    g = torch.Generator().manual_seed(cfg["seed"])
    tr_loader = DataLoader(WindowDataset(Xtr, ytr, mean, std, t["noise_std"], ch_idx), batch_size=t["batch_size"],
                           shuffle=True, generator=g, num_workers=t["num_workers"])
    va_loader = DataLoader(WindowDataset(Xva, yva, mean, std, 0.0, ch_idx), batch_size=t["batch_size"], shuffle=False,
                           num_workers=t["num_workers"])

    model = build_model(cfg["model"], in_channels=len(use)).to(device)
    pos_weight = torch.tensor([n_adl / n_fall if t["use_pos_weight"] else 1.0], device=device)
    loss_fn_w = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn = lambda lg, yb: loss_fn_w(lg, yb)  # noqa: E731
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    es = t["early_stopping"]
    maximise = es["monitor"] == "val_pr_auc"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max" if maximise else "min", factor=0.5,
                                                       patience=max(1, es["patience"] // 3))
    if verbose:
        print(f"Data source: {meta['data_source']} | train {len(ytr)} windows ({n_fall} fall / {n_adl} ADL), "
              f"val {len(yva)} | params {count_parameters(model):,} | pos_weight {pos_weight.item():.2f} | device {device}")

    history, best, best_state, best_epoch, bad = [], None, None, 0, 0
    for epoch in range(1, t["max_epochs"] + 1):
        tr_loss, tr_acc = _run_epoch(model, tr_loader, loss_fn, device, opt, t["grad_clip"])
        va_loss, va_acc = _run_epoch(model, va_loader, loss_fn, device)
        va_scores = score_windows(model, Xva, mean, std, device=device, ch_idx=ch_idx)
        va_m = compute_metrics(yva, va_scores)
        va_pr = va_m["pr_auc"] if va_m["pr_auc"] is not None else 0.0
        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc,
                   val_pr_auc=va_pr, val_f1_at_0p5=va_m["f1"], lr=opt.param_groups[0]["lr"])
        history.append(row)
        monitored = row[es["monitor"]]
        sched.step(monitored)
        improved = best is None or (monitored > best + es["min_delta"] if maximise else monitored < best - es["min_delta"])
        if improved:
            best, best_epoch, bad = monitored, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} "
                  f"acc {va_acc:.4f} PR-AUC {va_pr:.4f}{'  *' if improved else ''}")
        if bad >= es["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    va_scores = score_windows(model, Xva, mean, std, device=device, ch_idx=ch_idx)
    threshold = best_f1_threshold(yva, va_scores)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / CHECKPOINT_NAME
    save_checkpoint(ckpt_path, model, channels=use, window_size=meta["window_size"],
                    sampling_rate=meta["sampling_rate"], mean=mean, std=std, threshold=threshold,
                    data_source=meta["data_source"], seed=cfg["seed"],
                    extra=dict(best_epoch=best_epoch, epochs_run=len(history), monitor=es["monitor"],
                               train_subjects=meta["splits"]["train"]["subjects"],
                               val_subjects=meta["splits"]["val"]["subjects"],
                               test_subjects=meta["splits"]["test"]["subjects"],
                               rf_identical_split=meta["rf_identical_split"]))
    with open(artifacts_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(dict(history=history, best_epoch=best_epoch, threshold=threshold,
                       data_source=meta["data_source"], config=cfg), fh, indent=2)
    with open(artifacts_dir / "training_history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    banner = "SYNTHETIC DATA - software test only" if meta["data_source"] == "synthetic" else ""
    plot_history(history, artifacts_dir / "training_curves.png", banner)
    if verbose:
        print(f"Checkpoint: {ckpt_path}\nValidation-selected threshold: {threshold:.4f}")
    return dict(checkpoint=str(ckpt_path), best_epoch=best_epoch, epochs_run=len(history), threshold=threshold,
                data_source=meta["data_source"], history=history)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None, help="override paths.artifacts_dir")
    ap.add_argument("--epochs", type=int, default=None, help="override training.max_epochs")
    args = ap.parse_args()
    cfg = dp.load_config(args.config)
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    root = dp.find_project_root()
    res = train(cfg, dp.resolve(args.prepared_dir or cfg["paths"]["prepared_dir"], root),
                dp.resolve(args.artifacts_dir or cfg["paths"]["artifacts_dir"], root))
    if res["data_source"] == "synthetic":
        print("\n!! Trained on SYNTHETIC data - software test only, not a SisFall result !!")


if __name__ == "__main__":
    main()