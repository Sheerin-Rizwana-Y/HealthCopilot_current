"""Random Forest vs CNN-LSTM on the SAME held-out SisFall windows.

    python ml-model/fall_detection/research/cnn_lstm/compare_models.py

Protocol
  * Both models score the identical test windows (prepared by prepare_data.py): the RF's own
    window definition (non-overlapping 400-sample tiles; one impact tile per fall file) on the
    RF's own 6 held-out test subjects (26/6/6 participant-independent split, seed 42).
  * The Random Forest is loaded READ-ONLY from its existing .joblib and used as-is. Its eight
    statistical features are recomputed from the raw windows with the same formulas as
    preprocess_sisfall.py (acc_mag / gyro_mag mean, std[ddof=1], min, max, range, max).
  * Operating points: each model at the default 0.5 cut-off, and at the F1-optimal cut-off chosen
    on the VALIDATION subjects. ROC-AUC / PR-AUC are threshold-free.
  * Nothing here modifies the RF or any backend file. The comparison never swaps models.
Outputs go to <artifacts_dir>/comparison/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402
from evaluate import (C_BLUE, C_GRID, C_ORANGE, benchmark_latency, best_f1_threshold,  # noqa: E402
                      compute_metrics, plot_confusion, plot_roc_pr, score_windows)
from model import CHECKPOINT_NAME, count_parameters, load_checkpoint  # noqa: E402

RF_NAME, CNN_NAME = "Random Forest", "CNN-LSTM"


# --------------------------------------------------------------------------- RF features
def rf_features(windows: np.ndarray, channels: list[str], feature_columns: list[str]) -> pd.DataFrame:
    """Recompute the RF's 8 features from raw windows [N, C, T] (same formulas as preprocess_sisfall.py)."""
    w = np.asarray(windows, dtype=np.float64)
    ia = [channels.index(c) for c in ("adxl_x", "adxl_y", "adxl_z")]
    ig = [channels.index(c) for c in ("gyro_x", "gyro_y", "gyro_z")]
    acc = np.sqrt((w[:, ia, :] ** 2).sum(axis=1))      # [N, T]
    gyr = np.sqrt((w[:, ig, :] ** 2).sum(axis=1))
    feats = {
        "acc_mean": acc.mean(1), "acc_std": acc.std(1, ddof=1), "acc_min": acc.min(1), "acc_max": acc.max(1),
        "acc_range": acc.max(1) - acc.min(1),
        "gyro_mean": gyr.mean(1), "gyro_std": gyr.std(1, ddof=1), "gyro_max": gyr.max(1),
    }
    return pd.DataFrame({c: feats[c] for c in feature_columns})


def rf_total_nodes(rf) -> int:
    return int(sum(e.tree_.node_count for e in rf.estimators_))


# --------------------------------------------------------------------------- comparison
def run_comparison(cfg: dict, prepared_dir: Path, cnn_ckpt: Path, rf_path: Path, out_dir: Path,
                   verbose: bool = True) -> dict:
    meta = dp.load_prepared_meta(prepared_dir)
    synthetic = meta["data_source"] == "synthetic"
    wsec = meta["window_size"] / meta["sampling_rate"]
    ev = cfg["evaluation"]

    pkg = joblib.load(rf_path)                               # read-only use of the existing artifact
    rf, feat_cols = pkg["model"], pkg["feature_columns"]
    if pkg.get("window_size") != meta["window_size"] or pkg.get("sampling_rate") != meta["sampling_rate"]:
        raise ValueError("RF window/sampling configuration differs from the prepared windows")
    cnn, ckpt = load_checkpoint(cnn_ckpt)
    mean, std = ckpt["normalization"]["mean"], ckpt["normalization"]["std"]
    if ckpt["channels"] != meta["channels"]:
        raise ValueError("CNN-LSTM checkpoint channels differ from prepared windows")

    Xv, mv = dp.load_split(prepared_dir, "val")
    Xt, mt = dp.load_split(prepared_dir, "test")
    yv, yt = mv["label"].to_numpy(), mt["label"].to_numpy()
    assert (mt["kind"] == "eval").all() and (mv["kind"] == "eval").all(), "val/test must hold RF-identical windows only"
    assert set(meta["splits"]["test"]["subjects"]).isdisjoint(ckpt["extra"]["train_subjects"]), "test/train overlap"

    Xv_a, Xt_a = np.asarray(Xv), np.asarray(Xt)
    rf_pv = rf.predict_proba(rf_features(Xv_a, meta["channels"], feat_cols))[:, 1]
    rf_pt = rf.predict_proba(rf_features(Xt_a, meta["channels"], feat_cols))[:, 1]
    cn_pv = score_windows(cnn, Xv_a, mean, std)
    cn_pt = score_windows(cnn, Xt_a, mean, std)

    thr = {RF_NAME: best_f1_threshold(yv, rf_pv), CNN_NAME: best_f1_threshold(yv, cn_pv)}
    rows = {
        f"{RF_NAME} (0.5, as deployed)": compute_metrics(yt, rf_pt, 0.5, wsec),
        f"{RF_NAME} (val-tuned thr)": compute_metrics(yt, rf_pt, thr[RF_NAME], wsec),
        f"{CNN_NAME} (0.5)": compute_metrics(yt, cn_pt, 0.5, wsec),
        f"{CNN_NAME} (val-tuned thr)": compute_metrics(yt, cn_pt, thr[CNN_NAME], wsec),
    }

    # -------- cost: latency (one window, CPU, single call) and size
    rf.n_jobs = 1                                            # in-memory only; the file on disk is unchanged
    one = np.asarray(Xt_a[:1])
    lat_rf_pred = benchmark_latency(lambda: rf.predict_proba(rf_features(one, meta["channels"], feat_cols)),
                                    ev["latency_runs"], ev["latency_warmup"])
    def _cnn_call():
        with torch.no_grad():
            cnn(torch.from_numpy(dp.normalize(one, mean, std)))
    lat_cnn = benchmark_latency(_cnn_call, ev["latency_runs"], ev["latency_warmup"])
    cost = {
        RF_NAME: dict(file_bytes=Path(rf_path).stat().st_size, size="%.2f MB" % (Path(rf_path).stat().st_size / 1e6),
                      complexity=f"{rf.n_estimators} trees, {rf_total_nodes(rf):,} nodes", latency=lat_rf_pred,
                      latency_note="features + predict_proba, n_jobs=1, 1 window"),
        CNN_NAME: dict(file_bytes=Path(cnn_ckpt).stat().st_size, size="%.2f MB" % (Path(cnn_ckpt).stat().st_size / 1e6),
                       complexity=f"{count_parameters(cnn):,} parameters", latency=lat_cnn,
                       latency_note="forward pass on normalised window, 1 window, PyTorch CPU"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    banner = "SYNTHETIC DATA - software test only, NOT a SisFall result" if synthetic else ""

    # -------- tables
    cols = ["accuracy", "precision", "recall", "specificity", "f1", "false_positive_rate", "roc_auc", "pr_auc",
            "false_alarms_per_hour", "tp", "fn", "fp", "tn"]
    table = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in rows.items()}).T
    table.insert(0, "threshold", [v["threshold"] for v in rows.values()])
    table.to_csv(out_dir / "comparison_table.csv", index_label="model")

    # -------- plots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (name, v) in zip(axes, rows.items()):
        plot_confusion(ax, v["confusion_matrix"], f"{name}\nthr {v['threshold']:.3f}")
    if banner:
        fig.suptitle(banner, color="#b00020", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    plot_roc_pr({RF_NAME: (yt, rf_pt), CNN_NAME: (yt, cn_pt)}, out_dir / "roc_pr_curves.png", banner)

    metrics_plot = ["accuracy", "precision", "recall", "specificity", "f1"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.2
    styles = [(C_BLUE, 0.45), (C_BLUE, 1.0), (C_ORANGE, 0.45), (C_ORANGE, 1.0)]
    for i, ((name, v), (col, alpha)) in enumerate(zip(rows.items(), styles)):
        vals = [v[m] or 0 for m in metrics_plot]
        ax.bar(np.arange(len(metrics_plot)) + (i - 1.5) * w, vals, w * 0.92, color=col, alpha=alpha, label=name)
    ax.set_xticks(np.arange(len(metrics_plot)), [m.capitalize() for m in metrics_plot])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (held-out test subjects)")
    ax.grid(axis="y", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    if banner:
        ax.set_title(banner, color="#b00020", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_comparison.png", dpi=140)
    plt.close(fig)

    result = dict(data_source=meta["data_source"], rf_identical_split=meta["rf_identical_split"],
                  test_subjects=meta["splits"]["test"]["subjects"], n_test=len(yt),
                  n_test_fall=int((yt == 1).sum()), n_test_adl=int((yt == 0).sum()),
                  rows=rows, cost=cost, thresholds=thr)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    (out_dir / "evaluation_report.md").write_text(build_report(result, rows, cost, meta, ckpt, rf_path), encoding="utf-8")
    if verbose:
        print(table.drop(columns=["tp", "fn", "fp", "tn"]).round(4).to_string())
    return result


# --------------------------------------------------------------------------- report
def _f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def build_report(res: dict, rows: dict, cost: dict, meta: dict, ckpt: dict, rf_path: Path) -> str:
    synthetic = res["data_source"] == "synthetic"
    L: list[str] = ["# Random Forest vs CNN-LSTM - evaluation report", ""]
    if synthetic:
        L += ["> **SYNTHETIC SOFTWARE TEST - NOT A SisFall RESULT.** The data were generated by `synthetic.py` "
              "only to exercise the code. The numbers below say nothing about real fall-detection quality "
              "and must not be quoted.", ""]
    else:
        L += ["> Real SisFall recordings, participant-independent held-out test subjects.", ""]
    if not res["rf_identical_split"] and not synthetic:
        L += ["> **WARNING:** the subject split is not the RF's own 26/6/6 split, so the saved Random Forest "
              "may have been trained on some of these test subjects. The RF numbers below are then optimistic "
              "and the comparison is NOT valid.", ""]
    L += ["## Protocol", "",
          f"- Data source: **{res['data_source']}**; test subjects ({len(res['test_subjects'])}): "
          f"{', '.join(res['test_subjects'])}",
          f"- Test windows: {res['n_test']} ({res['n_test_fall']} fall, {res['n_test_adl']} ADL), "
          f"{meta['window_size']} samples @ {meta['sampling_rate']} Hz; identical windows for both models",
          f"- Split identical to the RF's 26/6/6 split (seed 42): **{res['rf_identical_split']}**",
          f"- Fall windows: the tile containing the peak acceleration (SisFall has no sample-level fall labels) "
          "- an *estimated* event window, so labels carry some noise for both models",
          f"- CNN-LSTM trained on `{ckpt['data_source']}` data, best epoch {ckpt['extra'].get('best_epoch')}, "
          f"channels {ckpt['channels']}",
          f"- RF artifact `{Path(rf_path).name}` used read-only and unchanged",
          f"- Operating points: default 0.5 and F1-optimal threshold chosen on validation subjects "
          f"(RF {res['thresholds'][RF_NAME]:.3f}, CNN-LSTM {res['thresholds'][CNN_NAME]:.3f})", ""]
    L += ["## Results (held-out test subjects)", "",
          "| Model / operating point | Acc | Prec | Recall | Spec | F1 | FPR | ROC-AUC | PR-AUC | False alarms/h | TP | FN | FP | TN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in rows.items():
        L.append(f"| {name} | {_f(v['accuracy'])} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['specificity'])} | "
                 f"{_f(v['f1'])} | {_f(v['false_positive_rate'])} | {_f(v['roc_auc'])} | {_f(v['pr_auc'])} | "
                 f"{_f(v['false_alarms_per_hour'], 1)} | {v['tp']} | {v['fn']} | {v['fp']} | {v['tn']} |")
    L += ["", "False alarms/hour = false-positive ADL windows per hour of laboratory ADL recording "
          "(contiguous 2 s tiles). Lab trials are not real-life wear time, so treat it as a relative number only.", "",
          "![confusion matrices](confusion_matrices.png)", "![ROC and PR](roc_pr_curves.png)",
          "![metrics](metrics_comparison.png)", ""]
    L += ["## Cost", "", "| Model | Size on disk | Complexity | Mean latency / window | p95 | Note |", "|---|---|---|---|---|---|"]
    for name, c in cost.items():
        L.append(f"| {name} | {c['size']} | {c['complexity']} | {c['latency']['mean_ms']:.3f} ms | "
                 f"{c['latency']['p95_ms']:.3f} ms | {c['latency_note']} |")
    L += ["", "## Which model performs better?", ""]
    if synthetic:
        L += ["No conclusion can be drawn: the data are synthetic (and the RF used for this software test was "
              "a throw-away model, not the project's artifact). Re-run `prepare_data.py`, `train.py` and "
              "`compare_models.py` on the real SisFall dataset to obtain a real comparison.", ""]
    else:
        rf, cn = rows[f"{RF_NAME} (0.5, as deployed)"], rows[f"{CNN_NAME} (val-tuned thr)"]
        def better(key, hi=True):
            a, b = rf[key], cn[key]
            if a is None or b is None or abs(a - b) < 1e-9:
                return "tie"
            return RF_NAME if (a > b) == hi else CNN_NAME
        L += [f"Comparing the RF as deployed with the CNN-LSTM at its validation-selected threshold, on "
              f"{res['n_test_fall']} fall and {res['n_test_adl']} ADL test windows:", "",
              f"- F1: {_f(rf['f1'])} (RF) vs {_f(cn['f1'])} (CNN-LSTM) -> {better('f1')}",
              f"- Recall: {_f(rf['recall'])} vs {_f(cn['recall'])} -> {better('recall')}",
              f"- Specificity: {_f(rf['specificity'])} vs {_f(cn['specificity'])} -> {better('specificity')}",
              f"- PR-AUC: {_f(rf['pr_auc'])} vs {_f(cn['pr_auc'])} -> {better('pr_auc')}",
              f"- False alarms/hour: {_f(rf['false_alarms_per_hour'], 1)} vs {_f(cn['false_alarms_per_hour'], 1)} -> "
              f"{better('false_alarms_per_hour', hi=False)}", "",
              "Caveats: only 6 test subjects (one subject can change recall noticeably), fall labels are estimated "
              "impact windows, there are no confidence intervals, and a single training seed was used. Differences "
              "of a few points are not evidence of a real difference. Any explanation for the gap (e.g. the CNN-LSTM "
              "seeing the raw waveform shape, versus the RF seeing only 8 window-level statistics; or the CNN-LSTM "
              "overfitting a small number of fall windows) is a hypothesis to test, not a finding. "
              "Repeat with several seeds and grouped cross-validation before any decision to promote a model.", ""]
    L += ["## Integration status", "",
          "The CNN-LSTM is a research model. It is not connected to the backend, emergency rules or caregiver "
          "notifications, and its predictions carry `alert_eligible: false`.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--rf-model", default=None, help="Random Forest .joblib (read-only)")
    args = ap.parse_args()
    cfg = dp.load_config(args.config)
    root = dp.find_project_root()
    art = dp.resolve(args.artifacts_dir or cfg["paths"]["artifacts_dir"], root)
    res = run_comparison(cfg, dp.resolve(args.prepared_dir or cfg["paths"]["prepared_dir"], root),
                         art / CHECKPOINT_NAME, dp.resolve(args.rf_model or cfg["paths"]["rf_model"], root),
                         art / "comparison")
    print(f"\nReport: {art / 'comparison' / 'evaluation_report.md'}")
    if res["data_source"] == "synthetic":
        print("!! SYNTHETIC DATA - software test only, not a SisFall result !!")


if __name__ == "__main__":
    main()
