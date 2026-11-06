"""Sanity checks for the CNN-LSTM's (suspiciously) strong SisFall result.

    python $C/sanity_checks.py                       # fast: per-activity, non-impact windows, single-number + untrained references
    python $C/sanity_checks.py --seeds 3             # + retrain with 3 seeds, report mean/std
    python $C/sanity_checks.py --shuffle             # + train on SHUFFLED labels (must score ~chance)
    python $C/sanity_checks.py --ablate              # + accelerometer-only and gyroscope-only models
    python $C/sanity_checks.py --all                 # everything (several trainings)

Reads the main checkpoint and the prepared windows; writes ONLY to <artifacts>/sanity/
(retrained models go to <artifacts>/sanity/<run>/, never over the main checkpoint).
The Random Forest .joblib is read-only.

What each check can tell you
  per-activity   : which ADL / fall types produce the errors (e.g. one activity driving every miss).
  non-impact     : what the models do on the pre-/post-impact windows of FALL recordings. SisFall does
                   not label them, so there are no accuracy numbers - only how often each is flagged.
                   Many post-impact flags = repeated alarms for one fall in continuous use.
  seeds          : stability of the result across training seeds (same split).
  shuffle        : model trained on random train+val labels. Scoring well anyway means EITHER a leak OR that the
                   classes differ so much in raw magnitude that even a random network separates them - which is why
                   the untrained-network and single-number-rule rows are reported beside it to tell the two apart.
  ablate         : how much each sensor contributes. Near-identical scores from accelerometer-only
                   and full models point to the impact signature carrying the task.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_pipeline as dp  # noqa: E402
from compare_models import rf_features  # noqa: E402
from evaluate import evaluate_checkpoint, score_windows  # noqa: E402
from model import CHECKPOINT_NAME, load_checkpoint  # noqa: E402
import train as train_mod  # noqa: E402

GROUPS = {"accel_only": ["adxl_x", "adxl_y", "adxl_z"], "gyro_only": ["gyro_x", "gyro_y", "gyro_z"]}


# --------------------------------------------------------------------------- helpers
def _scores_for(prepared: Path, ckpt_path: Path, rf_path: Path | None, split: str = "test"):
    meta = dp.load_prepared_meta(prepared)
    X, info = dp.load_split(prepared, split)
    model, ck = load_checkpoint(ckpt_path)
    ch_idx = [meta["channels"].index(c) for c in ck["channels"]]
    ch_idx = None if ch_idx == list(range(len(meta["channels"]))) else ch_idx
    out = {"CNN-LSTM": (score_windows(model, X, ck["normalization"]["mean"], ck["normalization"]["std"], ch_idx=ch_idx),
                        ck["threshold"])}
    if rf_path is not None and Path(rf_path).exists():
        pkg = joblib.load(rf_path)
        rf = pkg["model"]
        rf.n_jobs = 1
        p = rf.predict_proba(rf_features(np.asarray(X), meta["channels"], pkg["feature_columns"]))[:, 1]
        out["Random Forest"] = (p, 0.5)
    return meta, info, out


# --------------------------------------------------------------------------- 1. per activity
def per_activity(info: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """Error count/rate for every activity code (D01.. ADL: false alarms; F01.. falls: misses)."""
    y = info["label"].to_numpy()
    rows = []
    for act, g in info.groupby("activity"):
        idx = g.index.to_numpy()
        row = dict(activity=act, kind="fall" if act.startswith("F") else "ADL", n_windows=len(idx),
                   n_subjects=g["subject"].nunique())
        for name, (s, thr) in scores.items():
            pred = (s[idx] >= thr).astype(int)
            errors = int((pred != y[idx]).sum())
            row[f"{name} errors"] = errors
            row[f"{name} error rate"] = errors / len(idx)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. non-impact windows
def non_impact_windows(dataset_root: Path, subjects: list[str], window: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Complete tiles of the held-out subjects' FALL recordings that are NOT the impact tile."""
    wins, rows = [], []
    recs, _ = dp.discover_recordings(Path(dataset_root))
    for r in recs:
        if not r.is_fall or r.subject not in subjects:
            continue
        sig = dp.load_sisfall_recording(r.path)
        impact_tile = int(np.argmax(dp.accel_magnitude(sig))) // window
        for k, s in enumerate(dp.tile_starts(len(sig), window)):
            if k == impact_tile:
                continue
            wins.append(sig[s:s + window])
            rows.append(dict(subject=r.subject, recording=r.name, activity=r.activity, start=s,
                             position="before_impact" if k < impact_tile else "after_impact"))
    return (np.stack(wins) if wins else np.empty((0, window, 9))), pd.DataFrame(rows)


def non_impact_report(prepared: Path, ckpt_path: Path, rf_path: Path | None, dataset_root: Path) -> pd.DataFrame:
    meta = dp.load_prepared_meta(prepared)
    wins, info = non_impact_windows(dataset_root, meta["splits"]["test"]["subjects"], meta["window_size"])
    if len(info) == 0:
        return pd.DataFrame()
    ch = dp.channel_indices(meta["channels"])
    X = np.ascontiguousarray(wins[:, :, ch].transpose(0, 2, 1), dtype=np.float32)
    model, ck = load_checkpoint(ckpt_path)
    ci = [meta["channels"].index(c) for c in ck["channels"]]
    scores = {"CNN-LSTM": (score_windows(model, X, ck["normalization"]["mean"], ck["normalization"]["std"], ch_idx=ci),
                           ck["threshold"])}
    if rf_path is not None and Path(rf_path).exists():
        pkg = joblib.load(rf_path)
        pkg["model"].n_jobs = 1
        scores["Random Forest"] = (pkg["model"].predict_proba(rf_features(X, meta["channels"], pkg["feature_columns"]))[:, 1], 0.5)
    out = []
    for pos, g in info.groupby("position"):
        row = dict(position=pos, n_windows=len(g))
        for name, (s, thr) in scores.items():
            row[f"{name} flagged as fall"] = int((s[g.index.to_numpy()] >= thr).sum())
            row[f"{name} flagged %"] = 100.0 * row[f"{name} flagged as fall"] / len(g)
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- 3-5. retraining checks
def _train_eval(cfg: dict, prepared: Path, out_dir: Path, **kw) -> dict:
    res = train_mod.train(cfg, prepared, out_dir, verbose=False, **kw)
    ev = evaluate_checkpoint(out_dir / CHECKPOINT_NAME, prepared, "test", out_dir / "evaluation")
    return dict(best_epoch=res["best_epoch"], threshold=ev["threshold"],
                **{k: ev[k] for k in ("recall", "specificity", "precision", "f1", "roc_auc", "pr_auc", "fp", "fn",
                                      "n_fall", "n_adl", "false_alarms_per_hour")})


def seed_repeats(cfg, prepared, out, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = s
        print(f"  seed {s} ...", flush=True)
        rows.append(dict(run=f"seed_{s}", **_train_eval(c, prepared, out / f"seed_{s}")))
    return pd.DataFrame(rows)


def shuffled_labels(cfg, prepared, out, n_runs: int = 3) -> pd.DataFrame:
    """Train AND validate on randomly permuted labels; evaluate on the REAL test labels."""
    rows = []
    for k in range(n_runs):
        print(f"  shuffled labels (run {k}) ...", flush=True)
        c = copy.deepcopy(cfg)
        c["seed"] = cfg["seed"] + 1000 + k
        rows.append(dict(run=f"shuffled_labels_{k}", **_train_eval(c, prepared, out / f"shuffled_labels_{k}", shuffle_labels=True)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 6. single-feature baselines
def single_feature_auc(prepared: Path) -> pd.DataFrame:
    """ROC-AUC of each simple window statistic used by the RF, on the test windows (no training).
    Shows how separable the fall/ADL windows are before any model is involved."""
    from sklearn.metrics import roc_auc_score
    meta = dp.load_prepared_meta(prepared)
    X, info = dp.load_split(prepared, "test")
    cols = ["acc_mean", "acc_std", "acc_min", "acc_max", "acc_range", "gyro_mean", "gyro_std", "gyro_max"]
    f = rf_features(np.asarray(X), meta["channels"], cols)
    y = info["label"].to_numpy()
    rows = [dict(feature=c, roc_auc=float(roc_auc_score(y, f[c])),
                 separability=float(max(roc_auc_score(y, f[c]), 1 - roc_auc_score(y, f[c])))) for c in cols]
    return pd.DataFrame(rows).sort_values("separability", ascending=False).reset_index(drop=True)


def ablations(cfg, prepared, out) -> pd.DataFrame:
    meta = dp.load_prepared_meta(prepared)
    rows = []
    for name, chans in GROUPS.items():
        if not all(c in meta["channels"] for c in chans):
            continue
        print(f"  {name} ...", flush=True)
        rows.append(dict(run=name, **_train_eval(cfg, prepared, out / name, channels=chans)))
    return pd.DataFrame(rows)


def _channel_groups(meta: dict):
    ia = [meta["channels"].index(c) for c in ("adxl_x", "adxl_y", "adxl_z")]
    ig = [meta["channels"].index(c) for c in ("gyro_x", "gyro_y", "gyro_z")]
    return ia, ig


def magnitude_baselines(prepared: Path) -> pd.DataFrame:
    """Training-free reference: how well does ONE number (peak accelerometer / gyroscope magnitude in the
    window) separate fall windows from ADL windows? Threshold picked on validation subjects."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from evaluate import best_f1_threshold, compute_metrics
    meta = dp.load_prepared_meta(prepared)
    ia, ig = _channel_groups(meta)

    def peaks(X):
        x = np.asarray(X, dtype=np.float64)
        return {"peak accelerometer magnitude": np.sqrt((x[:, ia, :] ** 2).sum(1)).max(1),
                "peak gyroscope magnitude": np.sqrt((x[:, ig, :] ** 2).sum(1)).max(1)}
    Xv, iv = dp.load_split(prepared, "val")
    Xt, it = dp.load_split(prepared, "test")
    pv, pt = peaks(Xv), peaks(Xt)
    yv, yt = iv["label"].to_numpy(), it["label"].to_numpy()
    rows = []
    for name in pt:
        thr = best_f1_threshold(yv, pv[name])
        m = compute_metrics(yt, pt[name], thr)
        rows.append(dict(run=f"single feature: {name}", roc_auc=roc_auc_score(yt, pt[name]),
                         pr_auc=average_precision_score(yt, pt[name]), recall=m["recall"],
                         specificity=m["specificity"], f1=m["f1"]))
    return pd.DataFrame(rows)


def untrained_control(cfg: dict, prepared: Path, n_inits: int = 5) -> pd.DataFrame:
    """Random-weight (never trained) CNN-LSTMs scored on the real test labels. If these already separate
    the classes (|AUC-0.5| large) the raw signal differs so much that a shuffled-label model is not a clean null."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from model import build_model
    meta = dp.load_prepared_meta(prepared)
    mean = np.asarray(meta["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(meta["normalization"]["std"], dtype=np.float32)
    Xt, it = dp.load_split(prepared, "test")
    yt = it["label"].to_numpy()
    rows = []
    for k in range(n_inits):
        train_mod.set_seed(1000 + k)
        m = build_model(cfg["model"], in_channels=len(meta["channels"])).eval()
        sc = score_windows(m, Xt, mean, std)
        auc = roc_auc_score(yt, sc)
        rows.append(dict(run=f"untrained_init_{k}", roc_auc=auc, abs_auc_minus_half=abs(auc - 0.5),
                         pr_auc=average_precision_score(yt, sc)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, nd: int = 4) -> str:
    if df is None or df.empty:
        return "_not run_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.{nd}f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.to_numpy()) + "\n"


def build_report(meta: dict, sections: dict, prevalence: float) -> str:
    L = ["# CNN-LSTM sanity checks", ""]
    if meta["data_source"] == "synthetic":
        L += ["> **SYNTHETIC SOFTWARE TEST - NOT A SisFall RESULT.**", ""]
    L += [f"Data source: **{meta['data_source']}** | test subjects: {', '.join(meta['splits']['test']['subjects'])} | "
          f"fall prevalence in test windows: {prevalence:.3f}", ""]
    if "per_activity" in sections:
        L += ["## 1. Errors per activity", "Where do the (few) errors come from? Falls: misses; ADL: false alarms.", "",
              _md(sections["per_activity"])]
    if "non_impact" in sections:
        L += ["## 2. Pre-/post-impact windows of fall recordings (unlabelled)", "",
              "Not accuracy: SisFall has no labels for these windows. High 'after_impact' flag rates mean one fall "
              "would keep re-triggering in continuous use.", "", _md(sections["non_impact"], 2)]
    if "seeds" in sections and not sections["seeds"].empty:
        s = sections["seeds"]
        L += ["## 3. Training-seed stability", "", _md(s), "",
              "Mean +- std: " + ", ".join(f"{m} {s[m].mean():.4f} +- {s[m].std(ddof=0):.4f}"
                                          for m in ("recall", "specificity", "f1", "pr_auc")), "",
              "Interpretation guide: a std above ~0.02 on F1 means the single-run result should not be trusted.", ""]
    if any(k in sections for k in ("shuffle", "untrained", "baselines")):
        L += ["## 4. Label-shuffle control and reference points", ""]
    if "baselines" in sections:
        L += ["**Training-free single-number rules** (peak magnitude in the window):", "", _md(sections["baselines"]),
              "If one number already gives a very high ROC-AUC, the fall-vs-ADL task is easy by construction, "
              "which makes near-perfect scores less surprising (and less informative).", ""]
    if "untrained" in sections:
        u = sections["untrained"]
        L += ["**Untrained (random-weight) networks on the real test labels:**", "", _md(u), ""]
    if "shuffle" in sections:
        sh = sections["shuffle"]
        n1, n0 = sh.iloc[0]["n_fall"], sh.iloc[0]["n_adl"]
        sigma = float(np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)))      # std of ROC-AUC under "no signal"
        z = (sh["roc_auc"] - 0.5).abs() / sigma
        ok = bool((z < 3).all())
        verdict = ("PASS - random-label models are indistinguishable from chance, no sign of a label leak" if ok else
                   "FLAG - a model that never saw true labels still ranks the test windows better (or worse) than chance. "
                   "That is either a leak or classes that are trivially separable by signal strength. "
                   "Read the reference points above and section 6: if single simple features and random-weight networks "
                   "already separate the classes almost perfectly, "
                   "the task is easy and this is expected; if not, investigate the pipeline for a leak")
        L += ["**Shuffled-label models** (3 runs):", "",
              "Train AND validation labels are randomly permuted, so no true-label information reaches training, "
              "early stopping or threshold choice. Evaluated on the real test labels.", "", _md(sh),
              f"Expected ROC-AUC about 0.5 (std about {sigma:.3f} for {n1} fall / {n0} ADL windows; approximate, since "
              f"windows from one subject are not independent). |AUC-0.5| in std units: "
              f"{', '.join(f'{v:.1f}' for v in z)}. Chance PR-AUC about {prevalence:.3f}. Verdict: **{verdict}**.", ""]
    if "ablate" in sections:
        L += ["## 5. Sensor ablation", "", _md(sections["ablate"]),
              "Compare with the full 6-channel model from `evaluate.py`.", ""]
    if "single_feature" in sections:
        L += ["## 6. Single-feature baselines (no model, test windows)", "",
              "ROC-AUC of one simple statistic at a time. `separability` ignores direction (max of AUC and 1-AUC). "
              "Values near 1.0 mean the classes are almost separable by that statistic alone - the task is easy.", "",
              _md(sections["single_feature"]), ""]
    L += ["## Not covered here", "",
          "Grouped cross-validation over all 38 subjects (needs one `prepare_data.py` + `train.py` run per fold) and "
          "testing on the target wearable's real data. Do not promote the model before those.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--rf-model", default=None)
    ap.add_argument("--seeds", type=int, default=0, help="retrain with N seeds (0..N-1)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-non-impact", action="store_true", help="skip check 2 (needs the raw dataset)")
    a = ap.parse_args()
    cfg = dp.load_config(a.config)
    root = dp.find_project_root()
    prepared = dp.resolve(a.prepared_dir or cfg["paths"]["prepared_dir"], root)
    art = dp.resolve(a.artifacts_dir or cfg["paths"]["artifacts_dir"], root)
    rf = dp.resolve(a.rf_model or cfg["paths"]["rf_model"], root)
    ds = dp.resolve(a.dataset_root or cfg["paths"]["dataset_root"], root)
    out = art / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = art / CHECKPOINT_NAME

    meta, info, scores = _scores_for(prepared, ckpt, rf)
    prevalence = float((info["label"] == 1).mean())
    sections: dict = {"per_activity": per_activity(info, scores), "single_feature": single_feature_auc(prepared)}
    print("Per-activity errors and single-feature baselines done.")
    if not a.skip_non_impact:
        try:
            sections["non_impact"] = non_impact_report(prepared, ckpt, rf, ds)
            print("Non-impact windows done.")
        except FileNotFoundError as exc:
            print(f"Skipping non-impact check: {exc}")
    n_seeds = 3 if a.all and not a.seeds else a.seeds
    if n_seeds:
        print(f"Retraining with {n_seeds} seeds (this takes a while)...")
        sections["seeds"] = seed_repeats(cfg, prepared, out, list(range(n_seeds)))
    sections["baselines"] = magnitude_baselines(prepared)
    sections["untrained"] = untrained_control(cfg, prepared)
    if a.shuffle or a.all:
        sections["shuffle"] = shuffled_labels(cfg, prepared, out)
    if a.ablate or a.all:
        sections["ablate"] = ablations(cfg, prepared, out)

    report = build_report(meta, sections, prevalence)
    (out / "sanity_report.md").write_text(report, encoding="utf-8")
    with open(out / "sanity_results.json", "w", encoding="utf-8") as fh:
        json.dump({k: v.to_dict(orient="records") for k, v in sections.items()}, fh, indent=2)
    print("\n" + report)
    print(f"Saved: {out / 'sanity_report.md'}")


if __name__ == "__main__":
    main()