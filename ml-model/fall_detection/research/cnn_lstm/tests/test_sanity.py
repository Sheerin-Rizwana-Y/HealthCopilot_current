import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import data_pipeline as dp
import sanity_checks as sc
from compare_models import rf_features
from evaluate import evaluate_checkpoint
from model import CHECKPOINT_NAME, load_checkpoint

COLS = ["acc_mean", "acc_std", "acc_min", "acc_max", "acc_range", "gyro_mean", "gyro_std", "gyro_max"]


def _rf(prepared, tmp_path):
    X, info = dp.load_split(prepared[0], "train")
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(rf_features(np.asarray(X), dp.ALL_COLUMNS[:6], COLS), info.label)
    p = tmp_path / "rf.joblib"
    joblib.dump(dict(model=rf, feature_columns=COLS, sampling_rate=200, window_size=400, window_seconds=2), p)
    return p


def test_per_activity_and_non_impact(prepared, trained, synth_root, tmp_path):
    rf = _rf(prepared, tmp_path)
    meta, info, scores = sc._scores_for(prepared[0], trained[0] / CHECKPOINT_NAME, rf)
    t = sc.per_activity(info, scores)
    assert set(t.kind) == {"ADL", "fall"} and t.n_windows.sum() == len(info)
    assert {"CNN-LSTM errors", "Random Forest errors"} <= set(t.columns)
    ni = sc.non_impact_report(prepared[0], trained[0] / CHECKPOINT_NAME, rf, synth_root)
    assert set(ni.position) <= {"before_impact", "after_impact"} and (ni.n_windows > 0).all()


def test_non_impact_excludes_impact_tile(synth_root):
    subs = sorted({p.parent.name for p in synth_root.glob("S*/F*.txt")})[:2]
    wins, info = sc.non_impact_windows(synth_root, subs, 400)
    assert len(wins) == len(info) and info.recording.nunique() == 6
    for rec, g in info.groupby("recording"):
        sig = dp.load_sisfall_recording(next(synth_root.glob(f"*/{rec}")))
        n_tiles = len(dp.tile_starts(len(sig), 400))                       # 15 s -> 7 tiles
        impact_start = (int(np.argmax(dp.accel_magnitude(sig))) // 400) * 400
        assert len(g) == n_tiles - 1 and impact_start not in set(g.start)
        assert (g[g.start < impact_start].position == "before_impact").all()
        assert (g[g.start > impact_start].position == "after_impact").all()


def test_channel_subset_training_and_shuffle(cfg, prepared, tmp_path):
    import copy
    c = copy.deepcopy(cfg); c["training"]["max_epochs"] = 2
    r = sc._train_eval(c, prepared[0], tmp_path / "acc", channels=sc.GROUPS["accel_only"])
    _, ck = load_checkpoint(tmp_path / "acc" / CHECKPOINT_NAME)
    assert ck["channels"] == ["adxl_x", "adxl_y", "adxl_z"] and len(ck["normalization"]["mean"]) == 3
    assert 0 <= r["pr_auc"] <= 1
    s = sc.shuffled_labels(c, prepared[0], tmp_path)
    assert s.iloc[0]["run"].startswith("shuffled_labels")
    rep = sc.build_report(prepared[1], {"shuffle": s}, 0.1)
    assert "SYNTHETIC SOFTWARE TEST" in rep and "Label-shuffle control" in rep


def test_main_checkpoint_not_touched_by_sanity(trained, cfg, prepared, tmp_path):
    before = (trained[0] / CHECKPOINT_NAME).read_bytes()
    sc.seed_repeats({**cfg, "training": {**cfg["training"], "max_epochs": 1}}, prepared[0], tmp_path, [0])
    assert (trained[0] / CHECKPOINT_NAME).read_bytes() == before


def test_reference_controls(cfg, prepared):
    b = sc.magnitude_baselines(prepared[0])
    assert len(b) == 2 and ((b.roc_auc >= 0) & (b.roc_auc <= 1)).all()
    u = sc.untrained_control(cfg, prepared[0], n_inits=2)
    assert len(u) == 2 and (u.abs_auc_minus_half >= 0).all()
    rep = sc.build_report(prepared[1], {"baselines": b, "untrained": u}, 0.1)
    assert "single-number rules" in rep and "Untrained" in rep


def test_shuffle_randomises_validation_labels_too(cfg, prepared, tmp_path):
    """In shuffle mode the labels used for validation / early stopping / threshold choice must be a
    permutation of the real ones (same counts, different order) - otherwise the control leaks true labels."""
    import copy
    import train as T
    c = copy.deepcopy(cfg); c["training"]["max_epochs"] = 1
    seen = []
    orig = T.compute_metrics
    T.compute_metrics = lambda y, s, *a, **k: (seen.append(np.asarray(y).copy()), orig(y, s, *a, **k))[1]
    try:
        T.train(c, prepared[0], tmp_path / "s", verbose=False, shuffle_labels=True)
    finally:
        T.compute_metrics = orig
    real = dp.load_split(prepared[0], "val")[1]["label"].to_numpy()
    assert seen and seen[0].sum() == real.sum() and not np.array_equal(seen[0], real)


def test_shuffle_verdict_uses_untrained_reference():
    meta = {"data_source": "sisfall", "splits": {"test": {"subjects": ["S"]}}}
    sh = pd.DataFrame([dict(run="s0", roc_auc=0.62, pr_auc=0.1, n_fall=225, n_adl=3821)])
    ok = sc.build_report(meta, {"shuffle": sh, "untrained": pd.DataFrame([dict(run="u", roc_auc=0.15, abs_auc_minus_half=0.35, pr_auc=0.03)])}, 0.056)
    assert "Verdict: **PASS" in ok
    bad = sc.build_report(meta, {"shuffle": sh.assign(roc_auc=0.99), "untrained": pd.DataFrame([dict(run="u", roc_auc=0.45, abs_auc_minus_half=0.05, pr_auc=0.06)])}, 0.056)
    assert "Verdict: **FLAG" in bad