import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import data_pipeline as dp
from compare_models import rf_features, run_comparison
from model import CHECKPOINT_NAME

COLS = ["acc_mean", "acc_std", "acc_min", "acc_max", "acc_range", "gyro_mean", "gyro_std", "gyro_max"]


def test_comparison_end_to_end_on_synthetic(cfg, prepared, trained, tmp_path):
    out, _ = prepared
    art, _ = trained
    # THROW-AWAY RF trained on synthetic train windows (the project's real RF artifact is not used here)
    X, info = dp.load_split(out, "train")
    rf = RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=0).fit(
        rf_features(np.asarray(X), dp.ALL_COLUMNS[:6], COLS), info.label)
    rf_path = tmp_path / "throwaway_rf.joblib"
    joblib.dump(dict(model=rf, feature_columns=COLS, sampling_rate=200, window_size=400, window_seconds=2), rf_path)
    res = run_comparison(cfg, out, art / CHECKPOINT_NAME, rf_path, tmp_path / "cmp", verbose=False)
    for f in ("comparison_table.csv", "comparison.json", "evaluation_report.md", "confusion_matrices.png",
              "roc_pr_curves.png", "metrics_comparison.png"):
        assert (tmp_path / "cmp" / f).exists()
    report = (tmp_path / "cmp" / "evaluation_report.md").read_text()
    assert "SYNTHETIC SOFTWARE TEST" in report and "No conclusion can be drawn" in report
    assert res["data_source"] == "synthetic" and len(res["rows"]) == 4
    assert rf_path.exists()
