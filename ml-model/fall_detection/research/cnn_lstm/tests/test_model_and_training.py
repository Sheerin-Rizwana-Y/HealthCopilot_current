import numpy as np
import pytest
import torch

import data_pipeline as dp
from evaluate import best_f1_threshold, compute_metrics, evaluate_checkpoint, false_alarms_per_hour
from model import CHECKPOINT_NAME, CNNLSTM, count_parameters, load_checkpoint
from predict import FallDetectorCNNLSTM


@pytest.mark.parametrize("bi", [False, True])
def test_forward_shape_and_grad(bi):
    m = CNNLSTM(in_channels=6, bidirectional=bi)
    x = torch.randn(5, 6, 400)
    y = m(x)
    assert y.shape == (5,) and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    assert count_parameters(m) > 0


def test_forward_rejects_bad_input():
    with pytest.raises(ValueError):
        CNNLSTM()(torch.randn(6, 400))


def test_eval_mode_deterministic():
    m = CNNLSTM().eval()
    x = torch.randn(3, 6, 400)
    assert torch.equal(m(x), m(x))


def test_training_workflow_and_artifacts(trained, prepared):
    art, res = trained
    assert (art / CHECKPOINT_NAME).exists()
    for f in ("training_history.json", "training_history.csv", "training_curves.png"):
        assert (art / f).exists()
    assert 1 <= res["epochs_run"] <= 4 and res["data_source"] == "synthetic"
    h = res["history"][0]
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "val_pr_auc"} <= set(h)


def test_checkpoint_roundtrip(trained):
    art, _ = trained
    m1, ck = load_checkpoint(art / CHECKPOINT_NAME)
    m2, _ = load_checkpoint(art / CHECKPOINT_NAME)
    x = torch.randn(2, 6, 400)
    assert torch.equal(m1(x), m2(x))
    assert ck["data_source"] == "synthetic" and ck["channels"] == dp.ALL_COLUMNS[:6]
    assert len(ck["normalization"]["mean"]) == 6 and 0.0 <= ck["threshold"] <= 1.0
    assert ck["label_map"] == {"0": "adl", "1": "fall"}


def test_training_is_reproducible(cfg, prepared, tmp_path):
    import copy, train as T
    c = copy.deepcopy(cfg); c["training"]["max_epochs"] = 2
    a = T.train(c, prepared[0], tmp_path / "a", verbose=False)
    b = T.train(c, prepared[0], tmp_path / "b", verbose=False)
    assert [h["train_loss"] for h in a["history"]] == pytest.approx([h["train_loss"] for h in b["history"]], rel=1e-5)


def test_prediction_interface(trained, synth_root):
    art, _ = trained
    det = FallDetectorCNNLSTM.load(art / CHECKPOINT_NAME)
    sig = dp.load_sisfall_recording(next(synth_root.glob("SA01/F01_*.txt")))
    r = det.predict_window(sig[:400, :6])
    assert set(r) >= {"label", "is_fall", "fall_probability", "threshold", "alert_eligible", "experimental"}
    assert r["alert_eligible"] is False and r["experimental"] is True and 0 <= r["fall_probability"] <= 1
    assert det.predict_window(sig[:400, :]) == r                     # 9-column window accepted, same channels
    assert len(det.predict_file(next(synth_root.glob("SA01/F01_*.txt")))) == len(sig) // 400
    with pytest.raises(ValueError):
        det.predict_window(sig[:300, :6])
    with pytest.raises(ValueError):
        det.predict_window(np.full((400, 6), np.nan))


def test_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .2, .6, .1, .1, .1, .1, .1])
    m = compute_metrics(y, s, 0.5, window_seconds=2.0)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["accuracy"] == pytest.approx(0.8) and m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75) and m["specificity"] == pytest.approx(5 / 6)
    assert m["f1"] == pytest.approx(0.75) and m["false_positive_rate"] == pytest.approx(1 / 6)
    assert m["roc_auc"] == pytest.approx(23 / 24)                    # 23 of 24 fall/ADL pairs ranked correctly
    assert m["confusion_matrix"] == [[5, 1], [1, 3]]
    assert m["false_alarms_per_hour"] == pytest.approx(1 / (6 * 2 / 3600))
    assert false_alarms_per_hour(2, 1800, 2.0) == pytest.approx(2.0)
    assert best_f1_threshold(y, s) == pytest.approx(0.2)   # thr .2: TP4 FP1 -> F1 .889 beats thr .7 (.857)


def test_metrics_single_class_does_not_crash():
    m = compute_metrics(np.zeros(5), np.linspace(0, 1, 5))
    assert m["roc_auc"] is None and m["recall"] is None


def test_evaluate_checkpoint(trained, prepared, tmp_path):
    art, _ = trained
    res = evaluate_checkpoint(art / CHECKPOINT_NAME, prepared[0], "test", tmp_path)
    assert res["checkpoint_data_source"] == "synthetic" and res["n"] == res["tp"] + res["tn"] + res["fp"] + res["fn"]
    assert (tmp_path / "confusion_matrix_test.png").exists() and (tmp_path / "evaluation_test.json").exists()
