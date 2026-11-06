import ast
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_pipeline as dp
from compare_models import rf_features


def test_loader_parses_format_and_scales(tmp_path):
    f = tmp_path / "D01_SA01_R01.txt"
    f.write_text("   256,     0,     0,    16,     0,     0,  1024,     0,     0;\n"
                 "  -256,     0,     0,     0,    16,     0,     0,     0, -1024;\n")
    sig = dp.load_sisfall_recording(f)
    assert sig.shape == (2, 9)
    assert sig[0, 0] == pytest.approx(1.0)               # 256 * 32/8192 = 1 g
    assert sig[0, 3] == pytest.approx(16 * 4000 / 65536)  # deg/s
    assert sig[0, 6] == pytest.approx(1024 * 16 / 16384)  # MMA 1 g
    assert sig[1, 8] == pytest.approx(-1.0)              # trailing ';' removed, sign kept, order preserved


def test_loader_rejects_bad_files(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("1,2,3;\n")
    with pytest.raises(ValueError):
        dp.load_sisfall_recording(f)


def test_label_mapping_and_recording_parse(tmp_path):
    f = dp.parse_recording_name(Path("F05_SA03_R02.txt"))
    d = dp.parse_recording_name(Path("D12_SE01_R01.txt"))
    assert (f.label, d.label) == (dp.LABEL_FALL, dp.LABEL_ADL) == (1, 0)
    assert (f.subject, f.activity, f.trial) == ("SA03", "F05", "R02")
    assert dp.parse_recording_name(Path("Readme.txt")) is None


def test_window_generation_matches_rf_convention():
    n, win = 1250, 400
    sig = np.zeros((n, 9)); sig[:, 2] = 1.0
    w, s = dp.eval_windows(sig, False, win)
    assert w.shape == (3, 400, 9) and list(s) == [0, 400, 800]      # incomplete tail dropped
    sig[700, 0] = 5.0                                              # impact in tile 1 (400..799)
    w, s = dp.eval_windows(sig, True, win)
    assert w.shape == (1, 400, 9) and s[0] == 400
    sig[:, 0] = 0; sig[1249, 0] = 9.0                              # impact in the incomplete tail
    assert dp.eval_windows(sig, True, win)[0].shape[0] == 0


def test_fall_augmentation_contains_impact():
    rng = np.random.default_rng(0)
    sig = np.zeros((3000, 9)); sig[:, 2] = 1.0; sig[1500, 0] = 8.0
    w, s = dp.fall_augment_windows(sig, 400, 10, 20, rng)
    assert w.shape == (10, 400, 9)
    assert all(s_ <= 1500 < s_ + 400 for s_ in s)
    assert (w[:, :, 0].max(axis=1) == 8.0).all()


def test_subject_split_is_rf_identical_and_disjoint():
    from sklearn.model_selection import train_test_split
    subs = [f"SA{i:02d}" for i in range(1, 24)] + [f"SE{i:02d}" for i in range(1, 16)]
    sp = dp.split_subjects(subs, 6, 6, 42)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (26, 6, 6)
    tr, tmp = train_test_split(np.array(subs, dtype=object), test_size=12, random_state=42)   # RF script's calls
    va, te = train_test_split(tmp, test_size=6, random_state=42)
    assert sp["test"] == list(te) and sp["val"] == list(va) and sp["train"] == list(tr)
    assert dp.scaled_split_sizes(38, 6, 6) == (6, 6, True)
    assert dp.scaled_split_sizes(12, 6, 6)[2] is False


def test_prepared_dataset_shapes_labels_and_no_leakage(prepared, cfg):
    out, meta = prepared
    assert meta["data_source"] == "synthetic" and meta["n_failed"] == 0
    subj = {sp: set(meta["splits"][sp]["subjects"]) for sp in dp.SPLITS}
    assert subj["train"].isdisjoint(subj["val"]) and subj["train"].isdisjoint(subj["test"]) and subj["val"].isdisjoint(subj["test"])
    for sp in dp.SPLITS:
        X, info = dp.load_split(out, sp)
        assert X.shape[1:] == (6, 400) and X.dtype == np.float32 and len(info) == len(X)
        assert set(info.subject) <= subj[sp]
        assert set(info.label) == {0, 1}
        # label convention from file name: F* -> 1, D* -> 0
        assert (info[info.recording.str.startswith("F")].label == 1).all()
        assert (info[info.recording.str.startswith("D")].label == 0).all()
    _, mva = dp.load_split(out, "val"); _, mte = dp.load_split(out, "test")
    assert (mva.kind == "eval").all() and (mte.kind == "eval").all()      # no augmentation outside train
    assert not (set(mva.recording) & set(mte.recording))
    _, mtr = dp.load_split(out, "train")
    assert (mtr.kind == "aug").sum() == cfg["data"]["fall_aug_crops"] * (mtr[(mtr.label == 1) & (mtr.kind == "eval")].shape[0])


def test_normalization_uses_train_only(prepared):
    out, meta = prepared
    X, _ = dp.load_split(out, "train")
    x = np.asarray(X, dtype=np.float64)
    assert np.allclose(meta["normalization"]["mean"], x.mean(axis=(0, 2)), rtol=1e-4, atol=1e-5)
    assert np.allclose(meta["normalization"]["std"], x.std(axis=(0, 2)), rtol=1e-3)


def _rf_namespace(path: Path):
    """Pull ONLY the constants and functions out of the RF script (it runs a pipeline at import)."""
    import numpy, pandas
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("SAMPLING_RATE", "WINDOW_SECONDS", "WINDOW_SIZE"))]
    ns = {"np": numpy, "pd": pandas, "Path": Path}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_parity_with_existing_rf_preprocessing(synth_root):
    rf_src = dp.find_project_root() / "ml-model" / "fall_detection" / "preprocess_sisfall.py"
    if not rf_src.exists():
        pytest.skip("RF preprocessing script not found (tests run outside the project tree)")
    ns = _rf_namespace(rf_src)
    files = sorted(synth_root.glob("SA0[12]/*.txt"))[:8]
    assert files
    for f in files:
        info = dp.parse_recording_name(f)
        rf_df = ns["extract_window_features"](ns["load_sisfall_file"](f), is_fall_recording=info.is_fall)
        wins, _ = dp.eval_windows(dp.load_sisfall_recording(f), info.is_fall, 400)
        mine = rf_features(wins[:, :, :6].transpose(0, 2, 1).astype(np.float32), dp.ALL_COLUMNS[:6],
                           list(rf_df.columns.drop("label")))
        assert len(mine) == len(rf_df)
        np.testing.assert_allclose(mine.to_numpy(), rf_df.drop(columns="label").to_numpy(), rtol=1e-6, atol=1e-9)
        assert (rf_df.label == info.label).all()
