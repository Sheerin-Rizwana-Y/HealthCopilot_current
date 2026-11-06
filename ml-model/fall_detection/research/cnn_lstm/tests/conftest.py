import copy
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import data_pipeline as dp  # noqa: E402
import prepare_data  # noqa: E402
import synthetic  # noqa: E402
import train as train_mod  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = dp.load_config()
    c = copy.deepcopy(c)
    c["training"].update(max_epochs=4, batch_size=32)
    c["training"]["early_stopping"].update(patience=2)
    c["evaluation"].update(latency_runs=5, latency_warmup=1)
    c["data"]["fall_aug_crops"] = 6
    return c


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return synthetic.write_synthetic_sisfall(tmp_path_factory.mktemp("sis") / "SisFall_dataset",
                                             n_subjects=12, seed=1)


@pytest.fixture(scope="session")
def prepared(cfg, synth_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("prep")
    meta = prepare_data.prepare(cfg, synth_root, out, rf_csv=None, verbose=False)
    return out, meta


@pytest.fixture(scope="session")
def trained(cfg, prepared, tmp_path_factory):
    art = tmp_path_factory.mktemp("art")
    res = train_mod.train(cfg, prepared[0], art, verbose=False)
    return art, res
