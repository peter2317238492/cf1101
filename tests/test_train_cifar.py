import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import RandomSampler, Sampler

import src.cifar
import src.cifar.data as cifar_data
import src.cifar.train as T
from scripts.train_cifar import main, parse_overrides
from src.cifar.data import read_split, write_split_files
from src.cifar.train import build_loss, build_sampler, register_loss, register_sampler
from src.trainer import IMAGENET_MEAN, PROJECT_ROOT, TrainConfig
from tests.test_cifar_data import PER_CLASS, VAL_PER_CLASS, write_fake_cifar

CONFIG = PROJECT_ROOT / "configs" / "cifar_baseline.yaml"
N_TRAIN = 10 * (PER_CLASS - VAL_PER_CLASS)
FAST_CPU = ["model=tiny_test_net", "pretrained=false", "img_size=32", "bs=8", "epochs=1", "num_workers=0",
            "device=cpu", "amp=false", "channels_last=false", "deterministic=true"]


@pytest.fixture
def fake_cifar(tmp_path, monkeypatch):
    """Tiny CIFAR copy; the CLI's default split files are pointed at its own 100 / 20 split."""
    root, splits = tmp_path / "cifar10", tmp_path / "splits"
    write_split_files(write_fake_cifar(root), splits, val_per_class=VAL_PER_CLASS)
    monkeypatch.setattr(cifar_data, "SPLITS_DIR", splits)
    return root


@pytest.fixture
def cli(fake_cifar, tmp_path):
    def run(*args):
        return main(["--config", str(CONFIG), "--root", str(fake_cifar), "--set", *FAST_CPU,
                     f"out_dir={(tmp_path / 'runs').as_posix()}", f"results_csv={(tmp_path / 'runs.csv').as_posix()}",
                     *args])  # a test's own --set comes last and wins
    return run


@pytest.fixture
def registries():
    """Losses / samplers registered inside a test are dropped afterwards. Real modules under src/cifar are
    imported first, so their registrations survive the restore (they would not be re-imported later)."""
    T._autodiscover()
    saved = dict(T._LOSSES), dict(T._SAMPLERS)
    yield
    for table, old in zip((T._LOSSES, T._SAMPLERS), saved):
        table.clear()
        table.update(old)


def _write_indices(path, indices):
    pd.DataFrame({"index": indices}).to_csv(path, index=False)
    return str(path)


# ---------- CLI ----------

def test_parse_overrides_reads_yaml_values():
    got = parse_overrides(["lr=1e-3", "epochs=3", "n_train=null", "pretrained=true", "norm_mean=[0.5, 0.5, 0.5]",
                           "notes=a: b=c", "exp_name= x ", "patience=", "member=no", "loss=0123"])
    # text fields stay as typed (YAML would turn "no" into False and "a: b" into a dict); empty = null
    assert got == {"lr": "1e-3", "epochs": 3, "n_train": None, "pretrained": True, "norm_mean": [0.5, 0.5, 0.5],
                   "notes": "a: b=c", "exp_name": "x", "patience": None, "member": "no", "loss": "0123"}
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["lr"])
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["=1"])
    with pytest.raises(ValueError, match="not valid YAML"):
        parse_overrides(["norm_mean=[0.5,"])


def test_duplicate_seeds_are_refused(cli, tmp_path):
    with pytest.raises(ValueError, match="duplicate --seeds"):
        cli("--seeds", "0", "0")
    assert not (tmp_path / "runs").exists()


def test_baseline_config_is_the_pretrained_224_recipe():
    cfg = TrainConfig.from_yaml(CONFIG)
    assert (cfg.model, cfg.pretrained, cfg.freeze, cfg.img_size, cfg.num_classes) == ("resnet18", True, "none", 224, 10)
    assert (cfg.task, cfg.member, cfg.exp_name, cfg.loss, cfg.sampler) == (
        "cifar10", "B", "cifar_baseline_r18_pt224", "ce", "none")
    assert cfg.norm_mean == IMAGENET_MEAN and cfg.aug == "crop_flip"


def test_set_overrides_and_seeds_reach_the_runs(cli, tmp_path):
    cli("--seeds", "0", "1", "--set", "exp_name=over", "lr=0.05", "n_train=30", "label_smoothing=0.1")
    rows = pd.read_csv(tmp_path / "runs.csv")
    assert rows["seed"].tolist() == [0, 1] and set(rows["exp_name"]) == {"over"}
    assert rows["lr"].tolist() == [0.05, 0.05] and rows["n_train"].tolist() == [30, 30]
    assert set(rows["label_smoothing"]) == {0.1} and set(rows["model"]) == {"tiny_test_net"}
    with pytest.raises(FileExistsError, match="seed1"):  # refused before seed 2 trains
        cli("--seeds", "2", "1", "--set", "exp_name=over")
    assert len(pd.read_csv(tmp_path / "runs.csv")) == 2

    cli("--set", "exp_name=default_seed", "seed=3")  # no --seeds: the config's (overridden) seed
    assert pd.read_csv(tmp_path / "runs.csv")["seed"].tolist() == [0, 1, 3]


def test_unknown_config_field_fails_before_training(cli, tmp_path):
    with pytest.raises(ValueError, match="unknown TrainConfig field.*lrr"):
        cli("--set", "lrr=0.1")
    assert not (tmp_path / "runs").exists()


# ---------- --train-indices ----------

def test_train_indices_subset_is_used(cli, tmp_path):
    train = read_split("train")["index"].to_numpy()
    csv = _write_indices(tmp_path / "subset.csv", train[[3, 0, 9, 9, 40]])
    res = cli("--train-indices", csv, "--set", "exp_name=sub", "notes=imb test")
    row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
    assert row["n_train"] == 5
    assert row["notes"].startswith("imb test; idx=subset.csv@") and row["notes"].endswith(" (5)")
    assert res[0]["best_epoch"] == 1


def test_run_cifar_records_the_subset_from_python_too(fake_cifar, make_cfg, tmp_path):
    idx = read_split("train")["index"].to_numpy()[[1, 2, 3, 3]]
    cfg = make_cfg(model="tiny_test_net", num_classes=10, img_size=32, aug="none", bs=2, epochs=1,
                   norm_mean=cifar_data.CIFAR10_MEAN, norm_std=cifar_data.CIFAR10_STD)
    T.run_cifar(cfg, root=fake_cifar, train_indices=idx)
    T.run_cifar(make_cfg(**{**cfg.to_dict(), "seed": 1}), root=fake_cifar, train_indices=idx)
    notes = pd.read_csv(tmp_path / "results" / "runs.csv")["notes"].tolist()
    assert notes[0] == notes[1] == T.indices_note(idx) and notes[0].startswith("idx=@") and cfg.notes == ""


@pytest.mark.parametrize("bad, match", [
    ("val", "1 of 3 indices are not in the 45k train split .*1 of them are in the 5k val split"),
    ("out_of_range", "not in the 45k train split"),
    ("no_index_column", "'index' column"),
    ("float", "integers"),
])
def test_train_indices_outside_train_split_are_rejected(cli, tmp_path, bad, match):
    train, val = read_split("train")["index"].tolist(), read_split("val")["index"].tolist()
    path = tmp_path / "bad.csv"
    if bad == "val":
        _write_indices(path, [train[0], val[0], train[1]])
    elif bad == "out_of_range":
        _write_indices(path, [train[0], 10 ** 6])
    elif bad == "no_index_column":
        pd.DataFrame({"idx": train[:3]}).to_csv(path, index=False)
    else:
        path.write_text(f"index\n{train[0]}\n1.5\n")
    with pytest.raises(ValueError, match=match):
        cli("--train-indices", str(path))
    assert not (tmp_path / "runs").exists()


# ---------- loss / sampler registries ----------

class FirstN(Sampler):
    """Always yields positions 0..n-1 in order; counts how often it is iterated."""

    def __init__(self, n):
        self.n, self.iterations, self.generator = n, 0, None

    def __iter__(self):
        self.iterations += 1
        return iter(range(self.n))

    def __len__(self):
        return self.n


class CountingCE(nn.CrossEntropyLoss):
    calls = 0

    def forward(self, logits, y):
        CountingCE.calls += 1
        return super().forward(logits, y)


def test_registered_loss_and_sampler_are_used_and_recorded(cli, tmp_path, registries):
    seen = {}

    @register_loss("test_counting_ce")
    def counting_ce(cfg, train_labels):
        seen["loss_labels"] = train_labels
        return CountingCE(label_smoothing=cfg.label_smoothing)

    @register_sampler("test_first16")
    def first16(cfg, train_labels):
        seen["sampler_labels"], seen["sampler"] = train_labels, FirstN(16)
        return seen["sampler"]

    split = read_split("train")
    picked = split.iloc[[5, 0, 17, 17, 30, 31, 32, 33, 40, 41, 42, 43, 50, 51, 52, 53, 60, 61, 62, 63]]
    csv = _write_indices(tmp_path / "subset.csv", picked["index"])
    CountingCE.calls = 0
    cli("--train-indices", csv, "--set", "exp_name=custom", "loss=test_counting_ce", "sampler=test_first16",
        "epochs=2", "label_smoothing=0.05")

    # builders saw the labels of the actual training subset (after --train-indices), duplicates included
    assert seen["loss_labels"].tolist() == picked["label"].tolist() == seen["sampler_labels"].tolist()
    # the sampler drove the train loader (2 batches of 8 per epoch, not 20 // 8 shuffled ones), seeded from cfg.seed
    assert seen["sampler"].iterations == 2 and CountingCE.calls == 4
    assert seen["sampler"].generator.initial_seed() == 0
    row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
    assert (row["loss"], row["sampler"], row["n_train"]) == ("test_counting_ce", "test_first16", 20)
    metrics = json.loads((tmp_path / "runs" / "custom" / "seed0" / "metrics.json").read_text())
    assert metrics["criterion"] == {"type": "CountingCE", "label_smoothing": 0.05}


@pytest.mark.parametrize("field", ["loss", "sampler"])
def test_unknown_loss_or_sampler_lists_available(cli, tmp_path, field):
    with pytest.raises(KeyError, match=rf"unknown {field} 'nope'; available: .*'{'ce' if field == 'loss' else 'none'}'"):
        cli("--set", f"{field}=nope")
    assert not (tmp_path / "runs").exists()


def test_builtin_ce_and_none(make_cfg):
    cfg = make_cfg(label_smoothing=0.1)
    labels = np.array([0, 1, 1])
    crit = build_loss(cfg, labels)
    assert isinstance(crit, nn.CrossEntropyLoss) and crit.label_smoothing == 0.1 and crit.weight is None
    assert build_sampler(cfg, labels) is None


def test_modules_under_src_cifar_are_discovered(tmp_path, monkeypatch, registries, make_cfg):
    pkg_dir = tmp_path / "extra"
    pkg_dir.mkdir()
    (pkg_dir / "fake_imbalance.py").write_text(
        "from torch import nn\n"
        "from src.cifar.train import register_loss\n\n"
        "@register_loss('test_discovered')\n"
        "def build(cfg, train_labels):\n"
        "    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)\n")
    monkeypatch.setattr(src.cifar, "__path__", [*src.cifar.__path__, str(pkg_dir)])
    monkeypatch.delitem(sys.modules, "src.cifar.fake_imbalance", raising=False)
    try:
        assert "test_discovered" in T.available_losses()
        assert isinstance(build_loss(make_cfg(loss="test_discovered"), np.zeros(4)), nn.CrossEntropyLoss)
    finally:
        sys.modules.pop("src.cifar.fake_imbalance", None)


def test_duplicate_registration_is_refused(registries):
    with pytest.raises(ValueError, match="already registered"):
        register_loss("ce")(lambda cfg, labels: None)
    with pytest.raises(ValueError, match="already registered"):
        register_sampler("none")(lambda cfg, labels: None)
    register_loss("ce")(T._ce)  # the same builder again (a re-imported module) is fine


def test_build_cifar_loaders_refuses_a_sampler_name_it_cannot_apply(fake_cifar, make_cfg):
    cfg = make_cfg(model="tiny_test_net", num_classes=10, img_size=32, aug="none", sampler="weighted",
                   norm_mean=cifar_data.CIFAR10_MEAN, norm_std=cifar_data.CIFAR10_STD)
    with pytest.raises(ValueError, match="run_cifar"):
        cifar_data.build_cifar_loaders(cfg, root=fake_cifar)


def test_sampler_replaces_shuffle_in_cifar_loaders(fake_cifar, make_cfg):
    cfg = make_cfg(model="tiny_test_net", num_classes=10, img_size=32, aug="none", bs=4, sampler="first8",
                   norm_mean=cifar_data.CIFAR10_MEAN, norm_std=cifar_data.CIFAR10_STD)
    ds = cifar_data.build_cifar_datasets(cfg, root=fake_cifar)
    with pytest.raises(ValueError, match="cfg.sampler is 'none'"):
        cifar_data.cifar_loaders(make_cfg(model="tiny_test_net", num_classes=10, img_size=32), ds, sampler=FirstN(8))
    sampler = FirstN(8)
    loader = cifar_data.cifar_loaders(cfg, ds, sampler=sampler)["train"]
    assert loader.sampler is sampler and sampler.generator.initial_seed() == cfg.seed
    ys = torch.cat([y for _, y in loader]).tolist()
    assert ys == cifar_data.dataset_targets(ds["train"])[:8].tolist()
    cfg1 = make_cfg(model="tiny_test_net", num_classes=10, img_size=32, aug="none", bs=4, seed=1, sampler="first8",
                    norm_mean=cifar_data.CIFAR10_MEAN, norm_std=cifar_data.CIFAR10_STD)
    cifar_data.cifar_loaders(cfg1, ds, sampler=sampler)  # the same sampler reused for another seed
    assert sampler.generator.initial_seed() == 1
    shuffled = cifar_data.cifar_loaders(cfg, ds)["train"]
    assert isinstance(shuffled.sampler, RandomSampler) and len(shuffled) == N_TRAIN // 4
