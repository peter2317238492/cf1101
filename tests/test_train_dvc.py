"""scripts/train_dvc.py and its stand-in loader (src/dvc_standin.py) on a tiny fake ImageFolder."""
import random

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts import train_dvc
from src.dvc_standin import (balanced_subset, build_standin_loaders, eval_transform, resolve_root,
                             train_transform)
from src.trainer import TrainConfig, _seed_worker

N_TRAIN_PER_CLASS = 12  # cat.0 .. cat.11: string order (0, 1, 10, 11, 2, ...) differs from numeric order
N_VAL_PER_CLASS = 3


def _write_images(folder, prefix, n, seed):
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        colour = tuple(rng.randrange(256) for _ in range(3))
        Image.new("RGB", (24 + i % 5, 20), colour).save(folder / f"{prefix}.{i}.jpg", quality=90)


@pytest.fixture
def fake_dvc(tmp_path):
    root = tmp_path / "datasets"
    for s, (split, n) in enumerate((("train", N_TRAIN_PER_CLASS), ("val", N_VAL_PER_CLASS))):
        for k, cls in enumerate(("cat", "dog")):
            _write_images(root / split / cls, cls, n, seed=10 * s + k)
    return root


@pytest.fixture
def dvc_cfg(make_cfg):
    def _make(**overrides):
        return make_cfg(**{"task": "dvc", "img_size": 16, "bs": 4, "aug": "flip", **overrides})
    return _make


def _names(subset):
    return sorted(subset.dataset.samples[i][0].replace("\\", "/").rsplit("/", 1)[1] for i in subset.indices)


# ---------- stand-in loader ----------

def test_loaders_yield_cat0_dog1_and_cfg_sized_images(fake_dvc, dvc_cfg):
    loaders = build_standin_loaders(dvc_cfg(), root=fake_dvc)
    assert loaders["class_to_idx"] == {"cat": 0, "dog": 1}
    assert loaders["train"].dataset.class_to_idx == {"cat": 0, "dog": 1}
    assert len(loaders["train"].dataset) == 2 * N_TRAIN_PER_CLASS
    assert len(loaders["val"].dataset) == 2 * N_VAL_PER_CLASS
    x, y = next(iter(loaders["val"]))
    assert x.shape == (4, 3, 16, 16) and x.dtype == torch.float32 and y.dtype == torch.int64
    labels = [y for _, y in loaders["val"].dataset.samples]
    assert labels == [0] * N_VAL_PER_CLASS + [1] * N_VAL_PER_CLASS
    assert all(p.replace("\\", "/").split("/")[-2] == ("cat", "dog")[y] for p, y in loaders["val"].dataset.samples)


@pytest.mark.parametrize("classes", [("cat", "dog", "horse"), ("cats", "dogs"), ("dog", "zebra")])
def test_label_mapping_other_than_cat0_dog1_is_rejected(tmp_path, dvc_cfg, classes):
    for split in ("train", "val"):
        for cls in classes:
            _write_images(tmp_path / split / cls, cls, 2, seed=0)
    with pytest.raises(ValueError, match="class_to_idx must be"):
        build_standin_loaders(dvc_cfg(), root=tmp_path)


def test_n_train_takes_n_over_2_per_class_from_sorted_files_with_own_rng(fake_dvc, dvc_cfg):
    loaders = build_standin_loaders(dvc_cfg(n_train=6), root=fake_dvc)
    got = _names(loaders["train"].dataset)
    expected = []
    for cls in ("cat", "dog"):
        files = sorted(f"{cls}.{i}.jpg" for i in range(N_TRAIN_PER_CLASS))
        random.Random(42).shuffle(files)  # a fresh Random(42) per class
        expected += files[:3]
    assert got == sorted(expected)
    assert len(loaders["val"].dataset) == 2 * N_VAL_PER_CLASS  # val is never subsampled by n_train


def test_n_train_subsets_are_nested_and_seed_independent(fake_dvc, dvc_cfg):
    small = _names(build_standin_loaders(dvc_cfg(n_train=4, seed=0), root=fake_dvc)["train"].dataset)
    big = _names(build_standin_loaders(dvc_cfg(n_train=10, seed=1), root=fake_dvc)["train"].dataset)
    again = _names(build_standin_loaders(dvc_cfg(n_train=4, seed=2), root=fake_dvc)["train"].dataset)
    assert set(small) < set(big) and small == again


@pytest.mark.parametrize("n", [0, 3, 2 * N_TRAIN_PER_CLASS + 2])
def test_bad_n_train_is_rejected(fake_dvc, dvc_cfg, n):
    with pytest.raises(ValueError, match="n_train"):
        build_standin_loaders(dvc_cfg(n_train=n), root=fake_dvc)


def test_balanced_subset_none_keeps_everything(fake_dvc):
    ds = ImageFolder(fake_dvc / "train")
    assert balanced_subset(ds, None) is ds


def test_transforms_follow_the_contract(dvc_cfg):
    eval_tf = eval_transform(16, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    resize, crop, _, norm = eval_tf.transforms
    assert isinstance(resize, transforms.Resize) and resize.size == int(16 * 256 / 224)
    assert isinstance(crop, transforms.CenterCrop) and crop.size == (16, 16)
    assert list(norm.mean) == [0.1, 0.2, 0.3] and list(norm.std) == [0.4, 0.5, 0.6]
    flip = [type(t) for t in train_transform(dvc_cfg(aug="flip")).transforms]
    none = [type(t) for t in train_transform(dvc_cfg(aug="none")).transforms]
    assert transforms.RandomHorizontalFlip in flip and transforms.RandomHorizontalFlip not in none
    assert none == [type(t) for t in eval_tf.transforms]


def test_val_images_use_the_eval_transform_with_cfg_normalisation(fake_dvc, dvc_cfg):
    cfg = dvc_cfg(norm_mean=(0.5, 0.5, 0.5), norm_std=(0.25, 0.25, 0.25))
    val = build_standin_loaders(cfg, root=fake_dvc)["val"].dataset
    path, _ = val.samples[0]
    expected = eval_transform(16, cfg.norm_mean, cfg.norm_std)(Image.open(path).convert("RGB"))
    assert torch.equal(val[0][0], expected)


def test_unknown_and_unimplemented_augs_raise(fake_dvc, dvc_cfg):
    with pytest.raises(ValueError, match="flip_rot"):  # the message lists the agreed presets
        build_standin_loaders(dvc_cfg(aug="crop_flip"), root=fake_dvc)
    for aug in ("flip_rot", "full"):
        with pytest.raises(NotImplementedError, match="build_dvc_loaders"):
            build_standin_loaders(dvc_cfg(aug=aug), root=fake_dvc)


def test_sampler_name_is_refused_by_the_stand_in_loader(fake_dvc, dvc_cfg):
    with pytest.raises(ValueError, match="no sampler"):  # it would be recorded in runs.csv but never applied
        build_standin_loaders(dvc_cfg(sampler="weighted"), root=fake_dvc)


def test_train_loader_is_seeded_with_loader_kwargs(fake_dvc, dvc_cfg):
    def order(seed):
        loader = build_standin_loaders(dvc_cfg(seed=seed, aug="none"), root=fake_dvc)["train"]
        assert loader.worker_init_fn is _seed_worker and loader.generator.initial_seed() == seed
        return torch.cat([y for _, y in loader]).tolist(), torch.cat([x for x, _ in loader])

    (y0, x0), (y0b, x0b), (_, x1) = order(0), order(0), order(1)
    assert y0 == y0b and torch.equal(x0, x0b)
    assert len(y0) == 2 * N_TRAIN_PER_CLASS  # 24 images, bs 4: drop_last loses nothing
    assert not torch.equal(x0, x1)


def test_data_root_comes_from_env_then_project_default(fake_dvc, tmp_path, monkeypatch):
    monkeypatch.setenv("DVC_ROOT", str(fake_dvc))
    assert resolve_root() == fake_dvc
    assert resolve_root(fake_dvc) == fake_dvc
    monkeypatch.setenv("DVC_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="DVC_ROOT"):
        resolve_root()


def test_num_classes_must_be_two(fake_dvc, dvc_cfg):
    with pytest.raises(ValueError, match="2 classes"):
        build_standin_loaders(dvc_cfg(num_classes=10), root=fake_dvc)


# ---------- CLI: --set / --seeds ----------

def test_set_values_are_parsed_by_yaml_and_coerced_by_trainconfig(tmp_path):
    overrides = train_dvc.parse_overrides([
        "n_train=2000", "lr=1e-3", "patience=null", "amp=false", "norm_mean=[0.5, 0.5, 0.5]",
        "exp_name=dvc_lp_dryrun", "out_dir=C:/Users/x/runs", "notes=a=b", "epochs=2", "epochs=1"])
    assert overrides == {"n_train": 2000, "lr": "1e-3", "patience": None, "amp": False,
                         "norm_mean": [0.5, 0.5, 0.5], "exp_name": "dvc_lp_dryrun",
                         "out_dir": "C:/Users/x/runs", "notes": "a=b", "epochs": 1}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"exp_name": "base", "lr": 0.01, "epochs": 5, "seed": 3}), encoding="utf-8")
    (cfg,) = train_dvc.build_configs(cfg_path, None, overrides)
    assert (cfg.n_train, cfg.lr, cfg.patience, cfg.amp, cfg.norm_mean) == (2000, 1e-3, None, False, (0.5,) * 3)
    assert (cfg.exp_name, cfg.out_dir, cfg.epochs, cfg.seed) == ("dvc_lp_dryrun", "C:/Users/x/runs", 1, 3)
    assert cfg.notes == "a=b; stand-in loader"


@pytest.mark.parametrize("bad", ["n_train", "=3", ""])
def test_set_without_key_value_is_an_error(bad):
    with pytest.raises(ValueError, match="key=value"):
        train_dvc.parse_overrides([bad])


def test_set_unknown_field_and_seed_conflict_are_errors(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("exp_name: base\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown TrainConfig field"):
        train_dvc.build_configs(cfg_path, None, {"n_trian": 2000})
    with pytest.raises(ValueError, match="--seeds"):
        train_dvc.build_configs(cfg_path, [0, 1], {"seed": 2})
    cfgs = train_dvc.build_configs(cfg_path, [0, 1, 2], {})
    assert [c.seed for c in cfgs] == [0, 1, 2] and all(c.notes == "stand-in loader" for c in cfgs)


def test_cli_rejects_bad_set_before_training(tmp_path, capsys):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("exp_name: base\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_dvc.main(["--config", str(cfg_path), "--set", "n_trian=2000", "--data-root", str(tmp_path)])
    assert "n_trian" in capsys.readouterr().err


def test_main_trains_one_run_per_seed_through_run_experiment(fake_dvc, tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "exp_name": "tiny_dvc", "task": "dvc", "model": "tiny_test_net", "num_classes": 2, "aug": "flip",
        "img_size": 16, "bs": 4, "epochs": 1, "lr": 0.05, "scheduler": "none", "amp": False,
        "channels_last": False, "num_workers": 0, "device": "cpu",
        "out_dir": str(tmp_path / "runs"), "results_csv": str(tmp_path / "runs.csv")}), encoding="utf-8")
    results = train_dvc.main(["--config", str(cfg_path), "--seeds", "0", "1", "--data-root", str(fake_dvc),
                              "--set", "n_train=8", "notes=unit"])
    assert len(results) == 2 and all("model" not in r for r in results)
    rows = pd.read_csv(tmp_path / "runs.csv")
    assert rows["seed"].tolist() == [0, 1]
    assert rows["n_train"].tolist() == [8, 8] and rows["n_val"].tolist() == [6, 6]
    assert (rows["notes"] == "unit; stand-in loader").all() and (rows["task"] == "dvc").all()
    ckpt = torch.load(tmp_path / "runs" / "tiny_dvc" / "seed1" / "best.pt", weights_only=True)
    assert ckpt["class_to_idx"] == {"cat": 0, "dog": 1} and ckpt["img_size"] == 16


def test_cli_refuses_duplicate_seeds_and_finished_runs_before_training(fake_dvc, tmp_path, capsys):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "exp_name": "tiny_dvc", "task": "dvc", "model": "tiny_test_net", "aug": "none", "img_size": 16, "bs": 4,
        "epochs": 1, "scheduler": "none", "amp": False, "num_workers": 0, "device": "cpu",
        "out_dir": str(tmp_path / "runs"), "results_csv": str(tmp_path / "runs.csv")}), encoding="utf-8")
    cli = ["--config", str(cfg_path), "--data-root", str(fake_dvc), "--set", "n_train=8", "--seeds"]
    with pytest.raises(SystemExit):
        train_dvc.main(cli + ["0", "0"])
    assert "duplicate --seeds" in capsys.readouterr().err
    train_dvc.main(cli + ["1"])
    with pytest.raises(SystemExit):
        train_dvc.main(cli + ["0", "1"])  # seed 1 is done: refused before seed 0 trains
    assert "already holds a finished run" in capsys.readouterr().err
    assert pd.read_csv(tmp_path / "runs.csv")["seed"].tolist() == [1]
