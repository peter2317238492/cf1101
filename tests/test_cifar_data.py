import pickle
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Subset

from src.cifar.data import (
    CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, ROOT_ENV, SPLIT_FILES, SPLITS_DIR, balanced_subset,
    build_cifar_datasets, build_cifar_loaders, load_cifar10, make_split, read_split, resolve_root, write_split_files,
)
from src.trainer import run_experiment, set_seed

PER_CLASS, TEST_PER_CLASS, VAL_PER_CLASS = 12, 3, 2  # fake data: 120 train (100 / 20 split), 30 test


def _dump(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_fake_cifar(root):
    """Tiny copy of cifar-10-batches-py. Image i has R(0,0) = i and G(0,0) = 255 - i, so order and HWC layout
    can be checked after loading. Returns the train labels."""
    d = root / "cifar-10-batches-py"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, per_class, parts in (("data_batch", PER_CLASS, 5), ("test_batch", TEST_PER_CLASS, 1)):
        labels = rng.permutation(np.repeat(np.arange(10), per_class))
        data = rng.integers(0, 256, size=(len(labels), 3072), dtype=np.uint8)
        data[:, 0] = np.arange(len(labels))
        data[:, 1024] = 255 - np.arange(len(labels))
        for i, part in enumerate(np.array_split(np.arange(len(labels)), parts), 1):
            fname = f"{name}_{i}" if parts > 1 else name
            _dump(d / fname, {b"data": data[part], b"labels": labels[part].tolist()})
        if name == "data_batch":
            train_labels = labels
    _dump(d / "batches.meta", {b"label_names": [c.encode() for c in CIFAR10_CLASSES]})
    return train_labels


@pytest.fixture
def fake_cifar(tmp_path):
    root, splits = tmp_path / "cifar10", tmp_path / "splits"
    labels = write_fake_cifar(root)
    write_split_files(labels, splits, val_per_class=VAL_PER_CLASS)
    return root, splits


@pytest.fixture
def cifar_cfg(make_cfg):
    def _make(**overrides):
        base = dict(model="tiny_test_net", num_classes=10, img_size=32, aug="crop_flip", bs=8,
                    norm_mean=CIFAR10_MEAN, norm_std=CIFAR10_STD)
        base.update(overrides)
        return make_cfg(**base)

    return _make


# ---------- split ----------

def test_make_split_is_stratified_disjoint_and_complete():
    labels = np.random.default_rng(0).permutation(np.repeat(np.arange(10), 5000))
    train, val = make_split(labels)
    assert (len(train), len(val)) == (45000, 5000)
    assert np.bincount(labels[train]).tolist() == [4500] * 10
    assert np.bincount(labels[val]).tolist() == [500] * 10
    assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(50000))
    assert np.all(np.diff(train) > 0) and np.all(np.diff(val) > 0)


def test_make_split_depends_only_on_seed():
    labels = np.random.default_rng(0).permutation(np.repeat(np.arange(10), 50))
    a, b = make_split(labels, val_per_class=5), make_split(labels, val_per_class=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], make_split(labels, val_per_class=5, seed=43)[1])


def test_committed_split_files_are_the_seed42_split():
    train, val = read_split("train"), read_split("val")
    assert (len(train), len(val)) == (45000, 5000)
    assert np.bincount(train["label"]).tolist() == [4500] * 10
    assert np.bincount(val["label"]).tolist() == [500] * 10
    labels = np.full(50000, -1)
    labels[train["index"]], labels[val["index"]] = train["label"], val["label"]
    assert (labels >= 0).all()  # disjoint and covering all 50k
    regen_train, regen_val = make_split(labels)
    assert np.array_equal(regen_train, train["index"]) and np.array_equal(regen_val, val["index"])


def test_committed_split_files_regenerate_byte_identical(tmp_path):
    train, val = read_split("train"), read_split("val")
    labels = np.empty(50000, dtype=np.int64)
    labels[train["index"]], labels[val["index"]] = train["label"], val["label"]
    write_split_files(labels, tmp_path)
    for name in SPLIT_FILES.values():
        committed = (SPLITS_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        assert (tmp_path / name).read_bytes() == committed


def test_write_split_files_is_idempotent_and_guards_overwrite(tmp_path):
    labels = np.random.default_rng(0).permutation(np.repeat(np.arange(10), 20))
    assert write_split_files(labels, tmp_path, val_per_class=3) == {"train": "created", "val": "created"}
    first = {n: (tmp_path / n).read_bytes() for n in SPLIT_FILES.values()}
    assert write_split_files(labels, tmp_path, val_per_class=3) == {"train": "unchanged", "val": "unchanged"}
    assert {n: (tmp_path / n).read_bytes() for n in SPLIT_FILES.values()} == first
    assert b"\r" not in first[SPLIT_FILES["val"]]
    with pytest.raises(FileExistsError, match="force"):
        write_split_files(labels, tmp_path, val_per_class=3, seed=7)
    assert {n: (tmp_path / n).read_bytes() for n in SPLIT_FILES.values()} == first  # nothing half-written
    assert write_split_files(labels, tmp_path, val_per_class=3, seed=7, force=True)["val"] == "overwritten"


def test_balanced_subset_is_balanced_and_nested():
    labels = np.random.default_rng(0).permutation(np.repeat(np.arange(10), 30))
    small, big = balanced_subset(labels, 25), balanced_subset(labels, 100)
    assert set(small) <= set(big)
    assert np.bincount(labels[big]).tolist() == [10] * 10
    assert np.ptp(np.bincount(labels[small], minlength=10)) <= 1
    assert balanced_subset(labels, 25) == small
    with pytest.raises(ValueError, match="n_train"):
        balanced_subset(labels, 301)


# ---------- raw data ----------

def test_load_cifar10_keeps_file_order_and_hwc_layout(fake_cifar):
    root, _ = fake_cifar
    images, labels = load_cifar10(root)
    assert images.shape == (10 * PER_CLASS, 32, 32, 3) and images.dtype == np.uint8
    assert labels.dtype == np.int64 and np.bincount(labels).tolist() == [PER_CLASS] * 10
    assert images[:, 0, 0, 0].tolist() == list(range(len(labels)))
    assert images[:, 0, 0, 1].tolist() == [255 - i for i in range(len(labels))]
    test_images, test_labels = load_cifar10(root, train=False)
    assert test_images.shape == (10 * TEST_PER_CLASS, 32, 32, 3) and len(test_labels) == 10 * TEST_PER_CLASS


def test_resolve_root_accepts_parent_batch_dir_and_env(fake_cifar, monkeypatch):
    root, _ = fake_cifar
    batch_dir = root / "cifar-10-batches-py"
    assert resolve_root(root) == batch_dir
    assert resolve_root(batch_dir) == batch_dir
    monkeypatch.setenv(ROOT_ENV, str(root))
    assert resolve_root() == batch_dir


def test_missing_data_error_names_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=ROOT_ENV):
        load_cifar10(tmp_path / "nowhere")


# ---------- loaders ----------

def test_loaders_yield_normalized_float_images_and_int_labels(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    loaders = build_cifar_loaders(cifar_cfg(), root=root, splits_dir=splits)
    assert loaders["class_to_idx"] == {name: i for i, name in enumerate(CIFAR10_CLASSES)}
    sizes = {k: len(loaders[k].dataset) for k in ("train", "val", "test")}
    assert sizes == {"train": 10 * (PER_CLASS - VAL_PER_CLASS), "val": 10 * VAL_PER_CLASS, "test": 10 * TEST_PER_CLASS}
    for k in ("train", "val", "test"):
        x, y = next(iter(loaders[k]))
        assert x.shape == (8, 3, 32, 32) and x.dtype == torch.float32
        assert y.shape == (8,) and y.dtype == torch.int64
        assert x.min() < -1 and x.max() > 1  # normalized, not raw [0, 1]


def test_eval_images_use_cfg_normalization(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    ds = build_cifar_datasets(cifar_cfg(aug="none"), root=root, splits_dir=splits)
    images, labels = load_cifar10(root)
    for k in ("train", "val"):
        x, y = ds[k][0]
        i = ds[k].source_index[0]
        expected = (torch.from_numpy(images[i]).permute(2, 0, 1) / 255 - torch.tensor(CIFAR10_MEAN)[:, None, None]) \
            / torch.tensor(CIFAR10_STD)[:, None, None]
        assert torch.allclose(x, expected, atol=1e-5) and y == labels[i]
    assert set(ds["val"].source_index) == set(read_split("val", splits)["index"])


def test_unknown_aug_lists_presets(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    with pytest.raises(ValueError, match="crop_flip"):
        build_cifar_loaders(cifar_cfg(aug="cutmix"), root=root, splits_dir=splits)


def test_n_train_takes_balanced_subset_of_train_split(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    train = build_cifar_datasets(cifar_cfg(n_train=30), root=root, splits_dir=splits)["train"]
    assert isinstance(train, Subset) and len(train) == 30
    assert np.bincount(train.dataset.targets[train.indices]).tolist() == [3] * 10
    assert set(train.dataset.source_index[train.indices]) <= set(read_split("train", splits)["index"])


def test_train_indices_wrap_the_train_split_in_subset(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    picked = read_split("train", splits)["index"].to_numpy()[[5, 0, 17, 17]]
    train = build_cifar_loaders(cifar_cfg(), root=root, train_indices=picked, splits_dir=splits)["train"].dataset
    assert isinstance(train, Subset)
    assert train.dataset.source_index[train.indices].tolist() == picked.tolist()


@pytest.mark.parametrize("bad", ["val", "out_of_range", "empty"])
def test_train_indices_must_come_from_train_split(fake_cifar, cifar_cfg, bad):
    root, splits = fake_cifar
    indices = {"val": read_split("val", splits)["index"].tolist()[:1], "out_of_range": [10 ** 6], "empty": []}[bad]
    with pytest.raises(ValueError, match="train_indices"):
        build_cifar_loaders(cifar_cfg(), root=root, train_indices=indices, splits_dir=splits)


def test_train_indices_and_n_train_are_exclusive(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    with pytest.raises(ValueError, match="not both"):
        build_cifar_loaders(cifar_cfg(n_train=10), root=root, train_indices=[0], splits_dir=splits)


def test_split_files_must_match_the_data(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    path = splits / SPLIT_FILES["val"]
    df = pd.read_csv(path)
    df.loc[0, "label"] = (df.loc[0, "label"] + 1) % 10
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels disagree"):
        build_cifar_loaders(cifar_cfg(), root=root, splits_dir=splits)
    df.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="partition"):
        build_cifar_loaders(cifar_cfg(), root=root, splits_dir=splits)


def test_cfg_must_fit_cifar(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    with pytest.raises(ValueError, match="num_classes"):
        build_cifar_loaders(cifar_cfg(num_classes=2), root=root, splits_dir=splits)
    with pytest.raises(ValueError, match="32px"):
        build_cifar_loaders(cifar_cfg(model="cifar_resnet18", img_size=224), root=root, splits_dir=splits)


def test_img_size_resizes_for_models_without_fixed_input(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    loaders = build_cifar_loaders(cifar_cfg(model="resnet18", img_size=64), root=root, splits_dir=splits)
    assert next(iter(loaders["train"]))[0].shape == (8, 3, 64, 64)
    assert next(iter(loaders["val"]))[0].shape == (8, 3, 64, 64)


def test_train_loader_is_reproducible_given_seed(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    cfg = cifar_cfg()

    def first_batch():
        set_seed(cfg.seed, cfg.deterministic)
        return next(iter(build_cifar_loaders(cfg, root=root, splits_dir=splits)["train"]))

    (xa, ya), (xb, yb) = first_batch(), first_batch()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_datasets_pickle_for_windows_workers(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    ds = build_cifar_datasets(cifar_cfg(aug="none", n_train=20), root=root, splits_dir=splits)
    for k in ("train", "val", "test"):
        clone = pickle.loads(pickle.dumps(ds[k]))
        assert torch.equal(clone[3][0], ds[k][3][0]) and clone[3][1] == ds[k][3][1]


def test_run_experiment_on_fake_cifar_records_test_metrics(fake_cifar, cifar_cfg):
    root, splits = fake_cifar
    cfg = cifar_cfg(task="cifar10", epochs=1)
    res = run_experiment(cfg, partial(build_cifar_loaders, root=root, splits_dir=splits), eval_test=True)
    assert res["test_metrics"]["confusion"].shape == (10, 10)
    row = pd.read_csv(cfg.results_csv).iloc[-1]
    assert row["n_train"] == 10 * (PER_CLASS - VAL_PER_CLASS) and row["n_val"] == 10 * VAL_PER_CLASS
    assert 0 <= row["test_acc"] <= 1


# ---------- real data (skipped unless $CIFAR10_ROOT or data/cifar10 holds CIFAR-10) ----------

def test_real_data_matches_committed_split(tmp_path):
    try:
        resolve_root()
    except FileNotFoundError:
        pytest.skip(f"real CIFAR-10 not found; set {ROOT_ENV}")
    images, labels = load_cifar10()
    assert images.shape == (50000, 32, 32, 3) and np.bincount(labels).tolist() == [5000] * 10
    assert np.bincount(load_cifar10(train=False)[1]).tolist() == [1000] * 10
    train, val = read_split("train"), read_split("val")
    assert np.array_equal(labels[train["index"]], train["label"]) and np.array_equal(labels[val["index"]], val["label"])
    write_split_files(labels, tmp_path)
    for name in SPLIT_FILES.values():
        assert (tmp_path / name).read_bytes() == (SPLITS_DIR / name).read_bytes().replace(b"\r\n", b"\n")
