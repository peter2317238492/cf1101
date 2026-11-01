"""CIFAR-10 data shared by B (Q8) and C (Q9): official pickle batches, the fixed 45k/5k split, loaders.

    from src.cifar.data import CIFAR10_MEAN, CIFAR10_STD, build_cifar_loaders
    cfg = TrainConfig(task="cifar10", model="cifar_resnet18", num_classes=10, img_size=32, aug="crop_flip",
                      norm_mean=CIFAR10_MEAN, norm_std=CIFAR10_STD, ...)
    res = run_experiment(cfg, build_cifar_loaders, eval_test=True)

The 50k official training images were split once (seed 42, stratified, 4500/500 per class) into
splits/cifar10_train45k.csv and splits/cifar10_val5k.csv, columns `index,label`, where index is the
position in data_batch_1..5 concatenated. Those committed files are the only source of the split:
loaders read them and check them against the data. C's imbalanced subsets are drawn from the 45k
train split only; the official 10k test batch is for the final evaluation only (eval_test=True).

Data root: `root=` argument, else $CIFAR10_ROOT, else data/cifar10 under the project. It may name
the folder holding cifar-10-batches-py or cifar-10-batches-py itself.
"""
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.models.registry import model_input_size
from src.trainer import PROJECT_ROOT, loader_kwargs

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR10_CLASS_TO_IDX = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # per channel over the 50k official training images
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
ROOT_ENV = "CIFAR10_ROOT"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "cifar10"
BATCH_DIR = "cifar-10-batches-py"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLIT_FILES = {"train": "cifar10_train45k.csv", "val": "cifar10_val5k.csv"}
SPLIT_SEED = 42
VAL_PER_CLASS = 500

# cfg.aug presets, applied to training images only (at 32px, before any resize to cfg.img_size).
AUGMENTATIONS = {
    "none": lambda: [],
    "crop_flip": lambda: [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
}


# ---------- raw data ----------

def resolve_root(root=None):
    """Folder holding the pickle batches: root argument > $CIFAR10_ROOT > data/cifar10."""
    base = Path(root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)
    for d in (base / BATCH_DIR, base):
        if (d / "batches.meta").is_file():
            return d
    raise FileNotFoundError(f"no CIFAR-10 pickle batches ({BATCH_DIR}/batches.meta) under {base}; "
                            f"pass root=... or set {ROOT_ENV}")


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10(root=None, train=True):
    """Returns (images uint8 [N, 32, 32, 3], labels int64 [N]) in official order: data_batch_1..5
    concatenated for train (N = 50000), test_batch for test (N = 10000)."""
    d = resolve_root(root)
    names = tuple(n.decode() for n in _unpickle(d / "batches.meta")[b"label_names"])
    if names != CIFAR10_CLASSES:
        raise ValueError(f"{d}: label names {names} differ from {CIFAR10_CLASSES}")
    batches = [_unpickle(d / f) for f in ([f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"])]
    data = np.concatenate([b[b"data"] for b in batches])
    images = np.ascontiguousarray(data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    labels = np.concatenate([np.asarray(b[b"labels"], dtype=np.int64) for b in batches])
    return images, labels


# ---------- the shared 45k / 5k split ----------

def make_split(labels, val_per_class=VAL_PER_CLASS, seed=SPLIT_SEED):
    """Stratified split of training indices. One random.Random(seed) shuffles each class's sorted indices
    in class order; the first val_per_class go to val. Returns sorted (train_idx, val_idx) int64 arrays."""
    labels = np.asarray(labels)
    rng = random.Random(seed)
    train, val = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c).tolist()
        if len(idx) <= val_per_class:
            raise ValueError(f"class {c} has {len(idx)} images, cannot hold out {val_per_class}")
        rng.shuffle(idx)
        val += idx[:val_per_class]
        train += idx[val_per_class:]
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(val), dtype=np.int64)


def _split_csv_text(indices, labels):
    return "index,label\n" + "".join(f"{i},{labels[i]}\n" for i in indices)


def write_split_files(labels, splits_dir=SPLITS_DIR, val_per_class=VAL_PER_CLASS, seed=SPLIT_SEED, force=False):
    """Write the train/val split CSVs (LF line endings). Idempotent: identical files are left alone. A file
    whose content would change is overwritten only with force=True, since C reuses it. Returns {name: status}."""
    splits_dir = Path(splits_dir)
    texts = dict(zip(("train", "val"), (_split_csv_text(idx, labels) for idx in
                                        make_split(labels, val_per_class, seed))))
    status = {}
    for name, text in texts.items():
        path = splits_dir / SPLIT_FILES[name]
        if not path.exists():
            status[name] = "created"
        elif path.read_text(encoding="utf-8").replace("\r\n", "\n") == text:
            status[name] = "unchanged"
        elif force:
            status[name] = "overwritten"
        else:
            raise FileExistsError(f"{path} differs from the regenerated split; it is shared with the team, "
                                  f"so overwrite it only on purpose (force=True / --force)")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if status[name] != "unchanged":
            (splits_dir / SPLIT_FILES[name]).write_text(text, encoding="utf-8", newline="\n")
    return status


def read_split(name, splits_dir=None):
    """DataFrame[index, label] of the shared "train" (45k) or "val" (5k) split; index is into the 50k
    official training images."""
    path = Path(splits_dir or SPLITS_DIR) / SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run python -m scripts.make_cifar_splits")
    return pd.read_csv(path, dtype={"index": np.int64, "label": np.int64})


def _checked_split(labels, splits_dir):
    """(train_idx, val_idx) from the split files, after checking they partition the data and match its labels."""
    train, val = read_split("train", splits_dir), read_split("val", splits_dir)
    tr, va = train["index"].to_numpy(), val["index"].to_numpy()
    if not np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(labels))):
        raise ValueError(f"split files in {splits_dir or SPLITS_DIR} do not partition the "
                         f"{len(labels)} training images")
    if not (np.array_equal(labels[tr], train["label"]) and np.array_equal(labels[va], val["label"])):
        raise ValueError("split file labels disagree with the data: wrong data root or stale split files")
    return tr, va


def balanced_subset(labels, n, seed=SPLIT_SEED):
    """Positions of n samples, class-balanced (counts differ by at most 1) and nested: the n=1000 subset lies
    inside the n=5000 one. Each class's positions are shuffled with random.Random(seed), then classes are
    taken round-robin. Returns sorted positions into `labels`."""
    labels = np.asarray(labels)
    if not 0 < n <= len(labels):
        raise ValueError(f"n_train={n} must be in [1, {len(labels)}]")
    rng = random.Random(seed)
    ranked = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c).tolist()
        rng.shuffle(pos)
        ranked += [(rank, c, p) for rank, p in enumerate(pos)]
    ranked.sort()
    return sorted(p for _, _, p in ranked[:n])


def _train_positions(train_idx, indices, n_total):
    """Positions in the train split of the given official training indices; rejects anything outside it."""
    idx = np.asarray(indices).ravel()
    if idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("train_indices must be a non-empty sequence of ints")
    pos_of = np.full(n_total, -1, dtype=np.int64)
    pos_of[train_idx] = np.arange(len(train_idx))
    pos = np.where((idx >= 0) & (idx < n_total), pos_of[np.clip(idx, 0, n_total - 1)], -1)
    if (pos < 0).any():
        raise ValueError(f"{(pos < 0).sum()} train_indices are not in the 45k train split "
                         f"(e.g. {idx[pos < 0][:5].tolist()}); sample them from read_split('train')")
    return pos.tolist()


# ---------- datasets / loaders ----------

class CIFAR10Images(Dataset):
    """uint8 images [N, 32, 32, 3] + labels -> (transform(PIL image), int label). `source_index` is each
    sample's index in the official file order (train: data_batch_1..5, test: test_batch)."""

    def __init__(self, images, targets, source_index, transform):
        self.images, self.targets, self.source_index, self.transform = images, targets, source_index, transform
        self.class_to_idx = dict(CIFAR10_CLASS_TO_IDX)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.targets[i])


def cifar_transforms(cfg):
    """(train, eval) transforms: the cfg.aug preset (train only), a resize when cfg.img_size != 32,
    then ToTensor + Normalize(cfg.norm_mean, cfg.norm_std)."""
    if cfg.aug not in AUGMENTATIONS:
        raise ValueError(f"unknown CIFAR aug {cfg.aug!r}; choose one of {list(AUGMENTATIONS)}")
    resize = [] if cfg.img_size == 32 else [transforms.Resize(cfg.img_size)]
    tail = [transforms.ToTensor(), transforms.Normalize(cfg.norm_mean, cfg.norm_std)]
    return transforms.Compose(AUGMENTATIONS[cfg.aug]() + resize + tail), transforms.Compose(resize + tail)


def _check_cfg(cfg):
    if cfg.num_classes != len(CIFAR10_CLASSES):
        raise ValueError(f"CIFAR-10 has {len(CIFAR10_CLASSES)} classes but cfg.num_classes={cfg.num_classes}")
    try:
        fixed = model_input_size(cfg.model)
    except KeyError:  # unknown model: build_model reports it with the list of names
        fixed = None
    if fixed is not None and fixed != cfg.img_size:
        raise ValueError(f"model {cfg.model!r} expects {fixed}px inputs but cfg.img_size={cfg.img_size}")


def build_cifar_datasets(cfg, root=None, train_indices=None, splits_dir=None):
    """The datasets behind build_cifar_loaders: {"train", "val", "test", "class_to_idx"}.
    Use it to build your own train loader (e.g. with a WeightedRandomSampler); labels are in `.targets`."""
    _check_cfg(cfg)
    if train_indices is not None and cfg.n_train is not None:
        raise ValueError("pass train_indices or cfg.n_train, not both; subsample train_indices yourself")
    train_tf, eval_tf = cifar_transforms(cfg)
    images, labels = load_cifar10(root, train=True)
    test_images, test_labels = load_cifar10(root, train=False)
    tr, va = _checked_split(labels, splits_dir)
    train = CIFAR10Images(images[tr], labels[tr], tr, train_tf)
    if train_indices is not None:
        train = Subset(train, _train_positions(tr, train_indices, len(labels)))
    elif cfg.n_train is not None:
        train = Subset(train, balanced_subset(labels[tr], cfg.n_train))
    return {
        "train": train,
        "val": CIFAR10Images(images[va], labels[va], va, eval_tf),
        "test": CIFAR10Images(test_images, test_labels, np.arange(len(test_labels)), eval_tf),
        "class_to_idx": dict(CIFAR10_CLASS_TO_IDX),
    }


def build_cifar_loaders(cfg, root=None, train_indices=None, splits_dir=None):
    """Loaders for run_experiment. "train", "val" and "test" yield (x float32 [B, 3, S, S], label int64)
    with S = cfg.img_size; "class_to_idx" maps class names to labels.

    train: the 45k split, augmented with cfg.aug and shuffled reproducibly via loader_kwargs(cfg.seed).
      cfg.n_train takes a class-balanced, nested subset of it (same subset for every cfg.seed).
      train_indices (C's imbalanced subsets) wraps it in Subset; they are indices into the 50k official
      training images and must all come from the 45k split. Duplicates are kept (oversampling).
    val: the 5k split. test: the official 10k test batch; evaluate it once, at the end (eval_test=True).
    Bind root / train_indices / splits_dir with functools.partial before passing it to run_experiment."""
    ds = build_cifar_datasets(cfg, root, train_indices, splits_dir)
    workers = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return {
        "train": DataLoader(ds["train"], batch_size=cfg.bs, shuffle=True, drop_last=len(ds["train"]) > cfg.bs,
                            persistent_workers=cfg.num_workers > 0, **workers, **loader_kwargs(cfg.seed)),
        "val": DataLoader(ds["val"], batch_size=cfg.bs, shuffle=False, persistent_workers=cfg.num_workers > 0,
                          **workers),
        "test": DataLoader(ds["test"], batch_size=cfg.bs, shuffle=False, **workers),
        "class_to_idx": ds["class_to_idx"],
    }
