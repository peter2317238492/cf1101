"""STAND-IN Dogs-vs-Cats loaders, used by B's scripts until A's `build_dvc_loaders` lands.

    from functools import partial
    from src.dvc_standin import build_standin_loaders
    res = run_experiment(cfg, partial(build_standin_loaders, root=...))

It follows the agreed contract (labels, split rule, preprocessing) but reads the folders directly:
- ImageFolder on {root}/train and {root}/val; class_to_idx must be {'cat': 0, 'dog': 1}. Images go
  through PIL .convert('RGB') (ImageFolder's default loader).
- eval transform: Resize(int(img_size * 256 / 224)) (short side) -> CenterCrop(img_size) -> ToTensor
  -> Normalize(norm_mean, norm_std). Train adds RandomHorizontalFlip after the crop when aug == 'flip'.
- aug presets are `none / flip / flip_rot / full`; this stand-in implements only `none` and `flip`.
- cfg.n_train: per class, the class's files sorted by name, shuffled with their own random.Random(42),
  first n_train/2 kept. Subsets are nested (n=2000 lies inside n=5000) and the same for every seed.
- the train loader shuffles via loader_kwargs(cfg.seed); val is never shuffled.

Data root: `root=` argument, else $DVC_ROOT, else data/dogs-vs-cats/datasets under the project.
"""
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.registry import model_input_size
from src.trainer import DVC_CLASS_TO_IDX, PROJECT_ROOT, loader_kwargs

ROOT_ENV = "DVC_ROOT"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "dogs-vs-cats" / "datasets"
SUBSET_SEED = 42
DVC_AUG_PRESETS = ("none", "flip", "flip_rot", "full")  # agreed names
STANDIN_AUGS = ("none", "flip")                         # flip_rot / full arrive with A's loader


def resolve_root(root=None):
    """Folder holding train/ and val/: root argument > $DVC_ROOT > data/dogs-vs-cats/datasets."""
    base = Path(root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)
    missing = [s for s in ("train", "val") if not (base / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"Dogs-vs-Cats data not found: {base} has no {'/'.join(missing)} folder; "
                                f"pass --data-root / root=... or set {ROOT_ENV}")
    return base


def eval_transform(img_size, norm_mean, norm_std):
    """Test/val preprocessing. Build it from a checkpoint with
    eval_transform(ckpt['img_size'], ckpt['norm_mean'], ckpt['norm_std'])."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def train_transform(cfg):
    if cfg.aug not in DVC_AUG_PRESETS:
        raise ValueError(f"unknown Dogs-vs-Cats aug {cfg.aug!r}; the agreed presets are {list(DVC_AUG_PRESETS)}")
    if cfg.aug not in STANDIN_AUGS:
        raise NotImplementedError(f"aug={cfg.aug!r} arrives with A's build_dvc_loaders; "
                                  f"the stand-in loader implements only {list(STANDIN_AUGS)}")
    flip = [transforms.RandomHorizontalFlip()] if cfg.aug == "flip" else []
    return transforms.Compose([
        transforms.Resize(int(cfg.img_size * 256 / 224)),
        transforms.CenterCrop(cfg.img_size),
        *flip,
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])


def balanced_subset(ds, n, seed=SUBSET_SEED):
    """Subset of an ImageFolder with n/2 images per class (n=None keeps everything). For each class, its
    file paths are sorted, shuffled by a fresh random.Random(seed), and the first n/2 kept."""
    if n is None:
        return ds
    n_cls = len(ds.class_to_idx)
    if n <= 0 or n % n_cls:
        raise ValueError(f"n_train={n} must be a positive multiple of {n_cls} (n/{n_cls} images per class)")
    picked = []
    for cls in sorted(ds.class_to_idx.values()):
        idx = sorted((i for i, (_, y) in enumerate(ds.samples) if y == cls), key=lambda i: ds.samples[i][0])
        if n // n_cls > len(idx):
            raise ValueError(f"n_train={n} needs {n // n_cls} images of class {cls}, only {len(idx)} exist")
        random.Random(seed).shuffle(idx)
        picked += idx[: n // n_cls]
    return Subset(ds, sorted(picked))


def _image_folder(path, transform):
    ds = datasets.ImageFolder(path, transform)
    if ds.class_to_idx != DVC_CLASS_TO_IDX:
        raise ValueError(f"{path}: class_to_idx must be {DVC_CLASS_TO_IDX} (cat=0, dog=1), got {ds.class_to_idx}")
    return ds


def _check_cfg(cfg):
    if cfg.num_classes != len(DVC_CLASS_TO_IDX):
        raise ValueError(f"Dogs-vs-Cats has 2 classes but cfg.num_classes={cfg.num_classes}")
    try:
        fixed = model_input_size(cfg.model)
    except KeyError:  # unknown model: build_model reports it with the list of names
        fixed = None
    if fixed is not None and fixed != cfg.img_size:
        raise ValueError(f"model {cfg.model!r} expects {fixed}px inputs but cfg.img_size={cfg.img_size}")


def build_standin_loaders(cfg, root=None, n_val=None):
    """{"train", "val", "class_to_idx"} for run_experiment; both loaders yield (x [B, 3, S, S], label).
    train: {root}/train (or its cfg.n_train subset), cfg.aug, shuffled via loader_kwargs(cfg.seed).
    val: all of {root}/val, or a balanced n_val subset (smoke tests only)."""
    _check_cfg(cfg)
    if cfg.sampler != "none":  # runs.csv records cfg.sampler, so never accept one that is not applied
        raise ValueError(f"the stand-in loader has no sampler, but cfg.sampler={cfg.sampler!r}")
    base = resolve_root(root)
    train = balanced_subset(_image_folder(base / "train", train_transform(cfg)), cfg.n_train)
    val = balanced_subset(_image_folder(base / "val", eval_transform(cfg.img_size, cfg.norm_mean, cfg.norm_std)),
                          n_val)
    workers = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                   persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(train, batch_size=cfg.bs, shuffle=True, drop_last=len(train) > cfg.bs,
                            **workers, **loader_kwargs(cfg.seed)),
        "val": DataLoader(val, batch_size=cfg.bs, shuffle=False, **workers),
        "class_to_idx": dict(DVC_CLASS_TO_IDX),
    }
