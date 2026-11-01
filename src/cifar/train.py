"""CIFAR-10 training shared by B (Q8) and C (Q9): loss / sampler registries and the run behind
scripts/train_cifar.py.

    python -m scripts.train_cifar --config configs/cifar_baseline.yaml --seeds 0 1 2 \
        --set loss=focal exp_name=cifar_focal_r18_pt224 member=C

cfg.loss and cfg.sampler pick builders registered here by name; B provides "ce" and "none". Put new ones
in any module under src/cifar/ (e.g. src/cifar/imbalance.py; a subfolder needs an __init__.py): it is
imported automatically the first time an unknown name is requested, so no other file needs editing.
cfg.sampler only takes effect through run_cifar (scripts/train_cifar.py); build_cifar_loaders refuses it.

    import numpy as np, torch
    from torch import nn
    from torch.utils.data import WeightedRandomSampler
    from src.cifar.train import register_loss, register_sampler

    @register_loss("weighted_ce")
    def weighted_ce(cfg, train_labels):           # int64 labels of the actual training subset
        counts = np.bincount(train_labels, minlength=cfg.num_classes)
        w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    @register_sampler("weighted")
    def weighted(cfg, train_labels):
        w = 1.0 / np.bincount(train_labels)[train_labels]          # one weight per training sample
        return WeightedRandomSampler(w, num_samples=len(w), generator=torch.Generator().manual_seed(cfg.seed))

Loss builder -> nn.Module built with label_smoothing=cfg.label_smoothing (the trainer checks it), tensors
kept as buffers so .to(device) moves them; or None, meaning the trainer's plain CE (only for loss "ce").
Sampler builder -> torch.utils.data.Sampler over positions 0..len(train_labels)-1, or None for a plain
shuffle. With a sampler the train loader does not shuffle; a sampler whose `generator` is None is given
one seeded from cfg.seed.
"""
import dataclasses
import hashlib
import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn

from src.cifar.data import SPLIT_FILES, build_cifar_datasets, cifar_loaders, dataset_targets, read_split
from src.trainer import TrainConfig, run_experiment, set_seed

_LOSSES = {}
_SAMPLERS = {}


def _same_builder(a, b):
    """True for the same function, or its copy from a re-imported module (e.g. after a failed import)."""
    key = lambda f: (getattr(f, "__module__", None), getattr(f, "__qualname__", None))
    return a is b or (key(a) == key(b) and None not in key(a) and "<locals>" not in key(a)[1])


def _register(table, kind, name):
    def deco(builder):
        if name in table and not _same_builder(table[name], builder):
            raise ValueError(f"{kind} {name!r} is already registered by {table[name].__module__}")
        table[name] = builder
        return builder
    return deco


def register_loss(name):
    """Decorator: `builder(cfg, train_labels: np.ndarray) -> nn.Module | None`, selected by cfg.loss."""
    return _register(_LOSSES, "loss", name)


def register_sampler(name):
    """Decorator: `builder(cfg, train_labels: np.ndarray) -> Sampler | None`, selected by cfg.sampler."""
    return _register(_SAMPLERS, "sampler", name)


def _autodiscover():
    import src.cifar as pkg
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.name != __name__:
            try:
                importlib.import_module(info.name)
            except Exception as e:
                raise ImportError(f"auto-discovery of losses/samplers failed while importing {info.name}: {e}") from e


def _lookup(table, kind, name):
    if name not in table:
        _autodiscover()
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)}")
    return table[name]


def available_losses():
    _autodiscover()
    return sorted(_LOSSES)


def available_samplers():
    _autodiscover()
    return sorted(_SAMPLERS)


def build_loss(cfg, train_labels):
    """The criterion for cfg.loss (None = the trainer's plain CE)."""
    return _lookup(_LOSSES, "loss", cfg.loss)(cfg, np.asarray(train_labels, dtype=np.int64))


def build_sampler(cfg, train_labels):
    """The train sampler for cfg.sampler (None = plain shuffle)."""
    return _lookup(_SAMPLERS, "sampler", cfg.sampler)(cfg, np.asarray(train_labels, dtype=np.int64))


@register_loss("ce")
def _ce(cfg, train_labels):
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


@register_sampler("none")
def _no_sampler(cfg, train_labels):
    return None


# ---------- one run ----------

def load_train_indices(path, splits_dir=None):
    """int64 array from the `index` column of a CSV: positions in the 50k official training images, all
    of which must lie in the shared 45k train split. Duplicates are kept (oversampling)."""
    path = Path(path)
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError(f"{path}: needs an 'index' column (positions in the 50k official training images); "
                         f"found columns {list(df.columns)}")
    if df.empty or not pd.api.types.is_integer_dtype(df["index"]):
        raise ValueError(f"{path}: 'index' must be a non-empty column of integers (found dtype "
                         f"{df['index'].dtype}, {len(df)} rows); blank cells read as NaN")
    idx = df["index"].to_numpy(dtype=np.int64)
    outside = ~np.isin(idx, read_split("train", splits_dir)["index"].to_numpy())
    if outside.any():
        in_val = np.isin(idx[outside], read_split("val", splits_dir)["index"].to_numpy()).sum()
        raise ValueError(f"{path}: {outside.sum()} of {len(idx)} indices are not in the 45k train split "
                         f"{SPLIT_FILES['train']} ({in_val} of them are in the 5k val split), e.g. "
                         f"{idx[outside][:5].tolist()}; sample them from read_split('train')['index']")
    return idx


def indices_note(indices, name=""):
    """e.g. "idx=cifar10_imb_step_r0.1.csv@1a2b3c4d (24750)": which subset a run trained on."""
    indices = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha1(indices.astype("<i8").tobytes()).hexdigest()[:8]
    return f"idx={name}@{digest} ({len(indices)})"


def run_cifar(cfg, root=None, train_indices=None, splits_dir=None, eval_test=False, indices_name=""):
    """One CIFAR-10 run: datasets -> cfg.loss criterion and cfg.sampler sampler (both from the labels of the
    actual training subset) -> run_experiment. eval_test=True scores the best epoch once on the official
    10k test set (final runs only). With train_indices, indices_note(train_indices, indices_name) is appended
    to cfg.notes so runs.csv records the subset. Returns run_experiment's dict."""
    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)
    if train_indices is not None:
        note = indices_note(train_indices, indices_name)
        cfg = dataclasses.replace(cfg, notes=f"{cfg.notes}; {note}" if cfg.notes else note)
    _lookup(_LOSSES, "loss", cfg.loss)  # unknown names fail before any data is loaded
    _lookup(_SAMPLERS, "sampler", cfg.sampler)
    ds = build_cifar_datasets(cfg, root, train_indices, splits_dir)
    labels = dataset_targets(ds["train"])
    set_seed(cfg.seed, cfg.deterministic)  # a loss that draws random numbers is reproducible too
    criterion = build_loss(cfg, labels)
    # run_experiment seeds, then calls this, so the sampler is built under the run's seed
    return run_experiment(cfg, lambda c: cifar_loaders(c, ds, sampler=build_sampler(c, labels)),
                          criterion=criterion, eval_test=eval_test)
