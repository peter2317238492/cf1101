"""Trainer core shared by A (scratch CNN), B (pretrained / CIFAR) and C (CIFAR imbalance).

Entry point (one run = one config + one seed):

    from src.trainer import TrainConfig, run_experiment
    cfg = TrainConfig.from_yaml("configs/xxx.yaml", seed=1)
    result = run_experiment(cfg, build_loaders)

`build_loaders(cfg)` returns {"train": DL, "val": DL} plus optionally "test" (labelled, used only
with eval_test=True) and "class_to_idx". Train/val/test loaders yield (x, label); loaders passed
to `predict` yield (x, id). `run_experiment` seeds everything, builds loaders and the model (by
registry name), trains with `fit`, and leaves the best-epoch weights in `result["model"]`.
Every run writes runs/{exp_name}/seed{seed}/{config.yaml, history.csv, best.pt, metrics.json}
and appends one row to results/runs.csv. A finished run is never overwritten unless
cfg.overwrite=True.

Losses: pass `criterion=` for anything other than plain CE (weighted CE, focal, ...), set
cfg.loss to its name, and build it with label_smoothing=cfg.label_smoothing when that is
non-zero. Keep tensors inside a criterion as registered buffers (or nn.CrossEntropyLoss(weight=...))
so `.to(device)` moves them. Validation/test loss is always plain cross-entropy.

Windows note: scripts that use num_workers > 0 must be guarded by `if __name__ == "__main__":`.
"""
import csv
import json
import math
import os
import random
import subprocess
import time
import types
import typing
import warnings
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.models.registry import build_model, save_ckpt, split_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts", "configs", "tests", "splits")  # changes here mark a run's git commit as -dirty
DVC_CLASS_TO_IDX = {"cat": 0, "dog": 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Columns are only ever appended; older runs.csv files are migrated automatically.
RUNS_CSV_COLUMNS = [
    "run_id", "member", "task", "model", "pretrained", "freeze", "aug", "img_size", "n_train", "n_val",
    "optimizer", "lr", "wd", "bs", "epochs", "scheduler", "seed", "best_val_acc", "best_epoch", "val_loss",
    "test_acc", "macro_f1", "train_time_s", "ckpt_path", "git_commit", "notes",
    "exp_name", "loss", "sampler", "imb_ratio", "lr_backbone", "label_smoothing", "test_macro_f1",
]


@dataclass
class TrainConfig:
    # bookkeeping
    exp_name: str = "debug"
    task: str = "dvc"                  # dvc | cifar10 | cifar10_imb
    member: str = ""
    notes: str = ""
    overwrite: bool = False            # allow re-running a finished exp_name/seed
    # model
    model: str = "resnet18"            # name in src.models.registry
    num_classes: int = 2
    pretrained: bool = False
    freeze: str = "none"               # none | backbone | partial
    # data (consumed by the loader builder; recorded here so every run is self-describing)
    aug: str = "flip"
    img_size: int = 224
    n_train: int | None = None         # None = use the full training split
    norm_mean: tuple = IMAGENET_MEAN
    norm_std: tuple = IMAGENET_STD
    sampler: str = "none"              # applied by src.cifar.train.run_cifar; other loaders refuse non-"none"
    imb_ratio: float | None = None     # recorded only (CIFAR imbalance)
    # loss
    loss: str = "ce"                   # anything but "ce" must be passed as criterion=
    label_smoothing: float = 0.0
    # optimisation
    optimizer: str = "sgd"             # sgd | adamw
    lr: float = 0.01                   # head lr (and backbone lr unless lr_backbone is set)
    lr_backbone: float | None = None
    momentum: float = 0.9
    wd: float = 5e-4                   # not applied to biases / norm layers
    bs: int = 64
    epochs: int = 10
    scheduler: str = "cosine"          # cosine | step | onecycle | none (all stepped per batch)
    warmup_epochs: float = 0.0
    min_lr_ratio: float = 0.0          # cosine floor as a fraction of the initial lr
    step_epochs: int = 30              # step scheduler: decay every step_epochs
    gamma: float = 0.1
    grad_clip: float | None = None
    patience: int | None = None        # early stopping on val acc; None = off
    # runtime
    amp: bool = True
    channels_last: bool = False        # CUDA only; with cuDNN 9.10 trainable BatchNorm is 5-10x slower in it
    seed: int = 0
    deterministic: bool = False
    num_workers: int = 4
    device: str = "auto"               # auto | cuda | cpu
    log_every: int = 0                 # print every N batches; 0 = once per epoch
    out_dir: str = "runs"
    results_csv: str = "results/runs.csv"

    def __post_init__(self):
        # YAML reads 1e-3 as a string, sweeps pass numpy scalars, paths arrive as Path: normalise all.
        for f in fields(self):
            setattr(self, f.name, _coerce(f.name, getattr(self, f.name), f.type))
        if not self.exp_name.strip():
            raise ValueError("TrainConfig.exp_name must not be empty (it names the run directory)")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown TrainConfig field(s): {unknown}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path, **overrides):
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        d.update(overrides)
        return cls.from_dict(d)

    def to_dict(self):
        d = asdict(self)
        d["norm_mean"], d["norm_std"] = list(self.norm_mean), list(self.norm_std)
        return d

    @classmethod
    def parse_overrides(cls, pairs):
        """CLI `--set` pairs -> dict: ["lr=1e-3", "n_train=null", "notes=a: b"] -> {"lr": "1e-3", "n_train": None,
        "notes": "a: b"}. Splits on the first "="; later duplicates win. str fields keep the raw text;
        other values go through yaml.safe_load (empty = null). Field names are checked by from_dict."""
        str_fields = {f.name for f in fields(cls) if f.type is str}
        out = {}
        for pair in pairs or ():
            key, sep, raw = pair.partition("=")
            key, raw = key.strip(), raw.strip()
            if not sep or not key:
                raise ValueError(f"--set expects key=value, got {pair!r}")
            if key in str_fields:
                out[key] = raw
                continue
            if ":" in raw and not raw.startswith(("[", "{")):
                out[key] = raw  # YAML 1.1 reads 1:10 as the base-60 number 70; let TrainConfig reject it
                continue
            try:
                out[key] = yaml.safe_load(raw) if raw else None
            except yaml.YAMLError as e:
                raise ValueError(f"--set {pair!r}: not valid YAML ({e}); quote values that contain spaces, "
                                 f'e.g. "norm_mean=[0.5, 0.5, 0.5]"') from None
        return out


def _coerce(name, value, annotation):
    optional = typing.get_origin(annotation) in (types.UnionType, typing.Union)
    base = next(a for a in typing.get_args(annotation) if a is not type(None)) if optional else annotation
    if optional and (value is None or (isinstance(value, str) and value.lower() in ("none", "null"))):
        return None
    try:
        if base is bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        elif base is int and not isinstance(value, (bool, np.bool_)):
            f = float(value)
            if f.is_integer():
                return int(f)
        elif base is float and not isinstance(value, (bool, np.bool_)):
            return float(value)
        elif base is str and value is not None:
            return str(value)
        elif base is tuple:
            return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        pass
    raise ValueError(f"TrainConfig.{name}: cannot interpret {value!r} as {getattr(base, '__name__', base)}")


# ---------- seeding ----------

def set_seed(seed, deterministic=False):
    """Seed python / numpy / torch (CPU + CUDA). deterministic=True trades speed for repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def loader_kwargs(seed):
    """Pass to DataLoader(...) so shuffling and worker-side augmentation depend only on the seed."""
    return {"worker_init_fn": _seed_worker, "generator": torch.Generator().manual_seed(seed)}


# ---------- optimizer / scheduler ----------

def build_optimizer(model, cfg):
    """Trainable params only; no weight decay on 1-D params (biases, norm weights).
    With cfg.lr_backbone set, non-head params use that lr and the head uses cfg.lr."""
    if cfg.lr_backbone is None:
        parts = [("all", [p for p in model.parameters()], cfg.lr)]
    else:
        backbone, head = split_parameters(model)
        parts = [("backbone", backbone, cfg.lr_backbone), ("head", head, cfg.lr)]
    groups = []
    for name, params, lr in parts:
        params = [p for p in params if p.requires_grad]
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        for suffix, ps, wd in (("", decay, cfg.wd), ("_no_wd", no_decay, 0.0)):
            if ps:
                groups.append({"params": ps, "lr": lr, "weight_decay": wd, "name": name + suffix})
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.momentum > 0)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Per-batch LR schedule. Call scheduler.step() after every optimizer step."""
    total = max(1, cfg.epochs * steps_per_epoch)
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=total)
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if cfg.scheduler == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if cfg.scheduler == "step":
            return cfg.gamma ** ((step // steps_per_epoch) // cfg.step_epochs)
        if cfg.scheduler == "none":
            return 1.0
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

    factor(warmup)  # fail fast on an unknown scheduler name
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------- train / eval loops ----------

def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_device(model):
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _to_device(x, device, channels_last):
    x = x.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _set_train_mode(model):
    """model.train(), but BatchNorm layers whose params are all frozen stay in eval mode,
    so a frozen backbone keeps its pretrained running statistics."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg, channels_last=False):
    _set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(loader, 1):
        x = _to_device(x, device, channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            if i == 1 and logits.shape[-1] != cfg.num_classes:
                raise ValueError(f"model outputs {logits.shape[-1]} logits but cfg.num_classes={cfg.num_classes}")
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        n = y.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == y).sum().item()
        seen += n
        if cfg.log_every and i % cfg.log_every == 0:
            print(f"  step {i}/{len(loader)} loss {total_loss / seen:.4f} acc {correct / seen:.4f}", flush=True)
    return total_loss / max(1, seen), correct / max(1, seen)


@torch.no_grad()
def _forward_all(model, loader, device, amp, channels_last):
    """Returns (logits [N, K] float32 on CPU, list of the second batch element per sample)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    outs, tails = [], []
    for x, tail in loader:
        x = _to_device(x, device, channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        outs.append(logits.float().cpu())
        tails.extend(tail.tolist() if torch.is_tensor(tail) else list(tail))
    return torch.cat(outs), tails


def _macro_f1(confusion):
    tp = np.diag(confusion).astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    present = (support + predicted) > 0  # same label set as sklearn's default
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean()) if present.any() else 0.0


def evaluate(model, loader, device=None, amp=True, channels_last=False):
    """Loader yields (x, y). Returns acc, loss (plain CE), per_class_acc (recall), confusion
    (rows = true, cols = predicted), macro_f1, probs [N, K], preds [N], labels [N]."""
    device = device or _model_device(model)
    logits, tails = _forward_all(model, loader, device, amp, channels_last)
    labels = torch.as_tensor(tails, dtype=torch.long)
    k = logits.shape[1]
    preds = logits.argmax(1)
    confusion = torch.bincount(labels * k + preds, minlength=k * k).reshape(k, k).numpy()
    support = confusion.sum(1)
    per_class = np.divide(np.diag(confusion), support, out=np.zeros(k), where=support > 0)
    return {
        "acc": (preds == labels).sum().item() / max(1, len(labels)),
        "loss": F.cross_entropy(logits, labels).item(),
        "per_class_acc": per_class,
        "confusion": confusion,
        "macro_f1": _macro_f1(confusion),
        "probs": logits.softmax(1).numpy(),
        "preds": preds.numpy(),
        "labels": labels.numpy(),
    }


def predict(model, loader, device=None, amp=True, channels_last=False):
    """Loader yields (x, id); digit-string ids become ints. Returns
    DataFrame[id, prob_0..prob_{K-1}, pred] in loader order."""
    device = device or _model_device(model)
    logits, ids = _forward_all(model, loader, device, amp, channels_last)
    probs = logits.softmax(1).numpy()
    df = pd.DataFrame({"id": [int(i) if isinstance(i, str) and i.isdigit() else i for i in ids]})
    for j in range(probs.shape[1]):
        df[f"prob_{j}"] = probs[:, j]
    df["pred"] = probs.argmax(1)
    return df


# ---------- run bookkeeping ----------

def _git_commit(root=PROJECT_ROOT):
    """Short SHA of HEAD, with -dirty if CODE_DIRS hold uncommitted edits or new untracked files."""
    git = ["git", "-c", "safe.directory=*", "-C", str(root)]
    try:
        head = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        sha = head.stdout.strip()
        if head.returncode != 0 or not sha:
            return "no-commit"
        dirty = subprocess.run(git + ["status", "--porcelain", "--", *CODE_DIRS],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "no-commit"


def _robust_write(write_fn, what, retries=5, delay=2.0):
    """Run write_fn, retrying PermissionError (Windows: file open in Excel). Warns and returns False on failure."""
    err = None
    for attempt in range(retries):
        try:
            write_fn()
            return True
        except PermissionError as e:
            err = e
            if attempt < retries - 1:
                time.sleep(delay)
    warnings.warn(f"could not write {what} ({err}); close it in Excel or open a copy instead", stacklevel=2)
    return False


def _dataset_attr(dataset, name):
    while dataset is not None:
        if hasattr(dataset, name):
            return getattr(dataset, name)
        dataset = getattr(dataset, "dataset", None)  # unwrap Subset
    return None


def _num_samples(loader):
    indices = getattr(getattr(loader, "sampler", None), "indices", None)  # SubsetRandomSampler
    if indices is not None:
        return len(indices)
    try:
        return len(loader.dataset)
    except (TypeError, AttributeError):
        return ""


def _describe_criterion(criterion):
    desc = {"type": type(criterion).__name__}
    for key in ("label_smoothing", "gamma", "alpha", "weight"):
        v = getattr(criterion, key, None)
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        if isinstance(v, (bool, int, float, str, list)):
            desc[key] = v
    return desc


def _resolve_class_to_idx(explicit, train_loader, task):
    found = explicit if explicit is not None else _dataset_attr(getattr(train_loader, "dataset", None),
                                                                "class_to_idx")
    found = None if found is None else {str(k): int(v) for k, v in found.items()}
    if task == "dvc":
        if found is None:
            warnings.warn("task='dvc' but class_to_idx is unknown; return it from build_loaders "
                          "(key 'class_to_idx') or expose dataset.class_to_idx", stacklevel=3)
        elif found != DVC_CLASS_TO_IDX:
            raise ValueError(f"task='dvc' requires class_to_idx == {DVC_CLASS_TO_IDX} (cat=0, dog=1), got {found}")
    return found


def _check_runs_csv(path):
    """Fail before training if an existing runs.csv has columns this trainer does not know."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    unknown = [c for c in header if c not in RUNS_CSV_COLUMNS]
    if unknown:
        raise ValueError(f"{path} has columns {unknown} that are not in RUNS_CSV_COLUMNS; "
                         f"point results_csv elsewhere or fix the file")


def _as_int(text):
    """'0' / '0.0' (a pandas re-save) -> 0; blank or junk -> None."""
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def check_new_run(cfg):
    """Raise before any training if cfg.exp_name/seed already finished: in this checkout (run dir) or in the
    shared results_csv, which also holds teammates' rows. cfg.overwrite=True skips the check."""
    _check_runs_csv(cfg.results_csv)
    if cfg.overwrite:
        return
    run_dir = Path(cfg.out_dir) / cfg.exp_name / f"seed{cfg.seed}"
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"{run_dir} already holds a finished run; change exp_name/seed or set overwrite=True")
    path = Path(cfg.results_csv)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as f:
            done = [r.get("run_id", "") for r in csv.DictReader(f)
                    if (r.get("exp_name") or "").strip() == cfg.exp_name and _as_int(r.get("seed")) == cfg.seed]
        if done:
            raise FileExistsError(f"{path} already has exp_name={cfg.exp_name!r} seed={cfg.seed} (run {done[-1]}); "
                                  f"change exp_name/seed or set overwrite=True")


def _append_runs_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {c: row.get(c, "") for c in RUNS_CSV_COLUMNS}
    header = None
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), None)

    if header == RUNS_CSV_COLUMNS:
        def write():
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS).writerow(clean)
    else:  # new file, or an older header: rewrite with the current columns, keeping old values by name
        old_rows = []
        if header:
            with path.open(newline="", encoding="utf-8-sig") as f:
                old_rows = list(csv.DictReader(f))

        def write():
            tmp = path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows({c: r.get(c, "") for c in RUNS_CSV_COLUMNS} for r in old_rows)
                writer.writerow(clean)
            os.replace(tmp, path)

    if not _robust_write(write, str(path), retries=10, delay=3.0):
        pending = path.with_name(path.stem + ".pending.csv")
        with pending.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(clean)
        warnings.warn(f"run row saved to {pending}; append it to {path} later", stacklevel=2)


# ---------- fit ----------

def fit(model, train_loader, val_loader, cfg, criterion=None, test_loader=None, class_to_idx=None):
    """Train with per-epoch validation; keep the best val-acc epoch (ties broken by lower val loss).
    On return the model holds the best weights. If test_loader (labelled) is given, the best model
    is evaluated on it once and test metrics are recorded. Returns best_val_acc, best_epoch,
    ckpt_path, history, val_metrics, test_metrics (or None), run_dir, run_id.
    Prefer run_experiment(); if you call fit() directly, call set_seed(cfg.seed) before building
    the model and loaders and build the model with build_model()."""
    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)

    # ---- validate everything before touching the disk ----
    name = getattr(model, "model_name", None)
    if name is None:
        raise ValueError("model was not created by build_model(); register it in src/models and build it "
                         "with build_model() so its checkpoint can be reloaded")
    if name != cfg.model:
        raise ValueError(f"cfg.model={cfg.model!r} but the model passed in is {name!r}")
    if criterion is None:
        if cfg.loss != "ce":
            raise ValueError(f"cfg.loss={cfg.loss!r} needs criterion=...; only 'ce' is built in")
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    elif cfg.loss == "ce" and not (type(criterion) is nn.CrossEntropyLoss and criterion.weight is None):
        raise ValueError(f"cfg.loss='ce' but the criterion passed in is {type(criterion).__name__} "
                         f"{_describe_criterion(criterion)}; set cfg.loss to its name, runs.csv records cfg.loss")
    elif getattr(criterion, "label_smoothing", 0.0) != cfg.label_smoothing:
        raise ValueError(f"criterion has label_smoothing={getattr(criterion, 'label_smoothing', 0.0)} but "
                         f"cfg.label_smoothing={cfg.label_smoothing}; build the criterion with "
                         f"label_smoothing=cfg.label_smoothing")
    class_to_idx = _resolve_class_to_idx(class_to_idx, train_loader, cfg.task)
    check_new_run(cfg)
    run_dir = Path(cfg.out_dir) / cfg.exp_name / f"seed{cfg.seed}"
    if torch.initial_seed() != cfg.seed:
        warnings.warn(f"torch is not seeded with cfg.seed={cfg.seed}; call set_seed(cfg.seed) before building "
                      f"the model and loaders (or use run_experiment), otherwise the run is not reproducible",
                      stacklevel=2)

    device = _resolve_device(cfg.device)
    channels_last = cfg.channels_last and device.type == "cuda"
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    if isinstance(criterion, nn.Module):
        criterion.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    report_group = next((g for g in optimizer.param_groups if g["name"].startswith("head")),
                        optimizer.param_groups[0])

    run_id = f"{cfg.exp_name}_seed{cfg.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    _robust_write(lambda: (run_dir / "config.yaml").write_text(config_text, encoding="utf-8"),
                  run_dir / "config.yaml", retries=2, delay=1.0)
    ckpt_path = run_dir / "best.pt"
    criterion_desc = _describe_criterion(criterion)
    meta = {
        "cfg": cfg.to_dict(), "class_to_idx": class_to_idx, "img_size": cfg.img_size,
        "norm_mean": list(cfg.norm_mean), "norm_std": list(cfg.norm_std), "seed": cfg.seed,
        "git_commit": _git_commit(), "run_id": run_id, "criterion": criterion_desc,
    }

    history, best, best_state, bad_epochs = [], None, None, 0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        lr_now = report_group["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                                                device, cfg, channels_last)
        val = evaluate(model, val_loader, device=device, amp=cfg.amp, channels_last=channels_last)
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val["loss"],
               "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "lr": lr_now,
               "time_s": round(time.time() - t0, 2)}
        history.append(row)
        hist_df = pd.DataFrame(history)
        _robust_write(lambda: hist_df.to_csv(run_dir / "history.csv", index=False), run_dir / "history.csv",
                      retries=2, delay=1.0)
        print(f"[{cfg.exp_name} seed{cfg.seed}] epoch {epoch}/{cfg.epochs} "
              f"train {train_loss:.4f}/{train_acc:.4f} val {val['loss']:.4f}/{val['acc']:.4f} "
              f"lr {lr_now:.2e} {row['time_s']}s", flush=True)

        improved = best is None or val["acc"] > best["acc"] or (
            val["acc"] == best["acc"] and val["loss"] < best["loss"])
        if improved:
            best, best_epoch, bad_epochs = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_ckpt(ckpt_path, model, model_name=cfg.model, num_classes=cfg.num_classes,
                      val_acc=val["acc"], epoch=epoch, **meta)
        else:
            bad_epochs += 1
            if cfg.patience and bad_epochs >= cfg.patience:
                print(f"early stop: no val improvement for {bad_epochs} epochs", flush=True)
                break
    train_time = round(time.time() - start, 1)
    model.load_state_dict(best_state)

    test = None
    if test_loader is not None:
        test = evaluate(model, test_loader, device=device, amp=cfg.amp, channels_last=channels_last)
        print(f"[{cfg.exp_name} seed{cfg.seed}] test acc {test['acc']:.4f} macro-F1 {test['macro_f1']:.4f}",
              flush=True)

    n_train, n_val = _num_samples(train_loader), _num_samples(val_loader)
    metrics = {
        "run_id": run_id, "best_val_acc": best["acc"], "best_epoch": best_epoch, "val_loss": best["loss"],
        "macro_f1": best["macro_f1"], "per_class_acc": best["per_class_acc"].tolist(),
        "confusion": best["confusion"].tolist(), "epochs_ran": len(history), "train_time_s": train_time,
        "n_train": n_train, "n_val": n_val, "criterion": criterion_desc, "class_to_idx": class_to_idx,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch": torch.__version__, "git_commit": meta["git_commit"],
    }
    if test is not None:
        metrics.update({"test_acc": test["acc"], "test_loss": test["loss"], "test_macro_f1": test["macro_f1"],
                        "test_per_class_acc": test["per_class_acc"].tolist(),
                        "test_confusion": test["confusion"].tolist()})
    metrics_text = json.dumps(metrics, indent=1)
    _robust_write(lambda: (run_dir / "metrics.json").write_text(metrics_text, encoding="utf-8"),
                  run_dir / "metrics.json", retries=10, delay=3.0)
    _append_runs_csv(cfg.results_csv, {
        **cfg.to_dict(), "run_id": run_id, "n_train": n_train, "n_val": n_val, "best_val_acc": best["acc"],
        "best_epoch": best_epoch, "val_loss": best["loss"], "macro_f1": best["macro_f1"],
        "train_time_s": train_time, "ckpt_path": ckpt_path.as_posix(), "git_commit": meta["git_commit"],
        "test_acc": test["acc"] if test else "", "test_macro_f1": test["macro_f1"] if test else "",
    })
    return {"best_val_acc": best["acc"], "best_epoch": best_epoch, "ckpt_path": str(ckpt_path),
            "history": history, "val_metrics": best, "test_metrics": test, "run_dir": str(run_dir),
            "run_id": run_id}


def run_experiment(cfg, build_loaders, criterion=None, eval_test=False):
    """Seed -> build_loaders(cfg) -> build_model(cfg.model ...) -> fit. Returns fit()'s dict plus "model".
    eval_test=True evaluates the best model once on loaders["test"] (must be labelled, e.g. CIFAR test)."""
    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)
    if eval_test and cfg.task == "dvc":
        raise ValueError("the Dogs-vs-Cats test set has no labels; use predict() on it instead of eval_test=True")
    set_seed(cfg.seed, cfg.deterministic)
    loaders = build_loaders(cfg)
    if eval_test and "test" not in loaders:
        raise KeyError("eval_test=True needs build_loaders to return a labelled 'test' loader")
    model = build_model(cfg.model, cfg.num_classes, pretrained=cfg.pretrained, freeze=cfg.freeze)
    result = fit(model, loaders["train"], loaders["val"], cfg, criterion=criterion,
                 test_loader=loaders["test"] if eval_test else None, class_to_idx=loaders.get("class_to_idx"))
    result["model"] = model
    return result
