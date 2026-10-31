"""Shared test fixtures: a tiny registered CNN and linearly separable "blob" images."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.registry import register
from src.trainer import TrainConfig, loader_kwargs


class TinyNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.pool(F.relu(self.bn(self.conv(x)))).flatten(1))


@register("tiny_test_net", head=("fc",), partial=("bn",))
def build_tiny(num_classes, pretrained=False):
    return TinyNet(num_classes)


def make_blobs(n, num_classes, seed, size=8):
    """Images whose mean colour depends on the class, so a 1-conv net separates them in a few epochs."""
    centers = torch.randn(num_classes, 3, generator=torch.Generator().manual_seed(123)) * 2
    y = torch.arange(n) % num_classes
    noise = torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(seed))
    return centers[y][:, :, None, None] + 0.5 * noise, y


def blob_loaders(cfg, n_train=128, n_val=64):
    xtr, ytr = make_blobs(n_train, cfg.num_classes, seed=1)
    xva, yva = make_blobs(n_val, cfg.num_classes, seed=2)
    return {
        "train": DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.bs, shuffle=True, **loader_kwargs(cfg.seed)),
        "val": DataLoader(TensorDataset(xva, yva), batch_size=cfg.bs, shuffle=False),
    }


class FlipAfter(nn.Module):
    """Cross-entropy that trains on flipped binary labels after `n` calls, so later epochs get worse on purpose."""

    def __init__(self, n):
        super().__init__()
        self.n, self.calls = n, 0

    def forward(self, logits, y):
        self.calls += 1
        if self.calls > self.n:
            y = 1 - y
        return F.cross_entropy(logits, y)


@pytest.fixture
def make_cfg(tmp_path):
    def _make(**overrides):
        base = dict(
            exp_name="t", task="unit", model="tiny_test_net", num_classes=2, pretrained=False,
            optimizer="sgd", lr=0.1, wd=0.0, bs=16, epochs=3, scheduler="none",
            amp=False, channels_last=False, seed=0, deterministic=True, num_workers=0,
            device="cpu", out_dir=str(tmp_path / "runs"), results_csv=str(tmp_path / "results" / "runs.csv"),
        )
        base.update(overrides)
        return TrainConfig(**base)

    return _make
