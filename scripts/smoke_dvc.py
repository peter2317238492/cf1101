"""End-to-end smoke test of the trainer on real Dogs-vs-Cats images (GPU, AMP, worker processes).

Uses a stand-in loader (ImageFolder + class-balanced subset) until A's build_dvc_loaders lands.

    python -m scripts.smoke_dvc --n-train 1000 --n-val 500 --epochs 2
"""
import argparse
import random
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.trainer import TrainConfig, loader_kwargs, run_experiment

DATA_ROOT = Path("data/dogs-vs-cats/datasets")


def balanced_subset(ds, n, seed=42):
    """First n/2 of each class after a seeded shuffle of the sorted file list."""
    if not n:
        return ds
    rng = random.Random(seed)
    picked = []
    for cls in sorted(ds.class_to_idx.values()):
        idx = [i for i, (_, y) in enumerate(ds.samples) if y == cls]
        rng.shuffle(idx)
        picked += idx[: n // 2]
    return Subset(ds, sorted(picked))


def build_loaders(cfg, n_val):
    norm = transforms.Normalize(cfg.norm_mean, cfg.norm_std)
    train_tf = transforms.Compose([transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize(int(cfg.img_size * 256 / 224)),
                                  transforms.CenterCrop(cfg.img_size), transforms.ToTensor(), norm])
    train = datasets.ImageFolder(DATA_ROOT / "train", train_tf)
    val = datasets.ImageFolder(DATA_ROOT / "val", eval_tf)
    assert train.class_to_idx == {"cat": 0, "dog": 1}, train.class_to_idx
    common = dict(batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True,
                  persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(balanced_subset(train, cfg.n_train), shuffle=True, drop_last=True,
                            **common, **loader_kwargs(cfg.seed)),
        "val": DataLoader(balanced_subset(val, n_val), shuffle=False, **common),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--freeze", default="backbone")
    args = ap.parse_args()
    cfg = TrainConfig(exp_name=f"smoke_dvc_r18_{args.freeze}", member="B", task="dvc", model="resnet18",
                      pretrained=True, freeze=args.freeze, aug="rrc_flip", img_size=224, n_train=args.n_train,
                      optimizer="sgd", lr=0.01, wd=1e-4, bs=64, epochs=args.epochs, scheduler="cosine",
                      warmup_epochs=0.5, num_workers=4, notes="smoke test, stand-in loader",
                      out_dir="runs/smoke", results_csv="runs/smoke/runs.csv", overwrite=True)
    res = run_experiment(cfg, lambda c: build_loaders(c, args.n_val))
    print(f"best val acc {res['best_val_acc']:.4f} at epoch {res['best_epoch']} -> {res['ckpt_path']}")


if __name__ == "__main__":
    main()
