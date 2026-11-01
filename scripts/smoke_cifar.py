"""End-to-end smoke test of the CIFAR-10 loaders + trainer on real data (GPU, AMP, worker processes).

    python -m scripts.smoke_cifar --root E:/6483proj/data/cifar10 --epochs 1

One epoch on the full 45k reaches ~27% val on a laptop RTX 3070 Ti in ~1.5 min; --n-train 5000 is quicker
but stays near chance (only 39 SGD steps).
"""
import argparse
from functools import partial

from src.cifar.data import CIFAR10_MEAN, CIFAR10_STD, build_cifar_loaders
from src.trainer import TrainConfig, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="CIFAR-10 root (default: $CIFAR10_ROOT, then data/cifar10)")
    ap.add_argument("--n-train", type=int, default=0, help="0 = the full 45k train split")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    cfg = TrainConfig(exp_name="smoke_cifar_r18", member="B", task="cifar10", model="cifar_resnet18", num_classes=10,
                      pretrained=False, aug="crop_flip", img_size=32, n_train=args.n_train or None,
                      norm_mean=CIFAR10_MEAN, norm_std=CIFAR10_STD, optimizer="sgd", lr=0.1, wd=5e-4, bs=128,
                      epochs=args.epochs, scheduler="cosine", num_workers=4, notes="smoke test, not for reporting",
                      out_dir="runs/smoke", results_csv="runs/smoke/runs.csv", overwrite=True)
    res = run_experiment(cfg, partial(build_cifar_loaders, root=args.root), eval_test=True)
    print(f"best val acc {res['best_val_acc']:.4f} at epoch {res['best_epoch']}, "
          f"test acc {res['test_metrics']['acc']:.4f} -> {res['ckpt_path']}")


if __name__ == "__main__":
    main()
