"""End-to-end smoke test of the trainer on real Dogs-vs-Cats images (GPU, AMP, worker processes).

Uses the stand-in loader in src/dvc_standin.py (ImageFolder + class-balanced subsets) until A's
build_dvc_loaders lands. Data root: $DVC_ROOT, else data/dogs-vs-cats/datasets under the project.

    python -m scripts.smoke_dvc --n-train 1000 --n-val 500 --epochs 2
"""
import argparse

from src.dvc_standin import STANDIN_AUGS, build_standin_loaders
from src.trainer import TrainConfig, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--freeze", default="backbone")
    ap.add_argument("--aug", default="flip", choices=STANDIN_AUGS)
    args = ap.parse_args()
    cfg = TrainConfig(exp_name=f"smoke_dvc_r18_{args.freeze}", member="B", task="dvc", model="resnet18",
                      pretrained=True, freeze=args.freeze, aug=args.aug, img_size=224, n_train=args.n_train,
                      optimizer="sgd", lr=0.01, wd=1e-4, bs=64, epochs=args.epochs, scheduler="cosine",
                      warmup_epochs=0.5, num_workers=4, notes="smoke test, stand-in loader",
                      out_dir="runs/smoke", results_csv="runs/smoke/runs.csv", overwrite=True)
    res = run_experiment(cfg, lambda c: build_standin_loaders(c, n_val=args.n_val))
    print(f"best val acc {res['best_val_acc']:.4f} at epoch {res['best_epoch']} -> {res['ckpt_path']}")


if __name__ == "__main__":
    main()
