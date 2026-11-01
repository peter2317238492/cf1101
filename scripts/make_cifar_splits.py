"""Write the shared CIFAR-10 45k/5k split (seed 42, 4500/500 per class) to splits/. Idempotent: a rerun
leaves identical files untouched and refuses to change them without --force.

    python -m scripts.make_cifar_splits --root E:/6483proj/data/cifar10
"""
import argparse

from src.cifar.data import SPLIT_FILES, SPLITS_DIR, load_cifar10, write_split_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="CIFAR-10 root (default: $CIFAR10_ROOT, then data/cifar10)")
    ap.add_argument("--out", default=str(SPLITS_DIR))
    ap.add_argument("--force", action="store_true", help="overwrite split files whose content would change")
    args = ap.parse_args()
    _, labels = load_cifar10(args.root, train=True)
    for name, status in write_split_files(labels, args.out, force=args.force).items():
        print(f"{SPLIT_FILES[name]}: {status}")


if __name__ == "__main__":
    main()
