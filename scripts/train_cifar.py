"""Train on CIFAR-10 from a YAML config: B's balanced baseline and improvements (Q8), C's imbalance runs (Q9).

    python -m scripts.train_cifar --config configs/cifar_baseline.yaml [--seeds 0 1 2] [--set key=value ...]
                                  [--train-indices <csv>] [--root <cifar dir>] [--eval-test]

--seeds          seeds to run one after another (default: the config's seed)
--set            override TrainConfig fields: lr=1e-3 n_train=null loss=focal "norm_mean=[0.5, 0.5, 0.5]";
                 text fields (exp_name, notes, loss, ...) are kept as typed, other values are parsed as YAML
--train-indices  CSV with an `index` column: the training subset as positions in the 50k official training
                 images; every index must be in splits/cifar10_train45k.csv. Its file name and a hash of the
                 indices are appended to cfg.notes, so runs.csv says which subset a run used
--root           CIFAR-10 root (default: $CIFAR10_ROOT, then data/cifar10)
--eval-test      score the best epoch once on the official 10k test set; final runs only

cfg.loss / cfg.sampler select builders from the registries in src/cifar/train.py.
"""
import argparse
import statistics
from pathlib import Path

import torch
import yaml

from src.cifar.train import load_train_indices, run_cifar
from src.trainer import TrainConfig, check_new_run


def parse_overrides(pairs):
    """`--set` pairs -> dict of TrainConfig overrides; see TrainConfig.parse_overrides."""
    return TrainConfig.parse_overrides(pairs)


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m scripts.train_cifar", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="YAML file of TrainConfig fields")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the config's seed")
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="override TrainConfig fields; text fields kept as typed, others parsed as YAML (empty = null)")
    ap.add_argument("--train-indices", default=None, metavar="CSV",
                    help="CSV with an 'index' column; subset of splits/cifar10_train45k.csv")
    ap.add_argument("--root", default=None, help="CIFAR-10 root (default: $CIFAR10_ROOT, then data/cifar10)")
    ap.add_argument("--eval-test", action="store_true", help="evaluate the best epoch on the 10k test set once")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    overrides = parse_overrides(args.set)
    if args.seeds and "seed" in overrides:
        raise ValueError("give the seed(s) either with --seeds or with --set seed=..., not both")
    base.update(overrides)
    seeds = args.seeds or [TrainConfig.from_dict(base).seed]  # unknown fields / bad values fail here
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate --seeds {seeds}")
    cfgs = [TrainConfig.from_dict({**base, "seed": s}) for s in seeds]
    for cfg in cfgs:  # the trainer refuses these too, but only after the earlier seeds have trained
        check_new_run(cfg)
    train_indices = load_train_indices(args.train_indices) if args.train_indices else None
    indices_name = Path(args.train_indices).name if args.train_indices else ""

    results = []
    for cfg in cfgs:
        res = run_cifar(cfg, root=args.root, train_indices=train_indices, eval_test=args.eval_test,
                        indices_name=indices_name)
        del res["model"]  # free GPU memory before the next seed
        torch.cuda.empty_cache()
        test = f", test acc {res['test_metrics']['acc']:.4f}" if res["test_metrics"] else ""
        print(f"[{cfg.exp_name} seed{cfg.seed}] best val acc {res['best_val_acc']:.4f} at epoch "
              f"{res['best_epoch']}{test} -> {res['ckpt_path']}", flush=True)
        results.append(res)
    if len(results) > 1:
        accs = [r["best_val_acc"] for r in results]
        print(f"[{cfgs[0].exp_name}] best val acc over seeds {seeds}: "
              f"{statistics.mean(accs):.4f} +/- {statistics.stdev(accs):.4f} (sample std)", flush=True)
    return results


if __name__ == "__main__":
    main()
