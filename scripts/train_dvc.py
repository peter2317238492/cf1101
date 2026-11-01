"""Train a Dogs-vs-Cats model with the shared trainer: one YAML config, one run per seed.

    python -m scripts.train_dvc --config configs/dvc_r18_linear_probe.yaml
    python -m scripts.train_dvc --config configs/dvc_r18_linear_probe.yaml --seeds 0 1 2
    python -m scripts.train_dvc --config ... --set lr=3e-3 epochs=10 n_train=2000 exp_name=dvc_lp_n2000
    python -m scripts.train_dvc --config ... --data-root E:/6483proj/data/dogs-vs-cats/datasets

STAND-IN LOADER: the data comes from src.dvc_standin (ImageFolder on train/ and val/, contract
preprocessing, aug none/flip only, n_train = n/2 per class from the sorted file list with
random.Random(42)) until A's build_dvc_loaders lands. Then replace `build_loaders` below with A's
function and drop STANDIN_NOTE; runs made before the switch say "stand-in loader" in runs.csv notes.

--set KEY=VALUE ... overrides TrainConfig fields after the YAML is read: text fields (exp_name, notes, ...)
are kept as typed, other values are parsed with yaml.safe_load, so 1e-3, null, true and
"norm_mean=[0.5, 0.5, 0.5]" (quoted: it contains spaces) work; unknown keys are an error.
--seeds runs the config once per seed (default: the config's own seed); don't combine it with --set seed=.
Finished runs (this checkout or results/runs.csv) are refused before anything trains, unless overwrite=true.
--data-root defaults to $DVC_ROOT, then data/dogs-vs-cats/datasets under the project.
Outputs: runs/{exp_name}/seed{seed}/best.pt etc. and one row per seed in results/runs.csv.
"""
import argparse
import statistics
from functools import partial

from src.dvc_standin import build_standin_loaders, resolve_root
from src.trainer import TrainConfig, check_new_run, run_experiment

STANDIN_NOTE = "stand-in loader"


def parse_overrides(items):
    """`--set` pairs -> dict of TrainConfig overrides; see TrainConfig.parse_overrides."""
    return TrainConfig.parse_overrides(items)


def build_configs(config_path, seeds=None, overrides=None):
    """One TrainConfig per seed (or just the config's own seed), with overrides applied and the run
    tagged as using the stand-in loader."""
    overrides = dict(overrides or {})
    if seeds and "seed" in overrides:
        raise ValueError("give the seed(s) either with --seeds or with --set seed=..., not both")
    if seeds and len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate --seeds {seeds}")
    cfgs = [TrainConfig.from_yaml(config_path, **overrides, **({"seed": s} if s is not None else {}))
            for s in (seeds or [None])]
    for cfg in cfgs:
        if STANDIN_NOTE not in cfg.notes:
            cfg.notes = f"{cfg.notes}; {STANDIN_NOTE}" if cfg.notes else STANDIN_NOTE
    return cfgs


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="YAML file of TrainConfig fields")
    ap.add_argument("--seeds", type=int, nargs="+", help="run once per seed, e.g. --seeds 0 1 2")
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="override TrainConfig fields; text fields kept as typed, others parsed as YAML (empty = null)")
    ap.add_argument("--data-root", help="folder with train/ and val/ (default: $DVC_ROOT, then "
                                        "data/dogs-vs-cats/datasets)")
    return ap, ap.parse_args(argv)


def main(argv=None):
    ap, args = parse_args(argv)
    try:
        cfgs = build_configs(args.config, args.seeds, parse_overrides(args.overrides))
        for cfg in cfgs:  # the trainer refuses these too, but only after the earlier seeds have trained
            check_new_run(cfg)
        root = resolve_root(args.data_root)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        ap.error(str(e))
    build_loaders = partial(build_standin_loaders, root=root)
    results = []
    for cfg in cfgs:
        print(f"== {cfg.exp_name} seed {cfg.seed}: {cfg.model} pretrained={cfg.pretrained} freeze={cfg.freeze} "
              f"aug={cfg.aug} n_train={cfg.n_train} data={root}", flush=True)
        res = run_experiment(cfg, build_loaders)
        res.pop("model")  # weights are in best.pt; don't keep one model per seed on the GPU
        print(f"== {cfg.exp_name} seed {cfg.seed}: best val acc {res['best_val_acc']:.4f} at epoch "
              f"{res['best_epoch']} -> {res['ckpt_path']}", flush=True)
        results.append(res)
    if len(results) > 1:
        accs = [r["best_val_acc"] for r in results]
        print(f"== {cfgs[0].exp_name}: val acc {statistics.mean(accs):.4f} +- {statistics.stdev(accs):.4f} "
              f"over seeds {[c.seed for c in cfgs]}", flush=True)
    return results


if __name__ == "__main__":
    main()
