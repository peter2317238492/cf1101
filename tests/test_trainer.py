import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset

from src.models.registry import build_model, load_ckpt, register
from src.trainer import (
    RUNS_CSV_COLUMNS, TrainConfig, _git_commit, _robust_write, build_optimizer, build_scheduler, evaluate, fit,
    loader_kwargs, predict, run_experiment, set_seed,
)
from tests.conftest import FlipAfter, TinyNet, blob_loaders, make_blobs

CONTRACT_RUNS_COLUMNS = [
    "run_id", "member", "task", "model", "pretrained", "freeze", "aug", "img_size", "n_train", "n_val",
    "optimizer", "lr", "wd", "bs", "epochs", "scheduler", "seed", "best_val_acc", "best_epoch", "val_loss",
    "test_acc", "macro_f1", "train_time_s", "ckpt_path", "git_commit", "notes",
]


# ---------- config ----------

def test_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="lrr"):
        TrainConfig.from_dict({"lrr": 0.1})


def test_config_from_yaml_applies_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 0.5\nepochs: 7\n")
    cfg = TrainConfig.from_yaml(path, epochs=9)
    assert cfg.lr == 0.5
    assert cfg.epochs == 9


# ---------- seeding ----------

def test_loader_kwargs_make_shuffle_order_reproducible():
    ds = TensorDataset(torch.arange(100))

    def order(seed):
        return torch.cat([b[0] for b in DataLoader(ds, batch_size=10, shuffle=True, **loader_kwargs(seed))])

    assert torch.equal(order(7), order(7))
    assert not torch.equal(order(7), order(8))


def test_same_seed_gives_identical_history(make_cfg):
    h1 = run_experiment(make_cfg(exp_name="a"), blob_loaders)["history"]
    h2 = run_experiment(make_cfg(exp_name="b"), blob_loaders)["history"]
    h3 = run_experiment(make_cfg(exp_name="c", seed=1), blob_loaders)["history"]
    assert [r["val_loss"] for r in h1] == [r["val_loss"] for r in h2]
    assert [r["val_loss"] for r in h1] != [r["val_loss"] for r in h3]


# ---------- training ----------

def test_fit_learns_separable_blobs(make_cfg):
    res = run_experiment(make_cfg(epochs=5), blob_loaders)
    assert res["best_val_acc"] > 0.9


def test_injected_criterion_drives_training(make_cfg):
    res = run_experiment(make_cfg(epochs=5), blob_loaders, criterion=FlipAfter(16))
    assert res["history"][-1]["val_acc"] < 0.5  # only possible if the flipped-label criterion was used


def test_fit_restores_best_epoch_weights(make_cfg):
    res = run_experiment(make_cfg(epochs=5), blob_loaders, criterion=FlipAfter(16))
    assert res["history"][-1]["val_acc"] < res["best_val_acc"]
    val = blob_loaders(make_cfg())["val"]
    assert evaluate(res["model"], val, amp=False)["acc"] == pytest.approx(res["best_val_acc"])


def test_best_ckpt_reloads_with_same_val_acc(make_cfg):
    res = run_experiment(make_cfg(epochs=5), blob_loaders, criterion=FlipAfter(16))
    model, ckpt = load_ckpt(res["ckpt_path"])
    assert ckpt["model_name"] == "tiny_test_net"
    assert ckpt["epoch"] == res["best_epoch"]
    assert ckpt["seed"] == 0
    val = blob_loaders(make_cfg())["val"]
    assert evaluate(model, val, amp=False)["acc"] == pytest.approx(res["best_val_acc"])


def test_early_stopping_halts_after_patience(make_cfg):
    # frozen backbone + lr 0 => val metrics never improve after epoch 1
    res = run_experiment(make_cfg(freeze="backbone", lr=0.0, epochs=10, patience=2), blob_loaders)
    assert res["best_epoch"] == 1
    assert len(res["history"]) == 3


def test_frozen_backbone_weights_and_bn_stats_stay_fixed(make_cfg):
    res = run_experiment(make_cfg(freeze="backbone", epochs=2), blob_loaders)
    set_seed(0)
    fresh = build_model("tiny_test_net", 2, freeze="backbone")
    model = res["model"].cpu()
    assert torch.equal(model.conv.weight, fresh.conv.weight)
    assert torch.equal(model.bn.running_mean, fresh.bn.running_mean)
    assert torch.equal(model.bn.running_var, fresh.bn.running_var)
    assert not torch.equal(model.fc.weight, fresh.fc.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_amp_channels_last_training_on_cuda(make_cfg):
    res = run_experiment(make_cfg(device="cuda", amp=True, channels_last=True, deterministic=False, epochs=5),
                         blob_loaders)
    assert res["best_val_acc"] > 0.9
    assert all(np.isfinite(r["train_loss"]) for r in res["history"])


# ---------- outputs on disk ----------

def test_fit_writes_run_artifacts(make_cfg, tmp_path):
    cfg = make_cfg(exp_name="artifacts", seed=3, member="B", notes="hello")
    res = run_experiment(cfg, blob_loaders)
    run_dir = tmp_path / "runs" / "artifacts" / "seed3"
    assert Path(res["ckpt_path"]) == run_dir / "best.pt"
    for name in ["config.yaml", "history.csv", "best.pt", "metrics.json"]:
        assert (run_dir / name).exists(), name

    hist = pd.read_csv(run_dir / "history.csv")
    assert {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"} <= set(hist.columns)
    assert len(hist) == cfg.epochs
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["seed"] == 3
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["best_val_acc"] == pytest.approx(res["best_val_acc"])
    assert len(metrics["confusion"]) == 2

    rows = pd.read_csv(tmp_path / "results" / "runs.csv")
    assert list(rows.columns[: len(CONTRACT_RUNS_COLUMNS)]) == CONTRACT_RUNS_COLUMNS  # extra columns only appended
    row = rows.iloc[0]
    assert (row["member"], row["seed"], row["n_train"], row["n_val"], row["notes"]) == ("B", 3, 128, 64, "hello")
    assert row["best_val_acc"] == pytest.approx(res["best_val_acc"])


def test_runs_csv_appends_one_row_per_run(make_cfg, tmp_path):
    run_experiment(make_cfg(epochs=1), blob_loaders)
    run_experiment(make_cfg(epochs=1, seed=1), blob_loaders)
    rows = pd.read_csv(tmp_path / "results" / "runs.csv")
    assert rows["seed"].tolist() == [0, 1]


def test_ckpt_records_dataset_class_to_idx_and_preprocessing(make_cfg):
    class Named(TensorDataset):
        class_to_idx = {"cat": 0, "dog": 1}

    def loaders(cfg):
        x, y = make_blobs(128, 2, seed=1)
        return {"train": DataLoader(Subset(Named(x, y), range(100)), batch_size=16),
                "val": blob_loaders(cfg)["val"]}

    cfg = make_cfg(epochs=1)
    _, ckpt = load_ckpt(run_experiment(cfg, loaders)["ckpt_path"])
    assert ckpt["class_to_idx"] == {"cat": 0, "dog": 1}
    assert ckpt["img_size"] == cfg.img_size
    assert ckpt["norm_mean"] == list(cfg.norm_mean)


# ---------- metrics / prediction ----------

def test_evaluate_reports_known_metrics():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = F.one_hot(torch.tensor([0, 1, 1, 1, 2, 0]), 3).float() * 5
    m = evaluate(nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=4), amp=False)
    assert m["acc"] == pytest.approx(4 / 6)
    np.testing.assert_array_equal(m["confusion"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    np.testing.assert_allclose(m["per_class_acc"], [0.5, 1.0, 0.5])
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["loss"] == pytest.approx(F.cross_entropy(logits, labels).item(), rel=1e-5)
    np.testing.assert_allclose(m["probs"].sum(1), 1, rtol=1e-5)
    np.testing.assert_array_equal(m["labels"], labels.numpy())


def test_macro_f1_matches_sklearn_on_random_predictions():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 4, (200,), generator=g)
    logits = torch.randn(200, 5, generator=g)  # class 4 is predicted but never a true label
    m = evaluate(nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=64), amp=False)
    expected = f1_score(labels.numpy(), logits.argmax(1).numpy(), average="macro")
    assert m["macro_f1"] == pytest.approx(expected)


def test_macro_f1_ignores_class_absent_from_labels_and_preds():
    # 3 logits but class 2 never appears: sklearn averages over classes {0, 1} only
    labels = torch.tensor([0, 0, 1, 1])
    logits = F.one_hot(torch.tensor([0, 1, 1, 1]), 3).float()
    m = evaluate(nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=4), amp=False)
    assert m["macro_f1"] == pytest.approx(f1_score(labels.numpy(), [0, 1, 1, 1], average="macro"))


def test_predict_returns_probs_in_loader_order():
    ids = torch.tensor([5, 3, 9, 1])
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [3.0, 0.0]])
    df = predict(nn.Identity(), DataLoader(TensorDataset(logits, ids), batch_size=3), amp=False)
    assert list(df.columns) == ["id", "prob_0", "prob_1", "pred"]
    assert df["id"].tolist() == [5, 3, 9, 1]
    np.testing.assert_allclose(df[["prob_0", "prob_1"]].sum(1), 1, rtol=1e-6)
    assert df["pred"].tolist() == [0, 1, 0, 0]


# ---------- optimizer / scheduler ----------

def test_weight_decay_skips_bias_and_norm_params():
    model = build_model("tiny_test_net", 2)
    opt = build_optimizer(model, TrainConfig(optimizer="sgd", lr=0.1, wd=5e-4))
    seen = 0
    for g in opt.param_groups:
        for p in g["params"]:
            assert g["weight_decay"] == (5e-4 if p.ndim > 1 else 0.0)
            seen += 1
    assert seen == len(list(model.parameters()))


def test_lr_backbone_applies_to_non_head_params():
    model = build_model("tiny_test_net", 2)
    opt = build_optimizer(model, TrainConfig(optimizer="adamw", lr=1e-3, lr_backbone=1e-4, wd=0.01))
    assert isinstance(opt, torch.optim.AdamW)
    head = {id(p) for p in model.fc.parameters()}
    for g in opt.param_groups:
        for p in g["params"]:
            assert g["lr"] == (1e-3 if id(p) in head else 1e-4)


def test_optimizer_excludes_frozen_params():
    model = build_model("tiny_test_net", 2, freeze="backbone")
    opt = build_optimizer(model, TrainConfig())
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in model.fc.parameters()}


def _lr_trace(cfg, steps_per_epoch):
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = build_scheduler(opt, cfg, steps_per_epoch=steps_per_epoch)
    lrs = []
    for _ in range(cfg.epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_cosine_schedule_warms_up_then_decays():
    lrs = _lr_trace(TrainConfig(scheduler="cosine", warmup_epochs=1, epochs=3), steps_per_epoch=4)
    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))
    assert lrs[-1] < 0.1


def test_step_schedule_decays_every_step_epochs():
    lrs = _lr_trace(TrainConfig(scheduler="step", step_epochs=2, gamma=0.1, epochs=4), steps_per_epoch=2)
    assert lrs == pytest.approx([1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1])


def test_onecycle_schedule_peaks_at_base_lr():
    lrs = _lr_trace(TrainConfig(scheduler="onecycle", epochs=4), steps_per_epoch=10)
    assert max(lrs) == pytest.approx(1.0, rel=1e-3)
    assert lrs[0] < 0.1
    assert lrs[-1] < 0.01


# ---------- run bookkeeping ----------

def test_rerun_of_completed_run_raises_unless_overwrite(make_cfg):
    run_experiment(make_cfg(epochs=1), blob_loaders)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_experiment(make_cfg(epochs=1), blob_loaders)
    assert run_experiment(make_cfg(epochs=1, overwrite=True), blob_loaders)["best_epoch"] == 1


def test_run_ids_are_unique_and_returned(make_cfg, tmp_path):
    r1 = run_experiment(make_cfg(epochs=1, exp_name="u1"), blob_loaders)
    r2 = run_experiment(make_cfg(epochs=1, exp_name="u1", seed=1), blob_loaders)
    assert r1["run_id"] != r2["run_id"]
    assert pd.read_csv(tmp_path / "results" / "runs.csv")["run_id"].tolist() == [r1["run_id"], r2["run_id"]]


def test_runs_csv_with_older_header_is_migrated(make_cfg, tmp_path):
    csv_path = tmp_path / "results" / "runs.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame([{c: "" for c in CONTRACT_RUNS_COLUMNS} | {"run_id": "old_run", "seed": 9}]).to_csv(
        csv_path, index=False)
    run_experiment(make_cfg(epochs=1), blob_loaders)
    rows = pd.read_csv(csv_path)
    assert list(rows.columns) == RUNS_CSV_COLUMNS
    assert rows["run_id"].tolist()[0] == "old_run"
    assert len(rows) == 2


def test_runs_csv_with_unknown_columns_fails_before_training(make_cfg, tmp_path):
    csv_path = tmp_path / "results" / "runs.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("run_id,bogus\nx,y\n")
    with pytest.raises(ValueError, match="bogus"):
        run_experiment(make_cfg(epochs=1), blob_loaders)
    assert not (tmp_path / "runs" / "t").exists()


def test_robust_write_retries_transient_permission_error():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("locked by Excel")

    assert _robust_write(flaky, "x.csv", retries=5, delay=0) is True
    assert len(calls) == 3


def test_robust_write_gives_up_with_warning():
    def locked():
        raise PermissionError("locked by Excel")

    with pytest.warns(UserWarning, match="x.csv"):
        assert _robust_write(locked, "x.csv", retries=2, delay=0) is False


def test_git_commit_tracks_code_changes_only(tmp_path):
    git = ["git", "-c", "safe.directory=*", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    assert _git_commit(tmp_path) == "no-commit"
    subprocess.run(git + ["init", "-q"], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "runs.csv").write_text("run_id\n")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    sha = _git_commit(tmp_path)
    assert sha not in ("", "no-commit") and not sha.endswith("-dirty")
    (tmp_path / "notes.txt").write_text("untracked")
    (tmp_path / "results" / "runs.csv").write_text("run_id\nr1\n")
    assert _git_commit(tmp_path) == sha
    (tmp_path / "src" / "new_model.py").write_text("y = 1\n")  # new code that was never git-added
    assert _git_commit(tmp_path) == sha + "-dirty"
    (tmp_path / "src" / "new_model.py").unlink()
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert _git_commit(tmp_path) == sha + "-dirty"


def test_n_train_counts_subset_sampler_indices(make_cfg, tmp_path):
    def loaders(cfg):
        x, y = make_blobs(128, 2, seed=1)
        sampler = SubsetRandomSampler(list(range(40)), generator=torch.Generator().manual_seed(cfg.seed))
        return {"train": DataLoader(TensorDataset(x, y), batch_size=16, sampler=sampler),
                "val": blob_loaders(cfg)["val"]}

    run_experiment(make_cfg(epochs=1), loaders)
    assert pd.read_csv(tmp_path / "results" / "runs.csv")["n_train"].tolist() == [40]


# ---------- losses (C's imbalance runs) ----------

def test_injected_criterion_must_carry_cfg_label_smoothing(make_cfg):
    with pytest.raises(ValueError, match="label_smoothing"):
        run_experiment(make_cfg(label_smoothing=0.1), blob_loaders, criterion=nn.CrossEntropyLoss())
    res = run_experiment(make_cfg(label_smoothing=0.1, exp_name="ok"), blob_loaders,
                         criterion=nn.CrossEntropyLoss(label_smoothing=0.1))
    assert res["best_val_acc"] > 0.5


def test_non_ce_loss_name_requires_injected_criterion(make_cfg):
    with pytest.raises(ValueError, match="criterion"):
        run_experiment(make_cfg(loss="focal"), blob_loaders)


def test_loss_sampler_and_criterion_are_recorded(make_cfg, tmp_path):
    cfg = make_cfg(loss="weighted_ce", sampler="weighted", imb_ratio=0.1, exp_name="rec")
    run_experiment(cfg, blob_loaders, criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0])))
    row = pd.read_csv(tmp_path / "results" / "runs.csv").iloc[0]
    assert (row["exp_name"], row["loss"], row["sampler"], row["imb_ratio"]) == ("rec", "weighted_ce", "weighted", 0.1)
    metrics = json.loads((tmp_path / "runs" / "rec" / "seed0" / "metrics.json").read_text())
    assert metrics["criterion"]["type"] == "CrossEntropyLoss"
    assert metrics["criterion"]["weight"] == [1.0, 2.0]


# ---------- test-set evaluation (CIFAR) ----------

def test_eval_test_records_test_metrics(make_cfg, tmp_path):
    def loaders(cfg):
        x, y = make_blobs(64, 2, seed=3)
        return {**blob_loaders(cfg), "test": DataLoader(TensorDataset(x, y), batch_size=16)}

    res = run_experiment(make_cfg(epochs=4, exp_name="withtest"), loaders, eval_test=True)
    assert res["test_metrics"]["acc"] > 0.9
    row = pd.read_csv(tmp_path / "results" / "runs.csv").iloc[0]
    assert row["test_acc"] == pytest.approx(res["test_metrics"]["acc"])
    assert row["test_macro_f1"] == pytest.approx(res["test_metrics"]["macro_f1"])
    metrics = json.loads((tmp_path / "runs" / "withtest" / "seed0" / "metrics.json").read_text())
    assert metrics["test_acc"] == pytest.approx(res["test_metrics"]["acc"])
    assert len(metrics["test_confusion"]) == 2


# ---------- guarding direct fit() calls ----------

def test_fit_rejects_model_that_does_not_match_cfg_model(make_cfg):
    loaders = blob_loaders(make_cfg())
    with pytest.raises(ValueError, match="tiny_test_net"):
        fit(build_model("tiny_test_net", 2), loaders["train"], loaders["val"], make_cfg(model="resnet18"))


def test_fit_rejects_unregistered_model(make_cfg):
    loaders = blob_loaders(make_cfg())
    with pytest.raises(ValueError, match="build_model"):
        fit(TinyNet(2), loaders["train"], loaders["val"], make_cfg())


def test_fit_rejects_num_classes_mismatch(make_cfg):
    loaders = blob_loaders(make_cfg())
    set_seed(0)
    with pytest.raises(ValueError, match="num_classes"):
        fit(build_model("tiny_test_net", 2), loaders["train"], loaders["val"], make_cfg(num_classes=3))


def test_fit_warns_when_seed_was_not_set(make_cfg):
    torch.manual_seed(999)
    loaders = blob_loaders(make_cfg())
    with pytest.warns(UserWarning, match="set_seed"):
        fit(build_model("tiny_test_net", 2), loaders["train"], loaders["val"], make_cfg(epochs=1))


class _ViewNet(nn.Module):
    """Flattens a conv map with .view, as many hand-written CNNs do."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        h = F.relu(self.conv(x))
        return self.fc(h.view(h.size(0), -1))


register("view_test_net", head=("fc",), input_size=8)(lambda num_classes, pretrained=False: _ViewNet(num_classes))


def test_channels_last_is_ignored_on_cpu_for_view_flatten_models(make_cfg):
    res = run_experiment(make_cfg(model="view_test_net", channels_last=True, epochs=1), blob_loaders)
    assert res["best_epoch"] == 1


# ---------- labels / ids ----------

def test_class_to_idx_from_loaders_dict_is_saved(make_cfg):
    def loaders(cfg):
        return {**blob_loaders(cfg), "class_to_idx": {"cat": 0, "dog": 1}}

    _, ckpt = load_ckpt(run_experiment(make_cfg(epochs=1), loaders)["ckpt_path"])
    assert ckpt["class_to_idx"] == {"cat": 0, "dog": 1}


def test_dvc_task_rejects_swapped_labels(make_cfg):
    def loaders(cfg):
        return {**blob_loaders(cfg), "class_to_idx": {"dog": 0, "cat": 1}}

    with pytest.raises(ValueError, match="cat"):
        run_experiment(make_cfg(task="dvc", epochs=1), loaders)


def test_dvc_task_warns_when_class_to_idx_unknown(make_cfg):
    with pytest.warns(UserWarning, match="class_to_idx"):
        run_experiment(make_cfg(task="dvc", epochs=1), blob_loaders)


def test_predict_accepts_string_ids():
    class StrIds(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return torch.tensor([float(i), 0.0]), str(i + 10)

    df = predict(nn.Identity(), DataLoader(StrIds(), batch_size=2), amp=False)
    assert df["id"].tolist() == [10, 11, 12]


# ---------- config coercion ----------

def test_yaml_scientific_notation_becomes_float(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 1e-3\nlr_backbone: 1e-4\nwd: 5e-4\npatience: null\n")
    cfg = TrainConfig.from_yaml(path)
    assert (cfg.lr, cfg.lr_backbone, cfg.wd, cfg.patience) == (1e-3, 1e-4, 5e-4, None)
    assert isinstance(cfg.lr, float)


def test_config_normalizes_numpy_path_and_list_values(tmp_path):
    cfg = TrainConfig(lr=np.float64(0.1), epochs=np.int64(3), out_dir=tmp_path, norm_mean=[0.5, 0.5, 0.5])
    assert type(cfg.lr) is float and type(cfg.epochs) is int
    assert cfg.out_dir == str(tmp_path)
    assert cfg.norm_mean == (0.5, 0.5, 0.5)
    yaml.safe_dump(cfg.to_dict())


@pytest.mark.parametrize("field,value", [("amp", "maybe"), ("epochs", 2.5), ("lr", "fast")])
def test_config_rejects_uncoercible_values(field, value):
    with pytest.raises(ValueError, match=field):
        TrainConfig(**{field: value})
