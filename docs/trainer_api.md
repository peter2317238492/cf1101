# 训练器使用说明（给 A、C）

代码在 `src/trainer.py`（训练器）和 `src/models/registry.py`（模型注册表、checkpoint 读写）。

```bash
python -m pytest -q                                            # 148 个单元测试，约 20 秒（没找到真实 CIFAR 时跳过 1 个）
python -m scripts.smoke_dvc --n-train 1000 --n-val 500 --epochs 2   # 真实猫狗数据的 GPU 冒烟测试
python -m scripts.smoke_cifar --epochs 1                            # 真实 CIFAR-10 的 GPU 冒烟测试（45k，约 1 分钟；11/1 前 channels_last 默认开着时约 1.7 分钟）
```

数据不在 git 里。放在项目下的 `data/` 就能直接跑；放在别处（或在 worktree 里跑）时设环境变量 `DVC_ROOT`（指向 `dogs-vs-cats/datasets`）和 `CIFAR10_ROOT`（指向 `cifar10`）。

## 1. 最小用法

一次训练对应一个 config 加一个 seed。建议统一从 `run_experiment` 进入：它负责设种子、建 loader、建模型、训练，并把最好那个 epoch 的权重留在 `result["model"]` 里。

```python
from torch.utils.data import DataLoader
from src.trainer import TrainConfig, loader_kwargs, run_experiment

def build_loaders(cfg):
    ...  # 按 cfg.img_size / cfg.aug / cfg.n_train / cfg.norm_mean / cfg.norm_std 构造数据集
    return {
        "train": DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.num_workers,
                            pin_memory=True, persistent_workers=cfg.num_workers > 0, **loader_kwargs(cfg.seed)),
        "val": DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_workers),
        "class_to_idx": {"cat": 0, "dog": 1},   # 可选；dataset 自带 .class_to_idx 时可以不写
        # "test": 有标签的测试 loader（只有 CIFAR 有），配合 eval_test=True 使用
    }

if __name__ == "__main__":          # Windows 下 num_workers>0 时必须加这一句
    cfg = TrainConfig.from_yaml("configs/dvc_r18_linear_probe.yaml", seed=0, exp_name="debug_dvc", member="A",
                                out_dir="runs/debug", results_csv="runs/debug/runs.csv")  # 调试不写进总表
    res = run_experiment(cfg, build_loaders)
    print(res["best_val_acc"], res["ckpt_path"])
```

- 猫狗现成的命令行入口：`python -m scripts.train_dvc --config <yaml> [--seeds 0 1 2] [--set key=value ...] [--data-root DIR]`。A 的 `build_dvc_loaders` 到位之前，它用 `src/dvc_standin.py` 的临时 loader（ImageFolder 直读 train/ 和 val/，`aug` 只支持 `none` / `flip`），这些 run 的 `notes` 里带 "stand-in loader"。
- 训练集和验证集的 loader 产出 `(x, label)`；交给 `predict` 的 loader 产出 `(x, id)`。猫狗的 test 没有标签，`task: dvc` 时 `eval_test=True` 会直接报错，用 `predict`。
- `loader_kwargs(cfg.seed)` 把 shuffle 顺序和 worker 内的随机增强都固定到 seed 上。
- 多 seed 统一用 `[0, 1, 2]`，报告时写 mean±std。

## 2. A：注册从零训练的 CNN

把模型放在 `src/models/` 下的任意文件里（例如 `src/models/scratch_cnn.py`），训练器和 `load_ckpt` 会自动找到，不用手动 import。

```python
from torch import nn
from src.models.registry import register

@register("scratch_cnn_v1", head=("classifier",), input_size=128)
def scratch_cnn_v1(num_classes, pretrained=False):
    return ScratchCNN(num_classes)
```

规则：
- `forward` 输出 logits，不要在里面做 softmax。
- 展平特征图用 `torch.flatten(x, 1)`、`nn.Flatten()` 或 `reshape`，**不要用 `.view`**。cfg 里开了 `channels_last` 时 `.view` 会直接报错。`python -m pytest -q` 会自动检查所有已注册模型。
- `head` 写分类层的属性名，`freeze` 和 `lr_backbone` 靠它区分头和骨干。
- `input_size` 只有在全连接层要求固定输入尺寸时才写。

## 3. C：CIFAR 不平衡实验

### CIFAR-10 数据（`src/cifar/data.py`）

50k 官方训练集已按 seed 42 分层划成 45k train / 5k val（每类 4500/500），索引在 `splits/cifar10_train45k.csv` 和 `splits/cifar10_val5k.csv` 里（列为 `index,label`，index 是 data_batch_1..5 拼接后的位置）。这两个文件已提交，loader 直接读它们，并核对标签和数据是否一致。不要自己重新划分；`python -m scripts.make_cifar_splits` 重跑只会得到完全相同的文件。

```python
import dataclasses
from functools import partial
from src.cifar.data import CIFAR10_MEAN, CIFAR10_STD, build_cifar_loaders, read_split

cfg = TrainConfig(task="cifar10", model="cifar_resnet18", num_classes=10, img_size=32, aug="crop_flip",
                  norm_mean=CIFAR10_MEAN, norm_std=CIFAR10_STD, exp_name="debug_cifar", member="C",
                  out_dir="runs/debug", results_csv="runs/debug/runs.csv", ...)    # 调试不写进总表
res = run_experiment(cfg, build_cifar_loaders)                  # 只看 val；eval_test=True 只在最终 run 用

train = read_split("train")          # DataFrame[index, label]，不平衡子集只能从这里抽
imb_indices = ...                    # 按类从 train["index"] 里取
cfg_imb = dataclasses.replace(cfg, task="cifar10_imb", exp_name="debug_imb_wce", loss="weighted_ce")
res = run_experiment(cfg_imb, partial(build_cifar_loaders, train_indices=imb_indices), criterion=my_weighted_ce)
```

- 数据路径：`root=` 参数 > 环境变量 `CIFAR10_ROOT` > 项目下的 `data/cifar10`。指向 `cifar-10-batches-py` 本身或它的上一级都可以。在 worktree 里运行时设 `CIFAR10_ROOT=E:/6483proj/data/cifar10`。
- 返回 `train` / `val` / `test` 三个 loader，都产出 `(x, label)`，另外附带 `class_to_idx`。`test` 是官方 10k，只在最终评估时用一次（`eval_test=True`）。
- `aug` 可选 `none`、`crop_flip`（RandomCrop 32 padding 4 + 水平翻转），只作用于训练集。`img_size` 不是 32 时会把图放大（例如 ImageNet 预训练的 `resnet18` 用 224）；`cifar_resnet18` 必须配 `img_size: 32`。
- `n_train`：从 45k 里取一个按类均衡的嵌套子集，所有 seed 用的是同一个子集。它不能和 `train_indices` 同时使用。
- `train_indices`：传入后训练集变成 `Subset(train_45k, ...)`。不在 45k 里的索引（比如 val 的）会直接报错；允许重复索引，可以用来做过采样。
- **`cfg.sampler` 的名字只有走 `scripts/train_cifar.py`（或 `src.cifar.train.run_cifar`）才会生效**，所以用采样器时请走这两个入口。直接 `run_experiment(cfg, build_cifar_loaders)` 时如果 `cfg.sampler` 不是 `none`，`build_cifar_loaders` 会报错，免得 runs.csv 记了一个没用上的采样器。要自己搭 loader 就用 `build_cifar_datasets` + `cifar_loaders(cfg, ds, sampler=...)`，训练集标签用 `dataset_targets(ds["train"])` 取（已经处理了 `Subset`）。
- 自己传 `criterion=` 或 `sampler=` 时，`cfg.loss` / `cfg.sampler` 也必须写成它的名字（例如 `loss="weighted_ce"`），否则训练前就报错：runs.csv 记的是 cfg 里的名字。

### 训练入口 `scripts/train_cifar.py`

```bash
python -m scripts.train_cifar --config configs/cifar_baseline.yaml [--seeds 0 1 2] [--set key=value ...] [--train-indices <csv>] [--root <cifar dir>] [--eval-test]

# 例：一组不平衡实验（3 个 seed，只看 val）
python -m scripts.train_cifar --config configs/cifar_baseline.yaml --seeds 0 1 2 --train-indices splits/cifar10_imb_step_r0.1.csv --set task=cifar10_imb member=C exp_name=cifar_imb_focal_r0.1_base loss=focal imb_ratio=0.1
```

- `--seeds`：依次跑这几个 seed，默认用 config 里的 `seed`。跑之前会先检查每个 seed 是否已经跑完过（本地 run 目录或 `results/runs.csv` 里已有同名 `exp_name` + `seed`），跑完过的直接报错；重复的 seed 也报错。
- `--set`：覆盖任意 `TrainConfig` 字段，可以写多个。文字字段（`exp_name`、`notes`、`loss` 等）原样保留，其他值按 YAML 解析（`lr=1e-3`、`n_train=null`、`"norm_mean=[0.5, 0.5, 0.5]"`，带空格的要加引号）。字段名写错直接报错。
- `--train-indices`：CSV，必须有 `index` 列（50k 官方训练集里的位置）。所有索引都要在 `splits/cifar10_train45k.csv` 里，否则训练前就报错，并说明有几个落在 val 里。允许重复索引；不能和 `n_train` 同时用。文件名和索引的哈希会自动追加到 `notes`（例如 `idx=cifar10_imb_step_r0.1.csv@1a2b3c4d (24750)`），runs.csv 里就能看出每次 run 用的是哪个子集；从 Python 调 `run_cifar(cfg, train_indices=idx)` 也会记（没有文件名，只有哈希）。
- `--eval-test`：用最好那个 epoch 的模型在官方 10k test 上只评一次，结果写进 `runs.csv` 的 `test_acc` / `test_macro_f1` 和 `metrics.json` 的 `test_per_class_acc` / `test_confusion`。**只在最终 run 加**，平时默认关闭，只看 val。
- 平衡基线是 `configs/cifar_baseline.yaml`：ImageNet 预训练 `resnet18`，图放大到 224，全量微调，10 个 epoch。seed 0 的 best val acc 为 0.9698（第 10 个 epoch），训练 7.7 分钟（暂定：这次 run 的代码当时还没提交，记录为 `c2cf275-dirty`，提交后会重跑），见 `results/runs.csv` 中 `exp_name=cifar_baseline_r18_pt224` 那一行。
- 耗时（RTX 3070 Ti Laptop，45k，bs 128，AMP）：224 px 预训练 `resnet18` 每个 epoch 约 42 s；32 px 从零训练的 `cifar_resnet18`（`--set model=cifar_resnet18 img_size=32 pretrained=false exp_name=cifar_scratch_r18_32`，再把 `norm_mean` / `norm_std` 换成 CIFAR 的）每个 epoch 约 12 s。每次 run 另有约 45 s 固定开销（读数据、Windows 下启动 worker），都算在第 1 个 epoch 里。子集更小时，每个 epoch 的耗时按样本数等比例减少。
- Python 里调用：`from src.cifar.train import run_cifar, load_train_indices`，然后 `run_cifar(cfg, train_indices=idx, eval_test=False)`。

### 加 loss / sampler（注册表在 `src/cifar/train.py`）

`cfg.loss`、`cfg.sampler` 按名字从注册表里取。B 已经提供 `ce`（普通 CE，带 `cfg.label_smoothing`）和 `none`（不用采样器，普通 shuffle）。新方法写在 `src/cifar/` 下任意文件里（例如 `src/cifar/imbalance.py`），第一次用到未知名字时会自动 import，别的文件都不用改。

```python
import numpy as np, torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from src.cifar.train import register_loss, register_sampler

@register_loss("weighted_ce")
def weighted_ce(cfg, train_labels):        # 实际训练子集的标签（np.int64），已经按 train_indices / n_train 取过
    counts = np.bincount(train_labels, minlength=cfg.num_classes)
    w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

@register_sampler("weighted")
def weighted(cfg, train_labels):
    w = 1.0 / np.bincount(train_labels)[train_labels]       # 每个训练样本一个权重
    return WeightedRandomSampler(w, num_samples=len(w), generator=torch.Generator().manual_seed(cfg.seed))
```

- loss 返回 `nn.Module`，必须用 `label_smoothing=cfg.label_smoothing` 构造，不一致时训练器会报错。里面的张量用 `register_buffer` 或 `weight=` 存，这样 `.to(device)` 才会带过去。Focal loss 照这个写法自己写一个 `nn.Module`。
- sampler 返回 `torch.utils.data.Sampler`，产出训练子集里的位置 `0..len(train_labels)-1`；返回 `None` 就是普通 shuffle。用了 sampler 时 train loader 不再 shuffle；sampler 的 `generator` 是 None 时会自动用 `cfg.seed` 设好（同一个 sampler 换 seed 复用时也会重新设）。sampler 的参数不会单独记录，有参数的话写进名字或 `notes`。
- 名字写错会报错，并列出所有可用的名字；同名重复注册也会报错。放在子文件夹里的话，子文件夹要有 `__init__.py` 才会被自动找到。实际用的名字记在 `runs.csv` 的 `loss` / `sampler` 列，criterion 的类型和参数记在 `metrics.json` 的 `criterion`。
- 验证集和测试集的 loss 一律用普通 CE 计算，所以不同 loss 的实验之间可以直接比较。

## 4. 预测并生成 submission.csv（A）

```python
from src.models.registry import load_ckpt
from src.trainer import predict

model, ckpt = load_ckpt("runs/<exp>/seed0/best.pt")
model = model.to("cuda")                      # load_ckpt 返回的模型在 CPU 上，predict 在模型所在的设备上跑
# 测试图的预处理要和 ckpt["img_size"]、ckpt["norm_mean"]、ckpt["norm_std"] 保持一致
df = predict(model, test_loader)              # 返回列 id, prob_0, prob_1, pred；字符串数字 id 会自动转成 int
df = df.sort_values("id")                     # 按数值排序
df[["id", "pred"]].rename(columns={"pred": "label"}).to_csv("submission.csv", index=False)  # 1=dog, 0=cat
```

`task: dvc` 时训练器会强制检查 `class_to_idx == {'cat': 0, 'dog': 1}`，所以 `pred` 本身就是 1=dog、0=cat。

## 5. 输出文件

`runs/{exp_name}/seed{seed}/` 下有四个文件：

- `config.yaml`：本次使用的完整配置
- `history.csv`：每个 epoch 一行（train/val 的 loss 和 acc、val macro-F1、lr、耗时）
- `best.pt`：验证集最好那个 epoch 的权重，附带能完整重建模型的元数据（模型名、cfg、class_to_idx、预处理参数、seed、git commit、run_id、loss 描述）
- `metrics.json`：最佳 epoch 的指标、每类准确率、混淆矩阵、训练时长、设备

`results/runs.csv` 每次运行追加一行，是全组共用的结果总表，Q3、Q4、Q7、Q9 的表和图都从它生成。

- 同一个 `exp_name` 加 `seed` 已经完整跑过的话（本地有 `metrics.json`，或者 `results/runs.csv` 里已经有这一行，包括组员从别的电脑提交上来的），再跑会在训练前直接报错。确实要重跑就在 cfg 里加 `overwrite: true`。**总表里的旧行不要删**：union 合并会把别人那边还留着的旧行带回来。`aggregate.py` 对同一个 `exp_name` + `seed` 只取最新的 `run_id`。
- **不要用 Excel 直接打开 `runs.csv`**，Excel 会锁住文件。要看的话用 pandas、VS Code，或者先复制一份。万一写入失败，训练器会重试约 30 秒；还是失败就把这一行存到 `results/runs.pending.csv`，事后手动并回去。
- 以后只会在表末尾加新列。旧格式的 `runs.csv` 会自动迁移；如果文件里出现训练器不认识的列，训练开始前就会报错。
- `.gitattributes` 把 `runs.csv` 设成 union 合并：两个人各自追加的行在 git merge 时都会保留，不会冲突（行的先后可能交错，用 `run_id` 区分）。例外是训练器加了新列、某一边的表头被自动迁移过，那次合并要手动检查。

## 6. 常用配置字段

| 字段 | 默认值 | 说明 |
|---|---|---|
| `exp_name` / `member` / `task` / `notes` | debug / "" / dvc / "" | 记录用；`task` 取 `dvc`、`cifar10` 或 `cifar10_imb` |
| `model` / `num_classes` | resnet18 / 2 | 注册表里的模型名 |
| `pretrained` | **false** | 用 ImageNet 预训练权重时必须显式写 `true` |
| `freeze` | none | `none` / `backbone`（线性探测）/ `partial`（只训最后一个 stage 和 head） |
| `optimizer` / `lr` / `lr_backbone` / `momentum` / `wd` | sgd / 0.01 / null / 0.9 / 5e-4 | `lr_backbone` 设置后骨干和头分开设学习率；bias 和 BN 不加 weight decay |
| `scheduler` / `warmup_epochs` / `min_lr_ratio` | cosine / 0 / 0 | 也可选 `step`（配 `step_epochs`、`gamma`）、`onecycle`、`none`；都按 batch 更新 |
| `epochs` / `bs` / `patience` | 10 / 64 / null | `patience` 设了就按 val acc 早停 |
| `label_smoothing` / `grad_clip` | 0 / null | |
| `loss` / `sampler` / `imb_ratio` | ce / none / null | 不平衡实验用，见第 3 节 |
| `aug` / `img_size` / `n_train` / `norm_mean` / `norm_std` | flip / 224 / null / ImageNet 均值方差 | 由 loader 使用，训练器只负责记录；猫狗的 `aug` 只用约定的 `none` / `flip` / `flip_rot` / `full` |
| `amp` / `channels_last` / `deterministic` | true / false / false | 后两项只在 CUDA 上生效；`channels_last` 见下 |
| `num_workers` / `device` | 4 / auto | |
| `out_dir` / `results_csv` / `overwrite` | runs / results/runs.csv / false | |

- YAML 里 `1e-3`、`null` 这些写法都能正确解析。字段名写错会直接报错，不会被悄悄忽略。
- `channels_last` 默认是 `false`（11/1 起；之前是 `true`）。在这台机器上（torch 2.8 + cuDNN 9.10），channels_last 下 cuDNN 训练模式的 BatchNorm 特别慢，会让带可训练 BN 的模型整体慢 5～10 倍（R18@224 实测约 200 vs 1150 img/s），从零训练的 CNN 和微调都受影响。冻结骨干的线性探测反而快约 25%，那种 config 可以手动写 `channels_last: true`。
- 如果不用 `run_experiment`、自己调用 `fit()`：要先 `set_seed(cfg.seed)` 再建模型和 loader（否则会警告结果不可复现），模型也必须用 `build_model()` 创建。

## 7. 复现性

- 训练前先把代码 commit，checkpoint 里才会记录 SHA。没有提交时记为 `no-commit`；`src/`、`scripts/`、`configs/`、`tests/`、`splits/` 下有未提交改动时记为 `<sha>-dirty`，还没 `git add` 的新文件（比如新写的模型或 config）也算。
- 固定 seed 后，同一台机器上的结果基本可复现：冒烟测试重跑两次，验证集指标完全一致。cuDNN 和 AMP 仍可能带来极小差异，所以报告里的说法是"固定种子并对 3 个 seed 取平均"，不承诺逐位一致。
