# 训练器使用说明（给 A、C）

代码在 `src/trainer.py`（训练器）和 `src/models/registry.py`（模型注册表、checkpoint 读写）。

```bash
python -m pytest -q                                            # 73 个单元测试，约 15 秒
python -m scripts.smoke_dvc --n-train 1000 --n-val 500 --epochs 2   # 真实猫狗数据的 GPU 冒烟测试
```

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
    cfg = TrainConfig.from_yaml("configs/dvc_r18_linear_probe.yaml", seed=0)
    res = run_experiment(cfg, build_loaders)
    print(res["best_val_acc"], res["ckpt_path"])
```

- 训练集和验证集的 loader 产出 `(x, label)`；交给 `predict` 的 loader 产出 `(x, id)`。
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
- 展平特征图用 `torch.flatten(x, 1)`、`nn.Flatten()` 或 `reshape`，**不要用 `.view`**。GPU 上训练器用 channels_last 格式，`.view` 会直接报错。`python -m pytest -q` 会自动检查所有已注册模型。
- `head` 写分类层的属性名，`freeze` 和 `lr_backbone` 靠它区分头和骨干。
- `input_size` 只有在全连接层要求固定输入尺寸时才写。

## 3. C：CIFAR 不平衡实验

| 方法 | 实现方式 | cfg 里要写的字段 |
|---|---|---|
| 加权 CE | `criterion=nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)` | `loss: weighted_ce` |
| Focal loss | 自己写一个 `nn.Module`，里面的张量用 `register_buffer` 存（这样 `.to(device)` 才会带过去） | `loss: focal` |
| 重采样 | 在 `build_loaders` 里用 `WeightedRandomSampler(w, len(w), generator=torch.Generator().manual_seed(cfg.seed))`，同时去掉 `shuffle` | `sampler: weighted` |
| 不平衡子集 | `Subset(train_45k, imb_indices)`，索引从 `splits/` 读 | `imb_ratio: 0.1` |

```python
res = run_experiment(cfg, build_loaders, criterion=my_loss, eval_test=True)  # loaders 里要有带标签的 "test"
```

- `eval_test=True` 时，用最好那个 epoch 的模型在 test 上只评一次。结果写进 `runs.csv` 的 `test_acc` / `test_macro_f1` 两列，以及 `metrics.json` 的 `test_per_class_acc` / `test_confusion`。
- 以下两种情况会直接报错，防止记录和实际训练对不上：`cfg.loss` 不是 `ce` 却没传 `criterion`；传入的 criterion 的 label smoothing 和 `cfg.label_smoothing` 不一致。
- 验证集和测试集的 loss 一律用普通 CE 计算，所以不同 loss 的实验之间可以直接比较。

## 4. 预测并生成 submission.csv（A）

```python
from src.models.registry import load_ckpt
from src.trainer import predict

model, ckpt = load_ckpt("runs/<exp>/seed0/best.pt", map_location="cuda")
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

- 同一个 `exp_name` 加 `seed` 已经完整跑过的话，再跑会直接报错，不会覆盖。确实要重跑就在 cfg 里加 `overwrite: true`。
- **不要用 Excel 直接打开 `runs.csv`**，Excel 会锁住文件。要看的话用 pandas、VS Code，或者先复制一份。万一写入失败，训练器会重试约 30 秒；还是失败就把这一行存到 `results/runs.pending.csv`，事后手动并回去。
- 以后只会在表末尾加新列。旧格式的 `runs.csv` 会自动迁移；如果文件里出现训练器不认识的列，训练开始前就会报错。

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
| `aug` / `img_size` / `n_train` / `norm_mean` / `norm_std` | flip / 224 / null / ImageNet 均值方差 | 由 loader 使用，训练器只负责记录 |
| `amp` / `channels_last` / `deterministic` | true / true / false | 后两项只在 CUDA 上生效 |
| `num_workers` / `device` | 4 / auto | |
| `out_dir` / `results_csv` / `overwrite` | runs / results/runs.csv / false | |

- YAML 里 `1e-3`、`null` 这些写法都能正确解析。字段名写错会直接报错，不会被悄悄忽略。
- 如果不用 `run_experiment`、自己调用 `fit()`：要先 `set_seed(cfg.seed)` 再建模型和 loader（否则会警告结果不可复现），模型也必须用 `build_model()` 创建。

## 7. 复现性

- 训练前先把代码 commit，checkpoint 里才会记录 SHA。没有提交时记为 `no-commit`；`src/`、`scripts/`、`configs/`、`tests/` 下有未提交改动时记为 `<sha>-dirty`。
- 固定 seed 后，同一台机器上的结果基本可复现：冒烟测试重跑两次，验证集指标完全一致。cuDNN 和 AMP 仍可能带来极小差异，所以报告里的说法是"固定种子并对 3 个 seed 取平均"，不承诺逐位一致。
