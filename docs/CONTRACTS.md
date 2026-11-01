# CONTRACTS.md · 全组接口约定（草案，待 A、C 确认）

- 起草：B，2026-11-01，基于 `master` `c2cf275`。用法细节见 [trainer_api.md](trainer_api.md)，排期见 [B_任务依赖与计划.md](B_任务依赖与计划.md)
- 状态标记：✅ 已实现（写明位置）；🔲 待实现（写明谁、何时）
- **待确认·A / 待确认·C / 待确认·全员**：计划里没写、由 B 先提出的默认值。对应的人回复之前不算定论。请直接回答文末的两张清单
- ✅ 条目里的名字、路径、列名以代码为准；发现本文件和代码不一致，以代码为准，并在群里指出。标 `fb26e20` 的是 11/1 新增、已经提交并测过的内容（合并进 master 并 push 之后，pull 下来就有）

## 0. 变更规则

1. **只增不改**：冻结后的函数签名、`TrainConfig` 字段名、`runs.csv` 列名、预设名、文件路径只能新增，不能改名、删除或改变含义。新增的参数和字段必须有默认值，旧 config 不改也能照常运行。
2. **改约定先在群里说**：任何改动，只要涉及本文件或上面这些名字，都先在群里说明改什么、影响谁，然后再提交代码。
3. **CIFAR-10 官方 10k test 只用一次**：只在配置冻结后的最终评估中使用（`--eval-test` / `eval_test=True`），不能用它调参、选模型或挑 seed。

## 1. 标签和提交格式

| 约定 | 状态 |
|---|---|
| cat=0、dog=1。`task: dvc` 时训练器检查 `class_to_idx == {'cat': 0, 'dog': 1}`，不一致直接报错 | ✅ `src/trainer.py`：`DVC_CLASS_TO_IDX`、`_resolve_class_to_idx` |
| 测试集 id 取文件名中的数字（`test/1.jpg` → `1`）。test loader 给出的 id 只能是 int 或纯数字字符串（`"1"`，不能是 `"1.jpg"`）；`predict` 会把纯数字字符串转成 int | `predict` ✅；loader 一侧 🔲 A |
| `submission.csv`：表头正好是 `id,label`，共 500 行；id 按**数值**升序排列为 1..500，和 `docs/sampleSubmission.csv` 的 id 列逐行一致；label ∈ {0,1}，1=dog | 🔲 A：11/3 用 B 的线性探测 ckpt 打通流程，11/8 交最终版 |
| 校验脚本 `scripts/check_submission.py <csv>`：逐项检查上一行的要求，不通过时以非零码退出 | 🔲 A；路径和名字 **待确认·A** |
| 最终文件放在仓库根目录，文件名 `submission.csv` | **待确认·A** |

## 2. 猫狗划分规则

计划已定：每类文件名排序后用 `random.Random(42)` 抽样；划分文件包括 `splits/dvc_train.csv`、嵌套子集 `n2000` / `n5000`，以及所有方法共用的 `splits/dvc_val.csv`，列为 `path,label`。🔲 A 约 11/2 提交正式文件。

| 细节（计划没写，以下是 B 的提议） | 状态 |
|---|---|
| "排序"指 Python `sorted()` 的字符串序（`cat.10.jpg` 排在 `cat.2.jpg` 前面），和 `ImageFolder` 一致 | **待确认·A** |
| 每类各新建一个 `random.Random(42)`，用它 `shuffle` 该类排好序的列表；`n{N}` 取每类 shuffle 后的前 N/2 张，因此 n2000 ⊂ n5000 ⊂ dvc_train | **待确认·A** |
| 文件名：`splits/dvc_train.csv`（train 全部 20000 张）、`splits/dvc_train_n2000.csv`、`splits/dvc_train_n5000.csv`、`splits/dvc_val.csv`（val 全部 5000 张） | **待确认·A** |
| `path` 是相对 `data/dogs-vs-cats/datasets/` 的路径，用正斜杠，例如 `train/cat/cat.0.jpg`；`label` 取 0/1；行按 `path` 排序 | **待确认·A** |
| LF 换行，各平台上字节一致 | ✅ `.gitattributes`：`splits/*.csv text eol=lf` |
| `cfg.n_train`：`null` 表示用 `dvc_train.csv` 全部；任意偶数 N 都按上面的规则取每类前 N/2 张，结果和 `dvc_train_n{N}.csv` 一致（n2000、n5000 两个文件就是这条规则写到盘上的结果）；奇数或超出范围时报错 | **待确认·A** |
| A 交付之前的临时版：规则同上，只在内存里生成，不写文件 | ✅ `fb26e20`：`src/dvc_standin.py` 的 `balanced_subset`；正式结果一律改用 A 的文件 |

## 3. 预处理（猫狗）

| 约定 | 状态 |
|---|---|
| 读图后统一 `convert('RGB')` | 计划已定；🔲 A |
| 短边预缩放到 256（不能只缩到 128，否则预训练模型用不了 224 输入）。在线 Resize 还是写盘缓存由 A 决定，不影响接口 | 计划已定；🔲 A |
| 归一化统一用 ImageNet 均值方差：`IMAGENET_MEAN` / `IMAGENET_STD`，它们也是 `TrainConfig.norm_mean` / `norm_std` 的默认值。loader 从 cfg 读取，不要写死 | 常量 ✅ `src/trainer.py`；使用 🔲 A |
| 验证和测试 transform：`Resize(int(img_size * 256 / 224))` → `CenterCrop(img_size)` → `ToTensor` → `Normalize`（img_size=224 时就是先缩到 256 再裁 224） | B 的临时版已经这样做（`src/dvc_standin.py` 的 `eval_transform`，`fb26e20`）；**待确认·A** |
| `cfg.aug` 的预设名固定为 `none / flip / flip_rot / full`，默认 `flip`，只作用于训练集 | 名字 ✅（`TrainConfig.aug`，`docs/trainer_api.md` §6）；实现 🔲 A，约 11/2。B 的临时版只实现了 `none` 和 `flip`（flip 加在 CenterCrop 之后） |

各预设的具体内容由 A 实现。下表是 B 的提议，**待确认·A**：

| 预设 | 训练 transform（`ToTensor` + `Normalize` 之前） |
|---|---|
| `none` | 和验证 transform 相同 |
| `flip` | `none` 加 `RandomHorizontalFlip(0.5)` |
| `flip_rot` | `flip` 加 `RandomRotation(15)` |
| `full` | `RandomResizedCrop(img_size, scale=(0.6, 1.0))` + `RandomHorizontalFlip(0.5)` + `RandomRotation(15)` + `ColorJitter(0.2, 0.2, 0.2)` |

## 4. 猫狗数据接口（A 实现，约 11/2）

计划已定：`build_dvc_loaders(cfg) -> {'train', 'val', 'test'}`。train、val 的批次是 `(x, label)`，test 的批次是 `(x, id)`。

| 细节 | 状态 |
|---|---|
| 模块放在 `src/dvc/data.py`，用 `from src.dvc.data import build_dvc_loaders` 导入（和 `src/cifar/data.py` 对应） | **待确认·A** |
| 签名扩展为 `build_dvc_loaders(cfg, root=None)`。数据根目录的优先级：`root=` 参数 > 环境变量 `DVC_ROOT` > `data/dogs-vs-cats/datasets`（和 `CIFAR10_ROOT` 的规则相同；在 worktree 里运行时需要） | B 的临时版用的是同一规则（`fb26e20`）；A 的版本 **待确认·A** |
| 返回的 dict 里另外带 `'class_to_idx': {'cat': 0, 'dog': 1}`（或者让 dataset 带 `.class_to_idx` 属性）。两者都没有时，训练器只发警告，不检查标签映射 | 训练器行为 ✅；B 的临时版会返回这个键；A 的版本 **待确认·A** |
| x 是 float32 `[B, 3, S, S]`，S = `cfg.img_size`；loader 读取 `cfg.img_size / aug / n_train / norm_mean / norm_std / bs / num_workers` | 这些字段 ✅ 已在 `TrainConfig` 中；读取逻辑 🔲 A |
| train loader 设 `shuffle=True` 并传入 `**loader_kwargs(cfg.seed)`；val、test 设 `shuffle=False` | `loader_kwargs` ✅ `src/trainer.py`；🔲 A |
| 猫狗的 `'test'` 没有标签，所以 `task: dvc` **不能**用 `run_experiment(..., eval_test=True)`。预测一律走 `predict` | ✅ `fb26e20`：`run_experiment` 在 `task: dvc` 加 `eval_test=True` 时训练前就报错 |
| B 的猫狗训练入口 `scripts/train_dvc.py`（`--config` `--seeds` `--set` `--data-root`）目前接的是临时 loader `build_standin_loaders`（`src/dvc_standin.py`），这些 run 的 `notes` 带 "stand-in loader"；A 的 `build_dvc_loaders` 到了以后改接 A 的函数 | ✅ `fb26e20` |

## 5. Trainer API（✅ `src/trainer.py`，11/2 冻结，之后只增不改）

```python
set_seed(seed, deterministic=False)
loader_kwargs(seed)                        # -> {"worker_init_fn", "generator"}，传给 train DataLoader
run_experiment(cfg, build_loaders, criterion=None, eval_test=False)   # -> fit 的返回值 + "model"
fit(model, train_loader, val_loader, cfg, criterion=None, test_loader=None, class_to_idx=None)
    # -> best_val_acc, best_epoch, ckpt_path, history, val_metrics, test_metrics, run_dir, run_id
evaluate(model, loader, device=None, amp=True, channels_last=False)
    # -> acc, loss, per_class_acc, confusion, macro_f1, probs, preds, labels
predict(model, loader, device=None, amp=True, channels_last=False)
    # -> DataFrame[id, prob_0..prob_{K-1}, pred]
TrainConfig.from_yaml(path, **overrides) / TrainConfig.from_dict(d) / cfg.to_dict()
TrainConfig.parse_overrides(["lr=1e-3", ...])   # 命令行 --set 的解析，两个训练脚本共用（新增，`fb26e20`）
check_new_run(cfg)                         # 训练前检查 exp_name+seed 是否已跑完（新增，`fb26e20`；fit 内部也会调用）
```

`TrainConfig` 共 41 个字段，字段名写错会直接报错：

| 组 | 字段 |
|---|---|
| 记录 | `exp_name` `task`（dvc / cifar10 / cifar10_imb） `member` `notes` `overwrite` |
| 模型 | `model` `num_classes` `pretrained` `freeze`（none / backbone / partial） |
| 数据 | `aug` `img_size` `n_train` `norm_mean` `norm_std` `sampler` `imb_ratio` |
| 损失 | `loss` `label_smoothing` |
| 优化 | `optimizer`（sgd / adamw） `lr` `lr_backbone` `momentum` `wd` `bs` `epochs` `scheduler`（cosine / step / onecycle / none） `warmup_epochs` `min_lr_ratio` `step_epochs` `gamma` `grad_clip` `patience` |
| 运行 | `amp` `channels_last` `seed` `deterministic` `num_workers` `device`（auto / cuda / cpu） `log_every` `out_dir` `results_csv` |

- 多 seed 统一用 `[0, 1, 2]`，报告 mean±std（计划已定）。
- ✅ `cfg.loss` 不是 `ce` 却没有传 `criterion` 时报错；criterion 的 `label_smoothing` 和 `cfg.label_smoothing` 不一致时报错。从 `fb26e20` 起，反过来，`cfg.loss` 还是 `ce` 却传了加权 CE 或自定义 loss 时也报错：runs.csv 记的是 `cfg.loss`，名字必须和实际用的一致。`sampler` 同理，见 8.2。val 和 test 的 loss 一律按普通 CE 计算。
- ✅ 训练前先 commit，ckpt 里才会记录正确的 SHA（`src/` `scripts/` `configs/` `tests/` `splits/` 下有未提交改动或没 `git add` 的新文件时记为 `<sha>-dirty`；`splits/` 这一条从 `fb26e20` 起）。Windows 下 `num_workers>0` 的脚本必须写 `if __name__ == "__main__":`。
- ✅ `fb26e20`：`channels_last` 默认改为 `false`。这台机器上（cuDNN 9.10）channels_last 会让带可训练 BatchNorm 的模型训练慢 5～10 倍；只有冻结骨干的线性探测开了更快。只改默认值，字段名不变，旧 config 照常运行。

## 6. 模型和 checkpoint（✅ `src/models/registry.py`）

| 约定 | 状态 |
|---|---|
| `build_model(name, num_classes, pretrained=False, freeze="none")`，`freeze` 取值见 `FREEZE_MODES = ("none", "backbone", "partial")` | ✅ |
| 注册方式：`@register(name, head=("fc",), partial=(), input_size=None)`，builder 签名为 `(num_classes, pretrained=False) -> nn.Module`；`src/models/` 下的任何模块都会自动导入 | ✅ |
| 已注册的模型：`resnet18`、`resnet50`、`efficientnet_b0`、`cifar_resnet18`（`input_size=32`） | ✅ |
| `forward` 输出 logits；展平用 `torch.flatten(x, 1)`、`nn.Flatten()` 或 `reshape`，不能用 `.view`（`python -m pytest -q` 会检查所有已注册模型） | ✅ |
| ckpt 保存在 `runs/{exp_name}/seed{seed}/best.pt`，包含这些键：`model_name` `num_classes` `state_dict` `cfg` `class_to_idx` `img_size` `norm_mean` `norm_std` `seed` `git_commit` `run_id` `criterion` `val_acc` `epoch` | ✅ 由 `fit` 调用 `save_ckpt` 写入 |
| `load_ckpt(path, map_location="cpu") -> (model, ckpt)`，返回的模型在 CPU 上、处于 eval 模式、没有冻结任何层 | ✅ |
| A 的从零训练 CNN 注册进同一张表，名字 `scratch_cnn_v1`，文件 `src/models/scratch_cnn.py` | 🔲 A，日期待填；名字和文件 **待确认·A** |
| ckpt 交付：11/3 交线性探测版（`configs/dvc_r18_linear_probe.yaml` → `runs/dvc_r18_linear_probe/seed0/best.pt`），用来打通提交流程；11/7 冻结最终模型，规则是"正式 val 上 3 个 seed 的均值最高" | 11/3 版 ✅：commit `fb26e20`，临时 loader，val 0.9888，run_id `dvc_r18_linear_probe_seed0_20261101-211039-620057`；🔲 B（11/7 的最终模型由 A、B 共同决定） |
| `*.pt` 和 `runs/` 都在 `.gitignore` 里，ckpt 不能通过 git 传。提议：用网盘传 `best.pt`，同时在群里发 `run_id` 和 `git_commit` | **待确认·A** |
| 生成提交文件时，用胜出配置中 val acc 最高的那个 seed 的 `best.pt` | **待确认·A** |

## 7. 结果记录（✅ `src/trainer.py`）

- 每个 run 写入 `runs/{exp_name}/seed{seed}/` 下的 `config.yaml`、`history.csv`、`best.pt`、`metrics.json`。`history.csv` 的列为 `epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr,time_s`。
- 每个 run 向 `results/runs.csv` 追加一行，列由 `RUNS_CSV_COLUMNS` 定义（共 33 列）。新列只会加在末尾，旧表头会自动迁移；表里出现训练器不认识的列时，训练开始前就报错：

```
run_id,member,task,model,pretrained,freeze,aug,img_size,n_train,n_val,optimizer,lr,wd,bs,epochs,scheduler,seed,best_val_acc,best_epoch,val_loss,test_acc,macro_f1,train_time_s,ckpt_path,git_commit,notes,exp_name,loss,sampler,imb_ratio,lr_backbone,label_smoothing,test_macro_f1
```

- ✅ 同一个 `exp_name` 加 `seed` 已经跑完时（本地 run 目录里有 `metrics.json`，或者 `results/runs.csv` 里已有这一行，包括别人提交的），训练前就报错；确实要重跑就设 `overwrite: true`。总表里的旧行不要删（union 合并会把删掉的行带回来）；`aggregate.py` 对同一个 `exp_name` + `seed` 只取最新的 `run_id`。总表这一条检查从 `fb26e20` 起生效。
- ✅ `.gitattributes` 把 `results/runs.csv` 设为 `merge=union`，两边各自追加的行在合并时都会保留。
- ✅ 不要用 Excel 打开 `runs.csv`。写入失败时，这一行会存到 `results/runs.pending.csv`，之后手动并回去。
- 🔲 B `aggregate.py`（11/5）：Q3、Q4、Q7、Q9 的表和图都由脚本从 `runs.csv` 生成，不手抄数字。脚本按 `exp_name` 分组计算 mean±std。

以下是 B 的提议，**待确认·全员**：
- 一个配置对应一个 `exp_name`，3 个 seed 共用；不同配置不能重名，因为 `aggregate.py` 按 `exp_name` 分组。
- 命名以数据集前缀 `dvc_` 或 `cifar_` 开头，后接模型和变体，例如 `dvc_r18_linear_probe`、`cifar_baseline_r18_pt224`、`cifar_imb_focal_r0.1`。
- `member` 填 `A`、`B` 或 `C`。
- 调试和冒烟运行把 `results_csv` 指到 `runs/` 下面（例如 `scripts/smoke_*.py` 用的 `runs/smoke/runs.csv`），不要写进 `results/runs.csv`。
- `results/runs.csv` 和代码一起 commit、push。B 会先把这个文件提交到 master；在那之前已经在本地跑过训练的人，`git pull` 前先把自己的 `results/runs.csv` 挪开，pull 完再把自己的行追加回去，否则 pull 会因为"未跟踪文件会被覆盖"而中止。

## 8. CIFAR-10（B、C 共用）

### 8.1 数据和划分（✅ `src/cifar/data.py`、`splits/`）

| 约定 | 状态 |
|---|---|
| 45k/5k 划分文件：`splits/cifar10_train45k.csv`（45000 行）和 `splits/cifar10_val5k.csv`（5000 行），列为 `index,label`。`index` 是 `data_batch_1..5` 拼接后的位置（0..49999）。按 seed 42 分层划分，每类 4500/500。不要重新划分 | ✅ |
| `read_split("train" / "val") -> DataFrame[index, label]` | ✅ |
| `build_cifar_loaders(cfg, root=None, train_indices=None, splits_dir=None, sampler=None)` 返回 `train` / `val` / `test` 三个 loader，都产出 `(x, label)`，另外附带 `class_to_idx`。`build_cifar_datasets(cfg, root=None, train_indices=None, splits_dir=None)` 返回 dataset；`cifar_loaders(cfg, ds, sampler=None)` 用已有的 dataset 建 loader；`dataset_targets(ds["train"])` 取训练集标签 | ✅（`sampler=`、`cifar_loaders`、`dataset_targets` 从 `fb26e20` 起） |
| 数据根目录的优先级：`root=` 参数 > 环境变量 `CIFAR10_ROOT` > `data/cifar10` | ✅ |
| `train_indices` 必须是 45k 的子集，否则报 `ValueError`；允许重复（用于过采样）；不能和 `cfg.n_train` 同时使用 | ✅ `_train_positions` |
| val 始终是平衡的 5k，test 始终是官方 10k；C 的不平衡子集只能从 45k 中抽取 | ✅ loader 强制执行 |
| CIFAR 的 `cfg.aug` 是另一套名字，即 `AUGMENTATIONS` 的键，目前有 `none`、`crop_flip`。B 为 Q8 的改进新增预设时只增不改 | 已有部分 ✅；新增 🔲 B，11/6 前 |
| `cifar_resnet18` 必须配 `img_size: 32`，否则报错 | ✅ `_check_cfg` |

### 8.2 `scripts/train_cifar.py`（✅ `fb26e20`，11/4 前交付）

```bash
python -m scripts.train_cifar --config configs/cifar_baseline.yaml [--seeds 0 1 2] [--set key=value ...] [--train-indices <csv>] [--root <cifar dir>] [--eval-test]
```

| 参数 | 约定 |
|---|---|
| `--config` | YAML 配置文件 |
| `--seeds` | 要跑的 seed，例如 `0 1 2`；默认用 config 里的 seed。重复的 seed、已经跑完的 seed 在训练前就报错 |
| `--set key=value ...` | 覆盖任意 `TrainConfig` 字段。文字字段（`exp_name` `notes` `loss` `sampler` 等）原样保留，其他值按 YAML 解析；空值表示 null；带空格的值要加引号 |
| `--train-indices <csv>` | CSV 必须有 `index` 列，内容是 50k 官方训练集中的位置，并且必须是 `splits/cifar10_train45k.csv` 的子集。文件名和索引哈希自动追加到 `notes`，例如 `idx=cifar10_imb_step_r0.1.csv@1a2b3c4d (24750)`（`run_cifar` 从 Python 调用时也会记） |
| `--root <cifar dir>` | CIFAR 数据目录 |
| `--eval-test` | 用最好的模型在官方 10k test 上评估一次；只用于最终运行 |

loss 和 sampler 用注册表实现，注册表在 `src/cifar/train.py`：

```python
@register_loss("name")      # builder(cfg, train_labels) -> nn.Module
@register_sampler("name")   # builder(cfg, train_labels) -> Sampler | None
```

- B 提供 loss `ce` 和 sampler `none`。C 加 `weighted_ce`、`focal` 和重采样 sampler，代码可以放在 `src/cifar/` 下的任意模块里；这些模块会被自动导入，不用在别处写注册调用。`train_labels` 是实际训练子集（应用 `--train-indices` 之后）的标签。
- 用哪一个，就在 cfg 里写 `loss: <名字>` / `sampler: <名字>`（也可以用 `--set`）。**这两个名字只有走 `train_cifar.py` / `run_cifar` 才会生效。**
- ✅ 训练器会检查：criterion 的 `label_smoothing` 属性必须等于 `cfg.label_smoothing`（没有这个属性时按 0 算）。另外，criterion 内部的张量要用 `register_buffer` 保存，否则 `.to(device)` 不会把它们带过去；sampler 的 `generator` 为 None 时，由 `src/cifar/data.py` 的 `cifar_loaders` 用 `cfg.seed` 自动设好（同一个 sampler 换 seed 复用时也会重设）。
- 子文件夹里的模块要有 `__init__.py` 才会被自动找到；被自动导入的模块出错时，报错会指明是哪个模块。
- 注意：如果 `cifar_final.yaml` 设了 `label_smoothing > 0`，没有 `label_smoothing` 属性的 focal loss 会直接报错，见 C 清单第 6 条。
- 直接调用 `run_experiment(cfg, partial(build_cifar_loaders, ...), criterion=...)` 仍然可以用，但只有 loss 能通过 `criterion=` 传进去：`cfg.sampler` 不是 `none` 时 `build_cifar_loaders` 会报错，免得 runs.csv 记了一个没用上的采样器。统一用 `train_cifar.py` 最省事。
- B 的交付：11/4 交 `train_cifar.py`、`configs/cifar_baseline.yaml` 和平衡基线结果；11/6 硬冻结 `configs/cifar_final.yaml`；11/7 把平衡数据下 3 个 seed 的 test 结果交给 C 做对照。

### 8.3 C 的不平衡实验（B 的提议，**待确认·C**）

| 项目 | 提议 |
|---|---|
| 索引文件 | `splits/cifar10_imb_<profile>_r<imb_ratio>.csv`，列为 `index,label`，格式和 `cifar10_train45k.csv` 相同（`train_cifar.py` 只读 `index` 列），提交进 git |
| 抽样 | 从 `read_split("train")` 中按类抽取，用 `random.Random(42)`；seed 0/1/2 用同一个文件，子集固定，只换训练 seed |
| `imb_ratio` 的定义 | 少数类每类张数除以多数类每类张数（≤1）。例如 `0.1` 表示少数类每类 450 张、多数类每类 4500 张。`<profile>` 写进文件名和 `notes`，例如 `step`（指定其中几类为少数类）或 `exp`（指数长尾） |
| cfg | `task: cifar10_imb`、`member: C`；`imb_ratio` 用 `--set` 手动设成和文件名一致的值 |
| 重采样 sampler 的名字 | `weighted`（和 `docs/trainer_api.md` §3 的示例一致） |
| 最终评估 | 先在群里定下最终配置清单（方法 × ratio），然后每个配置在 seed 0/1/2 上各跑一次 `--eval-test` |

```bash
python -m scripts.train_cifar --config configs/cifar_final.yaml --seeds 0 1 2 \
    --train-indices splits/cifar10_imb_step_r0.1.csv \
    --set task=cifar10_imb member=C exp_name=cifar_imb_focal_r0.1 loss=focal imb_ratio=0.1
```

## 9. GPU

| 项目 | 状态 |
|---|---|
| B 的笔记本：RTX 3070 Ti Laptop，8GB，两个 GPU 冒烟测试都已跑通 | ✅ |
| A、C 有没有 GPU（型号、显存） | **待确认·A、C** |

以下排队规则是 B 的提议，**待确认·全员**，适用于全组只有这一块卡的情况：
- 同一块卡同一时间只跑一个训练。
- 白天（09:00–23:00）只跑调试和 30 分钟以内的短任务；长任务和多 seed 放到夜间（23:00–09:00）。
- 排队表放在群公告或共享表格里，列为：日期、时段、人、`exp_name`、预计时长、状态。先登记再跑。
- 发生冲突时，关键路径上的任务优先，依次是：11/4 前的 CIFAR 基线、11/6 前的 `cifar_final`、11/7 最终模型的 3 个 seed，然后才是其他任务。
- A、C 的任务有两种跑法：把 YAML 和 commit SHA 交给 B，由 B 夜间代跑；或者自己用 Colab / Kaggle（先 `git clone` 仓库，否则 ckpt 里的 commit 会记成 `no-commit`）。

## 10. 里程碑

| 日期 | 交付 | 谁 → 谁 | 状态 |
|---|---|---|---|
| 11/2 | trainer v1，API 冻结 | B → A、C | ✅ `c2cf275`，加上 `fb26e20` 的补充（`channels_last` 默认值、`parse_overrides`、`check_new_run`、`splits/` 计入 -dirty、criterion 名字检查）；以 `fb26e20` 为冻结版本 |
| 11/2 | 数据管线 v1：`build_dvc_loaders` 加正式 split | A → B | 🔲 A |
| 11/3 | 线性探测 ckpt，用来打通提交流程 | B → A | ✅ `fb26e20` 训练完成，val 0.9888；ckpt 走网盘 |
| 11/4 | `train_cifar.py` 和平衡基线 | B → C | ✅ `fb26e20`；基线 seed 0 val 0.9688（第 8 个 epoch） |
| 11/6 | `configs/cifar_final.yaml` 硬冻结 | B → C | 🔲 B |
| 11/7 | 冻结最终提交模型；完成 CIFAR 3 个 seed 的 test 评估 | A + B；B → C | 🔲 |
| 11/8 | 增强消融结果写入 `runs.csv`；交 `submission.csv` | A → B | 🔲 A |
| 11/9 | B 负责的脚本的运行说明并入 README | B → A | 🔲 B |
| 11/10 | 各章节初稿（B：Q4、Q6、Q7、Q8 和贡献说明） | 全员 → C | 🔲 |
| 11/12 | 终稿（C 整合，全员审读） | C | 🔲 |
| 11/13 中午前 | 在 NTULearn 提交报告和 `submission.csv`（官方截止 23:59）；由谁提交 **待确认·全员** | — | 🔲 |

- 全组最长的链：trainer v1（11/2）→ CIFAR 接口（11/4）→ `cifar_final`（11/6）→ C 的 Q9 实验和写作 → C 整合 → 11/13 提交。浮动时间只有约 1 天。
- README 结构由 A 负责。B 的提议（**待确认·A**）：①环境安装 ②数据放置 ③按题号列出复现命令（Q2–Q9，每人写自己负责的部分）④输出文件说明（`runs/`、`results/runs.csv`）⑤生成并校验 `submission.csv`。

---

## 需要 A 确认

- [ ] 1. 划分时文件名按 Python `sorted()` 的字符串序排序（第 2 节），同意吗？是 / 否
- [ ] 2. 每类各新建一个 `random.Random(42)` 来 shuffle，`n{N}` 取每类前 N/2 张，同意吗？是 / 否
- [ ] 3. 文件名用 `dvc_train.csv`（20000 张）/ `dvc_train_n2000.csv` / `dvc_train_n5000.csv` / `dvc_val.csv`（5000 张），同意吗？是 / 否；不同意请填你的命名：____
- [ ] 4. `path` 相对 `data/dogs-vs-cats/datasets/`、用正斜杠、行按 path 排序，同意吗？是 / 否
- [ ] 5. `cfg.n_train` 接受 `null` 和任意偶数 N（按同一规则现取，结果和 n2000 / n5000 文件一致），同意吗？是 / 否；如果只想支持固定几档，请填：____
- [ ] 6. 四个增强预设的内容按第 3 节的表，同意吗？是 / 否；不同意请填每个预设的 transform：____
- [ ] 7. 验证和测试 transform 用 `Resize(int(img_size*256/224))` + `CenterCrop(img_size)`，同意吗？是 / 否
- [ ] 8. `build_dvc_loaders(cfg, root=None)` 放在 `src/dvc/data.py`，环境变量叫 `DVC_ROOT`，同意吗？是 / 否；不同意请填路径：____
- [ ] 9. 返回的 dict 里带 `'class_to_idx'`，同意吗？是 / 否
- [ ] 10. loader v1 和正式 split 的交付日期：____（计划约 11/2）
- [ ] 11. 从零 CNN 叫 `scratch_cnn_v1`，放在 `src/models/scratch_cnn.py`，同意吗？是 / 否；不同意请填：____；注册进表的日期：____
- [ ] 12. 校验脚本叫 `scripts/check_submission.py`，最终文件是仓库根目录的 `submission.csv`，同意吗？是 / 否
- [ ] 13. ckpt 用什么方式交接：____（网盘 / U 盘 / 其他）；生成提交文件时用胜出配置中 val 最高的那个 seed，同意吗？是 / 否
- [ ] 14. 你的 GPU 型号和显存：____（没有就填"无"）
- [ ] 15. README 结构按第 10 节的提议，同意吗？是 / 否；B 的运行说明最晚哪天交：____（计划 11/9）
- [ ] 16. 第 0 节变更规则、第 7 节记录规则、第 9 节排队规则，同意吗？同意 / 有意见：____
- [ ] 17. 最后由谁在 NTULearn 点提交：____

## 需要 C 确认

- [ ] 1. 不平衡索引文件命名为 `splits/cifar10_imb_<profile>_r<imb_ratio>.csv`，列为 `index,label`，提交进 git，同意吗？是 / 否
- [ ] 2. `imb_ratio` 定义为少数类每类张数除以多数类每类张数（≤1），同意吗？是 / 否
- [ ] 3. 不平衡分布用哪种：____（`step`：请写明哪几类是少数类；或 `exp` 指数长尾）；要跑哪些 ratio：____
- [ ] 4. seed 0/1/2 共用同一个不平衡文件，抽样用 `random.Random(42)`，同意吗？是 / 否
- [ ] 5. 重采样 sampler 的注册名叫 `weighted`，同意吗？是 / 否；不同意请填：____
- [ ] 6. 如果 `cifar_final.yaml` 设了 `label_smoothing > 0`：(a) focal 自己实现 label smoothing 并暴露 `label_smoothing` 属性，还是 (b) Q9 的实验全部用 `--set label_smoothing=0`？选：____
- [ ] 7. Q9 的最终配置清单（方法 × ratio）哪天在群里定下来：____；定下来之后每个配置 × 3 个 seed 只跑一次 `--eval-test`，同意吗？是 / 否
- [ ] 8. 第 8.2 节 `train_cifar.py` 的参数够用吗？是 / 否；还缺什么：____
- [ ] 9. B 在 11/7 给你平衡数据下的 test 结果，来得及吗？是 / 否；最晚什么时候需要：____
- [ ] 10. 你的 GPU 型号和显存：____（没有就填"无"）；如果没有，选哪种方式：B 夜间代跑 / Colab / Kaggle
- [ ] 11. 11/10 交稿用什么格式：____（Word / LaTeX / Overleaf）；最后由谁在 NTULearn 点提交：____
- [ ] 12. 第 0 节变更规则、第 7 节记录规则、第 9 节排队规则，同意吗？同意 / 有意见：____
- [ ] 13. 不平衡实验的 cfg 写 `task: cifar10_imb`、`member: C`，`imb_ratio` 用 `--set` 手动设成和文件名一致的值（第 8.3 节），同意吗？是 / 否
