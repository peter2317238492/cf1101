# EE6483 Mini Project（Option 2）· 角色 B 任务依赖与计划

- 截止：**2026-11-13（周五）23:59**，报告和 `submission.csv` 通过 NTULearn 交给 TA
- 要求原文：[EE6483-Project2.pdf](EE6483-Project2.pdf)、[EE6483_project_briefing.pdf](EE6483_project_briefing.pdf)、[sampleSubmission.csv](sampleSubmission.csv)
- 数据：`data/dogs-vs-cats/datasets/{train,val}/{cat,dog}`（每类训练集 10000 张、验证集 2500 张）、`datasets/test/1..500.jpg`；CIFAR-10 存放在 `data/cifar10/`
- B 负责：Q4（超参，5 分）、Q6（错误样本分析，10 分）、Q7（不同模型和数据处理的对比，10 分）、Q8（CIFAR-10，10 分），代码方面负责训练器核心、传统方法对照、预训练骨干与调参、误差分析、CIFAR-10 分类器

## 结论

**B 大部分工作不用等 A 和 C，反而是 A 和 C 在等 B。**

- A 的从零训练 CNN、增强消融、多 seed 终跑都要用 B 写的**训练器核心**。
- C 的 Q9（CIFAR-10 不平衡）要在 B 的 **CIFAR-10 分类器**基础上做。
- B 真正要等组员的只有 3 件事：
  1. Q6 的最终分析：要等全组确定最终提交模型
  2. Q7 中"数据处理"那一半：要等 A 的增强消融结果
  3. Q4 中从零训练 CNN 那部分调参依据：要等 A 的调参记录

其余工作今天就能开始。依赖 A 数据管线的部分，先用 `ImageFolder` 直接读取，加一个按约定规则自己生成的临时划分，等 A 的版本好了再替换。

## ✅ 现在就能做（完全没有依赖）

| 任务 | 对应题目 | 产出 | 工时 |
|---|---|---|---|
| 和 A、C 开 30 分钟接口会，把下面的"今天要约定的接口"写成 `CONTRACTS.md`；建 git 仓库骨架；确认谁有 GPU | 协调 | CONTRACTS.md、仓库 | 1.5h |
| 补环境：`pip install scikit-learn scikit-image pandas matplotlib pyyaml`；预下载 ResNet-18/50 的 ImageNet 权重；做一次 GPU+AMP 冒烟测试 | 环境 | 依赖清单、权重缓存 | 0.5h |
| 模型注册表 `build_model(name, num_classes, pretrained, freeze)`：ResNet-18/50、CIFAR 版 ResNet-18，冻结策略分 none/backbone/partial 三种；checkpoint 读写 | 训练器核心 | `src/models/registry.py` | 1.5h |
| **给 500 张测试图人工打标签**（cat/dog/ambiguous）。Q6 要分析测试集上分对和分错的样本，但测试集没有标签；只用于误差分析，不用于训练或调参 | Q6 | `analysis/test_manual_labels.csv` | 1.5h |
| CIFAR-10 数据模块：从 50k 训练集按 seed 42 分层划出 45k/5k，划分索引存成文件，和 C 共用 | Q8 | `src/cifar/data.py`、`splits/` | 1.5h |
| 把计划使用的模型清单和改进技术清单交给 C，C 用来写 Q1 的"基线选择与改进动机" | 支撑 Q1 | 一页清单和 BibTeX | 0.5h |

## 🟡 现在开始做，先用临时替代品，等组员的东西到了再替换

| 任务 | 对应题目 | 先用什么顶上 | 什么时候替换 | 工时 |
|---|---|---|---|---|
| **训练器核心** `trainer.py`：set_seed、AMP、SGD/AdamW、cosine+warmup、早停、best.pt、每类准确率和混淆矩阵、可注入 criterion（C 的 Q9 要用）、结果写入 `results/runs.csv`。**11/2 冻结 API 并通知 A** | 训练器核心 | `torchvision FakeData` 或 ImageFolder 小子集 | A 的 loader v1 到了跑一次冒烟测试，trainer 本身不用改 | 4.5h |
| LR range test 和超参扫描、多 seed 启动器 | Q4 | 用 FakeData 加小网络跑通流程 | trainer v1 完成后直接用；A 的多 seed 终跑也可以用 | 1.5h |
| 传统方法对照：HOG+颜色直方图 → LogReg/LinearSVC/RandomForest（RBF-SVM 只在子集上跑）；只用 CPU，可以和 GPU 任务并行 | Q7 | 按约定规则自己生成的临时 split | A 提交正式 split 后用 CPU 重跑 | 3.5h |
| 预训练骨干首轮：ResNet-18 依次做线性探测、部分微调、全量微调，输入 224 | Q4/Q5 | ImageFolder 直读（断言 cat=0、dog=1），配最小 transform | 替换成 A 的 loader | 2.5h |
| 误差分析工具：自己实现 Grad-CAM（用 hook 实现，不新增依赖）、高置信对/错和边界样本网格、错因分类表 | Q6 | 未微调的 ImageNet ResNet 加伪造的预测 CSV | 11/7 最终模型定下来后一键重跑 | 2.5h |
| `aggregate.py`：从 runs.csv 自动生成 mean±std 表和图 | Q7 | 伪造一份 runs.csv | 真实结果写入后重跑 | 1.5h |
| CIFAR-10 基线：把猫狗方案原样迁移过来（预训练 ResNet-18，输出改为 10 类），作为"改进前"的参照 | Q8 | 用 trainer v0 | 如果最终选的是 A 的 CNN，补跑一行 | 2h |
| 给 C 的不平衡实验入口 `train_cifar.py`：今天先把命令行参数约定发给 C，**11/4 交付可运行版**；loss/sampler 用注册表实现，B 写 ce/none，C 加 weighted_ce/focal/重采样 | Q8 到 Q9 的接口 | 先接 FakeData，C 的不平衡索引先用占位文件 | 11/6 冻结 `cifar_final.yaml` | 1h |
| 报告骨架：Q4 参数表模板、Q7 对比表模板、Q8 中不依赖结果的部分（数据集介绍、"猫狗 → CIFAR"改动对照表） | 报告 | 数字先留占位 | 出结果后填 | 1h |

## ⏳ 必须等（等组员，或等 B 自己前面的步骤）

| 任务 | 对应题目 | 在等什么 | 等谁 | 工时 |
|---|---|---|---|---|
| Q4 在正式 split 上做调参扫描和最终配置的 3 个 seed（粗扫可以先在临时 split 上做） | Q4 | trainer v1；**A 的 loader v1 和正式 split（约 11/2）** | A，以及 B 自己 | 3h |
| 交 checkpoint 给 A：11/3 交线性探测版，用来打通提交流程；11/7 按"正式 val 上 3 seed 均值最高"的规则冻结最终模型 | 支撑 Q5 | **A 的从零训练 CNN 3 seed 结果**，参与最终模型选择 | A | 1h |
| CIFAR 改进消融：CIFAR 版 stem、Crop+Flip+Cutout/TrivialAugment、SGD+cosine+label smoothing。**11/6 硬冻结后交给 C** | Q8 | CIFAR 基线（B 自己） | B 自己 | 2.5h |
| 在 CIFAR-10 官方 10k 测试集上做最终评估（3 个 seed，只评估一次），平衡数据下的数值同时交给 C 做对照 | Q8 | `cifar_final.yaml` 冻结 | B 自己 | 1.5h |
| **Q6 最终误差分析和写作**：用最终提交模型对测试集推理，结合人工标签挑 1–2 个案例，配 Grad-CAM 讨论优缺点 | Q6 | **11/7 冻结的最终模型和 A 的 submission.csv（约 11/8）** | A（最终模型由 A、B 共同决定） | 2.5h |
| 写 Q4：每个参数列出候选值、最终选择和依据（实验、资源约束或文献） | Q4 | 扫描日志；**A 的从零训练 CNN 调参记录**；C 的 BibTeX | A、C | 1.5h |
| 写 Q7：模型维度（传统方法、从零 CNN、线性探测、各种微调、R18 与 R50 对比）和数据处理维度 | Q7 | **A 的增强消融结果（约定最晚 11/8 写入 runs.csv）** | A | 2h |
| 写 Q8：数据集、方法、改动、测试集结果和混淆矩阵 | Q8 | CIFAR 测试集结果 | B 自己 | 1.5h |
| B 负责脚本的运行说明，并入 README（Q3 要求可复现） | 支撑 Q3 | 代码冻结；README 结构由 A 定 | A | 0.5h |
| 交稿：11/10 把 Q4/6/7/8 正文、图表和贡献说明交给 C；11/11–12 审读整合稿 | 报告 | C 的模板和整合稿 | C | 1h |

## B 是谁的上游（这些不能拖）

- **A**：trainer v1（11/2），A 的 Q2、Q3、Q5 共 35 分都建立在它上面；还有模型注册表、ckpt 格式、predict 接口，以及 11/3 和 11/7 两次交付的 ckpt
- **C**：CIFAR 的 45k/5k 划分；`train_cifar.py` 和平衡基线（11/4）；`cifar_final.yaml`（11/6）；模型和技术清单（Q1 要用）；11/10 的四个章节

全队最长链是：trainer v1（11/2）→ CIFAR 接口（11/4）→ cifar_final（11/6）→ C 的 Q9 实验和写作 → C 整合 → 11/13 提交，**浮动只有约 1 天**。时间不够时，优先砍 RBF-SVM、EfficientNet、mixup，不要拖上面几项交付。

## 今天要和 A、C 约定的接口（写进 CONTRACTS.md）

1. **标签和提交格式**：cat=0、dog=1，代码里断言 `class_to_idx == {'cat':0,'dog':1}`；测试集 id 取文件名中的数字，**按数值排序**为 1..500；`submission.csv` 只有 `id,label` 两列共 500 行，由 A 生成并用脚本校验。
2. **划分规则**：每类文件名排序后用 `random.Random(42)` 抽样；`splits/dvc_train.csv`、嵌套子集 `n2000`/`n5000`、`dvc_val.csv`（所有方法共用），列为 `path,label`。A 提交之前，B 按同一规则自己生成临时版。
3. **预处理**：统一 `convert('RGB')`；短边预缩放到 256（不能只缩到 128，否则预训练模型用不了 224 输入）；统一用 ImageNet mean/std；增强预设名固定为 `none / flip / flip_rot / full`。
4. **数据接口**（A 实现，约 11/2）：`build_dvc_loaders(cfg) -> {'train','val','test'}`，test 批次返回 `(x, id)`。
5. **Trainer API**（B 实现，11/2 冻结，之后只增不改）：`fit / evaluate / predict / set_seed`，cfg 字段名固定，seed 统一用 `[0,1,2]`，结果报告 mean±std。
6. **模型和 ckpt**：`build_model(name, num_classes, pretrained, freeze)`，A 的 CNN 也注册进同一张表；ckpt 里保存 model_name、cfg、class_to_idx、img_size、归一化参数、seed 和 git commit。
7. **结果记录**：每个 run 都写入 `results/runs.csv`，列名固定。Q3、Q4、Q7、Q9 的表和图都由脚本生成，不手抄数字。
8. **CIFAR**：B、C 共用 45k/5k 划分；C 的不平衡子集只从这 45k 中采样；10k test 只在最终评估时用一次。
9. **GPU**：如果全组只有这台 8GB 笔记本，用排队表管理：白天调试，夜间长跑，同一时间只跑一个训练。
10. **里程碑**：11/2 trainer v1 和数据管线 v1 → 11/4 CIFAR 接口交 C → 11/6 冻结 CIFAR 配置 → 11/7 冻结最终模型 → 11/8 A 交增强消融结果和 submission.csv → 11/10 各章节初稿 → 11/12 终稿 → 11/13 中午前提交。

## 建议时间线（B）

| 日期 | 重点 |
|---|---|
| 10/31 周六 | 开接口会，写 CONTRACTS.md，建仓库；装依赖，预下载权重；把 train_cifar 的参数约定发给 C；trainer v0 用 FakeData 跑通 |
| 11/1 周日 | trainer v1 和模型注册表；CIFAR 数据模块和划分；晚上给测试集人工打标签；夜间跑 CIFAR 冒烟测试 |
| 11/2 周一 | **冻结 trainer API 并通知 A**；LR finder 和扫描器；预训练首轮；夜间跑 CIFAR 基线 |
| 11/3 周二 | 线性探测 ckpt 交 A；train_cifar 可运行；传统方法在 CPU 上开跑；夜间跑 Q4 粗扫 |
| 11/4 周三 | **train_cifar 和平衡基线交 C**；切换到 A 的正式 split；跑 Q4 细扫；夜间跑 CIFAR 改进第一批 |
| 11/5 周四 | 训练量和分辨率消融；传统方法跑完；Grad-CAM 和 aggregate.py；夜间跑 CIFAR 改进第二批 |
| 11/6 周五 | **冻结 cifar_final 交 C**；搭报告骨架；夜间跑 Q4 最终配置的 3 个 seed |
| 11/7 周六 | **和 A 冻结最终提交模型**；写 Q4；夜间跑 CIFAR 3 seed 并做测试集评估 |
| 11/8 周日 | 接收 A 的增强消融结果和 submission.csv；Q6 推理、Grad-CAM、挑案例 |
| 11/9 周一 | 写 Q6、Q8、Q7；运行说明并入 README |
| 11/10 周二 | **四个章节和贡献说明交 C** |
| 11/11–12 | 审读整合稿，核对 submission.csv 和报告中的数字 |
| 11/13 周五 | 中午前提交，B 待命 |

## 主要风险

- **GPU 只有一块 8GB 笔记本卡**：开 AMP，长任务放夜间排队，粗扫用小子集；必要时把 C 的实验放到 Colab 或 Kaggle 跑。
- **预训练模型在猫狗上几乎饱和（超过 98%）**：Q4 调参曲线会很平，需要配合从零 CNN 的调参证据；Q7 要讨论 ImageNet 本身包含猫狗类别带来的先验优势。
- **用 val 既选超参又报告 val 准确率**会有乐观偏差：报告多 seed 的 mean±std 并说明这一点；CIFAR 的 10k test 只评估一次。
- **测试集人工标签**：只能用于 Q6 误差分析，报告中要声明没有用于训练或选模。
- **Windows 环境**：DataLoader 需要 `if __name__ == '__main__'` 保护；JPEG 解码是瓶颈，预缩放后缓存；不要用 `torch.compile`。
- **提交文件出错**：id 按字符串排序会整体错位；标签映射写反会让预测整体反转。代码里加断言，再用脚本对照 sampleSubmission 校验。
