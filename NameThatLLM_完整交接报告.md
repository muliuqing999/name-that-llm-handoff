# Name That LLM 项目完整交接报告

本报告交接比赛项目的运行环境、下载资产、项目自训模型、Notebook、实验方法、提交成绩与恢复操作。读者可以先看第一章了解当前状态，再根据后续章节定位代码、权重和证据。报告以 2026 年 11 月 16 日本轮只读核验和运行结果为准，早期文档只作为历史记录使用。

这里有两条并行的工作线：一条是在学校双 PPU 上训练和验证项目模型；另一条是把公开竞赛方案放到 Kaggle 双 T4 环境复现和提交。本次优化的是第二条工作线，并没有把学校服务器上的自训模型自动接入最新公开方案。

编写时间：2026-11-16 16:50:48，日本时间。提交历史查询时间：`2026-11-16T07:46:48.916266+00:00`。

## 1 当前状态和交接范围

已完成的最新动作是：在 `liuqingmu096` 账号下运行 ozertuu 方案的优化版，并提交保存的 Version 2。正式提交编号为 `58421503`。该版本通过了 75 条数据的真实推理测试，六个主模型和三折 Native 都完成了推理，计时器记录总耗时约 959.9 秒。正式评测仍以 Kaggle 该提交的最新状态为准。

截至本轮历史查询，队伍中已经有真实线上公开成绩 `0.689`，来自 2026 年 11 月 14 日的 Dhananjay Version 10 复现提交 `58337205`。这不等于最新优化版已取得同样成绩，也不代表全部方法都是个人原创。此前 whitegg 路线的 `0.679`、最早完整复现的 `0.636` 也都有提交记录。

服务器中确认存在项目自训模型：两阶段 Qwen3 LoRA、五折 Qwen3 对照实验，以及 DeBERTa 和 ELECTRA 训练产物。自训 Qwen3 的分组 OOF 主指标约为 `0.609766`，它是本地开发验证成绩，不能直接写成 Kaggle 线上成绩。

本次交接区分四种证据：本轮实时查询、已保存的训练终态及权重元数据、早期报告，以及尚待验证的计划。某个文件名包含 `final`、某个提交状态显示 `COMPLETE`、某个目录里存在适配器，都不足以单独证明完整训练或成功评分。

附带的附件保存代码、Notebook、环境快照、训练摘要和提交回执，不复制服务器上的大体积模型权重。权重的准确位置、来源和清单在后文及机器可读附件中列出。认证令牌、私钥和密码不进入本交接包。

## 2 比赛任务与数据

比赛入口：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)。任务是根据一段模型生成文本，判断其来自 18 个具体模型中的哪一个。这是文本来源归因的多分类任务，不是“人写还是 AI 写”的二分类，也不是让本项目生成这些文本。

已核对的官方训练集有 2,699 行、150 个 `prompt_id` 和 18 个标签。149 个提示词各对应 18 条文本，另一个提示词少一条。主字段为 `id`、`prompt_id`、`text`、`model`。公开 `test.csv` 只有 3 行，是格式预览；正式评测会把保存的 Notebook 放到隐藏数据上重新执行。

主要指标是 Balanced Accuracy，即每个真实类别的召回率先分别计算，再取平均。普通准确率、某一个类别的准确率、二分类 AUROC 和这个指标不是一回事。标准实现可以用：

```python
from sklearn.metrics import balanced_accuracy_score
score = balanced_accuracy_score(y_true, y_pred)
```

训练集 SHA256 为 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`，公开测试文件 SHA256 为 `4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57`。这些值标识此前核验的数据副本；如果官方数据更新，需要重新记录，不应继续沿用旧哈希。

验证时按 `prompt_id` 分组。同一提示词的不同模型回答如果被随机拆到训练和验证两侧，模型可能依赖题材线索，得到过于乐观的结果。`prompt_id` 可以用于定义训练和验证划分；实际盲预测时不能把验证文本的真实分组身份当作已知输入。

最终输出是 `submission.parquet`，列顺序为 `id, model`。每个测试 ID 必须恰好出现一次，预测标签必须在官方 18 类中，不能含空值。只上传 3 行公开样例无法代表隐藏测试集的完整答案，正确的代码竞赛提交应绑定保存的 Notebook 版本。

### 2.1 三种数据规模的区别

公开 3 行数据用于格式检查；这次的 75 行用于测量真实执行，其中包含 3 条公开样例和 72 条按文本长度选取的计时文本；正式隐藏评测数据由 Kaggle 提供。75 行计时文本可能已被公开权重见过，所以本报告不计算或宣传它们的泛化准确率。

此前资料提到隐藏测试约 4,000 行。代码中也有面向 4,000 行的运行估计，但这不是本轮直接读取隐藏数据得出的数量。实际规模、长度和类别分布都可能不同。小样本耗时按行数放大时还会重复放大模型加载成本，因此只能作粗略参考，不能冒充全量实测。

## 3 项目发展过程

2026 年 10 月下旬，项目先完成赛题理解、文献与公开方案调研，建立字符、词语、结构与风格特征基线。历史文件记录过按提示词分组的 style-aware TF-IDF 和 LinearSVC 基线，OOF Balanced Accuracy 约为 `0.3844842158`。它是轻量基线，不应与后来多模型公开榜分数混在同一列而不标注评测来源。

随后接收固定版本的学长交接仓库，提交哈希为 `3b8e3331890281bcf4f41c5c2a37875ca8323811`。早期工作包括恢复 32 项输入资源、处理部分下载入口失效、验证 PPU 运行兼容性，以及保持底座、适配器、词表和分类头的匹配关系。

2026 年 10 月 28 日，完整推理方案在 Kaggle 双 T4 上完成公开预览和隐藏重跑，提交 `57706812` 得到真实公开分数 `0.636`。同一时期学校服务器开始形成独立训练路线，之后产生两阶段 Qwen3、分组交叉验证以及两个编码器的训练记录。

10 月 29 日前后，还进行了组件融合、盲分组和压力测试。有的候选开发分数提高，但未达到预设的采纳条件。相关权重和日志仍保留，不能把“实验完成”写成“候选已经上线”或“超过已评分基准”。

11 月 12 日至 14 日，项目复现 whitegg 和 Dhananjay 公开方案。当前队伍可见提交记录分别有 `0.679` 和 `0.689`。11 月 15 日提交 ozertuu 当前 Version 37 的复现版本，正式评测因运行超时失败。11 月 16 日的排查进一步确认原固定镜像内 PyTorch 不支持 CUDA，随后更换 GPU 镜像、缩减重复计算，并加入阶段超时和有效预测回退。

自动生成的提交表会列出完整可见记录、时间、提交人和错误信息。这里描述的是主要工作线；早期报告中的“尚未训练”“尚未提交”等结论有时间边界，不能覆盖后续真实训练记录。

## 4 三套环境分别负责什么

### 4.1 Windows 本机

本机用于整理代码、运行 Kaggle API 客户端、通过 SSH 管理学校服务器、收集回执和撰写报告。本轮大型模型推理没有在本机笔记本显卡上完成。当前工作目录为 `C:\PROJECT_ROOT`，本机 Python 来自 `D:\anaconda\python.exe`。

本轮查询到本机为 Intel Core i5-11400H、12 个逻辑处理器、约 16 GB 物理内存，显示设备包括 NVIDIA GeForce RTX 3050 Laptop GPU。显示适配器名称不能代替显存与训练能力实测；本轮没有在该显卡上进行模型训练或大模型性能验收。

Kaggle 客户端依赖位于工作区的 `.runtime-kaggle` 中。运行操作脚本时显式设置该目录为 `PYTHONPATH`。普通本机 Python 环境与 Kaggle Notebook 环境不是同一套环境，不能因为本机安装了某个 CUDA 版本，就认为 Kaggle 容器也具备相同能力。

### 4.2 学校服务器

服务器通过现有 SSH 授权连接，地址为 `root@TRAIN_SERVER_HOST:22`。使用本机已有专用密钥，不把密钥内容复制进项目。网络可用时连接命令示例如下：

```powershell
ssh.exe -F NUL -o BatchMode=yes -o StrictHostKeyChecking=yes `
  -i C:\PATH_TO_SSH_KEY -p 22 root@TRAIN_SERVER_HOST
```

本轮从 cgroup 读取到 CPU 配额 `2400000 / 100000 = 24`，内存上限 `180388626432` 字节，即 168 GiB。`os.cpu_count()` 和 CPU affinity 都暴露了宿主机的 256 个逻辑处理器，系统内存信息也显示宿主机大容量内存；这些不能当成容器可使用的配额。报告的资源配置采用容器限额。

加速卡是两张 PPU-ZW810E，每张实测显存 `103079215104` 字节，即 96 GiB。它们不是 NVIDIA RTX 4090。厂商通过兼容接口暴露 `torch.cuda`，所以 `torch.cuda.is_available()` 为真不代表底层就是 NVIDIA CUDA。两张卡的显存也不会自动成为单个程序可随意使用的一块 192 GiB 显存。

厂商 PyTorch 包版本为 `2.4.0+ppu1.5.0.oe`，运行时报告 `2.4.0`，CUDA 兼容版本字段为 `11.8`。本轮使用的隔离解释器为 `/root/name-that-llm/ppu-compat-env/bin/python`，Python 3.10.13。历史环境记录的 Driver 为 `1.5.1-1d747a`，SDK 为 `1.6.1-51ad16`；驱动和 SDK 的版本不是本轮重新逐项查询所得，按历史记录标注。

模型环境启动前加载厂商 SDK：

```bash
source /usr/local/PPU_SDK/envsetup.sh
/root/name-that-llm/ppu-compat-env/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

不要在现有 PPU 环境直接安装 NVIDIA PyTorch 来修复 CUDA 问题。此次需要修正的是 Kaggle Notebook 镜像，学校服务器仍应保留厂商运行库。早期 Transformers 5 与厂商 Torch 2.4 的兼容性障碍、bitsandbytes 原生导入失败，不能通过改版本字符串来绕过。

### 4.3 Kaggle 推理环境

最新优化 Version 2 实测为 PyTorch `2.11.0+cu128`、CUDA `12.8`、两张 Tesla T4。保存版本的镜像固定为此前该账号成功使用的 GPU 镜像：

```text
gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f
```

原 ozertuu 复现沿用的镜像为：

```text
gcr.io/kaggle-images/python@sha256:0717e81990be591f8c5f98f033d29ab23e9ffde057e5b3a6bd4371f93969e962
```

在本轮相同原镜像测试中，PyTorch 报出 `Torch not compiled with CUDA enabled`。只设置 `enable_gpu=true` 或机器类型为 T4，并不能修复镜像内 CPU-only PyTorch。原公开样例的 GLM 二进制另带 CUDA 库，能够完成自身 GPU 检查，也不能证明 PyTorch 主模型能使用 GPU。

Version 2 在加载权重前检查 PyTorch CUDA 编译信息、CUDA 可用性、设备数和两张显卡名称，并在两张卡上分别执行矩阵计算。失败则立即停止。`gpu-validation.json` 是该检查的实际回执。其余软件版本若未在本轮单独导出，不用学校服务器版本或本机版本代填。

## 5 文件与存储位置

学校服务器的运行目录主要是 `/root/name-that-llm`，持久保存目录主要是 `/workspace/name-that-llm`。公开基础模型和数据放在 `/kaggle/input`。项目自训权重主要放在 `/workspace/name-that-llm/web-pro-checkpoints`。这些目录的用途不同，不能只备份几份 JSON 回执就认为模型权重也已备份。

`/workspace` 是持久挂载，早期资料记录为 OSS 挂载。`df` 显示的超大文件系统容量不代表项目账户实际可用配额。运行时的大量小文件、临时模型 staging 和缓存适合放本地快盘；训练完成后再把重要权重、配置和日志复制到持久目录，并核对哈希。

Kaggle 的 `/kaggle/input` 由数据源挂载，通常只读；`/kaggle/temp` 用于中间状态和模型 staging；`/kaggle/working` 用于需要保存的输出。本次优化将大模型临时状态放在 `/kaggle/temp/ntl-optimized`，把预测、阶段日志、GPU 检查和运行报告保存在 `/kaggle/working`。

本交接包不会下载隐藏测试标签，也没有取得隐藏测试逐行真实答案。`submission.parquet` 是模型预测，不是官方标准答案。随包附带的 3 行文件只是公开样例输出。

### 5.1 服务器基础模型

本轮读取模型配置，确认了 Qwen3 14B、Qwen3 14B Base、Qwen3 8B、Qwen3 8B Base、Qwen3 4B Base、Qwen2.5 14B、Qwen2.5 7B 共七个基础模型版本。完整路径、层数、隐藏维度和配置 dtype 在自动附录列出。

比赛的 18 个标签里包含 GPT、Gemini、Claude 等模型名，不代表服务器上下载了这些闭源模型的完整权重。它们多数是要识别的目标类别；实际提取特征和分类的骨干是 Qwen、DeBERTa、ELECTRA 等可获取模型。

最新 Kaggle 方案还挂载 Llama 3.1 8B Base 和 Qwen3 Embedding 8B。这两项属于 Kaggle 当前挂载的数据源，不在本轮学校服务器七个基础模型目录清单中。报告不会把“在 Kaggle 可用”写成“已经下载到学校服务器”。

### 5.2 下载资产与自训产物的区别

公开 Dataset 中的 LoRA、分类头、词表、编码器权重和预拟合 SVC，属于下载的第三方资产。项目目录中有训练请求、执行步骤、终态、分组验证结果和最终权重的模型，才列入项目自训章节。适配器文件存在本身不证明训练者身份。

早期清单声明 32 项输入，包括 16 个 Dataset、8 个 Notebook 输出、7 个基础模型版本和比赛数据。四项失败下载后来分别按固定版本恢复：`nlm-l24-adapters` Version 7、`ntl-free-proxy-generations` Version 9、`ntl-handwriting-blend` Version 1、`ultrabert-380m-ntl` Version 1。修复记录涉及 161 个文件、8,837,637,760 字节。该历史恢复范围不等于所有输入都与作者评分时的历史版本逐字节相同。

本轮服务器清单使用当前文件目录统计逻辑大小，不重新哈希所有大权重，也不保证文件去重。准确的资产目录、文件数和逻辑字节数在自动附录中列出。模型是否能执行，还需要相应的运行日志，不能只看文件大小。

## 6 项目中实际训练过哪些模型

本章依据本轮从服务器取回的训练请求、终态、结果、权重元数据和持久目录清单。它证明项目存在这些训练实验及产物；报告不根据一个共享目录擅自推断每次操作由本人还是队友手工执行。个人答辩的责任分工仍应结合实际协作记录说明。

### 6.1 两阶段 Qwen3 LoRA

实验任务为 `qwen3_two_stage`，实验 ID 为 `qwen3-public-v1-20261029-015557-d5cd5503`，持久目录为 `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44`。

该实验从基础模型加新 LoRA 和分类头开始，先用公开数据训练，再用官方训练部分微调，并在固定分组留出上选取结果，最后对全部官方样本重新训练。选中的外部数据为 113,967 条，公开预训练 1 轮；官方微调 3 轮；全量官方重训 3 轮。训练组有 120 个提示词，验证组有 30 个，分组随机种子为 `20261028`。

底座为 Qwen3 14B Base，执行前 27 层，读取第 `2,4,6,9,13,17,22,27` 层的表示，最大长度 447。LoRA 的 rank 为 16、alpha 为 32、dropout 为 0.05；分类头 dropout 为 0.1。全局 batch 为 18，公开预训练学习率为 `2e-5`，官方微调 LoRA 学习率为 `5e-5`，分类头学习率为 `5e-4`。权重衰减为 0.01，warmup 比例为 0.1，种子为 42。

训练使用两张 PPU，结果记录的 rank 0 峰值显存约 57.31 GiB。记录的阶段分别为公开预训练 6,331 步、官方微调 357 步、全量官方重训 447 步，总计 7,135 步；累计训练耗时约 5,865.73 秒。`acceptance_only=false`，这不是只跑几步的验收示例。

留出集上的最佳盲分组 Balanced Accuracy 为 `0.6092592593`，出现在第 3 轮。这个分数属于 `holdout-best`，不是全量重训后 `final` 的独立验证分数。两者权重分别保存在 `holdout-best` 和 `final` 下。

盲分组预测只接收文本。该实验在训练侧拟合分组阈值，选中余弦阈值 0.05，最小分组 3、最大分组 32，centered 权重为 0.5。它没有把验证集真实 `prompt_id` 用作预测输入。

### 6.2 五折 Qwen3 对照训练

实验为 `qwen3-groupcv-v2-20261029-043240-a1`，持久目录为 `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198`。使用相同的公开预训练阶段，然后在官方数据上进行五折分组验证，比较固定的两种训练配方。

对照配方训练 3 轮，学习率 `5e-5`；候选配方训练 5 轮，学习率 `3e-5`。所有 2,699 条官方样本都有对应的折外预测。对照配方的盲分组 OOF Balanced Accuracy 为 `0.6097663435`，plain OOF 为 `0.5745712155`；候选分别为 `0.6053268705` 和 `0.5738329605`。增加轮数并降低学习率没有在这次对照中带来收益，所以最终选中对照配方。

最终 `final` 使用选中的 3 轮配方重训。`ensemble/member-0` 至 `member-4` 保存五个成员，`folds/control` 和 `folds/candidate` 保存各折证据，`cv-evidence/summary.json` 保存汇总。OOF 每行对应一个未见过该行的折模型，不能直接把这个分数说成“五模型融合的独立测试成绩”。

结果记录总训练步数为 11,546，累计训练时间约 11,645.29 秒。模型实际执行约 97.41 亿参数，其中约 4,490.86 万为可训练参数，采用 bfloat16 和 `flash_attention_2`。LoRA 模块包括注意力的 `q_proj, k_proj, v_proj, o_proj` 及 MLP 的 `gate_proj, up_proj, down_proj`。

预训练数据共有 113,967 行，日志记录该轮实际看到 113,958 行，少 9 行是完整全局 batch 与 `drop_last` 的结果。报告不会把输入文件行数直接写成所有样本都被优化器使用。

另一个容易误读的地方是，基础 `cfg` 中仍保留 `epochs=2.0`，而实际任务请求和最终元数据是选中配方训练 3 轮。恢复实验时应同时查看请求、阶段、epoch 和 step，不能只读静态默认配置。

### 6.3 DeBERTa 和 ELECTRA 训练

实验 ID 为 `encoders-r5-20261029-a1`。持久目录为 `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full`。终态记录 `status=completed`、退出码为 0、`durable_models_verified=true`，不是未完成的训练计划。

每个编码器使用 72,000 条公开样本预训练 1 轮，然后在官方数据上做五折、每折 3 轮训练，最后用全部官方样本重训 3 轮。最大长度 512，microbatch 为 16，梯度累积 4 次，有效 batch 为 64。公开阶段 encoder 学习率为 `2e-5`、head 为 `1e-4`；官方阶段分别为 `8e-6`、`4e-5`。

其他记录包括权重衰减 0.01、warmup 0.1、label smoothing 0.05、head dropout 0.1、梯度裁剪 1.0，采用 bfloat16 autocast 和 float32 参数，使用最后隐藏层的 masked mean pooling，预测只接收文本，不做 query centering。

DeBERTa 的分组 OOF plain Balanced Accuracy 为 `0.4979368630`；ELECTRA 为 `0.4149366145`。最终目录分别为 `deberta/final` 与 `electra/final`。每个目录中的 `encoder/model.safetensors` 是编码器权重，`head.safetensors` 是分类头，`meta.json` 说明类别顺序与训练配方。两类文件需要一起保存。

训练监督进程总墙钟时间约 2,076.09 秒。两个专家各自的累计耗时可能有重叠，不能简单相加后当成整个任务墙钟时间。终态和结果文件比某个停止更新、仍写着 `running` 的旧心跳快照更可靠。

### 6.4 组件融合实验及未采纳结果

`managed-components-v4-20261029-1200-a1` 完成了五折组件开发，复用了 control 权重，并拟合新的折内 readout。它不是又从零训练了一套完整 Qwen3 主干。

固定融合配方的开发提升为 `0.0033358190`，五折中四折为正，但没有达到预设 `0.005` 提升门槛。按提示词做的 2,000 次 paired bootstrap，其 95% 区间约为 `[-0.0007407407, 0.0077777778]`，下界没有高于零。因此 `adopted=false`、`deployment_ready=false`，保留原已评分基准是合理结论。

同一候选在 PPU 上做过真实 4,000 行推理，记录总墙钟约 380.57 秒。该结果只证明对应 PPU 硬件和对应候选的执行情况，不是 Kaggle T4 的隐藏测试耗时，也不能替代当前优化版的性能验收。

### 6.5 不能当成正式训练成果的文件

`backup-acceptance-two-stage-5a4005bc1814` 的元数据明确写着 `acceptance_only=true`，最后只到第 1 轮、第 2 步。它适合验证文件格式和运行链路，不适合作为正式模型效果证据。

公开 Dataset 中带有 `final`、`trainables` 或 `heads` 的权重，首先按下载资产处理。它们可能由公开方案作者训练，不能因为现在位于自己的服务器就记为个人训练。反过来，本章已经查到真实项目训练日志，也不应继续套用 10 月 27 日“没有新 LoRA 训练”的旧结论。

### 6.6 异步训练执行链路

项目还保留了 `mojohugo/web-pro-gpu-runner` 执行链路。服务器项目说明描述的流程是：网页对话准备实验请求，GitHub 保存配置，Windows Runner 经 SSH 调用双 PPU，训练结果和小型报告再回到 GitHub，大权重保留在训练主机与持久目录。关闭聊天窗口不等于后台训练立即停止，但 Runner、校园网连接、跳板和服务器仍需要在线。

正式训练触发文件是 main 分支的 `experiments/request.json`。修改说明文档或示例配置不等于发起训练。同一个 experiment ID 只能领取一次，失败重试应使用新 ID。执行器有 GPU 文件锁、独立时间上限和结果检查，不会无限调参，也不会在没有授权时自行向 Kaggle 提交。

`qwen3_two_stage` 使用 schema 2；后续分组交叉验证使用 schema 3；组件开发使用 schema 4。自训 Qwen3 的两阶段请求上限为 480 分钟，执行器允许的单任务上限在项目文档中记为 720 分钟。请求参数、源代码 commit、数据 SHA、权重 SHA 和终态一起标识一次训练，不能只用文件夹名字恢复实验。

项目说明中的离线预测入口是 `scripts/predict_offline.py`，Kaggle 提交模板是 `kaggle/submit.ipynb`。随包若包含这些文件，按“服务器当前工作副本”标注，它们不自动等于所有历史训练 job 的精确源码版本。训练记录中引用的固定 commit 和 checkpoint 元数据仍是版本核对依据。

## 7 Notebook 与方法路线

### 7.1 早期完整交接方案

固定学长仓库的主分支包含 SVC、Qwen3 14B LoRA、Qwen2.5 14B LoRA 和 DeBERTa。complement 分支包含风格特征、LightGBM、多个编码器、LoRA、token-attention 和 UltraBERT。主分支与 complement 按 85% 和 15% 混合；部分低置信样本再混入 ATLAS，最后按原条件进行分布校正。

TF-IDF 和 SVC 捕捉词语、字符、标点和书写结构；神经骨干提取上下文表示；plain 与 centered twin 用于区分文本内容和来源风格；Gaussian readout 用统计分布建模特征；ATLAS 根据多个来源适配器的评分补充判别信息。这些组件是不同信息来源的融合，不是简单重复训练同一个分类器。

这一完整推理 Notebook 的真实线上成绩为 `0.636`。公开三行执行耗时约 2,073.33 秒，只是预览流程的耗时，不能称作隐藏全量推理耗时。文件包中的原始结果、独立复算与提交回执保留了这个边界。

### 7.2 whitegg 路线

whitegg 方案使用预拟合风格分类器、多种模型 readout、Native 分支和融合。项目曾强制在公开样例上走真实模型推理进行诊断，随后把保存的代码版本提交到 Kaggle 隐藏重跑。当前队伍记录中存在多个 `0.679` 提交。

重复提交记录不代表训练了多个不同模型。有些是同一已验证 Notebook 的重试，报告在历史表中保留描述和提交编号，不能把它们当成独立方法对照。预测文件、模型权重和 Notebook 是三个不同层次的产物。

### 7.3 Dhananjay Version 10

2026 年 11 月 14 日固定复现 Dhananjay Version 10，保留公开代码、设置和贡献者署名，提交 `58337205` 的实际公开分数为 `0.689`。它是目前已核实的最高可见公开成绩。

这一成绩属于该账号／队伍的提交记录，但方案仍属于公开方案复现。不能写成“个人从零训练的模型达到 0.689”。如果报告要说明个人贡献，应列出环境配置、资产核验、复现、代码适配、运行验收和后续修改，而不是把作者全部算法据为己有。

### 7.4 ozertuu Version 37 原方案

用户给出的链接地址包含 `0-689-dual-svc-residual-refinement-apex`，但下载时的当前源版本是 37，正文声称的分数已变为 `0.696`。链接标题、正文宣传分数、当前版本号和自己账号的实测分数必须分别记录。

原方案涉及预拟合 SVC 核心、多种 LoRA readout、Native Qwen Embedding、GSQ 27B 和 GLM likelihood 重排。它在少于 50 条测试样本时触发 `FAST_COMMIT`，公开三行直接写入固定标签，并运行部分资产与环境检查。这个分支用于快速保存，不能当成完整模型复现成功的证明。

原始复现提交 `58379230` 没有产生分数，官方错误为 Notebook Timeout。本轮排查在相同固定镜像的真实推理测试中进一步复现 CPU-only PyTorch 问题。这是明确的环境缺陷；由于没有取得失败提交的隐藏分阶段日志，本报告不把某个具体隐藏模型阶段写成已经被证实的唯一超时位置。

## 8 本次优化版具体改了什么

### 8.1 修正 GPU 环境

改用此前成功使用的 GPU 镜像，并在 Notebook 开头验证 PyTorch CUDA 编译、两张 T4、两卡矩阵计算。这样可以在几十秒内发现不合适的运行环境，而不是让大模型在 CPU 上运行几十分钟。

CPU-only 环境的第一次计时测试中，没有任何主模型完成；六个模型中有直接失败，也有在阶段上限被终止。该次没有用于正式提交。GPU 修正后，六个主模型与 Native 全部完成，不能把第一次仅剩 CPU 基础预测的产物冒充完整优化结果。

### 8.2 减少重复推理

关闭 complement 整栈、需要额外遍历官方训练集的 Gaussian 校准、GSQ 27B 和 GLM likelihood 重排。Qwen3 8B、Qwen3 14B、Llama 8B 三个家族各保留 full-data 适配器及第 0、1 折，主模型适配器遍历由 21 次降至 12 次。Native 由 5 折降至 3 折。

这些修改减少运行量，也改变了原始融合配方。它们属于运行效率优化，不是已证实的准确率提升。新线上成绩必须等本次正式评测返回，不能承诺达到作者的 `0.696` 或超过已有 `0.689`。

### 8.3 保存有效结果并限制耗时

先用真正的预拟合分类器生成完整预测并原子保存。每完成一个主模型，核验 ID、标签顺序、形状和有限值后，再更新融合结果。某个模型失败或超时，不覆盖之前已经生成的有效预测。

每个模型独立运行，时间上限覆盖模型加载、磁盘读取、推理及子进程。主流程目标预算为 7.5 小时，外层 watchdog 为 7.8 小时，给保存产物和平台收尾留出时间。外层会跟踪子孙进程，包括另开 session 的嵌套模型进程，避免只杀父进程而留下仍占用 GPU 的子进程。

Native 每完成一折就原子保存结果，后续折超时仍可使用已完成的折。所有阶段记录状态、耗时、退出码和日志，不能把回退当成全部模型成功。正式隐藏评测可能选择性跳过超时增强阶段；最终应结合隐藏运行状态和分数判断效果。

### 8.4 分组与融合

语义分组表示只计算一次并缓存，主模型和 Native 使用同一份分组。类别顺序统一到固定的 18 类列表。plain 和 centered twin 结果按原相应权重合并，再进行 Sinkhorn 风格的分布校正；Native 采用 0.40 的混合份额，并保留有限值与模型一致性检查。

主模型与 Native 的融合不是把字符串标签投票后取众数，而是处理每类分数／概率。标签顺序错位会造成“程序运行但答案错误”，因此代码把 ID 顺序和标签顺序检查放在每次加载结果时执行。

## 9 本次验证覆盖了什么

本地五项回归检查覆盖超时终止、成功任务状态、旧预测保护、成员 ID 顺序拒绝和融合公式一致性。原始无超时 helper 对短时限契约失败，新 helper 能按时停止；不是只检查代码能否编译。

Kaggle Linux 又验证了独立 session 的子孙进程终止。旧的仅杀进程组行为在构造样例中留下了子进程，新的实现能终止它们，同时保持已经保存的预测文件不变。

模型测试使用 75 条真实文本。六个主模型全部完成，Native 第 0、1、2 折全部完成，没有非有限值行。Native 与主融合的逐折预测一致率约为 85.33%、80.00%、85.33%，通过预设的一致性保护条件。这是一致性诊断，不是准确率。

主计时约 959.9 秒，完整 Notebook 日志约 1,002.1 秒。前者是运行脚本计时，后者还包括环境检查、Notebook 转换与平台收尾，两者起止范围不同。阶段详细耗时在自动附录中保留，不把累计时间误称为单模型前向时间。

本轮没有重新训练公开方案的 LoRA，没有在隐藏测试标签上调参，没有声称 75 条测试可以证明泛化准确率，也没有取得新的 Private Score。正式提交已经受理，完整隐藏评测是否在目标时限内结束仍需等待官方结果。

## 10 提交与成绩怎么核实

当前操作固定账号为 `liuqingmu096`，队伍显示为 `mojohugo`。API 在执行操作前核验账号，创建的优化 Notebook 为私有。提交绑定保存的 Version 2，输出文件名为 `submission.parquet`，不是直接上传公开三行文件来代替完整预测。

提交详情里的 `COMPLETE` 表示该条提交处理结束，不自动等于成功。失败的超时提交也会出现这个状态；需要同时检查错误说明和是否有实际分数。`PENDING` 表示仍等待处理，分数为空时不能写成零分，也不能拿标题分数代替。

队伍中可见的提交不一定都由同一个人操作。自动附录列出提交人；来自 `mojohugo` 的历史条目按队伍记录描述，不直接记作 `liuqingmu096` 个人提交。一个队伍也不一定自动获得队友所有私有 Notebook 的代码访问权。

历史 `0.586` 提交注释提到 Qwen3 final refit，但本轮没有建立该提交与后文两个具体训练 job 的完整版本映射，因此不把它直接当作这些 job 的线上分数。需要进一步归因时，应回读对应 Notebook 版本、权重引用和哈希。

### 10.1 页面查看方法

1. 打开比赛的 Submissions 页面，找到提交编号或 Notebook 名称。
2. 看状态及错误详情。成功评分后记录分数和日期；报错时保留原错误信息。
3. 打开对应 Notebook，核对使用的是哪一个保存版本。
4. 在 Output 中查看运行报告、日志和公开预测文件。下载到的公开预览不能视作隐藏测试全量输出。
5. 报告中引用排名时，同时保存当日名次、总队伍数和时间。当前报告没有重新核实排行榜百分位，因此不估算课程的排名档位。

## 11 接手者如何继续运行

先确认要处理哪一条路线：学校自训模型、旧完整复现，还是当前 ozertuu 优化版。不要在旧项目目录里覆盖新配置，也不要把公开下载权重放进自训目录后改名为个人模型。

当前优化目录包含 `account-copy`、`runtime`、`tests`、构建脚本、操作脚本与 Version 2 证据。运行依赖是在 Kaggle 上挂载的输入源，不是本地存在一个 Notebook 就足以执行。迁移时同时保留 `kernel-metadata.json` 的 Dataset、Notebook 和 Model 引用。

查看最新正式提交可以使用已经配置好的本机环境：

```powershell
$env:PYTHONPATH='C:\PROJECT_ROOT\.runtime-kaggle'
python 'C:\PROJECT_ROOT\ozertuu-optimized-20261116\operate.py' detail --version 2 --ref 58421503
```

这条命令只查询已有提交。它依赖本机现有认证配置，不应把令牌复制进 Notebook、报告或聊天。操作脚本中的 `push` 和 `submit` 是会创建新运行或新提交的动作，不作为日常状态查询命令使用。

如需改代码，先在 `runtime` 修改，再运行回归检查和构建脚本，创建新保存版本，回读验证后再提交。构建脚本会根据源模板生成 worker，直接修改生成后的文件可能在下一次构建时被覆盖。

```powershell
python '.\tests\test_runtime.py'
python '.\build_notebook.py'
```

构建和测试本身不代表已上传，也不代表已取得线上成绩。正式提交脚本带有一次性回执保护；网络中断后先查看回执和 Kaggle 列表，再决定是否重试，不要删除回执后盲目重复提交。

### 11.1 恢复学校自训模型

Qwen3 推理需要 `meta.json`、`heads.safetensors`、`adapter/adapter_config.json`、`adapter/adapter_model.safetensors`，以及与它匹配的 Qwen3 14B Base。编码器推理需要 encoder 权重、head 权重、tokenizer 文件和 meta。只拷贝适配器而遗漏分类头、标签顺序或 tokenizer，会得到不可复现的结果。

自训 Qwen3 的 EOS 为 `151643`，最大长度为 447，执行 27 层。其他公开 complement checkpoint 可能使用不同 EOS，不能统一替换成底座默认值。最新优化版的主家族、clean 分支和 Native 各有自己的层数、长度与校准契约，配置附录会分别列出。

部分历史部分权重不包含优化器和完整随机状态，不能保证精确断点续训。要续训，应先确认是否有完整 optimizer、scheduler、随机状态与数据顺序；否则应作为新的实验记录，而不是声称无损恢复同一次训练。

## 12 课程报告和答辩可以怎样使用这些材料

课程模块 B 允许选择与课程相关、正在开展且不是学习赛的竞赛，1 至 3 人组队，报告、排行榜和答辩分别占 30%、50%、20%。是否采用某一天的公开榜或最终榜，需要结合老师的具体要求确认。报告必须使用课程提供的模板。

模块 A 并非只有附件一的贷款题，还包括腾讯广告算法大赛方案学习与优化。当前项目已有训练、验证、复现和线上提交证据，适合整理模块 B，但不能因此省略竞赛资格、材料和答辩要求的核实。

答辩要求是 15 分钟以内，重点说明针对赛题做了哪些工作，并结合实验结果分析。可以按“任务与数据、验证方法、轻量基线、自训对照、公开方案复现、运行故障与优化、线上结果和限制”组织内容。不要把模型下载列表当成主要成果，也不要大段粘贴代码代替方法解释。

自训部分适合展示五折分组验证、3 轮与 5 轮的对照，以及没有达到采纳门槛的融合实验。复现部分适合展示权重匹配、标签与 EOS 契约、离线环境和实际线上分数。本次优化适合展示 CPU-only 镜像问题、GPU 前置检查、进程树超时和原子预测回退。

准备提交材料时，保存最优线上成绩、完整历史提交记录、线上测评 ID、报告和 PPT。报告模板中的旧贷款项目名、旧日期和示例环境应按本项目真实情况修改。本交接报告是素材与技术交接，不直接冒充已按学校模板完成的课程实验报告或答辩 PPT。

### 12.1 贡献表述参考

可以根据实际分工表述为“完成数据与环境核验”“在项目服务器上执行并核验分组训练实验”“复现公开多模型方案并保留来源”“诊断 CPU-only PyTorch 问题并改进运行可靠性”。每项后面应能对应证据文件。

公开架构、公开权重与作者训练成果应注明来源。项目自己的训练日志和开发分数可以展示，但不能直接归为线上 `0.689` 的来源；多人协作的训练和提交也应按真实分工解释。个人贡献、项目成果、公开方法来源三者可以同时写清楚。

## 13 常见问题

### 为什么有 18 个模型标签 却只下载了几个基础模型

18 个标签是需要识别的来源。实际分类器可以用几个开源骨干提取文本特征，再输出 18 类概率，不必也通常无法下载全部闭源来源模型。

### LoRA 是否等于完整模型

不是。LoRA 保存基础模型上的增量参数，通常还要配合分类头、配置和 tokenizer。更换底座版本或 EOS 后，原来的适配器不一定仍然匹配。

### 看到 final 是否说明这是自己训练的

不是。公开作者的 checkpoint 也常叫 final。需要结合训练请求、日志、终态、元数据和来源判断。项目的验收样例也有 final，但 `acceptance_only=true`。

### OOF 0.6098 和线上 0.689 哪个更好

它们来自不同模型、数据划分和评测环境，不能简单比较高低。OOF 用于开发对照，线上分数来自隐藏评测。报告应该分别列出模型、数据、验证方式和来源。

### 新版本为什么没有立即显示分数

代码竞赛提交先受理，再在隐藏数据上重跑。排队和运行都需要时间。收到提交编号只证明请求已受理，不能证明评测成功。

### 原版样例通过 为什么隐藏评测仍超时

原版少于 50 行时走快速分支，跳过完整推理；正式数据才运行重模型。另外原固定镜像的 PyTorch 没有 CUDA 支持，单独 GLM 二进制检查通过也不能覆盖这个问题。

### 本次是否已经重新训练了六个大模型

没有。本次重跑的是公开预训练模型的优化推理流程。学校服务器上已有独立训练历史，见自训章节，不与本次混淆。

### 两张 PPU 是否等于两张 4090

不是同一种硬件和软件栈。96 GiB 显存说明容量，不直接说明算力，也不能替代前向、反向、算子和多卡通信的实际验证。

### 这个文件包能否离线独立运行所有模型

不能。包内包含 Notebook、脚本、配置、清单和证据，不包含全部大权重。接手者需要访问已有服务器目录或 Kaggle 对应输入源，并遵守原数据和模型的使用条件。

### 可以把失败记录删掉吗

不建议。失败记录解释环境修正和优化选择，也便于核实真正成功的版本。交接包保留失败与成功证据，并在文件名和正文中区分。

## 14 后续工作和仍未确认的事项

首先查看提交 `58421503` 是否完成以及是否取得有效分数。如果失败，优先保留官方错误，不根据公开样例推断隐藏阶段；若成功，记录新分数和查询时间，再与历史 `0.689`、`0.679` 作同类线上比较。

其次，如希望把学校自训模型加入新融合，应建立明确的消融实验，固定分组验证和标签顺序，验证额外组件在相同折上是否有独立收益，再测 Kaggle T4 全量预算。服务器上曾经跑过某模型，不代表它已进入当前提交。

本报告没有复查比赛最新截止日期、奖牌规则和老师采用的排名时点，没有拿到隐藏测试标签或 Private Score，也没有逐文件重算全部服务器大权重哈希。最终个人分工、某些历史提交与训练 job 的严格映射仍需补充协作证据。

新接手者应优先打开本包的最新状态文件、当前 Notebook、提交回执和 GPU 运行报告，再参考旧报告。历史状态冲突时，以时间更晚的官方记录和可核对的训练终态为准，同时保留旧文件而不覆盖它们。

## 15 自动生成的配置与证据附录

以下附录由本轮实际读取的 JSON、目录清单和 Kaggle 提交详情生成。技术参数、目录和来源引用尽量保留原值，便于接手者直接定位。正文的解释与附录的机器可读文件互相对应。

### 附录 A 最新提交状态

| 字段 | 本轮核验值 |
| --- | --- |
| 账号 | `liuqingmu096` |
| 队伍 | mojohugo |
| 提交编号 | `58421503` |
| 保存版本 | 优化 Notebook Version 2 |
| 状态 | 已受理，等待或正在评测 |
| 公开分数 | 尚未返回 |
| 错误 | 无 |
| 提交时间 UTC | 2026-11-16T07:28:16.697000 |
| 本轮实测推理行数 | 75 |
| 主流程计时 | 959.885 秒 |
| Notebook | https://www.kaggle.com/code/liuqingmu096/ntl-ozertuu-bounded-optimized-20261116 |
| 提交页面 | https://www.kaggle.com/competitions/name-that-llm/submissions |

### 附录 B 本机和服务器软件版本

| 包或字段 | Windows 本机 | 学校 PPU 隔离环境 | 最新 Kaggle 实测 |
| --- | --- | --- | --- |
| Python | 3.12.7 \| packaged by Anaconda, Inc. \| (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)] | 3.10.13 (main, Sep 10 2025, 10:23:41) [GCC 11.4.0] | 本轮日志路径为 Python 3.13，未独立导出完整版本号 |
| 解释器 | D:\anaconda\python.exe | /root/name-that-llm/ppu-compat-env/bin/python | Kaggle 保存版本运行环境 |
| torch | 2.7.1+cu118 | 2.4.0+ppu1.5.0.oe | 2.11.0+cu128 |
| transformers | 未安装或未单独采集 | 4.57.6 | 未在本轮单独导出，不用其他环境代填 |
| accelerate | 未安装或未单独采集 | 1.14.0 | 未在本轮单独导出，不用其他环境代填 |
| peft | 未安装或未单独采集 | 0.17.1 | 未在本轮单独导出，不用其他环境代填 |
| numpy | 1.26.4 | 2.0.2 | 未在本轮单独导出，不用其他环境代填 |
| pandas | 2.2.2 | 2.3.3 | 未在本轮单独导出，不用其他环境代填 |
| scipy | 未安装或未单独采集 | 1.15.3 | 未在本轮单独导出，不用其他环境代填 |
| scikit-learn | 1.5.1 | 1.7.2 | 未在本轮单独导出，不用其他环境代填 |
| lightgbm | 未安装或未单独采集 | 4.6.0 | 未在本轮单独导出，不用其他环境代填 |
| pyarrow | 未安装或未单独采集 | 18.1.0 | 未在本轮单独导出，不用其他环境代填 |
| tokenizers | 未安装或未单独采集 | 0.22.2 | 未在本轮单独导出，不用其他环境代填 |
| safetensors | 未安装或未单独采集 | 0.6.2 | 未在本轮单独导出，不用其他环境代填 |
| bitsandbytes | 未安装或未单独采集 | 未单独核验 | 未在本轮单独导出，不用其他环境代填 |
| protobuf | 未安装或未单独采集 | 3.20.3 | 未在本轮单独导出，不用其他环境代填 |

服务器资源限额原始读数如下。CPU affinity 和宿主机内存只用于解释观测差异，不用作资源分配依据。

```json
{
  "cgroup_limits": {
    "cpu/cpu.cfs_quota_us": "2400000",
    "cpu/cpu.cfs_period_us": "100000",
    "cpu,cpuacct/cpu.cfs_quota_us": "2400000",
    "cpu,cpuacct/cpu.cfs_period_us": "100000",
    "memory/memory.limit_in_bytes": "180388626432"
  },
  "host_cpu_count": 256,
  "cpu_affinity_count": 256,
  "torch_runtime": {
    "version": "2.4.0",
    "cuda_build": "11.8",
    "available": true,
    "device_count": 2,
    "devices": [
      {
        "name": "PPU-ZW810E",
        "total_memory_bytes": 103079215104
      },
      {
        "name": "PPU-ZW810E",
        "total_memory_bytes": 103079215104
      }
    ]
  }
}
```

本机硬件快照：

```json
{
  "logical_processors": 12,
  "memory_bytes": 17007173632,
  "cpu_names": [
    "11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz"
  ],
  "video_names": [
    "GameViewer Virtual Display Adapter",
    "NVIDIA GeForce RTX 3050 Laptop GPU"
  ]
}
```

### 附录 C 服务器基础模型版本

| 模型路径 | 模型类型 | 层数 | 隐藏维度 | 配置 dtype |
| --- | --- | --- | --- | --- |
| `/kaggle/input/models/qwen-lm/qwen-3/Transformers/14b/1` | qwen3 | 40 | 5120 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen-3/Transformers/14b-base/1` | qwen3 | 40 | 5120 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen-3/Transformers/8b/1` | qwen3 | 36 | 4096 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen-3/Transformers/8b-base/1` | qwen3 | 36 | 4096 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen-3/Transformers/4b-base/1` | qwen3 | 36 | 2560 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen2.5/Transformers/14b/1` | qwen2 | 48 | 5120 | bfloat16 |
| `/kaggle/input/models/qwen-lm/qwen2.5/Transformers/7b/1` | qwen2 | 28 | 3584 | bfloat16 |

### 附录 D 服务器下载资产目录清单

本轮扫描得到 27 个聚合目录条目，统计的逻辑字节总数为 `191,617,828,391`，约 178.46 GiB。模型目录按家族聚合，七个具体版本见上一表；这不是 27 个基础模型，也不是去重后的物理占盘量。比赛目录仅统计顶层结构。

| 服务器相对目录 | 文件数 | 逻辑字节数 | 换算 |
| --- | --- | --- | --- |
| `/kaggle/input/competitions` | 0 | 目录项 | 未单独统计 |
| `/kaggle/input/datasets/abdullah1afify/ntl-generations` | 4 | 911057 | 0.001 GiB |
| `/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors` | 10 | 19329294373 | 18.002 GiB |
| `/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters` | 88 | 3730983134 | 3.475 GiB |
| `/kaggle/input/datasets/dhruvtibarewal/nlm-matched-clean` | 3 | 83376272 | 0.078 GiB |
| `/kaggle/input/datasets/dhruvtibarewal/nlm-offline-wheels` | 4 | 45193795 | 0.042 GiB |
| `/kaggle/input/datasets/dophuchung/ntl-proxy-generations` | 2 | 891947 | 0.001 GiB |
| `/kaggle/input/datasets/fleongg/ntl-legacy-qwen24-recovery` | 15 | 138146537 | 0.129 GiB |
| `/kaggle/input/datasets/fleongg/ntl-quality-selected-public-svc-pool` | 7 | 22551878 | 0.021 GiB |
| `/kaggle/input/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1` | 15 | 462803915 | 0.431 GiB |
| `/kaggle/input/datasets/lucasboesen/ntl-free-proxy-generations` | 2 | 6068379 | 0.006 GiB |
| `/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend` | 15 | 1281363560 | 1.193 GiB |
| `/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora` | 11 | 317728259 | 0.296 GiB |
| `/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl` | 56 | 3819222687 | 3.557 GiB |
| `/kaggle/input/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset` | 13 | 97252706 | 0.091 GiB |
| `/kaggle/input/datasets/seyitkaangunes/ntl-extra-api-passages` | 3 | 4660571 | 0.004 GiB |
| `/kaggle/input/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | 17 | 321066208 | 0.299 GiB |
| `/kaggle/input/models/qwen-lm/qwen-3` | 69 | 99947796669 | 93.084 GiB |
| `/kaggle/input/models/qwen-lm/qwen2.5` | 32 | 44794491086 | 41.718 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-desklib-weights` | 6 | 1744440217 | 1.625 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-electra-large-weights` | 6 | 1337133945 | 1.245 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-extract-encoders2` | 11 | 2120538003 | 1.975 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25` | 8 | 11459987786 | 10.673 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-train-lora24-s7` | 30 | 319503562 | 0.298 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn` | 4 | 1368061 | 0.001 GiB |
| `/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn2` | 6 | 1907992 | 0.002 GiB |
| `/kaggle/input/notebooks/vnton37nguyn/ntl-train-lora-qwen25-instyle` | 24 | 229145792 | 0.213 GiB |

### 附录 E 原固定方案的三十二项声明

| 类型 | 引用 | 声明的相对目录 |
| --- | --- | --- |
| dataset | `dhruvtibarewal/nlm-l24-adapters` | `datasets/dhruvtibarewal/nlm-l24-adapters` |
| dataset | `seyitkaangunes/ntl-extra-api-passages` | `datasets/seyitkaangunes/ntl-extra-api-passages` |
| dataset | `abdullah1afify/ntl-generations` | `datasets/abdullah1afify/ntl-generations` |
| dataset | `dophuchung/ntl-proxy-generations` | `datasets/dophuchung/ntl-proxy-generations` |
| dataset | `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | `datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` |
| dataset | `dhruvtibarewal/nlm-matched-clean` | `datasets/dhruvtibarewal/nlm-matched-clean` |
| dataset | `dhruvtibarewal/nlm-offline-wheels` | `datasets/dhruvtibarewal/nlm-offline-wheels` |
| dataset | `josephadamski91/ntl-atlas-qwen3-8b-v1` | `datasets/josephadamski91/ntl-atlas-qwen3-8b-v1` |
| dataset | `lucasboesen/ntl-free-proxy-generations` | `datasets/lucasboesen/ntl-free-proxy-generations` |
| dataset | `fleongg/ntl-legacy-qwen24-recovery` | `datasets/fleongg/ntl-legacy-qwen24-recovery` |
| dataset | `fleongg/ntl-quality-selected-public-svc-pool` | `datasets/fleongg/ntl-quality-selected-public-svc-pool` |
| dataset | `nishantdahal/ntl-two-stage-lora` | `datasets/nishantdahal/ntl-two-stage-lora` |
| dataset | `nishantdahal/ntl-handwriting-blend` | `datasets/nishantdahal/ntl-handwriting-blend` |
| dataset | `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | `datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset` |
| dataset | `bbucxi/qwen35-9b-safetensors` | `datasets/bbucxi/qwen35-9b-safetensors` |
| dataset | `pjmathematician/ultrabert-380m-ntl` | `datasets/pjmathematician/ultrabert-380m-ntl` |
| kernel | `lucasboesen/name-that-llm-glm4-9b-trunc25` | `notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25` |
| kernel | `lucasboesen/name-that-llm-extract-encoders2` | `notebooks/lucasboesen/name-that-llm-extract-encoders2` |
| kernel | `lucasboesen/name-that-llm-electra-large-weights` | `notebooks/lucasboesen/name-that-llm-electra-large-weights` |
| kernel | `lucasboesen/name-that-llm-desklib-weights` | `notebooks/lucasboesen/name-that-llm-desklib-weights` |
| kernel | `lucasboesen/name-that-llm-train-lora24-s7` | `notebooks/lucasboesen/name-that-llm-train-lora24-s7` |
| kernel | `lucasboesen/name-that-llm-train-tokattn` | `notebooks/lucasboesen/name-that-llm-train-tokattn` |
| kernel | `lucasboesen/name-that-llm-train-tokattn2` | `notebooks/lucasboesen/name-that-llm-train-tokattn2` |
| kernel | `vnton37nguyn/ntl-train-lora-qwen25-instyle` | `notebooks/vnton37nguyn/ntl-train-lora-qwen25-instyle` |
| model | `qwen-lm/qwen-3/Transformers/14b-base/1` | `models/qwen-lm/qwen-3/Transformers/14b-base/1` |
| model | `qwen-lm/qwen-3/Transformers/8b-base/1` | `models/qwen-lm/qwen-3/Transformers/8b-base/1` |
| model | `qwen-lm/qwen-3/Transformers/14b/1` | `models/qwen-lm/qwen-3/Transformers/14b/1` |
| model | `qwen-lm/qwen-3/Transformers/8b/1` | `models/qwen-lm/qwen-3/Transformers/8b/1` |
| model | `qwen-lm/qwen-3/Transformers/4b-base/1` | `models/qwen-lm/qwen-3/Transformers/4b-base/1` |
| model | `qwen-lm/qwen2.5/Transformers/7b/1` | `models/qwen-lm/qwen2.5/Transformers/7b/1` |
| model | `qwen-lm/qwen2.5/Transformers/14b/1` | `models/qwen-lm/qwen2.5/Transformers/14b/1` |
| competition | `name-that-llm` | `competitions/name-that-llm` |

这里的声明来自固定交接仓库。当前文件是否存在以服务器实时清单为准；Dataset 和 Notebook 的历史版本不全都固定，不能只凭声明认定与原作者评分输入逐字节相同。

### 附录 F 最新优化版 Kaggle 输入源

#### Dataset 输入

| 序号 | 引用 |
| --- | --- |
| 1 | `dhruvtibarewal/nlm-l24-adapters` |
| 2 | `lightningv08/ntl-olmo2-native24-final-inference-assets` |
| 3 | `crtproduction/name-that-llm-open-pool` |
| 4 | `seyitkaangunes/ntl-extra-api-passages` |
| 5 | `abdullah1afify/ntl-generations` |
| 6 | `dophuchung/ntl-proxy-generations` |
| 7 | `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` |
| 8 | `dhruvtibarewal/nlm-matched-clean` |
| 9 | `dhruvtibarewal/nlm-offline-wheels` |
| 10 | `josephadamski91/ntl-atlas-qwen3-8b-v1` |
| 11 | `lucasboesen/ntl-free-proxy-generations` |
| 12 | `nuridudak/ntl-q14-lora` |
| 13 | `nuridudak/ntl-q14-lora-v4` |
| 14 | `kirilltym/ntl-spectral-style-heads-paired-4404` |
| 15 | `nuridudak/ntl-svc-prefit-pool-v4n` |
| 16 | `dhruvtibarewal/nlm-enc-v5` |
| 17 | `nuridudak/ntl-two-stage-encoders` |
| 18 | `nishantdahal/ntl-two-stage-lora` |
| 19 | `nishantdahal/ntl-handwriting-blend` |
| 20 | `dhruvtibarewal/nlm-l24-adapters-v5` |
| 21 | `prvsiyan/prvsiyans-extended-name-that-llm-dataset` |
| 22 | `bbucxi/qwen35-9b-safetensors` |
| 23 | `pjmathematician/ultrabert-380m-ntl` |

#### Notebook 输出输入

| 序号 | 引用 |
| --- | --- |
| 1 | `lucasboesen/name-that-llm-glm4-9b-trunc25` |
| 2 | `lucasboesen/name-that-llm-extract-encoders2` |
| 3 | `lucasboesen/name-that-llm-electra-large-weights` |
| 4 | `lucasboesen/name-that-llm-desklib-weights` |
| 5 | `lucasboesen/name-that-llm-train-lora24-s7` |
| 6 | `lucasboesen/name-that-llm-train-tokattn` |
| 7 | `lucasboesen/name-that-llm-train-tokattn2` |
| 8 | `vnton37nguyn/ntl-train-lora-qwen25-instyle` |

#### 基础模型输入

| 序号 | 引用 |
| --- | --- |
| 1 | `alexchilton/llama-3-1-8b-base/Transformers/default/1` |
| 2 | `qwen-lm/qwen-3/Transformers/8b-base/1` |
| 3 | `qwen-lm/qwen-3/Transformers/14b-base/1` |
| 4 | `qwen-lm/qwen-3-embedding/Transformers/8b/1` |
| 5 | `qwen-lm/qwen2.5/Transformers/14b/1` |

#### 比赛输入

| 序号 | 引用 |
| --- | --- |
| 1 | `name-that-llm` |

### 附录 G 完整可见提交历史

日期为官方接口返回的 UTC 时间。没有公开分数时，不能把 `COMPLETE` 单独解读为评分成功；完整错误文本见随包 JSON。

| 提交编号 | UTC 时间 | 提交人 | Public Score | 判断 | 原始说明 |
| --- | --- | --- | --- | --- | --- |
| `58421503` | 2026-11-16 07:28:16 | liuqingmu096 | 未返回 | 等待或未评分 | ozertuu v37 optimized bounded pipeline v2; real 75-row timing check; atomic fallback; liuqingmu096; 20261116 |
| `58379230` | 2026-11-15 09:50:45 | liuqingmu096 | 未返回 | 运行超时 | ozertuu v37 reproduction; original code/assets/credits preserved; liuqingmu096; 20261115 |
| `58337205` | 2026-11-14 08:26:32 | liuqingmu096 | 0.689 | 有评分 | Dhananjay Kothawale v10 reproduction; original code/settings/credits preserved; liuqingmu096; 20261114 |
| `58295446` | 2026-11-13 04:39:47 | liuqingmu096 | 0.679 | 有评分 | whitegg fuse33 liuqingmu096 retry1 of 58293228; unchanged verified v1; 20261113 |
| `58293228` | 2026-11-13 03:09:23 | liuqingmu096 | 0.679 | 有评分 | whitegg fuse33 verified ensemble; liuqingmu096 account; native50 sibling10; 20261113 |
| `58276197` | 2026-11-12 14:41:59 | mojohugo | 0.679 | 有评分 | whitegg fuse33 v1 forced real inference; native50 sibling10; 20261112 |
| `57833615` | 2026-10-31 09:03:49 | mojohugo | 0.661 | 有评分 | ATLAS20 R8 R4-only svc4 temperature 0.234793076894; mojohugo/ntl-r8-svc-calibrated-r4-20261031/1 |
| `57798970` | 2026-10-30 08:16:42 | mojohugo | 0.654 | 有评分 | ATLAS20 R7 expanded ensemble with calibrated StyleDistance 20% within four decoder groups; mojohugo/ntl-r7-four-style-expanded-20261030/4 |
| `57792375` | 2026-10-30 04:23:44 | mojohugo | 0.660 | 有评分 | ATLAS20 R6 expanded four-model blend 20%; mojohugo/ntl-expanded-r6-20261030/2 |
| `57778652` | 2026-10-29 17:49:23 | mojohugo | 0.662 | 有评分 | ATLAS20 cached CPU optimized atlas20-cachedcpu-20261030-r4; mojohugo/ntl-a20-r4-426fb9e8032af08b451e7134907b0a4f/1; sha 3c2d7486f7bc7232 |
| `57747844` | 2026-10-28 20:09:52 | mojohugo | 0.586 | 有评分 | Qwen3 run40740424907 final refit; notebook v2 |
| `57706812` | 2026-10-27 16:09:55 | liuqingmu096 | 0.636 | 有评分 | NTL fixed3b8e333 V1 all-members preview verified 2026-10-28 |
| `57569591` | 2026-10-23 18:37:11 | kiddingma | 0.601 | 有评分 | NTL-v7-361482022 |
| `57546749` | 2026-10-23 02:58:19 | kiddingma | 未返回 | 失败，见原错误 | Notebook NTL SOTA 0604 Dual LoRA Gaussian Hungarian \| Version 5 |
| `57503877` | 2026-10-22 01:03:50 | kiddingma | 未返回 | 失败，见原错误 | Notebook NTL SOTA 0604 Dual LoRA Gaussian Hungarian \| Version 2 |
| `57503817` | 2026-10-22 01:01:49 | kiddingma | 未返回 | 失败，见原错误 | Notebook v1_Vishak \| Version 2 |
| `57477922` | 2026-10-21 10:38:43 | kiddingma | 未返回 | 失败，见原错误 | Notebook LoRA x Adaptive Centering \| Version 2 |
| `57474150` | 2026-10-21 08:43:08 | kiddingma | 0.365 | 有评分 | Notebook notebook4c81c75745 \| Version 2 |
| `57440549` | 2026-10-20 13:51:07 | kiddingma | 0.053 | 有评分 |  |

### 附录 H 优化版逐阶段计时

| 阶段 | 状态 | 墙钟秒数 | 本次测试上限秒数 |
| --- | --- | --- | --- |
| semantic-groups | complete | 88.785 | 420 |
| deb | complete | 166.025 | 600 |
| q8v5 | complete | 73.945 | 600 |
| q14v5 | complete | 122.589 | 780 |
| n14b_clean | complete | 97.331 | 780 |
| ll8v5 | complete | 111.944 | 600 |
| q25_14 | complete | 83.445 | 600 |
| native | complete | 109.914 | 900 |

上述阶段时间包含模型加载和退出，不能按行数机械放大后称为真实 4,000 行耗时。完整运行报告中的线性估计会重复放大加载成本；该字段保留供审计，不作为已验证的隐藏全量时间。

### 附录 I 优化版每个模型的实际配置


```json
{
  "specs": [
    {
      "name": "deb",
      "type": "enc",
      "ckpt": "/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend/deberta_v3_large",
      "w": 1.0
    },
    {
      "name": "q8v5",
      "type": "l24",
      "path": "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
      "adapters": [
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q8v5",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q8v5_f0",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q8v5_f1"
      ],
      "keep": 24,
      "layers": [
        2,
        4,
        6,
        8,
        12,
        16,
        20,
        24
      ],
      "w": 2.0,
      "bs": 16,
      "max_len": 384,
      "twin": 0.35,
      "dp": true
    },
    {
      "name": "q14v5",
      "type": "l24",
      "path": "/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1",
      "adapters": [
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q14v5",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q14v5_f0",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/q14v5_f1"
      ],
      "keep": 27,
      "layers": [
        2,
        4,
        6,
        8,
        12,
        16,
        20,
        24
      ],
      "w": 3.0,
      "bs": 8,
      "max_len": 384,
      "twin": 0.35
    },
    {
      "name": "n14b_clean",
      "type": "l24",
      "path": "/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1",
      "adapters": [
        "/kaggle/input/datasets/nuridudak/ntl-q14-lora-v4",
        "/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend/lora_qwen3_14b"
      ],
      "adapter_weights": [
        0.72,
        0.28
      ],
      "keep": 27,
      "layers": [
        2,
        4,
        6,
        9,
        13,
        17,
        22,
        27
      ],
      "w": 2.5,
      "bs": 8,
      "max_len": 447,
      "twin": 0.35,
      "no_final_norm": true,
      "eos_id": 151643
    },
    {
      "name": "ll8v5",
      "type": "l24",
      "path": "/kaggle/input/models/alexchilton/llama-3-1-8b-base/transformers/default/1",
      "adapters": [
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/ll8v5",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/ll8v5_f0",
        "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5/ll8v5_f1"
      ],
      "keep": 24,
      "layers": [
        2,
        4,
        6,
        8,
        12,
        16,
        20,
        24
      ],
      "w": 1.0,
      "bs": 16,
      "max_len": 384,
      "twin": 0.35,
      "dp": true
    },
    {
      "name": "q25_14",
      "type": "l24",
      "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b/1",
      "adapters": [
        "/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend/lora_qwen25_14b"
      ],
      "keep": 32,
      "layers": [
        3,
        5,
        8,
        11,
        16,
        21,
        27,
        32
      ],
      "w": 1.0,
      "bs": 8,
      "max_len": 447,
      "twin": 0.35,
      "no_final_norm": true,
      "eos_id": 151643
    }
  ],
  "native_folds": [
    0,
    1,
    2
  ],
  "native_share": 0.4,
  "native_agreement": [
    0.8533333333333334,
    0.8,
    0.8533333333333334
  ],
  "gpu": {
    "ok": true,
    "torch_version": "2.11.0+cu128",
    "cuda_version": "12.8",
    "available": true,
    "device_count": 2,
    "devices": [
      "Tesla T4",
      "Tesla T4"
    ]
  }
}
```

### 附录 J 自训模型配置和结果索引

#### `/root/name-that-llm/pro-workspace/round5-encoder-review-a1/training-summary.json`


```json
{
  "observed_at": "2026-10-29T07:15:46.752128+00:00",
  "terminal_sha256": "5d83032f46f430fbf051e8d050355b352ca1042061f3ba583fcbf515f83bbd93",
  "status": "completed",
  "exit_code": 0,
  "started_at": "2026-10-29T05:55:31.161182+00:00",
  "finished_at": "2026-10-29T06:30:07.251551+00:00",
  "seconds": 2076.09,
  "cleanup_verified": true,
  "durable_models_verified": true,
  "experts": {
    "deberta": {
      "score": 0.49793686303753426,
      "fold_scores": [
        {
          "fold": 0,
          "plain_balanced_accuracy": 0.4796296296296295,
          "rows": 540
        },
        {
          "fold": 1,
          "plain_balanced_accuracy": 0.4874840357598978,
          "rows": 539
        },
        {
          "fold": 2,
          "plain_balanced_accuracy": 0.512962962962963,
          "rows": 540
        },
        {
          "fold": 3,
          "plain_balanced_accuracy": 0.49444444444444446,
          "rows": 540
        },
        {
          "fold": 4,
          "plain_balanced_accuracy": 0.5148148148148147,
          "rows": 540
        }
      ],
      "seconds": 2058.458392638713,
      "public_train_seconds": 1200.2834069877863,
      "public_rows": 72000,
      "final_train_seconds": 133.59213192015886,
      "final_checkpoint": "/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/final",
      "oof_npz_sha256": "5e34c7ceab075ec97839a00b64a7592f5ad7ed6f5b9f1909cb7194ccb6840aef"
    },
    "electra": {
      "score": 0.41493661446681585,
      "fold_scores": [
        {
          "fold": 0,
          "plain_balanced_accuracy": 0.4166666666666667,
          "rows": 540
        },
        {
          "fold": 1,
          "plain_balanced_accuracy": 0.4077266922094508,
          "rows": 539
        },
        {
          "fold": 2,
          "plain_balanced_accuracy": 0.38333333333333336,
          "rows": 540
        },
        {
          "fold": 3,
          "plain_balanced_accuracy": 0.43333333333333335,
          "rows": 540
        },
        {
          "fold": 4,
          "plain_balanced_accuracy": 0.43333333333333335,
          "rows": 540
        }
      ],
      "seconds": 1504.9794955067337,
      "public_train_seconds": 840.5051936097443,
      "public_rows": 72000,
      "final_train_seconds": 96.07188411056995,
      "final_checkpoint": "/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/final",
      "oof_npz_sha256": "b8bd6cd0c1bc6116e43df62e7f3679223af0fb9b3468bb46add25fe1d7a56686"
    }
  }
}
```

#### `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/final/meta.json`


```json
{
  "labels": [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3"
  ],
  "max_len": 512,
  "expert": "deberta",
  "stage": "official_refit_all",
  "complete_stage": true,
  "recipe": {
    "seed": 42,
    "fold_seed": 20261028,
    "folds": 5,
    "max_length": 512,
    "public_max_rows": 72000,
    "public_epochs": 1,
    "official_epochs": 3,
    "public_encoder_lr": 2e-05,
    "public_head_lr": 0.0001,
    "official_encoder_lr": 8e-06,
    "official_head_lr": 4e-05,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "schedule": "linear",
    "label_smoothing": 0.05,
    "head_dropout": 0.1,
    "clip_grad_norm": 1.0,
    "microbatch": 16,
    "gradient_accumulation": 4,
    "effective_batch": 64,
    "precision": "bfloat16_autocast_float32_parameters",
    "pooling": "last_hidden_state_masked_mean",
    "trainable_encoder": "all",
    "source_loss_weight": 1.0,
    "gradient_checkpointing": false,
    "loader_workers": 2,
    "omp_threads": 4,
    "expert_max_seconds": 10800,
    "parent_max_seconds": 10860,
    "smoke_optimizer_steps": 8,
    "smoke_query_rows": [
      8,
      36
    ],
    "heldout_selects_epoch": false,
    "query_centering": false
  },
  "official_rows_seen": 2699,
  "official_train_pool_rows": 2699,
  "official_train_pool_ids_sha256": "707edf645b0f09bba8f0983b5bc9f522595c95e8ca5b763b38babeb38200368d",
  "public_only_initialization": false,
  "full_R_initialization": false,
  "pooling": "last_hidden_state_masked_mean",
  "inference_dtype": "float32",
  "inference_inputs": [
    "text"
  ],
  "query_centering": false,
  "plan_sha256": "808a59e6af66de8c84609fb879005f5877471a069d0576eb381b8839f5f22253"
}
```

#### `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/final/meta.json`


```json
{
  "labels": [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3"
  ],
  "max_len": 512,
  "expert": "electra",
  "stage": "official_refit_all",
  "complete_stage": true,
  "recipe": {
    "seed": 42,
    "fold_seed": 20261028,
    "folds": 5,
    "max_length": 512,
    "public_max_rows": 72000,
    "public_epochs": 1,
    "official_epochs": 3,
    "public_encoder_lr": 2e-05,
    "public_head_lr": 0.0001,
    "official_encoder_lr": 8e-06,
    "official_head_lr": 4e-05,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "schedule": "linear",
    "label_smoothing": 0.05,
    "head_dropout": 0.1,
    "clip_grad_norm": 1.0,
    "microbatch": 16,
    "gradient_accumulation": 4,
    "effective_batch": 64,
    "precision": "bfloat16_autocast_float32_parameters",
    "pooling": "last_hidden_state_masked_mean",
    "trainable_encoder": "all",
    "source_loss_weight": 1.0,
    "gradient_checkpointing": false,
    "loader_workers": 2,
    "omp_threads": 4,
    "expert_max_seconds": 10800,
    "parent_max_seconds": 10860,
    "smoke_optimizer_steps": 8,
    "smoke_query_rows": [
      8,
      36
    ],
    "heldout_selects_epoch": false,
    "query_centering": false
  },
  "official_rows_seen": 2699,
  "official_train_pool_rows": 2699,
  "official_train_pool_ids_sha256": "707edf645b0f09bba8f0983b5bc9f522595c95e8ca5b763b38babeb38200368d",
  "public_only_initialization": false,
  "full_R_initialization": false,
  "pooling": "last_hidden_state_masked_mean",
  "inference_dtype": "float32",
  "inference_inputs": [
    "text"
  ],
  "query_centering": false,
  "plan_sha256": "808a59e6af66de8c84609fb879005f5877471a069d0576eb381b8839f5f22253"
}
```

#### `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44/final/meta.json`


```json
{
  "cfg": {
    "model_dir": "Qwen/Qwen3-14B-Base",
    "keep": 27,
    "layers": [
      2,
      4,
      6,
      9,
      13,
      17,
      22,
      27
    ],
    "max_len": 447,
    "r": 16,
    "alpha": 32,
    "ldrop": 0.05,
    "head_drop": 0.1,
    "lr_adapter": 5e-05,
    "lr_head": 0.0005,
    "wd": 0.01,
    "warmup": 0.1,
    "epochs": 2.0,
    "triplets_per_step": 6,
    "group": 3,
    "eval_bs": 32,
    "seed": 42,
    "grad_ckpt": true,
    "eos_id": 151643,
    "eos_note": "eos_id is the end token appended in training (<|endoftext|>)"
  },
  "labels": [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3"
  ],
  "request": {
    "schema_version": 2,
    "enabled": true,
    "experiment_id": "qwen3-public-v1-20261029-015557-d5cd5503",
    "task": "qwen3_two_stage",
    "seed": 42,
    "device": "cuda",
    "epochs": 3,
    "batch_size": 18,
    "learning_rate": 5e-05,
    "max_minutes": 480,
    "data_profile": "curated_v1",
    "pretrain_epochs": 1,
    "pretrain_learning_rate": 2e-05,
    "pretrain_max_rows": 120000,
    "refit_all": true
  },
  "stage": "official_refit_all",
  "epoch": 3,
  "step": 447,
  "initialization": "fresh base + random LoRA/heads",
  "inference_policy": {
    "name": "blind_text_v1",
    "cosine_threshold": 0.05,
    "max_group_size": 32,
    "min_group_size": 3,
    "centered_weight": 0.5
  },
  "data_sha256": "6baba592e086f15044f32665847c0df9cac4503da5102cc40bb3b253a75fc24e",
  "acceptance_only": false
}
```

#### `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/final/meta.json`


```json
{
  "cfg": {
    "model_dir": "Qwen/Qwen3-14B-Base",
    "keep": 27,
    "layers": [
      2,
      4,
      6,
      9,
      13,
      17,
      22,
      27
    ],
    "max_len": 447,
    "r": 16,
    "alpha": 32,
    "ldrop": 0.05,
    "head_drop": 0.1,
    "lr_adapter": 5e-05,
    "lr_head": 0.0005,
    "wd": 0.01,
    "warmup": 0.1,
    "epochs": 2.0,
    "triplets_per_step": 6,
    "group": 3,
    "eval_bs": 32,
    "seed": 42,
    "grad_ckpt": true,
    "eos_id": 151643,
    "eos_note": "eos_id is the end token appended in training (<|endoftext|>)"
  },
  "labels": [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3"
  ],
  "request": {
    "schema_version": 3,
    "enabled": true,
    "experiment_id": "qwen3-groupcv-v2-20261029-043240-a1",
    "task": "qwen3_groupcv",
    "seed": 42,
    "device": "cuda",
    "epochs": 5,
    "batch_size": 18,
    "learning_rate": 3e-05,
    "max_minutes": 300,
    "data_profile": "curated_v1",
    "pretrain_epochs": 1,
    "pretrain_learning_rate": 2e-05,
    "pretrain_max_rows": 120000,
    "refit_all": true,
    "cv_folds": 5,
    "control_epochs": 3,
    "control_learning_rate": 5e-05,
    "selection_min_delta": 0.005
  },
  "stage": "official_refit_all",
  "epoch": 3,
  "step": 447,
  "initialization": "shared fresh public stage1; independent fixed-recipe official fit",
  "inference_policy": {
    "name": "blind_text_v1",
    "cosine_threshold": 0.07,
    "max_group_size": 32,
    "min_group_size": 3,
    "centered_weight": 0.5
  },
  "data_sha256": "6baba592e086f15044f32665847c0df9cac4503da5102cc40bb3b253a75fc24e",
  "acceptance_only": false,
  "recipe": "control",
  "fold": null,
  "heldout_labels_select_epoch": false,
  "model_configuration": {
    "cfg": {
      "model_dir": "Qwen/Qwen3-14B-Base",
      "keep": 27,
      "layers": [
        2,
        4,
        6,
        9,
        13,
        17,
        22,
        27
      ],
      "max_len": 447,
      "r": 16,
      "alpha": 32,
      "ldrop": 0.05,
      "head_drop": 0.1,
      "lr_adapter": 5e-05,
      "lr_head": 0.0005,
      "wd": 0.01,
      "warmup": 0.1,
      "epochs": 2.0,
      "triplets_per_step": 6,
      "group": 3,
      "eval_bs": 32,
      "seed": 42,
      "grad_ckpt": true,
      "eos_id": 151643,
      "eos_note": "eos_id is the end token appended in training (<|endoftext|>)"
    },
    "labels": [
      "gpt-5.5",
      "gpt-5.6-luna",
      "gpt-5.6-sol",
      "gpt-5.6-terra",
      "gpt-6-astra",
      "gemini-3.1-pro-preview",
      "gemini-3.6-flash",
      "gemini-3.7-flash",
      "gemini-3.8-flash",
      "claude-opus-4-6",
      "claude-opus-4-8",
      "claude-opus-5",
      "claude-sonnet-5",
      "claude-fable-5",
      "qwen3.8-max",
      "glm-5.3",
      "glm-5.3-flash",
      "kimi-k3"
    ],
    "base_model": "/kaggle/input/models/qwen-lm/qwen-3/Transformers/14b-base/1",
    "base_config_sha256": "eb65771c8f108f88e4d87f771e1c9d40b7e35bc8070be8c97246c9446ccdf179",
    "source_meta_sha256": "293119872ff3f170ec85e8be02c80916e39048213de31cb45f8b84a2cf74d775",
    "lora_target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "attention": "flash_attention_2",
    "backbone_dtype": "bfloat16",
    "trainable_parameters": 44908580,
    "total_parameters": 9741243172
  }
}
```

#### `/workspace/name-that-llm/web-pro-checkpoints/managed-components-v4-20261029-1200-a1/development-decision.json`


```json
{
  "selected_recipe": "replayed_P",
  "development_gate_passed": false,
  "adopted": false,
  "adoption_scope": "development candidate only; does not replace the scored 0.636 anchor",
  "primary_comparison": "fixed primary minus replayed P",
  "primary_weights": {
    "P": 0.72,
    "S": 0.09,
    "E": 0.09,
    "H": 0.1
  },
  "diagnostic_core_weights": {
    "P": 0.8,
    "S": 0.1,
    "E": 0.1
  },
  "diagnostic_core_eligible_for_adoption": false,
  "delta": 0.003335819040517052,
  "plain_delta": 0.000738255033557067,
  "fold_wins": 4,
  "fold_deltas": [
    {
      "fold": 0,
      "blind_delta": 0.0018518518518518823,
      "plain_delta": 0.0018518518518518823
    },
    {
      "fold": 1,
      "blind_delta": 0.0037037037037037646,
      "plain_delta": -0.0018518518518518823
    },
    {
      "fold": 2,
      "blind_delta": -0.0018518518518518823,
      "plain_delta": 0.0037037037037038756
    },
    {
      "fold": 3,
      "blind_delta": 0.0037037037037036535,
      "plain_delta": -0.0018518518518517713
    },
    {
      "fold": 4,
      "blind_delta": 0.0092592592592593,
      "plain_delta": 0.0018518518518516602
    }
  ],
  "bootstrap": {
    "unit": "prompt",
    "paired": true,
    "seed": 20261029,
    "samples_requested": 2000,
    "samples_valid": 2000,
    "samples_missing_classes": 0,
    "missing_class_policy": "exclude replicate; fail if none remain",
    "statistic": "balanced_accuracy from summed per-prompt row-confusion counts",
    "confidence_level": 0.95,
    "lower": -0.0007407407407408223,
    "upper": 0.007777777777777664,
    "mean_delta": 0.0034064748048234176
  },
  "criteria": {
    "blind_delta_at_least_0_005": false,
    "paired_prompt_bootstrap_95_lower_above_zero": false,
    "at_least_three_fold_wins": true,
    "plain_delta_at_least_minus_0_005": true,
    "stress_mean_delta_at_least_minus_0_005": true,
    "stress_worst_case_delta_at_least_minus_0_02": true
  },
  "stress_mean_delta": 0.0003941230358772379,
  "stress_worst_case_delta": -0.012894074010574741,
  "deployment_ready": false,
  "kaggle_t4_full_4000_runtime_gate_passed": null,
  "independent_scored_anchor_comparison_passed": null,
  "original_scored_0_636_anchor_retained": true,
  "score_scope": "prompt-held-out development comparison; no Kaggle score or independent anchor superiority claim"
}
```

### 附录 K 项目训练权重文件索引

以下按本轮只读目录清单列出权重和元数据文件，未复制大权重。相同权重可能存在备份副本，不将重复文件数当成独立训练模型数。

| 服务器路径 | 字节数 | 换算 |
| --- | --- | --- |
| `/workspace/name-that-llm/web-pro-checkpoints/backup-acceptance-two-stage-5a4005bc1814/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/backup-acceptance-two-stage-5a4005bc1814/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/final/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/final/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-0/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-0/checkpoint/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-1/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-1/checkpoint/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-2/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-2/checkpoint/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-3/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-3/checkpoint/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-4/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/fold-4/checkpoint/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/public-initialization/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/deberta/public-initialization/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/final/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/final/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-0/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-0/checkpoint/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-1/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-1/checkpoint/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-2/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-2/checkpoint/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-3/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-3/checkpoint/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-4/checkpoint/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/fold-4/checkpoint/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/public-initialization/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/full/electra/public-initialization/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/smoke/deberta/public-initialization/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/smoke/deberta/public-initialization/encoder/model.safetensors` | 1,736,094,384 | 1.617 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/smoke/electra/public-initialization/head.safetensors` | 73,936 | 0.000 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared/smoke/electra/public-initialization/encoder/model.safetensors` | 1,336,413,848 | 1.245 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44/final/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44/final/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44/holdout-best/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-21109a6bb9e6425f92ab41c588bc6e44/holdout-best/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-0/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-0/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-1/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-1/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-2/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-2/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-3/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-3/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-4/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/ensemble/member-4/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/final/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/final/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-0/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-0/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-1/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-1/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-2/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-2/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-3/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-3/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-4/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/candidate/fold-4/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-0/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-0/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-1/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-1/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-2/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-2/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-3/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-3/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-4/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/folds/control/fold-4/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/stage1/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/job-3130182971224c338b0cb126a2033198/stage1/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/managed-components-v4-20261029-1200-a1/control/heads.safetensors` | 6,227,664 | 0.006 GiB |
| `/workspace/name-that-llm/web-pro-checkpoints/managed-components-v4-20261029-1200-a1/control/adapter/adapter_model.safetensors` | 173,456,688 | 0.162 GiB |

### 附录 L 随包 Notebook 和源代码

| 路线 | 本包文件 | SHA256 |
| --- | --- | --- |
| `当前优化GPU版V2` | `Notebooks\当前优化GPU版V2\optimized.ipynb` | `28fe388a3c9604d5c01fd946b582cf750ec320a3832ec6a21cf16c83487d53fd` |
| `原ozertuuV37超时版勿提交` | `Notebooks\原ozertuuV37超时版勿提交\ozertuu-v37-reproduction.ipynb` | `1cb8f3efe80909554427bcbd635f73b3fd7e323180ec49cef7052b3edcd2798e` |
| `DhananjayV10已评分0689` | `Notebooks\DhananjayV10已评分0689\dhananjay-v10-reproduction.ipynb` | `21ccb44ea968a43805d0e4a2e6d6fbdf1f320e6be4314de9335be552a1f93a07` |
| `whitegg已评分0679` | `Notebooks\whitegg已评分0679\ntl-whitegg-fuse33-real-inference-reproduction.ipynb` | `1715e2cc4ca6cb494a4313f2db25d47fbc423a6d1f6d1aed5ab9ae0f19e3174e` |
| `完整复现已评分0636` | `Notebooks\完整复现已评分0636\ntl_complement_fixed.ipynb` | `06b57cb41cf59b70678271877b7a08a748ddf6f7d5216aaad22485ac3e2a92f0` |
| `ATLAS队伍R4参考` | `Notebooks\ATLAS队伍R4参考\ntl-a20-r4-426fb9e8032af08b451e7134907b0a4f.ipynb` | `fe3f8bfbf2ba7ddf49cdd1714ba2d0a6869472876ac13ba75590b95f876a45a1` |
| `自训项目模板需核对版本` | `自训项目当前源码\kaggle\submit.ipynb` | `bc6463a8f7720317467fb1b0f6a8b21b5bcab7f4cd712a1ad0a5710df990fd75` |

另附服务器自训项目当前源码 `27` 个文本文件，该目录不是 Git 工作树，本轮未独立取得工作副本的 commit，原目录为 `/root/name-that-llm/pro-workspace/project`。它与历史 job 的具体 commit 需逐项核对；不能只因为文件同名就认定版本一致。

`当前优化版代码` 保存本轮完整运行脚本、构建入口、回归检查、元数据和操作入口。`证据` 保存环境、训练摘要、提交详情和实际运行日志。最早 2026 年 10 月 28 日的大交付包已存在于 `E:\kaggle`，本包不重复复制该大压缩包。

### 附录 M 证据优先级和恢复检查单

1. 官方提交详情决定是否成功评分，提交编号和保存版本必须匹配。
2. 训练终态、结果、权重哈希和实际文件共同决定模型训练完成情况。
3. 当前环境快照决定软件和硬件配置，旧文档只说明当时状态。
4. 公开样例、75 行计时和 OOF 是不同证据，不互相代替。
5. 迁移时保留基础模型版本、tokenizer、EOS、标签顺序、适配器、分类头和校准配置。
6. 新实验使用新的 ID，保留已有有效提交、原始回执和失败记录。

