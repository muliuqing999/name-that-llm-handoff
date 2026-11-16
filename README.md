# Name That LLM：复现、推理优化与训练交接

机器学习实践项目资料归档，整理日期 **2026-11-16**。包含环境配置、模型资源清单、自训实验记录、Kaggle Notebook、推理代码、故障分析和运行证据。

## 从这里开始

- [完整交接报告（Markdown，可在线阅读）](NameThatLLM_完整交接报告.md)
- [完整交接报告（HTML，下载后用浏览器打开）](NameThatLLM_完整交接报告.html)
- [完整公开版 ZIP](downloads/NameThatLLM_Public_Handoff_20261116.zip)
- [最新优化 Notebook](Notebooks/当前优化GPU版V2/optimized.ipynb)
- [优化代码与检查](当前优化版代码/)
- [项目自训源码与模板](自训项目当前源码/)
- [环境、训练、提交及运行证据](证据/)
- [SHA256 文件清单](文件清单与SHA256.json)

## 已验证的结果与边界

| 项目 | 结果 | 说明 |
| --- | --- | --- |
| Dhananjay V10 方案复现 | 公榜 0.689 | 历史提交 58337205 |
| whitegg 方案复现 | 公榜 0.679 | 历史复现记录见报告 |
| 早期完整复现 | 公榜 0.636 | 历史提交 57706812 |
| 最新有预算限制的推理优化 V2 | 已提交，尚未确认分数 | 提交 58421503；归档查询时为 PENDING |
| GPU 实测 | 主流程 959.8848 秒 | 75 行计时样本；不能作为独立准确率验证 |
| Qwen3 项目自训 5 折实验 | control blind OOF 0.6097663435 | 与 Kaggle 公榜分数分别记录 |

最新优化主要解决 CPU 版 PyTorch、隐藏评测超时、子进程回收和预测结果保底问题，使用两张 Tesla T4 实测。它是推理流程优化，没有在本轮重新训练 LoRA。提交状态是带查询时间的历史快照，不能据此推断当前 Kaggle 状态。

## 内容范围

资料包含原交接包的 8 份 Notebook 文件（其中 1 份是优化版镜像）、27 份项目自训源码和模板、详细模型清单及 19 条历史提交记录；另补入构建脚本所需的固定原版 Notebook 与源码。报告区分公开方案复现、项目自训、计时测试和公榜成绩；现有证据不能把全部团队工作都归为个人原创。

模型权重体积较大，原始交接包就未包含权重。本仓库保留来源、版本、配置和历史目录，运行时需要按来源及相应权限自行准备。包含的 submission.parquet 是运行生成的预测结果，不包含训练语料或原始竞赛数据集。

## 复现提示

1. 先读报告中的环境、资源挂载和方法章节，再选择对应 Notebook；目录名含“超时”或“诊断失败”的版本用于故障分析。
2. 最新 Kaggle 入口：[liuqingmu096 / ntl-ozertuu-bounded-optimized-20261116](https://www.kaggle.com/code/liuqingmu096/ntl-ozertuu-bounded-optimized-20261116)。私有 Notebook 或模型资源可能要求资源方授权。
3. Kaggle 代码依赖 Notebook 元数据中的输入挂载、指定 GPU 镜像及第三方模型，普通本机环境不能直接替代。
4. 本机操作脚本使用已有 Kaggle 授权且校验账号。复用时要自行配置认证和目标账号；本仓库未配置任何凭据，也不会自动执行提交。
5. 自训项目目录是服务器源码快照；完整运行还可能需要报告中列出的服务及资源。当前快照没有可验证的 Git HEAD，历史实验提交信息以证据为准。

构建和检查命令（在仓库根目录运行；需已安装 numpy、pandas、pyarrow）：

```powershell
python '当前优化版代码/build_notebook.py'
python -m unittest discover -s '当前优化版代码/tests' -p test_runtime.py -v
```

构建命令会重新写入 runtime 和 account-copy；归档中的固定原版放在 `ozertuu-reproduction-20261115`。

## 公开版处理

原始本地交接包保持不变。公开版将服务器连接地址、本机用户路径和 SSH 密钥位置替换为占位符，保留模型相对关系和训练记录。报告中的 `TRAIN_SERVER_HOST`、`C:\PROJECT_ROOT`、`C:\PATH_TO_SSH_KEY` 需要按自己的环境配置。历史证据里的哈希代表当时产物；公开版文件的当前校验值请使用本仓库根目录清单。

本资料是技术交接归档，尚未排入课程实验报告或答辩 PPT 模板。

## 来源与署名

- 竞赛：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- 最新优化所基于的公开方案：[ozertuu / Dual SVC Residual Refinement APEX](https://www.kaggle.com/code/ozertuu/0-689-dual-svc-residual-refinement-apex)
- 历史复现涉及 Dhananjay、whitegg 等作者，具体链接和版本见报告、Notebook 原始说明和元数据。
- 原始 Notebook、模型和数据的权利及许可属于各自权利人，使用和再分发应遵循原始来源许可。本仓库没有以统一许可证重新授权第三方内容。
