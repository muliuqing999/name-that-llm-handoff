# 网页 Pro：发起训练与另开对话读结果

这是异步流程。第一次 Pro 对话只提交任务并返回回执；训练由 GitHub Runner 和 PPU 主机独立执行，持续数小时也不需要保留聊天窗口。完成或失败后，报告自动写入 results 分支。另一次对话读取对应报告，才能分析结果或发起下一轮。不要在一次对话里循环等待长训练，也不要声称聊天会自行在几小时后唤醒。

仓库：`mojohugo/web-pro-gpu-runner`。使用当前网页 Pro 与 GitHub 连接，不调用 Codex、Dot、Work 或 OpenAI API 替代决策。GitHub 的实际读写权限需要以工具执行结果为准；先前读写和训练链路已实测通过。

## 第一次对话：只发起一轮

将下列提示词发给网页 Pro：

```text
请通过 GitHub 管理 mojohugo/web-pro-gpu-runner。保持当前 Pro 模式，不调用 Codex、Dot、Work 或 OpenAI API。

先读取 main 的 docs/WEB_PRO.md、docs/TRAINING_HANDOFF.md、docs/public-data-manifest.json、experiments/two-stage.example.json、experiments/request.json，以及 results 的 reports/latest.json。
先检查是否有正在运行或排队的训练。若有，只返回该任务状态，不新增任务。若工具无法确认运行状态，明确说明接单状态未确认，不重复提交来试探。

使用 two-stage.example.json 发起一轮：公开数据预训练 1 epoch、最多 120000 条按类别限额抽样，官方微调最多 3 epoch；根据固定留出验证选择 epoch，再从 stage1 权重出发用全部官方数据重新微调。保留 refit_all=true、seed=42、预训练学习率2e-5、微调学习率5e-5、batch_size=18、max_minutes=480。
用新的唯一 experiment_id，只修改并提交 main:experiments/request.json。此前的历史分数使用不同验证口径，不能作为新方案的提分证据。训练开始时执行器会做效率检查和最大长度显存检查。

拿到真实 commit SHA 后，不等待训练完成。若工具能读取 Actions，确认 queued/in_progress 并记录真实 run_id、run_url；没有这些信息就如实标记，不编造链接。将 experiment_id、request_commit_sha、run_id（未知则null）、run_url（未知则null）及“等待训练结果”状态写入 main:experiments/latest_receipt.json，这个文件不触发训练。
最后返回这份回执和后续读结果的方法，然后结束本次对话。不要自动提交第二轮，不要启动付费造数，也不要提交 Kaggle。
```

修改示例文件不会启动训练，只有对 main 的 `experiments/request.json` 的提交或手动 workflow_dispatch 会触发。确认未运行时再提交；同一 experiment_id 在执行器已领取后会被跳过。失败重试也使用新的 ID，避免误运行同一个耗时任务。

## 第二次对话：只读取、评估结果

数小时后可以在全新 Pro 对话发送：

```text
请通过 GitHub 查看 mojohugo/web-pro-gpu-runner 上一次训练结果。先读 main:docs/WEB_PRO.md、docs/TRAINING_HANDOFF.md、experiments/latest_receipt.json；若回执不存在，用 experiments/request.json 的 experiment_id 定位。
读取 Actions 状态，以及 results:reports/latest.json 和对应 reports/runs/<run_id>-<attempt>.json。核对 experiment_id、commit_sha、request、task、status，必须对应我发起的那一轮。latest 是旧实验时不能当成这轮结果。
如果仍运行，报告当前阶段/进度后结束对话，不重复提交。如果失败或 skipped，解释原因和原任务位置，不将部分权重当作训练完成。
如果 completed，确认 acceptance_only=false，分析 primary_metric、best_epoch、每类 recall、plain 与 blind 指标、阶段耗时、显存及数据版本。验证分数属于 selection_checkpoint；全量重训的 checkpoint 没有独立验证分数。不要把新指标和旧的真实分组诊断分数直接比较。
返回模型位置、durable_checkpoint、权重 SHA、适合与否及下一轮建议。这次先不发起新训练，也不提交 Kaggle；下一轮必须由我明确要求。
```

## 任务配置

`qwen3_two_stage` 使用 schema_version=2。固定两张 PPU、全局 batch 18、device=cuda（厂商兼容接口）。`data_profile` 仅允许已部署的 `curated_v1`，网页不能指定任意服务器路径。`pretrain_max_rows` 为 1,800–270,000，按类限额且不重复采样；预训练 epoch 为 1–2，官方微调 epoch 为 1–5，max_minutes 为 1–720。refit_all 为布尔值。首轮使用示例的 480 分钟上限。

旧任务 `qwen3_lora`、`ppu_benchmark`、`smoke_mlp` 继续使用 schema_version=1，最长 240 分钟。benchmark 和 acceptance_only 的短测都不是有效训练成绩。

## 机器和报告

- Windows Runner 经 Tailscale 跳板访问 PPU。两台 Windows、PPU 和 Runner 使用的本机网络代理要持续在线；关闭聊天不影响任务，关机/断网会影响执行或回传。
- 每个训练任务有独立输出目录、服务器 GPU 锁及超时。运行进度会写入 Actions 日志；同一实验 ID 不会再次启动。
- 结果在 `results:reports/runs/<run_id>-<attempt>.json`，latest 只是最近一次报告，不代表最好模型。报告包含完整请求、源 commit、数据哈希、验证口径、每类指标和权重位置。
- 正常完成后最终权重会复制到配置的持久目录并校验 SHA；原始日志和阶段权重在服务器。若持久复制失败，任务会报告失败，保留本地权重供排查。
- 每 500 个更新步保存部分权重。这不包含优化器和完整随机状态，不能声称支持精确自动断点续训。
- Actions artifact 只保存小型 JSON，不上传大权重。领取任务不依赖网页会话存活；回传仍依赖 Runner 和网络在线。

## 训练之后

按 [Kaggle 提交准备](KAGGLE_SUBMISSION.md) 导出最终权重，挂载基础模型和离线依赖，在 Kaggle 完成显存、全量运行时间与提交格式验收。当前两阶段 Qwen3 是一个候选组件，不等于完整 0.622 多模型集成，也不保证榜单提升。
