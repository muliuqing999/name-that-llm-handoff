# 网页 Pro → GitHub → 双 PPU：异步训练与结果回传

网页 Pro 提交配置后结束对话；Windows Runner 经 SSH 调用两张 PPU 独立训练，完成/失败报告自动写回 GitHub。数小时后，在另一段 Pro 对话读取对应报告。执行链路不调用 OpenAI API，也不依赖聊天窗口持续打开。

- [发起训练与另开对话读结果：两份提示词](docs/WEB_PRO.md)
- [首轮训练交接：数据、验证和参数](docs/TRAINING_HANDOFF.md)
- [已部署公开数据清单与哈希](docs/public-data-manifest.json)
- [Kaggle 离线提交要求与模板](docs/KAGGLE_SUBMISSION.md)
- [历史链路与性能验收记录](docs/VERIFICATION_2026-10-28.md)
- [最新结果](https://github.com/mojohugo/web-pro-gpu-runner/blob/results/reports/latest.json)
- [Actions](https://github.com/mojohugo/web-pro-gpu-runner/actions)

## 当前训练流程

`qwen3_two_stage` 从新 LoRA/分类头开始：公开数据预训练 → 固定官方分组留出验证 → 按最佳微调轮数用全部官方数据重训。已部署公开候选 183,298 条，首轮按类限额选中 113,967 条。数据中保留来源，排除多项已知风险；不保证所有原始标签独立核验，也不保证增加数据一定提分。

预测只接收文本。盲分组阈值在官方训练部分拟合，验证和离线推理共用函数，不使用真实验证 prompt_id。报告区分有留出分数的 selection_checkpoint 和全量重训后的 checkpoint。新指标不能直接与旧的真实分组诊断分数比较。

这是两阶段 Qwen3 候选组件，尚非完整 0.622 的 Qwen3、Qwen2.5、DeBERTa、SVC 集成。推理脚本和 Kaggle Notebook 模板已提供，但仍须训练后完成 Kaggle GPU 的离线全量耗时/显存验收。

## 发起任务

正式触发文件是 main:experiments/request.json。网页 Pro 读取 [two-stage.example.json](experiments/two-stage.example.json)，使用新 experiment_id 提交到触发文件。修改代码、说明或示例不会启动训练。

| task | schema | 用途 | 单任务上限 |
|---|---|---|---|
| qwen3_two_stage | 2 | 公开预训练、官方验证、可选全量重训 | 720 分钟，首轮配置 480 |
| qwen3_lora | 1 | 仅官方数据基线，已改为盲验证 | 240 分钟 |
| ppu_benchmark | 1 | 固定重复样本的双卡链路验收 | 240 分钟 |
| smoke_mlp | 1 | CPU/GPU 合成例子 | 240 分钟 |

PPU 任务固定两卡、全局 batch 18；device=cuda 指厂商兼容接口。**不要向服务器现有 PPU 环境安装 NVIDIA PyTorch。** requirements.txt 只服务旧 smoke_mlp。

同一 experiment_id 只能领取一次；失败重试必须用新 ID。工作流串行，服务器有 GPU 文件锁和独立超时。进度写入 Actions 日志，结果包括源 commit、完整请求、数据 SHA、主指标、每类召回率、分阶段耗时与权重哈希。

## 效率与保存

预训练分词有校验缓存；外部样本长度分桶减少 padding。训练前检查梯度检查点开/关的梯度一致性和最大长度显存，保留 15% 余量；仅在校验通过且前向/反向测速至少快 5% 时切换。相同软件、数据、模型与硬件复用最长 7 天的性能缓存，命中仍做显存压力检查。测速不更新参数，耗时不等于整轮提速。

训练在容器快盘进行，保存 stage1、每 500 步部分权重、holdout best 与 final。正式完成时将最终和验证候选复制到配置的持久目录并核验最终权重哈希。部分权重未包含优化器/完整随机状态，不支持精确自动断点续训。

## 异步运行条件

当前 Windows Runner、校园网 Tailscale 跳板、PPU 和 Runner 所用本机网络代理需保持在线。关闭聊天不影响后台任务；关机/网络中断可能影响执行或回传。报告位于 results 分支，Actions artifact 只保存小型 JSON，大权重留在训练主机与持久目录。

目前没有无人值守地自动唤醒网页 Pro 的机制。你在新的对话里读报告、确认下一轮；执行器不会自行无限调参或提交 Kaggle。

## 验证与生成数据

测试入口：`python -m unittest discover -s tests -v`。数据准备脚本：scripts/prepare_public_data.py。离线预测入口：scripts/predict_offline.py，提交模板：kaggle/submit.ipynb。

付费 API 造数尚未开启。已有公开数据先做对照训练；后续准确模型版本的定向补数方案见 [DATA_GENERATION.md](docs/DATA_GENERATION.md)。
