# Kaggle 离线提交准备

官方说明：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm/overview/citation)，核对日期 2026-10-29。

比赛提交的是可重新运行的 Kaggle Notebook。Notebook 读取 `/kaggle/input/competitions/name-that-llm/test.csv`，在 `/kaggle/working/submission.parquet` 写出 `id,model` 两列。每个测试 ID 恰好一次，不能有空值、重复 ID 或非官方的模型标签。CPU/GPU Notebook 均要求 9 小时以内、Internet 关闭。交互式样例很小，不能据此推断隐藏约 4,000 条测试数据的耗时。

## 需要挂载的资源

- 完整训练后报告 `checkpoint` 指向的 final 目录：LoRA adapter、heads.safetensors、meta.json，以及随权重保存的 predict_offline.py、blind_inference.py、model_features.py、validate_submission.py。
- 基础模型 Qwen3-14B-Base 和同版本分词器。Notebook 默认路径为 `/kaggle/input/models/qwen-lm/qwen-3/Transformers/14b-base/1`，挂载后检查实际路径。
- 离线依赖。已验证的训练组件版本为 Transformers 4.57.6、PEFT 0.17.1；推理还需要标准 NVIDIA/CPU PyTorch、Accelerate、safetensors、NumPy、pandas、PyArrow、scikit-learn。用 Kaggle Dependency Manager 或提前附加 wheels 处理缺失依赖；**不要复制 PPU 厂商 PyTorch 到 Kaggle**。实际 Kaggle 镜像组合仍需独立测试。

无需在评分 Notebook 重跑数小时训练，也无需将全部公开训练语料加载到推理过程。训练来源及可复现记录需继续保留。

## 提交流程

1. 正式训练 completed 且 acceptance_only=false 后，核对最终权重哈希，上传该 checkpoint 为本人 Kaggle Dataset/Model。
2. 导入 [submit.ipynb](../kaggle/submit.ipynb)，挂载候选权重、基础模型及必要依赖。若附加多个候选，显式指定 CHECKPOINT，不能随机取一个。
3. 启用合适 GPU，关闭网络。推理入口支持自动分配到多个 GPU，给激活保留显存；在不支持原生 BF16 的 NVIDIA GPU 上使用 FP16。单卡是否放得下、CPU offload 是否过慢，需要实际验收。
4. 先用公开/自有的代表性文本做约 4,000 条的离线压力测试，检查最长文本、峰值显存和端到端时长。不能使用隐藏测试标签调参。量化或多模型集成需要重新验证精度及总预算。
5. 正式运行并保存 Notebook 版本，确认生成 submission.parquet，再通过比赛提交入口提交该 Notebook 的输出。

当前交付包含推理入口、Notebook 模板和输出校验器；尚未在 Kaggle GPU 上执行，也没有正式训练的新模型或新榜单成绩。PPU 短测通过不等于 9 小时 Kaggle 验收通过。
