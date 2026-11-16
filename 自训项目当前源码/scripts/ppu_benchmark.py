"""Bounded real-weight Qwen3 LoRA training and DDP benchmark; saves no model."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model_features import feature_tensor

BASE = os.environ["NTL_BASE_MODEL"]
CHECKPOINT = os.environ["NTL_WARM_START"]
DATA = os.environ["NTL_TRAIN_CSV"]


def emit(event, **values):
    print(json.dumps({"event": event, "rank": int(os.environ.get("RANK", 0)), **values}), flush=True)


def initialize():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(8)
    torch.cuda.set_device(local)
    torch.manual_seed(42)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=900))
    return rank, world, local


def comm_probe(rank, world, local, output):
    t = torch.tensor([float(rank + 1)], device=f"cuda:{local}")
    if world > 1:
        dist.all_reduce(t)
    torch.cuda.synchronize()
    assert t.item() == world * (world + 1) / 2
    # Test backward as well as forward through the installed attention kernel.
    from flash_attn import flash_attn_func
    q = torch.randn(3, 448, 40, 128, device=f"cuda:{local}", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(3, 448, 8, 128, device=f"cuda:{local}", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    y = flash_attn_func(q, k, v, causal=True)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    assert all(x.grad is not None and torch.isfinite(x.grad).all().item() for x in (q, k, v))
    result = {"ok": True, "world_size": world, "all_reduce_sum": t.item(), "flash_attention_bf16_forward_backward": True}
    emit("communication_and_attention_pass", **result)
    if rank == 0:
        Path(output).write_text(json.dumps(result, indent=2))


class Classifier(nn.Module):
    def __init__(self, backbone, cfg, head_weights, local):
        super().__init__()
        self.backbone = backbone
        self.layers = cfg["layers"]
        hidden = backbone.config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.layers])
        self.plain = nn.Linear(hidden * len(self.layers), 18)
        self.centered = nn.Linear(hidden * len(self.layers), 18)
        self.dropout = nn.Dropout(cfg.get("head_drop", 0.1))
        self.centered_loss_weight = 1.0
        if head_weights is not None:
            self.norms.load_state_dict({k.removeprefix("norms."): v for k, v in head_weights.items() if k.startswith("norms.")})
            self.plain.load_state_dict({k.removeprefix("plain."): v for k, v in head_weights.items() if k.startswith("plain.")})
            self.centered.load_state_dict({k.removeprefix("centered."): v for k, v in head_weights.items() if k.startswith("centered.")})
        for module in (self.norms, self.plain, self.centered):
            module.to(device=f"cuda:{local}", dtype=torch.float32)

    def forward(self, ids, mask, labels):
        features = feature_tensor(self, ids, mask)
        grouped = features.reshape(-1, 3, features.shape[-1])
        centered = (grouped - grouped.mean(1, keepdim=True)).reshape_as(features)
        return nn.functional.cross_entropy(self.plain(self.dropout(features)), labels) + self.centered_loss_weight * nn.functional.cross_entropy(self.centered(self.dropout(centered)), labels)


def train(args, rank, world, local):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    meta = json.loads((Path(CHECKPOINT) / "meta.json").read_text())
    cfg = meta["cfg"]
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config.num_hidden_layers = cfg["keep"]
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:cfg["keep"]]
    emit("model_load_start", attention=args.attention)
    started = time.perf_counter()
    backbone = AutoModel.from_pretrained(BASE, config=config, torch_dtype=torch.bfloat16,
        device_map={"": local}, low_cpu_mem_usage=True, attn_implementation=args.attention, local_files_only=True)
    backbone.norm = nn.Identity()
    lora = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["ldrop"], bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    backbone = get_peft_model(backbone, lora)
    adapter_file = Path(CHECKPOINT) / "adapter/adapter_model.safetensors"
    loaded = set_peft_model_state_dict(backbone, load_file(str(adapter_file)))
    assert not loaded.unexpected_keys, loaded.unexpected_keys
    assert not [k for k in loaded.missing_keys if "lora_" in k], "Missing LoRA weights"
    if args.checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    model = Classifier(backbone, cfg, load_file(str(Path(CHECKPOINT) / "heads.safetensors")), local).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    emit("model_load_done", seconds=time.perf_counter() - started,
         trainable_parameters=sum(p.numel() for p in trainable), total_parameters=sum(p.numel() for p in model.parameters()))

    groups = {}
    with open(DATA) as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["prompt_id"], []).append(row)
    rng = random.Random(42)
    candidates = [v for v in groups.values() if len(v) >= 3]
    rng.shuffle(candidates)
    global_rows = []
    for group in candidates[:6]:
        global_rows.extend(rng.sample(group, 3))
    assert len(global_rows) == 18 and 18 % world == 0
    local_rows = global_rows[rank * (18 // world):(rank + 1) * (18 // world)]
    assert len(local_rows) % args.microbatch == 0 and args.microbatch % 3 == 0
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = cfg["eos_id"]
    tokens = tokenizer([r["text"] for r in global_rows], truncation=True, max_length=cfg["max_len"], add_special_tokens=False)["input_ids"]
    tokens = [row + [cfg["eos_id"]] for row in tokens]
    local_tokens = tokens[rank * (18 // world):(rank + 1) * (18 // world)]
    label_map = {label: i for i, label in enumerate(meta["labels"])}
    batches = []
    batch_lengths = []
    padded_tokens = 0
    for start in range(0, len(local_rows), args.microbatch):
        padding_kwargs = {"padding": "max_length", "max_length": cfg["max_len"] + 1} if args.padding == "max" else {"padding": True, "pad_to_multiple_of": 8}
        encoded = tokenizer.pad({"input_ids": local_tokens[start:start + args.microbatch]}, return_tensors="pt", **padding_kwargs)
        batch_lengths.append(encoded["input_ids"].shape[1])
        padded_tokens += encoded["input_ids"].numel()
        batches.append((encoded["input_ids"].to(local), encoded["attention_mask"].to(local),
            torch.tensor([label_map[r["model"]] for r in local_rows[start:start + args.microbatch]], device=f"cuda:{local}")))
    padded_total = torch.tensor(padded_tokens, device=f"cuda:{local}")
    if world > 1:
        dist.all_reduce(padded_total)
    padded_tokens = padded_total.item()
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01, foreach=False)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False, find_unused_parameters=False)
    emit("training_start", microbatch=args.microbatch, accumulation=len(batches), global_batch=18,
         padding=args.padding, batch_sequence_lengths=batch_lengths, padded_tokens_global=padded_tokens,
         sequence_length_cap=cfg["max_len"] + 1, mean_actual_tokens=statistics.mean(map(len, tokens)))
    torch.cuda.reset_peak_memory_stats()
    times = []
    losses = []
    for step in range(args.warmup + args.steps):
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for batch_index, (ids, mask, labels) in enumerate(batches):
            ctx = model.no_sync() if world > 1 and batch_index != len(batches) - 1 else contextlib.nullcontext()
            with ctx:
                loss = model(ids, mask, labels) / len(batches)
                loss.backward()
            step_loss += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
        assert torch.isfinite(norm).item(), "Non-finite gradients"
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=f"cuda:{local}")
        if world > 1:
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed = elapsed_tensor.item()
        emit("step", step=step, warmup=step < args.warmup, seconds=elapsed, loss=step_loss, grad_norm=norm.item())
        if step >= args.warmup:
            times.append(elapsed)
            losses.append(step_loss)
    # A compact, deterministically selected subset verifies synchronized updates.
    signature = torch.stack([p.detach().flatten()[:32].float().sum() for p in trainable[:12]])
    sync_error = 0.0
    if world > 1:
        signatures = [torch.empty_like(signature) for _ in range(world)]
        dist.all_gather(signatures, signature)
        sync_error = max((signature - other).abs().max().item() for other in signatures)
        assert sync_error < 1e-5, sync_error
    report = {"ok": True, "world_size": world, "model": "Qwen3-14B-Base first 27 layers",
        "dtype": "bfloat16 backbone, float32 adapters and classifier", "attention": args.attention,
        "checkpointing": args.checkpointing, "global_batch": 18, "microbatch_per_rank": args.microbatch,
        "accumulation_per_rank": len(batches), "sequence_length_cap": cfg["max_len"] + 1,
        "padding": args.padding, "batch_sequence_lengths_per_rank": batch_lengths, "padded_tokens_global": padded_tokens,
        "mean_actual_tokens": statistics.mean(map(len, tokens)), "step_seconds": times,
        "median_step_seconds": statistics.median(times), "samples_per_second": 18 / statistics.median(times),
        "padded_tokens_per_second": padded_tokens / statistics.median(times),
        "peak_allocated_GiB_per_rank": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_GiB_per_rank": torch.cuda.max_memory_reserved() / 1024**3,
        "losses": losses, "parameter_sync_max_error": sync_error,
        "data_sha256": hashlib.sha256(Path(DATA).read_bytes()).hexdigest(),
        "scope": "Fixed 18 real passages, grouped into same-prompt triplets. Training benchmark, not accuracy evaluation."}
    report["versions"] = {"torch": torch.__version__, "cuda_wrapper": torch.version.cuda}
    if rank == 0 and args.save_checkpoint:
        module = model.module if world > 1 else model
        torch.save({n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad}, args.save_checkpoint)
        report["scope"] += " Trainable parameter snapshot saved only to verify storage."
    emit("result", **report)
    if rank == 0:
        Path(args.output).write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["comm", "train"], default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--microbatch", type=int, default=3)
    parser.add_argument("--padding", choices=["max", "dynamic"], default="max")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--save-checkpoint")
    args = parser.parse_args()
    rank, world, local = initialize()
    try:
        if args.mode == "comm":
            comm_probe(rank, world, local, args.output)
        else:
            train(args, rank, world, local)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
