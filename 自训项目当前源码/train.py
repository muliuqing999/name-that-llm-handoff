"""Tiny synthetic PyTorch training task used to check the remote execution loop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from scripts.run_experiment import atomic_json, validate_request


class TrainingSetupError(RuntimeError):
    def __init__(self, error_type, message):
        super().__init__(message)
        self.error_type = error_type


def train(request, output_dir):
    # Deliberately lazy: request validation and disabled requests need no PyTorch.
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise TrainingSetupError("pytorch_unavailable", "PyTorch import failed; check the selected interpreter and wheel installation.") from error

    if request["device"] == "cuda" and not torch.cuda.is_available():
        raise TrainingSetupError("cuda_unavailable", "CUDA was requested but is unavailable. Install compatible NVIDIA drivers and CUDA-enabled PyTorch.")
    torch.manual_seed(request["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(request["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(request["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(request["seed"])
    inputs = torch.randn(5120, 16, generator=generator)
    teacher = torch.randn(16, 4, generator=generator)
    targets = (inputs @ teacher).argmax(dim=1)
    train_x, val_x = inputs[:4096].to(device), inputs[4096:].to(device)
    train_y, val_y = targets[:4096].to(device), targets[4096:].to(device)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=request["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    output_dir = Path(output_dir)
    best_loss, best_accuracy, best_epoch = float("inf"), 0.0, 0
    checkpoint = output_dir / "best.pt"
    started = time.monotonic()
    for epoch in range(1, request["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        for offset in range(0, len(order), request["batch_size"]):
            index = order[offset:offset + request["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[index]), train_y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(val_x)
            val_loss = loss_fn(predictions, val_y).item()
            val_accuracy = (predictions.argmax(dim=1) == val_y).float().mean().item()
        if val_loss < best_loss:
            best_loss, best_accuracy, best_epoch = val_loss, val_accuracy, epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy,
                        "request": request, "architecture": "16-64-ReLU-4"}, temporary)
            os.replace(temporary, checkpoint)
        print(json.dumps({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"metrics": {"best_val_loss": best_loss, "best_val_accuracy": best_accuracy,
                          "best_epoch": best_epoch, "training_seconds": round(time.monotonic() - started, 3)},
              "checkpoint": str(checkpoint.resolve()), "epochs_completed": request["epochs"],
              "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
              "hardware": {"device": str(device), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}}
    atomic_json(output_dir / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    request = validate_request(json.loads(Path(args.request).read_text(encoding="utf-8-sig")))
    if not request["enabled"]:
        raise SystemExit("Request is disabled; use scripts/run_experiment.py to report a skip")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        train(request, args.output_dir)
    except Exception as error:
        if isinstance(error, TrainingSetupError):
            summary = {"error_type": error.error_type, "error": str(error)}
        elif type(error).__name__ in ("OutOfMemoryError", "MemoryError"):
            summary = {"error_type": "out_of_memory", "error": "Training exhausted available memory. Reduce batch_size or free device memory."}
        else:
            summary = {"error_type": "training_runtime_error", "error": "Training failed inside PyTorch; inspect the local train.log for details."}
        # The full traceback remains on the GPU host; only these fixed summaries are published.
        atomic_json(Path(args.output_dir) / "error.json", summary)
        traceback.print_exc()
        sys.exit(1)
