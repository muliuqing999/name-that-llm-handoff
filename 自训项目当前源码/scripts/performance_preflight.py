"""Bounded checkpointing calibration. Probes never update model parameters.

Timing covers forward/backward only, before DDP and optimizer construction. It is
not an end-to-end training speedup claim. Batch size, tokens and loss stay fixed.
"""
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_cached(path, key, now=None):
    try:
        record = json.loads(Path(path).read_text())
        age = (time.time() if now is None else now) - record["created_at"]
        if (record["key"] == key and type(record["checkpointing"]) is bool
                and 0 <= age < 7 * 86400):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def choose_checkpointing(baseline, candidate):
    """Require correctness, memory headroom and at least 5% less probe time."""
    if not candidate or not candidate.get("safe") or not candidate.get("gradients_match"):
        return True
    old, new = baseline.get("mean_forward_backward_seconds"), candidate.get("mean_forward_backward_seconds")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (old, new)):
        return True
    return not (new <= old * 0.95)


def compare_gradients(actual, reference, loss, reference_loss):
    """BF16-aware checks on raw gradients AND the gradients the optimizer sees.

    Relative elementwise errors are unstable near zero. Keep that diagnostic,
    but require tight whole-vector error, bounded per-tensor error, unchanged
    loss and allclose after the training loop's unit-norm clipping instead.
    """
    import torch
    if (len(actual) != len(reference) or not any(a is not None for a in actual)
            or any((a is None) != (b is None) for a, b in zip(actual, reference))):
        # Keep collective keys identical even if only one rank has a problem.
        return {"gradients_match": False, "gradient_relative_l2": 1e30,
                "gradient_max_tensor_relative_l2": 1e30, "gradient_max_absolute_difference": 1e30,
                "raw_elementwise_match": False, "post_clip_gradients_match": False,
                "loss_absolute_difference": abs(loss-reference_loss),
                "gradient_relative_l2_limit": 1e-4, "gradient_tensor_relative_l2_limit": 0.02}
    pairs = [(a, b) for a, b in zip(actual, reference) if a is not None]
    checks = [(float((a-b).norm()), float(b.norm()), float((a-b).abs().max())) for a, b in pairs]
    norm_a = math.sqrt(sum(float(a.norm()) ** 2 for a, b in pairs))
    norm_b = math.sqrt(sum(b*b for a,b,c in checks))
    relative = math.sqrt(sum(a*a for a,b,c in checks)) / max(1e-12, norm_b)
    per_tensor = max(a / max(b, 1e-12) for a,b,c in checks)
    clip_a, clip_b = min(1.0, 1.0 / (norm_a + 1e-6)), min(1.0, 1.0 / (norm_b + 1e-6))
    post_clip = all(torch.allclose(a * clip_a, b * clip_b, rtol=0.005, atol=1e-5) for a,b in pairs)
    loss_delta = abs(loss - reference_loss)
    return {"gradients_match": bool(relative <= 1e-4 and per_tensor <= 0.02 and post_clip
                                    and loss_delta <= 1e-4 * max(1.0, abs(reference_loss))),
            "gradient_relative_l2": relative, "gradient_max_tensor_relative_l2": per_tensor,
            "gradient_max_absolute_difference": max(c for a,b,c in checks),
            "raw_elementwise_match": all(torch.allclose(a,b,rtol=0.005,atol=1e-5) for a,b in pairs),
            "post_clip_gradients_match": post_clip, "loss_absolute_difference": loss_delta,
            "gradient_relative_l2_limit": 1e-4, "gradient_tensor_relative_l2_limit": 0.02}


@contextlib.contextmanager
def preserve_probe_state(module, local=None):
    import torch
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError("Performance probes must precede optimizer/training updates")
    cpu_rng, python_rng = torch.get_rng_state(), random.getstate()
    device_rng = torch.cuda.get_rng_state(local) if local is not None else None
    modes = [(m, m.training) for m in module.modules()]
    buffers = [(b, b.detach().clone()) for b in module.buffers()]
    try:
        yield
    finally:
        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for child, training in modes:
            child.training = training
        torch.set_rng_state(cpu_rng)
        random.setstate(python_rng)
        if local is not None:
            torch.cuda.set_rng_state(device_rng, local)


def calibrate(module, cfg, training, make_batch, base, data_sha, output, rank, world, local, max_minutes):
    import torch
    import torch.distributed as dist

    started = time.monotonic()
    device = torch.device("cuda", local)
    params = [p for p in module.parameters() if p.requires_grad]

    def maximum(value):
        value = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()

    def all_ok(value):
        return maximum(0 if value else 1) == 0

    def set_mode(enabled):
        if enabled:
            module.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            module.backbone.gradient_checkpointing_disable()

    def publish(report):
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        report["timing_scope"] = "forward/backward only, without DDP synchronization or optimizer updates"
        if rank == 0:
            (output / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"event": "performance_preflight", **report}), flush=True)
        return report

    if max_minutes < 5:
        set_mode(True)
        return publish({"status": "skipped_small_time_budget", "checkpointing": True, "cache_hit": False})

    # Metadata fingerprints avoid reading tens of GB of immutable base weights.
    # Code, tokenizer/model asset metadata, dataset, vendor packages and device
    # capacity changes all invalidate the persistent performance choice.
    assets = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
              for p in sorted(Path(base).iterdir()) if p.is_file()]
    packages = {}
    for name in ("torch", "transformers", "peft", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in ("performance_preflight.py", "ppu_train.py", "ppu_benchmark.py", "training_data.py", "model_features.py")}
    props = torch.cuda.get_device_properties(local)
    driver = "unknown"
    try:
        info = subprocess.run(["/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"], capture_output=True, text=True, timeout=5)
        driver = next((line.strip() for line in info.stdout.splitlines() if "Driver Version" in line), "unknown")
    except (OSError, subprocess.TimeoutExpired):
        pass
    identity = {"version": 1, "cfg": cfg, "dataset": data_sha, "assets": assets, "packages": packages,
                "code": sources, "device": props.name, "total_memory": props.total_memory,
                "world": world, "local_batch": 9, "dtype": "bfloat16", "driver": driver,
                "runtime": {name: os.environ.get(name) for name in ("PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")}}
    key = fingerprint(identity)
    cache_file = output.parent / "performance-cache" / (key + ".json")
    shared = [load_cached(cache_file, key) if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0, device=device)
    cached = shared[0]

    # Keep same-prompt triplets intact. Use short/median/long training examples,
    # never validation data or validation labels, to select an execution mode.
    triplets = [group[i:i+3] for group in training for i in range(0, len(group) - 2, 3)]
    triplets.sort(key=lambda group: max(len(r["tokens"]) for r in group))
    probes = []
    for quantile in (0.1, 0.5, 0.9):
        start = int((len(triplets) - 6) * quantile) + rank * 3
        probes.append(make_batch([r for group in triplets[start:start+3] for r in group]))
    # Explicit worst-shape stress: even if this seed's batches happen to be
    # shorter, future shuffled batches may contain nine maximum-length rows.
    length = math.ceil((cfg["max_len"] + 1) / 8) * 8
    source, _, labels = probes[1]
    stress_ids = source.repeat(1, math.ceil(length / source.shape[1]))[:, :length].contiguous()
    stress = (stress_ids, torch.ones_like(stress_ids), labels)
    free, total = torch.cuda.mem_get_info(local)
    # Leave 15% of device memory for optimizer/DDP state, kernels and variation.
    available = min(total * 0.85, free + torch.cuda.memory_reserved(local) - total * 0.15)

    def backward(batch, clip=True):
        module.zero_grad(set_to_none=True)
        loss = module(*batch)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite preflight loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0 if clip else float("inf"), foreach=False)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Non-finite preflight gradient")
        return float(loss.detach())

    def stress_check(enabled):
        set_mode(enabled)
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(local)
        ok, reason = True, None
        try:
            backward(stress)
            torch.cuda.synchronize(local)
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
            ok, reason = False, type(error).__name__
        peak = torch.cuda.max_memory_reserved(local)
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        safe = all_ok(ok and peak <= available)
        return {"safe": safe, "peak_reserved_GiB": maximum(peak) / 1024**3,
                "memory_headroom_fraction": 0.15, "reason": reason if reason else (None if safe else "memory_budget_or_peer_failure")}

    with preserve_probe_state(module, local):
        module.train()
        if cached is not None:
            check = stress_check(cached["checkpointing"])
            if check["safe"]:
                result = {"status": "cached_and_stress_checked", "key": key, "cache_hit": True,
                          "checkpointing": cached["checkpointing"], "stress": check}
                return publish(result)
            # Retest the conservative mode if free memory changed since caching.
            if cached["checkpointing"]:
                raise RuntimeError("Cached conservative execution failed the memory/correctness preflight")

        reports, reference = [], None
        for enabled in (True, False):
            # Bounded tuning: never start the second candidate after 90 seconds.
            # The existing independent remote watchdog bounds the entire job.
            if not enabled and maximum(time.monotonic() - started) > 90:
                break
            check = stress_check(enabled)
            record = {"checkpointing": enabled, **check}
            if not check["safe"]:
                reports.append(record)
                if enabled:
                    raise RuntimeError("Conservative execution failed the memory/correctness preflight")
                continue

            # Check recomputation against ordinary autograd using identical data
            # and dropout disabled, while backbone.training remains True.
            dropouts = [m for m in module.modules() if isinstance(m, torch.nn.Dropout)]
            for m in dropouts:
                m.training = False
            loss = backward(probes[1], clip=False)
            current = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
            if reference is None:
                reference = (loss, current)
                matches = True
            else:
                comparison = compare_gradients(current, reference[1], loss, reference[0])
                matches = comparison["gradients_match"]
                for name, value in comparison.items():
                    if type(value) is bool:
                        record[name] = all_ok(value)
                    elif type(value) in (float, int):
                        record[name] = maximum(value)
                    else:
                        record[name] = value
            record["gradients_match"] = all_ok(matches)
            del current
            for m in dropouts:
                m.training = True
            times = []
            for repeat in range(3):  # One warm-up per shape, then two measurements.
                for batch in probes:
                    torch.cuda.synchronize(local)
                    tick = time.perf_counter()
                    backward(batch)
                    torch.cuda.synchronize(local)
                    elapsed = maximum(time.perf_counter() - tick)
                    if repeat:
                        times.append(elapsed)
            record["mean_forward_backward_seconds"] = statistics.mean(times)
            record["timed_batches"] = len(times)
            reports.append(record)
        selected = choose_checkpointing(reports[0], reports[1] if len(reports) > 1 else None)
        set_mode(selected)
        result = {"status": "calibrated", "key": key, "cache_hit": False,
                  "checkpointing": selected, "candidates": reports,
                  "selection_minimum_time_reduction": 0.05}
        if rank == 0 and len(reports) == 2:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                temp = cache_file.with_suffix(".tmp")
                temp.write_text(json.dumps({"key": key, "created_at": time.time(), "checkpointing": selected,
                                           "candidates": reports}, indent=2))
                temp.replace(cache_file)
            except OSError as error:
                result["cache_write_error"] = type(error).__name__
    return publish(result)
