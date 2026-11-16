"""Validate a bounded request, supervise training, and always write a small report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"schema_version", "enabled", "experiment_id", "task", "seed", "device",
          "epochs", "batch_size", "learning_rate", "max_minutes"}
TWO_STAGE_FIELDS = {"data_profile", "pretrain_epochs", "pretrain_learning_rate", "pretrain_max_rows", "refit_all"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    two_stage = request.get("task") == "qwen3_two_stage"
    if set(request) != (FIELDS | TWO_STAGE_FIELDS if two_stage else FIELDS):
        raise ValueError("Request must contain exactly the documented configuration fields")
    if type(request["schema_version"]) is not int or request["schema_version"] != (2 if two_stage else 1):
        raise ValueError("schema_version must match the task (two-stage=2; legacy=1)")
    if type(request["enabled"]) is not bool:
        raise ValueError("enabled must be a JSON boolean")
    if not isinstance(request["experiment_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request["experiment_id"]):
        raise ValueError("experiment_id must be 1-64 safe letters, digits, underscores or hyphens")
    if request["task"] not in {"smoke_mlp", "ppu_benchmark", "qwen3_lora", "qwen3_two_stage"} or request["device"] not in ("cuda", "cpu"):
        raise ValueError("Unknown task or device")
    if request["task"] != "smoke_mlp" and (request["device"] != "cuda" or request["batch_size"] != 18):
        raise ValueError("PPU tasks require device=cuda (vendor compatibility layer) and global batch_size=18")
    if request["task"] != "smoke_mlp" and request["epochs"] > 5:
        raise ValueError("PPU tasks allow at most five epochs; benchmark uses a fixed seven steps")
    bounds = {"seed": (0, 2**31 - 1), "epochs": (1, 100), "batch_size": (16, 4096)}
    for key, (lower, upper) in bounds.items():
        if type(request[key]) is not int or not lower <= request[key] <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    for key, lower, upper in (("learning_rate", 1e-6, 1e-3 if two_stage else 1.0), ("max_minutes", 1, 720 if two_stage else 240)):
        value = request[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be a finite number in [{lower}, {upper}]")
    if two_stage:
        if type(request['refit_all']) is not bool:
            raise ValueError('refit_all must be boolean')
        if request['data_profile'] != 'curated_v1':
            raise ValueError('Only the installed curated_v1 data profile is allowed')
        for key, lower, upper in [('pretrain_epochs',1,2),('pretrain_max_rows',1800,270000)]:
            if type(request[key]) is not int or not lower <= request[key] <= upper:
                raise ValueError(f'{key} is out of bounds')
        rate=request['pretrain_learning_rate']
        if type(rate) not in (float,int) or not math.isfinite(rate) or not 1e-6 <= rate <= 2e-4:
            raise ValueError('Invalid pretrain_learning_rate')
    return request


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Request contains a duplicate field")
        result[key] = value
    return result


def terminate_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        killed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                capture_output=True, timeout=20)
        if killed.returncode != 0 and process.poll() is None:
            process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=20)


def run_child(command, log_path, timeout):
    options = {"start_new_session": True} if os.name != "nt" else {}
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy(), **options)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            terminate_tree(process)
            raise


def execute(request_path, report_dir):
    started = time.monotonic()
    report_dir = Path(report_dir)
    run = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    run_id = f"{run}-{attempt}" if run.isdigit() and attempt.isdigit() else "local-" + uuid.uuid4().hex
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    report = {"schema_version": 1, "run_id": run_id, "status": "started",
              "started_at": utc_now(), "commit_sha": os.environ.get("GITHUB_SHA"),
              "run_url": f"{server}/{repository}/actions/runs/{run}" if run.isdigit() and repository else None,
              "experiment_id": None, "task": None, "device": None, "request_sha256": None,
              "metrics": {}, "checkpoint": None,
              "note": "Synthetic smoke test only; not an LLM or Kaggle quality benchmark."}

    def save():
        atomic_json(report_dir / f"{run_id}.json", report)
        atomic_json(report_dir / "latest.json", report)

    save()  # Replace stale prior-run status before any validation or import can fail.
    exit_code = 0
    try:
        raw = Path(request_path).read_bytes()
        report["request_sha256"] = hashlib.sha256(raw).hexdigest()
        request = validate_request(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=reject_duplicate_keys))
        report.update({key: request[key] for key in ("experiment_id", "task", "device")})
        report['request'] = request
        if not request["enabled"]:
            report["status"] = "skipped"
            report["reason"] = "Request is disabled"
        else:
            output_root = Path(os.environ.get("GPU_RUNNER_OUTPUT_ROOT", str(Path.home() / "gpu-training-runs"))).expanduser().resolve()
            if output_root == ROOT or ROOT in output_root.parents:
                raise ValueError("GPU_RUNNER_OUTPUT_ROOT must be outside the repository checkout")
            claimed = output_root / '.claimed-experiments' / (request['experiment_id'] + '.json')
            claimed.parent.mkdir(parents=True, exist_ok=True)
            try:
                with claimed.open('x',encoding='utf-8') as handle:
                    json.dump({'run_id':run_id,'request_sha256':report['request_sha256']},handle)
            except FileExistsError:
                previous=json.loads(claimed.read_text(encoding='utf-8'))
                report.update(status='skipped',reason='experiment_id_already_claimed',previous_run_id=previous['run_id'])
                save()
                return 0
            run_dir = output_root / request["experiment_id"] / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(run_dir / "request.json", request)
            report["output_directory"] = str(run_dir)
            save()
            entry = "train.py" if request["task"] == "smoke_mlp" else "scripts/remote_bridge.py"
            report["note"] = ("Synthetic smoke test only; not an LLM or Kaggle quality benchmark." if request["task"] == "smoke_mlp" else
                              "PPU benchmark uses 18 repeated passages; no quality claim." if request["task"] == "ppu_benchmark" else
                              "Fresh Qwen3 LoRA candidate with text-only blind validation; not the complete 0.622 ensemble.")
            code = run_child([sys.executable, str(ROOT / entry), "--request", str(run_dir / "request.json"),
                              "--output-dir", str(run_dir)], run_dir / "train.log", request["max_minutes"] * 60)
            if code:
                error_file = run_dir / "error.json"
                if error_file.is_file():
                    child_error = json.loads(error_file.read_text(encoding="utf-8"))
                    report["error_type"] = str(child_error["error_type"])[:80]
                    if 'remote_output_directory' in child_error:
                        report['remote_output_directory']=child_error['remote_output_directory']
                    raise RuntimeError(str(child_error["error"])[:300])
                raise RuntimeError(f"Training exited with code {code}; inspect the local train.log")
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            for key in ("metrics", "checkpoint", "versions", "hardware", "epochs_completed"):
                report[key] = result[key]
            for key in ("remote_output_directory", "data", "scope", "checkpoint_sha256", "performance", "validation", "durable_checkpoint", "selection_checkpoint", "acceptance_only"):
                if key in result:
                    report[key] = result[key]
            report["status"] = "completed"
    except subprocess.TimeoutExpired:
        report.update(status="failed", error="Training exceeded max_minutes; process tree terminated")
        exit_code = 1
    except KeyboardInterrupt:
        report.update(status="failed", error="Training interrupted; process tree terminated")
        exit_code = 130
    except (ValueError, RuntimeError) as error:
        report.update(status="failed", error=str(error)[:500])
        exit_code = 1
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}; inspect local configuration and logs")
        exit_code = 1
    finally:
        report.update(finished_at=utc_now(), duration_seconds=round(time.monotonic() - started, 3))
        save()
    print(f"Experiment status: {report['status']}; report: {report_dir / 'latest.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="experiments/request.json")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    sys.exit(execute(args.request, args.report_dir))
