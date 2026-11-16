"""Server-side supervisor: one GPU job at a time, independent wall-clock watchdog."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from run_experiment import validate_request


def atomic(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def work(root):
    request = json.loads((root / "request.json").read_text(encoding="utf-8-sig"))
    config = json.loads((root / "host.json").read_text())
    validate_request(request)
    if request["task"] not in {"ppu_benchmark", "qwen3_lora", "qwen3_two_stage"}:
        raise ValueError("Invalid remote request")
    lock = (root.parent / "gpu.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               NTL_BASE_MODEL=config["base_model"], NTL_WARM_START=config["warm_start"], NTL_TRAIN_CSV=config["train_csv"])
    if request['task']=='qwen3_two_stage':
        env['NTL_PUBLIC_MANIFEST']=config['public_manifest']
        env['NTL_DURABLE_ROOT']=config['durable_root']
    script = "ppu_benchmark.py" if request["task"] == "ppu_benchmark" else "ppu_train.py"
    command = [config["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2", str(root / script)]
    if script == "ppu_benchmark.py":
        command += ["--microbatch", "9", "--padding", "dynamic", "--output", str(root / "benchmark.json"), "--save-checkpoint", str(root / "benchmark-trainable.pt")]
    else:
        command += ["--request", str(root / "request.json"), "--output-dir", str(root)]
    shell = "source " + shlex.quote(config["sdk_env"]) + " >/dev/null && exec " + shlex.join(command)
    started = time.monotonic()
    with (root / "train.log").open("wb") as log:
        child = subprocess.Popen(["bash", "-lc", shell], env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / "child.pid").write_text(str(child.pid))
        atomic(root / "status.json", {"status": "running", "pid": child.pid})
        try:
            code = child.wait(timeout=max(30, request["max_minutes"] * 60 - 60))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise RuntimeError("Remote time limit exceeded")
    if code:
        raise RuntimeError("Training process failed with exit code " + str(code))
    if script == "ppu_benchmark.py":
        benchmark = json.loads((root / "benchmark.json").read_text())
        checkpoint = root / "benchmark-trainable.pt"
        atomic(root / "result.json", {"metrics": benchmark, "checkpoint": str(checkpoint),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "epochs_completed": 0, "versions": benchmark["versions"], "hardware": {"world_size": 2, "name": "PPU-ZW810E"},
               "scope": "Seven updates on 18 repeated examples. Trainable weights saved to verify persistence, not a competition model."})
    atomic(root / "status.json", {"status": "completed", "duration_seconds": time.monotonic() - started})


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    try:
        work(directory)
    except BaseException as error:
        atomic(directory / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)[:250]})
        raise
