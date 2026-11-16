"""Relay a fixed, versioned Python task to the configured SSH host; no remote GitHub credential."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def ssh(host, *arguments, timeout=30, stdin=None):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, *arguments],
                            input=stdin, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("SSH operation failed; inspect host connectivity and local relay log")
    return result.stdout.decode("utf-8")


def relay(request_path, output_dir):
    config = json.loads(Path(os.environ["GPU_TARGET_CONFIG"]).read_text(encoding="utf-8-sig"))
    host = config["ssh_host"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", host):
        raise ValueError("Use a configured SSH alias")
    job = "job-" + uuid.uuid4().hex
    remote = config["output_root"].rstrip("/") + "/" + job
    if not remote.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote):
        raise ValueError("Invalid remote output path")
    ssh(host, "mkdir -p " + shlex.quote(remote))
    files = [ROOT / "scripts" / name for name in ("remote_worker.py", "ppu_benchmark.py", "ppu_train.py", "performance_preflight.py", "training_data.py", "blind_inference.py", "model_features.py", "predict_offline.py", "validate_submission.py", "run_experiment.py")]
    for source in files + [Path(request_path)]:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(source), host + ":" + remote + "/"], check=True, timeout=60)
    # Only non-secret asset locations are transferred. SSH keys never leave Windows.
    config_bytes = json.dumps({k: config[k] for k in ("python", "sdk_env", "base_model", "warm_start", "train_csv", "public_manifest", "durable_root") if k in config}).encode()
    ssh(host, "/usr/local/bin/python -c " + shlex.quote("import sys,pathlib;pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())") + " " + shlex.quote(remote + "/host.json"), stdin=config_bytes)
    launcher = "import subprocess,sys; p=sys.argv[1]; f=open(p+'/supervisor.log','ab'); subprocess.Popen(['/usr/local/bin/python',p+'/remote_worker.py',p],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)"
    ssh(host, "/usr/local/bin/python -c " + shlex.quote(launcher) + " " + shlex.quote(remote))
    print("Remote job:", remote, flush=True)
    request = json.loads(Path(request_path).read_text(encoding="utf-8-sig"))
    deadline = time.monotonic() + request["max_minutes"] * 60 - 30
    status_reader = "import pathlib,sys,json; r=pathlib.Path(sys.argv[1]); p=r/'status.json'; s=json.loads(p.read_text()) if p.exists() else {}; q=r/'progress.json'; s.update(progress=json.loads(q.read_text())) if q.exists() else None; print(json.dumps(s))"
    failures = 0
    last_progress = None
    try:
        while time.monotonic() < deadline:
            time.sleep(10)
            try:
                status = json.loads(ssh(host, "/usr/local/bin/python -c " + shlex.quote(status_reader) + " " + shlex.quote(remote)))
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired):
                failures += 1
                if failures >= 6:
                    raise RuntimeError("SSH disconnected for six checks; remote watchdog still enforces its time limit")
                continue
            if status.get('progress') and status['progress'] != last_progress:
                last_progress = status['progress']
                print(json.dumps({'event':'remote_progress',**last_progress}),flush=True)
            if status.get("status") in ("completed", "failed"):
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", host + ":" + remote + "/train.log", str(Path(output_dir) / "remote-train.log")], timeout=60, check=True)
                if status["status"] == "failed":
                    raise RuntimeError("Remote task failed: " + str(status.get('error','inspect remote-train.log'))[:220])
                result = json.loads(ssh(host, "cat " + shlex.quote(remote + "/result.json")))
                result["remote_output_directory"] = remote
                (Path(output_dir) / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result), flush=True)
                return
        raise TimeoutError("Relay time budget exhausted")
    except BaseException as error:
        stopper = "import os,signal,pathlib,sys; p=pathlib.Path(sys.argv[1])/'child.pid'; os.killpg(int(p.read_text()),signal.SIGKILL) if p.exists() else None"
        try:
            ssh(host, "/usr/local/bin/python -c " + shlex.quote(stopper) + " " + shlex.quote(remote))
        except Exception:
            pass  # Independent server watchdog still bounds the process lifetime.
        (Path(output_dir)/'error.json').write_text(json.dumps({'error_type':type(error).__name__,
            'error':str(error)[:300],'remote_output_directory':remote}),encoding='utf-8')
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    relay(args.request, args.output_dir)
