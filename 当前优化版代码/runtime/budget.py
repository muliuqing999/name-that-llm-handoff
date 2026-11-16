"""Bound subprocess trees and preserve completed prediction checkpoints."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def kill_tree(proc):
    assert proc.pid > 1 and proc.pid != os.getpid()
    if proc.poll() is not None:
        return
    if os.name == 'posix':
        # A nested stage owns its own session. Follow OS parent-child links as
        # well as the outer process group so a nested watchdog cannot escape.
        def descendants(pid):
            result=[]
            try:
                tasks=list(Path(f'/proc/{pid}/task').iterdir())
            except (FileNotFoundError,PermissionError):
                return result
            children=set()
            for task in tasks:
                try:children.update(map(int,(task/'children').read_text().split()))
                except (FileNotFoundError,PermissionError):pass
            for child in children:
                result.extend(descendants(child));result.append(child)
            return result
        for pid in descendants(proc.pid):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        try:os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:pass
    else:
        # Local Windows tests have a single child. Production Kaggle is POSIX;
        # only its process-group branch is used for nested model workers.
        proc.kill()
    proc.wait(timeout=20)


def run_stage(name, cmd, seconds, output_dir, env=None):
    """A wall-clock limit covers reads, inference, and all nested subprocesses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if seconds <= 0:
        return dict(name=name, status='skipped_budget', seconds=0.0)
    flags = {'start_new_session': True} if os.name == 'posix' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    log_path = output_dir / (name + '.log')
    with log_path.open('w', encoding='utf-8') as log:
        stream=name=='pipeline'
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if stream else log,
            stderr=subprocess.STDOUT, env=env, text=True, bufsize=1, **flags)
        pump=None
        if stream:
            def forward():
                for line in proc.stdout:
                    log.write(line);log.flush()
                    print(line,end='',flush=True)
            pump=threading.Thread(target=forward,daemon=True);pump.start()
        try:
            code = proc.wait(timeout=seconds)
            status = 'complete' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            code, status = proc.returncode, 'timeout'
        if pump is not None:pump.join(timeout=5)
    result = dict(name=name, status=status, returncode=code, seconds=time.monotonic()-started, limit_seconds=seconds)
    atomic_json(output_dir / (name + '.status.json'), result)
    tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-8:]
    print(json.dumps(result), flush=True)
    print('\n'.join(tail), flush=True)
    return result
