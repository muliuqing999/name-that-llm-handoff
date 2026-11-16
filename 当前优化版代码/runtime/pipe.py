import gc
import os
import threading
import time
from collections import deque

import numpy as np


def _flat(x):
    if isinstance(x, dict):
        parts = [x[k] for k in sorted(x)]
    else:
        parts = x if isinstance(x, (tuple, list)) else [x]
    return np.concatenate([np.ravel(np.asarray(p, dtype=np.float64)) for p in parts])


def _agree(batches, work, workers, tag, log):
    probe = [batches[len(batches) // 2 + i] for i in range(workers)]
    try:
        ref = [work(b) for b in probe]
        if any(r is None for r in ref):
            return False
        got = [None] * workers
        err = []

        def go(i):
            try:
                got[i] = work(probe[i])
            except Exception as e:
                err.append(e)

        threads = [threading.Thread(target=go, args=(i,)) for i in range(workers)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ok = not err and all(g is not None and np.abs(_flat(r) - _flat(g)).max() <= 2e-2 * (np.abs(_flat(r)).mean() + 1e-6) for r, g in zip(ref, got))
    except Exception as e:
        log(f'{tag}: concurrent check raised {e!r:.100}')
        ok = False
    log(f'{tag}: concurrent check {"passed" if ok else "failed, single worker"}')
    return ok


def pipelined(batches, work, store, fallback, workers=2, deadline=None, tag='pipe', log=print):
    batches = list(batches)
    t0 = time.time()
    workers = min(workers, int(os.environ.get('NTL_PIPE_WORKERS', workers)))
    if workers < 2 or len(batches) < 6 or not _agree(batches, work, workers, tag, log):
        workers = 1
    queue = deque(batches)
    redo = []
    lock = threading.Lock()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if deadline is not None and time.time() >= deadline:
                return
            with lock:
                if not queue:
                    return
                b = queue.popleft()
            try:
                r = work(b)
            except Exception:
                with lock:
                    redo.append(b)
                stop.set()
                return
            with lock:
                if r is None:
                    redo.append(b)
                else:
                    store(b, r)

    if workers == 1:
        while queue and not (deadline is not None and time.time() >= deadline):
            b = queue.popleft()
            r = work(b)
            if r is None:
                redo.append(b)
            else:
                store(b, r)
    else:
        threads = [threading.Thread(target=loop) for _ in range(workers)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    rest = redo + list(queue)
    if rest:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    for b in rest:
        if deadline is not None and time.time() >= deadline:
            break
        fallback(b)
    log(f'{tag}: {len(batches)} batches, {workers} worker(s), {len(redo)} redone sequentially, {time.time() - t0:.0f}s')


def spans_devices(model):
    return len({p.device for p in model.parameters()}) > 1
