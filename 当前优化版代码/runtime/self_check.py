"""Exercise the production POSIX process-tree watchdog before loading any GPU model."""
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import time
import numpy as np
from budget import run_stage
from ensemble import checkpoint

assert os.name=='posix','Production watchdog requires Kaggle Linux process groups'
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    marker=root/'grandchild-survived'
    child="import time;from pathlib import Path;time.sleep(1);Path("+repr(str(marker))+").write_text('survived')"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"],start_new_session=True);print('READY',flush=True);time.sleep(5)"
    # Red: the previous process-group-only kill leaves an independently grouped
    # nested stage alive. Its short-lived fixture writes the marker and exits.
    old=subprocess.Popen([sys.executable,'-c',parent],stdout=subprocess.PIPE,text=True,start_new_session=True)
    assert old.stdout.readline().strip()=='READY'
    os.killpg(old.pid,signal.SIGKILL);old.wait(timeout=5)
    time.sleep(1.2)
    assert marker.exists(),'red fixture did not reproduce nested-session escape'
    marker.unlink()
    print('RED_CONFIRMED_NESTED_SESSION_ESCAPED_GROUP_ONLY_KILL',flush=True)
    pred=root/'submission.parquet'
    checkpoint(pred,['a','b'],np.arange(36,dtype=float).reshape(2,18))
    original=pred.read_bytes()
    result=run_stage('tree',[sys.executable,'-c',parent],.25,root)
    assert result['status']=='timeout' and result['seconds']<2
    time.sleep(1.1)
    assert not marker.exists(),'grandchild escaped timeout'
    assert pred.read_bytes()==original,'completed prediction lost'
print('POSIX_PROCESS_TREE_TIMEOUT_AND_CHECKPOINT_PASSED',flush=True)
