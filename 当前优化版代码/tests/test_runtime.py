"""Regression checks for the actual timeout and prediction-preservation failure modes."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from budget import run_stage
from ensemble import LABELS, checkpoint, combine, load_member


class RuntimeRegression(unittest.TestCase):
    def test_timeout_preserves_prior_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'submission.parquet'
            z=np.arange(36,dtype=float).reshape(2,18)
            checkpoint(path,['a','b'],z)
            before=path.read_bytes()
            started=time.monotonic()
            result=run_stage('slow',[sys.executable,'-c','import time; time.sleep(4)'],.2,tmp)
            self.assertEqual(result['status'],'timeout')
            self.assertLess(time.monotonic()-started,3.)
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(pd.read_parquet(path).id.tolist(),['a','b'])

    def test_success_is_not_reported_as_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=run_stage('quick',[sys.executable,'-c','print("done")'],5,tmp)
            self.assertEqual(result['status'],'complete')

    def test_invalid_result_cannot_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'s.parquet'
            checkpoint(path,['a'],np.arange(18,dtype=float)[None])
            before=path.read_bytes()
            with self.assertRaises(AssertionError):checkpoint(path,['a'],np.full((1,18),np.nan))
            self.assertEqual(before,path.read_bytes())

    def test_member_wrong_order_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.npz'
            np.savez(path,ids=['b','a'],labels=LABELS,term_test=np.ones((2,18)))
            with self.assertRaises(AssertionError):load_member(path,['a','b'])
            self.assertIn('test',load_member(path,['b','a']))

    def test_fusion_matches_pinned_original_formula(self):
        rng=np.random.RandomState(42)
        terms={'core':rng.randn(15,18),'main':rng.randn(15,18),'zero':np.zeros((15,18))}
        prior=np.arange(1,19,dtype=float)
        z=sum(terms.values())/2
        p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        q=p.copy();target=.65*p.mean(0)+.35*prior/prior.sum();target/=target.sum()
        for _ in range(100):
            q*=15*target[None]/np.clip(q.sum(0,keepdims=True),1e-12,None)
            q/=np.clip(q.sum(1,keepdims=True),1e-12,None)
        expected=.7*np.log(p+1e-12)+.3*np.log(q+1e-12)
        np.testing.assert_allclose(combine(terms,prior),expected,atol=1e-10)


if __name__=='__main__':
    # Red: the original helper waits beyond this contract, despite a notional stage budget.
    old=(ROOT.parent/'ozertuu-reproduction-20261115/code-review/cell_27.py').read_text()
    scope=dict(subprocess=subprocess,re=re)
    exec(old.split('COMP_SEC =')[0],scope)
    t=time.monotonic();scope['run']([sys.executable,'-c','import time;time.sleep(.7)'])
    old_elapsed=time.monotonic()-t
    assert old_elapsed>.5
    print('RED_CONFIRMED: original stage exceeded .5s contract:',round(old_elapsed,3),'s',flush=True)
    unittest.main()
