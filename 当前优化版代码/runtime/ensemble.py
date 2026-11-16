"""Label-safe fusion and atomic, fully validated submission checkpoints."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z-z.max(1, keepdims=True)
    p = np.exp(z)
    return p/p.sum(1, keepdims=True)


def decide_scores(p, prior, strength=.30, target_mix=.35):
    q = p.copy()
    marg = np.clip(p.mean(0), 1e-6, None)
    marg /= marg.sum()
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior/prior.sum()
    target = np.clip((1-target_mix)*marg+target_mix*prior, 1e-6, None)
    target /= target.sum()
    for _ in range(100):
        q *= len(q)*target[None, :]/np.clip(q.sum(0, keepdims=True), 1e-12, None)
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    return (1-strength)*np.log(p+1e-12)+strength*np.log(q+1e-12)


def combine(terms, prior):
    assert terms and all(v.ndim == 2 and v.shape[1] == 18 and np.isfinite(v).all() for v in terms.values())
    active = sum(float(abs(v).max()) > 0 for v in terms.values())
    return decide_scores(softmax(sum(terms.values())/max(1, active)), prior)


def load_member(path, ids):
    with np.load(path, allow_pickle=False) as z:
        assert z['ids'].astype(str).tolist() == list(map(str, ids)), 'member row order mismatch'
        assert z['labels'].astype(str).tolist() == LABELS, 'member label order mismatch'
        values = {k[5:]:z[k].copy() for k in z.files if k.startswith('term_')}
    assert values and all(v.shape == (len(ids),18) and np.isfinite(v).all() for v in values.values())
    return values


def checkpoint(path, ids, scores):
    ids = list(map(str, ids))
    assert len(ids) and len(set(ids)) == len(ids)
    scores = np.asarray(scores)
    assert scores.shape == (len(ids),18) and np.isfinite(scores).all()
    frame = pd.DataFrame({'id':ids, 'model':np.asarray(LABELS)[scores.argmax(1)]})
    path = Path(path)
    temp = path.with_name(path.name+'.tmp.parquet')
    frame.to_parquet(temp, index=False)
    assert pd.read_parquet(temp).equals(frame)
    os.replace(temp,path)
    return frame
