import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
FUNCTION_WORDS = frozenset('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
VIEWS = [('char', 3, 5, 250000, 1.), ('word', 1, 3, 180000, .7),
         ('shape', 2, 6, 60000, .5), ('function', 1, 3, 50000, .5)]


def shape_view(text):
    return re.sub(r'[^\W\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)


def function_view(text):
    return ' '.join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else 'CONTENT'
                    for t in re.findall(r'\w+|[^\w\s]', text))


def prepare(texts, view, keep=()):
    if view == 'shape':
        return [shape_view(t) for t in texts]
    if view == 'function':
        return [function_view(t) for t in texts]
    if view == 'distort':
        keep = set(keep)
        return [re.sub(r"[A-Za-z’']+", lambda m: m[0] if m[0].lower() in keep else '*', t) for t in texts]
    return texts


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(1, keepdims=True)


def sparse_group_centre(X, groups, lo=2, hi=18):
    X = csr_matrix(X)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()


def style_groups(texts):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    try:
        X = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000,
                            dtype=np.float32).fit_transform(texts)
        D = np.clip(1.0 - (X @ X.T).toarray(), 0., 1.)
    except ValueError as exc:
        if 'empty vocabulary' not in str(exc):
            raise
        D = np.ones((len(texts), len(texts)), dtype=np.float32) - np.eye(len(texts), dtype=np.float32)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)
    D = np.asarray(D / med, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.)
    return fcluster(linkage(squareform(D, checks=False), method='average'), .90, criterion='distance')


class Readout:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'metadata.json').read_text())
        if self.meta['labels'] != LABELS:
            raise ValueError('Unexpected canonical class order')
        self.vectorizers = [joblib.load(self.folder / e['file']) for e in self.meta['views']]
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert all(type(v) is TfidfVectorizer and not callable(v.analyzer) and
                   v.preprocessor is None and v.tokenizer is None for v in self.vectorizers)

    def matrix(self, texts):
        return hstack([v.transform(prepare(texts, e['view'], self.meta.get('keep', []))) * e['weight']
                       for v, e in zip(self.vectorizers, self.meta['views'])], format='csr')

    def raw(self, X, state='model'):
        with np.load(self.folder / (state + '.npz'), allow_pickle=False) as z:
            scores = np.asarray(X @ z['coef'].T) + z['intercept']
            classes = z['classes'].copy()
        if len(classes) == 2 and scores.shape[1] == 1:
            scores = np.column_stack([-scores[:, 0], scores[:, 0]])
        return scores, classes

    def scores(self, texts, state='model'):
        return self.raw(self.matrix(texts), state)


def core_votes(root, texts):
    votes = {}
    for key, weight in [('svc4', 12.0), ('tfidf', .25), ('distort', .10)]:
        r = Readout(Path(root) / 'core' / key)
        scores, classes = r.scores(texts)
        assert classes.tolist() == list(range(len(LABELS)))
        votes[key] = log_softmax(scores) * weight
    return votes


def complementary_style(root, texts):
    root = Path(root)
    r = Readout(root / 'comp' / 'stylegram')
    X = r.matrix(texts)
    d, classes = r.raw(X, 'plain')
    assert classes.tolist() == list(range(len(LABELS)))
    P = softmax(d)
    groups = style_groups(texts)
    sizes = np.bincount(groups)[groups]
    rows = (sizes >= 2) & (sizes <= 18)
    if rows.any():
        d, _ = r.raw(sparse_group_centre(X[rows], groups[rows]), 'twin')
        P[rows] = softmax(d)
    gates = {}
    for family in ['claude', 'gpt', 'gemini', 'glm']:
        g = Readout(root / 'comp' / 'gates' / family)
        d, classes = g.scores(texts)
        expected = [i for i, label in enumerate(LABELS) if label.startswith(family)]
        assert classes.tolist() == expected
        gates[family] = d
    return P, gates, groups
