import os, sys, time, gc, argparse, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from pipe import pipelined, spans_devices

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
L2I = {l: i for i, l in enumerate(LABELS)}

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--models', required=True)
p.add_argument('--out', default='submission.parquet')
p.add_argument('--save_final_logits', default='')
p.add_argument('--comp_path', default='')
p.add_argument('--comp_w', type=float, default=0.0)
p.add_argument('--tfidf_w', type=float, default=1.0)
p.add_argument('--distort_w', type=float, default=0.0)
p.add_argument('--adapt_thr', type=float, default=0.15)
p.add_argument('--adapt_w', type=float, default=0.0)
p.add_argument('--extra', default='')
p.add_argument('--extra_w', type=float, default=0.5)
p.add_argument('--svc_w', type=float, default=0.0)
p.add_argument('--pool', default='')
p.add_argument('--sk_div', type=int, default=1)
p.add_argument('--sinkhorn', type=float, default=0.0)
p.add_argument('--sinkhorn_target_mix', type=float, default=0.35)
p.add_argument('--atlas_dir', default='')
p.add_argument('--atlas_base', default='')
p.add_argument('--atlas_mode', default='off', choices=['off', 'head', 'lite', 'mix'])
p.add_argument('--atlas_member_w', type=float, default=0.0)
p.add_argument('--atlas_k', type=int, default=6)
p.add_argument('--atlas_w', type=float, default=0.0)
p.add_argument('--atlas_margin', type=float, default=0.0)
p.add_argument('--atlas_bs', type=int, default=4)
p.add_argument('--atlas_chunk', type=int, default=3)
p.add_argument('--atlas_workers', type=int, default=2)
p.add_argument('--atlas_deadline', type=float, default=0.0)
p.add_argument('--save_lp', default='')
p.add_argument('--native_dir', default='')
p.add_argument('--native_base', default='')
p.add_argument('--native_share', type=float, default=0.0)
p.add_argument('--native_cent', type=int, default=1)
p.add_argument('--native_deadline', type=float, default=0.0)
p.add_argument('--optional_deadline', type=float, default=0.0)
p.add_argument('--native_work', default='/kaggle/temp/native')
p.add_argument('--native_min_agree', type=float, default=0.35)
p.add_argument('--prefit_core', default='')
p.add_argument('--gsq_dir', default='')
p.add_argument('--gsq_share', type=float, default=0.0)
p.add_argument('--gsq_deadline', type=float, default=0.0)
p.add_argument('--gsq_timeout', type=float, default=0.0)
p.add_argument('--gsq_work', default='/kaggle/temp/gsq')
p.add_argument('--gsq_min_agree', type=float, default=0.30)
p.add_argument('--group_embed_base', default='')
p.add_argument('--group_embed_assets', default='')
p.add_argument('--group_work', default='/kaggle/temp/qgroups')
p.add_argument('--group_sim', type=float, default=0.70)
a = p.parse_args()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
def log(*x): print(f'[{time.time()-T0:7.0f}s]', *x, flush=True)

train = pd.read_csv(os.path.join(a.data, 'train.csv'))
test = pd.read_csv(os.path.join(a.data, 'test.csv'))
train['text'] = train.text.fillna(''); test['text'] = test.text.fillna('')
y = train.model.map(L2I).values; Gtr = train.prompt_id.values
log('train', len(train), 'test', len(test))

def capped_agglo(S, cap=18, thr=0.02):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab

from sklearn.feature_extraction.text import TfidfVectorizer
group_output = os.environ.get('NTL_GROUP_OUTPUT', '')
if group_output and os.path.isfile(group_output):
    Gte = np.load(group_output, allow_pickle=False)
elif os.environ.get('NTL_GROUP_EMBED'):
    E_q = np.load(os.environ['NTL_GROUP_EMBED'], allow_pickle=False).astype(np.float32)
    assert E_q.shape[0] == len(test) and np.isfinite(E_q).all()
    Gte = capped_agglo(E_q @ E_q.T, 18, .70)
else:
    try:
        v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=.05)
        Xc = v.fit_transform(test.text.values)
        Gte = capped_agglo((Xc @ Xc.T).toarray(), 18, .15)
    except ValueError:
        Gte = np.arange(len(test))
assert Gte.shape == (len(test),)
if group_output:
    np.save(group_output, Gte, allow_pickle=False)
log('shared groups', len(np.unique(Gte)))
LP = {}
EXT_T, EXT_Y = np.array([], dtype=object), np.array([], dtype=int)
EXTRA_DF = pd.DataFrame(columns=['text', 'model'])
def _norm_text_key(value):
    import unicodedata
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value or ''))).strip()
if a.extra:
    _E = pd.read_parquet(a.extra) if a.extra.endswith('.parquet') else pd.read_csv(a.extra)
    if not {'text', 'model'}.issubset(_E.columns):
        raise ValueError(f'External file must contain text/model columns: {a.extra}')
    _E = _E[_E.model.isin(LABELS)].copy()
    _E['text'] = _E.text.fillna('').astype(str)
    _official_keys = {_norm_text_key(t) for t in train.text.values}
    _seen = set(_official_keys)
    _keep = []
    for _t in _E.text.values:
        _k = _norm_text_key(_t)
        _ok = bool(_k) and _k not in _seen
        _keep.append(_ok)
        if _ok: _seen.add(_k)
    EXTRA_DF = _E.loc[_keep, ['text', 'model']].reset_index(drop=True)
    EXT_T = EXTRA_DF.text.astype(str).values
    EXT_Y = EXTRA_DF.model.map(L2I).values.astype(int)
    log('matched-clean extra rows after exact de-dup', len(EXT_T))
FIT_T = np.concatenate([train.text.values, EXT_T])
FIT_Y = np.concatenate([y, EXT_Y])
FIT_W = np.concatenate([np.ones(len(train), dtype=np.float32), np.full(len(EXT_T), a.extra_w, dtype=np.float32)])
if a.tfidf_w > 0:
    from sklearn.linear_model import LogisticRegression
    mats_tr, mats_te = [], []
    for kw in [dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000),
               dict(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000)]:
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, **kw).fit(FIT_T)
        mats_tr.append(vec.transform(FIT_T)); mats_te.append(vec.transform(test.text.values))
    LP['tfidf'] = LogisticRegression(C=20, max_iter=3000).fit(sp.hstack(mats_tr).tocsr(), FIT_Y, sample_weight=FIT_W).predict_log_proba(sp.hstack(mats_te).tocsr()) * a.tfidf_w
    gc.collect(); log('tfidf done')
else:
    LP['tfidf'] = np.zeros((len(test), 18), dtype=np.float32)
    log('tfidf skipped (zero ensemble weight)')
if a.svc_w > 0:
    import re as _re
    from scipy.sparse import hstack as _hs
    from sklearn.svm import LinearSVC
    _FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    _shape = lambda t: _re.sub(r"[^\W\d_]+", lambda m_: "X" if m_[0].isupper() else "x", t)
    _func = lambda t: " ".join(w.lower() if w.lower() in _FW or not w.isalnum() else "CONTENT" for w in _re.findall(r"\w+|[^\w\s]", t))
    _parts = []
    _official = train[['text', 'model']].copy(); _official['text'] = _official.text.fillna('').astype(str)
    _official['_srcw'] = 1.0
    _parts.append(_official)
    _seen = {_norm_text_key(t) for t in _official.text.values}
    if len(EXTRA_DF):
        _e = EXTRA_DF[['text', 'model']].copy(); _e['_srcw'] = float(a.extra_w)
        _keep = []
        for _t in _e.text.values:
            _k = _norm_text_key(_t); _ok = bool(_k) and _k not in _seen
            _keep.append(_ok)
            if _ok: _seen.add(_k)
        _e = _e.loc[_keep].copy(); _parts.append(_e)
    if a.pool and os.path.exists(a.pool):
        _P = pd.read_parquet(a.pool)
        if not {'text', 'model'}.issubset(_P.columns):
            raise ValueError(f'Pool must contain text/model columns: {a.pool}')
        _P = _P[_P.model.isin(LABELS)][['text', 'model']].copy()
        _P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35
        _keep = []
        for _t in _P.text.values:
            _k = _norm_text_key(_t); _ok = bool(_k) and _k not in _seen
            _keep.append(_ok)
            if _ok: _seen.add(_k)
        _P = _P.loc[_keep].copy(); _parts.append(_P)
    _fitdf = pd.concat(_parts, ignore_index=True)
    _fit = _fitdf.text.astype(str).tolist(); _fy = _fitdf.model.map(L2I).to_numpy(dtype=int); _sw = _fitdf['_srcw'].to_numpy(dtype=np.float32)
    log(f'svc4 fit rows={len(_fit)} source_weighted; external clean={len(EXTRA_DF)}')
    _A, _B = [], []
    for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]:
        prep = _shape if view == "shape" else _func if view == "function" else (lambda s_: s_)
        an = "char" if view == "shape" else "word" if view == "function" else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
        if an == "word": o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        _v = TfidfVectorizer(**o); _A.append(_v.fit_transform([prep(t) for t in _fit]) * wv); _B.append(_v.transform([prep(t) for t in test.text.values]) * wv)
    _svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
    _svc.fit(_hs(_A, format='csr'), _fy, sample_weight=_sw)
    _d = _svc.decision_function(_hs(_B, format='csr'))
    LP['svc4'] = (_d - _d.max(1, keepdims=True) - np.log(np.exp(_d - _d.max(1, keepdims=True)).sum(1, keepdims=True))) * a.svc_w
    del _A, _B, _parts, _fitdf, _svc; gc.collect(); log('svc4 done')
if a.distort_w > 0:
    import re, collections
    wre = re.compile(r"[A-Za-z’']+")
    cnt = collections.Counter(w.lower() for t in FIT_T for w in wre.findall(t))
    keep = {w for w, _ in cnt.most_common(100)}
    dist = lambda arr: [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in keep else '*', t) for t in arr]
    Dtr, Dte = dist(FIT_T), dist(test.text.values)
    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer='char', ngram_range=(2, 6), min_df=2),
          TfidfVectorizer(sublinear_tf=True, lowercase=True, token_pattern=r"[^\s]+", ngram_range=(1, 3), min_df=2)]
    Xa = sp.hstack([v_.fit_transform(Dtr) for v_ in vs]).tocsr(); Xb = sp.hstack([v_.transform(Dte) for v_ in vs]).tocsr()
    LP['distort'] = LogisticRegression(C=10, max_iter=3000).fit(Xa, FIT_Y, sample_weight=FIT_W).predict_log_proba(Xb) * a.distort_w
    log('distort done')
if a.prefit_core:
    from ntl660_readout import core_votes
    _pv = core_votes(a.prefit_core, test.text.tolist())
    assert set(_pv) == {'svc4', 'tfidf', 'distort'} and all(v.shape == (len(test), 18) and np.isfinite(v).all() for v in _pv.values())
    LP.update(_pv)
    log('prefit core votes loaded', {k: round(float(abs(v).max()), 2) for k, v in _pv.items()})
from transformers import AutoTokenizer, AutoModel
@torch.no_grad()
def l24_predict(spec, texts, groups=None, train_texts=None):
    from transformers import AutoConfig
    from safetensors.torch import load_file
    from concurrent.futures import ThreadPoolExecutor
    adapters = spec.get('adapters') or [spec['adapter']]
    aw = spec.get('adapter_weights')
    if aw is None or len(aw) != len(adapters): aw = [1.0 / len(adapters)] * len(adapters)
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(spec['path']); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = spec['keep']; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None: tc.layer_types = tc.layer_types[:spec['keep']]
    _ms = []
    if spec.get('dp') and torch.cuda.device_count() >= 2:
        try:
            for _d in range(2):
                _ms.append(AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map={'': _d}).eval())
                for _n in ('vision_tower', 'multi_modal_projector'):
                    if getattr(_ms[-1], _n, None) is not None: setattr(_ms[-1], _n, None)
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(_d)
                _ms[-1](input_ids=torch.full((spec.get('bs', 16), spec.get('max_len', 384)), tok.pad_token_id, device=f'cuda:{_d}'), output_hidden_states=True, use_cache=False)
                assert torch.cuda.max_memory_allocated(_d) < torch.cuda.get_device_properties(_d).total_memory - (1 << 30), 'under 1 GiB to spare'
        except Exception as _e:
            log(f'  l24 {spec["name"]}: one split copy instead of one per GPU ({_e!r:.80})'); _ms = []
        gc.collect(); torch.cuda.empty_cache()
    if not _ms: _ms = [AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()]
    m = _ms[0]
    if spec.get('no_final_norm'):
        for _m in _ms: _m.norm = torch.nn.Identity()
    mods = dict(m.named_modules()); first = next(m.parameters()).device; L = spec['layers']
    _mods = [dict(_m.named_modules()) for _m in _ms]
    _eos = spec.get('eos_id')
    _tk = lambda tx: [x + ([_eos] if _eos is not None else []) for x in tok(list(tx), truncation=True, max_length=spec.get('max_len', 384) - (1 if _eos is not None else 0))['input_ids']]
    enc = _tk(texts)
    order = np.argsort([len(x) for x in enc]); bs = spec.get('bs', 16)
    def lsm(z): return z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
    P_acc, C_acc = 0.0, 0.0; FEATS = None; OVF = [0, 0]
    if train_texts is not None:
        enc_tr = _tk(train_texts)
    def featurize(enc_, norms, hid):
        order_ = np.argsort([len(x) for x in enc_])
        Z_ = np.zeros((len(enc_), hid * len(L)), np.float32)
        batches = [order_[i:i + bs] for i in range(0, len(order_), bs)]

        def feats(mm, b, check=True):
            dev = next(mm.parameters()).device
            width = max(len(enc_[j]) for j in b)
            ids = torch.full((len(b), width), tok.pad_token_id)
            att = torch.zeros((len(b), width), dtype=torch.long)
            for r, j in enumerate(b):
                ids[r, :len(enc_[j])] = torch.tensor(enc_[j])
                att[r, :len(enc_[j])] = 1
            with torch.no_grad():
                hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
                if check and not all(torch.isfinite(hs[l]).all() for l in L):
                    return None
                mask = att.unsqueeze(-1).float()
                mask[:, 0] = 0
                mg = {d_: mask.to(d_) for d_ in {hs[l].device for l in L}}
                return torch.cat([nm(((hs[l].float() * mg[hs[l].device]).sum(1) / mg[hs[l].device].sum(1)).cpu()) for l, nm in zip(L, norms)], -1).numpy()

        def put(b, r):
            Z_[b] = r

        def checked(k, b):
            mm = _ms[k]
            r = feats(mm, b)
            if r is None:
                mm.to(torch.bfloat16)
                r = feats(mm, b, check=False)
                mm.to(torch.float16)
                OVF[k] += 1
            put(b, r)

        def run_copy(k, n):
            for b in batches[k::n]:
                checked(k, b)

        if len(_ms) == 1:
            if spans_devices(_ms[0]):
                pipelined(batches, lambda b: feats(_ms[0], b), put, lambda b: checked(0, b), workers=2, tag='l24 ' + spec['name'], log=log)
            else:
                for b in batches:
                    checked(0, b)
        else:
            try:
                with ThreadPoolExecutor(len(_ms)) as _ex:
                    for _f in [_ex.submit(run_copy, k, len(_ms)) for k in range(len(_ms))]:
                        _f.result()
            except Exception as _e:
                log(f'  l24 {spec["name"]}: two-copy pass failed ({_e!r:.80}), one copy from here on')
                del _ms[1:], _mods[1:]
                gc.collect()
                torch.cuda.empty_cache()
                for b in batches:
                    checked(0, b)
        return Z_
    for ai, ad in enumerate(adapters):
        adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
        acfg = json.load(open(os.path.join(adp, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(adp, 'adapter_model.safetensors')); deltas = []
        for k, v in sd.items():
            if '.lora_A.' not in k: continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods: name = 'model.' + name
            B = sd[k.replace('.lora_A.', '.lora_B.')]
            for _md in _mods:
                W = _md[name].weight
                W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
                deltas.append((W, B, v))
        if os.path.exists(os.path.join(ad, 'heads.pt')): H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu')
        else: H = dict(load_file(os.path.join(ad, 'heads.safetensors')))
        if os.path.exists(os.path.join(ad, 'meta.json')):
            _lab = json.load(open(os.path.join(ad, 'meta.json')))['labels']; _perm = [_lab.index(l) for l in LABELS]
            for _p in ('plain', 'centered'): H[f'{_p}.weight'] = H[f'{_p}.weight'][_perm]; H[f'{_p}.bias'] = H[f'{_p}.bias'][_perm]
        hid = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid) for _ in L]
        for i, nm in enumerate(norms): nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        Z = featurize(enc, norms, hid)
        if train_texts is not None and ai == len(adapters) - 1:
            FEATS = (Z.copy(), featurize(enc_tr, norms, hid))
        for W, B, A_ in deltas:
            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
        torch.cuda.empty_cache()
        del deltas, sd; gc.collect()
        hd = lambda p, X: lsm(X @ H[f'{p}.weight'].float().numpy().T + H[f'{p}.bias'].float().numpy())
        P_acc = P_acc + hd('plain', Z) * aw[ai]
        if groups is not None:
            Zc = Z.copy()
            for g in np.unique(groups):
                ii = groups == g; Zc[ii] -= Zc[ii].mean(0)
            C_acc = C_acc + hd('centered', Zc) * aw[ai]
        log(f'  l24 {spec["name"]}: adapter {os.path.basename(ad)} done (bf16 redo batches {sum(OVF)})')
    del m, mods, _ms, _mods; gc.collect(); torch.cuda.empty_cache()
    if train_texts is not None: return P_acc, (C_acc if groups is not None else None), FEATS
    return P_acc, (C_acc if groups is not None else None)

def gpu_mem():
    return ' '.join(f'gpu{i}:{torch.cuda.memory_allocated(i)/1e9:.1f}GB' for i in range(torch.cuda.device_count()))

specs = json.loads(a.models)
failed = []


def run_l24(spec):
    log('l24', spec['name'])
    out = {}
    if spec.get('gauss', 0) > 0:
        plain, cent, (Zte_g, Ztr_g) = l24_predict(spec, test.text.values, Gte if a.adapt_w > 0 else None, train_texts=train.text.values)
        import gauss as _ga
        prm = _ga.fit(Ztr_g, y, Gtr, dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
        prm['T1'] = prm['Tg'] = 3.0
        _sz = np.bincount(Gte)[Gte]
        _rows = (_sz >= 2) & (_sz <= 18)
        _q = _ga.scores(Ztr_g[[0, 1]], np.zeros(2, dtype=int), prm)
        assert np.allclose(np.exp(_q).sum(1), 1)
        _G = np.zeros((len(test), 18))
        if _rows.any():
            _G[_rows] = _ga.scores(Zte_g[_rows], np.unique(Gte[_rows], return_inverse=True)[1], prm)
        out[spec['name'] + '_gauss'] = _G * spec['gauss']
        log(f"  gaussian head {spec['name']}: {int(_rows.sum())} grouped test rows, weight {spec['gauss']}")
    else:
        plain, cent = l24_predict(spec, test.text.values, Gte if a.adapt_w > 0 else None)
    out[spec['name']] = plain * spec.get('w', 1.0)
    if cent is not None:
        rows = np.bincount(Gte)[Gte] >= 2
        out[spec['name'] + '_twin'] = np.where(rows[:, None], cent, 0.0) * a.adapt_w * spec.get('w', 1.0) * spec.get('twin', 1.0)
    LP.update(out)


def run_enc(spec):
    log('enc', spec['name'])
    from safetensors.torch import load_file as _lf
    _ck = spec['ckpt']
    _meta = json.load(open(os.path.join(_ck, 'meta.json')))
    _perm = [_meta['labels'].index(l) for l in LABELS]
    _hd = _lf(os.path.join(_ck, 'head.safetensors'))
    _W = _hd['weight'].float().numpy()[_perm]
    _b = _hd['bias'].float().numpy()[_perm]
    _etok = AutoTokenizer.from_pretrained(os.path.join(_ck, 'encoder'))
    _enc = AutoModel.from_pretrained(os.path.join(_ck, 'encoder'), dtype=torch.float32).to('cuda:0').eval()
    _ids = _etok(list(test.text.values), truncation=True, max_length=_meta['max_len'])['input_ids']
    _ord = np.argsort([len(x) for x in _ids], kind='stable')
    Z_ = np.zeros((len(_ids), _enc.config.hidden_size), np.float32)
    with torch.inference_mode():
        for _s in range(0, len(_ord), 32):
            _idx = _ord[_s:_s + 32]
            _bt = _etok.pad({'input_ids': [_ids[i] for i in _idx]}, return_tensors='pt').to('cuda:0')
            _m = _bt['attention_mask'].unsqueeze(-1).float()
            Z_[_idx] = ((_enc(**_bt).last_hidden_state * _m).sum(1) / _m.sum(1).clamp(min=1)).cpu().numpy()
    del _enc
    _zz = Z_ @ _W.T + _b
    LP[spec['name']] = (_zz - _zz.max(1, keepdims=True) - np.log(np.exp(_zz - _zz.max(1, keepdims=True)).sum(1, keepdims=True))) * spec.get('w', 1.0)


RUNNERS = {'l24': run_l24, 'enc': run_enc}
for spec in specs:
    if spec.get('optional') and a.optional_deadline and time.time() > a.optional_deadline:
        log('MEMBER SKIPPED (time):', spec['name'])
        failed.append(spec['name'])
        continue
    try:
        RUNNERS[spec['type']](spec)
    except Exception as _e:
        failed.append(spec['name'])
        log(f"MEMBER FAILED {spec['name']}: {_e!r}")
    gc.collect()
    torch.cuda.empty_cache()
    log(spec['name'], 'done', gpu_mem())
log('members failed:', failed if failed else 'none')

tot = sum(LP.values())
assert not failed, failed
member_output = os.environ['NTL_MEMBER_OUTPUT']
member_terms = {'term_'+k:np.asarray(v) for k,v in LP.items() if k not in ('tfidf','distort','svc4')}
assert member_terms and all(v.shape == (len(test),18) and np.isfinite(v).all() for v in member_terms.values())
np.savez(member_output+'.tmp.npz', ids=test.id.astype(str).values.astype('U'), labels=np.array(LABELS), **member_terms)
os.replace(member_output+'.tmp.npz',member_output)
if a.save_lp:
    np.save(a.save_lp, tot)
COMP_P = None
if a.comp_w > 0 and a.comp_path and os.path.exists(a.comp_path):
    try:
        _cm = json.load(open(a.comp_path.replace('comp_logit.npy', 'comp_meta.json')))
        assert _cm['ids'] == test['id'].astype(str).tolist(), 'comp stack row order differs'
        _cl = np.load(a.comp_path)[:, [_cm['labels'].index(l) for l in LABELS]]
        COMP_P = np.exp(_cl - _cl.max(1, keepdims=True))
        COMP_P /= COMP_P.sum(1, keepdims=True)
        assert np.isfinite(COMP_P).all()
        log(f"comp stack loaded ({_cm['seconds']:.0f}s)")
    except Exception as _e:
        log('COMP SKIPPED:', repr(_e))
        COMP_P = None
elif a.comp_w > 0:
    log('COMP SKIPPED: no stack output')


def blend(terms):
    total = sum(terms.values())
    active = sum(abs(v).max() > 0 for v in terms.values())
    lpn = total / max(1, active) if a.sk_div else total
    P = np.exp(lpn - lpn.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    if COMP_P is not None:
        P = (1 - a.comp_w) * P + a.comp_w * COMP_P
        P /= P.sum(1, keepdims=True)
    return P


def decide_scores(P):
    Q = P.copy()
    rho = float(np.clip(a.sinkhorn_target_mix, 0.0, 1.0))
    marg = np.clip(P.mean(0), 1e-6, None)
    marg /= marg.sum()
    prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0
    prior /= prior.sum()
    target = np.clip((1.0 - rho) * marg + rho * prior, 1e-6, None)
    target /= target.sum()
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)


def decide(P):
    return decide_scores(P).argmax(1)


def save_final_logits(z):
    if a.save_final_logits:
        np.savez(a.save_final_logits, ids=test.id.astype(str).values.astype('U'),
                 labels=np.array(LABELS), logits=np.asarray(z, dtype=np.float64))


def write_pred(pred, tag):
    sub = pd.DataFrame({'id': test.id.values, 'model': [LABELS[i] for i in pred]})
    assert sub.model.isin(LABELS).all() and sub.id.is_unique and len(sub) == len(test)
    sub.to_parquet(a.out, index=False)
    log(f'wrote {tag}', sub.model.value_counts().to_dict())


def write(P, tag):
    save_final_logits(decide_scores(P))
    write_pred(decide(P), tag)


P = blend(LP)
write(P, 'blend without ATLAS')


def atlas_lite(k):
    import atlas_infer
    scorer = atlas_infer.Scorer(a.atlas_base, a.atlas_dir, log=log)
    try:
        enc = scorer.encode(test.text.values)
        to_b = np.array([scorer.labels.index(l) for l in LABELS])
        cand = np.argsort(-P, 1)[:, :k]
        mean_lp, done = scorer.run_candidates(enc, to_b[cand], a.atlas_deadline or None, workers=a.atlas_workers)
        lp = scorer.candidate_logprobs(mean_lp[done], to_b[cand][done])
    finally:
        scorer.close()
    rows = np.repeat(lp.min(1, keepdims=True), 18, 1)
    np.put_along_axis(rows, cand[done], lp, 1)
    term = np.zeros((len(test), 18))
    term[done] = rows
    return term, done


def atlas_scores(rows, calibrate):
    import atlas_infer
    scorer = atlas_infer.Scorer(a.atlas_base, a.atlas_dir, log=log)
    try:
        enc = scorer.encode(test.text.values)
        out, done = scorer.run(enc, rows, deadline=a.atlas_deadline or None, batch=a.atlas_bs, chunk=a.atlas_chunk, workers=a.atlas_workers)
        order = [scorer.labels.index(l) for l in LABELS]
        if calibrate:
            res = np.zeros((len(test), 18))
            res[done] = scorer.calibrated(out[done])
        else:
            raw = out[..., 1:].mean(-1)
            res = np.divide(raw, raw.sum(1, keepdims=True), out=np.zeros_like(raw), where=raw.sum(1, keepdims=True) > 0)
    finally:
        scorer.close()
    return res[:, order], done


if a.atlas_mode != 'off' and a.atlas_dir:
    if a.atlas_deadline and time.time() > a.atlas_deadline - 600:
        log('ATLAS SKIPPED: not enough time left')
    else:
        try:
            if a.atlas_mode == 'lite':
                term, done = atlas_lite(a.atlas_k)
                P = blend({**LP, 'atlas': a.atlas_member_w * term})
                log(f'atlas lite top-{a.atlas_k} on {int(done.sum())}/{len(test)} rows at weight {a.atlas_member_w}', gpu_mem())
            elif a.atlas_mode == 'head':
                lp, done = atlas_scores(np.arange(len(test)), True)
                P = blend({**LP, 'atlas': a.atlas_member_w * lp})
                log(f'atlas head on {int(done.sum())}/{len(test)} rows at weight {a.atlas_member_w}', gpu_mem())
            else:
                top2 = np.sort(P, 1)
                margin = top2[:, -1] - top2[:, -2]
                rows = np.where(margin < a.atlas_margin)[0] if a.atlas_margin > 0 else np.arange(len(test))
                A_, done = atlas_scores(rows, False)
                use = rows[done[rows]]
                mixed = P.copy()
                mixed[use] += a.atlas_w * (A_[use] - P[use])
                P = mixed / mixed.sum(1, keepdims=True)
                log(f'atlas mixed on {len(use)}/{len(rows)} rows at weight {a.atlas_w}', gpu_mem())
        except Exception as _e:
            log('ATLAS SKIPPED:', repr(_e))
        write(P, 'final blend')


if a.gsq_share > 0 and a.gsq_dir:
    if a.gsq_deadline and time.time() > a.gsq_deadline:
        log('GSQ SKIPPED: not enough time left')
    else:
        try:
            import gsq_stage as _gs
            import shutil as _sh
            _sh.rmtree('/tmp/llama-3.1', ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            Q27 = _gs.run(a.gsq_dir, test['id'].astype(str).tolist(), test.text.tolist(), Gte, a.gsq_work, LABELS, log, a.gsq_timeout or None)
            _agree = float((Q27.argmax(1) == P.argmax(1)).mean())
            log(f'GSQ 27B agreement with the blend {_agree:.3f}')
            if _agree < a.gsq_min_agree:
                raise RuntimeError(f'GSQ agreement {_agree:.3f} below {a.gsq_min_agree}: refusing to use it')
            P = (1 - a.gsq_share) * P + a.gsq_share * Q27
            P /= P.sum(1, keepdims=True)
            write(P, 'blend with GSQ 27B')
        except Exception as _e:
            log('GSQ SKIPPED:', repr(_e))


def rowsoftmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def native_stage(P):
    import subprocess
    import native_q8
    os.makedirs(a.native_work, exist_ok=True)
    out = os.path.join(a.native_work, 'native.npz')
    if os.path.exists(out):
        os.remove(out)
    gc.collect()
    torch.cuda.empty_cache()
    cmd = [sys.executable, '-u', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native_q8.py'), '--base', a.native_base, '--assets', a.native_dir,
           '--query', os.path.join(a.data, 'test.csv'), '--out', out, '--work', a.native_work, '--deadline', str(a.native_deadline)]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    code = subprocess.run(cmd, env=env).returncode
    if not os.path.exists(out):
        raise RuntimeError(f'native worker produced nothing (exit code {code})')
    z = np.load(out, allow_pickle=False)
    folds = z['folds'].tolist()
    assert z['ids'].astype(str).tolist() == test['id'].astype(str).tolist(), 'native row order differs'
    labels = z['labels'].astype(str).tolist()
    order = [labels.index(l) for l in LABELS]
    states = [json.load(open(os.path.join(a.native_dir, 'calibration', f'fold{k}.json')))['models']['qwen8'] for k in folds]
    lp = native_q8.calibrate(z['plain'][:, :, order], z['centered'][:, :, order], z['centered_bias'][:, order], Gte, states, bool(a.native_cent))
    share = (a.native_share * z['finite'])[:, None]
    logP = np.log(np.maximum(P, 1e-15))
    agree = [float((lp[i].argmax(1) == P.argmax(1)).mean()) for i in range(len(folds))]
    log(f'native folds {folds} (worker exit {code}); agreement with the blend per fold', [round(v, 3) for v in agree],
        'rows without native', int((~z['finite']).sum()))
    if min(agree) < a.native_min_agree:
        raise RuntimeError(f'native agreement {min(agree):.3f} below {a.native_min_agree}: refusing to use it')
    probs = np.zeros_like(P)
    for i in range(len(folds)):
        probs += rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i]))) / len(folds)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)
    return probs


if a.native_share > 0 and a.native_dir and a.native_base:
    if a.native_deadline and time.time() > a.native_deadline - 1500:
        log('NATIVE SKIPPED: not enough time left')
    else:
        try:
            probs = native_stage(P)
            changed = int((probs.argmax(1) != decide(P)).sum())
            write_pred(probs.argmax(1), 'final blend with native q8')
            save_final_logits(np.log(np.maximum(probs, 1e-15)))
            log(f'native q8 at share {a.native_share}: {changed}/{len(test)} decisions differ from the blend-only decision', gpu_mem())
        except Exception as _e:
            log('NATIVE SKIPPED:', repr(_e))
