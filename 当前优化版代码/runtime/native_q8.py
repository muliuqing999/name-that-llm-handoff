import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

INSTRUCTION = 'Instruct: Identify the language model that generated the text based on its writing style.\nQuery: '
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
RANK = 16
SCALE = 2.0
TOKEN_LIMIT = 512
TERMINAL = 151643


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def calibrate(plain, centered, bias, groups, selections, use_centered=True):
    plain, centered, bias = (np.asarray(x, dtype=np.float64) for x in (plain, centered, bias))
    groups = np.asarray(groups)
    folds = plain.shape[0]
    assert plain.shape == centered.shape and plain.ndim == 3 and plain.shape[-1] == 18
    assert bias.shape == (folds, 18) and groups.shape == (plain.shape[1],) and len(selections) == folds
    assert all(np.isfinite(x).all() for x in (plain, centered, bias))
    _, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    eligible = (count[inverse] >= 2) & (count[inverse] <= 18)
    out = []
    for k, state in enumerate(selections):
        assert not state['outer_labels_used'] and state['centered_head_trained']
        assert state['plain_scale'] > 0 and state['centered_scale'] > 0
        means = np.zeros((len(count), 18))
        np.add.at(means, inverse, centered[k])
        means /= count[:, None]
        relative = centered[k] - means[inverse] + bias[k]
        a = log_softmax(plain[k] * state['plain_scale'])
        b = log_softmax(relative * state['centered_scale'])
        weight = float(state['selected_centered_weight'])
        assert 0 <= weight <= 0.5
        fraction = np.where(eligible & use_centered, weight, 0.0)[:, None]
        out.append(log_softmax((1 - fraction) * a + fraction * b))
    return np.stack(out)


def windows(tokenizer, texts):
    prefix = tokenizer(INSTRUCTION, add_special_tokens=False)['input_ids']
    assert tokenizer('', add_special_tokens=True)['input_ids'] == [TERMINAL], 'unexpected end-of-text handling'
    capacity = TOKEN_LIMIT - len(prefix) - 1
    first_capacity = TOKEN_LIMIT - 1
    full = tokenizer([INSTRUCTION + str(t) for t in texts], truncation=False, add_special_tokens=False)['input_ids']
    seqs, owners, weights = [], [], []
    for row, tokens in enumerate(full):
        assert len(tokens) > 0
        seqs.append(tokens[:first_capacity] + [TERMINAL])
        owners.append(row)
        weights.append(max(1, min(len(tokens), first_capacity) - len(prefix)))
        pos = first_capacity
        while pos < len(tokens):
            chunk = tokens[pos:pos + capacity]
            seqs.append(prefix + chunk + [TERMINAL])
            owners.append(row)
            weights.append(len(chunk))
            pos += len(chunk)
    return seqs, np.asarray(owners), np.asarray(weights, dtype=np.float64)


class NativeQ8:
    def __init__(self, base, assets, work, log=print, gpus=None):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.nn, self.log = torch, nn, log
        torch.set_num_threads(2)
        for key in ('allow_tf32', 'allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction'):
            setattr(torch.backends.cuda.matmul, key, False)
        torch.backends.cudnn.allow_tf32 = False
        self.assets = Path(assets)
        self.labels = json.loads((self.assets / 'manifest.json').read_text())['classes']
        origin = Path(base)
        index = json.loads((self.assets / 'base' / 'model.safetensors.index.json').read_text())
        shards = sorted(set(index['weight_map'].values()))
        small = ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json']
        self.staged = Path(work) / 'native-q8-base'
        self.staged.mkdir(parents=True, exist_ok=True)
        for name in small:
            src = self.assets / 'base' / name
            assert src.is_file(), str(src)
            dst = self.staged / name
            if not dst.exists():
                shutil.copyfile(src, dst)
        for name in shards:
            src, dst = origin / name, self.staged / name
            assert src.is_file(), str(src)
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
        for k in range(5):
            assert (self.assets / f'fold{k}' / 'trainables.safetensors').is_file(), k
        cfg = json.loads((self.staged / 'config.json').read_text())
        self.layers, self.hidden = cfg['num_hidden_layers'], cfg['hidden_size']
        assert self.layers == 36
        gpus = gpus if gpus is not None else torch.cuda.device_count()
        assert gpus >= 1
        self.head_device = 'cuda:0'
        place = {'embed_tokens': 0, 'norm': gpus - 1, 'rotary_emb': gpus - 1}
        cut = 16 if gpus >= 2 else self.layers
        place.update({f'layers.{i}': 0 if i < cut else 1 for i in range(self.layers)})
        self.tokenizer = AutoTokenizer.from_pretrained(self.staged, local_files_only=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        loaded = AutoModel.from_pretrained(self.staged, local_files_only=True, dtype=torch.float16, device_map=place,
                                           attn_implementation='sdpa', output_loading_info=True)
        if isinstance(loaded, tuple):
            self.model, info = loaded
            assert not any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs', 'unexpected_keys')), info
        else:
            self.model = loaded
        self.model.config.use_cache = False
        self.loras = {}
        for name, mod in list(self.model.named_modules()):
            if isinstance(mod, nn.Linear) and name.rsplit('.', 1)[-1] in SUFFIXES:
                wrapped = self._lora(mod)
                parent, attr = name.rsplit('.', 1)
                setattr(self.model.get_submodule(parent), attr, wrapped)
                self.loras[name] = wrapped
        assert len(self.loras) == self.layers * len(SUFFIXES), len(self.loras)
        dev = self.head_device
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden, device=dev, dtype=torch.float32) for _ in DEPTHS])
        self.plain = nn.Linear(self.hidden * len(DEPTHS), 18, device=dev, dtype=torch.float32)
        self.centered = nn.Linear(self.hidden * len(DEPTHS), 18, device=dev, dtype=torch.float32)
        self.model.requires_grad_(False).eval()
        for m in (self.norms, self.plain, self.centered):
            m.requires_grad_(False).eval()
        self.expected = {}
        for name, mod in self.loras.items():
            self.expected[f'encoder.backbone.base_model.model.{name}.lora_A.default.weight'] = mod.a.weight
            self.expected[f'encoder.backbone.base_model.model.{name}.lora_B.default.weight'] = mod.b.weight
        for prefix, m in (('norms', self.norms), ('plain', self.plain), ('centered', self.centered)):
            self.expected.update({f'{prefix}.{n}': p for n, p in m.named_parameters()})
        assert len(self.expected) == 2 * len(self.loras) + 2 * len(DEPTHS) + 4
        self.current = None

    def _lora(self, base):
        torch, nn = self.torch, self.nn

        class TaskLoRA(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.base_layer = layer
                dev = layer.weight.device
                self.a = nn.Linear(layer.in_features, RANK, bias=False, device=dev, dtype=torch.float32)
                self.b = nn.Linear(RANK, layer.out_features, bias=False, device=dev, dtype=torch.float32)

            def forward(self, x):
                result = self.base_layer(x)
                return (result + self.b(self.a(x.float())) * SCALE).to(result.dtype)

        return TaskLoRA(base)

    def load_fold(self, fold):
        if self.current == fold:
            return
        from safetensors.torch import load_file
        torch = self.torch
        state = load_file(str(self.assets / f'fold{fold}' / 'trainables.safetensors'))
        assert set(state) == set(self.expected), (sorted(set(state) - set(self.expected))[:3], sorted(set(self.expected) - set(state))[:3])
        with torch.inference_mode():
            for name, param in self.expected.items():
                assert state[name].shape == param.shape and state[name].dtype == torch.float32, name
                assert torch.isfinite(state[name]).all(), name
                param.copy_(state[name].to(param.device))
        del state
        gc.collect()
        self.current = fold

    def _features(self, seqs, idx):
        torch = self.torch
        batch = self.tokenizer.pad({'input_ids': [seqs[i] for i in idx]}, padding=True, return_tensors='pt').to('cuda:0')
        out = self.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        assert len(out.hidden_states) == self.layers + 1
        pos = batch['attention_mask'].sum(1) - 1
        rows = torch.arange(len(idx))
        parts = []
        for depth, norm in zip(DEPTHS, self.norms):
            hs = out.last_hidden_state if depth == self.layers else out.hidden_states[depth]
            pooled = hs[rows.to(hs.device), pos.to(hs.device)].float()
            if depth == self.layers:
                pooled = self.nn.functional.normalize(pooled, dim=-1)
            parts.append(norm(pooled.to(self.head_device)))
        return torch.cat(parts, dim=-1)

    def predict(self, texts, folds=range(5), batch_size=4, deadline=None, on_fold=None):
        torch = self.torch
        seqs, owners, weights = windows(self.tokenizer, texts)
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), seqs[i]))
        n = len(texts)
        res = {'plain': [], 'centered': [], 'centered_bias': [], 'folds': []}
        finite = np.ones(n, dtype=bool)
        for fold in folds:
            if deadline is not None and time.time() > deadline:
                self.log(f'native: deadline reached before fold {fold}')
                break
            self.load_fold(fold)
            t0 = time.time()
            vals = {'plain': np.zeros((n, 18)), 'centered': np.zeros((n, 18))}
            with torch.inference_mode():
                for s in range(0, len(order), batch_size):
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError('native deadline inside fold; completed-fold checkpoint retained')
                    idx = order[s:s + batch_size]
                    try:
                        feats = self._features(seqs, idx)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        feats = torch.cat([self._features(seqs, [i]) for i in idx], 0)
                    ok = torch.isfinite(feats).all(1).cpu().numpy()
                    if not ok.all():
                        for i, good in zip(idx, ok):
                            if not good:
                                finite[owners[i]] = False
                        feats = torch.nan_to_num(feats)
                    own = owners[idx]
                    for name, head in (('plain', self.plain), ('centered', self.centered)):
                        sc = head(feats).float().cpu().numpy().astype(np.float64)
                        np.add.at(vals[name], own, sc * weights[idx][:, None])
                    del feats
            denom = np.bincount(owners, weights=weights, minlength=n)[:, None]
            for name in ('plain', 'centered'):
                res[name].append(vals[name] / denom)
            res['centered_bias'].append(self.centered.bias.detach().cpu().numpy().astype(np.float64).copy())
            res['folds'].append(int(fold))
            self.log(f'native: fold {fold} done, {n} rows, {len(seqs)} windows, {time.time() - t0:.0f}s, non-finite rows {int((~finite).sum())}')
            if on_fold is not None:
                on_fold({k: np.stack(v) if k != 'folds' else np.asarray(v) for k, v in res.items()}, finite)
        return {k: np.stack(v) if k != 'folds' else np.asarray(v) for k, v in res.items()}, finite


def save(path, ids, labels, res, finite):
    tmp = str(path) + '.tmp.npz'
    np.savez(tmp, ids=np.asarray(ids, dtype=str), labels=np.asarray(labels), finite=finite, **res)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--deadline', type=float, default=0.0)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    T0 = time.time()

    def log(*x):
        print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)

    frame = pd.read_csv(a.query, dtype={'id': str})
    assert frame.id.is_unique and len(frame)
    if a.limit:
        frame = frame.iloc[:a.limit]
    texts = frame.text.fillna('').astype(str).tolist()
    model = NativeQ8(a.base, a.assets, a.work, log=log)
    log('native: backbone and heads built')
    folds=[int(x) for x in os.environ.get('NTL_NATIVE_FOLDS','0,1,2').split(',')]
    res, finite = model.predict(texts, folds=folds, batch_size=a.batch, deadline=a.deadline or None,
                                on_fold=lambda r, f: save(a.out, frame.id.values, model.labels, r, f))
    assert len(res['folds']), 'no fold finished'
    save(a.out, frame.id.values, model.labels, res, finite)
    log('native: saved', a.out, 'folds', res['folds'].tolist())


if __name__ == '__main__':
    main()
