import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def stage(base, assets, work):
    base, assets, staged = Path(base), Path(assets), Path(work) / 'qgroup-base'
    staged.mkdir(parents=True, exist_ok=True)
    index = json.loads((assets / 'base' / 'model.safetensors.index.json').read_text())
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
        target = staged / name
        if not target.exists():
            shutil.copyfile(assets / 'base' / name, target)
    for name in sorted(set(index['weight_map'].values())):
        source, target = base / name, staged / name
        assert source.is_file(), str(source)
        if not target.exists():
            try:
                os.symlink(source, target)
            except OSError:
                shutil.copyfile(source, target)
    return staged


def main():
    p = argparse.ArgumentParser()
    for key in ('base', 'assets', 'query', 'out', 'work'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--tok_budget', type=int, default=8192)
    p.add_argument('--mem_frac', type=float, default=0.88)
    a = p.parse_args()
    started = time.time()
    frame = pd.read_csv(a.query, dtype={'id': str})
    texts = frame.text.fillna('').astype(str).tolist()
    staged = stage(a.base, a.assets, a.work)
    tok = AutoTokenizer.from_pretrained(staged, local_files_only=True)
    auto, plain = tok('hello')['input_ids'], tok('hello', add_special_tokens=False)['input_ids']
    eos = auto[-1] if (len(auto) > len(plain) and auto[-1] != plain[-1]) else (tok.eos_token_id if tok.eos_token_id is not None else 151643)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    gpus = torch.cuda.device_count()
    assert gpus >= 1
    max_memory = {i: f'{int(torch.cuda.get_device_properties(i).total_memory * a.mem_frac / 2 ** 30)}GiB' for i in range(gpus)}
    model = AutoModel.from_pretrained(staged, local_files_only=True, dtype=torch.float16, device_map='auto', max_memory=max_memory).eval()
    first = next(model.parameters()).device
    ids = []
    for t in texts:
        x = tok(t, truncation=True, max_length=a.max_len - 1)['input_ids']
        ids.append(x if x[-1] == eos else x + [eos])
    order = np.argsort([len(x) for x in ids], kind='stable')
    emb = np.zeros((len(texts), model.config.get_text_config().hidden_size), np.float16)
    budget, done, ooms = a.tok_budget, 0, 0
    with torch.inference_mode():
        while done < len(order):
            end = done + 1
            while end < len(order) and (end - done + 1) * len(ids[order[end]]) <= budget:
                end += 1
            idx = order[done:end]
            try:
                width = max(len(ids[i]) for i in idx)
                inp = torch.full((len(idx), width), pad)
                att = torch.zeros((len(idx), width), dtype=torch.long)
                for r, i in enumerate(idx):
                    inp[r, :len(ids[i])] = torch.tensor(ids[i])
                    att[r, :len(ids[i])] = 1
                hidden = model(input_ids=inp.to(first), attention_mask=att.to(first), use_cache=False).last_hidden_state
                vec = hidden[torch.arange(len(idx), device=hidden.device), (att.sum(1) - 1).to(hidden.device)].float()
                emb[idx] = torch.nn.functional.normalize(vec, dim=-1).cpu().numpy().astype(np.float16)
                done = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ooms += 1
                budget = max(len(ids[order[done]]), budget // 2)
                print(f'qwen_groups: OOM, token budget -> {budget}', flush=True)
    assert np.isfinite(emb).all()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, emb)
    print(f'qwen_groups: {len(texts)} rows, dim {emb.shape[1]}, {ooms} OOM retries, {time.time() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
