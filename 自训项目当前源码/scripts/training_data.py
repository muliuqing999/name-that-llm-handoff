"""Deterministic data operations shared by preparation, training and tests."""
import hashlib
import random
import re


def text_key(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def content_key(text):
    return hashlib.sha256(" ".join(re.findall(r"\w+", text.casefold())).encode()).hexdigest()


def split_groups(rows, seed=20261028):
    groups = {}
    for row in rows:
        groups.setdefault(str(row["prompt_id"]), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 10:
        raise ValueError("At least ten prompt groups are required")
    cut = max(2, round(len(keys) * 0.2))
    return [groups[k] for k in keys[cut:]], [groups[k] for k in keys[:cut]], keys[:cut]


def balanced_subset(rows, labels, maximum, seed):
    """Cap each class without replacement; never invent or relabel samples."""
    by_label = {label: [] for label in labels}
    for row in rows:
        by_label[row['model']].append(row)
    if any(not value for value in by_label.values()):
        raise ValueError('External data must contain every competition label')
    rng = random.Random(seed)
    cap = max(1, maximum // len(labels))
    selected = []
    for label in labels:
        # Sorting makes selection independent of parquet file row order.
        group = sorted(by_label[label], key=lambda row: text_key(row['text']))
        rng.shuffle(group)
        selected.extend(group[:cap])
    rng.shuffle(selected)
    return selected


def stage_batches(groups, seed, grouped, global_batch=18):
    rng = random.Random(seed)
    if grouped:
        units = []
        for group in groups:
            group = list(group)
            rng.shuffle(group)
            units.extend(group[i:i+3] for i in range(0, len(group)-2, 3))
        rng.shuffle(units)
        rows = [row for unit in units for row in unit]
    else:
        rows = [row for group in groups for row in group]
        rng.shuffle(rows)
        # Sort within shuffled windows to reduce padding without changing samples.
        for start in range(0, len(rows), global_batch * 32):
            rows[start:start+global_batch*32] = sorted(rows[start:start+global_batch*32], key=lambda r: len(r['tokens']))
        batches = [rows[i:i+global_batch] for i in range(0, len(rows)-global_batch+1, global_batch)]
        rng.shuffle(batches)
        return batches
    return [rows[i:i+global_batch] for i in range(0, len(rows)-global_batch+1, global_batch)]
