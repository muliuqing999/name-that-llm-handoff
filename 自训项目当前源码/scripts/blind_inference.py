"""Test-compatible prediction. Never accepts prompt ids or target labels."""
import hashlib
import numpy as np

POLICY = {'name': 'blind_text_v1', 'cosine_threshold': 0.35, 'max_group_size': 32,
          'min_group_size': 3, 'centered_weight': 0.5}


def fit_policy(training_texts, training_prompt_ids):
    """Choose text-similarity threshold on official TRAIN groups only.

    Held-out texts, held-out prompt ids and model target labels are not accepted.
    The fitted policy is saved and reused unchanged on validation and test.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(training_texts)!=len(training_prompt_ids):raise ValueError('Training group length mismatch')
    x=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True,min_df=1,max_features=100000,dtype=np.float32).fit_transform(training_texts)
    a,b=np.triu_indices(len(training_texts),1)
    similarity=(x@x.T).toarray()[a,b]
    ids=np.asarray(training_prompt_ids)
    same=ids[a]==ids[b]
    records=[]
    for threshold in (0.03,0.05,0.07,0.10,0.12,0.15,0.18,0.22,0.28,0.35):
        proposed=similarity>=threshold
        tp=int((proposed&same).sum());precision=tp/max(1,int(proposed.sum()));recall=tp/max(1,int(same.sum()))
        records.append({'threshold':threshold,'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-12,precision+recall)})
    eligible=[r for r in records if r['precision']>=0.90]
    best=max(eligible,key=lambda r:(r['f1'],r['threshold'])) if eligible else max(records,key=lambda r:r['f1'])
    policy={**POLICY,'cosine_threshold':best['threshold'],
            'centered_weight':0.5 if best['precision']>=0.90 and best['recall']>=0.10 else 0.0}
    return policy,{'fit_rows':len(training_texts),'fit_prompt_groups':len(set(training_prompt_ids)),
                   'selection':'max pair F1 among thresholds with train pair precision >= 0.90; disable centering if recall < 0.10',
                   'selected':best,'candidates':records,'uses_heldout_data':False}


def blind_groups(texts, policy=None):
    """Deterministic complete-link greedy groups from text alone.

    No gold groups, assumed number of prompts, or per-class counts are used.
    The quadratic similarity matrix is bounded to the competition-sized batch.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    policy = policy or POLICY
    if len(texts) > 12000:
        raise ValueError('Inference batch exceeds the checked memory bound')
    if len(texts) < policy['min_group_size']:
        return [[i] for i in range(len(texts))]
    try:
        x = TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True,
                            min_df=1, max_features=100000, dtype=np.float32).fit_transform(texts)
    except ValueError:
        return [[i] for i in range(len(texts))]
    similarity = (x @ x.T).toarray()
    order = sorted(range(len(texts)), key=lambda i: hashlib.sha256(texts[i].encode()).digest())
    priority = {index: rank for rank,index in enumerate(order)}
    unused = set(order)
    groups = []
    for anchor in order:
        if anchor not in unused:
            continue
        group = [anchor]
        unused.remove(anchor)
        candidates = sorted((i for i in unused if similarity[anchor,i] >= policy['cosine_threshold']),
                            key=lambda i: (-float(similarity[anchor,i]), priority[i]))
        for candidate in candidates:
            if len(group) >= policy['max_group_size']:
                break
            if all(similarity[candidate, member] >= policy['cosine_threshold'] for member in group):
                group.append(candidate)
                unused.remove(candidate)
        groups.append(group)
    return groups


def softmax(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - values.max(axis=1, keepdims=True)
    ex = np.exp(values)
    return ex / ex.sum(axis=1, keepdims=True)


def predict_from_features(texts, features, plain_weight, plain_bias, centered_weight, centered_bias, policy=None):
    """Same function is called by holdout evaluation and offline inference."""
    policy = policy or POLICY
    if len(texts) != len(features):
        raise ValueError('Text/features length mismatch')
    plain = softmax(features @ plain_weight.T + plain_bias)
    result = plain.copy()
    groups = blind_groups(texts, policy)
    adjusted = 0
    for group in groups:
        if len(group) < policy['min_group_size']:
            continue
        f = features[group]
        centered = softmax((f - f.mean(axis=0, keepdims=True)) @ centered_weight.T + centered_bias)
        weight = policy['centered_weight']
        result[group] = (1-weight)*plain[group] + weight*centered
        adjusted += len(group)
    return {'probabilities': result, 'plain_probabilities': plain,
            'group_count':len(groups), 'centered_rows':adjusted, 'policy':dict(policy)}
