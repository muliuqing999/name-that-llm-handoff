"""Identical feature extraction for training evaluation and offline prediction."""
import torch


def feature_tensor(module, ids, mask):
    states=module.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
    pooled=[]
    for layer,norm in zip(module.layers,module.norms):
        device=norm.weight.device
        weights=mask.to(device).unsqueeze(-1).float().clone()
        weights[:,0]=0
        denominator=weights.sum(1).clamp(min=1)
        pooled.append(norm((states[layer].to(device).float()*weights).sum(1)/denominator))
    return torch.cat(pooled,dim=-1)
