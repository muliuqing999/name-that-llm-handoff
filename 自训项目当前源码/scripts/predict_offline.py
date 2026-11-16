"""Offline Kaggle-compatible inference from a saved candidate checkpoint.

Input is id/text only. No prompt_id, true label, network call or GPU vendor SDK.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from blind_inference import predict_from_features, POLICY


def load_model(base, checkpoint, device='auto', allow_acceptance=False):
    import torch
    from torch import nn
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    meta=json.loads((Path(checkpoint)/'meta.json').read_text())
    if meta.get('acceptance_only') and not allow_acceptance:
        raise ValueError('Acceptance-test weights must not be submitted')
    if meta.get('stage') not in {'official_finetune','official_refit_all'}:
        raise ValueError('Use a completed official fine-tune or refit checkpoint')
    cfg=meta['cfg']
    config=AutoConfig.from_pretrained(base,local_files_only=True)
    config.num_hidden_layers=cfg['keep'];config.use_cache=False
    if getattr(config,'layer_types',None) is not None:config.layer_types=config.layer_types[:cfg['keep']]
    cuda=torch.cuda.is_available() and device!='cpu'
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    placement={}
    if device=='auto' and cuda:
        placement={'device_map':'auto','max_memory':{i:int(torch.cuda.get_device_properties(i).total_memory*.78) for i in range(torch.cuda.device_count())}}
    else:
        placement={'device_map':{'':('cpu' if device=='auto' else device)}}
    backbone=AutoModel.from_pretrained(base,config=config,torch_dtype=dtype,
        local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa',**placement)
    backbone.norm=nn.Identity()
    backbone=PeftModel.from_pretrained(backbone,str(Path(checkpoint)/'adapter'),local_files_only=True)
    model=nn.Module()
    model.backbone=backbone;model.layers=cfg['layers']
    width=backbone.config.hidden_size
    model.norms=nn.ModuleList([nn.LayerNorm(width) for _ in model.layers])
    model.plain=nn.Linear(width*len(model.layers),18)
    model.centered=nn.Linear(width*len(model.layers),18)
    result=model.load_state_dict(load_file(str(Path(checkpoint)/'heads.safetensors')),strict=False)
    if result.unexpected_keys or any(not k.startswith('backbone.') for k in result.missing_keys):
        raise ValueError('Incomplete inference head checkpoint')
    input_device=backbone.get_input_embeddings().weight.device
    for head in (model.norms,model.plain,model.centered):head.to(device=input_device,dtype=torch.float32)
    model.eval()
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    tokenizer.padding_side='right';tokenizer.pad_token_id=cfg['eos_id']
    return model,tokenizer,meta


def predict(base, checkpoint, frame, batch_size=2, device='auto'):
    import torch
    from model_features import feature_tensor
    if not {'id','text'}.issubset(frame.columns) or frame[['id','text']].isna().any().any() or frame.id.duplicated().any():
        raise ValueError('Expected unique nonempty id/text rows')
    # Drop every other field before entering any prediction function.
    texts=frame['text'].astype(str).tolist()
    if not texts or any(not t.strip() for t in texts):raise ValueError('Empty inference input')
    model,tokenizer,meta=load_model(base,checkpoint,device)
    input_device=model.backbone.get_input_embeddings().weight.device
    cfg=meta['cfg']
    tokens=[tokenizer(t,truncation=True,max_length=cfg['max_len'],add_special_tokens=False)['input_ids']+[cfg['eos_id']] for t in texts]
    order=sorted(range(len(tokens)),key=lambda i:len(tokens[i]))
    features=np.empty((len(texts),model.plain.in_features),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(order),batch_size):
            indexes=order[start:start+batch_size]
            encoded=tokenizer.pad({'input_ids':[tokens[i] for i in indexes]},padding=True,pad_to_multiple_of=8,return_tensors='pt')
            features[indexes]=feature_tensor(model,encoded['input_ids'].to(input_device),encoded['attention_mask'].to(input_device)).cpu().numpy()
    arrays=[p.detach().cpu().numpy() for p in [model.plain.weight,model.plain.bias,model.centered.weight,model.centered.bias]]
    prediction=predict_from_features(texts,features,*arrays,policy=meta.get('inference_policy',POLICY))
    return np.asarray(meta['labels'])[prediction['probabilities'].argmax(1)], prediction


if __name__=='__main__':
    import pandas as pd
    p=argparse.ArgumentParser();p.add_argument('--base-model',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--test-csv',required=True);p.add_argument('--output',default='submission.parquet');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--device',default='auto')
    a=p.parse_args();frame=pd.read_csv(a.test_csv,dtype={'id':str})
    labels,diagnostics=predict(a.base_model,a.checkpoint,frame,a.batch_size,a.device)
    from validate_submission import validate_submission
    if Path(a.output).suffix!='.parquet':raise ValueError('Submission output must be Parquet')
    submission=pd.DataFrame({'id':frame.id,'model':labels})
    validate_submission(submission,frame.id,json.loads((Path(a.checkpoint)/'meta.json').read_text())['labels'])
    submission.to_parquet(a.output,index=False)
    print(json.dumps({'rows':len(frame),'output':a.output,'blind_groups':diagnostics['group_count'],'centered_rows':diagnostics['centered_rows']}))
