"""Fresh two-PPU LoRA training with optional public pretraining and blind evaluation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from ppu_benchmark import BASE, CHECKPOINT, DATA, Classifier, emit, initialize
from performance_preflight import calibrate
from training_data import split_groups, balanced_subset, stage_batches, content_key
from blind_inference import predict_from_features, POLICY, fit_policy
from model_features import feature_tensor


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_external(request, official):
    import pandas as pd
    path=Path(os.environ['NTL_PUBLIC_MANIFEST'])
    manifest=json.loads(path.read_text())
    if manifest['profile']!=request['data_profile'] or manifest['official_sha256']!=file_sha(DATA):
        raise ValueError('Public data profile or official dataset checksum changed')
    data_file=path.parent/manifest['file']
    if data_file.parent.resolve()!=path.parent.resolve() or file_sha(data_file)!=manifest['sha256']:
        raise ValueError('Public data path/checksum mismatch')
    data=pd.read_parquet(data_file)
    if len(data)!=manifest['rows'] or set(data.model)!=set(manifest['labels']):
        raise ValueError('Public data manifest mismatch')
    if data[['text','model']].isna().any().any() or data.content_sha256.duplicated().any():
        raise ValueError('Invalid or duplicate public rows')
    keys=set(content_key(row['text']) for row in official)
    if any(content_key(t) in keys for t in data.text):
        raise ValueError('Official response leaked into external data')
    records=data[['text','model','source_dataset']].to_dict('records')
    records=balanced_subset(records,manifest['labels'],request['pretrain_max_rows'],request['seed'])
    return records,manifest


def cached_tokenize(rows, tokenizer, cfg, key, output, rank):
    cache=output.parent/'token-cache'/f'{key}.json'
    if rank==0:
        cache.parent.mkdir(parents=True,exist_ok=True)
        valid=False
        if cache.exists():
            try:
                payload=json.loads(cache.read_text())
                valid=payload.get('key')==key and len(payload['tokens'])==len(rows)
            except (OSError,ValueError,KeyError):pass
        if not valid:
            tokens=[]
            for start in range(0,len(rows),256):
                tokenized=tokenizer([r['text'] for r in rows[start:start+256]],truncation=True,max_length=cfg['max_len'],add_special_tokens=False)['input_ids']
                tokens.extend(t+[cfg['eos_id']] for t in tokenized)
            temp=cache.with_suffix('.tmp')
            temp.write_text(json.dumps({'key':key,'tokens':tokens},separators=(',',':')))
            temp.replace(cache)
        emit('token_cache',cache_hit=valid,rows=len(rows))
    dist.barrier()
    payload=json.loads(cache.read_text())
    if payload['key']!=key or len(payload['tokens'])!=len(rows):raise ValueError('Invalid token cache')
    for row,tokens in zip(rows,payload['tokens']):row['tokens']=tokens


def evaluate(module, validation_rows, batch, labels, rank, world, local, inference_policy):
    module.eval()
    features=torch.zeros((len(validation_rows),module.plain.in_features),device=f'cuda:{local}',dtype=torch.float32)
    indexes=list(range(rank,len(validation_rows),world))
    with torch.no_grad():
        for start in range(0,len(indexes),9):
            ids_here=indexes[start:start+9]
            ids,mask,_=batch([validation_rows[i] for i in ids_here])
            features[ids_here]=feature_tensor(module,ids,mask)
    dist.all_reduce(features)
    prediction=predict_from_features([r['text'] for r in validation_rows],features.cpu().numpy(),
        *[p.detach().cpu().numpy() for p in (module.plain.weight,module.plain.bias,module.centered.weight,module.centered.bias)],policy=inference_policy)
    label_map={label:i for i,label in enumerate(labels)}
    truth=np.asarray([label_map[r['model']] for r in validation_rows])
    result={}
    for name,key in [('blind','probabilities'),('plain','plain_probabilities')]:
        predicted=prediction[key].argmax(1)
        confusion=np.bincount(truth*18+predicted,minlength=324).reshape(18,18)
        recall=np.diag(confusion)/confusion.sum(1).clip(min=1)
        result[name+'_balanced_accuracy']=float(recall.mean())
        result[name+'_per_class_recall']=dict(zip(labels,recall.tolist()))
        result[name+'_confusion_matrix']=confusion.tolist()
    result.update(blind_group_count=prediction['group_count'],blind_centered_rows=prediction['centered_rows'])
    return result,prediction


def run(request, output, rank, world, local, performance_only=False, acceptance_steps=0):
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
    from safetensors.torch import save_file,load_file
    started=time.monotonic()
    meta=json.loads((Path(CHECKPOINT)/'meta.json').read_text())
    cfg,labels=meta['cfg'],meta['labels']
    with open(DATA,encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if any(r['model'] not in labels or not r['text'].strip() for r in rows):raise ValueError('Invalid official data')
    training,validation,val_ids=split_groups(rows)
    validation_rows=sorted([r for g in validation for r in g],key=lambda r:r['id'])
    if set(r['model'] for r in validation_rows)!=set(labels):raise ValueError('Validation is missing classes')
    train_keys={content_key(r['text']) for g in training for r in g}
    if train_keys&{content_key(r['text']) for r in validation_rows}:raise ValueError('Official train/validation overlap')
    official_sha=file_sha(DATA)
    external,manifest=(load_external(request,rows) if request['task']=='qwen3_two_stage' else ([],None))
    if acceptance_steps:external=balanced_subset(external,labels,1800,request['seed']) if external else []
    if rank==0:
        (output/'split.json').write_text(json.dumps({'seed':20261028,'train_prompt_ids':sorted({r['prompt_id'] for g in training for r in g}),'validation_prompt_ids':val_ids},indent=2))
        if manifest:(output/'data-manifest.json').write_text(json.dumps(manifest,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    tokenizer.padding_side='right';tokenizer.pad_token_id=cfg['eos_id']
    all_rows=rows+external
    token_identity={'official':official_sha,'external':manifest['sha256'] if manifest else None,'selected_texts':hashlib.sha256(''.join(content_key(r['text']) for r in external).encode()).hexdigest(),'max_len':cfg['max_len'],'eos':cfg['eos_id'],
                    'tokenizer':[(p.name,file_sha(p)) for p in sorted(Path(BASE).glob('tokenizer*')) if p.is_file()]}
    data_sha=hashlib.sha256(json.dumps(token_identity,sort_keys=True).encode()).hexdigest()
    cached_tokenize(all_rows,tokenizer,cfg,data_sha,output,rank)
    torch.manual_seed(request['seed'])
    config=AutoConfig.from_pretrained(BASE,local_files_only=True)
    config.num_hidden_layers=cfg['keep'];config.use_cache=False
    if getattr(config,'layer_types',None) is not None:config.layer_types=config.layer_types[:cfg['keep']]
    backbone=AutoModel.from_pretrained(BASE,config=config,torch_dtype=torch.bfloat16,device_map={'':local},low_cpu_mem_usage=True,attn_implementation='flash_attention_2',local_files_only=True)
    backbone.norm=nn.Identity()
    backbone=get_peft_model(backbone,LoraConfig(r=cfg['r'],lora_alpha=cfg['alpha'],lora_dropout=cfg['ldrop'],bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    backbone.enable_input_require_grads()
    module=Classifier(backbone,cfg,None,local)
    label_map={label:i for i,label in enumerate(labels)}
    def batch(batch_rows):
        encoded=tokenizer.pad({'input_ids':[r['tokens'] for r in batch_rows]},padding=True,pad_to_multiple_of=8,return_tensors='pt')
        return encoded['input_ids'].to(local),encoded['attention_mask'].to(local),torch.tensor([label_map[r['model']] for r in batch_rows],device=f'cuda:{local}')
    probe_groups=training+([external[i:i+18] for i in range(0,len(external)-17,18)] if external else [])
    performance=calibrate(module,cfg,probe_groups,batch,BASE,data_sha,output,rank,world,local,request['max_minutes'])
    if performance_only:return
    model=DDP(module,device_ids=[local],broadcast_buffers=False)
    adapters=[p for p in module.backbone.parameters() if p.requires_grad]
    heads=[p for n,p in module.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    history=[];stage_history=[];best=-1.;best_epoch=0;global_step=0;has_refit=False
    train_flat=[r for group in training for r in group]
    inference_policy,policy_fit=fit_policy([r['text'] for r in train_flat],[r['prompt_id'] for r in train_flat])
    if rank==0:emit('blind_policy_fit',policy=inference_policy,selected=policy_fit['selected'])
    def save_checkpoint(name,stage,epoch,step):
        if rank!=0:return
        target=output/name;target.mkdir(exist_ok=True)
        module.backbone.save_pretrained(target/'adapter',safe_serialization=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in module.state_dict().items() if not k.startswith('backbone.')},str(target/'heads.safetensors'))
        (target/'meta.json').write_text(json.dumps({'cfg':cfg,'labels':labels,'request':request,'stage':stage,'epoch':epoch,'step':step,'initialization':'fresh base + random LoRA/heads','inference_policy':inference_policy,'data_sha256':data_sha,'acceptance_only':bool(acceptance_steps)},indent=2))
        for source in ['predict_offline.py','blind_inference.py','model_features.py','validate_submission.py']:
            shutil.copy2(Path(__file__).parent/source,target/source)
    stages=[]
    if external:stages.append(('public_pretrain',[external],request['pretrain_epochs'],request['pretrain_learning_rate'],False))
    stages.append(('official_finetune',training,request['epochs'],request['learning_rate'],True))
    torch.cuda.reset_peak_memory_stats()
    for stage,groups,epochs,rate,grouped in stages:
        stage_start=time.monotonic()
        if stage=='official_refit_all':
            loaded=set_peft_model_state_dict(module.backbone,load_file(str(output/'stage1/adapter/adapter_model.safetensors')))
            if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):raise ValueError('Incomplete stage-1 checkpoint')
            loaded=module.load_state_dict(load_file(str(output/'stage1/heads.safetensors')),strict=False)
            if loaded.unexpected_keys or any(not k.startswith('backbone.') for k in loaded.missing_keys):raise ValueError('Incomplete stage-1 heads')
            torch.manual_seed(request['seed']+100000)
        module.centered_loss_weight=1.0 if grouped else 0.0
        optimizer=torch.optim.AdamW([{'params':adapters,'lr':rate},{'params':heads,'lr':rate*10}],weight_decay=0.01,foreach=False)
        planned_steps=len(stage_batches(groups,request['seed'],grouped))*epochs
        if not planned_steps:raise ValueError('Empty training stage')
        warmup=max(1,math.ceil(planned_steps*.1))
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:(step+1)/warmup if step<warmup else max(0.,(planned_steps-step)/max(1,planned_steps-warmup)))
        stage_step=0
        emit('stage_start',stage=stage,rows=sum(map(len,groups)),epochs=epochs,planned_steps=planned_steps)
        for epoch in range(epochs):
            model.train()
            for batch_rows in stage_batches(groups,request['seed']+epoch,grouped):
                ids,mask,targets=batch(batch_rows[rank*9:(rank+1)*9])
                optimizer.zero_grad(set_to_none=True)
                loss=model(ids,mask,targets)
                if not torch.isfinite(loss).item():raise RuntimeError('Non-finite loss')
                loss.backward()
                norm=nn.utils.clip_grad_norm_(adapters+heads,1.0,foreach=False)
                if not torch.isfinite(norm).item():raise RuntimeError('Non-finite gradient')
                optimizer.step();schedule.step();global_step+=1;stage_step+=1
                if stage_step%100==0 or stage_step==1:
                    elapsed=time.monotonic()-stage_start
                    progress={'stage':stage,'epoch':epoch+1,'step':stage_step,'planned_stage_steps':planned_steps,'loss':loss.item(),'stage_seconds':elapsed,'rough_stage_remaining_seconds':elapsed/stage_step*(planned_steps-stage_step)}
                    if rank==0:
                        temp=output/'progress.tmp';temp.write_text(json.dumps(progress));temp.replace(output/'progress.json')
                        emit('training',**progress)
                if stage_step%500==0:
                    save_checkpoint('progress-checkpoint',stage,epoch+1,stage_step)
                    dist.barrier()
                if acceptance_steps and stage_step>=acceptance_steps:break
                if time.monotonic()-started>request['max_minutes']*60-180:
                    save_checkpoint('progress-checkpoint',stage,epoch+1,stage_step)
                    dist.barrier()
                    raise TimeoutError('Saved partial weights before time budget; not a completed candidate')
            if stage=='official_finetune':
                metrics,prediction=evaluate(module,validation_rows,batch,labels,rank,world,local,inference_policy)
                metrics['epoch']=epoch+1
                history.append(metrics)
                if metrics['blind_balanced_accuracy']>best:
                    best=metrics['blind_balanced_accuracy']
                    best_epoch=epoch+1
                    save_checkpoint('best',stage,epoch+1,stage_step)
                    if rank==0:
                        np.savez_compressed(output/'validation-predictions.npz',ids=np.asarray([r['id'] for r in validation_rows]),probabilities=prediction['probabilities'],plain_probabilities=prediction['plain_probabilities'],labels=np.asarray(labels))
                if rank==0:emit('validation',**metrics)
            if acceptance_steps:break
        if not grouped:save_checkpoint('stage1',stage,epoch+1,stage_step)
        if stage=='official_finetune' and request.get('refit_all') and external:
            all_groups={}
            for row in rows:all_groups.setdefault(row['prompt_id'],[]).append(row)
            stages.append(('official_refit_all',list(all_groups.values()),best_epoch,request['learning_rate'],True))
        if stage=='official_refit_all':
            save_checkpoint('final',stage,epoch+1,stage_step)
            has_refit=True
        stage_history.append({'stage':stage,'steps':stage_step,'epochs_completed':epoch+1,'seconds':time.monotonic()-stage_start,'rows':sum(map(len,groups))})
        del optimizer,schedule
        dist.barrier()
    if rank==0:
        checkpoint=output/('final' if has_refit else 'best')
        hashes={str(p.relative_to(checkpoint)):file_sha(p) for p in checkpoint.rglob('*') if p.is_file()}
        durable=None
        if os.environ.get('NTL_DURABLE_ROOT') and not acceptance_steps:
            durable=Path(os.environ['NTL_DURABLE_ROOT'])/output.name
            durable.mkdir(parents=True,exist_ok=False)
            shutil.copytree(checkpoint,durable/'final')
            if has_refit:shutil.copytree(output/'best',durable/'holdout-best')
            if manifest:shutil.copy2(output/'data-manifest.json',durable/'data-manifest.json')
            shutil.copy2(output/'split.json',durable/'split.json')
            for name,expected in hashes.items():
                if file_sha(durable/'final'/name)!=expected:raise RuntimeError('Durable checkpoint copy checksum mismatch')
        result={'metrics':{'best_balanced_accuracy':best,'best_epoch':best_epoch,'primary_metric':'blind_balanced_accuracy','history':history,'stages':stage_history,'steps':global_step,'training_seconds':time.monotonic()-started},
            'checkpoint':str(checkpoint),'selection_checkpoint':str(output/'best'),'checkpoint_sha256':hashes,'durable_checkpoint':str(durable/'final') if durable else None,
            'epochs_completed':len(history),'acceptance_only':bool(acceptance_steps),
            'versions':{'torch':torch.__version__},'hardware':{'world_size':world,'name':torch.cuda.get_device_name(local),'rank0_peak_GiB':torch.cuda.max_memory_allocated()/1024**3},
            'data':{'official_sha256':official_sha,'external_sha256':manifest['sha256'] if manifest else None,'external_rows_selected':len(external),'external_label_counts':dict(Counter(r['model'] for r in external)),'train_groups':len(training),'validation_groups':len(validation),'split_seed':20261028},
            'validation':{'prediction_inputs':['text'],'policy':inference_policy,'policy_fit':policy_fit,'primary_metric':'blind_balanced_accuracy','known_prompt_ids_used_for_prediction':False,'development_holdout':True,'not_comparable_with_prior_gold_group_scores':True,'refit_all_official_rows':has_refit,'score_applies_to':'holdout_selection_checkpoint; final refit has no independent validation score'},
            'performance':performance,'scope':'Acceptance test only; not a trained candidate.' if acceptance_steps else 'Fresh Qwen3 candidate; blind text-only holdout; not the full 0.622 ensemble and no leaderboard score implied.'}
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if durable:shutil.copy2(output/'result.json',durable/'result.json')
    dist.barrier()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--performance-only',action='store_true');p.add_argument('--acceptance-steps',type=int,default=0,help='Local bounded verification only; checkpoints are rejected by inference.')
    args=p.parse_args();rank,world,local=initialize()
    try:
        if world!=2:raise ValueError('Exactly two PPU ranks required')
        if not 0<=args.acceptance_steps<=5:raise ValueError('Acceptance steps must be 0-5')
        run(json.loads(Path(args.request).read_text(encoding='utf-8-sig')),Path(args.output_dir),rank,world,local,args.performance_only,args.acceptance_steps)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
