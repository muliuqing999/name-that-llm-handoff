"""Bounded inference reproduction of ozertuu v37; no new training claim."""
import glob
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from budget import atomic_json, run_stage
from ensemble import LABELS, checkpoint, combine, decide_scores, load_member, softmax

T0 = time.monotonic()
WORK=Path('/kaggle/working')
TEMP=Path('/kaggle/temp/ntl-optimized')
TEMP.mkdir(parents=True,exist_ok=True)
OUTPUT=WORK/'submission.parquet'
REPORT=WORK/'runtime-report.json'
PROGRESS=WORK/'progress'
PROGRESS.mkdir(exist_ok=True)
DATA=next(Path(p) for p in ['/kaggle/input/competitions/name-that-llm','/kaggle/input/name-that-llm'] if (Path(p)/'train.csv').exists())
train=pd.read_csv(DATA/'train.csv',dtype={'id':str})
actual=pd.read_csv(DATA/'test.csv',dtype={'id':str})
assert actual.id.is_unique and len(actual)
is_benchmark=len(actual)<50
# Public commit runs real inference on 72 length-stratified texts plus the actual sample.
# No accuracy is reported: the published heads may have seen official training rows.
if is_benchmark:
    ordered=train.assign(_length=train.text.fillna('').str.len()).sort_values('_length',kind='stable')
    pos=np.linspace(0,len(ordered)-1,72,dtype=int)
    bench=ordered.iloc[pos][['text']].copy().reset_index(drop=True)
    bench['id']=[f'timing_{i:04d}' for i in range(len(bench))]
    test=pd.concat([actual[['id','text']],bench[['id','text']]],ignore_index=True)
else:
    test=actual[['id','text']].copy()
test['text']=test.text.fillna('').astype(str)
data=TEMP/'data';data.mkdir(exist_ok=True)
train.to_csv(data/'train.csv',index=False)
test.to_csv(data/'test.csv',index=False)
ids=test.id.tolist()
prior=np.bincount(train.model.map({l:i for i,l in enumerate(LABELS)}),minlength=18)+1.
report=dict(account='liuqingmu096',source='ozertuu Version 37',is_timing_benchmark=is_benchmark,
    official_test_rows=len(actual),inference_rows=len(test),benchmark_accuracy_not_validated=True,
    removed=['complementary stack','Gaussian extra train inference','GSQ 27B','GLM likelihood reranker'],
    main_adapters_per_family=3,native_folds_requested=3,stages=[],completed_members=[],native_folds=[],
    elapsed_seconds=0.,mode='starting')
TOTAL=6600 if is_benchmark else 7.5*3600
# Deadline includes setup, CPU baseline, model loads, and every child process.
deadline=T0+TOTAL
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
         PYTHONUNBUFFERED='1',NTL_PIPE_WORKERS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def save_report():
    report['elapsed_seconds']=time.monotonic()-T0
    atomic_json(REPORT,report)


def first(pattern):
    hits=sorted(glob.glob('/kaggle/input/**/'+pattern,recursive=True))
    assert hits,pattern
    return hits[0]


def stage(name,cmd,maximum,minimum=10):
    left=deadline-time.monotonic()-120
    result=run_stage(name,cmd,min(maximum,max(0,left)),PROGRESS,env) if left>=minimum else dict(name=name,status='skipped_budget',seconds=0.)
    report['stages'].append(result);save_report()
    return result


# Resolve and extract only the original prefit archive; keep original path guards.
root=Path(first('ntl666-cache-parts.json')).parent
parts=json.loads((root/'ntl666-cache-parts.json').read_text())
assert parts['format']=='ntl666-cache-parts-v1'
assert [p['name'] for p in parts['parts']]==[f'ntl666-prefit.part{i:03d}.bin' for i in range(len(parts['parts']))]
archive=TEMP/'prefit.zip'
with archive.open('wb') as target:
    for part in parts['parts']:
        source=root/part['name'];assert source.stat().st_size==part['bytes']
        with source.open('rb') as stream:shutil.copyfileobj(stream,target,16*1024*1024)
assert archive.stat().st_size==parts['archive_bytes']
with zipfile.ZipFile(archive) as z:
    entries=z.infolist()
    assert len(entries)==parts['files'] and len({e.filename for e in entries})==len(entries)
    for e in entries:
        p=PurePosixPath(e.filename)
        assert not p.is_absolute() and '..' not in p.parts and chr(92) not in e.filename and p.parts[0]=='ntl666-prefit' and not e.is_dir()
    assert z.read('ntl666-prefit/COMPLETE')==b'ntl666-v14-data-numeric-v1\n'
    provenance=json.loads(z.read('ntl666-prefit/provenance.json'))
    assert provenance['status']=='COMPLETE' and provenance['pool_rows']==190588 and provenance['dataset_version']==13
    z.extractall(TEMP)
archive.unlink()
prefit=TEMP/'ntl666-prefit'
from ntl660_readout import core_votes
terms=core_votes(prefit,test.text.tolist())
assert all(x.shape==(len(test),18) and np.isfinite(x).all() for x in terms.values())
scores=combine(terms,prior)
checkpoint(OUTPUT,ids,scores)
report['mode']='prefit_baseline';report['setup_seconds']=time.monotonic()-T0
save_report()
print('BASELINE_CHECKPOINT_READY',len(ids),'rows',flush=True)

# These workers use full-precision published backbones, not GSQ/BNB quantization.
# Do not reinstall duplicate offline wheels from multiple mounted datasets.
configs=sorted(glob.glob('/kaggle/input/**/config.json',recursive=True))
def model(*parts):
    hits=[str(Path(c).parent) for c in configs if all(p in c.lower() for p in parts)]
    assert hits,parts
    return hits[0]
def adapter(name):return str(Path(first(name+'/heads.pt')).parent)
def meta(name):
    folder=Path(first(name+'/meta.json'));return str(folder.parent),json.loads(folder.read_text())['cfg']
B8=model('/qwen-3/','/8b-base/')
B14=model('/qwen-3/','/14b-base/')
BL=model('/llama-3-1-8b-base/')
B25=model('/qwen2.5/','/14b/')
NB=model('/qwen-3-embedding/','/8b/')
NATIVE=str(Path(first('ntl-native-q8-multidepth-fivefold-0679/manifest.json')).parent)
N14,N14C=meta('lora_qwen3_14b')
Q25,Q25C=meta('lora_qwen25_14b')
OQ4=str(Path(first('ntl-q14-lora-v4/heads.safetensors')).parent)
DEB=str(Path(first('deberta_v3_large/meta.json')).parent)
L=[2,4,6,8,12,16,20,24]
specs=[
 dict(name='deb',type='enc',ckpt=DEB,w=1.),
 dict(name='q8v5',type='l24',path=B8,adapters=[adapter('q8v5'),adapter('q8v5_f0'),adapter('q8v5_f1')],keep=24,layers=L,w=2.,bs=16,max_len=384,twin=.35,dp=True),
 dict(name='q14v5',type='l24',path=B14,adapters=[adapter('q14v5'),adapter('q14v5_f0'),adapter('q14v5_f1')],keep=27,layers=L,w=3.,bs=8,max_len=384,twin=.35),
 dict(name='n14b_clean',type='l24',path=B14,adapters=[OQ4,N14],adapter_weights=[.72,.28],keep=N14C['keep'],layers=N14C['layers'],w=2.5,bs=8,max_len=N14C['max_len'],twin=.35,no_final_norm=True,eos_id=N14C['eos_id']),
 dict(name='ll8v5',type='l24',path=BL,adapters=[adapter('ll8v5'),adapter('ll8v5_f0'),adapter('ll8v5_f1')],keep=24,layers=L,w=1.,bs=16,max_len=384,twin=.35,dp=True),
 dict(name='q25_14',type='l24',path=B25,adapters=[Q25],keep=Q25C['keep'],layers=Q25C['layers'],w=1.,bs=8,max_len=Q25C['max_len'],twin=.35,no_final_norm=True,eos_id=Q25C['eos_id'])]
report['specs']=specs;save_report()

# Cache semantic groups once; all members and native calibration use this same grouping.
group_file=TEMP/'groups.npy'
embed=TEMP/'group_emb.npy'
g=stage('semantic-groups',[sys.executable,'qwen_groups.py','--base',NB,'--assets',NATIVE,'--query',str(data/'test.csv'),'--out',str(embed),'--work',str(TEMP/'groups')],420 if is_benchmark else 1200)
if g['status']=='complete' and embed.exists():
    env['NTL_GROUP_EMBED']=str(embed)
for spec in specs:
    name=spec['name'];member=TEMP/(name+'.npz')
    command=[sys.executable,'infer_worker.py','--data',str(data),'--models',json.dumps([spec]),
        '--tfidf_w','0','--distort_w','0','--svc_w','0','--adapt_w','1','--adapt_thr','.15',
        '--sinkhorn','.30','--sinkhorn_target_mix','.35','--out',str(TEMP/(name+'.parquet'))]
    env['NTL_MEMBER_OUTPUT']=str(member);env['NTL_GROUP_OUTPUT']=str(group_file)
    limit=(600 if name not in ('q14v5','n14b_clean') else 780) if is_benchmark else (900 if name=='deb' else 3900)
    s=stage(name,command,limit)
    if s['status']=='complete' and member.exists():
        try:
            values=load_member(member,ids)
            assert name in values
            terms.update(values)
            scores=combine(terms,prior)
            checkpoint(OUTPUT,ids,scores)
            report['completed_members'].append(name);report['mode']='main_ensemble';save_report()
            print('ENSEMBLE_CHECKPOINT',name,len(ids),'rows',flush=True)
        except Exception as e:
            s['validation_error']=repr(e);save_report()

# Native saves every complete fold atomically. A killed later fold cannot erase prior folds.
native=TEMP/'native.npz'
env['NTL_NATIVE_FOLDS']='0,1,2'
stage('native',[sys.executable,'native_q8.py','--base',NB,'--assets',NATIVE,'--query',str(data/'test.csv'),
    '--out',str(native),'--work',str(TEMP/'native'),'--deadline',str(time.time()+max(0,deadline-time.monotonic()-180))],900 if is_benchmark else 6000)
if native.exists() and report['completed_members'] and group_file.exists():
    try:
        import native_q8
        with np.load(native,allow_pickle=False) as z:
            assert z['ids'].astype(str).tolist()==ids
            labels=z['labels'].astype(str).tolist();order=[labels.index(l) for l in LABELS]
            folds=z['folds'].tolist();assert folds and len(folds)<=3
            states=[json.loads((Path(NATIVE)/'calibration'/f'fold{k}.json').read_text())['models']['qwen8'] for k in folds]
            groups=np.load(group_file,allow_pickle=False);assert groups.shape==(len(ids),)
            lp=native_q8.calibrate(z['plain'][:,:,order],z['centered'][:,:,order],z['centered_bias'][:,order],groups,states,True)
            agrees=[float((x.argmax(1)==scores.argmax(1)).mean()) for x in lp]
            assert min(agrees)>=.35,('native agreement guard',agrees)
            share=(.40*z['finite'])[:,None]
            active=sum(float(abs(v).max())>0 for v in terms.values())
            base=softmax(sum(terms.values())/max(1,active))
            mixed=sum(softmax(decide_scores(softmax((1-share)*np.log(np.maximum(base,1e-15))+share*x),prior)) for x in lp)/len(lp)
            scores=np.log(np.maximum(mixed,1e-15))
            checkpoint(OUTPUT,ids,scores)
            report['native_folds']=folds;report['native_agreement']=agrees;report['mode']='main_plus_native'
    except Exception as e:report['native_validation_error']=repr(e)

if is_benchmark:
    full=pd.read_parquet(OUTPUT)
    full.to_parquet(WORK/'timing-predictions.parquet',index=False)
    checkpoint(OUTPUT,actual.id.tolist(),scores[:len(actual)])
final=pd.read_parquet(OUTPUT)
assert final.columns.tolist()==['id','model'] and final.id.tolist()==actual.id.tolist()
assert final.id.is_unique and final.model.isin(LABELS).all()
report['final_rows']=len(final);report['final_valid']=True
report['estimated_4000_inference_seconds']={s['name']:s['seconds']*4000/len(test) for s in report['stages'] if s['status']=='complete' and s['name'] not in ('offline-wheel',)}
save_report()
print('FINAL_VALIDATED',json.dumps(report),flush=True)
