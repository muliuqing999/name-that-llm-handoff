"""Build a private, bounded derivative of the pinned ozertuu v37 source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'ozertuu-reproduction-20261115'
RUNTIME=ROOT/'runtime'
OUT=ROOT/'account-copy'
OUT.mkdir(exist_ok=True)
worker=(SOURCE/'code-review/infer.py').read_text(encoding='utf-8')
start=worker.index('from sklearn.feature_extraction.text import TfidfVectorizer\ntry:')
end=worker.index('LP = {}',start)
worker=worker[:start]+'''from sklearn.feature_extraction.text import TfidfVectorizer
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
'''+worker[end:]
needle='tot = sum(LP.values())\n'
assert worker.count(needle)==1
worker=worker.replace(needle,needle+'''assert not failed, failed
member_output = os.environ['NTL_MEMBER_OUTPUT']
member_terms = {'term_'+k:np.asarray(v) for k,v in LP.items() if k not in ('tfidf','distort','svc4')}
assert member_terms and all(v.shape == (len(test),18) and np.isfinite(v).all() for v in member_terms.values())
np.savez(member_output+'.tmp.npz', ids=test.id.astype(str).values.astype('U'), labels=np.array(LABELS), **member_terms)
os.replace(member_output+'.tmp.npz',member_output)
''')
ast.parse(worker)
(RUNTIME/'infer_worker.py').write_text(worker,encoding='utf-8')
native=(SOURCE/'code-review/native_q8.py').read_text(encoding='utf-8')
needle='res, finite = model.predict(texts, batch_size=a.batch, deadline=a.deadline or None,'
assert native.count(needle)==1
native=native.replace(needle,"folds=[int(x) for x in os.environ.get('NTL_NATIVE_FOLDS','0,1,2').split(',')]\n    res, finite = model.predict(texts, folds=folds, batch_size=a.batch, deadline=a.deadline or None,")
# A deadline cannot interrupt a completed-fold checkpoint or return partial zero logits.
needle='''                for s in range(0, len(order), batch_size):
                    idx = order[s:s + batch_size]'''
assert native.count(needle)==1
native=native.replace(needle,'''                for s in range(0, len(order), batch_size):
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError('native deadline inside fold; completed-fold checkpoint retained')
                    idx = order[s:s + batch_size]''')
ast.parse(native)
(RUNTIME/'native_q8.py').write_text(native,encoding='utf-8')
for name in ['pipe.py','gauss.py','ntl660_readout.py','qwen_groups.py']:
    (RUNTIME/name).write_text((SOURCE/'code-review'/name).read_text(encoding='utf-8'),encoding='utf-8')

payload={p.name:p.read_text(encoding='utf-8') for p in sorted(RUNTIME.glob('*.py'))}
for s in payload.values():ast.parse(s)
code='''import json, os, pathlib, subprocess, sys
FILES = '''+repr(payload)+'''
for name,content in FILES.items():
    pathlib.Path(name).write_text(content,encoding='utf-8')
sys.path.insert(0,os.getcwd())
from budget import run_stage, atomic_json
import pandas as pd
import torch
gpu=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,available=torch.cuda.is_available(),device_count=torch.cuda.device_count())
assert gpu['cuda_version'] and gpu['available'] and gpu['device_count']==2, gpu
gpu['devices']=[torch.cuda.get_device_name(i) for i in range(2)]
for i in range(2):
    with torch.cuda.device(i):
        x=torch.ones((64,64),device=f'cuda:{i}')
        assert float((x@x)[0,0].item())==64.
        del x
atomic_json('/kaggle/working/gpu-validation.json',dict(ok=True,**gpu))
print('PYTORCH_GPU_VALIDATED',json.dumps(gpu),flush=True)
selfcheck=run_stage('self-check',[sys.executable,'self_check.py'],30,'/kaggle/working')
assert selfcheck['status']=='complete',selfcheck
data=next(pathlib.Path(p) for p in ['/kaggle/input/competitions/name-that-llm','/kaggle/input/name-that-llm'] if (pathlib.Path(p)/'test.csv').exists())
actual=pd.read_csv(data/'test.csv',dtype={'id':str})
benchmark=len(actual)<50
result=run_stage('pipeline',[sys.executable,'-u','run_optimized.py'],6900 if benchmark else 7.8*3600,'/kaggle/working',dict(os.environ,PYTHONUNBUFFERED='1'))
out=pathlib.Path('/kaggle/working/submission.parquet')
assert out.exists(),result
frame=pd.read_parquet(out)
assert frame.id.is_unique and frame.model.notna().all()
assert set(actual.id)<=set(frame.id)
frame=frame.set_index('id').loc[actual.id].reset_index()
assert frame.columns.tolist()==['id','model'] and frame.id.tolist()==actual.id.tolist()
temp=out.with_name('submission.final.tmp.parquet')
frame.to_parquet(temp,index=False)
os.replace(temp,out)
atomic_json('/kaggle/working/outer-validation.json',dict(valid=True,rows=len(frame),pipeline=result,public_timing_benchmark=benchmark))
print('OUTER_VALIDATED',len(frame),'rows',result,flush=True)
'''
ast.parse(code)
n=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},cells=[
 dict(cell_type='markdown',metadata={},source=[
 '# NTL: bounded runtime optimization of ozertuu Version 37\n',
 'Original: https://www.kaggle.com/code/ozertuu/0-689-dual-svc-residual-refinement-apex\n',
 'Reproduction and runtime optimization for liuqingmu096, 2026-11-16. Published pretrained assets and attribution are preserved; no new training is claimed.\n',
 'Changes: remove complementary stack, Gaussian training-set readout, GSQ and GLM likelihood stages; use full-data adapter plus folds 0/1 for each main family, and Native folds 0/1/2. Cache shared semantic groups.\n',
 'Write a real prefit baseline first, then atomic checkpoints after each successful model; independently bound all model subprocess trees. Overall inference target 7.5h and parent watchdog 7.8h.\n',
 'Public runs perform genuine inference on the sample plus 72 length-stratified timing texts. This is a throughput check, not an unbiased accuracy estimate. Official scores come only from completed competition submissions.\n']),
 dict(cell_type='code',metadata={},source=[code],outputs=[],execution_count=None)])
# Preserve every original contributor acknowledgement as reference, without repeating a result claim.
original=json.loads((SOURCE/'original.ipynb').read_text(encoding='utf-8'))
n['cells'].insert(1,dict(cell_type='markdown',metadata={},source=['## Original author and community credits (quoted source documentation)\n',''.join(original['cells'][0]['source'])]))
(OUT/'optimized.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
meta=json.loads((SOURCE/'account-copy/kernel-metadata.json').read_text())
meta.update(id='liuqingmu096/ntl-ozertuu-bounded-optimized-20261116',title='NTL ozertuu bounded optimized 20261116',code_file='optimized.ipynb',is_private=True)
# This account's successful Dhananjay/whitegg runs used this GPU-capable image.
# The original ozertuu image has CPU-only PyTorch, despite a GPU machine shape.
meta['docker_image']='gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
# Omit the large, explicitly disabled GLM likelihood assets and unused model backbones.
meta['dataset_sources']=[s for s in meta['dataset_sources'] if not s.startswith('jamesliebel/glm53')]
meta['kernel_sources']=[s for s in meta['kernel_sources'] if s!='estopadilla/ntl-glm-likelihood-train']
meta['model_sources']=[s for s in meta['model_sources'] if any(p in s for p in ('llama-3-1','/8b-base/','/14b-base/','qwen-3-embedding','qwen2.5/Transformers/14b/'))]
(OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
manifest=dict(account='liuqingmu096',kernel=meta['id'],source_version=37,
    files={k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},
    notebook_sha256=hashlib.sha256((OUT/'optimized.ipynb').read_bytes()).hexdigest(),
    original_code_changed=True,new_training=False,official_score=None)
(ROOT/'build-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(runtime_files=len(payload),kernel=meta['id'],model_sources=meta['model_sources']),indent=2))
