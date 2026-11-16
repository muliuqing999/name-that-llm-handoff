"""Require genuine multi-model inference and verified timeout protection before submission."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);a=p.parse_args()
E=ROOT/f'evidence-v{a.version}'
OUT=E/'output'
assert json.loads((E/'status.json').read_text())['status']=='COMPLETE'
assert json.loads((E/'saved-validation.json').read_text())['ok']
report=json.loads((OUT/'runtime-report.json').read_text())
outer=json.loads((OUT/'outer-validation.json').read_text())
assert outer['valid'] and outer['pipeline']['status']=='complete'
gpu=json.loads((OUT/'gpu-validation.json').read_text())
assert gpu['ok'] and gpu['cuda_version'] and gpu['available'] and gpu['device_count']==2
assert report['is_timing_benchmark'] and report['inference_rows']==75
assert report['final_valid'] and report['final_rows']==3
assert set(report['completed_members'])=={'deb','q8v5','q14v5','n14b_clean','ll8v5','q25_14'},report['completed_members']
assert report['native_folds']==[0,1,2],report.get('native_validation_error')
assert all(x['status']=='complete' and not x.get('validation_error') for x in report['stages']),report['stages']
assert 'POSIX_PROCESS_TREE_TIMEOUT_AND_CHECKPOINT_PASSED' in (OUT/'self-check.log').read_text()
frame=pd.read_parquet(OUT/'submission.parquet')
assert frame.columns.tolist()==['id','model']
assert frame.id.tolist()==['00353fb1c541','005f5a9a6139','0062a53d5487'] and frame.model.notna().all()
proof=dict(ok=True,account='liuqingmu096',version=a.version,real_inference_rows=75,
    main_members=report['completed_members'],native_folds=report['native_folds'],
    posix_process_tree_timeout_test=True,public_runtime_seconds=report['elapsed_seconds'],
    estimated_4000_inference_seconds=report['estimated_4000_inference_seconds'],
    accuracy_not_estimated_from_timing_fixture=True,official_score=None,
    submission_sha256=hashlib.sha256((OUT/'submission.parquet').read_bytes()).hexdigest())
(E/'submission-validation.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,ensure_ascii=False,indent=2))
