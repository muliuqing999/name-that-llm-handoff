"""Run on the existing PPU: stream only the named project assets to stdout."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

code=Path('/root/web-pro-gpu-runs/managed-encoder-smoke-r5-20261029-a1')
prepared=Path('/workspace/name-that-llm/web-pro-checkpoints/encoders-r5-20261029-a1/prepared')
plan=json.loads((prepared/'plan.json').read_text())
files=[]
for name in ['encoder_finetune.py','encoder_finetune_protocol.py','parallel_features.py','training_data.py','cv_protocol.py']:
    files.append((code/name,'project/reference/encoders-r5/'+name,None))
for name in ['train.csv','test.csv','sample_submission.parquet']:
    files.append((Path('/workspace/name-that-llm/data')/name,'datasets/competition/'+name,None))
public=Path(plan['paths']['public_manifest']).parent
for name in ['manifest.json','public_curated_v1.parquet']:
    files.append((public/name,'datasets/public-curated-v1/'+name,None))
files.append((Path(plan['paths']['labels_meta']),'datasets/competition/labels-meta.json',None))
files.append((prepared/'plan.json','reports/migration/ppu-source-plan.json',None))
for expert,data in plan['experts'].items():
    model=data['model']
    for relative,expected in model['files_sha256'].items():
        assert '..' not in Path(relative).parts
        files.append((Path(model['path'])/relative,'models/'+expert+'/'+relative,expected))
manifest={'source':'existing PPU experiment inputs','recipe_changed':False,'files':[]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True) as tar:
    for source,relative,expected in files:
        hasher=hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(chunk)
        sha=hasher.hexdigest()
        if expected is not None and sha!=expected:raise RuntimeError('Source weight checksum mismatch: '+relative)
        entry={'source':str(source),'relative':relative,'sha256':sha,'bytes':source.stat().st_size}
        manifest['files'].append(entry)
        tar.add(source,arcname=relative,recursive=False)
        print('COPIED',relative,entry['bytes'],file=sys.stderr,flush=True)
    data=json.dumps(manifest,indent=2).encode()
    info=tarfile.TarInfo('reports/migration/assets.json');info.size=len(data);info.mode=0o600
    tar.addfile(info,io.BytesIO(data))
