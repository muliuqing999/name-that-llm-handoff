"""Version-specific operations for the authorized liuqingmu096 optimization."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SLUG='liuqingmu096/ntl-ozertuu-bounded-optimized-20261116'
os.environ['KAGGLE_API_TOKEN']=str(Path.home()/'.kaggle/access_token.txt')
os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'api.kaggle.com','www.kaggle.com','kaggle.com']))
os.environ.pop('KAGGLE_USERNAME',None);os.environ.pop('KAGGLE_KEY',None)
from kaggle import api
api.authenticate()
assert api.get_config_value('username')=='liuqingmu096'
p=argparse.ArgumentParser()
p.add_argument('action',choices=['push','status','verify','logs','inspect','output','submit','detail'])
p.add_argument('--version',type=int,default=1)
p.add_argument('--ref',type=int)
a=p.parse_args();version=a.version
E=ROOT/f'evidence-v{version}';E.mkdir(exist_ok=True)
if a.action=='push':
    receipt=E/'push-receipt.json'
    with receipt.open('x') as f:json.dump(dict(state='started',account='liuqingmu096'),f)
    meta=json.loads((ROOT/'account-copy/kernel-metadata.json').read_text())
    assert meta['id']==SLUG and meta['is_private'] is True
    result=api.kernels_push(str(ROOT/'account-copy'),timeout='28800',acc='NvidiaTeslaT4')
    data=json.loads(result.to_json());assert data['versionNumber']==version,data
    receipt.write_text(json.dumps(data,indent=2))
elif a.action=='status':
    data=json.loads(api.kernels_status(SLUG+f'/{version}').to_json())
    (E/'status.json').write_text(json.dumps(data,indent=2))
elif a.action=='verify':
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as c:
        req=ApiGetKernelRequest();req.user_name='liuqingmu096';req.kernel_slug=SLUG.split('/')[1];req.version_label=f'v{version}'
        result=c.kernels.kernels_api_client.get_kernel(req)
    remote=json.loads(result.blob.source)
    local=json.loads((ROOT/'account-copy/optimized.ipynb').read_text(encoding='utf-8'))
    cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
    assert cells(remote)==cells(local) and result.metadata.is_private and result.metadata.ref==SLUG
    data=dict(ok=True,account='liuqingmu096',version=version,source_sha256=hashlib.sha256(result.blob.source.encode()).hexdigest(),machine_shape=result.metadata.machine_shape)
    (E/'saved-validation.json').write_text(json.dumps(data,indent=2))
elif a.action=='logs':
    import re,time,requests
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
    send=requests.sessions.Session.send
    def bounded(self,request,**kw):
        if kw.get('timeout') is None:kw['timeout']=(15,12)
        return send(self,request,**kw)
    requests.sessions.Session.send=bounded
    lines=[];started=time.monotonic();response=None;error=None
    try:
        with api.build_kaggle_client() as c:
            req=ApiGetKernelSessionLogsStreamRequest();req.user_name='liuqingmu096';req.kernel_slug=SLUG.split('/')[1];req.version_label=f'v{version}';req.wait_for_logs_url_seconds=5
            response=c.kernels.kernels_api_client.get_kernel_session_logs_stream(req);response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1024):
                line=raw.decode('utf-8',errors='replace') if isinstance(raw,bytes) else str(raw)
                lines.append(re.sub(r'(https?://[^\s\"<>?]+)\?[^\s\"<>]*',r'\1?[REDACTED]',line))
                if time.monotonic()-started>22 or len(lines)>20000:break
    except Exception as e:error=type(e).__name__
    finally:
        if response is not None:response.close()
    (E/'live.log').write_text('\n'.join(lines),encoding='utf-8')
    data=dict(lines=len(lines),tail=lines[-14:],bounded_read_error=error)
elif a.action=='inspect':
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as c:
        req=ApiListKernelSessionOutputRequest();req.user_name='liuqingmu096';req.kernel_slug=SLUG.split('/')[1];req.version_label=f'v{version}'
        result=c.kernels.kernels_api_client.list_kernel_session_output(req)
    data=dict(files=[f.file_name for f in result.files or []])
    if result.log and not result.log.startswith(('http://','https://')):
        events=json.loads(result.log);(E/'run-log.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        data['last_events']=events[-5:]
elif a.action=='output':
    (E/'output').mkdir(exist_ok=True)
    api.kernels_output(SLUG+f'/{version}',path=str(E/'output'),file_pattern=r'(^|/)(submission\.parquet|runtime-report\.json|outer-validation\.json|gpu-validation\.json|self-check\.log|.*\.status\.json|.*\.log)$')
    data=dict(downloaded=True)
elif a.action=='submit':
    assert json.loads(api.kernels_status(SLUG+f'/{version}').to_json())['status']=='COMPLETE'
    assert json.loads((E/'saved-validation.json').read_text())['ok']
    proof=json.loads((E/'submission-validation.json').read_text());assert proof['ok'] and proof['version']==version
    receipt=E/'submission-receipt.json'
    with receipt.open('x') as f:json.dump(dict(state='started',at=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
    result=api.competition_submit_code(file_name='submission.parquet',message=f'ozertuu v37 optimized bounded pipeline v{version}; real 75-row timing check; atomic fallback; liuqingmu096; 20261116',competition='name-that-llm',kernel=SLUG,kernel_version=version,quiet=True)
    data=json.loads(result.to_json());receipt.write_text(json.dumps(data,indent=2))
else:
    assert a.ref
    result=api.get_submission(a.ref)
    data=dict(ref=result.ref,status=result.status.name,error=result.error_description,score=result.public_score,submitted_by=result.submitted_by,team=result.team_name)
    (E/'submission-status.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
