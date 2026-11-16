"""Build a versioned candidate pool from downloaded public files; never calls APIs.

This reduces known risks, not a certification of source model identities.
No official passage text is written to the output pool or its public manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from training_data import text_key, content_key

AGGREGATORS = {
    'pukkei37/ntl-external-clean','pukkei37/ntl-pool-v2','ayhampar/ntl-pool-ft',
    'sunny114514/ntl-mega-corpus','dhruvtibarewal/nlm-matched-clean',
    'michaelsemenoff/ntl-pool-v3','philipvallen/ntl-pools','nuridudak/ntl-130k-clean-passages',
    'simonrba/ntl-pool-extra','asiralabi/ntl-src','alperen5252525/ntl-public-passage-pool-v1',
    'mrralphie/ntl-pool-b2m','dominicjamil/ntl-pools','swaroopmeher/ntl-public-data-sub1020',
}
REPLACE = {
    'willrice/name-that-llm-openrouter-harvest',
    'prvsiyan/prvsiyans-extended-name-that-llm-dataset',
    'elliotthankinson/ntl-exact-gens-v3',
    'lucasboesen/ntl-free-proxy-generations','mathleteutd/ntl-12-model-proxy-passages',
    'davidpiln/ntl-vertex-gemini-gen','aswathvanan/ntl-proxy-passages','abdullah1afify/ntl-generations',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def ngrams(text, n=8):
    words=re.findall(r'\w+',str(text).casefold())
    return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}


def build(audit, official_path, output):
    audit,output=Path(audit),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    official=pd.read_csv(official_path)
    labels=sorted(official.model.unique())
    if len(labels)!=18: raise ValueError('Expected 18 official labels')
    path=audit/'downloads/open-pool-v9/ntl_open_pool.parquet'
    pool=pd.read_parquet(path)
    steps=[]
    inputs=[{'ref':'crtproduction/name-that-llm-open-pool','version':9,'file':path.name,'sha256':sha(path)}]
    def filter_rows(frame, keep, reason):
        steps.append({'filter':reason,'before':len(frame),'removed':int((~keep).sum())})
        return frame.loc[keep].copy()
    flagged=pool[['flag_gemini37_routing_risk','flag_train_prompt_regen','flag_near_train_prompt']].fillna(False).any(axis=1)
    pool=filter_rows(pool,~flagged,'author_risk_flags')
    pool=filter_rows(pool,~pool.source_dataset.isin(AGGREGATORS|REPLACE),'exclude_secondary_pools_or_replace_with_recommended_files')
    pool=filter_rows(pool,~pool.source_file.str.contains(r'\.jsonl|raw_all_templates|pilot|trial|smoke',case=False,regex=True),'exclude_raw_logs_and_setting_trials')
    pool=pool[['text','model','prompt_group','source_dataset','source_file','source_license']]
    pool['identity_status']='source_label_not_independently_verified'
    pool['requested_model']='';pool['served_model']='';pool['setting']=''
    parts=[pool]
    manifests=json.loads((audit/'extras-manifest.json').read_text(encoding='utf-8'))
    meta={x['ref']:x for x in json.loads((audit/'catalog.json').read_text(encoding='utf-8'))['datasets']}
    for item in manifests:
        ref=item['ref']
        if ref not in REPLACE: continue
        if 'error' in item:raise ValueError('Missing required source '+ref)
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('Source checksum mismatch '+ref)
        d=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
        if 'model' not in d:d['model']=d['label']
        if 'discriminator_pass' in d:
            d=filter_rows(d,d.discriminator_pass.eq(True),'elliot_author_gate')
        if ref.startswith('prvsiyan/'):
            d=filter_rows(d,d['split'].isin(['new','proxy_v1']) & d.passes_funnel.eq(True),'prvsiyan_new_or_proxy_clean_only')
        if ref.startswith('davidpiln/'):
            d=filter_rows(d,d.model_version.eq(d.model) & d.finish_reason.eq('STOP'),'vertex_returned_version_and_finish')
        d['source_dataset']=ref;d['source_file']=item['file']
        d['source_license']=meta[ref].get('licenseName') or 'See pinned source metadata'
        d['prompt_group']=ref+':'+d.get('arm',pd.Series('',index=d.index)).fillna('').astype(str)+':'+d.get('prompt_id',pd.Series(d.index,index=d.index)).astype(str)
        d['requested_model']=d.get('requested_model',d.get('requested_model_id',d.get('model_id',d.get('model_slug',''))))
        d['served_model']=d.get('served_model',d.get('served_model_id',d.get('model_version',d.get('model_seen',''))))
        d['setting']=d.get('setting',d.get('reasoning_setting',d.get('reasoning','')))
        d['identity_status']='source_metadata_retained_not_independently_verified'
        parts.append(d[pool.columns])
        inputs.append({k:item[k] for k in ['ref','version','file','sha256']})
    data=pd.concat(parts,ignore_index=True).fillna('')
    data=filter_rows(data,data.model.isin(labels)&data.text.map(lambda x:isinstance(x,str) and bool(x.strip())),'valid_text_and_official_labels')
    # Preserve punctuation: it is a genuine feature for author attribution.
    original=data.text.copy()
    word_counts=data.text.str.split().str.len()
    shape=word_counts.between(30,300)&~original.str.contains(r'\n\s*\n|```|(?m:^\s*#{1,6}\s)',regex=True)
    data=filter_rows(data,shape,'single_paragraph_30_300_words_no_code_or_headings')
    data['text']=data.text.str.replace(r'\s+',' ',regex=True).str.strip()
    data['text_sha256']=data.text.map(text_key)
    data['content_sha256']=data.text.map(content_key)
    conflicts=data.groupby('content_sha256').model.nunique().loc[lambda s:s>1].index
    data=filter_rows(data,~data.content_sha256.isin(conflicts),'conflicting_labels_after_content_normalization')
    data=filter_rows(data,~data.content_sha256.duplicated(),'content_duplicates_ignoring_case_and_punctuation')
    official_keys=set(official.text.map(content_key))
    data=filter_rows(data,~data.content_sha256.isin(official_keys),'exact_official_content_overlap')
    official_grams=set().union(*(ngrams(t) for t in official.text))
    data=filter_rows(data,data.text.map(lambda t:len(ngrams(t)&official_grams)<2),'two_or_more_shared_official_word_8grams')
    # All official passages are used only as an exclusion list, without labels.
    # No evaluation target is used to assign an external label or tune thresholds.
    vector=TfidfVectorizer(stop_words='english',sublinear_tf=True,min_df=2,max_features=100000,dtype=np.float32)
    ref=vector.fit_transform(official.text)
    maxima=[]
    for start in range(0,len(data),1024):
        scores=vector.transform(data.text.iloc[start:start+1024])@ref.T
        maxima.extend(scores.max(axis=1).toarray().ravel().tolist())
    data['max_official_text_cosine']=maxima
    data=filter_rows(data,data.max_official_text_cosine.lt(0.30),'official_text_cosine_ge_0_30')
    data=data.sort_values('text_sha256').reset_index(drop=True)
    assert set(data.model)==set(labels)
    target=output/'public_curated_v1.parquet'
    data.to_parquet(target,index=False)
    report={'schema_version':1,'profile':'curated_v1','file':target.name,'sha256':sha(target),
        'rows':len(data),'labels':labels,'label_counts':{k:int(v) for k,v in data.model.value_counts().sort_index().items()},
        'source_counts':{k:int(v) for k,v in data.source_dataset.value_counts().items()},
        'official_sha256':sha(official_path),'filters':steps,'inputs':inputs,
        'scope':'Candidate public pool with known risk exclusions; no independent model-identity certification.',
        'validation_caveat':'Some upstream publishers filtered using official training distribution. Fixed holdout is developmental, not an untouched test.',
        'training_protocol':'Stage 1 uses individual-row plain-head loss: source-local prompt ids do not control centering. Stage 2 uses official training prompt triplets. Validation predictions consume text only.',
        'license':'Retain each upstream attribution and applicable terms; distribute combined derivative under CC BY-SA 4.0 where compatible.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(data),'sha256':report['sha256'],'label_counts':report['label_counts'],'output':str(output)}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-dir',required=True);p.add_argument('--official',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();build(a.audit_dir,a.official,a.output_dir)
