"""One group split before any crop extraction. Explicit canonical labels only."""
import argparse
import hashlib
import json
from pathlib import Path

def stable_split(group,seed=42):
    value=int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8],16)/0xffffffff
    return 'train' if value<.7 else 'validation' if value<.8 else 'calibration' if value<.9 else 'test'

def prepare(records,taxonomy,seed=42):
    hashes={}; groups={}; result=[]
    # Detect cross-dish duplicates first; shared images merge dish groups.
    parent={str(r['dish_id']):str(r['dish_id']) for r in records}
    def root(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for r in records:
        digest=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
        group=str(r['dish_id'])
        if digest in hashes: parent[root(group)]=root(hashes[digest])
        hashes[digest]=group; r['_digest']=digest
    seen={}
    for r in records:
        key=f'{r["dataset"]}:{r["label"]}'
        if key not in taxonomy: raise ValueError(f'Unmapped taxonomy label: {key}')
        canonical=taxonomy[key]
        if not canonical.get('name'): raise ValueError(f'Missing canonical name: {key}')
        digest=r.pop('_digest')
        annotation_key=(digest,tuple(r.get('bbox',[])))
        if annotation_key in seen:
            if seen[annotation_key]!=canonical['name']: raise ValueError('Duplicate image has conflicting class annotations; resolve manually')
            continue
        seen[annotation_key]=canonical['name']
        group=root(str(r['dish_id']))
        result.append({**r,'group':group,'sha256':digest,'class_name':canonical['name'],'fdc_id':canonical.get('fdc_id'),'split':stable_split(group,seed),'seed':seed})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--taxonomy',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    records=[json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()]
    result=prepare(records,json.loads(Path(a.taxonomy).read_text()),a.seed)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in result))
    print(json.dumps({split:sum(r['split']==split for r in result) for split in ['train','validation','calibration','test']}))
