"""Generate one immutable1904 CPU encoding in the qualified environment."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time
import numpy as np
from . import semantic_state_data as data
from . import semantic_state_training as old
from . import semantic_state_encoded_cache as cache


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared-pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    p=old.validate_protocol(args.prepared_pair)
    args.output.mkdir(parents=True,exist_ok=False)
    sources={v['path']:v['sha256'] for v in p['prepared_sources'].values()}
    for path in (Path(cache.__file__).resolve(),Path(__file__).resolve()):sources[str(path)]=data.sha(path)
    upstream={v['path']:v['sha256'] for values in (p['source_bindings'],p['qualification_bindings']) for v in values.values()}
    upstream[str((args.prepared_pair/'PROTOCOL.json').resolve())]=data.sha(args.prepared_pair/'PROTOCOL.json')
    recordings=cache.recording_bindings(p['source_data'])
    protocol=dict(study=cache.CACHE_STUDY,rows=1904,fit_rows=464,development_rows=1440,
        source_bindings=sources,recording_bindings=recordings,upstream_artifacts=upstream,
        original_pair_protocol=str((args.prepared_pair/'PROTOCOL.json').resolve()),original_model_source=p['source_endpoint'],
        model_source_bindings=p['source_bindings'],original_qualification=p['qualification'],
        source_data=p['source_data'],data=p['data'],host=socket.gethostname(),python=sys.executable,
        scope='Freeze the full qualified encoding, labels and causal histories; no rounding, threshold change, model forward or scientific repair claim. Original failed run remains unchanged.')
    cache.write_json(args.output/'PROTOCOL.json',protocol)
    started=time.monotonic();reports={}
    for role in cache.ROLES:
        print('CACHE_ENCODE_BEGIN',role,flush=True)
        dataset=data.SemanticStateDataset(p['source_data'],p['data'],role=role)
        checks={c:old.qualify_condition(dataset,c,p['fit_limits']) for c in old.CONDITIONS}
        if not all(r['passed'] for r in checks.values()):raise ValueError('Actual cache input qualification failed')
        readback=old.cache_observations(dataset,p,verify_qualified_fit=role=='fit')
        cache.save_role(args.output,dataset)
        loaded=cache.CachedSemanticDataset(args.output,role)
        if [cache.row_hash(row) for row in dataset.rows]!=loaded.cached_input_hashes.tolist():raise ValueError('Saved cache changed input bits')
        reports[role]=dict(conditions=checks,summary_and_original464=readback,saved_load=loaded.cache_readback,
            denominators=data.task_denominators(dataset))
        print('CACHE_ENCODE_COMPLETE',role,len(dataset),flush=True)
    paths=('fit.npz','fit.json','development_interpolation.npz','development_interpolation.json')
    for name,bindings in (('source',sources),('recording',recordings),('upstream',upstream)):
        if not all(data.sha(path)==digest for path,digest in bindings.items()):raise ValueError(name+' changed during cache generation')
    result=dict(passed=True,rows=1904,roles=reports,protocol_sha256=data.sha(args.output/'PROTOCOL.json'),
        artifacts={name:dict(path=name,sha256=data.sha(args.output/name)) for name in paths},
        original464_hashes_exact=True,rounding_applied=False,hash_tolerance_relaxed=False,
        model_forwards=0,optimizer_updates=0,physics_controls=0,elapsed_s=time.monotonic()-started)
    cache.write_json(args.output/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)


if __name__=='__main__':main()
