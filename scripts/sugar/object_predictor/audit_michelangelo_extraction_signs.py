"""Frozen full official AE endpoint: exact evaluation replay, then grid signs.

PREPARATION: ``... audit_michelangelo_extraction_signs check`` is CPU/meta only.
Root alone queues ``run --output NEW_DIRECTORY`` under the active resource/fd9.
No optimizer, backward, parameter change, checkpoint selection or GT generation.
All four saved evaluation logits AND latents must replay bit-exactly before ANY
dense grid evaluation. The original official extract_geometry then calls the
unchanged model.query_geometry; its actual float32 queries and logits are
captured, preserving its 10000-query chunks and 129^3 endpoint-inclusive order.
Original raw extracted vertices/faces are also compared with the saved endpoint.

GT occupancy/ambiguity arrays are loaded only as diagnostic labels, never passed
to encode/decode/query_geometry. Encoder inputs remain the saved true full-surface
XYZ/normals, so this remains a native AE representation diagnostic, not tactile
inference. FP = predicted occupied in GT exterior; FN = predicted empty in GT
interior (including possible cavities). Ambiguous surface nodes are retained but
excluded from sign metrics. Numerical and visual reconstruction gates stay fixed.
An integrity-successful audit can contain many sign errors: it is not a repaired
or qualified representation. Every fixed object, failure and attempted call is
reported; existing run/source/data files are read-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .qualify_michelangelo_ae import ABC_TO_VAE, CHECKPOINT_SHA, CONFIG, EXPERIMENT, OBJECT_IDS, VENDOR, digest
from .prepare_michelangelo_extraction_grid import SCOPE as GRID_SCOPE, official_grid
from .train_michelangelo_overfit import tensor

SOURCE = EXPERIMENT/'overfit_repair_v1/michelangelo_coverage_overfit_v2'
GRID = EXPERIMENT/'overfit_repair_v1/michelangelo_extraction_grid_diagnostic_v1'
SCOPE = 'frozen_full_official_ae_exact_extraction_grid_sign_audit_only'
EVAL_CHUNK = 4096
GRID_CHUNK = 10000
EXPECTED_TENSORS = 307
EXPECTED_ELEMENTS = 184659585
CALLS = ('encode', 'decode', 'evaluation_query_geometry', 'grid_query_geometry', 'official_extract_geometry')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def exact_comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite array cannot pass exact endpoint replay')
    same_shape = actual.shape == expected.shape
    return dict(exact=bool(same_shape and actual.dtype == expected.dtype
                           and array_digest(actual) == array_digest(expected)),
                actual_sha256=array_digest(actual), expected_sha256=array_digest(expected),
                actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                maximum_absolute_error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
                if same_shape and actual.size else None)


def sign_arrays_and_metrics(logits, labels, ambiguous):
    logits, labels, ambiguous = np.asarray(logits), np.asarray(labels), np.asarray(ambiguous)
    if (logits.ndim != 1 or logits.shape != labels.shape or logits.shape != ambiguous.shape
            or labels.dtype != bool or ambiguous.dtype != bool or not np.isfinite(logits).all()):
        raise ValueError('Require finite node-aligned logits and boolean labels/ambiguity')
    predicted = logits >= 0  # Same published occupancy convention; not an SDF.
    valid = ~ambiguous
    fp = valid & predicted & ~labels
    fn = valid & ~predicted & labels
    tp = int((valid & predicted & labels).sum())
    tn = int((valid & ~predicted & ~labels).sum())
    fp_count, fn_count = int(fp.sum()), int(fn.sum())
    occupied, exterior = int((valid & labels).sum()), int((valid & ~labels).sum())
    union = tp+fp_count+fn_count
    metrics = dict(total_nodes=len(labels), valid_nodes=int(valid.sum()), ambiguous_nodes=int(ambiguous.sum()),
                   true_positive=tp, true_negative=tn, false_positive=fp_count, false_negative=fn_count,
                   gt_occupied_valid_nodes=occupied, gt_exterior_valid_nodes=exterior,
                   false_positive_rate=fp_count/exterior if exterior else None,
                   false_negative_rate=fn_count/occupied if occupied else None,
                   occupancy_iou=tp/union if union else 1.,
                   occupancy_accuracy=(tp+tn)/int(valid.sum()) if valid.any() else None,
                   ambiguous_raw_sign_disagreements=int((ambiguous & (predicted != labels)).sum()),
                   zero_logit_nodes=int((logits == 0).sum()),
                   all_nonboundary_signs_correct=not (fp_count or fn_count))
    return dict(grid_false_positive=fp, grid_false_negative=fn, grid_valid_for_metrics=valid), metrics


def read_inputs(source, grid):
    result_path, protocol_path = source/'RESULT.json', source/'PROTOCOL.json'
    result, protocol = json.loads(result_path.read_text()), json.loads(protocol_path.read_text())
    grid_result_path = grid/'RESULT.json'
    grid_result = json.loads(grid_result_path.read_text())
    if (result['scope'] != 'full_surface_native_ae_coverage_overfit_only'
            or result['model_updates'] != 1000 or not result['complete']
            or result['initial_checkpoint_sha256'] != CHECKPOINT_SHA
            or protocol['sampling_revision'] != 2
            or tuple(row['object_id'] for row in result['cases']) != OBJECT_IDS
            or grid_result['scope'] != GRID_SCOPE or not grid_result['complete'] or not grid_result['passed']
            or tuple(row['object_id'] for row in grid_result['cases']) != OBJECT_IDS
            or grid_result['abc_to_vae_scale'] != ABC_TO_VAE):
        raise ValueError('Require the fixed complete coverage-v2 endpoint and all four prepared GT grids')
    checkpoint = source/result['checkpoint_file']
    if digest(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Actual endpoint checkpoint changed')
    for relative, sha in protocol['sources'].items():
        if digest(VENDOR/relative) != sha:
            raise ValueError('Official model/inference source changed: '+relative)
    grid_path = grid/grid_result['grid_file']
    if digest(grid_path) != grid_result['grid_sha256']:
        raise ValueError('Prepared extraction queries changed')
    with np.load(grid_path, allow_pickle=False) as saved:
        queries = saved['queries_vae'].copy()
        expected, shape, _ = official_grid()
        if (not exact_comparison(queries, expected)['exact']
                or not np.array_equal(saved['grid_shape'], shape)
                or not np.array_equal(saved['bounds_vae'], [-1.25]*3+[1.25]*3)):
            raise ValueError('Prepared queries do not exactly match the original official extractor')
    data, bindings = [], []
    for row, gt_row in zip(result['cases'], grid_result['cases']):
        oid = row['object_id']; case_path = source/row['output_file']; label_path = grid/gt_row['file']
        if (not row['complete'] or digest(case_path) != row['case_output_sha256']
                or digest(label_path) != gt_row['file_sha256']):
            raise ValueError('Missing/changed actual endpoint or grid-label case: '+oid)
        with np.load(case_path, allow_pickle=False) as z:
            case = {key: z[key].copy() for key in z.files}
        with np.load(label_path, allow_pickle=False) as z:
            labels = {key: z[key].copy() for key in
                      ('occupancy_labels','boundary_ambiguous','training_eligible','evaluation_overlap')}
        for key in labels:
            if labels[key].dtype != bool or labels[key].shape != (len(queries),):
                raise ValueError('Invalid complete grid mask: '+key)
        if not np.array_equal(labels['training_eligible'], ~(labels['boundary_ambiguous']|labels['evaluation_overlap'])):
            raise ValueError('Prepared grid eligibility changed')
        # Same exact original GT and old evaluation source, without passing either
        # GT mesh or labels to the official neural encoder/decoder.
        for role, key in [('vertices','truth_vertices_canonical'), ('faces','truth_faces')]:
            path = Path(gt_row['source_'+role])
            if digest(path) != gt_row['source_'+role+'_sha256'] or not np.array_equal(case[key], np.load(path,allow_pickle=False)):
                raise ValueError('Endpoint and grid use different GT '+role)
        eval_path = Path(gt_row['old_evaluation_file'])
        if digest(eval_path) != gt_row['old_evaluation_file_sha256']:
            raise ValueError('Frozen evaluation pool changed')
        with np.load(eval_path, allow_pickle=False) as old:
            for now_key, old_key in [('occupancy_queries_vae','eval_queries_vae'), ('occupancy_labels','eval_labels'),
                                     ('input_surface_xyz_vae','input_surface_xyz_vae'), ('input_normals','input_normals')]:
                if not exact_comparison(case[now_key],old[old_key])['exact']:
                    raise ValueError('Endpoint input/evaluation differs from frozen source: '+now_key)
        if case['shape_latent'].shape != (256,64) or case['occupancy_logits'].shape != (32768,):
            raise ValueError('Incomplete actual saved latent/evaluation')
        data.append(dict(object_id=oid, saved=case, grid=labels))
        bindings.append(dict(object_id=oid, case_file=str(case_path.resolve()), case_sha256=digest(case_path),
                             grid_label_file=str(label_path.resolve()), grid_label_sha256=digest(label_path),
                             old_evaluation_file=str(eval_path), old_evaluation_file_sha256=digest(eval_path)))
    receipts = dict(source_result=str(result_path.resolve()), source_result_sha256=digest(result_path),
                    source_protocol_sha256=digest(protocol_path),
                    checkpoint_file=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                    initial_checkpoint_sha256=CHECKPOINT_SHA, original_model_updates=1000,
                    grid_result=str(grid_result_path.resolve()), grid_result_sha256=digest(grid_result_path),
                    grid_file=str(grid_path.resolve()), grid_sha256=digest(grid_path), cases=bindings)
    return result, protocol, queries, data, receipts


def original_class_and_config(result):
    import yaml
    config = yaml.safe_load((VENDOR/CONFIG).read_text())['model']['params']['shape_module_cfg']
    if config != result['loading']['official_config']:
        raise ValueError('Released architecture config differs from actual trained architecture')
    sys.path.insert(0, str(VENDOR))
    module_name, name = config['target'].rsplit('.',1)
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != (VENDOR/'michelangelo/models/tsal/sal_perceiver.py').resolve():
        raise ValueError('Model class is not the official vendor source')
    return getattr(module,name), config


def load_endpoint(result, receipts, *, meta=False):
    import torch
    original, config = original_class_and_config(result)
    with torch.device('meta' if meta else 'cpu'):
        model = original(device=None,dtype=None,**config['params'])
    saved = torch.load(receipts['checkpoint_file'],map_location='meta' if meta else 'cpu',mmap=True,weights_only=False)
    if (saved['step'] != 1000 or saved['initial_checkpoint_sha256'] != CHECKPOINT_SHA
            or len(saved['model']) != EXPECTED_TENSORS
            or list(saved['parameter_names']) != [n for n,_ in model.named_parameters()]):
        raise ValueError('Incomplete/wrong actual full-model endpoint')
    state = model.state_dict()
    if state.keys() != saved['model'].keys():
        raise ValueError('Full official state dictionary differs')
    for key in state:
        if state[key].shape != saved['model'][key].shape or state[key].dtype != saved['model'][key].dtype:
            raise ValueError('Wrong official parameter shape/dtype: '+key)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_ELEMENTS:
        raise ValueError('Unexpected full official architecture size')
    if not meta:
        model.load_state_dict(saved['model'],strict=True)
        for key,value in model.state_dict().items():
            if not torch.equal(value,saved['model'][key]):
                raise ValueError('Actual checkpoint parameter load changed: '+key)
    del saved;gc.collect()
    return model.requires_grad_(False).eval(), dict(full_parameter_tensors=len(state),
        full_parameter_elements=EXPECTED_ELEMENTS, meta_only=meta, strict_loaded=not meta,
        model_class=config['target'], official_config=config)


def new_counts():
    return {name:dict(attempted=0,completed=0) for name in CALLS}


def counted(counts,name,function,*args,**kwargs):
    counts[name]['attempted'] += 1
    value = function(*args,**kwargs)
    counts[name]['completed'] += 1
    return value


def run(args):
    from .retained_execution import require_active_resource
    resource = require_active_resource()
    import torch
    result, _, queries, data, bindings = read_inputs(args.source,args.grid)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(scope=SCOPE,bindings=bindings,resource=resource,object_ids=list(OBJECT_IDS),
                    adapter_sha256=digest(Path(__file__)),model_updates=0,backward_calls=0,
                    physics_controls=0,checkpoint_selection=False,posterior='mode; sample_posterior=False',
                    eval_chunk=EVAL_CHUNK,grid_chunk=GRID_CHUNK,total_grid_nodes=len(queries),
                    gt_labels_encoder_input=False,original_geometric_gates_unchanged=True,
                    numerical_gate_note='Sign audit integrity is separate from existing reconstruction gates')
    write_json(args.output/'PROTOCOL.json',protocol)
    model,loading = load_endpoint(result,bindings)
    model=model.to('cuda').eval();device=torch.device('cuda')
    protocol['runtime'] = dict(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        parameter_dtype=str(next(model.parameters()).dtype),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        device_name=torch.cuda.get_device_name(device))
    write_json(args.output/'PROTOCOL.json',protocol)
    from michelangelo.models.tsal.inference_utils import extract_geometry
    cases=[];decoded_cases=[];replayed=[]
    # All four exact evaluation qualifications precede all dense grid forwards.
    with torch.no_grad():
        for item in data:
            saved=item['saved'];counts=new_counts();record=dict(object_id=item['object_id'],call_counts=counts,
                                                               equivalence_passed=False,complete=False)
            try:
                _,latent,_=counted(counts,'encode',model.encode,tensor(saved['input_surface_xyz_vae'],device),
                                  tensor(saved['input_normals'],device),sample_posterior=False)
                decoded=counted(counts,'decode',model.decode,latent)
                parts=[counted(counts,'evaluation_query_geometry',model.query_geometry,
                               tensor(saved['occupancy_queries_vae'][start:start+EVAL_CHUNK],device),decoded)
                       for start in range(0,len(saved['occupancy_labels']),EVAL_CHUNK)]
                values=torch.cat(parts,dim=1)[0].cpu().numpy()
                latent_cpu=latent[0].cpu().numpy()
                comparisons=dict(latent=exact_comparison(latent_cpu,saved['shape_latent']),
                                 evaluation_logits=exact_comparison(values,saved['occupancy_logits']))
                record.update(equivalence=comparisons,equivalence_passed=all(x['exact'] for x in comparisons.values()))
                decoded_cases.append(decoded);replayed.append((latent_cpu,values))
            except Exception:
                record['error']=traceback.format_exc();decoded_cases.append(None);replayed.append(None)
            cases.append(record)
        exact=all(row['equivalence_passed'] for row in cases)
        write_json(args.output/'ENDPOINT_EQUIVALENCE.json',dict(passed=exact,cases=cases,
                                                              dense_grid_calls_at_this_stage=0))
        print('GRID_ENDPOINT_EQUIVALENCE',exact,flush=True)
        if exact:
            for item,decoded,replay,record in zip(data,decoded_cases,replayed,cases):
                counts=record['call_counts'];saved=item['saved'];parts=[];offset=0
                try:
                    def geometry(actual_queries):
                        nonlocal offset
                        cpu_queries=actual_queries[0].detach().cpu().numpy()
                        end=offset+len(cpu_queries)
                        if not exact_comparison(cpu_queries,queries[offset:end])['exact']:
                            raise ValueError('Official extraction query coordinates/order differ from prepared grid')
                        logits=counted(counts,'grid_query_geometry',model.query_geometry,actual_queries,decoded)
                        parts.append(logits[0].detach().cpu().numpy().copy());offset=end
                        return logits
                    meshes,valid=counted(counts,'official_extract_geometry',extract_geometry,geometry,device,
                                         batch_size=1,octree_depth=7,num_chunks=GRID_CHUNK)
                    if offset != len(queries):
                        raise ValueError('Incomplete official dense grid query coverage')
                    logits=np.concatenate(parts)
                    error_arrays,metrics=sign_arrays_and_metrics(logits,item['grid']['occupancy_labels'],
                                                                 item['grid']['boundary_ambiguous'])
                    arrays=dict(grid_logits=logits,shape_latent=replay[0],evaluation_logits=replay[1],
                                evaluation_queries_vae=saved['occupancy_queries_vae'],evaluation_labels=saved['occupancy_labels'],
                                **{'grid_'+key:value for key,value in item['grid'].items()},**error_arrays)
                    path=args.output/(item['object_id']+'.npz');np.savez_compressed(path,**arrays)
                    record.update(grid_outputs_saved=True,file=path.name,file_sha256=digest(path),
                                  grid_queries_exact=True,grid_metrics=metrics)
                    if not bool(valid[0]):
                        raise ValueError('Official repeated extraction returned no mesh')
                    raw,faces=meshes[0]
                    comparison=dict(raw_vertices=exact_comparison(raw,saved['reconstruction_vertices_vae_raw_official']),
                                    faces=exact_comparison(faces,saved['reconstruction_faces']))
                    record.update(saved_mesh_replay=comparison,
                                  complete=all(x['exact'] for x in comparison.values()))
                    if not record['complete']:
                        record['error']='Raw official mesh/faces differ from the saved endpoint despite eval binding'
                except Exception:
                    record['error']=traceback.format_exc()
                write_json(args.output/'PROGRESS.json',cases)
                print('EXTRACTION_SIGN_CASE',json.dumps(record),flush=True)
    totals={name:{key:sum(row['call_counts'][name][key] for row in cases) for key in ('attempted','completed')} for name in CALLS}
    success=exact and all(row['complete'] for row in cases)
    report=dict(protocol,loading=loading,cases=cases,call_counts=totals,complete=success,
                audit_integrity_passed=success,representation_qualified=False,
                all_nonboundary_signs_correct=success and all(row['grid_metrics']['all_nonboundary_signs_correct'] for row in cases))
    write_json(args.output/'RESULT.json',report)
    return 0 if success else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','run'))
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--grid',type=Path,default=GRID)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='check':
        result,_,queries,data,bindings=read_inputs(args.source,args.grid)
        model,loading=load_endpoint(result,bindings,meta=True)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU/meta check unexpectedly initialized CUDA')
        print(json.dumps(dict(scope=SCOPE,passed=True,objects=[x['object_id'] for x in data],
                              grid_nodes=len(queries),loading=loading,cuda_initialized=torch.cuda.is_initialized(),
                              model_forwards=0,model_updates=0)))
        return 0
    if args.output is None:
        parser.error('run requires a new --output directory')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
