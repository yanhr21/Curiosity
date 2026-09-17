"""Read fixed-seed residuals and full official condition encoders on real TRAIN."""
import argparse
import hashlib
import json
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.replicate_generator_training_seed import GROUP,SOURCES,RUNS
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import PARENT,ActualBranchGeometryDataset,GeneratorWrapper,restore_geometry_state,write,CONTEXT_KEY
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state
from scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay import same_rng

CENTROID=GROUP/'frozen_four_endpoint_train_centroid'
OUT=GROUP/'frozen_four_endpoint_train_representation'
MODELS=dict(ranked18=SOURCES[0],self18=SOURCES[1],ranked18_seed272400=RUNS[0],self18_seed272400=RUNS[1])


def centroid_readback():
    result=json.loads((CENTROID/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    assert not (CENTROID/'SAVED_READBACK.json').exists()
    checks={}
    with np.load(CENTROID/'DECOMPOSITION.npz') as arrays:
        for phase,models in result['phases'].items():
            for name,row in models.items():
                key=f'{name}_{phase}';parts=arrays[key+'_parts'];total=arrays[key+'_total'];mean=arrays[key+'_normalized_mean'];target=arrays[key+'_normalized_target']
                e=mean-target
                checks[key+'_common_exact']=np.array_equal(parts[0],((e[0]+e[1])/2)**2)
                checks[key+'_contrast_exact']=np.array_equal(parts[1],((e[0]-e[1])/2)**2)
                checks[key+'_all288_decomposition']=bool(np.allclose(parts.sum(0),total,rtol=1e-10,atol=1e-12))
                checks[key+'_all_metrics_exact']=(float(parts[0].mean())==row['common_bias_mse'] and float(parts[1].mean())==row['contrast_bias_mse'] and float(parts[2].mean())==row['draw_variance'] and float(total.mean())==row['total_mse'])
                checks[key+'_mean_bias_and_threshold_exact']=float(parts[:2].sum(0).mean())==row['mean_bias_mse'] and bool(parts[:2].sum(0).mean()>row['mean_threshold'])==row['variance_eliminated_mean_still_fails']
        checks['all4models9TRAIN_no_reused_queries']=set(result['phases'])==set(map(str,[158,178,197,221,245,261,277,298,318])) and all(set(v)==set(MODELS) for v in result['phases'].values())
    checks['all_primary_sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in result['source_hashes'].items())
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(CENTROID/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_model_forwards=0,new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def distances(features):
    x=np.asarray(features,dtype=np.float64).reshape(18,-1)
    return ((x[:,None]-x[None])**2).mean(-1)


def run():
    torch.set_num_threads(1)
    assert json.loads((CENTROID/'SAVED_READBACK.json').read_text())['checks_passed']
    centroid=json.loads((CENTROID/'RESULT.json').read_text());assert centroid['plot_inspected']
    assert json.loads((GROUP/'COMPARISON.json').read_text())['checks_passed']
    OUT.mkdir(exist_ok=False)
    plan=json.loads((SOURCES[0]/'PROTOCOL.json').read_text());phases=plan['data']['train']['phases'];assert len(phases)==9
    data=ActualBranchGeometryDataset(**plan['branch_dataset']);assert data.real_case_count==18 and len(data)==144
    samples=[data[i] for i in range(18)]
    obs={k:torch.stack([s['obs'][k] for s in samples]) for k in samples[0]['obs']}
    target=torch.stack([s['action'] for s in samples])
    checks={};rows={};arrays={};hashes={};reference_obs=None;reference_target=None
    write(OUT/'PROTOCOL.json',dict(models={k:str(v) for k,v in MODELS.items()},train_phases=phases,real_train_cases=18,
        full_checkpoint_parameters=8319216,batched_encoder_calls=16,rows_per_encoder_call=18,denoiser_calls=0,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        scope='Four complete immutable official Generator checkpoints, actual18TRAIN only. Use original full condition encoder: object/history/selected-demo tokens, each256D. Four encoder calls per model (correct, paired geometry swap, zero context, swapped zero context), with full model/state/normalizer/RNG controls. Saved primary32draw residuals remain the performance evidence. Token distances and label-distance correlations are descriptive; no linear probe, new predictor, retrieval deployment, aligned hidden coordinates across seeds or causal claim.',
        automatic_next_action='Inspect all9token-distance panels and saved arrays alongside completed allfour residual decomposition and bothseed primary comparison. Separate stable residual bias from seed-sensitive conditioning. Predeclare one TRAIN-based representation or objective hypothesis supported across the fixed endpoints before any new bounded experiment. No thirdseed, noise averaging, teacher/weight/solver sweep, old513, generatedphysics or SMPbenefit claim.'))
    for name,source in MODELS.items():
        path=source/'demo_geometry/checkpoints/endpoint.ckpt';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic');policy.eval().requires_grad_(False)
        normalized=policy.normalizer.normalize(obs);ntarget=policy.normalizer['action'].normalize(target)
        if reference_obs is None:reference_obs={k:v.clone() for k,v in normalized.items()};reference_target=ntarget.clone()
        checks[name+'_full8319216']=sum(p.numel() for p in policy.parameters())==8319216
        checks[name+'_normalization_exact']=all(torch.equal(v,reference_obs[k]) for k,v in normalized.items()) and torch.equal(ntarget,reference_target)
        swapped={k:v.clone() for k,v in normalized.items()};swapped[CONTEXT_KEY]=swapped[CONTEXT_KEY].reshape(9,2,*swapped[CONTEXT_KEY].shape[1:]).flip(1).reshape_as(swapped[CONTEXT_KEY])
        checks[name+'_only_paired_geometry_swapped']=all(torch.equal(v,swapped[k]) for k,v in normalized.items() if k!=CONTEXT_KEY)
        before_rng=rng_state(torch.device('cpu'))
        with torch.inference_mode():
            correct=policy.obs_encoder(normalized,training=False);wrong=policy.obs_encoder(swapped,training=False)
            policy.obs_encoder.condition_mode='zero_context'
            zero=policy.obs_encoder(normalized,training=False);zero_wrong=policy.obs_encoder(swapped,training=False)
            policy.obs_encoder.condition_mode='demo_geometry'
        checks[name+'_encoder_rng_unchanged']=same_rng(before_rng,rng_state(torch.device('cpu')))
        checks[name+'_fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v,state[k]) for k,v in policy.state_dict().items())
        checks[name+'_no_gradients']=all(p.grad is None for p in policy.parameters())
        checks[name+'_all_shapes_finite']=all(v.shape==(18,3,256) and bool(torch.isfinite(v).all()) for v in (correct,wrong,zero,zero_wrong))
        checks[name+'_other_tokens_exact']=torch.equal(correct[:,:2],wrong[:,:2]) and torch.equal(correct[:,:2],zero[:,:2])
        checks[name+'_paired_target_tokens_swap_exact']=torch.equal(correct[:,2].reshape(9,2,256).flip(1).reshape(18,256),wrong[:,2])
        checks[name+'_zero_context_invariant']=torch.equal(zero,zero_wrong) and torch.equal(zero[:,2],zero[0:1,2].expand(18,256))
        arrays[name+'_correct_tokens']=correct.numpy();arrays[name+'_wrong_tokens']=wrong.numpy();arrays[name+'_zero_tokens']=zero.numpy();arrays[name+'_zero_wrong_tokens']=zero_wrong.numpy()
        token_dist={field:distances(correct[:,i].numpy()) for i,field in enumerate(('object','history','geometry'))}
        token_dist['joint']=sum(token_dist.values())/3
        target_distance=distances(ntarget.numpy());triangle=np.triu_indices(18,1)
        correlations={}
        for field,distance in token_dist.items():
            arrays[name+'_'+field+'_distance']=distance
            x=distance[triangle];y=target_distance[triangle]
            correlations[field]=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
        pair_rms={field:[float(distance[2*i,2*i+1]**.5) for i in range(9)] for field,distance in token_dist.items()}
        geometry_spread=float(np.sqrt(token_dist['geometry'][triangle].mean()))
        assert geometry_spread>0
        relative_pair=[v/geometry_spread for v in pair_rms['geometry']]
        checks[name+'_shared_world_state_history_tokens_exact']=all(v==0 for field in ('object','history') for v in pair_rms[field])
        rows[name]=dict(phase_pair_token_rms={str(p):{k:v[i] for k,v in pair_rms.items()} for i,p in enumerate(phases)},
            geometry_off_diagonal_rms=geometry_spread,phase_relative_geometry_pair_rms={str(p):relative_pair[i] for i,p in enumerate(phases)},
            all153_nonindependent_pair_distance_correlations=correlations,
            token_global_rms={field:float(np.sqrt((correct[:,i].numpy().astype(np.float64)**2).mean())) for i,field in enumerate(('object','history','geometry'))})
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        del policy,state
    arrays['normalized_target']=reference_target.numpy()
    for k,v in reference_obs.items():arrays['normalized_obs_'+k]=v.numpy()
    arrays['target_distance']=distances(reference_target.numpy());arrays['phases']=np.repeat(phases,2);arrays['branches']=np.tile([0,1],9)
    arrays['raw_geometry_distance']=distances(reference_obs[CONTEXT_KEY].numpy())
    np.savez_compressed(OUT/'REPRESENTATIONS.npz',**arrays)
    with np.load(OUT/'REPRESENTATIONS.npz') as a:checks['all_saved_arrays_exact']=set(a.files)==set(arrays) and all(np.array_equal(a[k],v) for k,v in arrays.items())
    checks['full_checkpoint_hashes_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    names=list(MODELS);labels=['ranked84','self84','ranked400','self400']
    for phase,ax in zip(phases,axes.flat):
        ax.bar(range(4),[rows[name]['phase_relative_geometry_pair_rms'][str(phase)] for name in names]);ax.set_xticks(range(4),labels);ax.set_title(f'TRAIN{phase}');ax.set_ylabel('pair distance / all-pair RMS spread');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Four official encoders: relative branch-token distances; scale invariant, descriptive only')
    fig.tight_layout();fig.savefig(OUT/'CONDITION_TOKEN_DISTANCES.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,models=rows,source_hashes=hashes,plot_inspected=False,
        batched_encoder_calls=16,denoiser_calls=0,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0))
    assert all(checks.values())
    print(json.dumps(dict(checks=len(checks),models=rows)),flush=True)


def readback():
    result=json.loads((OUT/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    assert not (OUT/'SAVED_READBACK.json').exists()
    checks={}
    with np.load(OUT/'REPRESENTATIONS.npz') as a:
        checks['target_distances_exact']=np.array_equal(distances(a['normalized_target']),a['target_distance'])
        for name in MODELS:
            c,w,z,zw=[a[name+'_'+k] for k in ('correct_tokens','wrong_tokens','zero_tokens','zero_wrong_tokens')]
            checks[name+'_wrong_target_swap_exact']=np.array_equal(c[:,2].reshape(9,2,256)[:,::-1].reshape(18,256),w[:,2])
            checks[name+'_zero_invariance_exact']=np.array_equal(z,zw)
            for i,field in enumerate(('object','history','geometry')):
                d=distances(c[:,i]);checks[name+field+'_distance_exact']=np.array_equal(d,a[name+'_'+field+'_distance'])
                checks[name+field+'_pair_metrics_exact']=all(float(d[2*j,2*j+1]**.5)==result['models'][name]['phase_pair_token_rms'][str(int(phase))][field] for j,phase in enumerate(a['phases'][::2]))
            checks[name+'_joint_distances_exact']=np.array_equal(sum(a[name+'_'+field+'_distance'] for field in ('object','history','geometry'))/3,a[name+'_joint_distance'])
            d=a[name+'_geometry_distance'];spread=float(np.sqrt(d[np.triu_indices(18,1)].mean()))
            checks[name+'_relative_geometry_distance_exact']=spread==result['models'][name]['geometry_off_diagonal_rms'] and all(float(d[2*j,2*j+1]**.5)/spread==result['models'][name]['phase_relative_geometry_pair_rms'][str(int(p))] for j,p in enumerate(a['phases'][::2]))
        checks['raw_geometry_distances_exact']=np.array_equal(distances(a['normalized_obs_'+CONTEXT_KEY]),a['raw_geometry_distance'])
    checks['all_full_checkpoint_sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in result['source_hashes'].items())
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(OUT/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_model_forwards=0,new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['centroid-readback','run','readback'],required=True)
    mode=parser.parse_args().mode
    {'centroid-readback':centroid_readback,'run':run,'readback':readback}[mode]()


if __name__=='__main__':main()
