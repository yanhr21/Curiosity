"""Audit existing measured robot inputs through recorded clocks and full encoders."""
import argparse
import hashlib
import json
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.audit_generator_terminal_objective import OUT as PREVIOUS,digest
from scripts.sugar.demo_following.demo_future.audit_generator_seed_representation import MODELS
from scripts.sugar.demo_following.demo_future.replicate_generator_training_seed import GROUP,SOURCES
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import PARENT,ActualBranchGeometryDataset,GeneratorWrapper,restore_geometry_state,write
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state
from scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay import same_rng

ROOT=Path(__file__).resolve().parents[4]
OUT=GROUP/'causal_robot_input_audit'


def run():
    torch.set_num_threads(1)
    decision=json.loads((PREVIOUS/'DECISION.json').read_text())
    assert decision['checks_passed'] and not decision['permits_preparing_matched_experiment']
    OUT.mkdir(exist_ok=False)
    plan=json.loads((SOURCES[1]/'PROTOCOL.json').read_text());phases=plan['data']['train']['phases']
    assert phases==[158,178,197,221,245,261,277,298,318]
    write(OUT/'PROTOCOL.json',dict(train_phases=phases,real_cases=18,models={k:str(v) for k,v in MODELS.items()},
        scope='Read only existing admitted TRAIN traces and full four Generator endpoints. Bind current29joint positions and3projected gravity values through pre-action trace, official raw/phase formatter, branch arrays and model observations. Check existing full encoders ignore these fields under use_last_action=True. Roll measured32fields across TRAIN phases only as a diagnostic input intervention; no new predictions, labels, sampling, parameters or physics.',
        prospective_hypothesis='Retain previous36D command and append current measured32D robot state to the existing official robot_state_net first Linear, with zero new columns and all old full weights/statistics retained. This may improve conditional future fitting; observed input omission does not prove missing-information impossibility or benefit. No homemade state encoder, new token network, fabricated SMP features or direct substitution of36command with32state.',
        automatic_next_action='After all18 causal bindings, four full-state/encoder controls, nineplots and independent saved readback pass, prepare faithful36to68 existing-input adaptation and full-model preservation preflight. Any training requires a separate matched protocol and CPU/H200BF16 checks. If actual fields or clocks fail, retain failure and resolve provenance; never manufacture state or launch training.',
        smp_scope='Official SMP uses measured body/joint/box state in216D ten-frame windows. This audit is only existing current32D robot observations; it does not invoke the SMP prior or claim an SMP latent or reward.',
        new_model_modules=0,new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0))
    dataset=ActualBranchGeometryDataset(**plan['branch_dataset']);samples=[dataset[i] for i in range(18)]
    obs={k:torch.stack([s['obs'][k] for s in samples]) for k in samples[0]['obs']}
    checks={};rows={};arrays={};hashes={};joint_order=None
    collector=ROOT/'scripts/sugar/demo_following/demo_future/collect_refiner_pair.py';text=collector.read_text()
    checks['collector_both_fields_saved_before_env_step']=text.index('save("joint_pos_before", robot.data.joint_pos)')<text.index('save("project_gravity", quat_apply(quat_inv(command.robot_base_quat_w)')<text.index('next_observations, reward, terminated, truncated, info = env.step(action)')
    for pi,phase in enumerate(phases):
        group=Path(plan['phase_corpora'][str(phase)]).parent
        # phase_corpora points directly at the audited branch_samples folder.
        assert (group/'PROTOCOL.json').exists() and json.loads((group/'PROTOCOL.json').read_text())['switch_control_frame']==phase
        for bi,arm in enumerate(('original','alternate')):
            trace_path=group/arm/'TRACE.npz';startup=group/arm/'STARTUP.npz';raw=group/'official_il_data'/f'{arm}_RAW.npz';branch=group/'branch_samples/BRANCH_SAMPLES.npz'
            with np.load(startup) as a:names=a['joint_names'].tolist()
            if joint_order is None:joint_order=names
            key=f'{phase}_{arm}';checks[key+'_same29joint_names']=names==joint_order and len(set(names))==29
            with np.load(trace_path) as a:
                joint=a['joint_pos_before'][phase,0].copy();gravity=a['project_gravity'][phase,0].copy();previous=a['reference_command'][phase-5,0].copy();target=a['reference_command'][phase+np.arange(8)*5,0].copy()
                checks[key+'_completed_actual_trace']=not bool(a['done'].any()) and int(a['reference_id'][phase])==(96 if bi==0 else 90)
            with np.load(raw) as a:
                checks[key+'_raw_joint_exact']=np.array_equal(joint,a['joint_pos'][phase])
                checks[key+'_raw_gravity_exact']=np.array_equal(gravity,a['project_gravity'][phase])
            with np.load(branch) as a:
                checks[key+'_branch_current32_exact']=np.array_equal(joint,a['joint_pos'][bi,0]) and np.array_equal(gravity,a['project_gravity'][bi,0])
            index=2*pi+bi
            checks[key+'_actual_model_obs32_exact']=np.array_equal(joint,obs['joint_pos'][index,0].numpy()) and np.array_equal(gravity,obs['project_gravity'][index,0].numpy())
            checks[key+'_previous_tminus5_and_future_labels_separate']=np.array_equal(previous,obs['last_action'][index,0].numpy()) and np.array_equal(target,samples[index]['action'].numpy())
            arrays[key+'_joint']=joint;arrays[key+'_gravity']=gravity;arrays[key+'_previous']=previous
            for path in (trace_path,startup,raw,branch):hashes[str(path)]=digest(path)
        checks[str(phase)+'_causal32_same_in_both_branches']=np.array_equal(arrays[f'{phase}_original_joint'],arrays[f'{phase}_alternate_joint']) and np.array_equal(arrays[f'{phase}_original_gravity'],arrays[f'{phase}_alternate_gravity'])
        gap=arrays[f'{phase}_original_joint']-arrays[f'{phase}_original_previous'][:29]
        rows[str(phase)]=dict(joint_vs_previous_command_rms=float(np.sqrt(np.mean(gap.astype(np.float64)**2))),joint_vs_previous_command_max=float(np.abs(gap).max()),gravity=arrays[f'{phase}_original_gravity'].tolist(),
            interpretation='Current measured joint(t) versus prior reference-command joints(t-5); difference includes clock offset and tracking, not a same-time control error.')
    intervention={k:v.clone() for k,v in obs.items()}
    for field in ('joint_pos','project_gravity'):intervention[field]=torch.roll(obs[field],2,0)
    checks['only_measured32_intervened']=all(torch.equal(obs[k],intervention[k]) for k in obs if k not in ('joint_pos','project_gravity'))
    checks['actual_measured_joint_values_changed']=not torch.equal(obs['joint_pos'],intervention['joint_pos'])
    for name,source in MODELS.items():
        path=source/'demo_geometry/checkpoints/endpoint.ckpt';hashes[str(path)]=digest(path)
        with path.open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic');policy.eval().requires_grad_(False)
        normalized=policy.normalizer.normalize(obs);other=policy.normalizer.normalize(intervention)
        before=rng_state(torch.device('cpu'))
        with torch.inference_mode():a=policy.obs_encoder(normalized,training=False);b=policy.obs_encoder(other,training=False)
        checks[name+'_full8319216_original_robot36']=sum(p.numel() for p in policy.parameters())==8319216 and policy.obs_encoder.robot_state_net[0].in_features==36 and policy.obs_encoder.use_last_action
        checks[name+'_all_encoder_outputs_ignore32_exact']=torch.equal(a,b) and a.shape==(18,3,256)
        checks[name+'_fullstate_rng_unchanged']=same_rng(before,rng_state(torch.device('cpu'))) and state.keys()==policy.state_dict().keys() and all(torch.equal(v,state[k]) for k,v in policy.state_dict().items())
        measured=torch.cat([normalized['joint_pos'],normalized['project_gravity']],-1)
        checks[name+'_existing_normalization32_finite']=measured.shape==(18,1,32) and bool(torch.isfinite(measured).all())
        arrays[name+'_normalized_measured32']=measured.numpy();arrays[name+'_tokens']=a.numpy();arrays[name+'_intervened_tokens']=b.numpy()
        del policy,state
    checks['all9_observed_joints_differ_from_previous_command']=all(v['joint_vs_previous_command_rms']>0 for v in rows.values())
    np.savez_compressed(OUT/'CAUSAL_ROBOT_INPUTS.npz',**arrays)
    with np.load(OUT/'CAUSAL_ROBOT_INPUTS.npz') as a:checks['all_saved_arrays_exact']=set(a.files)==set(arrays) and all(np.array_equal(a[k],v) for k,v in arrays.items())
    sources=[collector,ROOT/'scripts/sugar/demo_following/demo_future/export_tracker_il.py',ROOT/'SUGAR/scripts/sugar_rl/process_tracker_rollout.py',ROOT/'SUGAR/source/sugar_il/sugar_il/model/encoder/generator_state_encoder.py',ROOT/'scripts/sugar/demo_following/demo_future/generator_demo_geometry.py',ROOT/'scripts/sugar/demo_following/demo_future/score_refiner_smp.py']
    for p in sources:hashes[str(p)]=digest(p)
    checks['all_sources_unchanged']=all(digest(p)==v for p,v in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(18,12))
    for phase,ax in zip(phases,axes.flat):
        ax.plot(arrays[f'{phase}_original_joint'],label='measured joint(t)',marker='o',markersize=2);ax.plot(arrays[f'{phase}_original_previous'][:29],label='command joint(t-5)',marker='o',markersize=2)
        ax.set_title(f'TRAIN{phase}');ax.set_xlabel('original joint order');ax.set_ylabel('radians');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Existing causal measured robot state and retained previous command; bothbranches share current state')
    fig.tight_layout();fig.savefig(OUT/'CAUSAL_JOINT_STATE.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,joint_names=joint_order,source_hashes=hashes,plot_inspected=False,full_encoder_calls=8,new_denoiser_calls=0,new_optimizer_updates=0,new_physics_steps=0))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    print(json.dumps(dict(checks=len(checks),phases=rows)),flush=True)


def readback():
    result=json.loads((OUT/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    assert not (OUT/'SAVED_READBACK.json').exists();checks={}
    with np.load(OUT/'CAUSAL_ROBOT_INPUTS.npz') as a:
        for phase,row in result['phases'].items():
            gap=a[f'{phase}_original_joint']-a[f'{phase}_original_previous'][:29]
            checks[phase+'_gap_exact']=float(np.sqrt(np.mean(gap.astype(np.float64)**2)))==row['joint_vs_previous_command_rms'] and float(np.abs(gap).max())==row['joint_vs_previous_command_max']
            checks[phase+'_bothbranches_state_exact']=all(np.array_equal(a[f'{phase}_original_{field}'],a[f'{phase}_alternate_{field}']) for field in ('joint','gravity'))
        for name in MODELS:checks[name+'_encoder_invariance_exact']=np.array_equal(a[name+'_tokens'],a[name+'_intervened_tokens'])
    checks['all_sources_unchanged']=all(digest(p)==v for p,v in result['source_hashes'].items())
    assert all(checks.values())
    write(OUT/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_model_forwards=0,new_optimizer_updates=0,new_physics_steps=0))
    write(OUT/'DECISION.json',dict(checks_passed=True,prepare_existing_robot_input_extension=True,new_input_width=68,old_input_width=36,added_current_measured_fields=['joint_pos','project_gravity'],added_parameters=8192,full_parameter_count=8327408,
        next_action='Prepare samefullmodel36to68 robot-input adapter with zero new columns and unchanged normalizer; original fullparameter gradients/forwards/RNG exact and nonzero32column gradients in CPU/H200BF16 preflight before a separate matched training stage.',
        training_admitted=False,scope='Actual causal availability and demonstrated current omission only; not proof of usefulness, fullstate sufficiency or SMP benefit. Retain past36command and original architecture.'))
    print(json.dumps(dict(checks=len(checks),prepare_existing_robot_input_extension=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['run','readback'],required=True)
    {'run':run,'readback':readback}[parser.parse_args().mode]()


if __name__=='__main__':main()
