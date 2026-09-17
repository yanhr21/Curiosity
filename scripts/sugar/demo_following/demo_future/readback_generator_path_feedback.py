"""Replay official scheduler transitions and decompose saved path differences."""
import json
import os
import socket

import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, PARENT, GeneratorWrapper, write

HINGE = BASE / 'matched_generator_branch_paired_hinge025512'
SOFT = BASE / 'matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe'
OWN = HINGE / 'frozen_evaluation/own_reverse_process'
CROSS = HINGE / 'frozen_evaluation/recorded_path_transfer'
OUT = OWN / 'path_feedback_readback'


def main():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    for root in (SOFT, OWN, CROSS):
        d=json.loads((root/'RESULT.json').read_text())
        assert d['checks_passed'] and d['plot_inspected']
    protocol=json.loads((OWN/'PROTOCOL.json').read_text())
    OUT.mkdir(exist_ok=False)
    write(OUT/'PROTOCOL.json',dict(phases=protocol['phases'],seeds=protocol['seeds'],steps=16,
        scope='Saved fullsoftplus025 and fullhinge own16step paths plus hinge evaluated on identical softplus correct-path states. Replay all actual transitions using original DDPMScheduler.step and exact CUDA RNG; no denoiser forward, new sampled path, optimizer or physics. Compare both conditions for transition integrity; direct/state-feedback decomposition uses correct condition only because cross-model evaluations used correct-path xt.',
        definitions='direct=x0_hinge(xt_soft)-x0_soft(xt_soft); feedback=x0_hinge(xt_hinge)-x0_hinge(xt_soft). Their sum is the saved total clean-prediction difference. MSE direct effect and state effect telescope through the same intermediate prediction. This is a descriptive decomposition anchored to softplus states, not a unique causal attribution or deployable hybrid.',
        automatic_next_action='Inspect all nine decomposition curves and complete exact transition/RNG checks. Identify where TRAIN own-path errors depart from same-state errors before proposing one bounded next method diagnostic; do not select intermediate outputs, hybrid models, weights, solver grids or claim generated control/SMP benefit.',
        new_optimizer_updates=0,new_denoiser_forwards=0,new_sample_paths=0,new_physics_steps=0))
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    scheduler=policy.noise_scheduler
    scheduler.set_timesteps(16)
    assert scheduler.config.prediction_type=='epsilon'
    checks={};rows={};transition_count=0
    for phase in protocol['phases']:
        archives=[]
        for root in (SOFT,OWN):
            with np.load(root/f'phase_{phase}_steps_16.npz') as data:
                archives.append({k:data[k].copy() for k in data.files})
        a,b=archives
        with np.load(CROSS/f'phase_{phase}.npz') as data:
            cross=data['clean'][:,0].astype(np.float64)
            checks[f'{phase}_cross_target_exact']=np.array_equal(data['target'].astype(np.float64),a['normalized_actual_target'])
            checks[f'{phase}_cross_times_exact']=np.array_equal(data['times'],a['times'])
        checks[f'{phase}_targets_times_seeds_exact']=all(np.array_equal(a[k],b[k]) for k in ('times','seeds','normalized_actual_target'))
        checks[f'{phase}_official16times_exact']=a['times'].tolist()==scheduler.timesteps.tolist()
        checks[f'{phase}_initial_gaussian_shared_exact']=np.array_equal(a['xt'][:,:,:,0],b['xt'][:,:,:,0])
        reference_rng={}
        for model_index,data in enumerate(archives):
            initial_exact=clean_exact=previous_exact=rng_exact=True
            for si,seed in enumerate(protocol['seeds']):
                for condition in (0,1):
                    for branch in (0,1):
                        torch.manual_seed(seed)
                        initial=torch.randn((1,8,36),device='cuda')
                        initial_exact &= np.array_equal(initial.cpu().numpy()[0],data['xt'][si,condition,branch,0])
                        for ti,time in enumerate(scheduler.timesteps):
                            key=(si,condition,branch,ti)
                            rng=torch.cuda.get_rng_state().clone()
                            if model_index==0:reference_rng[key]=rng
                            else:rng_exact &= torch.equal(rng,reference_rng[key])
                            xt=torch.from_numpy(data['xt'][si,condition,branch,ti:ti+1]).cuda()
                            epsilon=torch.from_numpy(data['epsilon'][si,condition,branch,ti:ti+1]).cuda()
                            result=scheduler.step(epsilon,time,xt)
                            transition_count+=1
                            clean_exact &= np.array_equal(result.pred_original_sample.cpu().numpy()[0],data['pred_original'][si,condition,branch,ti])
                            previous_exact &= np.array_equal(result.prev_sample.cpu().numpy()[0],data['prev_sample'][si,condition,branch,ti])
            for label,value in (('initial',initial_exact),('clean',clean_exact),('previous',previous_exact),('rng',rng_exact)):
                checks[f'{phase}_{model_index}_{label}_exact']=bool(value)
        old=a['pred_original'][:,0].astype(np.float64);new=b['pred_original'][:,0].astype(np.float64)
        direct=cross-old;feedback=new-cross;total=new-old
        checks[f'{phase}_prediction_decomposition']=bool(np.allclose(direct+feedback,total,rtol=0,atol=1e-14))
        target=a['normalized_actual_target'][None,:,None]
        mse=[np.square(v-target).mean((0,1,3,4)) for v in (old,cross,new)]
        direct_mse_effect=mse[1]-mse[0];feedback_mse_effect=mse[2]-mse[1]
        checks[f'{phase}_mse_decomposition']=bool(np.allclose(direct_mse_effect+feedback_mse_effect,mse[2]-mse[0],rtol=0,atol=1e-14))
        np.savez_compressed(OUT/f'phase_{phase}.npz',direct=direct,feedback=feedback,total=total,
            delta_xt=b['xt'][:,0].astype(np.float64)-a['xt'][:,0],times=a['times'],
            old_mse=mse[0],hinge_on_old_mse=mse[1],hinge_own_mse=mse[2])
        rows[str(phase)]=dict(times=a['times'].tolist(),old_mse=mse[0].tolist(),hinge_on_old_mse=mse[1].tolist(),hinge_own_mse=mse[2].tolist(),
            direct_mse_effect=direct_mse_effect.tolist(),state_feedback_mse_effect=feedback_mse_effect.tolist(),
            final=dict(old_mse=float(mse[0][-1]),hinge_on_old_mse=float(mse[1][-1]),hinge_own_mse=float(mse[2][-1]),direct_effect=float(direct_mse_effect[-1]),state_feedback_effect=float(feedback_mse_effect[-1])))
        write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows,transition_count=transition_count))
        if not all(checks.values()):raise RuntimeError('Saved transition or path decomposition mismatch')
        print(json.dumps(dict(phase=phase,checks_passed=True,final=rows[str(phase)]['final'])),flush=True)
    checks['all4608scheduler_transitions']=transition_count==4608
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(protocol['phases'],axes.flat):
        row=rows[str(phase)]
        ax.plot(row['times'],row['direct_mse_effect'],label='direct model effect',color='C2')
        ax.plot(row['times'],row['state_feedback_mse_effect'],label='state feedback effect',color='C1')
        ax.plot(row['times'],np.asarray(row['hinge_own_mse'])-row['old_mse'],label='total MSE change',color='black')
        ax.axhline(0,color='gray',linewidth=.6);ax.invert_xaxis();ax.grid(alpha=.25)
        ax.set_title(f'Phase{phase}');ax.set_xlabel('Reverse diffusion time')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Saved4seed correct paths: exact official transitions; descriptive decomposition; no model update or physics')
    fig.tight_layout();fig.savefig(OUT/'PATH_FEEDBACK.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,
        plot_inspected=False,transition_count=transition_count,new_optimizer_updates=0,new_denoiser_forwards=0,new_sample_paths=0,new_physics_steps=0))
    assert all(checks.values())


if __name__=='__main__':main()
