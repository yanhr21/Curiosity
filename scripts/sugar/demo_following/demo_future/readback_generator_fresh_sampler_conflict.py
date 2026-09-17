"""Saved full-gradient replication on fresh official TRAIN sampling paths."""
import hashlib
import json

import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import PAIRED_OUT,write
from scripts.sugar.demo_following.demo_future.audit_generator_self_objective_vs_sampler import compare

FRESH=PAIRED_OUT/'fresh_full_sampler_gradient'
OBJECTIVE=PAIRED_OUT/'objective_vs_sampler_gradient'
OUT=FRESH/'objective_conflict_readback'


def main():
    torch.set_num_threads(8)
    result=json.loads((FRESH/'RESULT.json').read_text());read=json.loads((FRESH/'SAVED_MEAN_READBACK.json').read_text())
    assert result['checks_passed'] and result['plot_inspected'] and read['checks_passed']
    protocol=json.loads((FRESH/'PROTOCOL.json').read_text())
    objective=torch.load(OBJECTIVE/'MEAN_GRADIENTS.pt',map_location='cpu')
    reference=torch.load(FRESH/'MEAN_GRADIENTS.pt',map_location='cpu')
    checks=dict(parameter_names_exact=objective['parameter_names']==reference['parameter_names'],
        fixed_objective_file_unchanged=hashlib.sha256((OBJECTIVE/'MEAN_GRADIENTS.pt').read_bytes()).hexdigest()==protocol['fixed_objective_sha256'])
    OUT.mkdir(exist_ok=False)
    comparisons={key:compare(objective['parameter_names'],components,reference['means']) for key,components in objective['means'].items()}
    rows=result['rows'];means={};seed_rows={}
    for phase in protocol['train_phases']:
        seed_rows[str(phase)]={str(seed):float(np.mean([row['fixed_objective_vs_full_sampler']['full_model']['dot'] for row in rows if row['phase']==phase and row['seed']==seed])) for seed in protocol['seeds']}
    for key in reference['means']:
        selected=rows if key=='all_train' else [r for r in rows if r['seed']==int(key[5:])] if key.startswith('seed_') else [r for r in rows if r['phase']==int(key)]
        means[key]={}
        for group in ('full_model','geometry_columns','target_encoder','denoiser'):
            row_mean=float(np.mean([r['fixed_objective_vs_full_sampler'][group]['dot'] for r in selected]))
            value=comparisons['all_noise'][key]['total'][group]['descent_dot']
            checks[f'{key}_{group}_row_mean_dot_replay']=bool(np.isclose(row_mean,value,rtol=3e-5,atol=1e-8))
            means[key][group]=dict(saved_mean_dot=value,individual_dot_mean=row_mean)
    checks['all144_fresh_rows']=len(rows)==144
    checks['all_source_rows_finite']=all(np.isfinite(row['loss']) and all(np.isfinite(v) for value in row['fixed_objective_vs_full_sampler'].values() for k,v in value.items() if v is not None) for row in rows)
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    decision={}
    for phase in protocol['train_phases']:
        values=list(seed_rows[str(phase)].values());dot=comparisons['all_noise'][str(phase)]['total']['full_model']['descent_dot']
        negative=sum(v < -1e-8 for v in values)
        decision[str(phase)]=dict(mean_total_descent_dot=dot,negative_seed_means=negative,all8seed_means=seed_rows[str(phase)],conflict_replicated=dot < -1e-8 and negative>=6,prior_total_conflict=phase in protocol['prior_total_conflict_phases'])
    replicated=[p for p in protocol['prior_total_conflict_phases'] if decision[str(p)]['conflict_replicated']]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(protocol['train_phases'],axes.flat):
        row=decision[str(phase)];ax.plot(range(8),list(row['all8seed_means'].values()),marker='o',label='fresh seed branch mean')
        ax.axhline(row['mean_total_descent_dot'],ls='--',color='C1',label='fresh8 mean');ax.axhline(0,color='gray')
        ax.set_title(f'TRAIN{phase}: {row["negative_seed_means"]}/8 negative');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Fixed training-objective direction versus fresh full-sampler gradient; descent dot, no parameter update')
    fig.tight_layout();fig.savefig(OUT/'FRESH_CONFLICT.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=True,checks=checks,mean_comparisons=comparisons,row_mean_readback=means,phase_decision=decision,replicated_prior_conflict_phases=replicated,plot_inspected=False,
        new_model_forwards=0,new_gradients=0,new_optimizer_updates=0,new_physics_steps=0,
        scope='Fixed previously computed q+rank+aux gradient means versus fresh8seed full official terminal gradients. All TRAIN-only. Different sampling seed sets and FP32 diagnostic coupling retained. Negative dot means local first-order increase for this fixed raw gradient direction, not actual Adam or training failure. No rank removal or primary32 replacement.'))
    write(OUT/'DECISION.json',dict(checks_passed=True,replicated_prior_conflict_phases=replicated,
        next_action='bounded_fresh_full_terminal_vs_existing_objective_compatibility' if replicated else 'stored_adam_direction_vs_fresh_terminal_gradient',training_admitted=False,decision_rule=protocol['decision_rule']))
    print(json.dumps(dict(checks=len(checks),replicated_prior_conflict_phases=replicated,phase_decision=decision)),flush=True)


if __name__=='__main__':main()
