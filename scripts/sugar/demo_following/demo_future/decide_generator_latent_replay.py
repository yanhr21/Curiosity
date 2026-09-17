"""Apply the predeclared fresh TRAIN gradient rule; preserve all evidence."""
import json
from pathlib import Path

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, write

SOURCE=BASE/'matched_generator_branch_paired_rank025512'
REPLAY=BASE/'generator_train_diffusion_replay8'
AUDIT=REPLAY/'generated_state_gradient_audit'
RUN=BASE/'matched_generator_branch_latent_replay01512'


def main():
    protocol=json.loads((REPLAY/'NEXT_DECISION_PROTOCOL.json').read_text())
    amendment=json.loads((REPLAY/'MATCHED_CONTROL_AMENDMENT.json').read_text())
    audit=json.loads((AUDIT/'RESULT.json').read_text())
    assert audit['execution_completed'] and audit['plot_inspected']
    assert not (AUDIT/'DECISION.json').exists()
    weight=protocol['auxiliary_weight']
    assert weight==amendment['fixed_auxiliary_weight']==.1
    def direction(stats):
        a,b,d=stats['supervised_norm'],stats['generated_state_norm'],stats['dot']
        return dict(supervised_direction_dot=a*a+weight*d,generated_direction_dot=d+weight*b*b,cosine=stats['cosine'],relative_auxiliary_gradient_norm=weight*b/a)
    mean=direction(audit['aggregate']['paired_vs_generated']['full_model'])
    seeds={seed:direction(row['paired_vs_generated']['full_model']) for seed,row in audit['seed_summaries'].items()}
    checks=dict(all128_gradient_rows_and_full_integrity=audit['checks_passed'] and all(audit['checks'].values()) and len(audit['rows'])==128,
        all8_declared_seeds=set(seeds)==set(map(str,range(272230,272238))),
        mean_both_direction_dots_positive=min(mean['supervised_direction_dot'],mean['generated_direction_dot'])>0,
        all8seed_supervised_descent=all(v['supervised_direction_dot']>0 for v in seeds.values()),
        at_least6of8seed_generated_descent=sum(v['generated_direction_dot']>0 for v in seeds.values())>=6)
    passed=all(checks.values())
    decision=dict(checks=checks,permits_preparing_matched_objective_experiment=passed,weight=weight,mean=mean,seeds=seeds,
        negative_individual_rows=sum(r['paired_vs_generated']['full_model']['dot']<0 for r in audit['rows']),
        matched_control_amendment=str(REPLAY/'MATCHED_CONTROL_AMENDMENT.json'),
        scope='Fixed0.1 rule predeclared before fresh8seed gradients. Full raw FP32 eval directions at one learned model, not actual AdamW or generalization. Both experimental arms must receive identical auxiliary replay supervision; no coefficient sweep.',
        automatic_next_action=protocol['on_pass'] if passed else protocol['on_fail'])
    if passed:
        # The amendment overrides the original zero-arm skipping proposal.
        decision['automatic_next_action']=amendment['matched_training']+' '+amendment['exactness']+' Complete actual CPU/BF16 preflights, both512arms, all9phase5condition32draw/fullstates/Adam/readback/72panels and original criteria. No old endpoint extension or physics.'
        plan=json.loads((SOURCE/'PROTOCOL.json').read_text())
        plan.update(run_name=RUN.name,data_coverage_predecessor=str(SOURCE),generator_training_started=False,
            implementation_status='protocol_only_requires_replay_dataset_loss_and_full_actual_preflights',
            objective_audit=str(AUDIT/'RESULT.json'),objective_decision=str(AUDIT/'DECISION.json'),
            generated_state_objective=dict(weight=.1,apply_to_both_arms=True,replay_source=str(REPLAY),
                input_arrays=str(REPLAY/'TRAIN_GENERATED_STATE_INPUTS.npz'),seeds=list(range(272230,272238)),
                times=[45,42,39,36,33,30,27,24,21,18,15,12,9,6,3,0],
                seed_assignment='Original dataset index//14 assigns eight repeated rows for each realcase to eight frozen replay seeds. No new RNG.',
                time_assignment='At each official TRAIN forward use saved time index=(zero-based TRAIN forward counter modulo16), identical for all112rows. All8seed x16time x14case combinations covered32times over512updates. Counter outside model state is not a claim of exact interrupted resume.',
                objective='Original epsilon MSE+0.25scaledsoftplus paired ranking plus0.1*MSE(epsilon_prediction(saved_xt,t,causal_obs), (saved_xt-sqrt(alpha_t)*actual_normalized_future)/sqrt(1-alpha_t)). Detach savedxt and impliedepsilon target; no gradient through sampler. Full encoder/denoiser trained, oldnormalizer and original clipping unchanged.',
                rng='Compute old supervised/ranking loss first, preserve its loss/forwards/gradient and final CPU/CUDA RNG. Auxiliary fullforward uses current stream but restores it afterward; replay indexing is deterministic. Both arms receive exactsame xt/target/time/seed data.',
                expected_exposures_per_real_case=4096,expected_total_auxiliary_rows=57344),
            comparison_scope=amendment['matched_training']+' '+amendment['interpretation'],
            automatic_next_action=decision['automatic_next_action'])
        plan['training']=plan['optimization']=plan['comparison_scope']
        plan['paired_objective']=dict(weight=.25,margin_fraction=.1)
        for k in ('weight_selection','gradient_routing_selection'):
            if k in plan:plan['predecessor_'+k]=plan.pop(k)
        # The existing branch_dataset remains the causal/label authority. New
        # replay adapter must attach only extra TRAIN fields outside obs.
        write(REPLAY/'NEXT_MATCHED_PROTOCOL.json',plan)
    write(AUDIT/'DECISION.json',decision)
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
