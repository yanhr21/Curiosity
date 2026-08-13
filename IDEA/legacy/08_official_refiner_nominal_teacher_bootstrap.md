# Official Refiner Nominal-Teacher Bootstrap for Goal-Based Carrying

## Research Question

Can the accepted official SUGAR Refiner provide only the short nominal
contact-establishment behavior that the current causal goal policy is missing,
while a causal state/direct-TacSL policy retains authority to discover a lower
squat, asymmetric support, bottom support, regrasp, or other successful
strategy after the nominal bilateral clamp fails?

The official Refiner is admitted here as a frozen **nominal action teacher**,
not as the final policy, not as a checkpoint warm start for an incompatible
actor, not as curiosity, and not as permission to restore exact-reference
tracking as the exploration objective.

This is a focused child of
`IDEA/06_sugar_smp_tactile_strategy_exploration.md`. It consumes the passed
live-teacher gate recorded in
`DOCS/sugar_live_official_refiner_teacher_result_20260725.md`.

## Evidence That Motivates the Idea

Two separately audited facts now meet:

1. The fresh causal H2R1 goal policy loses left-palm direct TacSL on its first
   action even when its correct previous action and four-frame tactile history
   are restored. Its action is L2 `14.36` from the official continuation.
2. The exact official Refiner observation and frozen `model_10000.pt`
   reproduce the same-state official actor output to L2 `1.94e-6` and preserve
   bilateral direct pressure/two-axis shear for four live goal-scene steps.

This rejects another scalar contact reward as the immediate fix. The missing
piece is nominal action competence at the causal reset boundary.

## Two Matched Bootstrap Mechanisms

### A. Frozen-Teacher Residual

The causal goal policy outputs a residual action:

```text
a_executed = alpha_teacher * a_teacher + scale_residual * a_residual
```

`a_teacher` is produced online by the exact frozen official Refiner from its
declared selected nominal reference. `a_residual` is produced by the existing
SUGAR-native causal goal-policy architecture from actor-visible goal state and
direct TacSL history. The policy optimizer acts on the residual distribution;
the environment applies the declared transformed action.

At initialization, `alpha_teacher=1` and the residual mean must be exactly
zero. A zero-residual audit must reproduce the frozen teacher action before
any learning. After teacher release, `alpha_teacher=0` leaves the causal
residual policy in control. This is a state-dependent action
reparameterization, not an extra reward.

### B. Offline Behavior-Cloned Bootstrap

The same causal goal-policy architecture is trained offline to predict
official actor outputs from the existing motion-disjoint causal
state/direct-TacSL nominal dataset. It receives no online Refiner observation
and no future reference frames. It is then fine-tuned by the same policy
optimizer and compared under the same source states, physics, mounts, budgets,
and evaluation gates.

This branch tests whether the privileged/reference teacher can be compressed
into the deployable causal observation boundary before online exploration.
It must use the existing serious SUGAR-native tactile policy architecture; a
small local MLP or other placeholder is forbidden.

## Failure-Conditioned Release

Teacher influence is allowed only during nominal contact establishment. The
release condition must be computed from direct spatial TacSL and the already
locked causal strategy runtime, never from hidden success, future reference,
or an oracle outcome.

The default phase contract is:

```text
before tactile-confirmed nominal failure:
    alpha_teacher = 1

after failure_closed for the initial nominal strategy:
    alpha_teacher -> 0 over a short frozen release horizon
```

After release:

- original ICM continues to reward not-yet-predicted controllable
  transitions, including novel failures;
- frozen SMP continues to score generic SUGAR G1+box motion naturalness;
- slip, task outcome, repeated-strategy, and safety remain separate external
  ledgers; and
- teacher imitation may be logged but may not gate or scale ICM.

A hard immediate-off control and a short linear-release control must both be
reported. The schedule may not be selected from held-out strategy success.

## Causal and Deployment Boundary

The frozen Refiner teacher legitimately reads its declared nominal reference
only while generating the nominal action prior. The deployable residual/BC
actor reads only the existing causal goal observation and direct TacSL
history. It does not receive:

- future reference states;
- hidden mass, friction, or center of mass;
- reward or success;
- oracle slip;
- contact labels or aggregate wrenches presented as tactile; or
- a future teacher action.

The residual branch requires the teacher at runtime before release. The BC
branch does not. Neither branch proves real-robot transfer.

## Matched Scientific Test

The first controlled comparison has three arms:

1. current causal H2R1 goal policy control;
2. frozen-teacher residual bootstrap;
3. offline-BC causal bootstrap.

Every arm must share:

- official SUGAR action transform, G1, box, motion sources, and physics;
- unchanged global-v3 bilateral R15 mounts;
- direct `20 x 25` pressure and signed two-axis shear history;
- frozen SMP and original-ICM definitions;
- optimizer schedule, environment seeds, physics tuples, update budget, and
  held-out evaluation endpoints; and
- independent world/pressure/shear/GelSight visualization.

The first admission criterion is nominal bilateral-contact retention, not
alternative-strategy success. Only after a tactile-confirmed nominal failure
and teacher release may the experiment count recovery or a topology switch.

## Success and Rejection Gates

The mechanism is rejected if any of the following occurs:

- exact zero residual changes the teacher action;
- teacher/reference information leaks into the deployable causal actor;
- PPO log probabilities are computed for a different action variable than
  the declared residual policy;
- teacher anchoring remains active after the locked failure-release boundary;
- direct spatial TacSL is replaced by a contact proxy;
- ICM is gated by contact, success, slip, or teacher agreement; or
- improvement exists only in training traces and fails the frozen rendered
  contact gate.

Passing the nominal gate still does not prove recovery or an alternative
strategy. Those counts remain separately reported.

## First Executable Evidence

The exact-zero residual invariance gate now passes on a retained compute-node
allocation. A reusable adapter loads the accepted official checkpoint
strictly, reconstructs its official uncorrupted 890-D observation, disables
all gradients, constructs no optimizer, and applies the declared residual
transform. Across four live actions, exact-zero residual tensors return the
teacher actions bitwise while bilateral direct pressure and signed two-axis
shear remain present on all five seed/post records. The synchronized render
and independent `19/19` array audit pass. This does not yet prove PPO
residual-variable integration or a trained residual policy. See
`DOCS/sugar_official_refiner_zero_residual_result_20260725.md`.

The matched offline-BC contract is also frozen and independently audited. It
uses the existing serious SUGAR-native zero-preserving tactile actor, same-time
175-D causal state plus `[4,2,3,20,25]` direct-TacSL history, and the official
29-D pre-action actor output. The split is motion-disjoint at
`74/8/10` train/validation/test motions with `36,519/3,605/4,873` eligible
rows. The `40,124` train+validation rows are now materialized as 82
motion-level shards and every row reconstructs bitwise in the independent
`10/10` audit. See
`DOCS/sugar_official_refiner_bc_contract_result_20260725.md`.

The BC branch has now been executed through its predeclared endpoint and
rejected as the stable closed-loop foundation. The validation-selected serious
actor reaches held-out test MSE `0.006995`, but zero tactile is marginally
better (`0.99817x` the full-TacSL MSE), so the supervised nominal target does
not demonstrate useful tactile conditioning. Full- and zero-TacSL BC both
retain bilateral contact for one live step. Across a fresh four-step gate, the
full-TacSL BC candidate remains bilateral but exceeds the exact-official
control's active-taxel envelope from step 2 and reaches left/right load ratios
`12.78/4.65` at step 4. The independent `23/23` audit correctly confirms the
negative result and the synchronized render shows the expanding footprint.
The BC checkpoint remains a diagnostic/control, not the nominal foundation.
See `DOCS/sugar_official_refiner_bc_bootstrap_result_20260725.md`.

The causal teacher-release implementation also passes its no-learning audit.
It accepts only `initial_strategy_failed`, same-step `failure_closed`, and a
reset mask. Immediate release returns zero; the frozen four-step linear
control returns `0.75, 0.5, 0.25, 0`. An unfailed environment never releases
and only reset rearms the teacher. This does not yet prove the integrated PPO
residual-variable path or recovery.

The integrated residual-variable path now passes. One real
four-environment/24-step rollout starts the admitted serious actor at exact
zero residual mean, samples and log-scores the residual through upstream
RSL-RL PPO 3.0.1, applies
`alpha * official_teacher + 0.05 * residual` in the environment, and
independently reconstructs the native processed joint target. A real
direct-TacSL-confirmed failure releases the teacher to zero; original ICM
continues to score the actual applied action and remains positive after its
bootstrap row. The teacher and official-architecture TinyMDM remain frozen,
one PPO update and one independent ICM update complete, and the combined
checkpoint reloads exactly. The producer/independent audits pass `22/22` and
`16/16`.

The frozen checkpoint camera replay also passes `11/11` producer and `15/15`
independent checks with synchronized world, direct pressure/signed shear, and
official GelSight RGB/depth. It makes the limitation visible: contact is lost
after release and the episode resets; this is an interface result, not learned
recovery. Recovery/alternative counts remain `0/0`. See
`DOCS/sugar_official_refiner_residual_ppo_integration_result_20260725.md`.

The next eight-update H2R1 gate now passes with all five direct-TacSL stress
roles active: 3,780 valid original-ICM transitions, 22 real failure releases,
336 zero-teacher control samples, finite fixed-low-LR PPO, frozen teacher and
TinyMDM, and exact update-1/4/8 combined reload. In a fresh frozen evaluation,
the policy produces two frames of unilateral post-failure palm contact and
moves the box up by at most `0.11071 m`, but never restores bilateral contact
and resets at step 17. This is the first concrete hint of a post-failure
contact attempt, not recovery or an alternative strategy. It motivates the
fixed 64-update segment without changing the reward definition or selecting
on this one seed. See
`DOCS/sugar_official_refiner_residual_h2r1_eight_update_result_20260725.md`.

The fixed 64-update endpoint now closes the longer-training hypothesis as a
negative behavioral result. The serious upstream-PPO/direct-TacSL/SMP/ICM
stack remains numerically sound across 30,386 valid ICM transitions, 1,280
PPO Adam steps, 40 failure releases, and exact update-1/16/64 reload. But its
fresh deterministic rollout is effectively unchanged from update 8: both
fail at step 3, reset at step 17, have only two unilateral post-failure
contact frames, and have no bilateral recontact.

The matched stochastic control provides the important explanation. With the
same training chain and common action seed, update 1 and update 64 both lift
the box about `0.48 m` along the same single-hand-looking trajectory. However,
their sampled residual L2 is `5.305/5.307` while their learned-mean L2 is only
`0.0041/0.0383`; the teacher remains fully active and the tactile failure gate
never fires. That trajectory is therefore raw Gaussian exploration around
the nominal teacher, not learned post-failure recovery. The next idea gate is
no longer “train longer”: separately measure learned policy shift versus
exploration noise over paired fixed seeds, while preserving original ICM's
stochastic discovery semantics. See
`DOCS/sugar_official_refiner_residual_h2r1_64_update_result_20260725.md`.

That paired measurement is complete. Across 20 common environment/Gaussian
streams, update 64 adds two failures/resets, avoids none, and lowers mean final
height by `0.04317 m`; its learned residual mean is still about 159 times
smaller than the sampled residual. The independent exact-ICM follow-up finds
slightly broader frozen-feature support but no reliable increase in frozen
update-1 forward error: paired mean `-0.06389`, 95% interval
`[-0.21107,0.04297]`. This is a negative policy-discovery result, not an
outcome-shaped definition of curiosity. The next bounded idea is a matched
policy-credit ablation with original ICM still scored/learned in every arm,
while its and SMP's contributions to PPO reward are separately disabled.

The policy-credit ablation is also complete. Full, SMP-credit-zero, and
ICM-credit-zero policies all fail the unchanged frozen-ICM discovery gate.
Their zero-teacher exposure is only `3.19%/2.36%/3.23%`, so the active idea
is now a bounded exposure intervention: preserve the full mix and original
ICM, but allow 64 actions after the first raw drop in each episode before the
effective drop termination resumes. Fall, workspace, and timeout safety remain
unchanged. Frozen evaluation has no grace, and recovery/alternative counts
remain `0/0`.

The exposure arm is now also negative for discovery: it creates more
zero-teacher experience but not broader frozen-ICM support, and the rendered
no-grace policy falls after unilateral contact. The adapter's next
architectural boundary is blockwise teacher authority. Releasing the arm block
while retaining leg/waist support is the smallest test of whether the current
failure is caused by coupling manipulation freedom to whole-body collapse.
