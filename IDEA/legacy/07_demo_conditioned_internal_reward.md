# Demo-Conditioned Internal Reward for SUGAR Carrying

## Research Question

Can a learned reward model encourage the G1 policy to preserve the intent and
natural motion of a human demonstration without reading a frame-by-frame
ground-truth trajectory at runtime, while still allowing original-ICM
exploration to discover a different grasp or posture after the demonstrated
two-palm side clamp fails?

The proposed model predicts whether the robot's recent and likely near-future
motion is compatible with a selected demonstration. Its output is a learned
imitation or style reward, provisionally named `r_demo_pred`. It is not the ICM
intrinsic signal, not the frozen SMP score, and not task success.

This is a companion to
`IDEA/06_sugar_smp_tactile_strategy_exploration.md`, not a replacement for the
accepted SUGAR control or the original-ICM discovery mainline.

## Core Proposal in One Line

Learn a causal scorer
`f(live prefix, fixed demo condition) -> predicted future demo mismatch`, then
convert the prediction into a soft imitation reward. Offline labels may use
true future trajectory error; online scoring may not read the future
trajectory. The better semantic variant compares motion in an audited
established representation instead of treating coordinate L2 as the meaning
of the demonstration.

Its role in the current system is precise:

- `r_demo_pred` softly preserves carrying intent and motion quality;
- frozen SMP supplies generic SUGAR G1+box naturalness;
- original ICM rewards not-yet-predicted controllable transitions, including
  novel failures; and
- external objectives decide whether the box was actually lifted and carried
  safely.

This separation can permit a lower squat, asymmetric regrasp, or bottom
support while avoiding arbitrary motion. It helps only if the target and
post-failure schedule tolerate successful alternatives. A predictor trained
only on the nominal side-clamp demonstration would reproduce the same
overconstraint in learned form.

## Plain-Language Meaning and Immediate Insight

The proposal is to learn **how to judge whether the robot still understands a
demonstration**, rather than require the robot to copy every demonstrated pose.
During offline training, a causal live prefix and a fixed demonstration
condition are paired with a future mismatch or semantic-compatibility label.
During policy training, the frozen predictor turns only the live prefix and
demo condition into a soft reward. During ordinary deployment, the policy
needs neither the ground-truth future trajectory nor a reward input.

Its contribution and limitation are equally important:

- it can bridge timing and morphology differences between a human video and
  G1, and provide denser direction than terminal task success;
- it evaluates whether an explored behavior remains compatible with the
  requested carrying intent, while original ICM—not this predictor—drives
  discovery of not-yet-predicted controllable transitions;
- it cannot invent a lower squat, hand switch, or bottom-support strategy from
  nominal bilateral side-clamp positives alone; such alternatives must appear
  in, or be correctly preferred by, the supervision; and
- it therefore guides and selects exploration but does not replace either ICM
  discovery or external success/safety truth.

## Operational Meaning

The proposal can be reduced to one train/runtime/deployment contract:

| Boundary | Available information | Required output |
| --- | --- | --- |
| Offline reward-model training | fixed selected-demo condition, causal robot/box/direct-TacSL prefix, and separately stored future alignment labels | predict future trajectory mismatch or audited semantic compatibility |
| Policy training | fixed selected-demo condition and the live causal prefix only | convert the frozen prediction into a soft demonstration reward |
| Ordinary deployment | actor observations only | actor action; neither the true future trajectory nor the reward is required |
| Optional online planning/adaptation | fixed selected-demo condition and live causal prefix only | frozen reward-model score without future GT lookup |

The selected-demo condition says *which behavior is requested*; it is not a
step-indexed answer giving the robot the correct current or future pose. The
simple target predicts coordinate-level future trajectory loss. The more
general target predicts compatibility in a source-audited motion
representation, with generic frozen SMP naturalness logged separately.

The immediate research value is not a new optimizer. It is a softer bridge
between exact SUGAR tracking and unconstrained exploration:

- `r_demo_pred` supplies direction toward demonstrated carrying intent;
- original ICM independently supplies pressure to discover controllable
  transitions that the ICM model has not yet learned;
- external task and safety terms decide whether the box was actually carried;
  and
- after a failed clamp, the imitation schedule must permit a lower posture,
  asymmetric support, or recontact instead of forcing the original palm path.

This value is conditional on the data. With nominal side-clamp positives only,
the predictor would become a learned version of exact reference
overconstraint. At least one faithful stable release-to-recontact recovery and
successful alternative-strategy positive is therefore an admission
requirement, not an optional improvement.

## Why This Helps

The official SUGAR Refiner receives explicit reference features and is trained
to track a particular processed human motion. That gives a strong target but
also makes a materially different successful solution look wrong. Removing
the reference entirely creates the opposite problem: a goal-based policy can
lift the box in unnatural or unstable ways before the frozen SMP prior and
safety objectives have enough influence.

A learned reward model offers a middle layer:

- during offline training, ground-truth demonstration alignment supplies its
  labels;
- during policy rollout, the model estimates demonstration compatibility from
  a causal observation window and a fixed demonstration command;
- during ordinary policy evaluation, the actor does not consume reward at all;
  if online adaptation is studied, the frozen reward model can still score the
  live prefix without access to future ground truth.

The value is therefore not simply “test without reward.” Standard deployed
policies already act without receiving reward. The useful claim is narrower:
the learned reward can be evaluated online without looking up the corresponding
future reference frames or privileged simulator outcomes.

## A Necessary Conditioning Boundary

An unconditional reward model cannot know which of SUGAR's 922 motions the
robot is intended to follow. It can learn only a generic notion of
SUGAR-like carrying. Following a selected demonstration therefore requires an
explicit command such as:

- a demonstration or motion identity;
- a fixed embedding computed from the complete demonstration before rollout;
- a task/phase embedding derived from the demonstration; or
- a small bank of admissible demonstration embeddings.

This condition is allowed because it states the desired behavior. It must not
silently contain the current ground-truth reference pose, the future robot
state, hidden mass/friction, success, or simulator slip oracle. A no-condition
model is retained only as a generic motion-quality control.

## Two Reward-Model Targets

### 1. Causal Trajectory-Loss Prediction

The direct baseline trains a model to predict a future-window trajectory loss:

```text
L_traj(t:t+H) =
    w_body  * local body-motion error
  + w_box   * local box-motion error
  + w_phase * temporal/phase alignment error
```

The label is computed offline by aligning an executed or perturbed SUGAR
window to its selected demonstration. The runtime input contains only the live
prefix up to `t`, actor-visible task context, and the fixed demonstration
condition. A calibrated compatibility potential can be formed as:

```text
Phi_demo(t) = exp(-alpha * softplus(L_hat_traj(t)))
```

This branch is interpretable and can answer whether the robot is likely to
continue following the selected video. Its main risk is inheriting every
defect of the exact trajectory metric: phase sensitivity, morphology mismatch,
and punishment of a valid alternative strategy.

`Phi_demo(t)` is a prediction of a future outcome, not an observed
instantaneous reward. Repeating it as positive reward at every step can reward
waiting or prolonging the episode. The admitted dense integration is therefore
potential-style:

```text
r_demo_pred(t) = eta_t * (gamma * Phi_demo(t+1) - Phi_demo(t))
```

with a frozen predictor, policy-matched discount, and separately verified
episode boundary. A sparse terminal `Phi_demo(T)` remains the matched control.
Raw repeated `Phi_demo(t)` is an exploit control, not the main method.

### 2. Semantic Motion-Space Compatibility

The more general branch evaluates compatibility in an established pretrained
motion representation. For this workspace the first admitted representation
is the frozen official MimicKit `TinyMDMModel` already trained on audited SUGAR
G1+box windows. The existing ESM/SDS score is a generic SUGAR motion prior; it
does not by itself identify a selected demonstration.

A demo-conditioned semantic reward must therefore compare the live window and
selected-demo condition through an audited official representation or a
faithful established reward-learning method. It may use:

- frozen official TinyMDM denoising energy as a generic naturalness term;
- phase-tolerant distance between live and demo windows in an audited frozen
  official feature/energy space; or
- a separately trained demo-conditioned reward head whose backbone and
  training source are frozen before policy optimization.

The semantic branch should tolerate small timing and pose differences better
than raw trajectory loss. It must not be called an “SMP latent reward” unless
the exact official feature or energy being used is identified and validated.
The public TinyMDM model is primarily a diffusion denoiser, not an automatically
semantic, demonstration-conditioned encoder.

The 2026-07-24 pinned-source audit closes this ambiguity. Official
`TinyMDMModel.forward()` returns a diffusion training loss,
`TinyStableMotionDiTModel.forward()` returns the flattened denoiser prediction,
and `ESM_SDS_loss()` returns per-timestep denoising MSE. The admitted G1+box
checkpoint uses unconditional `DiT`; it exposes no embedding API. Public
`CondDiT` class labels are not selected-demo embeddings and its
`global_hidden_states` argument is unused. Raw normalized `10 x 216` motion
features and arbitrary transformer hooks are therefore not admitted SMP
semantic latents.

The only admitted SMP-derived auxiliary representation is the frozen
per-timestep SDS error vector, explicitly named a stochastic generic energy
descriptor. A separately trained demo-conditioned head may consume it only
after the alternative-positive gate, with a raw-feature ablation and fixed
noise/timestep accounting. That head remains a new reward model and its output
remains `r_demo_pred`, while official SMP remains the separate `r_smp`.

If an established method such as T-REX is selected for reward learning, its
official repository, architecture, training objective, and released assets
must be used faithfully. A small local ranking MLP may not be presented as
T-REX or as serious reward-model evidence.

The pinned official T-REX audit now gives this idea a precise comparison.
T-REX learns one additive state reward from ranked trajectories; partial
trajectory returns are compared with the official softmax cross-entropy
objective. After fitting it needs neither online GT reward nor an online
demonstration, which validates the deployment principle behind an internal
reward model. But it is task-wide, per-state, and not selected-demo
conditioned. Adding a demo encoder or sequence predictor would be a new method,
not official T-REX.

Accordingly, official T-REX is retained as `r_trex_task`, a serious
ranked-reward baseline. It does not replace `r_demo_pred`. Its ranking data
must also contain an admitted alternative success: if every good example uses
the nominal bilateral clamp, T-REX has no supervision that makes another
support strategy good.

Official XIRL is a closer comparison for the semantic version of this idea.
Its TCC representation learns task progress from cross-embodiment RGB videos,
then its released reward embeds the current rendered frame and scores negative
distance to the mean terminal demonstration embedding. It therefore proves
that an offline video-trained semantic reward can run during policy learning
without the corresponding future GT trajectory.

The boundary is still strict. Standard XIRL is task-wide and
current-frame-to-goal; it is not conditioned on a selected full
demonstration, does not score full trajectory compatibility, and has no
robot/box/direct-TacSL sequence input. It may tolerate different paths to a
visually similar goal, but cannot by itself distinguish a stable bottom
support from a visually plausible unsafe shortcut. A selected-demo or tactile
extension is a new project method, not untouched official XIRL.

Accordingly, XIRL is retained as `r_xirl_goal`, an official visual
task-progress control. Its checked source exposes no applicable trained
SUGAR/G1/CarryBox checkpoint. Because XIRL reward fitting and policy-training
inference require RGB, the stopped RGB branch is not reactivated without a new
explicit user instruction. Static source and method compatibility are frozen
in
`DOCS/sugar_demo_reward_official_xirl_compatibility_audit_20260724.md`.

Official RoboCLIP is the more direct comparison to the selected-video semantic
proposal. It freezes an S3D/HowTo100M video-language model, encodes one
selected demonstration and the robot's completed rollout, and returns their
unscaled embedding dot product as a sparse terminal reward. Unlike XIRL's
mean-goal image distance, it conditions on one full demonstration and compares
temporal behavior. This validates the proposal's core intuition without
requiring task-specific reward-model fitting.

The remaining gap is now narrower and more precise. Standard RoboCLIP is an
episode-terminal RGB reward. It has no causal state/direct-TacSL prefix, no
dense per-step prediction, and no load/slip signal. A rolling predictor or
tactile extension is a new project method, not official RoboCLIP. Its
style-transfer behavior can also preserve the nominal side clamp, so
alternative-positive and post-failure relaxation gates remain necessary.

Accordingly, RoboCLIP is retained as `r_roboclip_demo`, the official
selected-video/full-rollout semantic control. The checked parent repository
has no top-level license, and its linked S3D weights/dictionary are not
locally downloaded or hash-pinned. Their official fallback filenames and
sizes are now frozen, but artifact-specific licensing and content identity
remain unresolved. Together with the stopped RGB branch, this keeps runtime
and adaptation inactive. Records:
`DOCS/sugar_demo_reward_official_roboclip_compatibility_audit_20260724.md`
and
`DOCS/sugar_demo_reward_s3d_pytorch_artifact_metadata_audit_20260724.md`.

The S3D dependency now has a more precise identity boundary. The pinned
PyTorch port points to DeepMind's versioned TFHub S3D v1 model, whose official
tutorial specifies `[0,1]` `B x T x H x W x 3` inputs, 32-frame
`224 x 224` preprocessing, video/text signatures, and raw dot-product
similarity. This provides an official S3D backend control, but does not prove
that its variables, embeddings, or reward scale equal the PyTorch artifacts
linked by RoboCLIP.

Therefore every teacher dataset must declare one of two backends:
`roboclip_released_pytorch` or `deepmind_tfhub_v1`. Their scores cannot be
mixed before strict matched-input embedding and raw-dot-product equivalence.
The official Kaggle model API now resolves the TFHub v1 artifact license as
`Apache-2.0` and freezes version ID `1444` plus its exact
four-file/`131,583,009`-byte manifest. It publishes no content hashes, and it
does not resolve the RoboCLIP-linked PyTorch artifacts. Frozen records are
`DOCS/sugar_demo_reward_s3d_backend_equivalence_protocol_20260724.md`
and
`DOCS/sugar_demo_reward_s3d_official_artifact_metadata_audit_20260724.md`,
plus the PyTorch release metadata boundary in
`DOCS/sugar_demo_reward_s3d_pytorch_artifact_metadata_audit_20260724.md`.

The future comparison is now implemented as official-backend glue, not as a
new model: a common-input builder applies the official tutorial preprocessing
with explicit SUGAR frame indices; an independent input verifier re-decodes
every source and requires bitwise tensor equality; one isolated runner
requires the official PyTorch fallback filenames/sizes, content hashes, and
artifact-specific license before strictly loading the pinned source; one
isolated runner uses the official TFHub loader on a complete local SavedModel
tree; and an independent verifier alone compares embeddings and raw rewards.
All five reject missing authorization, licenses, hashes, compute allocation,
or input audit. They remain unexecuted.

Official-artifact acquisition is now a separate fail-closed pre-stage rather
than an informal download command. Its hash-bound request requires explicit
acquisition/model/network authorization, a retained Slurm allocation,
artifact-specific license evidence, exact frozen metadata, ignored output,
streamed byte limits, per-file hashes, cleanup on failure, and atomic
publication. The TFHub/Kaggle four-file path is statically ready but
unexecuted. The RoboCLIP-linked PyTorch path is deliberately unexecutable
because its binary-artifact license remains unresolved. This advances
provenance and reproducibility only; it generates no semantic label and does
not reduce the zero alternative-positive blocker. Record:
`DOCS/sugar_demo_reward_s3d_artifact_acquisition_protocol_20260724.md`.

### RoboCLIP Teacher to Causal State/TacSL Student

RoboCLIP suggests a concrete semantic target for the user's predictor:

```text
offline teacher label:
  y_sem = dot(S3D(selected demo video), S3D(complete rollout video))

causal student:
  (selected demo condition, robot/box/direct-TacSL prefix)
      -> Phi_t = expected terminal y_sem
```

The teacher sees complete RGB videos only during offline label construction.
Its backend identity and the selected-demo embedding generated by that same
backend remain fixed per dataset and checkpoint. The student never sees
rollout RGB, teacher score, future state, future tactile, hidden physics, or
outcome at runtime. It is frozen before policy optimization and provides
either a sparse terminal prediction or the potential-difference feedback
above.

This is not official RoboCLIP: it distills an official terminal visual score
into a new causal state/tactile model. It also does not make the teacher
tactile. Direct pressure/shear can improve early prediction of later semantic
failure, while external slip/lift/safety remain the outcome truth.

The offline supervision bridge is now implemented without selecting or
training a student. For each completed rollout it consumes exactly one
declared-backend 512-D S3D embedding and writes exactly one float32 terminal
raw dot-product target. Dataset splitting is frozen before causal-prefix
expansion; rollout and row identities are retained only in offline audit
fields and are forbidden student inputs. A separate verifier reloads the
official-backend evidence, recomputes every dot product bitwise, and
reconstructs every terminal and prefix row exactly.

The bridge is intentionally fail-closed on semantic support, not just code
correctness. Its admitted dataset must contain nominal success, failure,
release-to-recontact recovery, alternative-strategy success, and
reward-hacking negatives. Faithful recovery and stable alternative-strategy
success currently both have count `0`, so the builder refuses before
embedding consumption and no teacher label exists. This prevents the nominal
side-clamp demonstration from silently becoming a universal reward that
suppresses the very strategy changes this project is meant to discover.

The support decision is now executable rather than a manually asserted
boolean. An independent admission auditor reconstructs the bilateral
direct-TacSL failure, contact-free release/re-arm, later same-episode direct
recontact, changed object-frame palm topology, actual lift, stable goal, and
unchanged safety from the source trace. It requires raw per-hand
`3 x 20 x 25` normal/signed-shear fields, synchronized world and tactile
visuals, and GelSight RGB/depth response evidence. A strategy family is
repeatable only after two distinct rollouts, seeds, and genuine physics
tuples pass. Object geometry remains audit-only and is never relabeled as
tactile.

The evidence-collection boundary below that admission is also executable.
Current C4 cannot be retroactively upgraded: it has direct TacSL and world RGB
but lacks complete per-step goal/safety/palm-center fields and both GelSight
RGB/depth streams. A new unregistered one-environment config adds only the
official dual-R15 camera path and world camera to the same coherent goal task.
The guarded single-pass collector now loads only the admitted frozen
SUGAR-native tactile policy, uses deterministic inference without PPO/SMP/ICM
updates, records every raw goal/safety/action/TacSL field, encodes one world
frame per policy step, and saves raw dual-GelSight event windows. Future
evidence must currently use that one-pass path. Exact initial-state/physics/
action camera replay remains a frozen future contract and is rejected until
its own producer and complete provenance verifier exist.
Images never enter the actor or reward, and open-loop replay never counts as
another policy success. The runner is implemented but has not received
runtime validation or authorization.

The exact static contracts are
`DOCS/sugar_demo_reward_roboclip_causal_distillation_protocol_20260724.md`
and
`DOCS/sugar_demo_reward_s3d_backend_equivalence_protocol_20260724.md`.
The executable offline label boundary is frozen in
`DOCS/sugar_demo_reward_s3d_causal_prefix_teacher_label_protocol_20260724.md`.
Its upstream positive-support admission is frozen in
`DOCS/sugar_demo_reward_strategy_support_admission_protocol_20260724.md`.
The lower collection/readiness boundary is frozen in
`DOCS/sugar_demo_reward_strategy_support_collection_protocol_20260724.md`.
The branch remains unexecuted behind the artifact acquisition/content-hash,
RGB-authorization, and stable alternative-strategy-positive gates. The
RoboCLIP-linked PyTorch artifact license remains a separate blocker.

## Reward and Constraint Ledgers

The integrated system has five primary quantities plus three learned-reward
controls:

1. `r_smp`: frozen official TinyMDM ESM/SDS motion-naturalness score;
2. `r_demo_pred`: learned compatibility with the selected demonstration;
3. `r_icm`: original-ICM pre-update forward prediction error for new
   controllable transitions;
4. external task/slip/recovery/repeated-strategy/safety objectives; and
5. PPO as the policy optimizer.

The additional controls are `r_trex_task`, the official task-wide ranked
reward, and `r_xirl_goal`, the official task-wide visual goal-progress reward.
The third is `r_roboclip_demo`, the official selected-video/full-rollout
semantic reward. They are compared with `r_demo_pred`, not silently combined
with it.

`r_demo_pred` may influence policy learning, but it may not be added to the ICM
forward target, used to gate ICM novelty, or reported as curiosity. A novel
failed attempt can have high `r_icm` and low `r_demo_pred`; that disagreement
is informative and must remain visible in logs.

## Preserving Exploration After a Failed Clamp

If the only selected demonstration uses a symmetric side clamp, a globally
strong imitation reward will actively suppress the desired lower squat,
bottom support, asymmetric regrasp, or torso/forearm support. The reward must
therefore be phase-aware:

- before failure, score whole-body and box compatibility with the selected
  demo;
- during approach, keep enough temporal tolerance to avoid requiring exact
  simulator phase;
- after a tactile-confirmed failed clamp, retain lower-body balance, torso
  naturalness, box safety, and task-intent compatibility;
- reduce or remove exact palm-to-box trajectory matching during regrasp;
- never reduce `r_icm` merely because the new attempt is dissimilar to the
  demonstration; and
- restore stronger demonstration compatibility only after the new contact
  topology has stabilized, if doing so does not force a return to the failed
  strategy.

A cleaner future option is a bank of demonstration conditions containing
multiple valid carry strategies. Until such official data exists, the method
must not pretend that one side-clamp video specifies a bottom-support target.

## Training Data and Causal Contract

Positive and negative windows must be generated from audited official SUGAR
rollouts and declared simulation rollouts:

- positive matched motion/demo windows with temporal offsets;
- same-motion phase shifts and speed changes;
- mismatched official motion identities;
- joint/body or box-motion corruptions already used by the SMP prior audit;
- live nominal, heavy, low-friction, slip, and recovery rollouts;
- successful natural deviations and unsafe/high-error deviations, when
  available; and
- held-out motion identities and physics combinations reserved before fitting.

Every sample records the source motion, alignment rule, label components,
actor-visible input fields, demonstration condition, split, and provenance.
Train/validation/test splits are by original motion identity and physics
combination, not randomly mixed windows from the same rollout.

At inference step `t`, the predictor may read only:

- observations at or before `t`;
- the same non-privileged tactile/proprio/task fields available to the actor;
- reset-valid causal history; and
- the fixed selected-demo condition.

Future executed states, future reference frames, reward, success, hidden
physics, oracle slip, and critic-only fields are forbidden.

## Calibration and Reward-Hacking Tests

Low supervised loss is not enough. Before policy integration the model must
pass:

- ranking and calibration on held-out motion identities;
- phase-shift and speed-warp robustness;
- distinction between matched natural motion and frozen/static shortcuts;
- monotonic degradation under controlled body, box, and contact corruption;
- causal prefix truncation and future-reference leakage audits;
- physics-shift tests for actor-hidden mass and friction;
- disagreement analysis against raw trajectory loss and frozen SMP ESM;
- adversarial or policy-generated high-predicted-reward/low-true-alignment
  searches; and
- frozen-checkpoint reproducibility with exact preprocessing hashes.

After policy training, true trajectory loss is an evaluation-only oracle.
Report both predicted and true alignment to expose reward hacking. A high
predicted reward cannot substitute for rendered task behavior.

## Experimental Controls

Matched policy experiments must include:

- no demonstration reward;
- exact ground-truth trajectory reward during training, labeled non-deployable
  oracle control;
- learned causal trajectory-loss reward;
- frozen generic SMP ESM/SDS only;
- demo-conditioned semantic reward;
- combined SMP + learned demo reward without ICM;
- SMP + original ICM without learned demo reward; and
- SMP + learned demo reward + original ICM with the post-failure schedule.

All controls keep the same official SUGAR action/physics, TacSL observations,
policy optimizer, rollout budget, seeds, and held-out evaluation split.

## Success Criteria

This idea is supported only if the learned reward:

- predicts held-out alignment and ordering without future-reference leakage;
- improves selected-demo following before failure compared with no demo reward;
- avoids materially reducing post-failure strategy diversity relative to
  SMP+ICM;
- does not force repeated return to the failed side clamp;
- remains calibrated under held-out mass/friction and tactile perturbations;
- cannot be trivially exploited for high predicted reward with frozen or
  unsafe behavior;
- remains separately attributable from SMP, ICM, and task outcomes; and
- is supported by synchronized rendered policy behavior, predicted/true
  alignment traces, tactile fields, ICM novelty, and strategy state.

Until those gates pass, the correct status is “a predeclared learned
demonstration-reward hypothesis,” not a solved video-following or sim-to-real
result.

## Current Execution Boundary (2026-07-23)

The idea and its leakage/claim boundaries are now specified, but no reward
predictor has been trained and no policy has been optimized with
`r_demo_pred`. Plan 06's nominal 20-environment/eight-update SUGAR-native
preflight now passes under the separately named zero-preserving fixed-`1e-5`
contract. Its separately locked five-role direct-TacSL H2R1 gate now also
passes with all stress roles active from ICM initialization, raw TacSL
provenance, exact shared-history accounting, and unchanged semantic ledgers.

Plan-07 Stage R0 now passes. The versioned causal/prohibited-input contract,
exact 922-trajectory SUGAR manifest, all 922 exported shard hashes, admitted
200,000-iteration official TinyMDM/EMA, twelve train/validation/test
motion-disjoint TacSL physics sequences, and four causal goal-task replay
schemas were re-audited on an H200. The exact
official SMP runtime path is finite on seeded train/validation/test samples,
but its admitted claim remains an unconditional generic SUGAR G1+box
naturalness energy—not a selected-demo semantic encoder.

The reference portion of R1 passes: all 922 demonstrations have a fixed
32-window official-normalized numeric condition, and 349,736 audit-only pairs
cover matched, phase-shifted, speed-warped, mismatched, frozen, body-only,
box-only, and impossible-rotation labels without crossing original motion-ID
splits.

The live candidate portion now also passes its construction boundary. Nine
frozen-policy rollouts over train motion 6, validation motion 18, and test
motion 29 under nominal/heavy/low-friction physics produce 4,998 strict
ten-step trajectory rows and 180 separately encoded terminal-only failure
rows. Every model-input row contains only a causal actor-visible vector prefix,
reset/valid mask, direct spatial TacSL history, and one fixed numeric demo
condition. Future executed/demo endpoints and body/box losses are stored only
as offline labels. Phase and contact-alignment components remain explicitly
invalid; no weighted reward or reward transform exists.

This resolves the file/schema part of R1 but exposes the decisive data blocker.
All current rollouts fail, only 253/5,178 rows contain direct contact, only 51
contain a slip-start event, and zero occur after the initial strategy is
declared failed. There is no successful nominal or alternative strategy and
only one selected demo per split. A serious predictor trained now would learn
overlapping failure trajectories, not semantic compatibility or post-failure
strategy choice. Predictor and policy training therefore remain unauthorized.

An audit of the accepted official SUGAR Refiner control narrows this blocker.
Its pinned `model_10000` record proves 922/1,000 completed
reference-tracking rollouts, so genuine nominal alignment-positive source
states exist. However, curation removed the raw rollout archives, and the 922
retained processed records contain only body/joint/object state plus binary
proxy contact. They omit the exact goal-task actor vector, applied action,
reset/alive/termination fields, selected reference frame, and direct spatial
TacSL pressure/shear history. A historical 196/256-completion tactile-policy
NPZ likewise retains only aggregate summaries, not causal taxel time series.
Neither source is directly trainable.

That fresh read-only instrumented path is now implemented and bitwise audited.
The exact 175-D goal-task view and direct four-frame TacSL history match the
registered goal task with maximum absolute delta zero, while the frozen
official policy still reads only its original reference-tracking observation.
A four-environment motion-0 diagnostic then produced four exclusive
`trajectory_complete` sequences, 481 transitions, complete causal
actor/action/TacSL records, and native rendered RGB without updating any model.

This removes the source-interface and zero-positive blockers, but not the
training-data blocker. Only 7/1,928 causal observations contain direct contact;
the strongest footprint is unilateral left-palm contact, and only one selected
demonstration is represented. `trajectory_complete` labels nominal demo
alignment only and is not goal-task success, recovery, or alternative-strategy
discovery.

The subsequent exhaustive frozen-control scan changes one important dataset
boundary. Of all 77 official train-split motions, 74 complete exclusively and
therefore provide distinct nominal demo-alignment positives. Among those 74,
39 have any direct first-episode TacSL contact, only motion 42 has any
synchronous bilateral force, and that contact lasts only two observations.
Geometry analysis shows that many missing contacts are sub-millimeter
near-misses, but a train-derived mount-depth correction fails to improve
bilateral validation contact and violates the predeclared active-taxel-fraction
bound. The global v3 mount therefore remains unchanged.

This does **not** mean nominal alignment labels should wait for bilateral
tactile contact. A learned demo-compatibility target asks whether a rollout
follows its selected motion; official exclusive `trajectory_complete` supplies
that nominal label, while spatial TacSL is retained as a separately reported
causal modality. Bilateral direct TacSL becomes a hard requirement only for
claims about tactile-confirmed failed clamps, recovery timing, contact-topology
switching, or tactile-conditioned alternative strategies.

The R1 data gate is consequently split:

1. build a motion-disjoint train/validation/test nominal multi-demo causal
   label source from successful frozen official rollouts, without inventing a
   bilateral-contact requirement; and
2. independently collect tactile-confirmed failed-clamp/recovery sequences and
   verified successful alternative strategies before fitting or integrating a
   predictor intended to guide post-failure strategy discovery.

No predictor or policy training is authorized by the nominal source alone.

The first branch is now complete as a dataset boundary. Frozen official
train/validation/test sources provide 74/8/10 distinct successful motions and
43,341 causal ten-step rows. Every row has a causal `10 x 175` actor prefix,
reset/valid masks, direct four-frame two-hand TacSL, and a fixed numeric demo
condition. Future executed/demo endpoints and unweighted body/box components
are stored offline only. An independent audit reconstructs both components
with maximum absolute difference zero and proves motion-ID split isolation.

This is the direct trajectory-loss supervision proposed by the idea, not yet a
learned internal reward. Phase/contact components, body/box weighting, reward
transform, recovery data, and successful alternative strategies remain open.
The semantic branch is also still only a hypothesis; using the official
TinyMDM normalizer does not make these raw feature distances an SMP semantic
latent reward.

The first R2 paired audit now confirms the central limitation quantitatively.
At 43,717 exactly paired anchors, a different valid official SUGAR motion has
higher unweighted body+box loss than frozen behavior in 98.31% of cases,
body-only corruption in 98.44%, box-position corruption in 97.13%, and an
impossible box rotation in 84.73%. All exceed the predeclared 75% risk bound,
with the same pattern present in train, validation, and test.

Exact trajectory-loss prediction is therefore retained as the interpretable
baseline, but rejected as a strong global post-failure reward. A predictor can
accurately estimate a badly chosen target; causal prediction accuracy alone
does not make the target compatible with strategy discovery. The semantic
branch and post-failure component schedule are necessary rather than optional,
and the explicit alternative-strategy tolerance gate remains open.

A retrospective first-episode audit of the frozen official train source now
separates usable failure negatives from the still-missing recovery data. Of 77
train motions, 40 contain direct spatial TacSL, 36 contain a slip-start event,
35 close a failure, and 33 satisfy a strict causal failure-negative definition
requiring a three-observation direct-contact run, recent slip start, the
initial-failure latch, and spatial pressure/shear at closure.

None of those failures is preceded by the required three-observation bilateral
direct-contact run. Motion 42, the only source with any synchronous bilateral
TacSL, closes a left-contact failure at observation 251 and becomes bilateral
only afterward at observations 256–257. Thirty-five failed sequences later
reach official `trajectory_complete`, but that is reference completion, not
goal-task success or policy-conditioned recovery. The 33 failure negatives are
admitted as source material; bilateral failed-clamp, strict recovery, and
successful alternative-strategy gates remain false. Predictor and policy
training remain unauthorized.

The 33 admitted events are now materialized as fixed 31-observation sequences:
ten causal observations before failure, the failure observation, and twenty
afterward. Actor-visible policy state, complete direct TacSL, and the fixed
numeric demo condition are physically separated from offline event roles,
motion identity, reference frames, and termination provenance. Independent
reconstruction is exact for every input and label. This resolves the
failure-negative serialization task, but it does not convert any row into
bilateral failed-clamp, recovery, or alternative-success evidence.

Authoritative live result:
`DOCS/sugar_demo_reward_live_prefix_future_labels_result_20260723.md`.

Authoritative positive-source compatibility result:
`DOCS/sugar_demo_reward_positive_source_compatibility_result_20260723.md`.

Authoritative fresh frozen-official positive result:
`DOCS/sugar_demo_reward_frozen_official_positive_result_20260723.md`.

Authoritative nominal multi-demo label result:
`DOCS/sugar_demo_reward_nominal_multidemo_labels_result_20260723.md`.

Authoritative exact-loss tolerance result:
`DOCS/sugar_demo_reward_exact_label_strategy_tolerance_result_20260723.md`.

Authoritative tactile failure/recovery boundary result:
`DOCS/sugar_demo_reward_tactile_failure_boundary_result_20260723.md`.

## C3 Source Boundary for Future Predictor Positives

The reward-predictor branch does not create its own recovery labels. Its first
eligible post-failure positive must be produced by the Stage-I C3/C4 causal
protocol. C3 first requires one initial restore of the frozen motion-45 source,
followed by at least three physics-integrated same-episode bilateral direct-
TacSL observations; per-frame state replay is forbidden.

The later mass/friction event is recorded as external curriculum and excluded
from original-ICM/PPO storage around the event. A valid C3 negative requires
the pre-intervention bilateral run plus post-intervention TacSL slip and
failure closure. A valid positive additionally requires the unchanged C4
release, later-attempt, topology-change, lift, and post-lift goal gates.
Therefore C3-P0 clamp persistence alone adds zero predictor rows and does not
authorize `r_demo_pred` fitting.

Authoritative protocol:
`DOCS/sugar_stage_i_c3_bilateral_failure_protocol_20260723.md`.

C3-P0 has now passed the clamp prerequisite in all 20 coherent environments
with three-to-four physics-integrated bilateral observations and no later
state writes. It still contributes exactly zero reward-model rows. The
intervention boundary and magnitudes are now fixed, but fitting remains
blocked until C3-P1 supplies strict failures and C4 supplies strict recovery/
successful-alternative positives. Authoritative result:
`DOCS/sugar_stage_i_c3_p0_bilateral_clamp_result_20260723.md`.

C3-P1 has now executed once at that frozen boundary. Its strict failure counts
for `[no-op, mass/inertia 3x, matched low friction, combined]` are
`[0, 1, 0, 1]`. The full pre-registered intervention-coverage gate is therefore
a no-result because low friction alone produces no strict failure. Environment
1 (mass-only) and environment 3 (combined) nevertheless reconstruct as two
independently valid failure-negative source events. They add no positive:
`r_demo_pred` fitting remains blocked until C4 produces a same-episode
release, later changed-topology attempt, actual lift, and post-lift stable
goal, plus a verified successful alternative strategy. The external property
intervention remains curriculum, never ICM novelty. Authoritative result:
`DOCS/sugar_stage_i_c3_p1_bilateral_failure_result_20260724.md`.

## C4-P0 Consequence for the Internal Reward

The exact C3-to-C4 handoff now passes. Strict failure histories from
environments 1 and 3 enter the official policy/SMP/original-ICM/PPO stack in
the same episode, with the external intervention and all 140 prelude
transitions excluded from learned storage. This establishes a valid future
collection boundary for post-failure positives.

It does not supply a positive yet. In the first 24 stored steps, environment 3
releases and re-arms, but there is no later changed-topology contact, qualified
lift, or goal-stable recovery. The predictor therefore still has two admitted
C3 failure negatives and zero strict recovery/alternative-success positives.
`r_demo_pred` fitting remains blocked. A longer C4-P1 may collect positives,
but it may not reinterpret release alone, low similarity alone, ICM magnitude,
or a dropped box as semantic success.

Authoritative result:
`DOCS/sugar_stage_i_c4_p0_same_episode_handoff_result_20260724.md`.

## C4-P1 Frozen Collection Boundary

The next source attempt is now frozen before execution. It reloads the curated
global-74 learned/RNG state, recreates the exact C3 identity-matrix failure,
and runs 16 complete 24-step updates to the fixed global-90 endpoint. Recovery
cannot stop the run or select an intermediate checkpoint. Only environments 1
and 3 are positive-eligible, and the unchanged release, later direct-contact,
attempt, topology-similarity, `0.10 m` lift, goal-stability, and no-prior-reset
predicates all remain mandatory.

Even a numeric C4-P1 candidate does not immediately authorize predictor
fitting. It must first pass independent world-RGB/direct-TacSL/topology audit
and show a successful alternative strategy rather than a bounce or drop.

Authoritative frozen protocol:
`DOCS/sugar_stage_i_c4_p1_fixed_postfailure_exploration_protocol_20260724.md`.

The fixed endpoint has now completed as a no-result. It adds no reward-model
positive: one eligible branch re-arms, but both original failed-clamp episodes
reset after only 9/11 stored policy steps, before any later contact or attempt
2. The remaining updates are post-reset and predictor-ineligible. Predictor
fitting remains blocked; the next dependency is a per-term termination audit
and, only if justified, a bounded recovery-grace contract.

Authoritative result:
`DOCS/sugar_stage_i_c4_p1_fixed_postfailure_exploration_result_20260724.md`.

## C4-P2/P3 Consequence for Predictor Admission

C4-P2 does not add a predictor row. Its exact fresh-restart replay gate fails,
and its sole-drop result applies directly only to the instrumented replication.
It supports an episode-design diagnosis, not a recovery label.

C4-P3 is frozen as a bounded data-collection opportunity: raw drop is retained
as a causal field while only its effective reset is suppressed for 256 steps
in original failed branches. The grace value, suppressed flag, and termination
provenance are offline metadata and must not enter the live-prefix predictor
input. Even if C4-P3 reaches a later contact, predictor fitting remains blocked
until the unchanged topology, lift, goal-stability, no-reset, tactile, and
rendered-world gates establish a true positive.

C4-P3 reaches no later contact. It lengthens the original failed episodes but
both terminate through unchanged unsafe-fall safety, so it adds zero predictor
positives. This gives the internal-reward idea a concrete dependency: before
training a causal trajectory-loss or semantic motion-latent predictor for
recovery, the dataset needs at least one faithful stable release-to-recontact
positive. Training on current negatives alone would learn failure/termination
separation, not the intended recovery-following reward.

## Official Source Search and Motion-37 Falsification

The all-official source audit initially finds one promising direct-TacSL
candidate. In the stored 77-motion run, motion 37 changes from right-only
contact to left-only contact across 25 no-contact observations, later lifts
`0.590 m`, and completes the official reference. It is the only source to pass
the contact-switch/lift/completion screen.

A preregistered fresh replay then falsifies its use as a stable positive.
Predeclared env 0 has no stable right precursor. Env 3 nearly repeats the
switch but a right-hand touch at observation 196 breaks the ten-frame release;
post-hoc environment selection is forbidden. All four executions still finish
the official trajectory, so this is contact-transition non-reproducibility,
not a broken baseline.

This sharpens the learned-reward lesson: a semantic reward cannot be made
strategy-tolerant merely by mining one accidental contact ordering from a
nominal reference rollout. It needs a deliberately stable, repeatable
multi-strategy demonstration source. Predictor fitting remains blocked.

## Public SUGAR and OmniContact Source Result

The next two official-source paths are now exhausted without adding a positive.
The public SUGAR release does not expose its original RGB-D human-video bank,
source mapping, camera/depth metadata, or video-to-motion preprocessing code.
Its three public CarryBox robot-result videos all show the nominal bilateral
side grasp; the public failure-recovery video is PickBottle, not CarryBox.

The complete 13-file CarryBox subset publicly exposed by the official
OmniContact GitHub repository is structurally auditable but also nominal.
Across 9,811 frames its hand labels contain only bilateral contact or no
contact—zero left-only and zero right-only frames. Motion 12 contains four
adjacent release/recontact structures between repeated carry cycles, but all
four return to the same bilateral topology. The maximum object-local wrist
shift is `0.0371 m`, below the frozen `0.12 m` distinct-geometry threshold.

This result reinforces the core idea rather than weakening it: a learned
semantic reward can tolerate a new strategy only if its supervision contains
or ranks such a strategy. Nominal bilateral demonstrations can train ordinary
following, but they cannot truthfully supervise hand switching or bottom
support. OmniContact binary contact is a semantic annotation, not tactile;
neither source authorizes `r_demo_pred` fitting for recovery.

Authoritative results:

- `DOCS/sugar_public_human_video_source_audit_result_20260724.md`
- `DOCS/sugar_omnicontact_source_compatibility_result_20260724.md`

## GRAIL Boxed-Pickup Metadata Gate

The next official-source candidate is NVIDIA GRAIL's released G1
`pickup_table`/`pickup_ground` data. At the fixed public dataset commit, the
complete boxed-food/drink metadata selection contains 155 files and all pass
exact download-hash and restricted Joblib decoding checks. However, the
decoded metadata contains only object identity and, for tabletop episodes,
table pose/size. It exposes no time-series hand contact, trajectory length, or
faithful source identifier.

The preregistered G0 gate therefore stops before robot/object/video expansion
and contributes zero candidate rows. This is a schema/provenance rejection,
not evidence that the underlying GRAIL motions lack useful behavior. It also
clarifies the data requirement implied by this idea: an official physics
dataset is not enough by name alone; a candidate must expose auditable temporal
behavior and provenance before it can teach semantic tolerance. No GRAIL
SUGAR/TacSL replay or predictor fitting is authorized by this result.

Authoritative result:
`DOCS/sugar_grail_boxed_pickup_source_compatibility_result_20260724.md`.
The complete negative package is independently rehashed under
`grail_boxed_pickup_metadata_gate_negative_20260724`, curation manifest
`94cf8387d1d64bd5fba5e26282a3f6ef2bf5ad8c481a727ff8aa347e34f884fa`.

## Humanoid Everyday Semantic-Topology Evidence

The official Humanoid Everyday screen adds a narrower but positive semantic
result. Its real-G1 football task visibly transports an unsupported object
with both hands; its dumpling-toy task visibly transports an unsupported
object between desks with the right hand while the left arm remains free.
This supports the idea that a semantic reward should recognize a shared
transport outcome across different hand topologies instead of enforcing exact
joint tracking.

It does not solve the positive-data blocker. The objects and scenes differ,
neither sample contains a nominal CarryBox failure followed by an alternative
success, and there is no paired weight intervention, hand switch, recontact,
or bottom support. The public LeRobot conversion also contains no hand
pressure. These rows are auxiliary representation-test sources only; they are
not predictor positives and do not authorize `r_demo_pred` fitting.

Authoritative result:
`DOCS/sugar_humanoid_everyday_carry_source_compatibility_result_20260724.md`.
The independently verified local package has curation manifest
`58f7c3f2563ed5ab9e7e75d8da2563ca06d33dfb4ecd5b3bfcbba7e87cc79d7e`.

## OmniRetarget Box-Carry Candidate

OmniRetarget is the first newly screened public source that declares a
substantial G1+box trajectory release rather than only task names or project
videos: the fixed MIT-licensed `robot-object` archive contains about three
hours of OMOMO-derived 43D G1/base/object qpos. Its advertised diversity makes
it a serious candidate for alternative box-carrying semantic examples.

It is not yet a positive. The declared NPZ schema has no action, torque,
contact, force, mass intervention, TacSL, reward, or physics-success field.
The complete archive must first pass provenance/schema audit, exact-asset
kinematic screening, and full-motion rendering. Any survivor still needs fresh
SUGAR physics and direct dual-palm TacSL replay before it can supervise the
predictor.

Frozen protocol:
`DOCS/sugar_omniretarget_boxcarry_source_compatibility_protocol_20260724.md`.

The complete OR0/OR1r1 audit now passes all 1,826 files, 323,812 frames, and
312 original motion families. Exact upstream G1/rubber-hand/largebox geometry
then nominates 57 original motions, and OR3 renders and independently decodes
every full motion. Manual full-timeline inspection retains three complementary
physical-replay candidates:

- `sub12_largebox_078_original`: sustained bottom-plus-side support;
- `sub12_largebox_020_original`: sustained unilateral bottom support; and
- `sub16_largebox_010_original`: long bottom-dominant transport.

This is useful evidence for the learned-reward idea because the same audit
also produces semantic boundary controls. In particular,
`sub12_largebox_039_original` passes a numerical low-posture screen but the
full timeline shows terminal placement, not sustained low carrying;
release/recontact-only candidates likewise need not establish a stable new
support strategy. A useful reward predictor must therefore distinguish
task-phase semantics and mechanically viable support from isolated pose or
distance cues.

The frozen family follow-up confirms that all `15/15` official augmented
siblings retain the three selected roles and that every original reproduces
OR2 exactly. This strengthens geometric consistency but not data diversity:
the augmentations remain derived kinematic variants, not independent
successful alternatives.

None of these rows is a predictor positive. The source largebox is
approximately `0.47115 x 0.45873 x 0.40790 m` with upstream mass `0.1 kg`,
whereas official SUGAR CarryBox uses a different `small_box` asset with a
`0.5 kg` task mass. The released trajectory is not conditioned on object
weight, so it cannot demonstrate weight-triggered strategy choice. The exact
source-to-SUGAR joint permutation, official Holosoma conversion path, asset
retargeting requirement, and fresh SUGAR/direct-TacSL replay gate are frozen
in
`DOCS/sugar_omniretarget_or4_sugar_tacsl_bridge_protocol_20260724.md`.
Fixed-source inspection confirms that released `human_joints` is the official
retargeter's preprocessed human-motion field and that the official API
accepts separate demo/current object point arrays. The official loader,
however, constructs them from the same 100 seeded surface samples and only
applies `smpl_scale`; the optimizer assumes indexed point correspondence.
This proves a faithful largebox-reproduction seam, not an arbitrary
largebox-to-SUGAR-smallbox mapping. No official cross-mesh correspondence path
has been found, and independently sampling or normalizing two meshes is not
admitted. Runtime also remains unexecuted because upstream pins NumPy `2.3.5`
while the approved prebuilt SUGAR environment has NumPy `1.26.0`, and
compute-node dependency installation is prohibited.

The same-day official release follow-up closes source-package acquisition:
the exact `holosoma-retargeting==0.1.0` wheel is cached by release SHA, passes
full ZIP integrity, and reproduces the three audited official-main source
blobs. It is not a self-contained runtime and remains uninstalled. Inspection
of all 66 visible official refs and all 28 Python files in the wheel still
finds no indexed largebox-to-smallbox correspondence. This strengthens the
blocker rather than authorizing a local mapping.

The exact environment surface is now also closed. Official setup uses Python
3.11, `numpy==2.3.5`, `Miniconda3-latest`, an unpinned pip upgrade, and
fourteen unpinned direct dependencies. The repository `uv.lock` contains only
three header lines and no package resolution. A future environment therefore
must be labeled an official-wheel-compatible locally frozen resolution, not
an author-published lock. Its two-prefix resolver/offline-rebuild protocol is
frozen but explicitly unauthorized and unexecuted in
`DOCS/sugar_omniretarget_holosoma_isolated_runtime_protocol_20260724.md`.

The wheel also contains the raw `sub3_largebox_003.pt`, height table,
largebox, G1, and combined MuJoCo assets that produced a released 196-frame
target in the public archive. The first runtime test can therefore be a pure
official-input/official-entry-point round trip with no adapter at all. That
gate is frozen in
`DOCS/sugar_omniretarget_holosoma_largebox_roundtrip_protocol_20260724.md`;
execution still waits for an approved isolated runtime.

The entire comparison side is now prepared without weakening that boundary.
After the target-blind official runner closes its output, an independent
array/FK verifier may first open the released target. A second stage reuses
the frozen OR2 exact-mesh and OR3 official-asset implementations to require
identical topology traces and render all 196 frames side by side; a final
independent decoder must verify the video and twelve endpoint-inclusive
filmstrip frames before the round trip can be marked complete. These are
kinematic runtime-equivalence gates only and still add zero predictor-positive
rows.

Before any retarget run, the exact SUGAR-smallbox asset audit was separately
frozen by file identity and method and now passes. Its one collision-tagged
composed mesh contains 50,004 vertices with exact local dimensions
`0.40004499 x 0.54605001 x 0.53614101 m`; both runtime configs use unit scale
and `0.5 kg`, and the three-view render is valid. This resolves geometry
ambiguity only; vertices are not automatically the uniform surface points
expected by the official retargeter.

Correspondence result:
`DOCS/sugar_omniretarget_official_object_point_correspondence_audit_20260724.md`.
Release/runtime result:
`DOCS/sugar_omniretarget_holosoma_release_runtime_audit_20260724.md`.
Asset result:
`DOCS/sugar_omniretarget_or4_m1_smallbox_asset_result_20260724.md`.

Official SMP representation result:
`DOCS/sugar_demo_reward_official_smp_representation_audit_20260724.md`.
It resolves the representation boundary but adds no semantic model and no
predictor-positive row.

Official T-REX result:
`DOCS/sugar_demo_reward_official_trex_compatibility_audit_20260724.md`.
It admits a task-wide ranking baseline, not the selected-demo predictor, and
adds no applicable checkpoint or training authorization.

Official XIRL result:
`DOCS/sugar_demo_reward_official_xirl_compatibility_audit_20260724.md`.
It admits an official semantic visual-progress control and confirms GT-free
reward inference, but standard XIRL is task-wide,
current-RGB-to-mean-goal, non-tactile, and not the selected-demo sequence
predictor. Its RGB path remains inactive.

Official RoboCLIP result:
`DOCS/sugar_demo_reward_official_roboclip_compatibility_audit_20260724.md`.
It is the closest selected-video semantic control found, but remains a sparse
terminal RGB/full-rollout score with no causal state/direct-TacSL prefix. Its
unlicensed parent source plus unresolved linked-artifact license/content
hashes keep it inactive. Its official fallback filenames and sizes are
nevertheless frozen.

S3D backend identity result:
`DOCS/sugar_demo_reward_s3d_backend_equivalence_protocol_20260724.md`.
The versioned DeepMind TFHub v1 endpoint, official interface, Apache-2.0
artifact license, version ID, and four-file manifest are identified. No
artifact is acquired and no PyTorch/TFHub embedding or raw-reward equivalence
is claimed. The fail-closed official-backend extractors and independent
verifier plus the official-spec input preparation/re-decode audit are
implemented, but no decoded video, input manifest, embedding, or score exists.
Artifact metadata record:
`DOCS/sugar_demo_reward_s3d_official_artifact_metadata_audit_20260724.md`.
PyTorch artifact metadata record:
`DOCS/sugar_demo_reward_s3d_pytorch_artifact_metadata_audit_20260724.md`.
Artifact acquisition protocol:
`DOCS/sugar_demo_reward_s3d_artifact_acquisition_protocol_20260724.md`.
Its static audit passes, but no request, artifact body, model load, RGB decode,
embedding, reward, label, or training result exists.

S3D terminal-teacher to causal-prefix label bridge:
`DOCS/sugar_demo_reward_s3d_causal_prefix_teacher_label_protocol_20260724.md`.
The builder and independent verifier pass static audit, including
split-before-prefix expansion, one exact float32 raw dot product per rollout,
audit-only identity fields, and exact row reconstruction. The semantic-support
gate remains closed at zero faithful recoveries and zero stable alternative
successes; no request, embedding consumption, label, student, reward, or
policy result exists.

Strategy-support admission:
`DOCS/sugar_demo_reward_strategy_support_admission_protocol_20260724.md`.
The independent auditor and static contract pass without consuming taxel
data. They formalize exact same-episode recovery and repeatable
alternative-topology evidence, while retaining the authoritative C4-P3
negative result: two releases/re-arms but zero later direct contacts, zero
recoveries, and zero alternative successes. No request or admission output
exists. Every future episode now binds its own passed collection audit and
fresh single-pass record; trace, raw camera archive, world video, rollout
identity, seed, and read-back physics must all come from that same run.

Strategy-support collection:
`DOCS/sugar_demo_reward_strategy_support_collection_protocol_20260724.md`.
The unregistered camera-only coherent-task config, guarded frozen-policy
single-pass collector, and independent field/equivalence auditor pass static
checks. The collector binds every-step world RGB plus raw-frame hashes and
event-window direct-TacSL/GelSight evidence. Current C4 remains incompatible;
the authorized runtime probe is now complete and independently audited.

Its source-interface audit now additionally pins the exact goal-policy and
pure-discovery termination terms, RSL-RL checkpoint wrapper, manager reset
ordering, declared-physics readback, and seven official R15/TAXIM runtime
assets. Native TacSL shear is `[environment, taxel, xy]`; the new collector
transposes it before `[hand, channel, 20, 25]` archival. The frozen generic
C4 shear archive did not, so those signed-shear maps are excluded, while its
normal-based zero-recontact result and the official policy/ICM tactile path
remain valid. The collection auditor now also requires the exact run record
and recursively binds the original authorization request, checkpoint,
declared/read-back physics, seven runtime assets, fourteen source hashes,
every output, and the 16-observation minimum. This is still static preflight,
plus runtime evidence validation, not collected supervision.

Runtime result:
`DOCS/sugar_demo_reward_strategy_support_collection_result_20260724.md`.
The collector now self-binds the unchanged global-v3 mount offsets and the
official contact-free GelSight baseline. Exact motion-45 state replay verifies
bilateral taxel-resolved normal force and signed shear. A faithful official
compliant-contact backport plus a camera-fresh `0.25--4 mm` normal-press sweep
now passes independent bilateral pressure/shear and GelSight RGB/depth
response audit. Exact nominal replay remains below the optical load threshold,
and the fresh Stage-H probe still has 23 observations, zero direct contact,
no failure/later attempt/goal success, and unsafe termination. The sensor
response result contributes no predictor row: it has neither policy behavior
nor selected-demo semantic supervision. Contact-seeded H2R1 and C4-P3
global-90 now add real frozen-policy behavior, but both discard one hand on
their first action and have zero post-action bilateral contact/load frames.
These are useful contact-retention negatives, not recovery or alternative
positives. Recovery and repeatable-alternative counts remain `0/0`, so
`r_demo_pred` fitting stays closed. Record:
`DOCS/sugar_tacsl_gelsight_contact_load_policy_preflight_result_20260724.md`.

Consolidated method selection:
`DOCS/sugar_demo_reward_method_selection_result_20260724.md`.
No audited official source matches the complete selected-demo + dense causal
state/direct-TacSL interface. `r_demo_pred` is therefore an explicitly new
project method, with architecture selection and fitting still behind the
alternative-positive gate.

Authoritative OR0--OR3 result:
`DOCS/sugar_omniretarget_boxcarry_source_compatibility_result_20260724.md`.
The compact independently verified evidence package has curation manifest
`0dbb267962c4b78d359b8d941ac86023c6cd960106f285227a6facda0de4416a`.

## CHORD-Inspired Predictor Target (2026-07-27)

CHORD suggests that the future internal predictor should compare functional
object-contact effects, not only raw hand/body trajectory error. A possible
new target is future demo/task-compatible object-wrench sufficiency learned
from completed rollouts, with current causal state and direct-TacSL history as
student input.

This does not admit a model or training run. The prediction would remain a
potential/value estimate, not observed instantaneous reward, and would require
the existing discount-matched potential-difference and leakage audits. Exact
CHORD is reference-conditioned and not test-time-GT-free; the proposed causal
student would therefore be a new project method. It remains blocked by the
same positive recovery/alternative-support gate as the other semantic
predictor variants.

Audit:
`DOCS/sugar_chord_contact_wrench_guidance_audit_20260727.md`.
