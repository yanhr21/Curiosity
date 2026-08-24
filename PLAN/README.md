# Plan index

## Current priority: demo following

### Executable official-skill router: complete

The physically valid shared-checkpoint baseline is complete. One checkpoint contains the exact
released CarryBox and KickBox Tracker actors plus a learned router that reads only the frozen
798-D causal selected-demo condition. Temporal validation routes Carry45/Kick21 with `100%`
accuracy and positive margin; both official experts remain parameter-exact.

Matched frozen physics passes: Carry condition in CarryBox gives `18/20` successes, `0/20` falls
and `0.43204 m` mean maximum lift; Kick condition in KickBox gives `20/20` successes, `0/20` falls,
`0.09892` foot-contact fraction and `1.06092 m` planar displacement. Student actions equal the
selected released expert exactly. Within each domain, the condition swap uses bitwise-identical
initial robot/object state and prefix action.

The counterfactuals define the boundary. CarryBox routed to the Kick expert leaves Carry and makes
foot interaction, but later emits a raw action of `5.87e11`, violates the fixed `25` envelope and
falls in `9/20`; it is rejected. KickBox routed to the Carry expert remains stable but still kicks
because the task-specific generator is unchanged: foot contact drops `0.09892 -> 0.02838` and
planar displacement drops `1.06092 -> 0.72466 m`. This is a causal two-skill router, not arbitrary
demo imitation.

The preceding full-510D shared-MLP route is now a retained negative diagnostic. Offline BC reached
MSE `0.00682` but failed Carry `0/20` with `20/20` falls. Three serious DAgger stages over official
visited-state labels improved transient behavior but ended at `6/20` Carry and `14/20` falls. Do
not extend that MLP by adding optimizer steps. The next method must preserve executable skill
stability while reducing task-generator coupling, for example a shared skill prior/latent and
state-aware safe transition policy trained on multiple official skills.

### Reference-aware matched pair: complete, partial single-seed shift

The from-scratch seed/action-seed `161587/161588` pair is complete. Both arms use the same
CarryBox45 Refiner teacher, frame-197 causal phase, 20 environments, startup-physics readback,
optimizer, reward weights and 64-update budget; only CarryBox45 versus KickBox21 selected reward
differs. Both training proofs pass `65/65` checks, and update-32/update-64 frozen evaluation uses 20
identical physics profiles per arm.

At update 64, correct/unrelated maximum lift is `0.69332/0.69666 m`, bilateral-contact fraction is
`0.83335/0.83447`, lifted fraction is `0.61142/0.61644`, lifted-transport fraction is
`0.94514/0.94141`, ground-transport fraction is `0.05486/0.05859`, and orbit rate is
`0.37166/0.37547 rad/s`. Physical falls are `0/20` in both arms. Three of four predeclared
directions move toward Kick21 at update 64; update 32 also records `3/4`, with a different missing
direction. Both arms nevertheless retain bilateral Carry and foot-box contact remains near one
frame per episode. This is selected-demo-conditioned behavior change, not complete semantic
following.

The independent-seed replication `161589/161590 -> 171589` is complete. At update 64 it repeats the
same missing direction and nearly the same three deltas as seed161587: lifted transport
`-0.00372/-0.00386`, ground transport `+0.00372/+0.00386`, and orbit
`+0.00381/+0.00369 rad/s`. Update 32 does not replicate (`3/4` versus `1/4`). Both seeds retain
bilateral Carry and near-zero foot interaction.

The fixed 4x feedback-scale overfit diagnostic is also complete. It changes only `eta` and
`reward_clip`, yet update-64 directions fall from `3/4` to `1/4`; ground-transport and orbit effects
reverse, and unrelated foot contact does not increase. The unrelated cumulative feedback reaches
`-48.64` versus baseline `-11.99`, so simple signal magnitude is rejected as the demonstrated
bottleneck. No further scale is planned.

The first actionable deployment experiment is complete. One serious SUGAR actor/checkpoint trains
on both selected-demo conditions. Before every action it receives a 798-D condition made from the
frozen 11.386M predictor's own selected-demo projection, causal representation, mismatch,
uncertainty, risk, phase and readiness. Future labels and GT trajectory error remain forbidden.
Frozen evaluation restores the same update-64 checkpoint and exact initial state twice, changing
only Carry45 versus Kick21 conditioning. Residual actions change by mean/max absolute
`0.01319/0.37943`; the unrelated condition moves in `3/4` predeclared directions, but both rollouts
remain bilateral Carry. Maximum lift is `0.68367/0.66868 m` and physical falls are `0/20` versus
`1/20`. This establishes same-policy demo-conditioned modulation, not complete semantic following.

The fixed action-direction topology diagnostic is also complete. It keeps that same serious
SUGAR actor family, frozen 11.386M causal predictor and CarryBox45 Refiner execution baseline. For
both official Carry45 and Kick21 state sequences, each causal state is paired with both demo
conditions: the correct target is exact-zero residual, and the unrelated target is the released
`Kick21 Tracker action - Carry45 Tracker action`. Future actions are training labels only and are
absent from frozen evaluation. Exactly 3000 optimizer steps reduce dataset MSE from `1.48955` to
`0.10764`; critic and tactile encoder remain unchanged. Swapping only the condition on the same
state changes residual actions by mean/max `0.98783/12.1281`.

The same step-3000 checkpoint is then evaluated from the same Carry initial state and physics over
20 profiles. Correct preserves Carry (`0.68792 m` mean lift, `0.84006` bilateral-contact fraction,
`0/20` falls). Unrelated leaves the Carry solution (`0.00267 m` lift, zero bilateral contact,
`0.99764` ground-transport fraction and increased foot-box contact), but falls in `15/20` profiles.
All four predeclared behavior directions move toward Kick21, yet the video shows an unstable leg/
orbit response rather than successful KickBox imitation. This proves that explicit action-direction
supervision can create a same-checkpoint behavior split; it does not establish semantic following.

That next matched experiment has now been completed by the official-skill router above. It proves
executable selection only at the two released skill endpoints; task-generator coupling and safe
cross-skill transition remain open. No reward-scale sweep, toy teacher or future-action input to
the deployed actor is permitted.

The first causal selected-demo experiment is complete:

1. CarryBox45 teacher-only, zero-residual evaluation passes bilateral contact and 5 cm lift in all
   20 profiles;
2. both learned arms use the same fixed CarryBox45 teacher, seeds, initialization, physics,
   optimizer, reward weights and 64-update budget;
3. only the selected reward demo differs: CarryBox45 versus unrelated KickBox21;
4. frozen outcomes are `16/20` versus `18/20` success, with two physical root-height falls per arm;
5. selected-demo reward changes the checkpoint and rollout, but correct-demo superiority and
   semantic obedience are not established.

The direct predictor-independent behavior audit is complete. It uses only robot/object state and
rigid hand-contact force from the existing traces; predictor loss, demo reward and training loss are
excluded. CarryBox45 lifts the box by `0.7639 m` and completes `81.41%` of its horizontal box path
while lifted, whereas KickBox21 never crosses the `0.05 m` lift threshold and performs all box
motion at ground level.

Under the common CarryBox45 teacher, both learned arms remain Carry-like. Compared with the correct
arm, the KickBox21-reward arm has `+0.0350` lifted-frame fraction, `+0.0323` lifted-transport
fraction, `-0.0323` ground-transport fraction and `-0.0050 rad/s` orbit rate. Thus none of the four
declared semantic directions is observed. The two transport fractions are complementary views, not
independent tests. The reward changes the Carry solution, but semantic demo following is not
established.

The serial three-seed repeat is complete. Training/action seed pairs are `161581/161582`,
`161583/161584`, and `161585/161586`; frozen evaluation seeds are `171581`, `171583`, and `171585`.
For unrelated minus correct, lifted-frame deltas are `+0.0350/+0.0179/-0.0058`, lifted-transport
deltas are `+0.0323/+0.0132/-0.0277`, and orbit-rate deltas are
`-0.0050/-0.0294/-0.0115 rad/s`. Thus the Kick-like direction occurs in only `1/3` seeds for
lift/transport and `0/3` for orbit. Seed161585's partial `3/4` shift does not replicate. The final
multiseed verdict is `stable_semantic_following=false`.

### Teacher-authority learnability diagnostic：已完成

The fixed-physics overfit pair resumed both seed161581 update-64 endpoints and executed exactly 64
new updates. The common official CarryBox45 teacher was annealed from `1.0` to a nonzero `0.25`
floor in both arms; task, initialization, optimizer, physics and frozen evaluation stayed matched.
Both endpoint proofs and frozen evaluations passed their execution checks.

The behavioral gate failed decisively. In 20 frozen profiles per arm, correct and unrelated both
have zero bilateral-contact fraction, zero lifted fraction and zero lifted-transport fraction;
both have zero foot-to-box contact, and `0/4` Kick-like directions are observed. Episodes terminate
after about `0.88 s`. This is behavioral collapse after reducing teacher authority, not semantic
separation, so the schedule is not repeated across seeds.

### Current branch: contact/event internal reward redesign

The official reference-corpus audit now covers 100 CarryBox and 99 KickBox motions. Binary source
contact is retained only as a reference-event proxy. Carry contact frames select a hand as the
nearest named effector in `95.46%` on average; Kick contact frames select a foot in `99.78%`.
Carry median lifted-moving fraction is `40.85%`; Kick is exactly zero. Thus the desired semantic
targets exist in the reference data.

The actual-rollout target corpus and predictor admission gate are now complete. The corpus contains
100 CarryBox and 99 KickBox source motions, each with 700 same-clock frames. Actual left/right
hand/foot contact is the exact 0.1 N threshold of named body-to-box filtered force; event duration
is reset-bounded and motion regime is episode-relative. No reference binary label is used as an
actual target. Carry median bilateral contact/longest hand event/lift are `0.3293`, `4.60 s` and
`0.490 m`; Kick median foot contact/longest foot event/lift are `0.0414`, `0.22 s` and `0.0066 m`.

The first 510-D predictor passed ordinary held-out gates but failed deployment and reward semantics.
The goal actor exposes a 121-D causal core, and the target builder independently minimized over 32
demo windows at every frame. That free alignment allowed a trajectory to keep matching an easy
static phase. Fixed Carry45/Kick21 scoring consequently preferred Carry45 even on held-out Kick
rollouts. Removing uncertainty did not fix it; directly recomputed targets isolated phase alignment
as the cause. This version is rejected evidence, not the active reward.

The corrected dataset binds every target to causal normalized episode phase. With that single
scientific change, direct validation/test targets prefer Carry45 for Carry and Kick21 for Kick. The
active V3 remains a 6-layer, 384-D serious Transformer and has `11,386,010` parameters. It reads the
past `10 x 121` deployable core, a fixed numeric selected demo and one normalized phase scalar; no
future event, task ID or motion ID is an input. Seed271303 freezes epoch 20. Validation/test MAE is
`0.1771/0.1560` against constant `0.2803/0.2566`; zero-demo is `0.2945/0.2766`, permuted-demo is
`0.2018/0.1761`, median Spearman is `0.677/0.694`, and all 12 model gates pass.

Validation-only calibration gives nominal-90% coverage of `97.13%/97.77%` on validation/test, with
`91.86%` minimum test target coverage. The fixed Carry45/Kick21 scale audit passes all ten checks:
both held-out splits prefer their matching task, and held-out Carry receives positive mean feedback
under Carry45 and negative mean feedback under Kick21. Dense feedback is
`eta * (exp(-calibrated_event_risk) - train_baseline)`, with `eta=0.2427623309` and clip
`0.1431077421`; it is not potential-difference shaping because the purpose is to change the policy
objective toward the selected demonstration.

This established a causal deployable reward on the official-Tracker corpus before optimization,
not policy obedience or transfer to Refiner-policy rollouts.

Implementation admission is complete: the frozen event scorer consumes the exact 121-D policy
prefix online, uses a reset-bounded causal clock, waits for ten states, adds dense feedback after the
unchanged SMP/original-ICM calculation, and exposes no future event label to the actor. The matched
launcher uses seed/action-seed `161587/161588`; both arms use the same CarryBox45 fixed teacher and
differ only by `selected_option=correct/unrelated`. Frozen evaluation is predeclared at updates 32
and 64 with 20 matched physics profiles per checkpoint.
Formal GPU admission also passes for both arms on H200: correct selects CarryBox45 row 37 and
unrelated selects KickBox21 row 97; both report 121-D input, clock-phase alignment, frozen eval mode,
zero trainable parameters, no environment creation and zero policy updates.

The subsequent online gate corrected an experiment-composition bug: `explicit_zero_control` had
zeroed policy tensors but still instantiated the dual-R15 TacSL scene. The matched demo experiment
now uses the original no-TacSL SUGAR G1/CarryBox scene. On fresh H200 job257815/server54, a minimal
Isaac Sim canary and then both correct/unrelated 24-step real-environment smokes pass. They exercise
the frozen Refiner, actor, SMP, original ICM, phase-aware reward and storage with zero optimizer
updates and unchanged parameters. Earlier cross-node `ERROR_DEVICE_LOST` runs are runtime failures,
not model evidence. Under the identical unoptimized rollout, correct and unrelated have exactly
matched actions/base reward but different ready-step demo reward (`0.04013` versus `0.01734`),
confirming online selected-demo sensitivity.

The common-teacher physical prerequisite is now admitted separately. A corrected no-TacSL frozen
evaluator runs exact-zero residual for 400 steps over 20 nominal profiles; every profile has
bilateral contact, `0.6854--0.7224 m` maximum lift and no physical robot fall. This removes teacher
inability and accidental TacSL construction as explanations for a future matched-policy result.

The active no-TacSL runner now records the exact official-SUGAR startup materials, object mass,
inertia and COM in every formal proof, and frozen evaluation restores that record. Repeated online
smokes show exact correct/unrelated physics equality as well as exact action/base-reward equality;
the only online difference remains the selected-demo feedback.

The pre-optimization reward-to-gradient admission passes for both arms. On the same collected
trajectory, the runner recomputes GAE and normalized advantages after removing only selected-demo
feedback, then differentiates the exact clipped PPO actor surrogate without calling an optimizer.
Correct changes return/advantage by `0.45412/0.25342` and the actor gradient by L2 `0.07804`;
unrelated changes them by `0.23489/0.16923/0.04430`. Parameters and optimizer counters remain
unchanged. The selected-demo signal therefore reaches the policy learning direction.

The fixed-one teacher does not mask the student. In both admitted arms the exact execution formula
is `teacher + residual` with coefficient/scale `1.0/1.0`; the sampled 29-D residual reaches the
ActionManager raw input exactly. Inverse joint scale/offset round-trip error is only `4.77e-7`
against the existing `2e-6` float32 tolerance. The student therefore retains full residual authority
while the common teacher preserves a matched Carry baseline.

Runner probes now fail closed on a separate machine-readable result. A zero process return code is
insufficient because Isaac shutdown can mask an inner Python failure.

### Historical reset-zero phase-event result: complete, negative

The historical matched experiment ran both arms serially for exactly 64 updates. Both 65-check
training proofs pass; update-32/update-64 checkpoints are finite and reload exactly. The evaluator
restores the same 20 startup-physics profiles into both update slices, expands only the invariant
fixed-one wrapper state from 20 to 40 environments, and gates the replica closest to the recorded
source origin at the original `2e-6` action tolerance. Frozen evaluation seed is `171587`.

At update 64, correct/unrelated maximum lift is `0.69453/0.69419 m`, bilateral-contact fraction is
`0.83348/0.83252`, lifted fraction is `0.61292/0.61214`, lifted-transport fraction is
`0.94127/0.94219`, and physical falls are `0/20` in both arms. The unrelated arm moves in only one
of four predeclared Kick-like directions; update 32 moves in two of four, with all behavioral deltas
very small. Both synchronized videos show a Carry solution. The result therefore rejects semantic
demo following under this reward and budget; it does not reject the fact that selected reward
changes gradients and checkpoints.

The frozen score itself fails an online semantic sanity check. On the correct update-64 Carry
trajectory, mean predicted mismatch is `0.96986` to Carry45 and `0.89087` to Kick21. Exact 121-D
prefixes have now been recollected from both arms, and a scorer-only ablation reproduces every old
runtime signal within float32/model tolerance before changing one variable: the initial phase.

With the deployed reset-zero clock, `Kick risk - Carry risk` is approximately
`-0.080~-0.082`; only `30.1%~30.6%` of ready frames and `0/20` profiles prefer Carry in each of the
four arm/update blocks. Starting the first episode from the restored CarryBox45 reference frame
`197` changes the margin to `+0.324~+0.328`; `85.7%~86.1%` of ready frames and `20/20` profiles in
all four blocks prefer Carry. The online inversion is therefore caused directly by the nonzero
reference state being paired with a zero phase clock. The scorer, runner and frozen evaluator now
initialize phase from the same reset reference frame.

A separate Tracker-to-Refiner state shift remains measurable. Official Tracker test has normalized
state `mean|z|/p95/p99 = 0.668/1.923/2.882`; the current Carry rollouts are approximately
`1.035/3.212/5.420`. Phase correction is nevertheless sufficient for the necessary Carry-domain
gate. It does not establish independent Kick-domain transfer or policy semantic following, and the
old policies were trained under the wrong phase clock. The corrected policy pair described at the
top of this file has now run and supersedes this historical result for the active causal question.

The missing official Generator/Tracker Kick direction has now been tested without policy training.
The exact motion-disjoint predictor test split `9/19/.../89` supplies nine 700-step, 121-D online trajectories;
all nine contain foot-to-box contact and at least 1 cm planar object displacement. With the deployed
fixed-650 phase clock, frozen Kick21 risk is lower than Carry45 risk by `0.06508` on average,
`50.50%` of ready valid frames prefer Kick21, and `8/9` motion-level means prefer Kick21. Motion29
prefers Carry45 and is retained as a counterexample. A source-duration-normalized clock gives `9/9`
only as an evaluation diagnostic and is not a deployed input. This passes the declared
official Generator/Tracker Kick transfer gate; it does not establish Refiner-plus-residual Kick transfer.

### Completed matched diagnostic and next replication

The corrected scorer gates are complete. On job258074, both 24-step online smokes use initial step
197 and execute zero optimizer updates; mean ready reward/risk is `+0.04804/0.31539` for Carry45
and `-0.00338/0.65776` for Kick21. The frozen Carry evaluator passes all four arm/update blocks with
`20/20` profile preference and `+0.324~+0.328` mean margins. Proceed serially:

1. retain the completed corrected zero-optimizer online and frozen Carry evidence;
2. retain the completed motion-disjoint official Generator/Tracker Kick gate and its motion29
   counterexample;
3. retain the released-artifact boundary: this workspace provides the official KickBox Generator
   and Tracker but no frozen Kick Refiner checkpoint; do not substitute a toy policy;
4. retain the completed reference-aware seed161587 pair and its `3/4` update-32/update-64 behavior
   shifts without upgrading them to semantic-following evidence;
5. run exactly one independent-seed matched replication before changing reward weights, teacher
   authority or update budget;
6. keep selected-demo SMP out until official TinyMDM passes an independent semantic-extension gate.

### Expected behavior, not just reward score

For the correct Carry demo, the expected interaction is: approach the box, establish bilateral hand
contact, lift above `0.05 m`, transport it predominantly while lifted, and keep the robot/object
geometry coupled. For the unrelated Kick demo, genuine conditioning must instead cause an
observable shift toward ground-level object motion, more motion around the box, and a different
body/object contact mode, even though the external task still asks for CarryBox completion. Task
success and demo adherence are reported separately.

The exact source timelines make this concrete. CarryBox45 has one continuous binary hand-contact
proxy interval from frame `245` to `541` (`4.90--10.82 s`), crosses the 5 cm lift threshold at frame
`286` (`5.72 s`), peaks at frame `350` (`7.00 s`) and remains lifted through frame `508`
(`10.16 s`). KickBox21 has 14 intermittent binary foot-contact proxy intervals, its closest named
end-effector is the right ankle at frame `190` (`3.80 s`), and it never crosses 5 cm. These binary
labels describe reference contact roles only; they are not tactile force. The learned unrelated
arm must measurably move toward this event structure, not merely change predictor score.

This standard follows the behavior-level structure used in physics-based interaction imitation:
[DeepMimic](https://arxiv.org/abs/1804.02717) separates imitation from the task objective;
[PhysHOI](https://arxiv.org/abs/2312.04393) explicitly evaluates body-object contact topology;
[InterMimic](https://arxiv.org/abs/2502.20390) checks object deviation, joint-object relations and
required-contact duration; and [CHORD](https://nvidia-isaac.github.io/video_to_data/chord/) argues
that object-centric contact wrench should measure how contact moves the object. Therefore policy
loss, predicted reward or task success alone is not a demo-following verdict.

The official selected-demo TinyMDM gate is also complete. Exact selected-clip identity passes, but
the independent CarryBox96/KickBox22 semantic extension fails. Policy integration is not
scientifically supported by this result, and an arbitrary Transformer hidden state must not be
called an official SMP latent.

Training, frozen evaluation, evidence paths and exact commands are consolidated in
[`DOCS/reproducibility.md`](../DOCS/reproducibility.md). The policy entrypoint remains
`scripts/sugar/demo_following/run_matched_state_predictor.py`; reference-event feasibility is
reproduced with `scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py`.

## Frozen historical plan

[`15_online_patch_tactile_mass_adaptation/plan.md`](15_online_patch_tactile_mass_adaptation/plan.md)
retains the exact online 54-patch tactile and sudden-mass protocol. All audited source defects were
fixed, but no valid corrected matched Z/P/PS comparison was completed. The line is frozen and is
not the current execution queue.
