# Plan index

## Current priority: demo following

### Executable full official-skill router: complete

The physically valid shared-checkpoint baseline is complete. One checkpoint contains the exact
released CarryBox and KickBox Tracker actors plus a learned router that reads only the frozen
798-D causal selected-demo condition. Temporal validation routes Carry45/Kick21 with `100%`
accuracy and positive margin; both official experts remain parameter-exact. Runtime selection now
routes the corresponding released Generator together with the Tracker, because the first 36
dimensions of the 510-D Tracker observation are task-specific Generator commands.

Matched frozen physics passes: Carry45 in CarryBox gives `18/20` Carry, `0/20` falls and
`0.43204 m` mean maximum lift; Kick21 in KickBox gives `20/20` Kick, `0/20` falls, `0.09892`
foot-contact fraction and `1.06092 m` planar displacement. Within each domain, the condition swap
uses bitwise-identical initial state, post-prefix state and prefix action.

The complete-pair counterfactuals define the boundary. In the same CarryBox scene, selecting
Kick21 now produces `19/20` Kick, `0/20` Carry and `0/20` falls; maximum raw action is `5.1712`.
The earlier Tracker-only swap reached `5.87e11` and fell in `9/20`, so joint Generator+Tracker
routing fixes that causal failure. The reverse BIGBOX/KickBox-to-Carry45 transfer reaches only
`8/20` Carry and raw action `68.437`, so it is rejected. This is an executable endpoint skill
router on the compatible SMALLBOX scene, not arbitrary-demo imitation or solved cross-asset
transition.

The preceding full-510D shared-MLP route is now a retained negative diagnostic. Offline BC reached
MSE `0.00682` but failed Carry `0/20` with `20/20` falls. Three serious DAgger stages over official
visited-state labels improved transient behavior but ended at `6/20` Carry and `14/20` falls. Do
not extend that MLP by adding optimizer steps. The next method must preserve the stable SMALLBOX
Carry/Kick result while learning an official-skill prior/latent and state-aware safe transition
across object geometry, target pose and initialization. No toy latent model is allowed.

Three matched `2 x 2` audits now localize that reverse failure. Asset × motion context is a
crossover: matched Carry-small and Kick-big contexts pass while the crossed pairs fail. Holding
Carry context fixed, SMALLBOX passes with either small or `1.5x` big nominal mass, whereas BIGBOX
fails with either mass; geometry is sufficient and mass is not. Holding physical context fixed,
Carry initialization passes with either Carry or Kick target pose, whereas Kick initialization
fails with either target; changing the goal alone is insufficient. The next safe transition must
therefore react to causal state/geometry compatibility, not select by target or mass label.

Two parameter-free transition gates have now been rejected. A synchronized shadow
Generator+Tracker was first made exactly equivalent to the direct released pair: on the compatible
CarryBox-to-Kick21 route, all 650 command, action, object-state and robot-state frames are bitwise
equal and both produce `20/20` Kick with no fall. On BIGBOX-to-Carry45, however, the current-action
envelope triggers only after the state has left both experts' stable distributions; the candidate
reaches `2.17e5`, the fallback reaches `3.85e5`, and the fall is not prevented. The selected
Generator's own released min/max normalizer also fails as an early OOD separator: in the first 100
frames the incompatible route has lower outside-range fraction than the compatible route
(`0.00132` versus `0.00625` mean). Do not tune either threshold.

The causal transition-risk predictor has now been implemented and tested. It is a serious
11.012M-parameter, 6-layer, 384-D Transformer over the exact past `10 x 539` deployable prefix:
510-D official Tracker observation plus the current 29-D released Carry candidate action. Dataset
profiles are disjoint between train/validation, and test uses disjoint seed/context traces. Future
fall/contact/action-invalidity defines the profile label only and never enters inference. A fixed
eight-profile overfit passes `5/5` at 500 steps. The frozen formal checkpoint selects threshold
`0.715` using validation first-50 frames only; held-out test AUROC is `0.7430`, balanced accuracy
`0.6536`, probability gap `0.2655`, and Brier `0.2258` versus prevalence baseline `0.2331`.

That offline pass does not transfer to safe endpoint switching. In the matched BIGBOX online test,
the first 50 frames and candidate actions are exact between direct and fallback arms. Nine causal
samples latch `10/20` profiles to the official Kick pair at frame 49. The composed arm then records
one physical fall, leaves the action envelope and reaches its first non-finite Tracker transition
at frame 447. The failing environment has risk `0.8885 > 0.715` and was already using fallback;
therefore this is not a classifier false negative. Executing Carry for 50 frames and abruptly
switching to Kick can place even the official fallback outside its stable distribution.

The earliest available anchor-9 audit is also complete. Its threshold `0.84` is selected only on
validation anchor 9. Held-out AUROC `0.7239`, balanced accuracy `0.6553` and probability gap
`0.3564` show that frames 0--9 contain ranking signal, but test Brier `0.2773` is worse than the
prevalence baseline `0.2331`; the predeclared calibration gate fails. Do not launch an online
anchor-9 hard switch or weaken the gate. Replace hard endpoint switching with a learned causal
transition/recovery controller trained from official endpoint rollouts; preserve both exact
released skills and do not substitute a toy latent or hand-written world model.

### Fixed Carry-9 to Kick recovery: complete, locally positive but saturated

The first continuous recovery diagnostic executes one released Kick alignment step followed by
nine live released Carry Generator+Tracker steps before each student episode. The prefix is online
PhysX and contributes zero PPO transitions. The student is tensor-exact initialized from the
released Kick Tracker and retains the official `510 -> 512/256/128 -> 29` actor, 890-D privileged
critic and repository BCPPO. A frozen released-Kick action-mean anchor, fixed `1e-5` learning rate,
`0.05` exploration std and no observation corruption keep the fixed diagnostic matched to frozen
evaluation.

At update 64, the matched 20-profile frozen pair is structurally valid and starts from bitwise-
identical robot/object/joint/observation tensors. Baseline and trained are both `20/20` Kick with
zero falls. Training changes mean planar displacement `0.17136 -> 0.18634 m`, foot-contact fraction
`0.0632 -> 0.0674`, and per-frame reward `0.072629 -> 0.073203`. This proves a small local recovery
improvement while preserving the released skill, but there is no success-rate headroom. The
attempted 128-update extension is invalid after rare timeout-only parallel outliers contaminate the
critic at update 67; only the finite update-64 checkpoint is retained.

Next, run a no-training prefix-length frontier with the frozen released Kick pair. Choose one
predeclared length where all inputs/actions remain finite but Kick success is below the admitted
criterion, then run the same baseline/update-64 matched recovery comparison. Only after that fixed
frontier succeeds should geometry and seed generalization be added.

### Prefix frontier and prefix-41 recovery: complete, physically negative

The frozen seed181630 sweep covers `9/17/25/33/41/49/57/65/73/81/89/97`. No point satisfies both
an upright handoff and fewer than 10/20 safe Kick outcomes. Prefix 41 is retained separately as the
maximum upright failure boundary: minimum handoff root height is `0.6680 m`, all traces are finite,
safe Kick is `14/20`, and falls are `6/20`. Prefix 49 is not admitted because its minimum handoff
root height is `0.6476 m`.

At prefix 41, the unconstrained update-64 policy increases displacement but changes falls
`2 -> 3`; the physical-invalid-penalty arm holds falls at `2 -> 2` but leaves safe success at
`17/20 -> 17/20`. Both are negative under the physical-outcome contract. Stop BCPPO/reward-scale
iteration on this topology. The next implementation must preserve both released endpoint skills
while adding a serious shared skill prior and state-aware transition objective.

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
6. retain this phase-predictor result as historical; the later task-wide conditional TinyMDM gate
   and online diagnostic below supersede the old single-clip SMP restriction.

### Official conditional TinyMDM: prior admitted, absolute reward rejected

The official MimicKit audit now uses all available Carry/Kick motions with motion IDs disjoint
across train/validation/test. The exact representation is the released `compute_disc_obs` contract
plus the 15-D box state, `10 x 216` at 50 Hz. Independent Carry/Kick TinyMDM checkpoints correctly
classify all `19/19` test motions at motion level, but are rejected for online comparison because
their separately learned normalizers and energy scales are not shared.

The admitted model is one official `CondTinyStableMotionDiTModel`: 2,836,864 parameters, two class
labels, one normalizer and 50,000 training iterations. It classifies `19/19` motion-disjoint test
motions; correct-condition window preference is `96.998%` on Carry and `92.678%` on Kick. On actual
prefix41 recovery traces, its Kick energy also ranks every observed pre-fall event above same-clock
safe profiles. A training-data-only transform anchors median matching-class reward to 0.5.

The first policy diagnostic keeps the released Kick teacher and actor initialization, executes one
Kick alignment plus 41 live Carry steps before PPO, and trains correct Kick-class versus wrong
Carry-class arms with the same seed171632, 64 updates, safety penalty and `0.5/0.5` task/prior
weights. The online smoke proves exact offline/online feature agreement (`2.38e-7` maximum error),
finite rewards, private scorer RNG and no future labels. Frozen seed181632 gives the same `16/20`
safe Kick and `3/20` falls in both arms. Wrong Carry produces more foot contact (`0.0762` versus
`0.0540`) and displacement (`0.17009` versus `0.14705 m`). This establishes causal condition use
without correct-condition physical benefit.

Do not repeat the absolute occupancy reward with another weight or update count. It is near-zero on
states far from the Kick manifold, while the wrong Carry condition supplies dense reward for
remaining stable in the transitional Carry-like state. The next fixed diagnostic uses the same
admitted official checkpoint and a matched-noise causal progress/transition objective. It must
reward improvement between consecutive causal windows, preserve the released endpoint teacher,
and exclude future outcomes from deployment.

### Matched-noise progress: repeatable condition effect, no robust physical advantage

The fixed progress reward scores consecutive causal windows with the same private diffusion noise
and uses the selected-class normalized-loss decrease. It passed the zero-optimizer online gate and
was then run without weight, teacher or update changes for training seeds `171632/171633/171634`
and disjoint frozen seeds `181632/181634/181636`.

Across 60 profiles per arm, correct Kick versus wrong Carry totals `52 vs 50` safe kicks and equal
`5 vs 5` falls. The mean correct-minus-wrong deltas are `+0.01492 m` net displacement,
`-0.00847` foot-contact fraction, `-0.02102 m` planar path and `+0.00096 m` maximum root-height
loss. Net displacement improves and foot contact decreases in every seed, while fall, root-height
and task-reward directions are inconsistent. Only seed171632 passes the per-seed physical-advantage
rule. The admitted conclusion is repeatable condition-dependent behavior without seed-robust
physical benefit; do not call this general demo following or repeat a weight/update sweep.

The next fixed objective is a causal contrastive transition margin: change in
`loss(alternative) - loss(selected)` across adjacent windows with matched diffusion noise. It keeps
the same official checkpoint and online feature contract while removing generic motion progress
that helps both semantic classes.

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

The old selected-clip TinyMDM gate remains a useful memorization counterexample, but it is
superseded by the motion-disjoint shared conditional model above. The shared prior is admitted;
its first absolute online reward is not. An arbitrary Transformer hidden state must still never be
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
