# Global Agent Rules

## 1. Absolute ban on human authorization gates; current priority: executable demo-conditioned topology

- Hard user rule, reaffirmed 2026-08-25: no in-scope workflow may require, request or wait for human
  authorization. This includes approval flags, confirmation prompts, sentinel files, environment
  variables, manual checkpoint admission, permission to evaluate, permission to start the next
  predeclared stage or seed, and repeated requests for the user to say "continue".
- Execute the documented active plan autonomously through training, endpoint inspection, frozen
  evaluation, rendering, documentation and the next predeclared matched stage. A scientific check
  must produce a machine-readable pass/fail decision and the agent must act on it; it may never be
  converted into a human decision gate.
- Safety, exact process control, non-overwrite rules and scientific validity checks remain required,
  but they are automatic execution constraints rather than authorization gates. Stop only at a
  completed scientific objective or a genuine external blocker that cannot be resolved in scope.
- Any code or document that makes progress depend on a manual approval, confirmation, sentinel or
  repeated user response is a defect and must be removed. This rule supersedes conflicting wording
  elsewhere in the repository.

The tactile/mass-adaptation line in
`PLAN/15_online_patch_tactile_mass_adaptation/plan.md` is frozen while demo following is the active
queue. Do not interleave new tactile training, evaluation or scale tuning with that queue.

The predeclared three-seed demo-following repeat and the seed161581 teacher-floor learnability
diagnostic are complete. The latter annealed the common CarryBox45 teacher from `1.0` to `0.25` but
collapsed both arms: no bilateral hold, no 5 cm lift, no foot-to-box contact and `0/4` Kick-like
directions. Do not repeat that schedule across seeds. The first event predictor used a 510-D Tracker
prefix and per-row free selection among 32 demo windows. It is rejected for deployment: the actor
exposes a 121-D causal core, and free-window matching lets every frame jump to an easy static demo
phase. Do not restore that target or call its passing MAE a valid reward.

The frozen predictor is seed271303: an 11,386,010-parameter, 6-layer, 384-D phase-aware
causal Transformer over the exact past `10 x 121` deployable core, a fixed numeric demo and a
causal normalized clock phase. Its training, calibration and fixed CarryBox45/KickBox21 bidirectional
reward-scale gates pass on the official Generator/Tracker corpus.

The phase-aware dense runtime is connected to the serious SUGAR rollout integrator. The historical
seed161587 matched pair is complete at 64 updates with the same CarryBox45 teacher and physics; only
the selected demo differs. Frozen update-32/update-64 behavior remains nearly identical Carry in
both arms (`2/4` then `1/4` predeclared Kick-like directions). Do not claim semantic following.

The old frozen scorer failed transfer on these actual Refiner-plus-residual rollouts because a state
restored at CarryBox45 reference frame `197` was paired with a phase clock starting at zero. An exact
121-D scorer-only audit reproduces the old runtime and then changes only this offset. All four
correct/unrelated x update-32/update-64 blocks move from `0/20` to `20/20` Carry-preferring profiles;
the mean `Kick risk - Carry risk` changes from approximately `-0.081` to `+0.326`. The scorer,
training runner and frozen evaluator must initialize the first causal clock from the restored reset
reference frame. Do not restore the zero-phase behavior.

Tracker-to-Refiner state distribution shift remains measurable, but phase correction alone passes
the necessary Carry-domain semantic gate. The corrected online gates are now complete on retained
H200 job258074: correct and unrelated both execute 24 real environment steps with zero optimizer
updates, unchanged parameters and exact initial episode step `197`. Mean ready-step reward/risk is
`+0.04804/0.31539` for Carry45 and `-0.00338/0.65776` for Kick21 on the identical unoptimized Carry
rollout. Corrected frozen evaluation also passes for both arms at updates 32/64; all four 20-profile
blocks prefer Carry45, with mean `Kick risk - Carry risk=+0.324~+0.328` and
Carry-preferred frame fraction `85.71%~86.12%`.

The frozen predictor has also passed a motion-disjoint
official Generator/Tracker Kick-domain gate on source motions `9/19/.../89`: all nine rollouts contain
foot-to-box contact and at least 1 cm planar object motion; under the deployed fixed-650 clock the
mean `Kick risk - Carry risk` is `-0.06508`, `8/9` motion profiles prefer Kick21, and the
Kick-preferred ready-frame fraction is `50.50%`. Motion29 remains a counterexample, so this is a
Tracker-domain transfer result, not a universal scorer claim. Refiner-plus-residual Kick transfer
has not been tested, and the existing Carry policy pair was trained under the wrong clock. The
released-artifact audit confirms that the official Kick inference path is the Generator/Tracker
pair and no frozen Kick Refiner checkpoint is available. The necessary scorer gates are therefore
complete at their declared reproducible scope.

The first from-scratch reference-aware matched pair is now complete at update 64. Both arms pass
`65/65` training checks and the frozen evaluator restores identical physics over 20 profiles per
checkpoint. At both update 32 and 64, the unrelated Kick21-reward arm moves in `3/4` predeclared
semantic directions, but both arms still execute a bilateral Carry solution and actual foot-box
contact remains approximately one frame per episode. This proves a behavior shift under the
selected reward; one training seed does not establish semantic following or replication.

The independent-seed replication `161589/161590 -> 171589` is now complete. At update 64 it repeats
the first seed's exact `3/4` direction pattern: unrelated has less lifted transport, more ground
transport and more orbit, while lifted-frame time does not decrease. The paired deltas are nearly
identical across seeds (`-0.00372/-0.00386`, `+0.00372/+0.00386`, and
`+0.00381/+0.00369 rad/s`). Update-32 behavior does not replicate (`3/4` versus `1/4`). Both update-64
arms remain bilateral Carry, foot-box contact stays near one frame per episode and physical falls
are zero. This establishes a reproducible small behavior shift at update 64, not Kick interaction
or complete semantic following.

The fixed 4x feedback-strength diagnostic is complete. Both arms pass `65/65` training checks and
matched frozen evaluation, but update-64 semantic directions fall from baseline `3/4` to `1/4`.
The unrelated-minus-correct ground-transport and orbit effects reverse to `-0.02801` and
`-0.03546 rad/s`; foot-contact advantage is also negative (`-0.00535`). Unrelated cumulative demo
feedback grows from `-11.99` to `-48.64`, while predicted loss improves only `0.00320` and behavior
remains Carry. Signal magnitude alone is not the bottleneck. Do not run another scale or describe
the 4x result as improvement.

The first shared-checkpoint actionable-conditioning experiment is complete at update 64. One
official SUGAR actor was trained across ten Carry45 and ten Kick21 selected-demo environments while
the common teacher, task and physics remained CarryBox45. Its 798-D causal condition is built from
the frozen 11.386M predictor's own selected-demo projection, causal representation, predicted
mismatch, uncertainty, risk, phase and readiness; future/GT events never enter the actor. A
zero-optimizer smoke proves that swapping only the demo changes the actor input and hidden state and
changes the exact PPO surrogate gradient.

Frozen evaluation loads the same `policy.pt` twice from an identical initial state and changes only
Carry45 versus Kick21 conditioning. The residual actions differ (mean/max absolute difference
`0.01319/0.37943`), and the unrelated condition moves in `3/4` predeclared directions, but both
conditions remain bilateral Carry. Mean maximum lift is `0.68367/0.66868 m`; physical falls are
`0/20` versus `1/20`. This proves same-policy demo-conditioned modulation, not semantic switching or
complete Kick following. Do not increase reward scale or claim success from this single seed; the
next matched experiment must address contact-topology generation under the same shared-policy rule.

The fixed action-direction topology diagnostic is now complete. It reused the existing serious
SUGAR `512/256/128` actor, frozen 11.386M causal predictor and official Carry45/Kick21 Tracker
corpus. For each Carry and Kick causal state, correct and unrelated conditions were both present;
the residual labels were exact zero and `Kick21 Tracker action - Carry45 Tracker action`. Future
actions were training labels only and never entered the deployed actor. Exactly 3000 actor-only
optimizer steps reduced MSE from `1.48955` to `0.10764`; critic and tactile encoder remained
bitwise unchanged.

Frozen evaluation loaded the same step-3000 checkpoint and identical Carry initial state twice,
changing only the selected-demo condition over 20 profiles. Correct preserved Carry with
`0.68792 m` mean lift, `0.84006` bilateral-contact fraction and `0/20` falls. Unrelated left the
Carry solution with `0.00267 m` mean lift, zero bilateral contact, `0.99764` ground-transport
fraction and increased foot contact, but fell in `15/20`. The videos show unstable leg/orbit
behavior, not a successful Kick. This proves that official action-direction supervision can create
a strong same-checkpoint behavior split; it does not prove semantic demo following.

Do not extend the fixed Carry Refiner residual diagnostic or run another reward-scale sweep. The
full-510D shared-MLP follow-up and its three-stage DAgger diagnostic are also complete and negative:
offline MSE does not preserve closed-loop stability. The admitted executable baseline is one
checkpoint with parameter-exact released Carry/Kick Tracker experts and a learned router over the
frozen 798-D causal selected-demo condition. Runtime must route the matching released Generator
together with the Tracker: its 36-D command is part of the 510-D Tracker input. Never restore a
Tracker-only counterfactual and describe its OOD action as demo following.

The complete-pair frozen result is asymmetric and must be reported exactly. In the identical
SMALLBOX/Carry scene, Carry45 produces `18/20` Carry and Kick21 produces `19/20` Kick, with zero
falls in both arms; the joint route reduces the old Carry-to-Kick raw-action maximum from `5.87e11`
to `5.1712`. Matched BIGBOX Kick21 produces `20/20` Kick with zero falls. BIGBOX-to-Carry45 reaches
only `8/20` Carry and raw action `68.437`, so it remains rejected. This proves compatible-scene
selection of two released endpoint skills, not arbitrary-demo following, a continuous skill latent
or cross-asset transition. Future action labels remain training-only; no toy policy, teacher or
model may replace official SUGAR components. The next serious method must preserve the passing
SMALLBOX switch while learning a faithful skill prior and safe state-aware transition across object
geometry, target pose and initialization. Its stages run continuously without approval flags,
sentinels, manual checkpoint gates or repeated user confirmation.

The cross-context decomposition and two parameter-free safety audits are complete. BIGBOX geometry,
not its `1.5x` nominal mass or target pose alone, is sufficient to break Carry45. The online shadow
Generator+Tracker is now bitwise equivalent to the direct pair for all 650 compatible-route frames.
Nevertheless, current-action magnitude is a late signal: the candidate and fallback reach
`2.17e5/3.85e5` after the state is already invalid. The released Generator normalizer is not an
early OOD gate either; failed Carry-on-BIGBOX has lower first-100-frame outside-range fraction than
successful Kick-on-SMALLBOX (`0.00132/0.00625`). Do not tune these thresholds. The next serious
method is a causal transition-risk predictor trained across official multi-context rollouts, with
future failure used only as a training label and held-out contexts reserved for frozen evaluation.

Both arms subsequently passed the formal inner-runner admission on retained H200 job257762. The
probe starts Isaac Sim/Vulkan, validates the full protocol and loads the frozen correct Carry45 or
unrelated Kick21 model, but creates no environment and executes zero PPO updates. The allocation is
held after the probes. This is stronger execution-readiness evidence, still not training evidence.

The next online-runtime audit found that the old `explicit_zero_control` composition still
constructed the dual-R15 TacSL scene even though its tensors were zero. Demo-only controls now use
`NoTactileGoalRobotEnvCfg`, the original SUGAR G1/CarryBox scene with no VisuoTactileSensor assets.
Do not restore the TacSL scene in a demo-only arm or claim zero sensor reads from zeroed tensors.

The formal 24-step, zero-optimizer rollout smoke now passes for both arms on the fresh H200
job257815/server54. Each arm executes the original no-TacSL SUGAR scene, frozen Refiner, actor,
SMP, original ICM, phase-aware event reward and rollout storage for 24 online steps while leaving
all policy/ICM parameters and optimizer counters unchanged. Correct uses CarryBox45; unrelated uses
KickBox21. This is online integration evidence, not policy-training or demo-obedience evidence.

The earlier job257762/server60 and job257794/server45 GPUs entered Isaac Sim 5.1
`ERROR_DEVICE_LOST`; stale Kit descendants were found and killed by their exact recorded PGIDs, but
minimal `SimulationApp` canaries still failed afterward. A fresh-allocation canary passed with the
system NVIDIA ICD `/etc/vulkan/icd.d/nvidia_icd.json`, and the matched runner now uses that same ICD.
After any Vulkan device loss, do not reuse that GPU for Isaac evidence in the same allocation;
retain the allocation as required and move the Isaac gate to a fresh GPU.

The shared CarryBox45 teacher-only prerequisite also passes on job257815. The evaluator now uses
`NoTactileGoalRobotEnvCfg` rather than the old zero-tensor TacSL scene, writes nominal object/robot
physics directly through PhysX and requires an explicit no-tactile-scene proof. Across 20 nominal
profiles and 400 control steps, the exact-zero residual has maximum lift `0.6854--0.7224 m`,
bilateral rigid contact for `153--156` frames and zero physical robot falls. Do not reuse the older
teacher-only result lacking the scene proof; the admitted directory name is
`teacher_only_gate_no_tactile_v2`.

The no-TacSL scene still inherits official SUGAR startup randomization. Every formal phase-event
training proof must therefore record exact per-environment object/robot material tensors and object
mass, inertia and COM under `no_tactile_startup_physics`; frozen evaluation restores those values
before rollout. The corrected correct/unrelated online smokes on job257815 record bitwise-identical
startup physics, actions and base rewards while retaining the selected-demo reward difference.
Never evaluate a phase-event checkpoint whose training proof lacks this physics record.

Both arms also pass a zero-optimizer reward-to-gradient admission on job257815. The same stored
rollout is evaluated with total reward and with only selected-demo feedback removed; PPO returns,
normalized advantages and the exact clipped actor-surrogate gradient all change. Correct has actor
gradient delta L2 `0.07804`; unrelated has `0.04430`. Policy parameters and optimizer counters stay
unchanged. This proves the reward reaches the learning signal, not that an optimized policy follows
the demo. Probe success must be read from the separate machine-readable result; never accept process
return code alone because Isaac shutdown can mask an inner exception.

The same online gate proves that `fixed_one` does not suppress the student residual. Both arms use
the exact action `teacher + residual` with teacher coefficient and residual scale both `1.0`; the
nonzero 29-D residual reaches the ActionManager raw input exactly. The inverse joint scale/offset
round trip has maximum float32 error `4.77e-7`, below the frozen `2e-6` tolerance. Do not reinterpret
fixed teacher authority as zero student authority.

Routine audits, dataset builds and the active matched pair proceed through the documented queue.
Corrected online Carry, frozen Carry and official Generator/Tracker Kick gates have passed. Run the
one new phase-corrected pair now; do not wait for another user confirmation.

The completed same-teacher experiment fixes CarryBox45 as teacher in both arms and changes only the
selected reward demo, CarryBox45 versus KickBox21. Across training seeds 161581/161583/161585,
the predeclared Kick-like direction appears in only one seed for lift/transport and zero seeds for
orbit rate. This proves policy change, not correct-demo superiority or semantic obedience.

For the demo goal task, physical fall means relative robot-root height loss `>=0.35 m`. Do not use
reference-anchor posture or `root up-z < 0.5`: the admitted CarryBox45 source legitimately crouches
to root up-z about `0.191`, recovers and carries.

## 2. No degraded placeholder models

- Never present a hand-written toy MLP/VAE/Transformer/world model as progress on T-Rex, VQ-VAE,
  SMP or another serious released method.
- Use official repositories, released checkpoints and faithful architectures first. Write only
  adapters and glue needed to connect them to project data.
- If official code or weights are unavailable or incompatible, report the blocker. A simplified
  diagnostic must be labelled diagnostic and cannot be called a faithful implementation.
- When Plan 15 becomes active again, it retains the serious SUGAR policy, official Tracker warm start,
  frozen Refiner teacher, repository BCPPO and anatomical patch Transformer. No offline tactile
  replay or taxel-CNN substitute is allowed.

## 3. Demo-following evidence contract

- The early motion45/motion96 result compares two CarryBox demonstrations and is not an unrelated
  demo experiment.
- The archived 1216-update experiment changed both teacher and selected demo, and its goal task
  still penalized tactile contact. It is diagnostic history, not an active result.
- The current frozen internal reward predictor is a serious 11,386,010-parameter causal Transformer
  over body, box position, box rotation, box velocity, four-limb contact mismatch, event duration
  and motion regime. It reads only the past 10-frame 121-D actor core, the selected numeric demo and
  causal normalized phase. It is not an SMP-latent predictor.
- The official MimicKit TinyMDM currently provides a shared generic motion prior. Exact
  single-clip identity passes, but the independent CarryBox96/KickBox22 semantic extension fails.
  Do not call an arbitrary hidden state an official SMP latent and do not integrate the selected
  prior into policy before a semantic gate passes.
- A valid policy comparison keeps teacher, initialization, update budget, seeds, reward weights,
  physics and frozen evaluation identical. Only the selected reward demo may differ.
- The historical reset-zero seed161587 comparison remains invalid because its correct Carry rollout
  was scored closer to Kick21. The new reference-aware seed161587 pair is distinct: both arms start
  the causal clock at restored frame 197 and pass the scorer and physics contracts.
- The new pair moves in `3/4` declared semantic directions at both update 32 and 64, but both arms
  remain bilateral Carry and foot-box contact stays near zero. Treat it as a promising single-seed
  behavior shift, not established semantic following.
- The behavior audit is independent of the reward predictor. With the same CarryBox45 teacher, the
  KickBox21-reward arm remains Carry-like. Seeds 161581/161583 observe `0/4` predeclared semantic
  directions and seed161585 observes a non-replicated `3/4`; orbit rate moves opposite to Kick21
  in all three. This is within-Carry behavior change, not semantic obedience. Task success and
  predictor loss remain separate.
- The teacher-floor diagnostic is also negative: correct and unrelated both lose the Carry
  interaction before any meaningful semantic separation. Its synchronized videos are failure
  evidence, not proof that the unrelated demo was followed.
- Official reference binary contact labels may define reference-event supervision only. Across
  100 CarryBox and 99 KickBox motions they cleanly separate hand/foot role and lifted/ground object
  motion, but they are not tactile force or actual rollout contact. The completed actual corpus
  instead uses named left/right hand/foot body-to-box filtered force, reset-bounded duration and
  episode-relative regime over 100 CarryBox plus 99 KickBox motions. Target alignment is fixed by
  causal normalized clock phase; per-frame free-window minimization is forbidden. The phase-aware
  6-layer predictor passes motion-disjoint full/zero/permuted gates. Its validation-only variance
  calibration yields `97.77%` mean test coverage for nominal 90% intervals, with `91.86%` minimum
  per-target coverage. Fixed CarryBox45/KickBox21 scoring prefers the matching task on both held-out
  splits. This proves a deployable selected-demo-conditioned reward signal, not policy following.
- The reward is dense compatibility feedback,
  `eta * (exp(-calibrated_event_risk) - fixed_train_baseline)`, not potential-difference shaping.
  The frozen scale is `eta=0.2427623309`, clip `0.1431077421`; future targets and predictor scores
  never enter the actor observation. Policy success must still be judged independently.
- The original seed-161581 traces do not contain per-body pose or foot-contact state. The repeated
  frozen evaluator now archives named body positions and left/right foot-to-box contact forces as
  evaluation-only evidence; these fields never enter the actor or reward.
- One policy-training seed is one experimental replicate. Multiple physics profiles do not replace
  independent training seeds.
- Never replace official MimicKit/TinyMDM or SUGAR components with a toy implementation.

## 4. Frozen tactile contract

- Backend: IsaacLab/PhysX only. Newton may supply assets but is not the simulation environment.
- Each hand has exactly 27 physical anatomical patches: palm `4 x 3`, plus proximal/middle/distal
  on thumb, index, middle, ring and little finger.
- A patch is the policy unit. Official R15 taxels are the TacSL physics/audit backend, never policy
  tokens.
- Every live patch record contains contact, normal load, mean pressure, signed local-XY shear and
  friction utilization. PS additionally uses causal slip evidence and
  `NO_CONTACT/STICK/INCIPIENT/GROSS` state.
- Never substitute `hands_contact_label`, ordinary ContactSensor, object state, generated values or
  saved/offline traces for live tactile input.
- All sensing and slip inference used by training must be generated inside the current rollout
  before the next actor call.

The only training-time slip interface is causal and batch-stateful:

```python
PatchSlipDetector.update(
    contact,
    normal_load_n,
    mean_pressure_pa,
    shear_xy_n,
    friction_utilization,
    timestamp_s,
    reset_mask,
)
```

Object motion, relative contact velocity, mass factor, jump flag, reward and future frames are
evaluation labels only. They may not enter the callable or deployed actor.

## 5. Frozen actor, teacher and mass-event contract

- The deployed actor uses the existing 504-D no-measured-object-state Tracker-command/
  proprioception contract.
- Official Refiner 890-D observation and the privileged critic are training-only. Joint motion can
  leak load, so any final claim must report tactile benefit over proprioception.
- Frozen Refiner controls the same complete G1 from CarryBox45 frame 0 until the box remains at
  least 0.05 m lifted for ten consecutive control frames.
- Handoff occurs without reset, teleport, replay or sensor-history replacement.
- The scheduler waits a matched 10--50 frames after handoff, then changes mass and inertia between
  actor calls. Nominal mass is `0.3023375869 kg`; fixed factors are
  `1.0x/1.5x/3x/6x/10x`.
- Mass factor, jump flag and handoff mask never enter the actor. P/PS require bilateral patch
  contact for ten frames before each event. Z performs zero TacSL reads.
- Teacher-prefix transitions receive no PPO surrogate/value/entropy credit.

## 6. Frozen matched formal training

When this line becomes active again, run exactly three branches serially:

- `Z`: exact-zero patch/slip tensors and zero TacSL reads;
- `P`: live patch contact/load/pressure/shear/friction with zero slip fields;
- `PS`: the same live patch signals plus the causal slip callable.

They share the anatomical patch-token encoder, SUGAR `512/256/128` actor, 29-D action, official
Tracker initialization, frozen Refiner, BCPPO, optimizer, reward, physics, mass sampling and seeds.

Formal seeds are `151014/151015/151016`, each exactly 3000 updates. Repository BCPPO stages are:

- 0--499: pure distillation;
- 500--999: critic warmup;
- 1000--1999: PPO-authority ramp;
- 2000--2999: full PPO with shared `stage3_distill_weight_floor=0.25`.

Never extend a seed beyond `model_2999.pt`. At each endpoint, automatically freeze and inspect the
checkpoint, then continue to the predeclared evaluation or next stage when its machine-readable
criteria pass; do not wait for human authorization. If an endpoint is behaviorally invalid,
automatically run one fixed-condition serious overfit diagnostic rather than spending another
formal budget.

## 7. Frozen evaluation and evidence

Pair checkpoints one-to-one with disjoint evaluation seeds:

- `151014 -> 152014`;
- `151015 -> 152015`;
- `151016 -> 152016`.

Each pair receives 20 profiles for each of five mass conditions: 300 rollouts per completed branch.
Do not evaluate every checkpoint on every evaluation seed or add profiles to only one branch.

Formal traces run at least 450 control frames. Camera-free traces are authoritative statistics.
Camera-enabled rollouts prove only their own rollout and are not frame-exact replays.

A positive result requires matched frozen-policy improvement in physical hold, recovery or safe
lowering, with nominal behavior reported separately. Losses, gradients, predicted reward, nonzero
action differences and a favorable video prove signal use at most.

Final H.264 evidence must show the complete G1/CarryBox world and both readable 27-patch maps on one
clock. Contact, pressure, signed shear and slip must be visible. Mass/jump overlays are evaluator
only and must be labelled hidden from the actor.

## 8. Frozen tactile evidence status

The recorded Z/P/PS endpoints do not establish tactile policy benefit. The corrected tactile-only
model1100 diagnostic gives `14/20` physical holds and `6/20` strict successes, with no matched
corrected Z/P/PS comparison. Do not describe Plan 15 sensing, slip or policy benefit as validated,
and do not resume it by changing normalization scale alone.

## 9. Heavy-box friction feasibility

The independent frozen-Refiner sweep at friction `0.5/1.0/1.5/2.0` for `6x/10x` is complete. At
6x the height losses are `0.5589/0.5429/0.02636/0.06596 m`; only `mu=1.5` satisfies the 5 cm hold
criterion. Every 10x condition drops. This is a feasibility search, not a tactile-policy result or
a monotonic friction-response estimate.

## 10. GPU allocation safety

- Prefer retained allocations long enough to finish training and review; lower CPU/memory requests
  are acceptable when they acquire a GPU faster.
- Never voluntarily release a granted GPU allocation merely because a child task or agent turn
  finishes. Keep the compute shell alive for review and follow-up.
- Enter the granted allocation with an explicit `srun --jobid=...` compute step before launching
  work. An `salloc` prompt can remain on a login node even though `SLURM_JOB_ID` is set.
- Launch long work from that compute step through
  `scripts/sugar/native_tactile/launch_retained_child.sh`, recording the exact Slurm step and child
  PID/PGID. The launcher must refuse missing `SLURM_STEP_ID` and login-node hosts.
- To change tasks, terminate only the recorded child process group. Never send generic Ctrl+C to
  the allocation shell and never cancel a retained job unless the user explicitly requests it.
- Formal training/evaluation remains serial under one pipeline lock. No concurrent writers to one
  seed directory.
- Do not run IsaacLab simulation or GPU training on the login node.

## 11. Repository hygiene

- `experiments/`, checkpoints, traces, videos, datasets and runtime logs are local-only and ignored.
- Root `legacy/` is the single ignored archive for failed, obsolete or superseded work.
- Active docs are README, `DOCS/reproducibility.md`, PLAN and TODO indexes, plus frozen Plan 15.
  Do not create streams of small status Markdown files.
- Keep only active, referenced code. Move explicit old branches, one-off diagnostics and superseded
  renderers to `legacy/` rather than leaving ambiguous entrypoints.
- Do not add routine SHA256 manifests, checksum ladders, duplicate validation scripts or defensive
  version matrices. Use direct outcome checks appropriate to the risk.
- Never commit or push experiment outputs or large binary artifacts. Before push, inspect Git
  status, staged paths and object sizes.
