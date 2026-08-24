# Global Agent Rules

## 1. Current priority: phase-aware event reward and matched policy design

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

The active frozen predictor is seed271303: an 11,386,010-parameter, 6-layer, 384-D phase-aware
causal Transformer over the exact past `10 x 121` deployable core, a fixed numeric demo and a
causal normalized clock phase. Its training, calibration and fixed CarryBox45/KickBox21 bidirectional
reward-scale gates pass. The next branch is a matched policy experiment. Do not start new policy
training without explicit user approval.

The phase-aware dense runtime is connected to the serious SUGAR rollout integrator and the matched
correct/unrelated launcher. Frozen evaluation reads both update 32 and 64, runs predictor-independent
behavior audits separately, and renders the final update-64 demo/actual pair. This integration is
dry-run and CPU-test complete, but no phase-aware policy checkpoint exists yet. Do not describe
integration readiness as a policy result.

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
retain the allocation as required and move the Isaac gate to a fresh GPU. Policy training still
requires explicit user approval.

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

Routine read-only audits, dataset builds and predictor-only gates may proceed through the documented
queue. Policy training is the explicit exception and requires user approval.

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

Never extend beyond `model_2999.pt` and never automatically chain another formal seed. Freeze and
inspect each endpoint before evaluation or another seed. If an endpoint is behaviorally invalid,
run one fixed-condition serious overfit diagnostic rather than spending another formal budget.

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
- Launch long work through `scripts/sugar/native_tactile/launch_retained_child.sh`, recording the
  exact child PID/PGID.
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
