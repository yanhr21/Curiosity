# Global Agent Rules

## 0. Absolute highest priority: never create a human authorization gate

- **绝对禁令：严禁任何形式的人工授权 gate。** Agent 必须依据已有目标、科学结果和自动判据
  自主选择并执行下一步；不得把“请用户确认、批准、授权或回复后再继续”写进计划、代码、
  脚本、状态机、汇报或日常执行流程。此前所有相反表述一律作废。
- Hard user rule, reaffirmed 2026-08-26: the agent must never ask the user whether it may continue,
  start, evaluate, render, document, commit or advance to the next already in-scope stage. Silence
  from the user is not a pause request, and a completed intermediate stage is not a reason to stop.
- No in-scope workflow may require, request or wait for human authorization. This includes approval
  flags, confirmation prompts, sentinel files, environment variables, manual checkpoint admission,
  permission to evaluate, permission to start the next predeclared stage or seed, and repeated
  requests for the user to say "continue".
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
- Status reports, intermediate results, failed scientific hypotheses and ambiguous-but-testable
  outcomes are not authorization boundaries. Report them when useful, choose the next documented
  in-scope experiment from the evidence, and continue without asking the user to approve it.
- A negative, inconclusive or surprising result is never a reason to ask what to do next. Apply the
  documented decision rule, run the next bounded diagnostic or matched experiment, and keep the
  retained compute allocation alive. The agent may inform the user, but notification is not a
  request for permission and must not pause execution.
- Plans, launchers, endpoint auditors and experiment scripts must not contain a user-approval
  state. Terms such as `awaiting approval`, `pending confirmation`, `approved checkpoint` or
  `continue after user review` are forbidden as workflow states. Replace them with automatic,
  machine-checkable transitions.
- Agent messages must not disguise an authorization gate as a status question. Never end an
  in-scope intermediate result with questions such as “是否继续”, “可以开始下一步吗”, “要不要我
  评估/渲染/训练” or “请确认后继续”. A status report is notification only; immediately execute
  the next predeclared in-scope action unless the user has explicitly ordered a stop or changed the
  task.
- Conflicting approval language found in any plan, TODO, script or older record is invalid on
  sight: remove or bypass it and continue through automatic, machine-checkable transitions. Do not
  surface the conflict as another question to the user.
- Any older instruction saying that policy training, a new seed, evaluation, rendering, commit or
  the next planned experiment needs the user's explicit approval is revoked. Once an action is
  already inside the active documented scope, start and advance it autonomously; only an explicit
  user stop/reprioritization or a genuine external blocker may interrupt that execution.

The user may still interrupt, reprioritize or stop work at any time. That user control must never be
implemented as a prerequisite approval gate for ordinary autonomous progress.

## 1. Current task: imported bugfix audit and generative overfit (2026-09-11)

Newest user objective: pull updated GitHub sugar, distinguish genuine bug fixes from alternate
data/GPU adaptations, then continue overfit until all metrics and actual visuals are normal.
This explicitly supersedes the old no-new-overfit terminal below for this investigation only.
Pulled 4bcf8582 over 7312737a. Preserve full model, original PhysX corpus, old caches/results;
do not use the new kinematic rasterizer or substitute joint positions for executed actions.
Retained allocation 291647/server31/step0 in tmux curiosity_pzw_bugfix_h200_20260911 is the
only current H200 allocation (16 CPUs/256 GiB/one day). Old 288297 was externally cancelled.
Initial current-code audit passed 23 CPU tests and official full-width forward/backward checks.
First new endpoint is overfit_resampled_noise_20260911: 32 resampled-noise updates, same eight
TRAIN cases, imported optimization settings explicitly labelled a local method variant, then
eight-draw endpoint probes and all eight actual TRAIN renders. Retain model and optimizer state.
This endpoint is not the full objective: if negative, diagnose remaining causes and continue
bounded evidence-driven overfit; do not relabel execution as success or open formal/physics.
Comparison and completion criteria: experiments/demo_following/paper_zero_wam_v1/
bugfix_audit_20260911/CODE_COMPARISON.json. Current status: DOCS/current_status.md.

### Prior completed scientific terminal (2026-09-10; historical)

Current user task: repository housekeeping. Archive superseded experiments and documents under
root `legacy/`, preserve results and dependencies, and keep active entrypoints concise.
The latest scientific objective is COMPLETE NEGATIVE, not an instruction to restart training.

- Repaired full-width paper Zero-WAM overfit completed exactly 32 applied updates, endpoint probes
  and all eight actual TRAIN-case renders. No update 33, formal update 701 or physics follows.
- Fixed-noise video/action/IFP final-to-initial ratios: 1.666156 / 0.222756 / 0.162461.
  Independent-noise/time ratios: 3.097468 / 0.581112 / 0.854718. Fitting and prompt gates fail.
- All eight generated TRAIN cases show block artifacts and damaged geometry; frozen-VAE controls
  are close to ground truth. Code interface tests are not whole-model correctness or overfit success.
- This is a 10,680,751,069-parameter local paper reconstruction, not the authors' released code.
  Keep 30x3072 independent video/action experts and four full-width IFP heads; no toy substitution.
- Preserve original overfit, fixed-noise diagnostic, repaired caches and formal step-700 artifacts.
  Local paper reconstruction performs no SHA256, digest or hash validation.
- Training, evaluation and rendering children have exited. Allocation 288297 was still RUNNING
  on server23 at cleanup, held by tmux `curiosity_pzw_overfit_h200_20260909`; do not cancel it,
  create duplicate requests or interpret this dated observation as a future live-state assertion.
- Tactile, HOST and BPP lines are inactive; their archived contracts are historical constraints,
  not launch instructions. Future methodological proposals have not been executed.
- Latest evidence and entrypoints: [current status](DOCS/current_status.md).
  Complete prior rules/history: [archived AGENTS](legacy/repository_cleanup_20260910/AGENTS.md).
  Archive path mapping and restore guidance: [archive guide](DOCS/archive.md).
  Archive snapshots may contain contradictory old status/launch language; never treat it as a
  current task. Durable frozen contracts below remain binding if that line is explicitly resumed.

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
- The official MimicKit TinyMDM provides a shared task-level motion prior. Exact single-clip
  identity passes, while same-task extension fails; the admitted shared conditional model passes
  motion-disjoint Carry/Kick classification but conditions only on task class. The released API has
  no stable selected-demo representation. Do not call an arbitrary hidden state an official SMP
  latent or claim that task-class energy measures deviation from a specified demonstration.
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
- Active docs are root README, `DOCS/README.md`, `DOCS/current_status.md` and `DOCS/archive.md`.
  Old PLAN, TODO and reproducibility/project-page documents are archived, not execution queues.
  Do not recreate root PLAN/TODO directories or streams of small status Markdown files.
- Keep only active, referenced code. Move explicit old branches, one-off diagnostics and superseded
  renderers to `legacy/` rather than leaving ambiguous entrypoints.
- Do not add routine SHA256 manifests, checksum ladders, duplicate validation scripts or defensive
  version matrices. Use direct outcome checks appropriate to the risk.
- Never commit or push experiment outputs or large binary artifacts. Before push, inspect Git
  status, staged paths and object sizes.
