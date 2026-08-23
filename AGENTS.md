# Global Agent Rules

## 1. Current priority: demo following and SMP audit

The tactile/mass-adaptation line in
`PLAN/15_online_patch_tactile_mass_adaptation/plan.md` is frozen. Do not launch new tactile
training, evaluation or scale tuning unless the user explicitly resumes it.

The current task is to consolidate demo-following evidence, distinguish implemented components
from proposals, build a predictor-independent behavior adherence metric, and design the next
matched multi-seed experiment. Do not start new policy training without explicit user approval.

The completed same-teacher experiment fixes CarryBox45 as teacher in both arms and changes only the
selected reward demo, CarryBox45 versus KickBox21. Frozen success is `16/20` versus `18/20`, with
two physical root-height falls per arm. This proves policy change, not correct-demo superiority or
semantic obedience.

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
- Plan 15, if explicitly resumed, retains the serious SUGAR policy, official Tracker warm start,
  frozen Refiner teacher, repository BCPPO and anatomical patch Transformer. No offline tactile
  replay or taxel-CNN substitute is allowed.

## 3. Demo-following evidence contract

- The early motion45/motion96 result compares two CarryBox demonstrations and is not an unrelated
  demo experiment.
- The archived 1216-update experiment changed both teacher and selected demo, and its goal task
  still penalized tactile contact. It is diagnostic history, not an active result.
- The current internal reward predictor is a serious causal future-mismatch predictor over body,
  box position, box rotation and box velocity. It is not an SMP-latent predictor.
- The official MimicKit TinyMDM currently provides a shared generic motion prior. Exact
  single-clip identity passes, but the independent CarryBox96/KickBox22 semantic extension fails.
  Do not call an arbitrary hidden state an official SMP latent and do not integrate the selected
  prior into policy before a semantic gate passes.
- A valid policy comparison keeps teacher, initialization, update budget, seeds, reward weights,
  physics and frozen evaluation identical. Only the selected reward demo may differ.
- Before another training budget, define direct behavior-level adherence independent of the reward
  predictor. Task success and predictor loss must be reported separately.
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

If explicitly resumed, run exactly three branches serially:

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
