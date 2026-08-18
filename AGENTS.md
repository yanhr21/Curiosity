# Global Agent Rules

## 1. Highest priority: Plan 15 only

The only active research plan is
`PLAN/15_online_patch_tactile_mass_adaptation/plan.md`; the only active task list is
`TODO/15_online_patch_tactile_mass_adaptation/todo.md`. All older plans, RGB, demo following,
ICM/Curiosity, Newton simulation, deformable demos and soft-body training are legacy until the
user explicitly changes priority.

The scientific question is whether live bilateral whole-hand tactile improves frozen physical
behavior after a complete SUGAR G1 has already lifted CarryBox and PhysX mass/inertia changes
online without changing geometry or appearance. The claim is incremental benefit over deployable
proprioception, never “only tactile can sense mass.”

## 2. No degraded placeholder models

- Never present a hand-written toy MLP/VAE/Transformer/world model as progress on T-Rex,
  VQ-VAE or another serious released method.
- Use official repositories, released checkpoints and faithful architectures first. Write only
  adapter/glue code needed to connect them to project data.
- If official code or weights are unavailable/incompatible, report the blocker. Any simplified
  diagnostic must be labeled diagnostic and cannot be called a faithful implementation.
- Plan 15 must retain the serious SUGAR policy, official Tracker warm start, frozen Refiner teacher,
  repository BCPPO and anatomical patch Transformer. No offline tactile replay or taxel-CNN
  substitute is allowed.

## 3. Online tactile contract

- Backend: IsaacLab/PhysX only.
- Each hand has exactly 27 physical anatomical patches: palm `4 x 3`, plus
  proximal/middle/distal on thumb, index, middle, ring and little finger.
- A patch is the policy unit. Official R15 taxels are the TacSL physics/audit backend, never policy
  tokens.
- Every live patch record contains contact, normal load, mean pressure, signed local-XY shear and
  friction utilization. PS additionally uses causal slip evidence and
  `NO_CONTACT/STICK/INCIPIENT/GROSS` state.
- Never substitute `hands_contact_label`, ordinary ContactSensor, object state, generated values or
  saved/offline traces.
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

## 4. Actor, teacher and mass-event contract

- The deployed actor uses the existing `504-D` no-measured-object-state Tracker-command/
  proprioception contract.
- Official Refiner `890-D` observation and the privileged critic are training-only. Joint motion
  can leak load, so final claims must report tactile benefit over proprioception.
- Frozen Refiner controls the same complete G1 from motion-45 frame 0 until the box remains at least
  `0.05 m` lifted for 10 consecutive control frames.
- Handoff to the student occurs without reset, teleport, replay or sensor-history replacement.
- The mass scheduler waits a matched `10--50` frames after handoff, then changes mass and inertia
  between control actions. Nominal mass is `0.3023375869 kg`; fixed factors are
  `1.0x/1.5x/3x/6x/10x`.
- Mass factor, jump flag and handoff mask never enter the actor. P/PS traces must independently show
  bilateral patch contact for the 10 frames before each event. Z makes zero TacSL reads.
- Teacher-prefix transitions are excluded from PPO surrogate/value/entropy credit.

## 5. Matched formal training

Run exactly three branches serially:

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

Never extend beyond `model_2999.pt`. Never automatically chain the next formal seed. At each
endpoint, freeze and inspect checkpoint finiteness, live handoff, mass readback, the 80-frame
physical window, action continuity and synchronized world/54-patch video. Only then may the user or
agent explicitly start frozen evaluation or the next seed.

If an endpoint is behaviorally invalid or ambiguous, run one fixed-condition serious overfit
diagnostic before spending another formal budget. An overfit is diagnostic, not a formal branch.

## 6. Frozen evaluation and evidence

Pair checkpoints one-to-one with disjoint evaluation seeds:

- `151014 -> 152014`;
- `151015 -> 152015`;
- `151016 -> 152016`.

Each pair receives 20 profiles for each of five mass conditions: exactly 300 rollouts per completed
branch. Do not evaluate every checkpoint on every evaluation seed and do not add profiles to only
one branch.

Formal traces run at least 450 control frames. Camera-free traces are the authoritative statistics.
Camera-enabled rollouts are outcome-sensitive near boundaries and prove only their own rollout;
never call them frame-exact replays.

A positive result requires matched frozen-policy improvement in physical hold, recovery or safe
lowering, with nominal behavior reported separately. Losses, gradients, predicted reward, nonzero
action differences and a favorable video prove signal use at most.

Final H.264 evidence must show the complete G1/CarryBox world view and both readable 27-patch maps
on one clock. Contact, pressure, signed shear and slip must be visible. Mass/jump overlays are
evaluator-only and must be labeled hidden from the actor.

## 7. Current accepted evidence (2026-08-18)

Sensing and slip are complete:

- continuous patch change precedes all 119 heavy-box drops, median lead 21 frames; contact binary
  median lead is 15 frames;
- continuous patch change precedes all 133 profiles with sag at least `0.02 m`; contact binary
  covers 81/133;
- controlled R15 slip state counts are STICK `109/111`, INCIPIENT `109/109`, GROSS `19/20`;
- CarryBox 3x slip evaluation has precision `1.0`, recall `0.9971`, median delay 0 and p95 1 frame.

This establishes an information advantage over binary contact, not policy benefit.

Formal frozen results in factor order `1x/1.5x/3x/6x/10x`:

- Z, three seeds: holds `59,59,52,1,0`; drops `0,0,2,58,59`;
- P, three seeds: holds `59,59,49,0,0`; drops `0,0,8,59,59`;
- P does not establish benefit over Z and trends worse at 3x;
- PS seed 151014: 19 eligible profiles/factor, holds `19,19,10,0,0`, drops
  `0,0,6,19,19`;
- PS seed 151015: 20 profiles/factor, holds `20,20,16,0,0`, drops
  `0,0,4,20,20`;
- PS seed 151016: 20 profiles/factor, holds `20,19,7,0,0`, drops
  `0,1,12,20,20`;
- PS aggregate holds are `59,58,33,0,0`; drops are `0,1,22,59,59`;
- at 3x, PS-P hold difference is `-0.2712` with paired hierarchical-bootstrap 95% CI
  `[-0.4655,-0.0667]`; PS-P drop difference is `+0.2373`, CI `[0.1053,0.3833]`;
- current P and PS therefore do not establish tactile-policy benefit. PS is significantly worse
  than P at 3x under the matched frozen evaluation.

The exact three-branch comparison is complete. The next serial action is the separate 6x/10x
friction-feasibility sweep and, if needed, a stronger-grip/lower-posture response or serious
fixed-condition overfit.

## 8. Heavy-box friction feasibility

The separate frozen-Refiner sweep at static/dynamic friction
`0.5/0.5`, `1.0/1.0`, `1.5/1.5`, `2.0/2.0` for `6x/10x` is complete. All eight runs have exact
material/mass readback, pre-jump bilateral contact and a complete post-jump window. At 6x the
height losses are `0.5589/0.5429/0.02636/0.06596 m`: only `mu=1.5` satisfies the 5-cm hold
criterion. Every 10x condition drops. These results remain separate from Z/P/PS.

The sweep is a feasibility search, not a monotonic friction-response estimate: friction changes
the pickup dynamics and shifts handoff/jump timing. The verified `6x, mu=1.5` hold satisfies the
required physical-success gate, so a stronger-grip/lower-posture overfit is not required for this
gate. Its camera-enabled 450-frame evidence also holds with `0.02552 m` maximum height loss and is
stored at `experiments/online_patch_tactile_mass_adaptation/visualizations/`
`official_refiner_mu1p5_6x_friction_hold_single_env/official_refiner_mu1p5_6x_world_bilateral27.mp4`.

## 9. GPU allocation safety

- Prefer retained allocations long enough to finish training and review; lower CPU/memory requests
  are acceptable when they acquire a GPU faster.
- Never voluntarily release a granted GPU allocation merely because one child task finished or the
  agent turn ends. Keep the compute shell alive for review and follow-up work.
- Launch long work through `scripts/sugar/native_tactile/launch_retained_child.sh`, recording the
  exact child PID/PGID.
- To change tasks, terminate only the recorded child process group. Never send a generic `Ctrl+C`
  to the allocation shell and never cancel retained jobs unless the user explicitly requests it.
- Multiple retained allocations may exist, but formal training/evaluation remains serial under one
  pipeline lock. No concurrent writers to the same seed directory.
- Do not run IsaacLab simulation or GPU training on the login node.

## 10. Repository hygiene

- `experiments/`, checkpoints, traces, videos, datasets and runtime logs are local-only and ignored.
- The root `legacy/` is the single archive for failed, obsolete or superseded work and is ignored.
- Active docs are README, DOCS, Plan 15 and TODO 15. Do not create streams of small status markdown
  files; merge durable conclusions into these documents.
- Keep only active, referenced code. Move explicit old branches, one-off diagnostics and superseded
  renderers to `legacy/` rather than leaving ambiguous entrypoints.
- Do not add routine SHA256 manifests, checksum ladders, duplicate validation scripts or defensive
  version matrices. Use direct outcome checks appropriate to the risk.
- Never commit/push experiment outputs or large binary artifacts. Before push, inspect Git status,
  staged paths and object sizes.
