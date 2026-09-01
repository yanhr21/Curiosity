# Global Agent Rules

> **Read this first.** Sections 1, 2, 9 and 10 are current and global. Sections 3–8 are the
> **Plan 15 contract** and describe the *archived* IsaacLab/PhysX line — they are kept because
> that code is still in the tree and because their traps generalise, but they are not the
> contract for current work. Where a section says "IsaacLab/PhysX only" or treats Newton as
> legacy, § 1 overrides it.

## 1. Highest priority: Plan 16, the Newton tactile rewrite

The active research plan is
[`PLAN/16_newton_tactile_rewrite/plan.md`](PLAN/16_newton_tactile_rewrite/plan.md); the active
task list is [`TODO/16_newton_tactile_rewrite/todo.md`](TODO/16_newton_tactile_rewrite/todo.md).
All current work lives in `sugar_newton/`. Start from [`README.md`](README.md) § Start here.

Plan 16 **supersedes Plan 15**, which reached a null result on IsaacLab/PhysX that a
115-finding audit then showed was not trustworthy. Plan 15 stays in the tree as the record of
that experiment and its audit; its code targets a host that does not exist on this cluster.
RGB, demo following, ICM/Curiosity, deformable demos and soft-body training remain legacy.

The scientific question is unchanged: whether live bilateral whole-hand tactile improves frozen
physical behavior after a complete SUGAR G1 has already lifted CarryBox and the mass/inertia
changes online without changing geometry or appearance. The claim is incremental benefit over
deployable proprioception, never "only tactile can sense mass."

The backend is now **Newton**, vendored as the `third_party/newton` submodule. It must stay a
clean diff against upstream: the audit's sharpest finding was a local edit inside vendored
IsaacLab that was indistinguishable from upstream by inspection and caused the shear leak. Do
not fix Newton in place — report the blocker.

Before touching any tactile channel or reward term, read
[`claude_context/findings.md`](claude_context/findings.md). It is the reason this line exists.

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

## 3. Online tactile contract *(Plan 15, archived line)*

> The patch layout, the "a patch is the policy unit" rule and the privileged-signal
> prohibition all carry over to Plan 16. The backend line does not: Plan 16 is Newton, and its
> channel set is defined in `PLAN/16_newton_tactile_rewrite/plan.md` § 4 and implemented in
> `sugar_newton/tactile/reducer.py`. Two of the channels below are the ones the audit found
> broken — see § "Audit addendum".

- Backend: IsaacLab/PhysX only. **(Plan 15 only; Plan 16 is Newton.)**
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

## 4. Actor, teacher and mass-event contract *(Plan 15, archived line)*

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

## 5. Matched formal training *(Plan 15, archived line)*

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

## 6. Frozen evaluation and evidence *(Plan 15, archived line)*

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

## 7. Plan 15's final evidence (2026-08-18) *(archived line; superseded by the audit below)*

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

## 8. Heavy-box friction feasibility *(Plan 15, archived line)*

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

Plan 16 runs on **this** cluster (OCI ORD) through SLURM. Details and failure modes are in
[`claude_context/operations.md`](claude_context/operations.md) § This repo.

- Hold a **named** dev-node container with `sbatch slurm/devnode.sbatch` and send work in with
  `slurm/devrun.sh`. The container name is what makes the GL/Xvfb installs persist across
  steps; a fresh `srun --container-image` starts from the pristine image every time.
- Prefer `interactive_singlenode` for one GPU: it has ~1243 nodes against `interactive`'s 10,
  so it actually schedules. `interactive` is needed for the 8-GPU variant, whose QOS the
  singlenode partition rejects.
- Allocations expire hard at 4 h and **say nothing** — the shell silently falls back to the
  login node, so work then fails with "no NVIDIA driver" or half-runs on CPU. Confirm a
  RUNNING job before sending work in.
- Never voluntarily release a granted allocation merely because one task finished or the agent
  turn ends. Keep it for review and follow-up.
- Newton's CPU device works on the login node, and the analytic validator is meant to be run
  there. Do not run GPU training or rendering on the login node.

## 10. Repository hygiene

- `experiments/`, `legacy/`, checkpoints, traces, videos, datasets and runtime logs are
  local-only and ignored. Never commit or push them. Before push, inspect git status, staged
  paths and object sizes.
- Assets are **not** in git and are not supposed to be. See [`ASSETS.md`](ASSETS.md) for what
  is missing and the two commands that restore it.
- `third_party/newton` is a submodule and must stay a clean diff against upstream (§ 1).
- Active docs are `README.md`, `ASSETS.md`, the `sugar_newton/**/README.md` files, `DOCS/`,
  Plan/TODO 16 and `claude_context/`. **Update the existing document.** Do not create streams
  of small status markdown files, and do not append a new copy of a section to the end of a
  README — that had happened three times over in `sugar_newton/rl/README.md`, leaving a file
  that contradicted itself about whether BCPPO worked.
- Keep only active, referenced code. Move superseded entrypoints to `legacy/` rather than
  leaving them ambiguous.
- Do not add routine SHA256 manifests, checksum ladders, duplicate validation scripts or
  defensive version matrices. Use direct outcome checks appropriate to the risk. (The pinned
  asset archives in `SUGAR/_downloads/` are the deliberate exception: an unpinned teacher
  checkpoint had already cost one campaign.)

## Audit addendum — 2026-08-19

A full correctness audit (115 findings, `claude_context/findings.md`) found that several
rules above are **stated correctly but violated by the implementation**. Read this before
treating any rule above as satisfied.

This is the addendum that motivated Plan 16, so it is the most transferable part of this file:
the "New standing rules" below apply to the Newton line too, and the specific defects are the
ones a port must not reproduce. Everything here describes the *IsaacLab* implementation.

### Rules violated in the Plan 15 code

- **§3 "Object motion, relative contact velocity … may not enter the callable or deployed
  actor."** Violated. `shear_xy_n` and `friction_utilization` — two of the six inputs to
  `PatchSlipDetector.update` — are computed inside TacSL from `relative_velocity_world`.
  The callable consumes the simulator's object-relative contact velocity one transform
  removed. It genuinely never reads object *pose*, mass, jump flag, reward or future
  frames.
- **§3 "Every live patch record contains … friction utilization."** The channel exists but
  cannot measure friction: it divides by the *sensor's* fixed `mu = 0.5`, the same constant
  TacSL already used to cap the shear numerator, so it is invariant to the object's PhysX
  material. Under stick it reduces to `2·tan θ` — pure taxel-frame geometry — and fires the
  0.60 incipient-slip trigger at θ ≥ 16.7° with **zero** relative motion.
- **§3 "Never substitute … ordinary ContactSensor."** Honoured in the observation path, but
  note the *reward* path depends entirely on ordinary ContactSensors, and two of those
  terms are broken (below).

### New standing rules

- **Never trust a reward term's weight column alone in this repo.** `feet_air_time` carries
  weight `+5.0` on a function that is always ≤ 0 — it is a penalty. `hoi_contact` carries
  `+1.0` but reads ContactSensors on hand links whose collision subtrees are deactivated at
  spawn, so it is dead and supplies **no behavioural gradient**. `undesired_contacts`
  (`−1.0`) counts all 54 elastomer patches; six in contact cancel the entire achievable
  positive reward.
- **Never report a Plan-15 number without naming the evaluation view.** Every published
  figure came from `--physical-outcome-view`, which suppresses all six SUGAR terminations,
  so `eligible` means only "the jump landed by frame 370". The stricter
  `strict_sugar_hold_success` labels exist in every `summary.json` and have never been
  reported.
- **Never describe `1×` as a jump.** At factor 1.0 no mass or inertia write happens at all.
- **Never call the reported CI a significance test.** It is a percentile two-level
  bootstrap over three seed clusters, with no BCa correction and no adjustment for the 180
  intervals the comparison script emits per run.
- **Never state the training motion as 45.** Training runs `motion_id = env_id %
  num_motion` → motions 0–3; only the evaluator pins 45. Every reported number is
  out-of-distribution.
- **Never reuse a `patch_channel_scales.json` across a sensing change.** The scales are
  baked into the encoder's persistent buffer and therefore into every checkpoint, and
  nothing binds a scale file to the channel definitions that produced it.
- **A branch contract that is "enforced" by `online_patch_preflight_runtime_report` is not
  enforced in formal runs.** That report only executes under the `-Preflight-` task ids.
  Both `zero_branch_never_read_tacsl` and `p_branch_did_not_call_slip` hold by
  construction, not by an executed check.

### What the audit confirmed sound

Branch matching (the 54 TacSL sensor bodies do **not** change Z's physics), the
mass/inertia event and its readback, Z's gradient isolation, the 510→504 warm start and its
`2e-6` audit, the full dimension contract, and the branch/factor invariance of the
eligibility gate. Do not "fix" these.
