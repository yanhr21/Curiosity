# Curiosity — project context

**Read [`claude_context/operations.md`](claude_context/operations.md) before running
anything.** It carries the portable cluster rules, this machine's values, and —
most importantly here — the fact that **this repo does not execute on this
machine**. Plan 15 runs on a separate host at `/public/home/yanhongru/Curiosity`.
This checkout is for reading, editing, testing and committing source.

The browsable version of this context is `claude_context/index.html`
(`python3 claude_context/serve.py`).

## What this repo is

A single active research question, in IsaacLab/PhysX: **after a complete SUGAR G1
has already lifted the official CarryBox, does live bilateral whole-hand tactile
improve frozen physical behaviour when the box's mass changes online?** Geometry,
appearance, reference trajectory and contact targets are held fixed; only
mass/inertia changes.

The claim under test is deliberately narrow: an **incremental** benefit over the
504-D deployable Tracker-command/proprioception base. Mass also leaks into
proprioception through joint sag and tracking error, so "only tactile can sense
weight" is not a claim this design can support.

- Active design: `PLAN/15_online_patch_tactile_mass_adaptation/plan.md`
- Active task list: `TODO/15_online_patch_tactile_mass_adaptation/todo.md`
- Global agent rules: `AGENTS.md`
- Everything else — RGB, demo following, ICM/Curiosity, Newton simulator,
  deformable and soft-body training — is **legacy** and out of the queue.

## Branch and layout

Branch `sugar`, remote `git@github.com:yanhr21/Curiosity.git`. Note the top-level
layout differs completely from the older `2026_7_14_mike` branch (which had
`src/`): `sugar` is `SUGAR/` + vendored `IsaacLab/` + `scripts/` + `tests/`.

| where | what |
|---|---|
| `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/` | env cfgs, tactile observation terms, slip detector, mass jump, teacher handoff |
| `SUGAR/source/sugar_rl/sugar_rl/utils/` | patch encoder, actor-critic, BCPPO |
| `SUGAR/scripts/sugar_rl/` | train / evaluate / compare launchers |
| `scripts/sugar/native_tactile/` | sensing sweeps, slip calibration, visualization, retained-child launcher |
| `tests/native_tactile/` | the only code runnable off the GPU host |
| `IsaacLab/` | vendored IsaacLab v2.3.2, including the TacSL `VisuoTactileSensor` |
| `experiments/`, `legacy/` | gitignored; contents exist **only on the runtime host** |

## The three branches

Matched in every respect except the tactile observation. Same official Tracker
warm start, frozen official Refiner teacher, BCPPO, 512/256/128 actor, 29-D
action, physics, reward, seeds `151014/151015/151016` and 3000-update budget.

- **Z** — patch and slip tensors exactly zero; never reads TacSL.
- **P** — live contact / normal load / pressure / signed XY shear / friction
  utilization; slip fields exactly zero.
- **PS** — P plus the causal `PatchSlipDetector` state and score.

## Current result — a negative one

As of 2026-08-18, all nine endpoints and all 900 frozen rollouts are complete
(300 per branch: 3 seed pairs × 5 mass factors × 20 profiles).

Eligible physical holds, ordered `1× / 1.5× / 3× / 6× / 10×`:

| branch | holds | drops |
|---|---|---|
| Z | `59, 59, 52, 1, 0` | `0, 0, 2, 58, 59` |
| P | `59, 59, 49, 0, 0` | `0, 0, 8, 59, 59` |
| PS | `59, 58, 33, 0, 0` | `0, 1, 22, 59, 59` |

**P has not demonstrated any benefit over Z**, and at 3× the paired PS−P hold
difference is `-0.2712` with a hierarchical-bootstrap 95% CI of
`[-0.4655, -0.0667]` — PS is significantly *worse* than P. The separate
high-friction feasibility sweep found `6×` holds only at `mu=1.5`
(height loss `0.02636 m`); `10×` drops at every friction tested.

This coexists with a real, measured **information** advantage: continuous patch
signal separates mass ~13 frames after the jump versus ~35 for proprioception,
and precedes `133/133` sag events where binary contact covers only `81/133`. The
information is there; the training has not converted it into better frozen
physical behaviour.

## In flight — a correctness audit, 2026-08-19

The null result prompted an audit rather than more training: first by hand, then by nine
independent Opus auditors — one per branch on structure, one per major module on depth.
**115 findings** across three rounds. Full entries with file and line are in
`claude_context/findings.md` and on the Findings tab of the context page. Nothing has been
fixed yet.

Three of my own earlier claims were **refuted** by the audit and are corrected below; the
log records both versions, because a finding that had to be withdrawn is as useful as one
that held.

### What survives — do not rebuild

- **Branch matching is real, and this was the biggest risk to the design.** The worry that
  the 54 TacSL sensor bodies might change Z's physics is false: the sensors create
  read-only PhysX views and never write forces back, neither observation term consumes
  RNG, and the jump/handoff schedules are deterministic. Z's trajectory is matched to
  P/PS at the same seed.
- **The mass event.** Written at the action boundary, inertia scaled by exactly
  `target/default`, both values read back, raises past `rtol=1e-6`.
- **Z's gradient isolation**, measured rather than argued: all 41 encoder parameters are
  in the optimizer, every gradient is exactly zero, and five Adam steps leave them
  bitwise unchanged.
- **The eligibility gate cannot bias the comparison.** The 59 denominator is 60 minus one
  seed-151014 profile whose jump never lands by frame 370. Because the jump is scheduled
  off the *frozen teacher's* pickup and fires unconditionally thereafter, that exclusion
  is branch- and factor-invariant. The statistic confirms it: −16/59 = −0.27119, exactly
  the reported −0.2712.
- Every dimension in the contract (504 / 890 / 632 / 1944 / 128 / 29) holds and is
  asserted at runtime.

### What does not — four independent confounds, all pushing the same way

**1. The reward penalises grasping, and the term meant to reward it is dead.**
`hoi_contact` (+1.0) reads ContactSensors on the two `*_rubber_hand` links, whose
collision subtrees the robot spawner *deactivates at spawn*. `is_contact` is therefore
permanently false and the term degenerates to `(False == contact_label)` — a lookup into a
frozen `.npy` keyed by (motion, frame). It is a pure function of the reference clock and
**supplies no behavioural gradient at all**. What is left is `undesired_contacts` (−1.0), a
*count* of bodies whose net force exceeds 0.1 N, matching all 54 elastomer patches (87 of
91 bodies), scaled by `dt` to −0.02 per body per step. Against a maximum achievable
positive reward of 5.125, **six patches in contact cancel everything** — and a bilateral
carry puts far more than six over threshold. **Nothing in the 21 reward terms asks the
policy to hold the box up.**

**2. The tactile channels cannot carry the signal under test.** `friction_utilization`
divides by the *sensor's* fixed `mu = 0.5`, the same constant TacSL already used to cap
the shear numerator, so it is invariant to the object's real material friction — while
training randomizes that friction over `U[0.2, 0.8]` per environment, giving a true
contact coefficient of 0.35–0.65. Worse, TacSL projects the **total** force `fc + ft` into
a *per-taxel* frame, so under stick — where the friction term vanishes — the shear channel
is a geometric leak of the **normal** force, `k_n·d·sinθ`, which scales *with* grasp load.
Measured on the real reducer at exactly zero relative velocity: 8.6° off-centre gives
utilization 0.304, **17.2° gives 0.622 — past the slip detector's 0.60 incipient trigger
with no slip at all**. Under stick utilization ≈ `2·tanθ`, so INCIPIENT fires at 16.7° of
misalignment; any value above 1.0 is proof of geometric contamination by construction. And
after p99.5 max-scaling the binary channels sit at exactly 1.0 while shear sits at ~0.05 —
the two channels the experiment is about are the smallest in the tensor.

**3. Trained on motions 0–3, evaluated only on motion 45.** `start_init_env_ratio = 1.0`
puts every training env on the protected branch (`motion_id = env_id % num_motion`, 4
envs); the evaluator pins 45 by monkey-patching `_sample_init_state`. Every headline number
is an out-of-distribution measurement, and neither the plan nor the README says so.

**4. The evaluation is looser than it reads.** Every reported number came from
`--physical-outcome-view`, which monkey-patches `termination_manager.compute` to return
all-`False`: SUGAR's six reference terminations are computed and kept as labels but never
reset or end a rollout. `eligible` therefore means only "the mass jump landed by frame
370", not "the rollout stayed inside the SUGAR contract" — and the stricter
`strict_sugar_hold_success` labels exist in every summary and have never been reported.
`1×` is not a 1× jump: no mass write happens at all, so it is a pure no-perturbation
control. The 95 % CI is a *percentile* two-level bootstrap over only **3 seed clusters**,
with no BCa correction and no adjustment for the 180 intervals the script emits per run —
a strong point estimate with an optimistic interval. And the 20 "profiles" per run are
near-replicates: same motion, same frame 0, no push or observation randomization, differing
only in a deterministic jump delay and four per-env friction draws that repeat.

### The strongest single candidate for PS < P

`_online_patch_slip_history` returns early on a `common_step_counter` match **before**
calling `detector.update`, so on that call the detector never receives `reset_mask`. But
`ManagerBasedEnv.reset()` runs `_reset_idx` and then `observation_manager.compute()`
**without incrementing the counter**, and the evaluator calls `env.reset()` once per batch.
So for every batch after the first the reset is **silently swallowed**: the detector's
`previous_*` buffers, its `gross_evidence_count` and its GROSS latch survive the episode
boundary, and it differences the new episode's first frame against the previous episode's
last. Measured on the real module: `slip_score` saturates at 2.0 instead of 0.778, and a
GROSS latch from episode A survives every frame of episode B. **P carries no differencing
state, so the identical cache bug costs P only a stale first observation.** This bug is
asymmetric, it favours P over PS, and it sits on the evaluation path that produced the
−0.2712.

GROSS is also a latch: `retained_gross` holds a patch at GROSS for as long as *any*
incipient evidence persists — including a static geometric `utilization ≥ 0.60` — and
nothing but a reset clears it. And the detector's thresholds were calibrated on a **flat**
R15 capsule, a geometry in which the taxel-frame misalignment that dominates the 54 curved
anatomical pads does not exist.

### Smaller, but real

The distillation loss is not masked by the handoff mask while the PPO terms are, so the
frozen **nominal-mass** Refiner — trained on 0.5–2.0× and never on 3/6/10× — is a live
regression target inside the post-jump window. `configure_tactile_actor_finetune` overrides
the parent's freeze-and-mask implementation *without calling `super()`*, installs no
gradient mask, and is a no-op: the 504 warm-started Tracker columns train from update 0.
Only ~3 % of the 128-D patch embedding's norm varies with the tactile input; the rest is a
constant DC offset, and `warm_start_tactile_gain = 0.01` is undone by Adam inside one
BCPPO update. The warm start silently overwrites the configured `learning_rate`.

### What to do next

1. **Fix the reward.** Exclude `.*_anatomical_.*` from `undesired_contacts`; point
   `hoi_contact` at the patch bodies; consider adding a term that actually rewards holding
   the box. Cheapest change, furthest upstream.
2. **Fix the swallowed reset** in `_online_patch_slip_history` — hoist the reset out of the
   step-counter guard. It is a few lines and it is the best candidate for PS < P.
3. **Measure before rebuilding the sensing.** Per-channel variance and mass/friction mutual
   information during the *hold* phase, on saved traces. No GPU needed.
4. **Rebuild the sensor reduction** in the contact frame, with a minimum-load gate and the
   object's real material friction. Note the per-taxel frame that causes the leak is a
   **local** modification — upstream IsaacLab v2.3.2 uses one constant quaternion per
   sensor — so this is fixable without touching vendor force equations. Only a tangential
   spring with stick memory would be a genuine vendor change.
5. **Close the train/eval motion split**, and report the strict-view numbers alongside the
   physical-outcome ones.

Existing checkpoints do **not** survive any of this: the channel scales are baked into the
encoder's persistent buffer and therefore into every checkpoint, and nothing binds a scale
file to the channel definitions that produced it.

## The ten problems, in plain English, with locations

Paths relative to the repo root; `$R` = `SUGAR/source/sugar_rl/sugar_rl`. Every line number
below was checked against source. Technical entries with full evidence are in
`claude_context/findings.md`; the same table is on the context page under **The ten**.

| # | Problem | Where | Fix size |
|---|---|---|---|
| 1 | **Reward punishes touching the box.** A penalty counts every body in contact with anything; the 54 tactile pads were never excluded. | `$R/tasks/…/train_refiner/carry_box_refiner_env_cfg.py:86-98` — **line 93** is the regex. Sensor = all bodies: `base_refiner_env_cfg.py:72`. Reduction: `IsaacLab/…/envs/mdp/rewards.py:260-268` | **One regex.** Add `(?!.*_anatomical_.*)` on line 93 |
| 2 | **The term meant to reward touching is dead.** It reads sensors on hands whose collision shapes were switched off when the pads were added. | Term `carry_box_refiner_env_cfg.py:99-112`; impl `$R/tasks/locomanip/mdp/rewards.py:142-172` (**line 167**); cause `$R/assets/robots/anatomical_whole_hand_tacsl_g1.py:1093`, disable at **1136**; dead sensors `base_refiner_env_cfg.py:93-108` | **Small.** Repoint the two ContactSensors at the pads |
| 3 | **Nothing rewards holding the box up.** Scored only on matching the reference motion. | The full 21 terms: `base_refiner_env_cfg.py:298-399` + `carry_box_refiner_env_cfg.py:84-116` — the absence is the finding. Hold/drop live only in `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:339-360` | **Design decision** |
| 4 | **The "friction" channel cannot see friction.** Divides by a fixed 0.5 the simulator already used on the numerator. | `$R/tasks/locomanip/online_patch_tactile.py:144-146`; **line 237** shows `mu` is the *sensor's*. Numerator cap: `IsaacLab/…/tacsl_sensor/visuotactile_sensor.py:1005-1007`. Real friction randomized: `base_refiner_env_cfg.py:274-281` | **Medium** |
| 5 | **"Shear" mostly measures geometry.** Off-centre contact reports slipping with nothing moving. | `visuotactile_sensor.py:1018` sums normal+friction; **1023-1024** split the *total* and name the parts. Angle-dependent frame: **564-608** — a **local edit**, not upstream | **Medium, and local.** Project onto the SDF normal (already computed) |
| 6 | **Slip detector gets stuck on.** Stays at GROSS while any weak evidence continues — including the fake signal from #5. | `$R/tasks/locomanip/patch_slip.py:263-264`; cleared only at **line 135** | **Small** |
| 7 | **The reset is silently skipped between scoring batches** — the detector carries stuck state into the next episode. **P has no detector, so P is unaffected.** | `$R/tasks/locomanip/online_patch_tactile.py:446` returns early, **before** `detector.update` at **460**. Counter only moves in `step()`: `IsaacLab/…/manager_based_rl_env.py:203`. Bypassing reset: `evaluate_online_patch_mass_bcppo.py:546` | **A few lines.** Move the reset above the line-446 guard |
| 8 | **Trained on motions 0–3, tested only on motion 45.** | `carry_box_online_patch_tactile_mass_env_cfg.py:259` (`start_init_env_ratio = 1.0`) → the `env_id % num_motion` branch in `$R/tasks/locomanip/mdp/commands.py`. Evaluator override: `evaluate_online_patch_mass_bcppo.py:479-487`, default 45 at **line 41** | **Experiment-design decision** |
| 9 | **Scoring ran with the failure checks off.** "Eligible" only means the weight change landed in time. | `evaluate_online_patch_mass_bcppo.py:495` (replacement `compute`), **547** (reset stub), **729** (frame validity skipped). Always passed by `scripts/sugar/native_tactile/run_plan15_frozen_seed.sh:63` | **Free** — the strict numbers already exist in every `summary.json` |
| 10 | **The confidence interval is weak.** 3 seed clusters, simplest bootstrap, 180 uncorrected intervals per run. | `SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py:119` (plain `np.percentile`), **196** (method string), **line 21** `METRICS` × 5 factors × 3 pairs | **Free** — restate it |

**Start with #1 and #7.** One is a single regex, the other is moving a few lines above a
guard, and both sit upstream of everything else. **#9 and #10 cost nothing** — the stricter
numbers are already in the result files. Everything else is real work.


## Standing context

- The physical scene, the 27-patch anatomy, the mass-jump mechanism and the
  no-reset Refiner→student handoff are all verified working. The mass jump
  scales inertia consistently and reads back both values; that part is sound.
- R15 taxels are the physics and audit backend. **A patch is the policy unit** —
  never a taxel.
- These are high-fidelity simulated tactile signals. They are not GelSight
  calibration and not sim-to-real.
- A camera-enabled rollout only ever describes its own rollout; it is never a
  frame-by-frame replay of a camera-free formal trace.
