# Curiosity

Whole-hand tactile sensing for the SUGAR G1 carry task, rebuilt on the **Newton** physics
engine.

The scientific question is unchanged from the previous line: after a complete SUGAR G1 has
already lifted CarryBox, does live whole-hand tactile improve behaviour when the box's mass
changes online without changing its geometry or appearance? The claim being tested is an
*incremental* benefit over deployable proprioception, never "only tactile can sense mass" --
mass leaks into proprioception through joint sag and tracking error.

The previous line answered that on IsaacLab/PhysX and got a null result. A 115-finding audit
then showed the null was not trustworthy: the reward paid the policy for *not* touching the
box, and the friction channel could not see friction. **That audit is why this line exists**,
and it is required reading before touching any tactile channel or reward term:
[`claude_context/findings.md`](claude_context/findings.md).

| | |
|---|---|
| active plan | [Plan 16 — Newton tactile rewrite](PLAN/16_newton_tactile_rewrite/plan.md) |
| task list | [TODO 16](TODO/16_newton_tactile_rewrite/todo.md) |
| branch | `2026_9_1_tactile_newton` — self-contained; Newton is the `third_party/newton` submodule |
| runs on | this cluster (OCI ORD), conda env `robotbaby`, SLURM dev node |
| prior line | Plan 15 on IsaacLab/PhysX — [below](#prior-line-plan-15-on-isaaclabphysx), superseded but not deleted |

## Important caveats

Things that have already cost days, or that will silently produce a wrong number if you do not
know them. Each one says what the problem is and what to do about it.

### The IsaacLab-to-Newton port (`sugar_swap`)

**A weld-adjacent self-collision blocked the hip bow.** Fixed, but understand it before porting
another robot. `pelvis` declares no collision geometry of its own — the pelvis collider lives on
`pelvis_contour_link`, which is *welded* to `pelvis`. Newton's importer filters shapes on the same
body and on bodies joined directly by a joint, but neither rule looks **through** a fixed joint, so
it filtered `pelvis` (which has no shape, hence nothing) and left the real pelvis collider free to
collide with `pelvis`'s own children, both hip pitch links. MuJoCo's `filterparent` and PhysX's
articulation adjacency rule both *do* look through: bodies with zero degrees of freedom between them
are one rigid unit. The cost was 20.6 kN against a 33 kg robot, the hip bow pinned at 25° where
IsaacLab reaches 58°, and hands that never came within 2.5 cm of the box.
*Solution:* `_weld_filter_pairs` in `sugar_swap/builder.py` applies the weld-aware rule and is now
the default. `RB_SELF_COLLISION=raw` reinstates the old behaviour for A/B; `=0` filters every
robot-vs-robot pair and is a diagnostic only, since SUGAR trained with self-collision on. When
porting a new URDF, check for colliders sitting on welded links before anything else.

**Every Newton refiner run before that fix is invalid.** They trained in an environment where the
robot could not bow past about 10° of hip flexion, so the task was physically impossible and the
reward plateau at ~12–16 says nothing about learning. Do not compare against those curves.

**Convex hulling inflates thin cosmetic shells, and the hands are among them.** `builder.py` runs
`approximate_meshes("convex_hull")` on every mesh because MuJoCo only collides convex geoms.
Measured hull/raw volume: `logo_link` 39.6×, `waist_support_link` 16.7×, `pelvis_contour_link`
15.9×, wrist roll links 10.7×, and both `rubber_hand`s 2.35×; genuinely convex links are fine
(`hip_pitch` 1.08×, `pelvis` 1.09×). The bow no longer depends on this, but the rubber hands are the
*grasping* surfaces, so contact geometry there is coarser than PhysX's. *Solution if it matters:*
convex decomposition into several pieces for the non-convex meshes, at a collision-performance cost.

**Two randomized material properties silently do not survive MuJoCo.** Static and dynamic friction
collapse to a single coefficient (we keep dynamic, so the robot's 0.3–1.6 static range is
unrepresented), and restitution 0.0–0.5 is a complete no-op because `SolverMuJoCo` never reads
`shape_material_restitution`. Not fixable without leaving MuJoCo; treat both as absent when
reasoning about the randomization distribution.

**Domain randomization has to be announced to the solver.** `SolverMuJoCo` steps its own `mjw_model`,
so writing mass, COM or friction into the Newton model is invisible until `notify_model_changed`.
Fixed, but it was a silent no-op for a stretch of training, and the same trap applies to anything
else written after build. Note it is `mode='startup'`, so the run gets a fixed population of
dynamics rather than a fresh draw per episode.

**`effort_limit_sim` outranks the URDF.** SUGAR leaves `effort_limit` at `None` but sets
`effort_limit_sim` on every actuator group, and that is what IsaacLab hands the simulator; it is also
the numerator of the action scale (`0.25 * effort_limit_sim / stiffness`). Preferring the URDF ran
six joints (both ankles, waist pitch and roll) at 35 N·m instead of 50 and made the policy command
displacements they could not reach. Fixed.

**The two importers order bodies differently and nothing guards it.** `hip_pitch`/`pelvis_contour`
and `head`/`logo` come out in different positions. The tracked-body names happen to match today, so
rewards are correct — but adding a tracked body would silently reward the wrong link. *Solution:*
assert the name-to-index mapping rather than trusting order. Not yet done.

**Post-step magnitudes diverge even where shapes match exactly.** `joint_torque` 1.81×,
`motion_body_ang_vel` 0.52×, `joint_limit` 0.93×, and contact forces 0.23–2.8×, all with identical
shapes and zero residual. Unresolved; do not treat these channels as calibrated across engines.

**`ls_iterations` is a guess.** It is mapped from PhysX's `solver_velocity_iteration_count` (4), but
MuJoCo's line search is a different algorithm whose own default is 50. It costs almost nothing in
speed, so the faithful value should be decided on accuracy grounds; it has not been.

**Contact margin must stay 0.** The solver subtracts `(margin0 + margin1)` from separation, so any
nonzero margin leaves the grasp resting that far off the surface. This was a real bug once.

**The friction cone is pyramidal, and that is a dynamics choice.** Pyramidal is both faster and
closer to PhysX, but switching to elliptic under a policy trained on pyramidal is not valid — adopt
a change like this by retraining. `RB_CONE` / `RB_IMPRATIO` override.

**Newton is not run-to-run reproducible at a fixed seed.** Repeat replays of one checkpoint gave box
lifts of 0.0385, 0.0574 and 0.0632 m. Do not read a few percent between single replays as a real
difference, and do not quote five significant figures from one.

**The two engines do not draw the same reset.** At reset the robots differ by 8.8 cm horizontally
and 3.3° in yaw while the box is bit-identical, so a Newton replay and an IsaacLab replay are not a
perfectly matched pair. Pin the reset RNG before any claim that hinges on small differences.

### Training and comparison

**envs/rank is the scaling knob, not GPU count.** 32 GPUs at 128 envs/rank runs at 28% efficiency;
at 512 envs/rank, 92%. Four-node DDP does not pay at SUGAR's batch (2.40 s/iter on 32 GPUs versus
2.59 on 8), so prefer a single 8-GPU node unless the total env count rises.

**SUGAR's `--num_envs` is per rank; `sugar_swap`'s `RB_ENVS` is a total.** The inconsistency is
deliberate — it keeps `RB_GPUS` changing only the wall clock and never the batch — but reading
SUGAR's convention into ours silently multiplies the batch by the rank count.

**The curriculum pool rebuilds at every leg boundary.** `MotionCommand.count` starts at 0 and
`init_pool` is never checkpointed, so a resumed leg re-serves its 1,000-iteration warmup. Unfixed,
and second-order, but it wastes the start of every leg.

**Co-resident GPU work on a DDP node costs every rank, not one.** A 5-minute video loop run with
`--overlap` on the training node dragged the whole run from 2.8 to 7–12 s/iter, because all-reduce
blocks on the slowest rank. Give side jobs their own node.

### Cluster and tooling

**Interactive partitions expire hard at 4 h, silently.** On expiry a screen window falls back to the
login node and work fails with "no NVIDIA driver", or half-runs on CPU. Chain legs with
`--dependency=afterany` — `afterok` stalls, because a leg killed by the wall clock exits non-zero and
that is the normal way a leg ends.

**`find /` or `find ~` on this Lustre filesystem hangs until killed.** Chained with `;`, everything
useful after it never runs. Use targeted `grep`/`ls` with explicit paths.

**wandb has four separate traps.** Resumed legs raise `ConfigError` unless `Config.update` is allowed
to change values (fixed by a startup hook). Shared mode cannot append to a run whose other writer is
not also in shared mode: it ignores `step=` and drops history rows while the mp4 files still upload,
so media lands in the Files tab with no panel — use a companion run and make the iteration a
`step_metric` instead of wandb's step. Re-syncing a *live* offline run wipes its history (263 rows to
0); snapshot to a separate id. And a stray `WANDB_MODE=offline` rides `--export=ALL` into jobs, so
pin it explicitly.

**Newton writes eval mp4s at 30 fps for a 50 Hz sim.** Playback is 1.67× too slow, which reads as
sluggish motion and has already caused one wrong conclusion. Re-encode before comparing videos.

**`scontrol show job` does not print the export list.** A leg's `RB_*` settings are not recoverable
from the job record, so the run log is the provenance trail — which is why the builder prints its
self-collision mode and batch shape at startup.

## Start here

```bash
git clone --recurse-submodules <url> && cd Curiosity
bash env/setup_env.sh                 # login node, ~10 GB, creates conda env `robotbaby`
bash SUGAR/_downloads/fetch_assets.sh # ~700 MB of assets; NOT in git, see ASSETS.md
```

Then the cheapest useful command, which needs **no GPU and no container** — Newton runs on
the login node against the CPU device:

```bash
source env/activate.sh
python -m sugar_newton.validation.incline     # analytic tactile ground truth, ~45 s
```

For anything on a GPU, hold a dev node and send work into it:

```bash
sbatch slurm/devnode.sbatch                   # 1 GPU, 4 h, named pyxis container
bash slurm/devrun.sh "source env/activate.sh && python -m sugar_newton.validation.g1_carrybox_policy"
```

`slurm/devnode.sbatch` must be submitted **from the repo root** (it locates the repo via
`SLURM_SUBMIT_DIR`). It runs `slurm/setup_container.sh` once, which installs the GL/Xvfb
system libraries and writes the NVIDIA glvnd ICD that headless EGL needs; because the
container is *named*, later `devrun.sh` calls attach to the same container root and keep
those installs. `slurm/devnode8.sbatch` is the 8-GPU variant for sweeps
(`RB_STATE=slurm/.devnode8 bash slurm/devrun.sh …`).

Nothing resolves outside the checkout: `env/activate.sh` puts the repo and the in-repo
`third_party/newton` submodule on `PYTHONPATH`, so editing the submodule takes effect with no
reinstall. There is no `uv`, no `.venv` and no sibling clone any more.

## Where things are

```text
sugar_newton/            the Newton line -- all current work
  validation/            single-purpose checks, each runnable alone
  rl/                    CarryBox env + the BCPPO training port
  tactile/reducer.py     PatchTactile: Newton contacts -> per-patch channels
  hand/patches.py        the anatomical patch layout
sugar_swap/              run SUGAR's UNMODIFIED IsaacLab code on Newton (reusable technique)
env/                     conda env spec, activate.sh, setup_env.sh
slurm/                   dev-node sbatch, devrun.sh, container + GL setup
third_party/newton       Newton, as a submodule -- must stay a clean upstream diff
SUGAR/                   vendored SUGAR: the policy, BCPPO, the frozen Refiner teacher
PLAN/ TODO/ DOCS/        design, task lists, the CarryBox reproduction record
claude_context/          the Plan-15 audit (findings.md) and operating notes
scripts/ IsaacLab/       the Plan-15 IsaacLab line -- does not run on this cluster
experiments/ legacy/      gitignored; local outputs only
```

Authoritative detail lives next to the code, not here. Read in this order:

1. [`sugar_newton/README.md`](sugar_newton/README.md) — the tactile core and the analytic
   validator, plus platform facts learned the hard way (contact-index round-tripping, the
   `add_shape_*` density trap, the compliant contact's ballistic envelope).
2. [`sugar_newton/rl/README.md`](sugar_newton/rl/README.md) — the long record: throughput,
   decimation, the closed loop, the `margin` bug, and what is faithful to SUGAR.
3. [`sugar_swap/README.md`](sugar_swap/README.md) — how to run an IsaacLab project on Newton
   by substituting modules instead of reimplementing the environment, written up as a reusable
   recipe. Read this before porting any further IsaacLab system by hand.
4. [`ASSETS.md`](ASSETS.md) — what the branch does not carry and how to get it.
5. [`sugar_newton/rl/SETUP.md`](sugar_newton/rl/SETUP.md) — the two pinning traps if the env
   is ever re-resolved.

## Current state

**Works and is measured.** The pretrained SUGAR tracker transfers to Newton and lifts the
box; BCPPO trains on Newton and logs to wandb; the tactile reducer is validated against
analytic ground truth on an inclined plane; the closed loop (policy → physics → render →
visual+tactile → policy) is instrumented end to end.

**Open, as of 2026-09-02: training a policy from scratch in Newton does not learn to
stand.** SUGAR's refiner stage is ported (`rl/train_refiner.py`: privileged 890-D actor,
plain PPO, `RB_STAGE=refiner`) and runs data-parallel on 8 GPUs at SUGAR's own 4096 worlds,
logging to wandb with eval video. It optimises for ~100 iterations, then one update
destroys the policy, and every evaluation shows a 0.000 m lift with no load-bearing
contacts. Two causes are identified and being fixed: the reward's **regularisation weights
are not SUGAR's** (scaled down by up to 1e-7, and `feet_air_time` at +5.0 — SUGAR's largest
positive term — is one of four omitted contact terms), and the refiner is plain PPO with no
BC curriculum, so it lacks the stabiliser BCPPO relies on. Details and numbers in
[`sugar_newton/rl/README.md`](sugar_newton/rl/README.md).

**The reward-fidelity half of that is being fixed structurally rather than by patching.**
[`sugar_swap/`](sugar_swap/README.md) runs SUGAR's *unmodified* source on Newton by
substituting Newton-backed modules into `sys.modules` for the Isaac Sim-specific parts of
IsaacLab, while reusing IsaacLab's managers, math and MDP term library verbatim. Isaac Sim is
never booted. As of 2026-09-02 SUGAR's ~3,300-line MDP package imports verbatim and its
refiner config constructs with all **21 reward terms at SUGAR's own weights** — including the
four `sugar_newton/` omitted — so the reward stops being something to re-audit. The write-up is
a reusable recipe: prefer it over hand-porting the next IsaacLab system.

**As of 2026-09-03 `sugar_swap` builds, steps and trains data-parallel on 8 GPUs at SUGAR's
exact 98,304-sample batch** — and it took one more environmental bug to get there. The released
SUGAR refiner checkpoint, which lifts the box in IsaacLab, could not lift it in Newton: a
self-collision that PhysX excludes and Newton did not was pinning the hip bow at 25° against
IsaacLab's 58°, so the hands never reached the box. With that fixed the same checkpoint
reproduces IsaacLab closely (pelvis pitch 58.1° vs 58.4°, lift 0.655 m against a 0.628 m mocap
reference, full-length episodes). Read
[the port caveat](#the-isaaclab-to-newton-port-sugar_swap) before interpreting any Newton
refiner curve: every run before this fix trained where the task was physically impossible, so
their reward plateau carries no information. A fresh 10,001-iteration run is in flight against
the IsaacLab reference.

Separately, the throughput gap that looked like a Newton-vs-PhysX engine problem is **not
one**. Measured head-to-head on one A100 on 2026-09-02, the entire gap is the collider
representation, and the solver iterations we blamed are nearly irrelevant:

| envs | configuration | env-steps/s |
|---|---|---|
| 512 | Newton baseline (triangle mesh, 100/50 iters) | 142 |
| 512 | Newton, convex colliders only | 6,010 |
| 512 | Newton, 8/4 solver iters only | 153 |
| 512 | Newton, PhysX-matched (convex + 8/4 + self-collisions) | 6,508 |
| 512 | IsaacLab (PhysX), SUGAR's own code | 4,949 |
| 4096 | Newton, convex colliders only | 13,209 |
| 4096 | Newton, PhysX-matched | 9,214 |
| 4096 | IsaacLab (PhysX), SUGAR's own code | 26,735 |
| 4096 | SUGAR's implied rate (1 GPU, ~1 day for 30001 iters) | 34,100 |

Convex hulls cut the collider from 310,692 triangles to 3,348 and buy **42x on their own**;
going from 100/50 to 8/4 solver iterations buys only **1.08x**. At equal world count Newton
is *faster* than IsaacLab (6,508 vs 4,949 at 512), but IsaacLab scales far better — its step
time grows only 1.48x for 8x the worlds against Newton's 5.65x, so IsaacLab is overhead-bound
at 512 while Newton is already compute-saturated there. At Newton's best rate the full
30001-iteration recipe is **~2.6 days on one GPU** rather than the ~240 days the
triangle-mesh baseline implied. Reproduction and per-row detail in
[`experiments/isaac/README.md`](experiments/isaac/README.md).

Note the convex rows are **not tactile-grade**: they exist to bound the achievable training
speed. Tactile work still needs the triangle-mesh path, so this is a per-purpose choice, not
a global switch.

Three silent-truncation bugs were found and fixed on 2026-08-31, and they invalidated every
number measured before them. The pattern is worth internalising because it recurred three
times: **a limit sized from measurements taken while a different limit was clipping.**

- The box weighed **4.39 kg instead of 0.5 kg** (8.78×). `ShapeConfig.density` defaults to
  1000 kg/m³ and `add_shape_mesh` *adds* the shape's mass to the body. The robot's links were
  unaffected because the URDF importer overwrites accumulated mass, which made the symptom
  look like a wrist-localised contact problem. It was not: the wrists were holding 8.8× the
  intended load. Check with `python -m sugar_newton.validation.check_masses`.
- `njmax` was 2048 while the playback path used 16384.
- Hydroelastic iso buffers dropped contacts at low SDF resolution (1596 times in one rollout).

With the mass corrected, the tracker reaches about two thirds of the reference lift —
consistently across clips, not a single lucky run:

| clip | lift | reference | fraction |
|---|---|---|---|
| `data_000` | 0.430 m | 0.628 m | 68 % |
| `data_001` | 0.458 m | 0.692 m | 66 % |
| `data_005` | 0.460 m | 0.643 m | 72 % |

**Throughput was the headline problem and is now understood.** It is collision detection, not
rigging, rendering, observations, launch overhead or the solver — measured by wrapping the
env's own `pipeline.collide` and `solver.step` in synchronising timers, where collide is 76 %
of the step at one world and 94.5 % at sixteen. Because the dominant term scales linearly with
world count, batching amortised nothing and aggregate throughput was flat at ~11–14
env-steps/s from 1 to 16 worlds.

The fix is quadric decimation of the box collider, `box_tris=2000`, a 50× triangle reduction
that moves the surface by 1.6 mm in the worst case — half the carton's wall thickness. It is
decimation, **not** hulling and not convex decomposition: the mesh stays non-convex and the
carton stays open, so the concavity the grasp depends on survives. That buys 15× throughput
(104 env-steps/s at 16 worlds, 207 at 512) and restores scaling with world count, while the
lift moves −0.7 % against a run-to-run spread of 0.07 m. SUGAR's ~5.8 M env-steps becomes
about 8 hours rather than about 5 days.

**The grasp was floating on a 4.5 mm air gap.** `default_shape_cfg.margin` was 0.005, an
artifact carried over from an older Allegro scene. In Newton the solver subtracts
`margin0 + margin1` from the separation, so a non-zero margin makes shapes *rest* that far
apart — it is PhysX's `restOffset`, not `contactOffset` (that is `gap`). Newton's own default
is 0. With `margin=0` the fingers penetrate −0.0 to 0.9 mm instead of hovering 4.0–4.7 mm
away, and the lift improves from 65 % to 69 % of reference at no cost in speed or stability.

**Open.** In rough priority order:

- The analytic validator **reports one failure** on CPU, and it is the test scene rather than
  the sensor. At the `critical + 5°` case the block accelerates off the measurement patch, so
  no contact is seen in any of the 40 measured steps. The slip channels themselves are fine:
  at `--mu 0.3` the 20° case is genuinely sliding and reports `gross_slip_fraction = 1.000`
  with nonzero slip velocity, and assertions 1–4 (normal load, utilization, exactly-zero slip
  under stick, utilization ceiling) pass at every seated angle. This is the weakness
  `sugar_newton/README.md` § Open already flags — the free-sliding assertions are qualitative
  and want a prescribed-velocity scene, where tangential velocity is an input not an outcome.
  Until then `python -m sugar_newton.validation.incline` exits non-zero on a healthy tree, so
  do not use it as a bare pass/fail gate.
- Absolute contact-force scale is unexplained: peaks swing between tens and thousands of
  newtons depending on the measurement path. Relative comparisons have been sound; absolute
  grip force is not yet quotable.
- The decimation error budget (`box_tris=2000`, `hand_tris=5000`) was justified against the
  5 mm margin that has since been removed, so it needs re-deriving. `plot_hand_tactile.py`
  still draws a 5 mm reference line.
- Contact area and peak pressure (Plan 16 §4, channels 9–10) are unimplemented; they need the
  hydroelastic contact surface. The friction *scale* path is GPU-only and has not run.
- Four reward terms remain omitted (`rewards.OMITTED`), all per-body contact-force terms.
  The tactile work has made those forces reachable, so the blocker is wiring — but read the
  audit first, because two of these exact terms were broken in the IsaacLab line.

## Rules that carry over

These came out of the audit and cost real campaigns. They are simulator-independent.

- **Never trust a reward term's weight column.** In the IsaacLab line `hoi_contact` carried
  `+1.0` but read contact sensors on links whose collision subtrees were deactivated at spawn,
  so it was dead; `undesired_contacts` carried `−1.0` across all 54 patches, so six patches in
  contact cancelled the entire achievable positive reward. Nothing raised.
- **A broken tactile sensor looks like the tactile-free branch winning.** The zero branch never
  touches the sensors, so any misconfiguration degrades only the branches under test — and the
  experiment's whole output is that comparison. Check contact counters before believing a
  zero-favourable result.
- **Name the evaluation view with every number.** Every published Plan-15 figure came from
  `--physical-outcome-view`, which suppresses all six SUGAR terminations.
- **A limit sized while another limit is clipping is not a measurement.** Three separate bugs
  in this repo had that shape.
- **Never let privileged signals into the actor**: measured object state, mass factor, jump
  flag, relative contact velocity, RGB and future frames are evaluation labels only.
- **`third_party/newton` must stay a clean diff against upstream.** The audit's sharpest
  finding was a local edit inside vendored IsaacLab that was indistinguishable from upstream by
  inspection and caused the shear leak. Do not fix Newton in place; report it.
- **Never commit `experiments/`, `legacy/`, checkpoints, traces or videos.**

## Prior line: Plan 15 on IsaacLab/PhysX

Complete and superseded. Kept because the null result and its audit are the reason for the
current design. It does **not** run on this cluster — that code targeted a separate host
rooted at `/public/home/yanhongru/Curiosity`, and every checkpoint, trace and video it cites
lives only there. A README path starting with `experiments/` will not resolve here.

The experiment: a frozen Refiner lifted the box, control passed to a student with no reset,
then mass and inertia changed online by `1.0/1.5/3/6/10×`. Three branches differed *only* in
tactile observation — `Z` exact-zero, `P` live patch channels, `PS` adding causal slip — with
three seeds each, 3000 updates, and 300 frozen rollouts per branch.

The result was null and then worse than null: at 3× the `PS−P` hold difference was `−0.2712`,
95 % CI `[−0.4655, −0.0667]`. Continuous patch signals did carry a real *information*
advantage (they preceded all 119 heavy-box drops, median lead 21 frames, versus 15 for binary
contact), but that never became a policy benefit.

The audit then found why the comparison could not be trusted, and also confirmed what was
sound — branch matching genuinely does not perturb the zero branch's physics, the mass event
and its readback are correct, and the eligibility gate is branch-invariant. Do not "fix" those.
Full detail, 115 findings with `file:line`, in
[`claude_context/findings.md`](claude_context/findings.md); the browsable version is
`claude_context/index.html` (`python3 claude_context/serve.py`). Design and reproduction
commands are in [Plan 15](PLAN/15_online_patch_tactile_mass_adaptation/plan.md) and
[DOCS](DOCS/sugar_carrybox_reproduction_full_record.md).
