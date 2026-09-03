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
four `sugar_newton/` omitted — so the reward stops being something to re-audit. Model
construction (`builder.py`) is the remaining piece before environments can be stepped. The
write-up is a reusable recipe: prefer it over hand-porting the next IsaacLab system.

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
