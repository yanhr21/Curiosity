# sugar_swap — run unmodified IsaacLab code on Newton

This package runs SUGAR's **original, unedited** source on the Newton physics engine. Isaac
Sim is never booted, and neither `SUGAR/` nor `IsaacLab/` is patched.

It exists because the alternative was tried first and went badly. `sugar_newton/` is a
from-scratch reimplementation of SUGAR's environment, and it drifted: four reward terms were
missing (including `feet_air_time`, weight **+5.0**), three regularisation weights were
scaled differently, and IsaacLab's `weight * dt` convention was not applied. Those gaps were
found by archaeology months later, and they are the leading explanation for the ported
refiner never learning to stand. Reimplementation makes fidelity a thing you audit forever;
substitution makes it structural.

The technique is general. It is written up here as a recipe because we expect to reuse it for
larger IsaacLab-based systems.

---

## 1. The idea

IsaacLab is not one thing. Its modules divide almost perfectly in two:

| kind | modules | depends on Isaac Sim? |
|---|---|---|
| Backend-agnostic | `utils.math`, `utils.string`, `utils.configclass`, `managers.*`, `envs.mdp.*` | No — pure torch/Python |
| Isaac Sim-specific | `assets`, `sensors`, `scene`, `sim`, `envs` (the env classes), `markers`, `terrains`, `actuators` | Yes — USD, PhysX tensor views |

The managers and the MDP term library only ever touch the simulator through a narrow data
interface: `asset.data.<attribute>`. So if you supply objects that present that interface and
are backed by a different engine, *all the logic above them runs unchanged*.

Python makes the substitution trivial, because `import` consults `sys.modules` before it
touches the filesystem. Register replacements there first and every later
`from isaaclab.assets import Articulation` — including the ones inside IsaacLab's own
managers — resolves to yours:

```python
from sugar_swap import bootstrap
bootstrap.install()          # must precede any isaaclab / sugar_rl import

import sugar_rl.tasks        # SUGAR's real code, now on Newton
```

The result for SUGAR: **~3,300 lines** of `commands.py`, `observations.py`, `rewards.py`,
`terminations.py` and `events.py` run verbatim, and so do IsaacLab's six managers and its
whole term library.

### What it needs at runtime

Verified on 2026-09-02 in the **Newton** environment (`env/activate.sh`) on an A100, with no
`isaacsim` or `omni*` package installed anywhere — which is the point: the swap removes the
Isaac Sim dependency rather than merely declining to use it.

IsaacLab and SUGAR are made importable by path, not installed. Four entries are needed, on
top of what `activate.sh` already exports:

```bash
source env/activate.sh
export PYTHONPATH="$PYTHONPATH:$RB_REPO/IsaacLab/source/isaaclab:$RB_REPO/IsaacLab/source/isaaclab_tasks:$RB_REPO/SUGAR/source/sugar_rl:$RB_REPO/SUGAR/source/sugar_il"
```

`sugar_il` is easy to miss and is not optional: SUGAR's `locomanip/mdp/commands.py` imports
its diffusion-policy wrapper at module scope, so the whole IL stack must be importable just
to build a config. That is also why six of the ten extra wheels below have nothing to do with
IsaacLab.

```bash
pip install toml prettytable h5py dill omegaconf diffusers einops zarr timm "gymnasium==1.2.0"
```

`gymnasium` is pinned to 1.2.0 to match the reference `isaaclab` env. These are **not** in
`env/environment.yml` yet; adding them is a deliberate decision about the env spec.

---

## 2. Why it turned out to be cheap

Three measurements decided feasibility. Repeat them before attempting this on a new project.

**The data interface is tiny.** Across all of SUGAR's MDP code there are only 19 distinct
`.data.*` attributes:

```bash
rg -o "\.data\.[a-z_]+" <project>/mdp/*.py | sort | uniq -c | sort -rn
```

**The Omniverse coupling in the reusable half is only logging.** `isaaclab.utils.math`
imports `omni.log`; the managers additionally call `omni.kit.app.get_app_interface()` (twice,
for debug-visualisation event streams) and `omni.timeline.get_timeline_interface()` (once, to
defer scene-entity resolution). Nothing else. All three are satisfied by `_stubs.py`.

That last one is worth knowing: `ManagerBase.__init__` only subscribes to the timeline when
`env.sim.is_playing()` is false. Our `SimulationContext.is_playing()` returns `True`, so
terms resolve synchronously and the timeline stub is never reached.

**Only one module in the term library is genuinely coupled.** `isaaclab/envs/mdp/events.py`
imports `carb`, `omni.physics.tensors`, `isaacsim.core.utils` and `pxr` at module scope.
Everything else in `envs/mdp/` needs at most `omni.log`.

```bash
rg -n "^\s*(import|from)\s+(omni|carb|pxr|isaacsim)" <isaaclab>/envs/mdp/
```

---

## 3. Module map

| file | role |
|---|---|
| `bootstrap.py` | The entry point. Substitutes everything, in a load-bearing order. |
| `_stubs.py` | `omni.log`, `omni.kit.app`, `omni.timeline`, `carb`, `pxr` stand-ins. |
| `lenient.py` | Config base that accepts any field and **records what Newton ignored**. |
| `shadows.py` | `isaaclab.sim` / `markers` / `terrains` / `actuators` — pure USD-authoring configs. |
| `data.py` | The heart: `ArticulationData` / `RigidObjectData` computed from Newton state. |
| `assets.py` | `Articulation`, `RigidObject`, and the `*Cfg` classes. |
| `sensors.py` | `ContactSensor`: Newton's contact list reduced per body, on the GPU. |
| `physx_view.py` | `root_physx_view` stand-in, so IsaacLab's randomisation events run unmodified. |
| `scene.py` | `InteractiveScene`: owns the Newton model and the index maps. |
| `env.py` | `ManagerBasedRLEnv` and `SimulationContext`. |
| `events.py` | Extracts four event terms out of IsaacLab's coupled `events.py` verbatim. |
| `tasks_utils.py` | Loads `isaaclab_tasks.utils` by file path, bypassing its Isaac Sim `__init__`. |
| `builder.py` | Builds the Newton model: convex-hull colliders, solver and collision pipeline, index maps. Carries the friction-cone decision; see Status. |

---

## 4. The three conventions that differ

Get these wrong and nothing crashes — the policy just trains on corrupted observations. They
are converted in exactly one place, `data.py`, and each is checked against ground truth.

| quantity | Newton | IsaacLab |
|---|---|---|
| Orientation | `body_q[3:7]`, **xyzw** | `root_quat_w`, **wxyz** |
| Body velocity | `body_qd` = (linear, angular) | separate `*_lin_vel_w` / `*_ang_vel_w` |
| Joint indexing | root free joint + object coords included | actuated joints only |

The quaternion conversions were verified to be bit-exact against IsaacLab's own
`quat_rotate_inverse` and `quat_apply`, and against the wxyz→xyzw mapping that our working
Newton env (`sugar_newton/rl/carrybox_env.py`) already uses when writing state. Free-joint
`joint_qd` is linear-then-angular, which differs from the usual Warp spatial-vector ordering,
so `scene.joint_dof_of_coord()` exists to keep position and velocity indices straight (a free
joint spans 7 coordinates but 6 degrees of freedom).

Two quantities Newton does not report at all are reconstructed the way IsaacLab does:
`joint_acc` by finite-differencing joint velocity across the step, and `applied_torque` from
the implicit PD law the solver was handed, clipped to the effort limit. These are the least
exact part of the swap. They are only consumed as squared penalties, which is why that is
tolerable — but it is a real difference, not a free one.

---

## 5. Verbatim reuse, including one unusual case

Reused unmodified from IsaacLab: `utils.math`, `utils.string` (so SUGAR's `joint_names_expr`
regexes resolve to *identical* joint orderings — get this wrong and the action vector is
silently permuted), `utils.configclass`, all six managers, the whole MDP term library, and
`isaaclab_tasks.utils.importer` (which decides which task modules get registered).

`events.py` is the unusual case. SUGAR needs four terms from IsaacLab's Isaac Sim-coupled
`events.py`, and those four touch the simulator *only* through `asset.root_physx_view` —
which `physx_view.py` implements. So rather than transcribe them (and risk changing the
randomisation distributions, the exact failure mode this package exists to prevent), we parse
the file and `exec` just those definitions. A `_FORBIDDEN` scan raises at import if a future
IsaacLab version adds a USD call to any of them, so the assumption fails loudly.

`randomize_joint_default_pos` is *not* in IsaacLab — it is SUGAR's own, and comes in verbatim
with the rest of SUGAR.

---

## 6. Fidelity accounting

Substitution can still silently drop physics. `lenient.py` records every IsaacLab config
field the Newton backend never reads, so the unfaithful surface is enumerable instead of
folkloric:

```python
from sugar_swap.lenient import report_ignored
print(report_ignored())
```

For SUGAR's refiner config that is **23 fields**. Most are cosmetic (light colour, MDL
materials, USD authoring). Three are physics-relevant and must be wired up explicitly in
`builder.py`:

- `solver_position_iteration_count` and `solver_velocity_iteration_count` — SUGAR uses PhysX's
  **8/4**, where our hand-written Newton port used 100/50. Benchmarking showed this is worth
  only ~1.08x throughput, so it is a fidelity issue rather than a speed one.
- `enabled_self_collisions` — SUGAR has these **on**; our port had them off. Worth ~1.43x.

Also dropped, and worth remembering: `RigidBodyPropertiesCfg` damping and
`max_depenetration_velocity`, and `RigidBodyMaterialCfg`'s friction *combine mode* (Newton
carries one `mu` per shape, so a randomisation that sets static and dynamic friction apart
collapses to the dynamic value).

---

## 7. Gotchas, in the order they appeared

Anyone repeating this will hit these. All are one-line fixes once understood.

1. **`isaaclab.utils.math` fails to import** even though it is pure torch — it imports
   `omni.log`. The coupling is at that level everywhere; do not conclude the code is dirty.
2. **`'omni' is not a package`.** A stub built with `types.ModuleType` needs `__path__` set
   before submodule imports of it will resolve.
3. **Whack-a-mole through IsaacLab's `__init__` chains.** `envs/mdp/__init__.py` does
   `from .actions import *`, which imports *every* action term, which imports every sensor and
   `pxr`. Names SUGAR never uses must still resolve. Prefer explicit unimplemented
   placeholders that raise on construction (see `sensors._unimplemented`) over permissive
   mocks, so an accidental dependency shows up instead of silently doing nothing.
4. **Marker presets are mutated at class-definition time.** `commands_cfg.py` runs
   `cfg.markers["arrow"].scale = ...` during import, so synthesised marker configs need a
   self-populating dict (`shadows._MarkerDict`), or import dies with `KeyError: 'arrow'`.
5. **Config groups must default to instances, not `None`.** SUGAR's `__post_init__` does
   `self.sim.dt = 0.005` without ever assigning `self.sim`, exactly as against IsaacLab's base
   class. `sim` also needs a `physx` sub-config.
6. **`sugar_rl.tasks.__init__` imports `isaaclab_tasks.utils`,** whose package `__init__`
   boots Kit and blocks on the EULA prompt. Load the individual source files by path instead
   (`tasks_utils._load_by_path`).
7. **Do not sync the GPU in the sensor.** `sugar_newton/tactile/handmap.py` reads the contact
   buffer with `.numpy()`, which is fine for one video frame and ruinous per training step.
   `sensors.py` masks against the contact count on-device with an `arange` and reduces with
   `index_add_`.
8. **Mass randomisation must write the inverse arrays.** Newton's solver consumes
   `body_inv_mass` / `body_inv_inertia`; writing only `body_mass` leaves the dynamics
   untouched, so the randomisation appears configured and does nothing.
9. **Do not stub a module the target environment really has.** The `pxr` stub exists for
   IsaacLab's task-space action terms, but the Newton environment ships a real `usd-core`,
   and stubbing over it silently breaks USD asset loading — SUGAR's carried box is a `.usd`,
   and Newton's importer does `from pxr import Usd` lazily, so the failure surfaces far from
   the cause. `_stubs._install_pxr` now defers to the real package whenever it imports. The
   general rule: a stub should be conditional on the real thing being absent.

10. **wandb refuses the config a resumed leg writes.** Reusing one wandb run across chained
    SLURM legs — the point of `bind_wandb_run` — means leg 2 calls
    `wandb.config.update({"runner_cfg": ...})` against a run that already has one, and the two
    differ because `resume` is False on leg 1 and True after. wandb's default is to refuse, so
    every resumed leg died with `ConfigError: Attempted to change value of key "runner_cfg"`
    *after* the environment was built and the checkpoint loaded — late enough to read as a
    training failure, and it burned a four-leg chain in six minutes. Two fixes look right and
    are not: `wandb.Settings(allow_val_change=True)` is consulted by `Config._sanitize` only
    under Jupyter (`wandb_config.py:277`), and there is no environment variable for it. rsl_rl
    builds the writer itself, so there is no argument to pass either. `run_dir.
    _allow_config_val_change` defaults the parameter on `Config.update` instead. Overwriting is
    correct, not a lesser evil: the later leg's config describes the process actually running.

11. **A fix that lands mid-run takes effect at the next resume, and can look like a resume
    bug.** `refiner_swap_4096` spiked its value loss 8x and dropped its reward from 10.5 to 4.2
    at exactly the step it resumed, which reads as a bad checkpoint. It was not: the
    `notify_model_changed` fix (gotcha 8) landed *while* that leg was running, and Python had
    already imported the old module, so the leg trained its whole life with randomisation
    silently discarded — nominal mass and friction in all 4096 envs. The resumed leg was the
    first to import the fixed module, so SUGAR's three `mode="startup"` terms bit for the first
    time: box mass log-uniform 0.5-2.0x with inertia recomputed, box friction 0.2-0.8, robot
    friction 0.3-1.6 with restitution 0-0.5. A critic calibrated on one nominal box is wrong
    once mass varies fourfold, and the task really did get harder. Reward recovered to ~8.7
    within fifteen iterations.

    Tell the two apart before touching the checkpoint code, because the distinguishing evidence
    is cheap. A save/load fault shows a **learning rate reset to the config default** and a
    **discontinuous action-noise std**; both were continuous here (7.59e-5 held, std 0.3768 ->
    0.3777), the surrogate loss never moved, and there was exactly one spike in 545 steps. The
    draw itself is reproducible across legs — `torch.manual_seed(seed + rank)` runs before env
    construction — so this is a one-time shift at the fix boundary, not a per-leg reshuffle.
    Note the corollary for cross-run comparisons: the draw depends on rank, so runs at
    different GPU counts get different randomisation samples.

12. **`--export=ALL` carries an interactive `WANDB_MODE` onto the cluster, and re-syncing an
    offline run DESTROYS its history.** Two traps, and the second is what turns the first from
    an annoyance into data loss. A stray `WANDB_MODE=offline` left in a submitting shell from
    unrelated debugging followed a production 4-node run onto the nodes (job 33725508), and
    nothing looked wrong: training ran, the id was minted and recorded in `run_meta.json`,
    checkpoints landed — the run simply never appeared in the UI. `swap_train_leg.sh` now pins
    `WANDB_MODE=${RB_WANDB_MODE:-online}` next to where it re-sources the key, for the same
    reason: the environment a leg inherits cannot be trusted.

    The recovery attempt is the sharper lesson. Running `wandb sync` on an offline run that is
    **still being written** replaces the run's history rather than appending — it went from
    263 logged rows to **zero**, keeping only a summary. Do not sync a live offline run to
    watch it progress. Either wait for the leg to finish and sync once, or sync to a separate
    id (`wandb sync <dir> --id <other> --no-mark-synced`), which leaves the directory eligible
    for a proper full sync later. `--append` exists for the incremental case but does not undo
    a history that has already been clobbered.

13. **The one-leg-at-a-time lock deadlocks a multi-node leg.** The heartbeat in
    `swap_train_leg.sh` guards against two independent *legs* sharing a log directory, but a
    4-node leg deliberately runs that script once per node. Nodes 1-3 found node 0's heartbeat,
    concluded a rival leg held the directory, and slept 25 minutes while node 0's `torchrun`
    blocked at a rendezvous they never joined. It presents as a hang with no error, and looks
    like a network problem rather than a lock. Only node 0 owns the heartbeat now
    (`SLURM_NODEID`); the others go straight to the rendezvous.

14. **The importer's parent filter does not look through a fixed joint, and that one gap cost
    the whole task.** This is the most expensive gotcha here, and the first thing to check when
    porting a new URDF. `pelvis` declares no collision geometry of its own; the pelvis collider
    lives on `pelvis_contour_link`, *welded* to `pelvis`. Newton filters shapes on the same body
    and on bodies joined directly by a joint, so it filtered `pelvis` — which has no shape, hence
    nothing — and left the real pelvis collider free to collide with `pelvis`'s own children,
    both `*_hip_pitch_link`s. MuJoCo's `filterparent` and PhysX's articulation adjacency rule
    both *do* look through: bodies with zero degrees of freedom between them are one rigid unit,
    and contacts are excluded within such a unit and between it and its parent unit. The hulls
    interfere from about 10° of hip flexion, where the motion needs 72°, and the constraint
    carried 20.6 kN against a 33 kg robot — so the released refiner checkpoint, which lifts the
    box in IsaacLab, bowed to 25° instead of 58° and never reached it. Note the failure did *not*
    look like a collision bug: no joint was at a position limit, no actuator was saturated, and
    there was 47 N·m of surplus drive against an 18.5 N·m gravity load. It looked like a
    tracking or tuning problem for days. `_weld_filter_pairs` applies the weld-aware rule and is
    the default; `RB_SELF_COLLISION=raw` reinstates the old behaviour for A/B, and `=0` filters
    everything, which is a diagnostic only because SUGAR trained with self-collision on. Beware
    the log line: it reports 176 pairs filtered, but 167 of those Newton had already excluded
    under its own rules, and only 9 are new — compare live pair counts, not that number.

15. **Convex hulling a thin cosmetic shell produces a solid blob.** MuJoCo only collides convex
    geoms, so `approximate_meshes("convex_hull")` runs on every mesh, and for shells the hull
    bears no resemblance to the part: `logo_link` 39.6×, `waist_support_link` 16.7×,
    `pelvis_contour_link` 15.9× (an 82 cm³ shell becomes a 1,307 cm³ lump filling the pelvis
    cavity), wrist roll links 10.7×, both `rubber_hand`s 2.35×. Genuinely convex links are fine
    (`hip_pitch_link` 1.08×, `pelvis` 1.09×). Hulling was adopted for collision performance and
    is what made gotcha 14 reachable at all. Fixing the pair filter is enough for the bow, so
    hulling is retained — but the rubber hands are the *grasping* surfaces, so their contact
    geometry is still coarser than PhysX's. The fix, if that matters, is convex decomposition
    into several pieces per mesh, which costs collision time. Hulling does **not** change
    `body_mass` for either the robot or the box; that was checked.

---

## 8. Status

Verified, with Isaac Sim never booting:

- `bootstrap.install()` succeeds.
- IsaacLab's six managers and its full MDP term library import and bind to Newton objects.
- `import sugar_rl.tasks.locomanip.mdp` — SUGAR's ~3,300 lines, verbatim.
- SUGAR's refiner config constructs with **21 reward terms and 6 termination terms**,
  including the four the hand-written port omitted (`feet_air_time` +5.0, `hoi_contact` +1.0,
  `undesired_contacts` -1.0, `feet_slide` -0.1) and the correct regularisation weights
  (-2.5e-07 / -1e-05 / -0.1). Reward faithfulness is now structural.
- Timing confirmed from SUGAR's own config: `dt=0.005`, `decimation=4`, i.e. 50 Hz control.

- `builder.py` is written; environments instantiate and step. Colliders are convex hulls
  because MuJoCo only collides convex geoms and the triangle midphase costs 42x. They are *not*
  a match for what PhysX simulates on the shell meshes — see gotchas 14 and 15.
- **Behaviourally validated against IsaacLab on identical weights**, which is a stronger check
  than zero-step equivalence and is what caught the self-collision bug. The released SUGAR
  refiner checkpoint replayed in `sugar_swap` reaches pelvis pitch 58.1° against IsaacLab's
  58.4°, pelvis height 0.709 m against 0.711 m, and lifts the box 0.655 m against a 0.628 m
  mocap reference, running full-length episodes with no load-bearing self-contact. Before the
  gotcha-14 fix the same checkpoint bowed to 25° and lifted 0.057 m.
- **Numerically equivalent to IsaacLab at zero step: 1080/1080 terms across 6 pinned states.**
  Given the same input state, every observation, reward and termination matches. After one
  substep the two diverge at ~1e-3, which is two different integrators and not a porting
  error — the harness threshold is 1e-5 precisely because a real convention bug is O(0.1).
  `experiments/equiv/` holds the harness; `diff_terms.py` prints the table.
- **8-GPU DDP verified and measured**: 26,534 env-steps/s at SUGAR's 4096-env batch, ~4.6x a
  same-node single GPU. The shortfall from 8x is not communication (the all-reduce is 0.10 s
  of a 3.70 s iteration) but that a 512-env rank does not saturate an A100.
- **Multi-node works, and what it costs is batch size, not efficiency.** 4 nodes x 8 GPUs form
  one 32-rank world (`swap_train32.sbatch`, static rendezvous with `MASTER_ADDR` resolved
  outside the container because the container has no SLURM client). The thing to size is
  **envs per rank, not GPU count** — an A100 saturates around 512 envs and idles below that,
  where fixed per-step cost dominates:

  | GPUs | envs/rank | total envs | env-steps/s | per GPU | efficiency |
  |---|---|---|---|---|---|
  | 1 | 4096 | 4096 | 5,028 | 5,028 | — |
  | 8 | 512 | 4096 | 40,461 | 5,058 | **100%** (8.0x linear) |
  | 32 | 128 | 4096 | 45,300 | 1,416 | **28%** |
  | 32 | 512 | 16384 | 149,682 | 4,678 | **92%** |

  Holding SUGAR's 4096-env batch across 32 GPUs is the trap: 128 envs/rank buys 1.12x over
  8 GPUs for four times the hardware. Filling the same four nodes to 512 envs/rank instead
  gives **3.3x the throughput at the same 2.6 s iteration** — the ranks were starved, not the
  interconnect. The price is that the update is 393,216 samples rather than SUGAR's 98,304, so
  it is a different experiment: SUGAR's ~10k-iteration refiner budget is ~2,500 iterations
  here. Scale the env count with the GPU count, or stay on one node.
- **Domain randomisation reaches the solver.** It did not before: `SolverMuJoCo` steps its own
  `mjw_model`, so every mass/COM/friction write was silently discarded until `physx_view.py`
  started calling `notify_model_changed`. `experiments/equiv/check_rand_propagation.py`
  asserts it, including a control that disables the notify and confirms the write then goes
  nowhere.
- **Randomisation is per-env and matches the configured distributions**, measured on the real
  refiner env at 512 envs by `experiments/equiv/check_rand_distribution.py`, reading
  `mjw_model` rather than the staging copy. Box mass: 512 distinct values spanning
  0.499-1.988x nominal, against a configured log-uniform 0.5-2.0 (nominal 0.50 kg, so
  0.25-1.00 kg). Box friction: exactly 64 distinct values, matching `num_buckets=64`, over
  0.202-0.790 against a configured 0.2-0.8. Robot friction: 64 distinct over 0.353-1.160.
  That propagation check is the narrower test — it writes ONE value to every env, so it
  cannot see a draw that reaches the solver but is identical everywhere, which is why this
  second script exists.
- **It is `mode="startup"`, not per-episode, and that is SUGAR's own choice.** Each env draws
  once at construction and keeps that body for the whole run; resetting every env leaves the
  masses bit-identical (max delta exactly 0). So the batch is a fixed population of 4096
  dynamics rather than a fresh sample per episode. Do not describe it as "randomising each
  training sample". The draw is reproducible across chained legs because
  `torch.manual_seed(seed + rank)` precedes construction — but it therefore depends on rank,
  so runs at different GPU counts train on different randomisation samples.
- **Two randomised properties do not survive MuJoCo, and both are silent.** Newton carries one
  friction coefficient per shape, so SUGAR's separate static and dynamic ranges collapse and
  `set_material_properties` keeps the dynamic one: the robot's configured static range
  0.3-1.6 is not represented, and measured friction tops out at 1.16, tracking the dynamic
  0.3-1.2. Restitution is worse — SUGAR randomises it 0.0-0.5, `Model` has
  `shape_material_restitution`, and `SolverMuJoCo` never reads it, so that term is a complete
  no-op. Neither is fixable without leaving MuJoCo, which models contacts through
  `solref`/`solimp` rather than a restitution coefficient; they are listed here so nobody
  reads the config and assumes both are live.

Remaining:

- End-to-end verification: the released refiner checkpoint should produce equivalent behaviour
  on the Newton backend. This is also the cleanest test of whether the 890-D privileged
  observation is assembled correctly, which the hand-written port got wrong.
- **The friction cone is the open decision.** `builder.py` sets `cone="elliptic"`,
  `impratio=20.0`; neither appears in SUGAR's config and MuJoCo's defaults are
  `pyramidal`/`1.0`. Pyramidal is 1.48-1.51x faster *and* closer to the reference — PhysX
  linearises friction into a pyramid and cannot express an elliptic cone — cutting the worst
  contact-force disagreement from 949 N to 67 N and taking step-1 agreement from 605/1170 to
  798/1170. Keep `impratio=20`: at 1.0 the pyramid is *slower* than elliptic (0.68x). It is a
  real dynamics change, so adopt it by retraining, not by switching under a trained policy.

---

## 9. Recipe for a new IsaacLab project

1. **Scope it** with the three commands in §2: the `.data.*` surface, the Omniverse imports in
   the reusable half, and the coupled modules in the term library. If the data surface is
   small and the coupling is logging, this approach works.
2. **Enumerate the import surface** the project actually needs, so the shadows are complete:
   ```bash
   rg -o "from (isaaclab[a-z_.]*) import ([^#\n]+)" -r '$1 :: $2' <project>/ | sort -u
   ```
3. **Reuse `_stubs.py`, `lenient.py`, `shadows.py` and `tasks_utils.py` as they are** — they
   are project-independent.
4. **Extend `data.py`** with any attributes the new project reads. This is the only file that
   should need real thought. Raise on anything unimplemented; never return zeros.
5. **Add sensors as needed.** `ContactSensor` is the hard one and is done; cameras and
   ray-casters are currently placeholders that raise.
6. **Check the fidelity report** (§6) and consciously decide, field by field, what to wire and
   what to accept losing.

The load-bearing rule: **`bootstrap.install()` must run before any `isaaclab` import.** It
raises if `isaaclab.assets` is already in `sys.modules`, because a substitution that arrives
late is worse than one that never happened — some call sites would hold the real class and
some the substitute.
