# Newton — Working Context

Living summary of the environment, findings, tools, and gotchas from the Newton
physics-engine work (FPS-vs-collision benchmarking, tactile-data extraction, and
multi-material / FEM tactile demos). Detailed notes live in `claude_context/`:
`tactile_pipeline.md` (design/architecture) and `experimental_conclusions.md`
(results). Action items in `TODOs.md`.

---

## 1. Environment & repo

- **conda env `newton`** (Python 3.12): `newton 1.4.0.dev0` (editable from this repo),
  `warp-lang 1.14.0`, `mujoco-warp 3.8.1`, `usd-core 26.3`, `trimesh`, `scipy`,
  `matplotlib`, and `torch` (cu128, added for the policy/anymal examples).
- Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate newton`.
- Run examples: `python -m newton.examples <name> [--viewer gl|usd|null] ...`.
- Hardware: 2× NVIDIA RTX 6000 Ada (48 GB). Pin per-process with `CUDA_VISIBLE_DEVICES`.
- **Remote**: pushed to `mikewang` =
  `ssh://git@gitlab-master.nvidia.com:12051/shengzew/newton_mikewang.git`.
  Work branch: `shengzew/fps-collision-benchmarks` (main = pristine upstream Newton).
  The clone was shallow + missing LFS; had to `git fetch --unshallow` and
  `git lfs push --all` before the first push.

## 2. Demos (USD exports)

All 9 robot examples were rendered to USD (gitignored):
`allegro_hand, anymal_c_walk, anymal_d, cartpole, g1, h1, panda_hydro, policy, ur10`
via `python -m newton.examples <name> --viewer usd --output-path <name>.usd --headless`.
`anymal_c_walk` and `policy` require `torch`.

USD note: `/root/model/com` etc. are **debug overlays** (center-of-mass points,
inertia boxes, joint frames) written by the viewer — the real geometry is under
`/root/geometry/*`. COM data lives in timeSamples (animated), not the default value.

## 3. FPS vs collision-quality study

**Scripts:** `bench_fps_vs_collision.py` (XPBD box-pyramid, sweep `--iterations`),
`bench_panda_hydro.py` (drives `panda_hydro`, overrides `iters/ls/impratio/kh/substeps`,
reports penetration + grasp-lift + blow-up), `plot_results.py`, `plot_panda.py`,
`plot_kh_coupling.py`, `run_sweep_*.sh` / `run_kh_*.sh`.

**Penetration metric** (solver-agnostic, computed in numpy from contact buffers):
`d = dot(n, bx_b − bx_a) − (margin0+margin1)`; penetration `= max(0, −d)`. Negative = penetrating.

**Findings:**
- **XPBD boxes**: penetration ∝ 1/iterations, FPS ∝ 1/iterations — a genuine speed↔quality dial.
- **panda_hydro** (MuJoCo "newton" solver + hydroelastic SDF): the usual numerical knobs are
  *not* effective penetration levers — `iterations` saturates by ~5, `ls_iterations` does
  nothing, `substeps`/`impratio` only cost FPS. The lever is **`kh`** (hydroelastic stiffness).
- **`kh` trades against `dt`, not FPS.** MuJoCo's implicit integrator caps the *effective*
  stiffness at ≈1/dt: at large dt (substeps=2) `kh` is inert (1e11=1e14=1e15 all ~1.1 mm);
  at substeps=10 `kh` helps down to a dt-floor (~0.13 mm by `kh=1e13`) then saturates. To go
  lower you must raise `kh` AND substeps together → FPS cost returns (0.033 mm cost 25 FPS vs
  73 default). No blow-ups (the implicit cap absorbs the stiffness). Library default `kh=1e10`,
  example uses `1e11` — moderate for cross-scene/mass/dt stability.
- **panda_hydro cost is collision-bound** (SDF queries), not solver-bound; `sdf_max_resolution`
  is the real FPS lever.

## 4. Tactile data extraction

**Scripts:** `tactile_video.py` (full pick-place tactile video: scene render + per-pad
pressure maps + grip-force trace + shear-on-pencil graph; supports a downward-push mode),
`tactile_frame420.py` (single-frame diagnostic: 3D world scatter + pen outline + discrete
eval-point overlay), `tactile_push_probe.py` (headless push validation), `tactile_probe.py`.

**APIs:**
- `SensorContact` (`newton.sensors`): `total_force`, `total_force_friction` per sensing object;
  `force_matrix`, `force_matrix_friction` per counterpart — **all world frame**. `[i,j]` = force
  on sensing object `i` from counterpart `j`. Create the sensor **before** `model.contacts()`
  (it requests the `force` extended attribute). Per step: `solver.step` →
  `solver.update_contacts(contacts)` → `sensor.update(state, contacts)` → read `.numpy()`.
- Hydroelastic contact surface (dense spatial map):
  `collision_pipeline.hydroelastic_sdf.get_contact_surface()` → `ContactSurfaceData`
  (`contact_surface_point` = 3 verts/face, `contact_surface_depth` = signed depth/face,
  `contact_surface_shape_pair`, `face_contact_count`). Enable with
  `HydroelasticSDF.Config(output_contact_surface=True)`.

**Sign conventions (verified, not assumed):**
- Gravity: `model.gravity = [0,0,−9.81]` (read). −Z is down.
- Force: on the *sensing object*, world frame; +z = up (verified: resting sphere reads +mg up).
- **Depth is SIGNED, negative = penetrating.** The hydroelastic law is
  `pressure = −kh·signed_depth` (`linear_pressure`, `sdf_hydroelastic.py:128`), positive only
  for `signed_depth < 0`. So **pressure = kh·max(0, −depth)**.

**Pad geometry:** each finger body carries the URDF collision mesh(es) **plus** an added pad
mesh (highest shape index → `pad = max(shapes on finger body)`). Pad mesh ≈ 4×6 cm; its face
normal is the thin local axis. Contacts confirmed to be **only `pad↔pen`** even under hard push.

**Visualization choices (in `tactile_video.py`):**
- Pad pressure map: pressure interpolated (linear; cubic overshoots on near-collinear points)
  from the contact-surface face centroids (~85 points at grasp, ~1 mm spacing set by
  `sdf_max_resolution=64`).
- Axes gravity-aligned (v = −gravity-in-plane, per frame); window **anchored to the pad's fixed
  geometry** (center = pad origin, normal = pad thin axis, fixed ±2 cm) so it doesn't balloon as
  the contact slides along the long pad.
- Overlays: pen outline (cyan stadium, projected capsule), aggregate shear arrow (white), gravity
  reference (gold "g"), and the shear-on-pencil time graph.

**Findings:**
- **Holding the light pen** (weight ≈ 0.11 N): the two pads apply an **opposing vertical shear
  couple** (`cos∠(L,R) ≈ −1`), net ≈ 0. Verified *not* a frame bug — pen is static (not falling),
  both pads' "up" axes are identical (`v_hat_L·v_hat_R = +1`), and the opposition is in the raw
  world-frame sensor output.
- **Pushing the pen into the cup bottom** (`--push-after`): the cup supplies the upward reaction,
  so **both pads shear the pen downward together** (`cos∠ → +1`, ~12–17 N each) — same-direction
  shear under a real common load. Measured directly on the pen via
  `SensorContact(sensing="object", counterpart=fingers)` — world frame, no projection.

## 5. Multi-material & FEM tactile demos

**Goal:** show how a robot would sense different materials (metal, rubber, plastic,
wood, porcelain), with per-material videos (scene + pressure maps + measurements).

**Material model decision (verified reasoning):** all *rigid/hard* materials share
the **same rigid `kh`** — hydroelastic stiffness does not differentiate rigid contact;
they differ only by **friction (`mu`), density, and restitution**. Only *genuinely
deformable* materials become FEM (differentiated by **Young's modulus `E`**). Coupling
softness to a low `kh` fails: it causes deep penetration (grip slips), so soft = FEM.

**Rigid materials** (`tactile_video.py`, `--material`): `RIGID_KH = 1e12`; `MATERIALS` =
wood/metal/porcelain/glass differing only by `mu`/`restitution`/`density`. Material applied
by monkeypatching `add_shape_capsule` to set `cfg.kh/mu/restitution/density` on the pen.
Full 720-frame pick-and-place videos rendered: `tactile_material_{metal,wood,porcelain}.mp4`.

**FEM indenter** (`tactile_fem.py`): a spherical indenter presses a soft FEM block
(`add_soft_grid`, `SolverVBD`). Materials gel/soft_rubber/rubber/firm_rubber (E = 5e4 … 5e6,
ν=0.45). Displacement-controlled (prescribed descent to 12 mm); per-material force is the
**Hertz normal force** `F = (4/3)·E*·√R·d^1.5` (`E* = E/(1−ν²)`) from the simulated depth.
4 mp4s rendered; force span at 12 mm indent: gel 15.5 N → soft_rubber 62 N → rubber 311 N →
firm_rubber 1554 N. Composite = deforming scene + radial pressure map + indentation/area/force.

**FEM Franka rod grasp** (`tactile_rod_franka.py`) — the deformable pick-and-place. A Franka
Panda grasps a **deformable rubber rod** (`add_soft_grid`, E=1e6) through a full
approach→descend→pinch→lift→carry→place→release sequence. Adapted from
`newton.examples.softbody.example_softbody_franka` (Franka + rubber duck). Confirmed in sim:
the rod is gripped by friction and **lifted ~180 mm** off the table, deforming while held;
`tactile_rod_franka.mp4` (1110 frames) rendered. (Tactile measurement panels still to add.)

**Soft pencil in the EXACT panda_hydro scene** (`example_panda_soft_rod.py` + composite
`tactile_soft_rod_video.py`) — the faithful soft variant of `tactile_material_metal.mp4`.
The original example file is **forked verbatim** and only the object+solver change: the rigid
capsule pencil → a **round soft FEM rod** (a tetrahedralized cylinder of the same 0.5 cm
radius / 14 cm length via `add_soft_mesh` + a local `cylinder_tetmesh` helper; Newton has no
soft primitive, so a soft body must be a tet mesh). Arm stays on `SolverMuJoCo`; the rod is
`SolverVBD`. `builder.color()` before finalize; `ShapeFlags.COLLIDE_PARTICLES` set on the
rigid shapes; run **eager** (VBD's per-frame BVH rebuild does not CUDA-graph-capture). Rendered
720-frame `tactile_material_rubber_soft.mp4` + 3-way `tactile_compare_metal_wood_softrubber.mp4`
(metal/wood/soft labeled with sim FPS). Committed on `shengzew/fps-collision-benchmarks` (55958d82).

**Coupling architecture (the key to a friction grasp):** rigid robot and FEM rod live in
**one model / one state**; two solvers step it alternately each substep. Either rigid solver
works because `SolverVBD(integrate_with_external_rigid_solver=True)` only *reads* `body_q`/
`body_qd` from the state (for particle-body contact + friction) and never moves rigid bodies —
so whichever solver updated those poses drives the friction:
- **Featherstone** (`tactile_rod_franka.py`, cup scene): run as a **kinematic integrator** —
  set `joint_qd` to the IK velocity target, zero gravity + particle_count during its step, then
  restore. Gives real body velocities VBD uses for friction.
- **MuJoCo** (`example_panda_soft_rod.py`, panda scene): keep the original `SolverMuJoCo` arm —
  MuJoCo silently ignores particles, and VBD reads its updated body velocities. **One-way**
  coupling (the rod never pushes the rigid fingers back). Earlier notes said MuJoCo "does not
  couple to VBD" — that was a convention of the shipped examples, not a hard limit; it works and
  is *more* faithful (preserves the exact panda_hydro arm dynamics, hydroelastic overlay, scene).

## 6. Technical gotchas

- **Headless GL render** (no X): set `pyglet.options["headless"]=True` *before* importing
  newton/pyglet, use `ViewerGL(headless=True)`, capture with `viewer.get_frame().numpy()` →
  `(H,W,3)` uint8 (top-left origin). Uses EGL (present); no `xvfb` on this box.
- **mujoco_warp + CUDA graphs**: re-capturing an already-captured graph (e.g. calling the
  example's `capture()` twice to change substeps) → `CUDA error 700 illegal memory access`.
  Capture once; to change substeps, skip the in-`__init__` capture and capture once afterward.
- `contacts.force` (the `force` extended attribute) is a **6-vector per contact** (wrench), not vec3.
- numpy 2 removed `ndarray.ptp()` → use `np.ptp()`.
- **Trajectory override**: to push the gripper down after grasping, replace `ex.set_joint_targets`
  with a custom function that sets the IK target lower each frame and keeps the grip closed
  (`install_push` in `tactile_video.py`).
- **VBD + free rigid body**: with `integrate_with_external_rigid_solver=False`, VBD wrongly integrates
  a free rigid body (a gravity-only indenter *rises*); with `=True` it leaves the body fixed
  (expects an external solver). For a kinematic indenter, use `=True` and **prescribe `body_q`**
  each substep. To make a rigid body *dynamic*, drive it with a real solver (e.g. Featherstone).
- **Prescribed kinematic body transmits NO friction.** Overwriting `body_q` every substep
  (teleport) drives *normal* force fine (squeeze/indent works) but resets the contact/friction
  anchor each step, so no *static* friction accumulates — a prescribed jaw squeezes a soft rod
  but can never grip-and-lift it (`tactile_rod_probe.py` is the negative result). Friction grasp
  needs solver-integrated body velocities → Featherstone-as-kinematic-integrator + VBD.
- **FEM block initial penetration → launch**: spawn a soft body so its bottom particles sit
  ~one `particle_radius` above the support; placing them exactly on the surface ejects/launches
  the body at frame 0 (and it can then slide off-center).
- **VBD stiffness is iteration-limited, not modulus-limited.** A stiff FEM rod stays floppy if
  VBD has too few iterations *regardless of E* — raising E from 8e5→1e8 gave the **same** bend at
  5 iterations. The lever for a stiff/straight rod is **iterations** (≈50), not E. Past the
  convergence limit, more E does nothing.
- **A round soft rod rolls.** A cylinder rod rolls/slides off the table before the grasp. Seat a
  small **flat facet at the bottom** (offset the cross-section ring phase by ½ step) and spawn at
  exact rest height (no bounce).
- **One-way coupling → close the gripper to a rod-thickness GAP, not 0.** The soft rod can't push
  the rigid fingers back, so commanding full close crushes/squirts it (and the rigid pads jam on
  each other at ~30 mm finger gap). Command the closed finger pos so the pad gap ≈ rod diameter
  (`GRIP_CLOSE≈0.016` for the 1 cm rod).
- **Match a reference scene by FORKING its code, not rebuilding.** GL shape colors are a
  deterministic palette indexed by shape order, so a from-scratch `ModelBuilder` recolors
  everything (yellow→cyan table, magenta→blue cup) and easily drops pieces (`add_ground_plane`).
  Copy the original example file and change only the object/solver.
- **Pad heatmap is PAD-centric, not object-centric**: anchor the window to the pad's fixed frame
  (pad origin + thin-axis normal, gravity-up vertical) in the finger's local frame; the object/rod
  projection then *moves* within the fixed window as the contact slides (same for rigid + soft).

## 7. File inventory (repo root)

- Benchmarks: `bench_fps_vs_collision.py`, `bench_panda_hydro.py`, `plot_*.py`, `run_*.sh`,
  `run_panda_hydro_kh.py`.
- Tactile (rigid grasp): `tactile_video.py`, `tactile_frame420.py`, `tactile_push_probe.py`,
  `tactile_probe.py`.
- Tactile (FEM): `tactile_fem.py` (sphere-on-block indenter), `tactile_rod_franka.py` (Franka
  grasps deformable rod — the working pick-and-place), `tactile_fem_probe.py` (VBD indenter
  validation), `tactile_rod_probe.py` / `tactile_rod_render.py` (prescribed-jaw grasp — negative
  result: squeezes but cannot lift), `fem_test.py` (early feasibility).
- Tactile (FEM, panda scene — the faithful soft-rod deliverable, committed): `example_panda_soft_rod.py`
  (fork of `example_robot_panda_hydro` with a round soft FEM pencil, MuJoCo arm + VBD rod),
  `tactile_soft_rod_video.py` (its composite, pad-centric compression), `render_soft_rod.py`
  (single-frame preview driver), `stitch_compare.sh` (metal|wood|soft side-by-side with FPS labels).
  Superseded reconstructions (NOT committed): `tactile_soft_rod.py`, `tactile_pen_soft.py`,
  `tactile_rod_{cup,panda}.py` (from-scratch rebuilds — wrong scene/colors).
- Generation pipeline (submodules + `genpipe/`): PixelDiT (text→image) → TRELLIS.2 (image→3D GLB) →
  Newton. `genpipe/{trellis_image_to_glb.py, run_pipeline.sh, README.md}`; submodules under
  `third_party/{PixelDiT,TRELLIS.2}`. Newton clock scene: `example_panda_clock_metal.py` (fork of
  `example_robot_panda_hydro.py` — rigid TRELLIS-generated clock, metal material, grip + release-from-
  high into cup) and `tactile_clock_metal.py` (tactile measurement composite → `tactile_material_clock_metal.mp4`).
  Profiling + fleet scaling (10k/1M objects) in `claude_context/experimental_conclusions.md` §8 and `genpipe/README.md`.
  **Setup+run+cluster guide: `genpipe/RUNBOOK.md`** (authoritative; env bootstrap `genpipe/cluster_setup.sh`).
  Cluster (oci-ord Slurm) bring-up in progress — repo cloned + `pixeldit` env done; `trellis2` env must
  finish inside a Slurm job (login node kills heavy builds). State in `experimental_conclusions.md` §9.
- Generated (gitignored): `*.usd`, `results_*.csv`, plot/diagnostic `*.png`, `*.mp4`,
  `tactile_frames*/`, `fem_frames*/`, `rodfr_frames/`, `*.log`, `demo_logs/`, `pipeline_out/`,
  `tactile_clock_frames/`.
