# Newton — Working Context

Living summary of the environment, findings, tools, and gotchas from the Newton
physics-engine work (FPS-vs-collision benchmarking and tactile-data extraction).
Detailed scratch notes live in `claude_context/` (currently empty).

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

## 5. Technical gotchas

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

## 6. File inventory (repo root)

- Benchmarks: `bench_fps_vs_collision.py`, `bench_panda_hydro.py`, `plot_*.py`, `run_*.sh`,
  `run_panda_hydro_kh.py`.
- Tactile: `tactile_video.py`, `tactile_frame420.py`, `tactile_push_probe.py`, `tactile_probe.py`.
- Generated (gitignored): `*.usd`, `results_*.csv`, plot/diagnostic `*.png`, `*.mp4`,
  `tactile_frames*/`, `demo_logs/`.
