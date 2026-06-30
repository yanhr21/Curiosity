# Tactile Pipeline — Design & Architecture

Design notes for the tactile demos: solver setup, contact/sensor wiring, the
deformable-grasp coupling, and the video-composite pipeline. Results live in
`experimental_conclusions.md`; high-level overview in `../context.md`.

---

## 1. Three scene families

| Scene | Object | Solver(s) | Drive | Force signal |
|---|---|---|---|---|
| Rigid grasp (`tactile_video.py`) | rigid pen (capsule) | SolverMuJoCo + hydroelastic SDF | panda_hydro IK waypoints | `SensorContact` (world frame) + hydroelastic pressure |
| FEM indenter (`tactile_fem.py`) | soft block (`add_soft_grid`) | SolverVBD | prescribed sphere descent | Hertz `F=(4/3)E*√R d^1.5` from depth |
| FEM grasp (`tactile_rod_franka.py`) | soft rod (`add_soft_grid`) | Featherstone (robot) + VBD (rod) | Franka IK keyframes | (panels TODO) |
| FEM pencil, panda scene (`example_panda_soft_rod.py`) | round soft rod (`add_soft_mesh` cylinder) | **SolverMuJoCo (arm) + VBD (rod)** | original panda_hydro IK waypoints | VBD soft-contact compression [mm] |

## 2. Rigid grasp pipeline (`tactile_video.py`)

- Built on `newton.examples.robot.example_robot_panda_hydro.Example` (Franka + hydroelastic SDF,
  `SolverMuJoCo` solver="newton"). Two monkeypatches at import:
  - `HydroelasticSDF.Config.__init__` → force `output_contact_surface=True`.
  - `CollisionPipeline.contacts` → `request_contact_attributes("force")` before allocation.
- **Two `SensorContact`s**: (a) fingers←pen for per-pad grip/shear; (b) pen←fingers for
  shear *on the pen* (world frame, no projection) to settle direction-consistency questions.
- Per step: `ex.step()` → `ex.render()` → `solver.update_contacts` → `sensor.update` →
  read hydroelastic `get_contact_surface()`.
- **Pad window**: anchored to the pad's fixed geometry (origin + thin-axis normal, ±2 cm),
  not the moving contact points, so the heatmap doesn't balloon as the contact slides.
- **Material override**: `--material` monkeypatches `add_shape_capsule` to set
  `cfg.kh/mu/restitution/density`. Rigid materials share `RIGID_KH=1e12`.

## 3. FEM indenter pipeline (`tactile_fem.py`)

- `add_ground_plane` + `add_soft_grid` block + spherical indenter rigid body. `builder.color()`
  before finalize (required for VBD).
- `SolverVBD(iterations=10, integrate_with_external_rigid_solver=True,
  particle_enable_self_contact=False, particle_enable_tile_solve=False,
  rigid_contact_hard=False, rigid_body_particle_contact_buffer_size=512)`.
- Indenter is **kinematically prescribed** (set `body_q` + `body_qd` each substep). VBD with
  `external_rigid=True` does not integrate it, so prescription is required.
- `lame(E, ν)` → `k_mu=E/(2(1+ν))`, `k_lambda=Eν/((1+ν)(1−2ν))`.
- Force is **analytic Hertz** (displacement is prescribed; VBD prevents penetration so a
  penalty-force read is ~0). Honest label in the panel: "Hertz normal force".

## 4. FEM grasp coupling (`tactile_rod_franka.py`) — the important one

Adapted from `newton.examples.softbody.example_softbody_franka` (Franka + rubber duck). One
`ModelBuilder` holds the Franka URDF, table, ground, and the soft rod. **Two solvers, one
state, alternating each substep:**

```
per substep:
  clear_forces
  # robot as KINEMATIC INTEGRATOR (produces body velocities for friction):
  particle_count = 0; gravity = 0; shape_contact_pair_count = 0
  state.joint_qd = target_joint_qd            # = (IK target_q - current_q)/frame_dt
  robot_solver.step(...)                       # SolverFeatherstone
  particle_f.zero_(); restore particle_count + gravity
  # deformable + contacts:
  collision_pipeline.collide(state, contacts)
  soft_solver.step(...)                        # SolverVBD, external_rigid=True
  swap states
```

- **Why Featherstone, not prescribed `body_q`**: a teleported body carries no velocity, so VBD
  applies no tangential (static-friction) impulse → the rod slips. Featherstone integrates the
  arm from `joint_qd` velocity targets, giving bodies real velocities VBD friction can act on.
- **MuJoCo also couples (see §6).** Featherstone is the shipped-example recipe, but MuJoCo works
  too: VBD only *reads* `body_q`/`body_qd`, so any rigid solver that updates them drives friction.
  MuJoCo is preferred for the panda scene (keeps the exact arm dynamics + hydroelastic look).
- IK: `newton.ik.IKSolver` (position + rotation + joint-limit objectives), EE link_offset
  `(0,0,0.22)` along the tool axis. Gripper fingers set directly via `set_gripper_q` kernel
  (last two joint coords), `finger_pos = activation·0.04`.
- Contact material: `model.soft_contact_ke=2e6, kd=2e-1, mu=1.0`; `shape_material_mu=1.5`.
- **Grasp tuning that worked**: rod 10.8×1.8×2.4 cm (E=1e6, ρ=200), spawn bottom 5 mm above
  the table (avoid launch), descend EE to pz=0.20 to straddle the short rod, `gripper_close=0.18`
  (close hard on the 18 mm width). Result: gripped + lifted ~180 mm.

## 5. Video composite pipeline (shared)

- Headless GL: `pyglet.options["headless"]=True` before importing newton; `ViewerGL(headless=True)`;
  `viewer.log_state(state)` renders deforming soft bodies; `viewer.get_frame().numpy()` →
  `(H,W,3)` uint8.
- Two-pass: pass 1 simulates + caches scene PNGs + measurement arrays; pass 2 composites with
  matplotlib (scene + heatmap via `scipy.interpolate.griddata` linear + trace panels), then
  `ffmpeg` to mp4. Custom diverging colormap dark-blue→red for pressure/compression.

## 6. Soft pencil in the EXACT panda scene (`example_panda_soft_rod.py`) — the faithful variant

Reproduces `tactile_material_metal.mp4` with a soft pencil by **forking the original example file**
(`example_robot_panda_hydro.py`) and changing ONLY the object + solver — everything else (arm,
pads, table, cup, camera, IK waypoints, hydroelastic overlay, shape colors, ground) is byte-for-byte
the original. A from-scratch `ModelBuilder` is wrong: shape colors are a deterministic palette
indexed by shape order, so a rebuild silently recolors the scene and drops pieces.

- **Object**: rigid capsule (`add_shape_capsule`, r=0.005, L=0.14) → round soft FEM rod via
  `add_soft_mesh(vertices, indices)` with a local `cylinder_tetmesh(radius, length)` helper
  (structured center-fan prisms → 3 tets each, reordered to positive volume; a **flat facet
  seated at the bottom** so it doesn't roll). No rigid "object" body.
- **Solvers**: original `SolverMuJoCo` arm + `SolverVBD` rod, one model/state, alternating each
  substep (`collide → mujoco.step → vbd.step → swap`). `builder.color()` before finalize;
  `ShapeFlags.COLLIDE_PARTICLES` on every rigid shape; run **eager** (`self.graph=None`).
- **Stiffness**: `E=2e7, ν=0.45, ρ=200`, **50 VBD iterations** — iterations (not E) set apparent
  stiffness; with few iterations the rod stays floppy at any E (see conclusions §7).
- **Grip**: one-way coupling means the rod can't stop the rigid fingers, so the original
  `gripper=0.06·(1−t)` (full close) crushes/squirts it; remap the *closed* finger pos to a
  rod-thickness gap (`GRIP_CLOSE≈0.016`). Grasp reaches slightly lower (`grasping_offset z=0.12`).
- **Tactile panels** (`tactile_soft_rod_video.py`): a soft body has no hydroelastic surface, so the
  pad heatmap shows **VBD soft-contact compression [mm]** (`contacts.soft_contact_*` masked to the
  pad shape), grip force ≈ `kₑ·Σcompression·area`, plus a soft-body rod bend & height panel. The
  pad view is **pad-centric** exactly like the rigid one (fixed pad frame, gravity-up, rod
  projection moving within the window). `stitch_compare.sh` hstacks metal|wood|soft with FPS labels.
- **Performance**: soft (MuJoCo + VBD 50 it, eager) ≈ **5 fps** vs rigid (MuJoCo + hydroelastic,
  CUDA graph) ≈ **82 fps** — ~16× slower (VBD iterations + eager + per-substep collision).
