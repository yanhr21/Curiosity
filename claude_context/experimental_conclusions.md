# Experimental Conclusions

Results and verified findings. Design/architecture in `tactile_pipeline.md`;
environment + gotchas in `../context.md`.

---

## 1. FPS vs collision quality

- **XPBD boxes**: penetration ∝ 1/iterations and FPS ∝ 1/iterations — a genuine speed↔quality dial.
- **panda_hydro (MuJoCo "newton" + hydroelastic SDF)**: `iterations` saturates by ~5,
  `ls_iterations` is inert, `substeps`/`impratio` only cost FPS. The effective penetration lever
  is **`kh`**, but it trades against **`dt`**, not FPS: MuJoCo's implicit integrator caps effective
  stiffness at ≈1/dt.
  - substeps=2: `kh` inert (1e11=1e14=1e15 → ~1.1 mm).
  - substeps=10: `kh` helps to a dt-floor (~0.13 mm by `kh=1e13`), then saturates.
  - To go lower, raise `kh` **and** substeps together → FPS cost returns (0.033 mm @ 25 FPS vs
    73 FPS default). No blow-ups (implicit cap absorbs stiffness).
- panda_hydro is **collision-bound** (SDF queries); `sdf_max_resolution` is the real FPS lever.

## 2. Tactile grip on the rigid pen

- **Per-pad force metric**: use per-pad magnitude `‖tf[i] − tfr[i]‖`, NOT the vector sum of both
  pads (`(tf−tfr).sum(0)` ≈ 0 because the two pads oppose — earlier this wrongly read as "no contact").
- **Holding the light pen** (≈0.11 N): the two pads apply an **opposing vertical shear couple**
  (`cos∠(L,R) ≈ −1`), net ≈ 0. Verified *not* a frame bug (pen static; both pads' up-axes identical;
  opposition is in raw world-frame sensor output).
- **Pushing the pen into the cup bottom** (`--push-after`): cup supplies the upward reaction, so
  **both pads shear the pen downward together** (`cos∠ → +1`, ~12–17 N each).

## 3. Material model

- Confirmed reasoning: **rigid/hard materials share one rigid `kh`**; stiffness does not
  differentiate rigid contact. They differ only by **friction, density, restitution**.
- **Soft materials must be FEM** (Young's modulus `E`), not a low `kh` — a low `kh` couples
  softness to deep penetration and the grip slips (rubber at `kh=2e8` penetrated 1.31 mm / 26% of
  the pen radius and was not lifted).

## 4. FEM indenter — Hertz force span (12 mm prescribed indentation, ν=0.45)

| Material | E [Pa] | Hertz force |
|---|---|---|
| gel | 5e4 | 15.5 N |
| soft_rubber | 2e5 | 62 N |
| rubber | 1e6 | 311 N |
| firm_rubber | 5e6 | 1554 N |

Force scales cleanly with stiffness at fixed indentation (~100× over the E range). Block deforms
realistically (surface dimples, sides bulge, no pass-through).

## 5. Deformable grasp — what holds vs what slips

- **Prescribed-jaw grasp (`tactile_rod_probe.py`): FAILS to lift.** Across 5 configs (pinch depth,
  `ke`/`kf`/`mu` up to 5.0, iterations up to 20, lift speed, stiffer rubber, longer jaws,
  `rigid_contact_hard=True`) the rod **squeezes in Y but is never dragged up in Z** — it stays on
  the ground. Root cause: teleporting `body_q` each substep resets the friction anchor → no static
  friction. Visual confirmation: `tactile_rod_probe.mp4` (jaws lift, rod stays on the table).
- **Featherstone+VBD grasp (`tactile_rod_franka.py`): WORKS.** Franka grasps the deformable rod by
  friction and **lifts it ~180 mm** (rod mean-z 218 → 398 mm; zmin 204 → 379 mm), deforming while
  held. `tactile_rod_franka.mp4` (1110 frames) rendered. The fix is solver-integrated body
  velocities (Featherstone as kinematic integrator), per `example_softbody_franka`.
- Grasp tuning: rod must spawn slightly above the table (else launched at frame 0), EE must descend
  far enough to straddle the short rod (pz=0.20), and the gripper must close hard (`gripper_close=0.18`)
  on the 18 mm width.

## 6. Soft pencil in the EXACT panda scene (MuJoCo + VBD) — what worked

Faithful soft variant of `tactile_material_metal.mp4` (`example_panda_soft_rod.py`, forked from
the original example; composite `tactile_soft_rod_video.py`).

- **MuJoCo + VBD couple on one model (one-way).** MuJoCo silently ignores particles; VBD with
  `integrate_with_external_rigid_solver=True` only reads `body_q`/`body_qd` for friction and never
  moves rigid bodies. So the original MuJoCo arm drives the VBD rod's friction — no Featherstone
  needed. This keeps the *exact* panda arm dynamics + hydroelastic overlay. (Corrects the earlier
  "MuJoCo doesn't couple to VBD" note — that was the shipped-example convention, not a limit.)
- **Stiffness is iteration-limited, not modulus-limited.** With 5 VBD iterations the rod stayed
  floppy (90° fold) no matter the modulus: E = 8e5 → 1e8 gave the **same** ~25 mm bend. Raising to
  **50 iterations** at E=2e7 cut the bend to ~5–11 mm (stiff, nearly straight). Iterations are the
  lever; E past the convergence limit does nothing.
- **Grasp required three fixes** (one-way soft body, round object): close the gripper to a
  rod-thickness gap (`GRIP_CLOSE≈0.016`, not 0 — else it crushes/jams the pads); seat a flat facet
  at the cylinder bottom (else it rolls off the table pre-grasp); spawn at exact rest height (else
  it launches). With these, the rod is gripped and lifted/carried (z 104 → ~320 mm).
- **Scene fidelity comes from forking, not constants.** A from-scratch rebuild recolored the scene
  (yellow→cyan table, magenta→blue cup) and lost the floor because GL colors are palette-indexed by
  shape order. Forking the example reproduced metal's exact look automatically.
- **Performance (rigid vs non-rigid).** Rigid (MuJoCo + hydroelastic, CUDA-graph-captured):
  **12.3 ms/frame (~82 fps)**. Soft (MuJoCo arm + VBD 50 it, eager): **~196–201 ms/frame (~5 fps)**
  — **~16× slower**, from 500 VBD iterations/frame, eager execution (VBD's BVH rebuild can't be
  graph-captured), and per-substep collision. GL render cost itself is identical.

## 7. Status of deliverables

- Rigid-material videos: `tactile_material_{metal,wood,porcelain}.mp4` (720 frames). **Done.**
- FEM indenter videos: `tactile_fem_{gel,soft_rubber,rubber,firm_rubber}.mp4` (200 frames). **Done.**
- FEM Franka rod grasp: `tactile_rod_franka.mp4` (1110 frames), scene only. **Done; tactile panels pending.**
- **Soft pencil, panda scene**: `tactile_material_rubber_soft.mp4` (720 frames, full composite with
  pad-centric compression panels) + `tactile_compare_metal_wood_softrubber.mp4` (3-way, FPS-labeled).
  **Done.** Code committed on `shengzew/fps-collision-benchmarks` (55958d82).

## 8. Generation pipeline (PixelDiT→TRELLIS.2→Newton) — clock scene + profiling

Generative asset pipeline vendored as submodules (see `genpipe/README.md`): PixelDiT text→image →
TRELLIS.2 image→3D GLB → Newton rigid body. Verified end-to-end (brass alarm clock).

- **Metal clock in the exact panda_hydro scene** (`example_panda_clock_metal.py`, fork of
  `example_robot_panda_hydro.py`): the TRELLIS GLB loaded as a rigid body (convex-hull collision +
  hydroelastic SDF + full mesh as visual overlay), metal material (μ=0.5, kh=1e12, **density derived
  from mass/volume** = mass ÷ hull volume), gripped and released **from high above into the cup**
  (`grip_dx=0.03` centers the jaws). At the default 2× scale (7.2 cm) the clock exceeds the ~5 cm cup
  opening, so it perches on the rim (shrink to ~1× or enlarge the cup to seat it inside).
- **Tactile measurement video** (`tactile_clock_metal.py`, adapted from `tactile_video.py`):
  same composite as `tactile_material_wood.mp4` (scene + per-pad pressure heatmap + grip-force +
  shear-on-object + material-signature). Two fixes for the clock: **auto-zoom the pad window to the
  actual contact** (was a fixed ±2 cm box), and **compliant pads** (`pad_kh=1e10`) so they conform to
  the rounded clock → broad ~100 mm² patch at realistic ~3 MPa (the rigid pad only touched the ~8.5 mm
  dome → tiny 35 mm² patch; pressure proxy scaled by the pad, not the rigid clock kh). →
  `tactile_material_clock_metal.mp4` (570 frames, 19 s).

**Profiling (RTX 6000 Ada, one GPU, models resident):** image ~8–9 s; 3D object ~62 s end-to-end
(range 46–84 s by complexity); Newton grasp sim ~5 s (8.6 ms/frame). Model loads: PixelDiT ~12 s,
TRELLIS 4B ~47 s + ~64 s first-object JIT warmup. Tactile video cost is ~80 % matplotlib compositing
(333 ms/frame), not physics.

**Scaling (measured):** both stages are **compute-bound** — batching (PixelDiT `--bs`, flat 8.6→9.6
s/img) or concurrency (TRELLIS 1/2/3 procs, each slows ∝ N; GPU 100 % util) gives **no throughput
gain** on one GPU; memory is not the limit (~6–22 GB of 48 GB). (`num_samples>1` batching is broken
in the 4B pipeline — tensor-shape bug 64≠128.) N objects ≈ N × single-object time → scale out across
GPUs (1 stream/GPU optimal), not up on one.

**Fleet estimate** (~71 s/object; 8 GPU/node, linear in total GPUs; Ada, H100 ~1.5–2× faster).
Hydroelastic-SDF conversion is near-free (~0.3 s/object build, ~0.5 MB/object, or 0 if rebuilt at
sim-load):

| objects | gen compute | 8 nodes (64 GPU) | 16 nodes (128 GPU) | storage (GLB+img+SDF) |
|---|---|---|---|---|
| 10k | ~197 GPU-hr | ~3 h | ~1.5 h | ~77 GB |
| 1M | ~19,700 GPU-hr | ~13 days | ~6.4 days | **~7.7 TB** (7.1 TB GLB + 75 GB img + 0.5 TB SDF) |

At 1M scale: budget for retries/failures and shard the output (subdirs) — 1M files on Lustre.

## 9. Cluster setup (oci-ord Slurm) — in progress

Bootstrapping the SDG pipeline on the OCI-ord cluster (`sdg/newton` under
`/lustre/fsw/portfolios/nvr/users/shengzew`), driven via a `screen` session `ord_pod`. Full
guide + gotchas + live state: **`genpipe/RUNBOOK.md`** (authoritative). Highlights:

- Repo cloned (branch + submodules + eigen). **`pixeldit` env done** (torch 2.5.0+cu124 imports).
- **`trellis2` env incomplete** — the **login node SIGKILLs heavy builds** (torch install died
  mid-way). Fix: re-run `genpipe/cluster_setup.sh "8.0;9.0"` **inside a Slurm job** (account
  `nvr_nxp_visionconferencing`, 8 GPU/node; no system nvcc → conda cuda-toolkit 12.4). Open
  question: whether compute nodes have internet for the pip/clone steps.
- Screen-channel gotchas (documented in RUNBOOK): shell vars come back empty → absolute paths;
  ~64-col wrap mangles long lines → widen; avoid escaped quotes / regex brackets; background long
  ops + poll logs.
