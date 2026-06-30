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
