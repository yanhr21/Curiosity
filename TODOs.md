# TODOs

Action items and pending work. Status notes in `claude_context/experimental_conclusions.md`.

## In progress
- [ ] **Cup-scene deformable grasp**: extend `tactile_rod_franka.py` to the panda_hydro-style
      layout (rod + gripper + cup + table) — pick the deformable rod and place it **into a cup**.
      Keep the Featherstone+VBD coupling (the working recipe).

## Next
- [ ] Port the **tactile measurement panels** to the cup-scene `tactile_rod_franka.py`. The panel
      pipeline now exists for the panda-scene soft rod (`tactile_soft_rod_video.py`: VBD-compression
      pad heatmap + grip-force + rod bend/height) — reuse it.
- [ ] Optionally render the FEM rod grasp across rod stiffnesses (gel→firm_rubber) for a
      material-comparison set, mirroring the indenter videos.

## Verifications pending
- [ ] Confirm cup placement actually lands the rod in the cup (visual + final particle positions).
- [ ] Spot-check the full carry/place/release of `tactile_rod_franka.mp4` (sim verified to "hold";
      visually confirm release + settle).
- [ ] Spot-check the soft pencil's place/release **into the cup** in `tactile_material_rubber_soft.mp4`
      (grasp+lift+carry verified to ~frame 300; end-of-sequence placement not yet frame-checked).

## Done
- [x] FPS-vs-collision study (XPBD + panda_hydro kh/dt coupling).
- [x] Tactile data extraction + rigid grasp videos (pen, push-into-cup, shear-on-pen).
- [x] Material model decision (rigid share kh; soft → FEM).
- [x] Rigid-material videos (metal/wood/porcelain, 720 frames).
- [x] FEM indenter videos (gel/soft_rubber/rubber/firm_rubber, Hertz force).
- [x] Deformable rod grasp working via Featherstone+VBD coupling (lifts ~180 mm).
- [x] Documented the prescribed-jaw negative result (squeezes, cannot lift).
- [x] Soft pencil in the EXACT panda scene (`example_panda_soft_rod.py`, fork of the original
      example): MuJoCo arm + VBD rod (one-way), round cylinder-tet rod, 50 VBD iters for stiffness,
      grip/round-rod/spawn fixes. 720-frame composite `tactile_material_rubber_soft.mp4` +
      pad-centric panels + 3-way `tactile_compare_metal_wood_softrubber.mp4` (FPS-labeled). Committed (55958d82).
- [x] Proved MuJoCo+VBD couple (one-way: VBD reads body velocities) — corrects the earlier
      "only Featherstone drives VBD friction" note.
- [x] Measured rigid-vs-soft render perf: ~82 fps (rigid, CUDA graph) vs ~5 fps (soft FEM, eager) — ~16×.
