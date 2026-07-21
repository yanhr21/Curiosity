# TODOs

Action items and pending work. Status notes in `claude_context/experimental_conclusions.md`.

## SDG generation pipeline (PixelDiT→TRELLIS.2→Newton) — cluster bring-up
Full guide + live state in `genpipe/RUNBOOK.md`.
- [ ] **Finish `trellis2` env on the cluster** — re-run `genpipe/cluster_setup.sh "8.0;9.0"`
      **inside a Slurm job** (login node SIGKILLs the heavy torch install). Account
      `nvr_nxp_visionconferencing`, `-p backfill_singlenode --gres=gpu:1`.
- [ ] **Check compute-node internet** (`srun … git ls-remote …`) — decides whether pip/clone can
      run in the job or must be staged from the login node / offline wheels.
- [ ] **Verify envs via GPU `srun`** — `import torch/xformers/nvdiffrast/o_voxel/flex_gemm/cumesh`,
      `torch.cuda.is_available()` (task #15).
- [ ] Build the **`newton` sim env** on the cluster (if sim will run cluster-side).
- [ ] **Export `HF_TOKEN`** (DINOv3-authorized; account `McMvMc`) before any generation run.
- [ ] First end-to-end generation on the cluster, then a **multi-GPU batch** (1 stream/GPU) for the
      10k / 1M set; shard outputs into subdirs (7.7 TB / 1M files on Lustre).

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
- [x] **SDG generation pipeline** vendored (PixelDiT + TRELLIS.2 submodules + `genpipe/`), verified
      end-to-end (text→image→GLB). Commits dbe55ef8/c21dfb3e/c87b3e01/c41254a1.
- [x] **Metal clock Newton scene** (`example_panda_clock_metal.py`) + **tactile measurement video**
      (`tactile_clock_metal.py` → `tactile_material_clock_metal.mp4`, compliant pad + auto-zoom).
- [x] **Profiling + scaling study**: ~8 s/image, ~62 s/object, ~5 s sim; both stages compute-bound
      (no batch/concurrency speedup on 1 GPU). Fleet: 10k ≈ 3 h/77 GB (8 nodes); 1M ≈ 13 days/7.7 TB.
- [x] **`genpipe/RUNBOOK.md`** written; cluster (oci-ord) repo cloned + `pixeldit` env built.
