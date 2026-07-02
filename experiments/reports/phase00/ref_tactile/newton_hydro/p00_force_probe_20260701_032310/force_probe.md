# Phase 00 Direct Force Probe Failure

- run_tag: `p00_force_probe_20260701_032310`
- status: `failed_cuda_illegal_memory_access`
- run exit: `1`
- official example: `newton.examples.robot.example_robot_panda_hydro`
- attempted method: request `Contacts.force`, recreate compatible collision-pipeline contacts, step the official example, then call `SolverMuJoCo.update_contacts()` after each frame
- log: `logs/newton/phase00/ref_tactile/newton_hydro/p00_force_probe_20260701_032310.srun.log`
- inner log: `logs/newton/phase00/ref_tactile/newton_hydro/p00_force_probe_20260701_032310/force_probe.log`

Observed failure: `Warp CUDA error 700: an illegal memory access was encountered`.

Interpretation: direct solver force export through `SolverMuJoCo.update_contacts()` is not currently valid for the official Panda hydro Newton-contacts path. Direct `Ft` remains a blocker; current tactile evidence must stay labeled as `hydro_proxy.*` until a faithful official force path is found.
