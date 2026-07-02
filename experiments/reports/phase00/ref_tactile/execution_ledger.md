# Phase 00 Reference Tactile Execution Ledger

## 2026-07-01

### Documentation Reset

- Wrote the reference-video tactile reset into `IDEA/idea.md`.
- Wrote the hard reference-video tactile rule into `AGENTS.md`.
- Archived previous contact-count active Phase 00/01 plan and TODO under:
  - `PLAN/legacy_20260630_contact_proxy_stopgate/`
  - `TODO/legacy_20260630_contact_proxy_stopgate/`
- Created active:
  - `PLAN/00_ref_tactile_env/plan.md`
  - `TODO/00_ref_tactile_env/todo.md`

### Source Audit

- Wrote `experiments/reports/phase00/ref_tactile/source_audit.md`.
- Fetched latest Newton upstream metadata.
- Created latest Newton v1.3.0 worktree:
  `external/newton_v1.3`.
- Confirmed local Taccel matches upstream main:
  `cb23bc251b531ba6908a3788c2f91423cd543149`.
- Cloned HydroShear:
  `external/hydroshear`,
  commit `a53a51cb74f0608ca53839415d7f1964a99f1db0`.
- Attempted IsaacLabTactile clone; failed with:
  `fetch-pack: unexpected disconnect while reading sideband packet`.
  This is a comparison-reference acquisition blocker, not a blocker for the
  Newton/Taccel mainline.

### Active-Curiosity Reference Audit

- Cloned APPLE:
  `external/APPLE`,
  commit `4b1d71fadb786d865d4ee29a184ab408b9605083`.
- Cloned Tactile MNIST:
  `external/tactile-mnist`,
  commit `9e4e59139e9349ab361a3b9297f4815724ad6387`.
- Wrote audit report:
  `experiments/reports/phase00/ref_tactile/curiosity_reference_audit.md`.
- Wrote matrix:
  `experiments/configs/phase00/ref_tactile/curiosity_reference_matrix_v1.json`.
- Classification: source/text audit only. APPLE and Tactile MNIST are
  secondary Gate 00G active tactile perception references for future
  closed-loop curiosity design. They are not Newton-native grasp checkpoints,
  not Gate 00D/00E/00F completion, and not current curiosity-training success.

### Gate 00F Environment Availability Preflight

- Wrote lightweight checker:
  `experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh`.
- Ran only file/executable checks on the login node; no package import,
  installation, simulation, rendering, training, model loading, official demo,
  or dataset conversion was run.
- Status:
  `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`.
- Report:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_availability.md`.
- Result:
  `envs/univtac/conda/bin/python` missing,
  `envs/univtac/.venv/bin/python` missing,
  `envs/tacauchy/conda/bin/python` missing,
  `envs/tacauchy/.venv/bin/python` missing,
  `git-lfs` missing,
  `cmake` missing,
  `nvcc` missing.
- Classification: Gate 00F remains blocked by missing approved prebuilt
  official reference environments. Curiosity training remains disallowed.
- Updated the compute-side reference sanity runner so future probes accept
  `UNIVTAC_PYTHON` / `TACAUCHY_PYTHON`, `envs/<target>/conda/bin/python`, or
  `envs/<target>/.venv/bin/python`. This keeps the runner aligned with the
  controlled conda-prefix environment plan and prevents false missing-env
  blockers after the conda envs are prepared.
- Updated the tmux launcher so it refuses to consume a Slurm allocation when
  the target reference Python is missing, unless
  `ALLOW_MISSING_REFERENCE_ENV_BLOCKER_RUN=1` is explicitly set. This keeps
  future official-sanity attempts from wasting GPU time on known missing-env
  blockers.
- Updated `src/newton_tactile_curiosity/phase00_gate_review.py` and the
  Gate-review runners so `reference_env_availability` is an explicit Gate 00F
  check sourced from
  `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`.

### Active Evidence Index

- Wrote:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`.
- Purpose:
  keep current Phase 00 evidence paths easy to find without treating the long
  ledger as the working entry point.
- Classification:
  organization/index only. It is not a new experiment, not Gate 00D/00E/00F
  completion, and not curiosity success.

### Environment Spec

- Wrote dense tactile schema:
  `experiments/configs/phase00/ref_tactile/dense_tactile_schema_v1.json`.
- Wrote steel-first scene spec:
  `experiments/configs/phase00/ref_tactile/steel_scene_spec_v1.json`.

### Official Sanity Preparation

- Wrote compute-only official sanity runner:
  `experiments/configs/phase00/ref_tactile/run_official_sanity_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_official_sanity_tmux.sh`.
- Ran login-node lightweight syntax checks only:
  `bash -n` passed for both scripts.

### Allocation State

- Created tmux session: `curiosity_phase00_ref_tactile`.
- Requested H200 Slurm allocation:
  - job ID: `160324`
  - job name: `curiosity_p00_ref_tactile`
  - first state: `PENDING`
  - first reason: `Priority`
  - later state: `RUNNING`
  - node: `server30`

### Official Sanity Result

- Run tag: `p00_ref_sanity_20260701_013203`
- Slurm job: `160324`
- Host: `server30`
- GPU: `NVIDIA H200`
- Summary:
  `experiments/outputs/phase00/ref_tactile/sanity/p00_ref_sanity_20260701_013203/official_sanity_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/sanity/p00_ref_sanity_20260701_013203/official_sanity.md`
- Newton result:
  - commit: `ce11136b3a28390944f7fe5a32801b31d8aa5670`
  - command:
    `python -m newton.examples.sensors.example_sensor_contact --device cuda:0 --viewer null --num-frames 120 --test --quiet`
  - status: `pass`
- Taccel result:
  - commit: `cb23bc251b531ba6908a3788c2f91423cd543149`
  - command:
    `python -m examples.peg --num_envs 1 --export_mesh`
  - status: `pass`

No simulation, rendering, training, model loading, or dataset conversion has
been run on the login node.

### Reference Diagnostic Runner Preparation

- Wrote Taccel-based dense tactile environment diagnostic:
  `src/newton_tactile_curiosity/phase00_ref_tactile_diagnostic.py`.
- Wrote compute-only diagnostic runner:
  `experiments/configs/phase00/ref_tactile/run_ref_diagnostic_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_ref_diagnostic_tmux.sh`.
- The runner exports:
  - Taccel tactile RGB/depth/normal arrays;
  - marker-flow panels;
  - Taccel contact/collision/friction force summaries when the official API
    exposes them;
  - scene mesh projection video panels and sampled PLY scene frames;
  - source arrays under
    `experiments/outputs/phase00/ref_tactile/diag/<run_tag>/`;
  - video under
    `experiments/visuals/phase00/ref_tactile/diag/<run_tag>/`.
- Classification: environment diagnostic only, not training, not base-model
  completion, and not curiosity success.

### Reference Diagnostic Attempts And Blockers

- Ran Taccel Panda/Tac-Man-style diagnostic:
  `p00_ref_diag_20260701_014305`.
  Outcome: video/NPZ/PLY artifacts generated, but marker displacement was later
  found to be based on a bad local/rest comparison. Not valid contact evidence.
- Ran fixed Taccel Panda/Tac-Man-style diagnostic:
  `p00_ref_diagfix_20260701_014957`.
  Outcome: `status=partial_pass_force_gap`, `force_nonzero=false`.
- Ran Tac-Man geometry diagnostic:
  `p00_ref_tacman_20260701_015357`.
  Outcome: `status=partial_pass_force_gap`, `force_nonzero=false`.
- Ran collision probe:
  `p00_ref_collisionprobe_20260701_015627`.
  Outcome: `collision_nonzero=false`, `max_collision_count=0`; videos are
  partial visual/debug assets only.
- Ran contact placement sweep:
  `p00_contact_sweep_20260701_020413`.
  Outcome: all swept object x positions had `max_collision_count=0` and
  `max_gel_force_norm=0`.
- Ran official peg-style contact diagnostics:
  `p00_peg_contact_20260701_021622`,
  `p00_peg_contact_fix_20260701_021857`, and
  `p00_peg_contact_full_20260701_022114`.
  Outcome: even the full 200-step instrumented path reported
  `max_collision_count=0`, `max_force_norm=0`, and `max_deform_mean_m=0`.

Current blocker: Taccel-generated visual/soft-body assets are not yet producing
valid nonzero contact mechanics through the instrumented APIs. These attempts
must not be reported as dense tactile success, base-model success, or curiosity
training progress. The next faithful path is Newton official contact/hydro
mechanics first, then provenance-labeled tactile map construction.

### Newton Official Panda Hydro Base Evidence

- Ran official Newton Panda hydro pick/lift test on the held H200 allocation:
  `p00_newton_panda_hydro_20260701_022557`.
- Slurm job: `160324`
- Host: `server30`
- Log:
  `logs/newton/phase00/ref_tactile/newton_hydro/p00_newton_panda_hydro_20260701_022557.srun.log`
- Command:
  `python -m newton.examples.robot.example_robot_panda_hydro --device cuda:0 --viewer null --num-frames 240 --test --quiet --scene cube --world-count 1`
- Result: `P00_NEWTON_PANDA_HYDRO_EXIT=0`.
- Classification: positive official base-grasp/hydro-mechanics evidence. This
  is not dense tactile success yet because the run did not export synchronized
  rollout video, pad-resolved tactile maps, or the full dense tactile schema.
- Next faithful action: rerun the same official Newton path with a headless
  visual output backend, then add provenance-labeled hydro/contact field export
  before any base-model or curiosity success claim.

### Newton Official Panda Hydro USD Visual Evidence

- Ran official Newton Panda hydro with the USD viewer:
  `p00_panda_usd_20260701_023155`.
- Slurm job: `160324`
- Host: `server30`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_panda_usd_20260701_023155/newton_hydro_usd.md`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_panda_usd_20260701_023155/newton_hydro_usd_summary.json`
- Visual asset:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_panda_usd_20260701_023155/panda_hydro.usd`
- Result: `status=pass`, `run_exit=0`, USD size `6902837` bytes.
- Classification: positive official Newton hydro base visual evidence. This is
  still not dense tactile success because the run lacks synchronized
  pad-resolved tactile maps, force/shear timeseries, and manual visual
  inspection.

### Newton Hydro Tactile Export And Video Evidence

- Wrote official-Newton-based tactile exporter:
  `src/newton_tactile_curiosity/phase00_newton_hydro_export.py`.
- Wrote enhanced tactile visualizer:
  `src/newton_tactile_curiosity/phase00_enhance_tactile_visuals.py`.
- Successful run tag: `p00_hydro_tac_avi3_20260701_024826`.
- Slurm job: `160324`
- Host: `server30`
- Official base reused:
  `newton.examples.robot.example_robot_panda_hydro`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/hydro_tactile_summary.json`
- NPZ source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/hydro_tactile_timeseries.npz`
- Primary tactile video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/tactile_maps.avi`
- Enhanced tactile video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/tactile_maps_enhanced.avi`
- Enhanced tactile sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/tactile_sheet_enhanced.png`
- Metrics visual:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_hydro_tac_avi3_20260701_024826/metrics.svg`
- Positive metrics:
  - status: `pass`
  - frames: `240`
  - max object lift: `0.22351960837841034` m
  - max hydro face count: `4171`
  - max rigid contact count: `106`
  - max penetration: `0.0006419424898922443` m
  - left/right active tactile frames: `170` / `170`
  - enhanced active frames: `171`
- Manual visual inspection:
  - raw sheet was too low contrast because a few large peaks dominated global
    scaling;
  - enhanced sheet shows persistent lower-pad contact patches from frame `69`
    through frame `239`;
  - contact maps are still sparse and lower-edge concentrated, not
    reference-video-level tactile richness.
- Failed intermediate video-export attempts are diagnostic only:
  - `p00_hydro_tac_20260701_023711`: failed because Newton venv lacks
    `matplotlib`;
  - `p00_hydro_tac_nompl_20260701_023916`: failed because no `ffmpeg` command
    or imageio/cv2/av encoder exists;
  - `p00_hydro_tac_avi_20260701_024439` and
    `p00_hydro_tac_avi2_20260701_024629`: failed due local AVI header bugs,
    while still writing dense PPM frames.
- Classification: major positive Phase 00 evidence for official Newton base
  grasp plus provenance-labeled hydro-derived left/right tactile maps. It is
  not curiosity training success and not final reference-level tactile
  completion because force/shear fields remain incomplete
  (`force_norm_sum_max=0.0`), visual scene and tactile maps are not yet fused
  into one synchronized rollout, and tactile patches are sparse.

### Synchronized Newton Hydro Scene + Tactile Diagnostic

- Wrote synchronized scene/tactile diagnostic exporter:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`.
- Wrote compute-only runner:
  `experiments/configs/phase00/ref_tactile/run_sync_hydro_diagnostic_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`.
- Successful run tag: `p00_sync_hydro_20260701_025818`.
- Slurm job: `160324`
- Host: `server30`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_hydro_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_hydro_timeseries.npz`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_sync_hydro_20260701_025818/sync_hydro_diagnostic.md`
- File verification:
  - AVI: `1180 x 700`, `30.00 fps`
  - sheet: `2360 x 2800` JPEG
- Positive metrics:
  - status: `pass`
  - frames: `240`
  - max object lift: `0.2235533893108368` m
  - max hydro face count: `4171`
  - max raw hydro contact count: `4171`
  - max contact area sum: `0.0034376755356788635` m^2
  - max `hydro_proxy.Fn`: `2931.3955078125`
  - max `hydro_proxy.shear_motion`: `0.03598056361079216` m
  - left/right active tactile frames: `170` / `170`
- Manual visual inspection:
  - scene schematic, left/right tactile patches, object-z, contact area,
    `Fn` proxy, and shear-motion proxy are synchronized in one visual asset;
  - tactile patches are visible but still small and concentrated near the pad
    lower edge;
  - this improves over tactile-only AVI but is still not a reference-video-like
    dense tactile rendering.
- Classification: major positive Phase 00 synchronized diagnostic evidence.
  It is not final Gate 00D completion because the scene panel is a schematic
  from `body_q`, not a USD/photoreal render; `Ft` and pad-resolved shear vector
  fields are proxy-only; and steel-specific stress/strain calibration is still
  missing. It is not training and not curiosity success.

### Newton Hydro Base Mechanics Diagnostic

- Extended synchronized diagnostic exporter:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`.
- Added mechanics fields:
  - `hydro_proxy.stress = hydro_proxy.Fn / contact_area`;
  - force-weighted mean contact normal decoded from Newton reducer normals;
  - `hydro_proxy.Ft_capacity = max(shape_material_mu_pair) * hydro_proxy.Fn`;
  - left/right normal force, stress, tangential-capacity proxy, and force
    balance;
  - lift/hold/drop/slip/safety statistics;
  - separated instrumented simulation/export FPS and video render FPS.
- Successful run tag: `p00_base_mech_20260701_030544`.
- Slurm job: `160324`
- Host: `server30`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_hydro_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_hydro_timeseries.npz`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_base_mech_20260701_030544/sync_hydro_diagnostic.md`
- File verification:
  - AVI: `1180 x 760`, `30.00 fps`
  - sheet: `2360 x 3040` JPEG
- Positive mechanics metrics:
  - status: `pass`
  - frames: `240`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above lift threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.22364932298660278` m
  - final object height: `0.33636319637298584` m
  - max hydro face count: `4171`
  - max raw hydro contact count: `4171`
  - max contact area sum: `0.003433220088481903` m^2
  - max `hydro_proxy.Fn`: `2936.611083984375`
  - max `hydro_proxy.stress`: `1489732.5`
  - max left/right stress proxy: `13193867.0` / `8213480.0`
  - max `hydro_proxy.Ft_capacity`: `2936.611083984375`
  - mean active force balance ratio: `0.954063892364502`
  - max `hydro_proxy.shear_motion`: `0.03536156192421913` m
  - max object acceleration: `1.310336709022522` m/s^2
  - left/right active tactile frames: `168` / `168`
  - instrumented sim/export FPS: `36.448492620587935`
  - render FPS: `40.79951131027666`
- Manual visual inspection:
  - contact sheet visibly synchronizes scene schematic, left/right tactile
    maps, object height, contact area, normal-force proxy, stress proxy,
    tangential-capacity proxy, and shear-motion proxy;
  - tactile maps remain sparse and lower-edge concentrated, not
    reference-video-level dense tactile richness.
- Calibration finding:
  - observed `shape_material_mu` is `[1.0]`;
  - observed `shape_material_kh` is `[10000000000.0, 99999997952.0]`;
  - this does not match the current steel-first spec target
    `mu=0.3`, `kh=1e12`.
- Classification: stronger official Newton base mechanics evidence. It is not
  steel-calibrated Gate 00D/00E completion, not direct `Ft`, not pad-resolved
  shear-vector evidence, not training, and not curiosity success.

### Newton Hydro Official Runtime Benchmark

- Wrote official null-viewer benchmark runner:
  `experiments/configs/phase00/ref_tactile/run_newton_hydro_benchmark_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`.
- Successful run tag: `p00_hydro_bench_20260701_030813`.
- Slurm job: `160324`
- Host: `server30`
- Official example:
  `newton.examples.robot.example_robot_panda_hydro`
- Command path:
  `python -m newton.examples.robot.example_robot_panda_hydro --device cuda:0 --viewer null --benchmark 10 --num-frames 720 --quiet --scene cube --world-count 1`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_hydro_bench_20260701_030813/newton_hydro_benchmark_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_hydro_bench_20260701_030813/newton_hydro_benchmark.md`
- Log:
  `logs/newton/phase00/ref_tactile/newton_hydro/p00_hydro_bench_20260701_030813.srun.log`
- Result:
  - run exit: `0`
  - benchmark: `67.5 FPS`
  - frames/elapsed: `675` frames in `10.01` s
  - target FPS: `82`
  - minimum accepted FPS: `60`
  - meets minimum 60 FPS: `true`
  - meets target 82 FPS: `false`
- Classification: positive official runtime baseline above 60 FPS but below
  the user's 82 FPS target. This is not tactile export throughput, not
  training, and not curiosity success. The 82 FPS target remains an open gap.

### Newton Hydro Direct Force Probe Failure

- Wrote direct-force availability probe:
  `src/newton_tactile_curiosity/phase00_force_probe.py`.
- Wrote compute-only runner:
  `experiments/configs/phase00/ref_tactile/run_force_probe_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_force_probe_tmux.sh`.
- Run tag: `p00_force_probe_20260701_032310`.
- Slurm job: `160324`
- Host: `server30`
- Attempted method:
  request `Contacts.force`, recreate official collision-pipeline contacts,
  step the official Panda hydro example, then call
  `SolverMuJoCo.update_contacts()` after each frame.
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_force_probe_20260701_032310/force_probe_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_force_probe_20260701_032310/force_probe.md`
- Log:
  `logs/newton/phase00/ref_tactile/newton_hydro/p00_force_probe_20260701_032310.srun.log`
- Result:
  - `PHASE00_FORCE_PROBE_SRUN_EXIT=1`
  - observed `Warp CUDA error 700: an illegal memory access was encountered`
  - no valid nonzero direct-force arrays were produced
- Interpretation: direct solver force export through
  `SolverMuJoCo.update_contacts()` is not currently valid for the official
  Panda hydro Newton-contacts path. Direct `Ft` remains a blocker; current
  force/shear evidence must stay explicitly labeled as `hydro_proxy.*` until a
  faithful official force path is found.

### Newton Hydro Steel-Spec Material Calibration Diagnostic

- Updated synchronized diagnostic exporter to accept explicit material
  overrides:
  - `--material-label`
  - `--override-mu`
  - `--override-kh`
- Updated runner/launcher so override parameters are persisted in the tmux
  env file.
- Invalid parameter-loss run:
  `p00_steel_calib_20260701_032556`.
  - Cause: initial launcher did not persist `MATERIAL_LABEL`, `OVERRIDE_MU`,
    or `OVERRIDE_KH` into the remote env file.
  - Action: moved artifacts out of active evidence paths into grouped
    `invalid_param_loss/` folders under outputs, visuals, reports, and logs.
  - This run must not be used as steel calibration evidence.
- Successful run tag: `p00_steel_v1_20260701_032709`.
- Slurm job: `160324`
- Host: `server30`
- Requested material override:
  - label: `steel_spec_v1`
  - `mu=0.3`
  - `kh=1000000000000`
- Observed material arrays:
  - `observed_shape_material_mu_unique=[0.30000001192092896]`
  - `observed_shape_material_kh_unique=[999999995904.0]`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_hydro_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_hydro_timeseries.npz`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_steel_v1_20260701_032709/sync_hydro_diagnostic.md`
- Positive metrics:
  - status: `pass`
  - frames: `240`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above lift threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.2235182225704193` m
  - max contact area sum: `0.0032340704929083586` m^2
  - max `hydro_proxy.Fn`: `22572.54296875`
  - max `hydro_proxy.stress`: `6979607.5`
  - max left/right stress proxy: `19897082.0` / `14456556.0`
  - max `hydro_proxy.Ft_capacity`: `6771.763671875`
  - max `hydro_proxy.shear_motion`: `0.07048462331295013` m
  - max object acceleration: `1.2139862775802612` m/s^2
  - left/right active tactile frames: `169` / `168`
  - instrumented sim/export FPS: `39.2169025629407`
- Manual visual inspection:
  - sheet is nonblank and synchronized;
  - object lift/contact/stress/Ft-capacity/shear curves respond to the
    grasp/lift phases;
  - tactile patches remain sparse and lower-edge concentrated.
- Classification: positive steel-spec candidate mechanics diagnostic. Gate 00D
  and Gate 00E still remain open because direct `Ft`, pad-resolved shear
  vector fields, reference-video-level dense tactile richness, photoreal/USD
  fused scene panels, and 82 FPS target closure are still missing.

### Newton Hydro Grid Tactile Diagnostic

- Updated synchronized diagnostic exporter:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`.
- Added design record:
  `experiments/reports/phase00/ref_tactile/tactile_representation_decision.md`.
- New exported fields:
  - `left_fn_map`, `right_fn_map`
  - `left_stress_map`, `right_stress_map`
  - `left_deform_proxy_map`, `right_deform_proxy_map`
  - `left_shear_vector_y_map`, `left_shear_vector_z_map`
  - `right_shear_vector_y_map`, `right_shear_vector_z_map`
  - `left_shear_magnitude_map`, `right_shear_magnitude_map`
- Provenance:
  - `Fn` map from Newton hydro reducer `contact_area * penetration *
    effective_hydro_stiffness`.
  - stress map from `Fn / contact_area`.
  - deformation map from reducer penetration/compression.
  - shear vector map from frame-to-frame weighted contact-center motion in
    each pad's local tactile plane.
  - all fields remain explicitly `hydro_proxy.*`; direct `Ft` is not solved.
- Successful run tag: `p00_grid_v1_20260701_033556`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_grid_v1 RUN_TAG=p00_grid_v1_20260701_033556 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`
- Srun exit: `0`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_hydro_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_hydro_summary.json`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_grid_v1_20260701_033556/sync_hydro_diagnostic.md`
- Visual metadata:
  - AVI: `1180 x 940`, `30 fps`
  - sheet: `2360 x 3760`
- Positive metrics:
  - status: `pass`
  - frames: `240`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above lift threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.2234652042388916` m
  - max contact area sum: `0.003176062600687146` m^2
  - max `hydro_proxy.Fn`: `22551.130859375`
  - max `hydro_proxy.stress`: `7100342.0`
  - max `hydro_proxy.Ft_capacity`: `6765.33935546875`
  - max left/right grid `Fn` map: `3512.72900390625` /
    `1440.114013671875`
  - max left/right grid stress map: `7983843840.0` / `3274585600.0`
  - max left/right deformation map: `0.015967687591910362` /
    `0.0065491702407598495`
  - max left/right shear-magnitude map: `165.96604919433594` /
    `42.3233757019043`
  - active left/right grid `Fn` frames: `166` / `168`
  - active left/right grid shear frames: `165` / `167`
  - instrumented sim/export FPS: `34.76415467411698`
  - render FPS: `34.38684913408402`
- Manual visual inspection:
  - contact sheet is nonblank and synchronized;
  - scene panel, grid `Fn`, shear-vector, deformation maps, and mechanics
    curves advance together;
  - tactile evidence is stronger than raw sparse pressure maps and scalar
    contact count;
  - contact remains concentrated near lower pad edges and direct `Ft` is still
    absent.
- Classification: positive dense-grid tactile export improvement for the
  Phase 00 base environment. Gate 00D and Gate 00E remain open because this is
  still `hydro_proxy.*` evidence, not direct tangential force or reference-grade
  tactile camera rendering, and the scene panel is still schematic.

### Newton Direct Force Source Audit

- Wrote source audit:
  `experiments/reports/phase00/ref_tactile/direct_force_path_audit.md`.
- Finding:
  - Newton official `SensorContact` can report total force and friction, but it
    requires `Contacts.force` to be populated by `solver.update_contacts(...)`.
  - In Newton v1.3.0 `SolverMuJoCo.update_contacts()` converts MuJoCo contact
    data (`mjw_data.contact`, `mj_data.efc.force`) into Newton `Contacts`.
  - The active official Panda hydro example uses
    `SolverMuJoCo(..., use_mujoco_contacts=False, ...)` with a Newton
    hydroelastic collision pipeline.
- Interpretation:
  - `p00_force_probe_20260701_032310` failed because it forced a
    MuJoCo-contact force export path onto the hydroelastic Newton-contact path.
  - Direct `Ft` / `SensorContact.total_force_friction` remains blocked for the
    current official Panda hydro base.
  - Current fields must remain labeled `hydro_proxy.*`.

### Newton Hydro F6 Proxy Diagnostic

- Updated synchronized diagnostic exporter:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`.
- Added T-Rex-aligned proxy arrays:
  - `left_f6_normal_proxy`, `right_f6_normal_proxy`
  - `left_f6_ft_capacity_proxy`, `right_f6_ft_capacity_proxy`
  - `left_f6_combined_proxy`, `right_f6_combined_proxy`
- Provenance:
  - normal F6 proxy from Newton hydro reducer normal forces and contact points;
  - `Ft_capacity` F6 proxy from friction-capacity force along
    contact-center-motion tangent;
  - combined F6 proxy is normal plus `Ft_capacity` proxy;
  - these are not official T-Rex tactile force observations and not direct
    hardware-like tactile force.
- Successful run tag: `p00_f6_v1_20260701_034033`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_f6_v1 RUN_TAG=p00_f6_v1_20260701_034033 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`
- Srun exit: `0`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_hydro_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_hydro_summary.json`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_f6_v1_20260701_034033/sync_hydro_diagnostic.md`
- Visual metadata:
  - AVI: `1180 x 940`, `30 fps`
  - sheet: `2360 x 3760`
- Positive metrics:
  - status: `pass`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above lift threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.2235504388809204` m
  - max `hydro_proxy.Fn`: `22694.48046875`
  - max `hydro_proxy.stress`: `7009656.5`
  - max left/right F6 normal proxy norm: `1606.2017822265625` /
    `2068.567138671875`
  - max left/right F6 `Ft_capacity` proxy norm: `506.7914123535156` /
    `672.7354125976562`
  - max left/right F6 combined proxy norm: `1477.2451171875` /
    `2114.81494140625`
  - active left/right grid shear frames: `166` / `167`
  - instrumented sim/export FPS: `34.57001017834973`
  - render FPS: `34.59251749729773`
- Manual visual inspection:
  - contact sheet remains nonblank and synchronized;
  - tactile panels remain comparable to `p00_grid_v1`;
  - F6 evidence is source-array/summary evidence, not a visual success claim.
- Classification: positive schema-bridge improvement for later T-Rex-style
  conversion. Gate 00D and Gate 00E remain open because F6 arrays are proxy
  wrenches and direct force/shear plus reference-grade tactile rendering are
  still missing.

### Newton MuJoCo SensorContact Direct-Force Variant

- Wrote diagnostic script:
  `src/newton_tactile_curiosity/phase00_mujoco_sensor_probe.py`.
- Wrote compute-only runner:
  `experiments/configs/phase00/ref_tactile/run_mujoco_sensor_probe_in_alloc.sh`.
- Wrote tmux-held launcher:
  `experiments/configs/phase00/ref_tactile/launch_mujoco_sensor_probe_tmux.sh`.
- Diagnostic purpose:
  verify whether Newton's official `SensorContact` path can export direct
  force and tangential/friction force on a related Panda grasp variant.
- Variant:
  official Panda scene/waypoints with
  `SolverMuJoCo(use_mujoco_contacts=True)`, no Newton hydro collision pipeline.
  This is not the active hydro tactile base and not a replacement for it.
- Successful run tag: `p00_mjc_sensor_v1_20260701_034541`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_mjc_sensor_v1 RUN_TAG=p00_mjc_sensor_v1_20260701_034541 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_mujoco_sensor_probe_tmux.sh`
- Srun exit: `0`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/mujoco_sensor/p00_mjc_sensor_v1_20260701_034541/mujoco_sensor_probe_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/mujoco_sensor/p00_mjc_sensor_v1_20260701_034541/mujoco_sensor_probe_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/mujoco_sensor/p00_mjc_sensor_v1_20260701_034541/mujoco_sensor_probe.md`
- Positive metrics:
  - status: `pass_nonzero_friction`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - max contact count: `6`
  - max total force norm: `29.69374656677246`
  - max total friction norm: `8.532435417175293`
  - max per-counterpart force norm: `18.80536460876465`
  - max per-counterpart friction norm: `4.5348076820373535`
  - nonzero total force frames: `171`
  - nonzero total friction frames: `171`
  - lift success over `0.15` m: `true`
  - first lift frame: `171`
  - hold frames above threshold: `69`
  - max object lift: `0.21197126805782318` m
  - update errors: `0`
- Classification: positive direct-force comparison evidence. It shows that
  official Newton `SensorContact` can provide force/friction on a MuJoCo-contact
  variant, but it does not solve direct `Ft` for the active hydro tactile base
  and cannot close Gate 00D or Gate 00E by itself.

### Newton Hydro Hot-Cache Runtime Benchmark

- Updated benchmark launcher to preserve `SCENE`, `WORLD_COUNT`, `NUM_FRAMES`,
  and `BENCHMARK_SECONDS` in the tmux env file:
  `experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`.
- Successful run tag: `p00_bench_hot_20260701_034952`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_bench_hot RUN_TAG=p00_bench_hot_20260701_034952 BENCHMARK_SECONDS=30 NUM_FRAMES=2400 SCENE=cube WORLD_COUNT=1 bash experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`
- Srun exit: `0`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_hot_20260701_034952/newton_hydro_benchmark_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_bench_hot_20260701_034952/newton_hydro_benchmark.md`
- Log:
  `logs/newton/phase00/ref_tactile/newton_hydro/p00_bench_hot_20260701_034952.srun.log`
- Result:
  - benchmark: `79.2 FPS`
  - frames/elapsed: `2377` frames in `30.00` s
  - target FPS: `82`
  - minimum accepted FPS: `60`
  - meets minimum 60 FPS: `true`
  - meets target 82 FPS: `false`
- Classification: positive hot-cache runtime improvement over the earlier
  `67.5 FPS` run, but still below the user's `82 FPS` target. This is not
  tactile export throughput, not training, and not curiosity success.

### Newton Hydro 60-Second Runtime Benchmark

- Successful run tag: `p00_bench_60_20260701_035208`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_bench_60 RUN_TAG=p00_bench_60_20260701_035208 BENCHMARK_SECONDS=60 NUM_FRAMES=5200 SCENE=cube WORLD_COUNT=1 bash experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`
- Srun exit: `0`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_60_20260701_035208/newton_hydro_benchmark_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_bench_60_20260701_035208/newton_hydro_benchmark.md`
- Log:
  `logs/newton/phase00/ref_tactile/newton_hydro/p00_bench_60_20260701_035208.srun.log`
- Result:
  - benchmark: `79.1 FPS`
  - frames/elapsed: `4749` frames in `60.01` s
  - target FPS: `82`
  - minimum accepted FPS: `60`
  - meets minimum 60 FPS: `true`
  - meets target 82 FPS: `false`
- Classification: stable official hydro runtime evidence near `79 FPS`, still
  below the user's `82 FPS` target. Gate 00E remains open on performance.

### Newton Main 82 FPS Benchmark And F6 Tactile Compatibility

- Added independent official Newton main worktree:
  `external/newton_main`.
- Commit:
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`.
- The stable v1.3.0 worktree remains at:
  `external/newton_v1.3`.
- Updated launchers to preserve `NEWTON_ROOT` in tmux env files:
  - `experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`
  - `experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`

#### Main Benchmark

- Successful run tag: `p00_bench_main_20260701_035529`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_bench_main RUN_TAG=p00_bench_main_20260701_035529 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main BENCHMARK_SECONDS=30 NUM_FRAMES=2600 SCENE=cube WORLD_COUNT=1 bash experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh`
- Srun exit: `0`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_main_20260701_035529/newton_hydro_benchmark_summary.json`
- Result:
  - benchmark: `92.6 FPS`
  - frames/elapsed: `2597` frames in `28.04` s
  - target FPS: `82`
  - meets target 82 FPS: `true`
- Classification: positive latest-main performance evidence. It meets the
  user's `82 FPS` base-runtime target for the official Panda hydro null-viewer
  benchmark. This is still not tactile export throughput and not training.

#### Main F6/Grid Tactile Diagnostic

- Successful run tag: `p00_main_f6_v1_20260701_035926`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_main_f6_v1 RUN_TAG=p00_main_f6_v1_20260701_035926 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`
- Srun exit: `0`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_main_f6_v1_20260701_035926/sync_hydro_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_main_f6_v1_20260701_035926/sync_hydro_summary.json`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_f6_v1_20260701_035926/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_f6_v1_20260701_035926/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_main_f6_v1_20260701_035926/sync_hydro_diagnostic.md`
- Positive metrics:
  - status: `pass`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above lift threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.22340291738510132` m
  - max `hydro_proxy.Fn`: `22732.283203125`
  - max `hydro_proxy.stress`: `7027056.5`
  - max left/right F6 combined proxy norm: `598.2024536132812` /
    `602.2731323242188`
  - left/right grid shear active frames: `166` / `167`
  - instrumented sim/export FPS: `34.338542357891384`
  - render FPS: `34.81133350686305`
- Manual visual inspection:
  - contact sheet is nonblank and synchronized;
  - scene, grid `Fn`, shear-vector, deformation maps, and mechanics curves
    advance together;
  - tactile response remains lower-pad concentrated and proxy-only.
- Classification: major Phase 00 progress. Latest official Newton main now
  satisfies the `82 FPS` runtime target and remains compatible with the current
  steel-spec grid/F6 tactile exporter. Gate 00D/00E still remain open on direct
  hydro `Ft`, reference-grade tactile density, and USD/photoreal fusion.

### Newton Main Calibrated-View Tactile Diagnostic

- Updated synchronized diagnostic exporter:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`.
- Added calibrated tactile visualization arrays:
  - `left_calibrated_view_fn_map`, `right_calibrated_view_fn_map`
  - `left_calibrated_view_stress_map`, `right_calibrated_view_stress_map`
  - `left_calibrated_view_deform_proxy_map`,
    `right_calibrated_view_deform_proxy_map`
  - `left_calibrated_view_shear_vector_y_map`,
    `left_calibrated_view_shear_vector_z_map`
  - `right_calibrated_view_shear_vector_y_map`,
    `right_calibrated_view_shear_vector_z_map`
  - `left_calibrated_view_shear_magnitude_map`,
    `right_calibrated_view_shear_magnitude_map`
- Provenance:
  - raw maps are preserved;
  - calibrated maps reuse the same Newton hydro proxy samples;
  - the visualization window is derived from the rollout's 1%-99% local-yz
    contact range;
  - this is a view calibration, not a new physical tactile sensor and not
    direct `Ft`.
- Successful run tag: `p00_calib_view_v1_20260701_040715`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_calib_view_v1 RUN_TAG=p00_calib_view_v1_20260701_040715 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`
- Srun exit: `0`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_hydro_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_hydro_summary.json`
- Synchronized video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_calib_view_v1_20260701_040715/sync_hydro_diagnostic.md`
- Positive metrics:
  - status: `pass`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - lift success over `0.15` m: `true`
  - first lift frame: `169`
  - hold frames above threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.22356301546096802` m
  - max `hydro_proxy.Fn`: `22606.865234375`
  - max raw left/right Fn nonzero cell ratio:
    `0.03515625` / `0.03515625`
  - max calibrated left/right Fn nonzero cell ratio:
    `0.236328125` / `0.236328125`
  - max left/right calibrated-view Fn map:
    `1349.3370361328125` / `505.255615234375`
  - max left/right calibrated-view shear magnitude:
    `28.28270149230957` / `15.270296096801758`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
- Manual visual inspection:
  - contact sheet is nonblank and synchronized;
  - calibrated Fn/shear/deform panels are visibly more readable than the raw
    lower-edge/full-body maps;
  - scene panel is still schematic;
  - tactile evidence remains `hydro_proxy.*`.
- Classification: positive reference-video-alignment visualization improvement.
  It reduces the "all contact is squeezed into an edge/corner" artifact without
  changing the underlying proxy physics. Gate 00D/00E still remain open on
  direct hydro `Ft`, photoreal/USD fusion, and validated gel/marker tactile
  rendering.

### Newton Main Official USD Scene Export

- Successful run tag: `p00_main_usd_v1_20260701_041900`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_main_usd_v1 RUN_TAG=p00_main_usd_v1_20260701_041900 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv SCENE=cube WORLD_COUNT=1 NUM_FRAMES=240 bash experiments/configs/phase00/ref_tactile/launch_newton_hydro_usd_tmux.sh`
- Srun exit: `0`
- Official command inside allocation:
  `python -m newton.examples.robot.example_robot_panda_hydro --device cuda:0 --viewer usd --output-path panda_hydro.usd --num-frames 240 --test --quiet --scene cube --world-count 1`
- Newton root:
  `/public/home/yanhongru/Curiosity/external/newton_main`
- Newton commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/newton_hydro_usd_summary.json`
- USD:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/panda_hydro.usd`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/newton_hydro_usd.md`
- Positive metrics:
  - status: `pass`
  - official Newton hydro base evidence: `true`
  - USD exists: `true`
  - USD size bytes: `6903124`
  - traceback absent: `true`
- Classification: positive official scene/geometry/rollout evidence on latest
  Newton main. This is not training, not curiosity success, and not dense
  tactile success. Gate 00D remains open until this official scene evidence is
  rasterized or otherwise fused with the calibrated tactile/mechanics rollout
  and direct `Ft` or a faithful official force path is available.

### USD Raster Capability Probe

- Valid run tag: `p00_usd_probe_v2_20260701_042430`.
- Invalid/mixed-source run tag: `p00_usd_probe_v1_20260701_042300`.
- Slurm job: `160324`
- Host: `server30`
- Valid command:
  `RUN_TAG=p00_usd_probe_v2_20260701_042430 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv USD_PATH=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/panda_hydro.usd bash experiments/configs/phase00/ref_tactile/run_usd_scene_capability_probe_in_alloc.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_usd_probe_v2_20260701_042430/usd_scene_capability_probe.json`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_usd_probe_v2_20260701_042430/usd_scene_capability_probe.md`
- Positive evidence:
  - USD stage opens: `true`
  - prim count: `220`
  - time codes: `0.0` to `239.0`
  - time codes per second: `60.0`
  - `newton.viewer` import path:
    `/public/home/yanhongru/Curiosity/external/newton_main/newton/viewer.py`
- Blocker evidence:
  - `usdrecord`: unavailable
  - `usdview`: unavailable
  - `usdcat`: unavailable
  - `ffmpeg`: unavailable
  - `pxr.UsdAppUtils`: unavailable
  - `pxr.UsdImagingGL`: unavailable
- Classification: useful blocker/route decision. Direct USD rasterization is
  blocked in the current prebuilt environment, so active scene fusion should
  use official Newton `SensorTiledCamera` unless a faithful prebuilt USD render
  path becomes available.

### Newton Main SensorTiledCamera Scene-Frame Probe

- Successful run tag: `p00_scene_cam_v3_20260701_043330`.
- Failed diagnostic tags:
  - `p00_scene_cam_v1_20260701_042900`: invalid camera-count/ray-count
    interface usage.
  - `p00_scene_cam_v2_20260701_043100`: SensorTiledCamera rendered, but custom
    AVI header packing failed.
- Slurm job: `160324`
- Host: `server30`
- Command:
  `RUN_TAG=p00_scene_cam_v3_20260701_043330 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv SCENE=cube STEPS=180 SAMPLES=12 WIDTH=256 HEIGHT=256 bash experiments/configs/phase00/ref_tactile/run_scene_frame_probe_in_alloc.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_scene_frame_probe.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/scene_frame_probe_summary.json`
- AVI:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/scene_camera.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/scene_camera_sheet.jpg`
- Frame directory:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_scene_cam_v3_20260701_043330/frames/`
- Positive metrics:
  - nonblank: `true`
  - camera views: `head`, `right_wrist`, `left_wrist`
  - sampled frames: `12`
  - AVI metadata: `768 x 308`, `12 fps`
  - sheet metadata: `2220 x 1160`
  - pixel min/max/std: `1` / `255` / `97.0467646595397`
  - object initial/max/lift z: `0.1200004294514656` /
    `0.3123775124549866` / `0.19237708300352097`
- Manual visual inspection:
  - sheet is nonblank;
  - Panda, hand, object, and tabletop are visible from three camera views;
  - object/gripper motion is visible across sampled frames.
- Classification: positive real-scene rendering path on latest official Newton
  main. This replaces the schematic scene panel as the next fusion source, but
  it is not yet dense tactile success or curiosity success. Gate 00D remains
  open until SensorTiledCamera scene frames are fused with calibrated
  tactile/mechanics panels and direct `Ft` or a faithful force path is
  available.

### Newton Main Fused SensorTiledCamera Scene + Calibrated Tactile Diagnostic

- Successful run tag: `p00_fused_cam_v1_20260701_043900`.
- Slurm job: `160324`
- Host: `server30`
- Command launcher:
  `JOB_ID=160324 WINDOW_NAME=p00_fused_cam_v1 RUN_TAG=p00_fused_cam_v1_20260701_043900 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 SCENE_CAMERA=1 SCENE_CAMERA_WIDTH=256 SCENE_CAMERA_HEIGHT=256 NUM_FRAMES=180 bash experiments/configs/phase00/ref_tactile/launch_sync_hydro_diagnostic_tmux.sh`
- Important reproducibility note:
  - The launcher did not yet forward `NUM_FRAMES` when this run was launched,
    so the actual run used the runner default `num_frames=240`.
  - The launcher has since been patched to forward `DEVICE`, `NUM_FRAMES`,
    `SCENE`, `MAP_SIZE`, and `FPS` as well as scene-camera variables.
- Source:
  `src/newton_tactile_curiosity/phase00_sync_hydro_diagnostic.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_hydro_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_hydro_timeseries.npz`
- Synchronized fused video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_scene_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_scene_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_fused_cam_v1_20260701_043900/sync_hydro_diagnostic.md`
- Positive metrics:
  - status: `pass`
  - actual frames: `240`
  - AVI metadata: `1180 x 940`, `30 fps`
  - sheet metadata: `2360 x 3760`
  - scene camera enabled: `true`
  - scene camera nonblank: `true`
  - scene camera pixel std: `96.05477790619898`
  - scene panel:
    `synchronized Newton SensorTiledCamera head/right_wrist/left_wrist scene frames`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - lift success over `0.15` m: `true`
  - hold frames above threshold: `71`
  - drop detected after lift: `false`
  - max object lift: `0.22351396083831787` m
  - max `hydro_proxy.Fn`: `22550.27734375`
  - max `hydro_proxy.stress`: `6975598.0`
  - max left/right calibrated-view Fn map:
    `571.0975341796875` / `255.0796661376953`
  - max left/right calibrated Fn nonzero cell ratio:
    `0.2470703125` / `0.2470703125`
  - max left/right calibrated-view shear magnitude:
    `13.113311767578125` / `8.042720794677734`
  - instrumented sim/export FPS: `20.529331599550922`
  - render FPS: `29.030832616995557`
- Manual visual inspection:
  - contact sheet is nonblank;
  - real head/right-wrist/left-wrist scene frames are fused into the same
    panels as calibrated `Fn`, shear-vector, deformation maps, and mechanics
    curves;
  - object/gripper motion and tactile/mechanics response advance together;
  - this is a major improvement over the previous schematic scene panel.
- Classification: major Phase 00 visual-fusion evidence. It closes the
  schematic-scene weakness for the active Newton hydro diagnostic, while still
  remaining environment/base evidence only. It is not training and not
  curiosity success. Gate 00D remains open on direct solver `Ft`, direct
  pad-resolved shear force, and validated gel/marker-style tactile rendering
  comparable to the reference video.

### Newton Main MJWarp Direct-Force Array Audit

- Short-horizon audit tag: `p00_mjw_force_audit_v1_20260701_045000`.
- Full-horizon audit tag: `p00_mjw_force_audit_v2_20260701_045700`.
- Slurm job: `160324`
- Host: `server30`
- Launcher commands:
  - `JOB_ID=160324 WINDOW_NAME=p00_mjw_audit_v1 RUN_TAG=p00_mjw_force_audit_v1_20260701_045000 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=90 SCENE=cube bash experiments/configs/phase00/ref_tactile/launch_mjw_force_audit_tmux.sh`
  - `JOB_ID=160324 WINDOW_NAME=p00_mjw_audit_v2 RUN_TAG=p00_mjw_force_audit_v2_20260701_045700 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube bash experiments/configs/phase00/ref_tactile/launch_mjw_force_audit_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_force_audit.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_mjw_force_audit_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_mjw_force_audit_tmux.sh`
- v1 summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_force_audit_v1_20260701_045000/mjw_force_audit_summary.json`
- v2 summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_force_audit_v2_20260701_045700/mjw_force_audit_summary.json`
- v2 timeseries:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_force_audit_v2_20260701_045700/mjw_force_audit_timeseries.npz`
- v2 report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_mjw_force_audit_v2_20260701_045700/mjw_force_audit.md`
- Method:
  step official `newton.examples.robot.example_robot_panda_hydro` on latest
  Newton main and read `SolverMuJoCo.mjw_data.contact.*` plus
  `mjw_data.efc.force` directly. This intentionally does not call
  `SolverMuJoCo.update_contacts()` because
  `p00_force_probe_20260701_032310` showed CUDA illegal memory access in that
  official writeback path.
- Shared official evidence:
  - Newton root: `/public/home/yanhongru/Curiosity/external/newton_main`
  - Newton commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
  - GPU: `NVIDIA H200`
  - not training: `true`
  - not curiosity success: `true`
- v1 interpretation:
  - status: `partial_mjw_force_nonzero_no_pad_object_force`
  - frames: `90`
  - frames with contacts: `90`
  - frames with nonzero EFC: `90`
  - max EFC abs sum: `151.57574462890625`
  - max tangent EFC abs sum: `80.7594985961914`
  - frames with pad-object force: `0`
  - explanation: the 90-frame horizon proves bottom-level EFC force arrays are
    readable and nonzero, but it is too short for pad-object force in this
    scripted grasp/lift sequence.
- v2 positive evidence:
  - status: `pass_mjw_force_arrays_nonzero`
  - frames: `240`
  - frames with contacts: `240`
  - frames with nonzero EFC: `240`
  - max `nacon`: `107`
  - max valid EFC address count: `315`
  - max EFC abs sum: `500.2020568847656`
  - max normal EFC abs sum: `227.2883758544922`
  - max tangent EFC abs sum: `272.9135437011719`
  - frames with pad-object force: `128`
  - max pad-object contact count: `66`
  - max pad-object EFC abs sum: `253.05938720703125`
  - max pad-object tangent EFC abs sum: `141.14100646972656`
  - read errors: `0`
- Classification:
  positive official bottom-level force-path evidence. Direct force data exists
  in `mjw_data.efc.force` and includes pad-object normal/tangential constraint
  force during the full 240-frame grasp/lift sequence. This does not yet close
  the tactile gate because it is a candidate MJWarp direct-force extraction
  path, not a validated official `SensorContact` output on the hydro base.
  Next step is a candidate pad-resolved MJWarp force exporter, then validation
  against official `SensorContact` on a compatible MuJoCo-contact scene before
  any direct tactile success claim.

### Candidate MJWarp Direct-Force Tactile Export

- Failed smoke tag:
  `p00_mjw_direct_smoke_v1_20260701_051200`
  - Cause: script called `example.test_final()` at 120 frames, before the
    official Panda hydro lift horizon completed. This was a diagnostic script
    issue, not force-path evidence.
  - Fix: `example.test_final()` is now recorded as a nonblocking final-test
    status so short diagnostics can still write summary/video evidence.
- Passing smoke tag:
  `p00_mjw_direct_smoke_v2_20260701_052200`
  - frames: `120`
  - scene camera: `false`
  - status: `pass_candidate_direct_force_export`
  - expected limitation: official final test failed nonblocking because the
    120-frame horizon lifted only `0.0011140704154968262` m.
- Full visual/export tag:
  `p00_mjw_direct_v1_20260701_052900`
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_mjw_direct_v1 RUN_TAG=p00_mjw_direct_v1_20260701_052900 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube SCENE_CAMERA=1 SCENE_CAMERA_WIDTH=256 SCENE_CAMERA_HEIGHT=256 bash experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_mjw_direct_tactile_export_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile_timeseries.npz`
- Video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_mjw_direct_v1_20260701_052900/candidate_mjw_direct_tactile.md`
- Official evidence:
  - Newton root: `/public/home/yanhongru/Curiosity/external/newton_main`
  - Newton commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
  - GPU: `NVIDIA H200`
  - official example:
    `newton.examples.robot.example_robot_panda_hydro`
  - official final test status: `pass`
  - scene camera enabled: `true`
  - read errors: `0`
  - not training: `true`
  - not curiosity success: `true`
- Positive metrics:
  - status: `pass_candidate_direct_force_export`
  - frames: `240`
  - max object lift: `0.2241058051586151` m
  - max `nacon`: `107`
  - frames with pad-object contacts: `127`
  - max pad-object contact count: `65`
  - max pad-object candidate `Fn` sum: `48.28089141845703`
  - max pad-object candidate `Ft` sum: `48.28089141845703`
  - max left/right candidate `Fn` sum:
    `28.225444793701172` / `20.202743530273438`
  - max left/right candidate `Ft` sum:
    `28.225444793701172` / `20.14565658569336`
  - max left/right candidate `Fn` map:
    `13.648624420166016` / `11.802962303161621`
  - max left/right candidate `Ft` map:
    `13.648625373840332` / `11.802962303161621`
  - max left/right candidate `Fn` nonzero cell ratio:
    `0.3154296875` / `0.31640625`
  - video size: approximately `665 MB`
  - sheet size: approximately `428 KB`
- Manual visual inspection:
  - contact sheet is nonblank;
  - head/right-wrist/left-wrist Newton `SensorTiledCamera` views show Panda,
    gripper, tabletop, and object;
  - candidate left/right `Fn` heatmaps, candidate left/right `Ft` heatmaps,
    and shear arrows appear after the grasp contact window;
  - object-z and candidate force curves rise together during the contact/lift
    segment.
- Classification:
  major candidate direct-force tactile milestone. It upgrades the active
  visual asset from hydro-proxy-only tactile maps to synchronized scene +
  candidate direct normal/tangential force maps derived from official MJWarp
  EFC arrays. It is still not final direct tactile validation and not curiosity
  training success. The next faithful gate is validation against official
  `SensorContact`/`update_contacts` on a compatible MuJoCo-contact scene, then
  steel-spec merge and gel/marker-style tactile comparison.

### Candidate MJWarp vs Official SensorContact Alignment

- Successful run tag: `p00_mjw_align_v1_20260701_055200`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_mjw_align_v1 RUN_TAG=p00_mjw_align_v1_20260701_055200 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_mjw_sensor_alignment_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_sensor_alignment_probe.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_mjw_sensor_alignment_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_mjw_sensor_alignment_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_timeseries.npz`
- Report:
  `experiments/reports/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment.md`
- Official evidence:
  - Newton root: `/public/home/yanhongru/Curiosity/external/newton_main`
  - Newton commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
  - GPU: `NVIDIA H200`
  - variant: official Panda scene/waypoints with
    `SolverMuJoCo(use_mujoco_contacts=True)`, no Newton hydro collision
    pipeline
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - update errors: `0`
  - not training: `true`
  - not curiosity success: `true`
- Alignment target:
  candidate MJWarp EFC frame mapping versus official
  `SensorContact.force_matrix` and `force_matrix_friction`.
- Positive metrics:
  - status: `pass_candidate_sensor_alignment`
  - best force sign: `shape0_negative`
  - best friction sign: `shape0_negative`
  - force active vector count: `291`
  - force relative RMSE: `3.2491620810680347e-08`
  - force mean cosine: `1.0`
  - force norm correlation: `0.9999999999999551`
  - friction active vector count: `291`
  - friction relative RMSE: `2.0018143688320552e-07`
  - friction mean cosine: `1.0`
  - friction norm correlation: `0.999999999999998`
  - max official/candidate force norm:
    `18.80228614807129` / `18.802288055419922`
  - max official/candidate friction norm:
    `4.5354437828063965` / `4.5354437828063965`
  - max object lift: `0.2126164436340332` m
  - lift success over `0.15` m: `true`
  - hold frames above lift threshold: `69`
- Classification:
  strong positive validation for the candidate EFC frame mapping on a
  compatible MuJoCo-contact scene. The candidate direct-force exporter should
  use the `shape0_negative` sign convention. This does not itself prove active
  hydro direct tactile completion; the next gate is applying the validated
  mapping to the steel-spec hydro fused diagnostic and then comparing against
  reference-video-level tactile rendering requirements.

### Steel-Spec Validated-Sign Candidate Direct-Force Tactile Export

- Successful run tag: `p00_mjw_direct_steel_v1_20260701_060500`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_mjw_direct_steel_v1 RUN_TAG=p00_mjw_direct_steel_v1_20260701_060500 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube SCENE_CAMERA=1 SCENE_CAMERA_WIDTH=256 SCENE_CAMERA_HEIGHT=256 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_timeseries.npz`
- Video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile.md`
- Official evidence:
  - Newton root: `/public/home/yanhongru/Curiosity/external/newton_main`
  - Newton commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`
  - GPU: `NVIDIA H200`
  - official example:
    `newton.examples.robot.example_robot_panda_hydro`
  - material override: `steel_spec_v1`, `mu=0.3`, `kh=1e12`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
  - `material_notify_status=pass`
  - force sign convention:
    `shape0_negative` validated by `p00_mjw_align_v1_20260701_055200`
  - scene camera enabled: `true`
  - official final test status: `pass`
  - read errors: `0`
  - not training: `true`
  - not curiosity success: `true`
- Positive metrics:
  - status: `pass_candidate_direct_force_export`
  - frames: `240`
  - max object lift: `0.2225421965122223` m
  - max `nacon`: `106`
  - frames with pad-object contacts: `146`
  - max pad-object contact count: `51`
  - max pad-object candidate `Fn` sum: `40.099632263183594`
  - max pad-object candidate `Ft` sum: `12.027974128723145`
  - max left/right candidate `Fn` sum:
    `20.045833587646484` / `20.085206985473633`
  - max left/right candidate `Ft` sum:
    `6.0137505531311035` / `6.025561332702637`
  - max left/right candidate `Fn` map:
    `8.926953315734863` / `9.650286674499512`
  - max left/right candidate `Ft` map:
    `2.6780858039855957` / `2.8950860500335693`
  - max left/right candidate `Fn` nonzero cell ratio:
    `0.279296875` / `0.2783203125`
  - video size: approximately `665 MB`
  - sheet size: approximately `430 KB`
- Manual visual inspection:
  - contact sheet is nonblank;
  - head/right-wrist/left-wrist scene views show Panda, gripper, tabletop, and
    object;
  - candidate left/right `Fn` and `Ft` heatmaps appear after the contact
    window and stay synchronized with the object-lift and force curves;
  - `Ft` magnitude is roughly consistent with the steel-spec `mu=0.3`
    friction scale.
- Classification:
  major Phase 00 direct-force tactile asset. Compared with the earlier
  hydro-proxy-only visuals, this run uses a `SensorContact`-validated MJWarp
  EFC mapping, steel-spec material settings, and synchronized real scene
  frames. It is still candidate active-hydro tactile evidence rather than final
  reference-video-equivalent tactile output; the next gate is reference-video
  density/gel/marker comparison and final gate review before curiosity
  training restarts.

### Reference Video vs Steel-Spec Candidate Comparison

- Failed/partial run tag: `p00_refcmp_v1_20260701_063000`.
  - Cause: decoder dependency gap; `cv2` was not available in the prebuilt
    environment.
- Partial run tag: `p00_refcmp_v2_20260701_064300`.
  - Cause: candidate DIB AVI decoding was solved, but the reference MP4 still
    needed an approved decoder path.
- Environment preparation:
  - Installed shared-filesystem decoder dependencies into the local Newton venv
    outside compute allocation:
    `imageio==2.37.3` and `imageio-ffmpeg==0.6.0`.
  - This was dependency preparation, not simulation, rendering, training, or
    data conversion on the login node.
- Successful run tag: `p00_refcmp_v3_20260701_065300`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_refcmp_v3 RUN_TAG=p00_refcmp_v3_20260701_065300 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv REFERENCE_VIDEO=/public/home/yanhongru/Curiosity/0780e5ec3fdb26b63ae63de0f49f07c4.mp4 CANDIDATE_VIDEO=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile.avi CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_summary.json bash experiments/configs/phase00/ref_tactile/launch_reference_video_compare_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_reference_video_compare.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_reference_video_compare_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_reference_video_compare_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_video_compare_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_video_compare.md`
- Visual outputs:
  - `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_sheet.jpg`
  - `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/candidate_sheet.jpg`
  - `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_vs_candidate_sheet.jpg`
- Decode evidence:
  - reference decoder: `imageio_ffmpeg`
  - reference frames: `720`
  - reference FPS: `30.0`
  - reference size: `2846 x 1510`
  - candidate decoder: `avi_dib_builtin`
  - candidate frames: `240`
  - candidate FPS: `30.00030000300003`
  - candidate size: `1180 x 820`
- Visual metrics:
  - reference nonblank: `true`
  - candidate nonblank: `true`
  - reference pixel-std mean: `92.9096450805664`
  - candidate pixel-std mean: `87.54474639892578`
  - reference edge-density mean: `0.09730502218008041`
  - candidate edge-density mean: `0.06210779771208763`
  - reference colorfulness mean: `32.0333366394043`
  - candidate colorfulness mean: `27.778425216674805`
- Gate checklist:
  - candidate currently has real Newton `SensorTiledCamera` head/right-wrist/
    left-wrist scene views;
  - candidate has left/right candidate direct `Fn` maps;
  - candidate has left/right candidate direct `Ft` maps;
  - candidate has left/right pad-local shear arrows;
  - candidate has object-z and candidate force curves;
  - candidate has steel-spec `mu`/`kh` material evidence;
  - candidate has compatible-scene `SensorContact` alignment from
    `p00_mjw_align_v1_20260701_055200`.
- Manual visual inspection:
  - the reference side shows a richer multi-column tactile diagnostic layout,
    blue tactile heatmaps, vector/line overlays, and multiple time-series
    mechanics panels;
  - the candidate side shows three Newton camera views, left/right candidate
    `Fn`/`Ft` maps, shear arrows, and object-z/force curves;
  - the candidate is nonblank and materially closer than hydro-proxy-only
    visuals, but it is less dense than the reference and lacks gel/marker-style
    tactile camera channels plus direct contact-normal/contact-area overlays.
- Remaining gaps:
  - gel/marker-style tactile camera rendering comparable to the reference
    video;
  - validated photometric/deformation marker tracking;
  - direct visual overlay of contact normals and contact area in the same
    direct-force video;
  - channel-by-channel semantic matching beyond frame-level metrics;
  - final Gate 00D/00E review before restarting curiosity training.
- Classification:
  positive Phase 00 reference-alignment and gap-definition asset. This is not
  training, not curiosity success, and not Gate 00D/00E completion. Curiosity
  training remains disallowed until the remaining tactile sensor/rendering
  gaps and final base-gate review are closed.

### Candidate Normal/Area Overlay Direct-Force Tactile Export

- Successful run tag: `p00_mjw_normarea_v1_20260701_071900`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_mjw_normarea_v1 RUN_TAG=p00_mjw_normarea_v1_20260701_071900 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube SCENE_CAMERA=1 SCENE_CAMERA_WIDTH=256 SCENE_CAMERA_HEIGHT=256 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_timeseries.npz`
- Video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile.md`
- Positive metrics:
  - status: `pass_candidate_direct_force_export`
  - official final test status: `pass`
  - read errors: `0`
  - frames: `240`
  - frames with pad-object contacts: `147`
  - max object lift: `0.22243636846542358` m
  - max pad-object candidate `Fn` sum: `40.0997428894043`
  - max pad-object candidate `Ft` sum: `12.027881622314453`
  - max left/right contact-area proxy cell ratio:
    `0.2900390625` / `0.279296875`
  - max left/right contact-area proxy map:
    `9.216480255126953` / `8.891389846801758`
  - max left/right normal-yz norm:
    `9.213287353515625` / `8.88884162902832`
  - material notify status: `pass`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
- Manual visual inspection:
  - sheet is nonblank;
  - real Newton head/right-wrist/left-wrist scene views remain present;
  - both pads now show three tactile rows during contact: `Fn + normal`,
    `Ft vector`, and `contact area proxy + normal`;
  - overlays are synchronized with the grasp/lift window and object-z/force
    curves.
- Classification:
  positive Phase 00 diagnostic improvement. The previous direct visual overlay
  gap for normals and contact-area proxy is now addressed for the candidate
  MJWarp direct-force video. This does not validate real contact-area semantics
  and does not provide gel/marker-style tactile camera rendering.

### Reference Video Comparison After Normal/Area Overlay

- Successful run tag: `p00_refcmp_normarea_v2_20260701_073000`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_refcmp_normarea_v2 RUN_TAG=p00_refcmp_normarea_v2_20260701_073000 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv REFERENCE_VIDEO=/public/home/yanhongru/Curiosity/0780e5ec3fdb26b63ae63de0f49f07c4.mp4 CANDIDATE_VIDEO=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile.avi CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_normarea_v1_20260701_071900/candidate_mjw_direct_tactile_summary.json bash experiments/configs/phase00/ref_tactile/launch_reference_video_compare_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_normarea_v2_20260701_073000/reference_video_compare_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/ref_compare/p00_refcmp_normarea_v2_20260701_073000/reference_video_compare.md`
- Comparison sheet:
  `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_normarea_v2_20260701_073000/reference_vs_candidate_sheet.jpg`
- Updated gate checklist:
  - current candidate channels now include contact-normal overlay from MJWarp
    `contact.frame`;
  - current candidate channels now include contact-area proxy overlay from
    pad-object point-contact density;
  - curiosity training remains disallowed.
- Remaining reference-video gaps:
  - gel/marker-style tactile camera rendering comparable to the reference
    video;
  - validated photometric/deformation marker tracking on the pad surface;
  - validated real contact-area semantics beyond the current point-contact
    density proxy;
  - reference-video channel-by-channel semantic matching beyond frame-level
    metrics;
  - final Gate 00D/00E review before restarting curiosity training.
- Manual visual inspection:
  - side-by-side sheet confirms the candidate is now more complete than
    `p00_refcmp_v3`: it includes `Fn + normal`, `Ft vector`, and
    `contact area proxy + normal` panels;
  - the reference video is still richer, especially in gel/marker-style tactile
    appearance, dense multi-panel tactile fields, and mechanics plots.
- Classification:
  positive reference-alignment improvement, not training and not curiosity
  success. Gate 00D/00E remain open.

### Candidate Gel/Marker-Style Direct-Force Rendering

- Successful run tag: `p00_mjw_marker_v1_20260701_074200`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_mjw_marker_v1 RUN_TAG=p00_mjw_marker_v1_20260701_074200 NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_main NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv NUM_FRAMES=240 SCENE=cube SCENE_CAMERA=1 SCENE_CAMERA_WIDTH=256 SCENE_CAMERA_HEIGHT=256 MATERIAL_LABEL=steel_spec_v1 OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 bash experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_timeseries.npz`
- Video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_sheet.jpg`
- Report:
  `experiments/reports/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.md`
- Positive metrics:
  - status: `pass_candidate_direct_force_export`
  - official final test status: `pass`
  - read errors: `0`
  - frames: `240`
  - frames with pad-object contacts: `146`
  - max object lift: `0.2225111573934555` m
  - max pad-object candidate `Fn` sum: `41.90861511230469`
  - max pad-object candidate `Ft` sum: `12.294239044189453`
  - max left/right candidate marker-flow norm:
    `4.690944671630859` / `3.1349213123321533`
  - material notify status: `pass`
  - observed `shape_material_mu=[0.30000001192092896]`
  - observed `shape_material_kh=[999999995904.0]`
- Added candidate channel:
  blue gel-like marker/deformation panel with marker displacement overlay
  derived from candidate `Fn`, `Ft`, contact normals, and contact-area proxy.
- Manual visual inspection:
  - sheet is nonblank;
  - marker panels show a regular blue gel-like grid before contact;
  - during contact/lift, both pads develop centered marker deformation/flow;
  - marker panels are synchronized with `Fn`, `Ft`, contact-area proxy,
    object-z, and force curves.
- Classification:
  positive candidate rendering improvement. This is not official Taccel output,
  not hardware photometric marker validation, not training, and not curiosity
  success.

### Reference Video Comparison After Candidate Marker Rendering

- Successful run tag: `p00_refcmp_marker_v1_20260701_074900`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_refcmp_marker_v1 RUN_TAG=p00_refcmp_marker_v1_20260701_074900 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv REFERENCE_VIDEO=/public/home/yanhongru/Curiosity/0780e5ec3fdb26b63ae63de0f49f07c4.mp4 CANDIDATE_VIDEO=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.avi CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json bash experiments/configs/phase00/ref_tactile/launch_reference_video_compare_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_video_compare_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_video_compare.md`
- Comparison sheet:
  `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_vs_candidate_sheet.jpg`
- Updated gate checklist:
  - current candidate channels include real Newton scene views;
  - current candidate channels include direct `Fn` and `Ft` maps;
  - current candidate channels include shear arrows, contact-normal overlay,
    and contact-area proxy overlay;
  - current candidate channels include candidate gel/marker-style rendering
    derived from direct-force fields;
  - curiosity training remains disallowed.
- Remaining reference-video gaps:
  - validated gel/marker photometric semantics comparable to the reference
    video;
  - validated photometric/deformation marker tracking on the pad surface;
  - validated real contact-area semantics beyond the current point-contact
    density proxy;
  - reference-video channel-by-channel semantic matching beyond frame-level
    visual metrics;
  - final Gate 00D/00E review before restarting curiosity training.
- Manual visual inspection:
  - side-by-side sheet shows the candidate is visually closer than
    `p00_refcmp_normarea_v2` because it now includes separate blue marker-style
    panels with deformation/flow;
  - the reference video still has more mature multi-panel tactile semantics,
    richer heatmap/marker coupling, and denser mechanics plots.
- Classification:
  positive reference-alignment improvement, not Gate 00D/00E completion and not
  curiosity success.

### Strict Phase 00 Gate 00D/00E Review

- Superseded run tag: `p00_gate_review_v1_20260701_080300`.
  - Cause: review script read alignment RMSE from legacy top-level field names
    and incorrectly marked the already-positive `p00_mjw_align_v1` alignment
    check as failed. This run is retained as script-debug evidence only.
- Successful current run tag: `p00_gate_review_v2_20260701_080800`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_gate_review_v2 RUN_TAG=p00_gate_review_v2_20260701_080800 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv bash experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_gate_review.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v2_20260701_080800/phase00_gate_review_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v2_20260701_080800/phase00_gate_review.md`
- Evidence inputs:
  - benchmark:
    `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_main_20260701_035529/newton_hydro_benchmark_summary.json`
  - candidate tactile:
    `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json`
  - reference comparison:
    `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_video_compare_summary.json`
  - alignment:
    `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json`
- Passed checks:
  - `official_newton_runtime_82_fps`
  - `base_grasp_lift_final_test`
  - `steel_spec_material`
  - `candidate_direct_fn_ft`
  - `sensorcontact_alignment`
  - `normal_and_area_proxy_overlay`
  - `candidate_gel_marker_render`
  - `reference_comparison_assets`
- Failed checks: none.
- Gate status:
  - overall status: `open_not_curiosity_ready`
  - Gate 00D: `open_reference_semantics_blocked`
  - Gate 00E: `open_tactile_validation_blocked`
  - curiosity training allowed: `false`
- Hard blockers:
  - validated gel/marker photometric semantics comparable to the reference
    video;
  - validated photometric/deformation marker tracking on the pad surface;
  - validated real contact-area semantics beyond the current point-contact
    density proxy;
  - reference-video channel-by-channel semantic matching beyond frame-level
    visual metrics.
- Classification:
  major bookkeeping/evidence milestone. The current base environment passes
  all mechanical/runtime/candidate-visual evidence checks, but it still cannot
  close Gate 00D/00E and cannot start curiosity training because tactile
  semantics remain candidate/proxy rather than validated reference-level
  tactile sensing.

### Channel-Level Semantic Layout Audit

- Successful run tag: `p00_chan_audit_v1_20260701_082100`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_chan_audit_v1 RUN_TAG=p00_chan_audit_v1_20260701_082100 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv bash experiments/configs/phase00/ref_tactile/launch_channel_semantic_audit_tmux.sh`
- Source:
  `src/newton_tactile_curiosity/phase00_channel_semantic_audit.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_channel_semantic_audit_in_alloc.sh`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_channel_semantic_audit_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit.md`
- Visual audit sheet:
  `experiments/visuals/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_sheet.jpg`
- Status:
  `pass_channel_audit_open_validation`
- Passed checks:
  - `candidate_scene_channel`
  - `candidate_marker_render_channel`
  - `candidate_force_heatmap_channels`
  - `candidate_area_proxy_channel`
  - `candidate_mechanics_curve_channel`
  - `reference_scene_tactile_mechanics_layout`
- Failed checks: none.
- Manual visual inspection:
  - audit sheet boxes the reference scene/tactile/mechanics bands and the
    candidate scene, marker, `Fn`, `Ft`, area, and curves regions;
  - candidate channel layout is now explicit and comparable at the visual
    layout level.
- Classification:
  positive channel-level audit milestone. This is not photometric validation,
  not real tactile-area validation, not final semantic equivalence, and not
  curiosity success.

### Gate Review With Channel Audit

- Successful run tag: `p00_gate_review_v3_20260701_082600`.
- Slurm job: `160324`
- Host: `server30`
- Launcher command:
  `JOB_ID=160324 WINDOW_NAME=p00_gate_review_v3 RUN_TAG=p00_gate_review_v3_20260701_082600 NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv bash experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v3_20260701_082600/phase00_gate_review_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v3_20260701_082600/phase00_gate_review.md`
- Added evidence input:
  `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_summary.json`
- Passed checks:
  - `official_newton_runtime_82_fps`
  - `base_grasp_lift_final_test`
  - `steel_spec_material`
  - `candidate_direct_fn_ft`
  - `sensorcontact_alignment`
  - `normal_and_area_proxy_overlay`
  - `candidate_gel_marker_render`
  - `reference_comparison_assets`
  - `channel_semantic_layout_audit`
- Failed checks: none.
- Updated hard blockers:
  - validated gel/marker photometric semantics comparable to the reference
    video;
  - validated photometric/deformation marker tracking on the pad surface;
  - validated real contact-area semantics beyond the current point-contact
    density proxy;
  - validated channel-level semantic equivalence beyond current layout audit.
- Classification:
  positive gate-review refinement. The project now has channel-level layout
  evidence, but Gate 00D remains `open_reference_semantics_blocked`, Gate 00E
  remains `open_tactile_validation_blocked`, and curiosity training remains
  disallowed.

### 2026 Tactile Reference Scan

- Date: 2026-07-01.
- New relevant references from web scan:
  - UniVTAC official GitHub: `https://github.com/univtac/UniVTAC`
  - UniVTAC project page: `https://univtac.github.io/`
  - Tacmap arXiv: `https://arxiv.org/abs/2602.21625`
  - TaCauchy arXiv/html: `https://arxiv.org/html/2606.20426`
  - ControlTac project/paper: `https://dongyuluo.github.io/controltac/` and
    `https://arxiv.org/abs/2505.20498`
- Interpretation:
  These are candidate reference/comparison paths for solving the remaining
  semantic-validation blockers. Do not replace the current Newton mainline or
  claim compatibility until official code/config sanity passes.

### Official Tactile Semantic Reference Source Audit

- Date: 2026-07-01.
- Login-node scope:
  source clone and text/document inspection only. No simulation, rendering,
  dependency installation, model loading, dataset conversion, training, or
  Python-heavy validation was run on the login node.
- UniVTAC:
  - official repository: `https://github.com/univtac/UniVTAC`
  - local path: `external/UniVTAC`
  - local commit: `05bcd3edb92237107efa40105292a24f1a9fd761`
  - role: official Isaac Lab/TacEx visuo-tactile manipulation benchmark and
    policy reference
  - relevant upstream fields: left/right tactile `rgb`, `rgb_marker`, `depth`,
    `marker`, tactile pose, head/wrist RGB, joint and end-effector state
  - relevant upstream baselines: ACT tactile-full, ACT vision-only, ablations,
    ViTAL-style visuo-tactile pretraining path
- TaCauchy:
  - official repository: `https://github.com/figsama/TaCauchy`
  - local path: `external/TaCauchy`
  - local commit: `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
  - role: official Isaac Sim/Lab plus UIPC FEM tactile semantic reference
  - relevant upstream semantics: Cauchy stress, normal pressure, tangential
    traction, adaptive mesh refinement, force-field visualization, tactile RGB
- New report:
  `experiments/reports/phase00/ref_tactile/semantic_reference_audit.md`
- New reference matrix:
  `experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`
- Classification:
  positive source-audit milestone for Gate 00F. This does not close Gate 00D,
  Gate 00E, or Gate 00F. Curiosity training remains disallowed until official
  compute-side sanity and semantic mapping either pass or record faithful
  blockers.

### Gate 00F Official Reference Sanity/Blocker Probes

- Date: 2026-07-01.
- Held allocation:
  - Slurm job: `160450`
  - Host: `server02`
  - Job name: `curiosity_p00_ref_sem_1gpu_1day`
  - Launcher:
    `experiments/configs/phase00/ref_tactile/launch_phase00_ref_allocation_tmux.sh`
  - Allocation was released after blocker probes because no further approved
    compute work was available.
- UniVTAC probe:
  - run tag: `p00_ref_univtac_sanity_v1_20260701_054900`
  - runner:
    `experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh`
  - summary:
    `experiments/outputs/phase00/ref_tactile/reference_sanity/p00_ref_univtac_sanity_v1_20260701_054900/univtac_official_reference_sanity_summary.json`
  - report:
    `experiments/reports/phase00/ref_tactile/reference_sanity/p00_ref_univtac_sanity_v1_20260701_054900/univtac_official_reference_sanity.md`
  - status: `blocked_missing_prebuilt_environment`
  - observed commit matched expected:
    `05bcd3edb92237107efa40105292a24f1a9fd761`
  - blocker:
    no executable prebuilt Python found; set `UNIVTAC_PYTHON` or prepare
    `envs/univtac/.venv` before official compute sanity.
- TaCauchy probe:
  - run tag: `p00_ref_tacauchy_sanity_v1_20260701_054900`
  - runner:
    `experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh`
  - summary:
    `experiments/outputs/phase00/ref_tactile/reference_sanity/p00_ref_tacauchy_sanity_v1_20260701_054900/tacauchy_official_reference_sanity_summary.json`
  - report:
    `experiments/reports/phase00/ref_tactile/reference_sanity/p00_ref_tacauchy_sanity_v1_20260701_054900/tacauchy_official_reference_sanity.md`
  - status: `blocked_missing_prebuilt_environment`
  - observed commit matched expected:
    `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
  - blocker:
    no executable prebuilt Python found; set `TACAUCHY_PYTHON` or prepare
    `envs/tacauchy/.venv` before official compute sanity.
- Classification:
  honest official-reference blocker evidence. This is not training, not Gate
  00D/00E/00F completion, and not curiosity success.

### Gate Review V4 With Official Semantic Reference Blockers

- Successful run tag: `p00_gate_review_v4_20260701_055100`.
- Slurm job: `160450`
- Host: `server02`
- Source:
  `src/newton_tactile_curiosity/phase00_gate_review.py`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v4_20260701_055100/phase00_gate_review_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v4_20260701_055100/phase00_gate_review.md`
- Passed checks:
  - `official_newton_runtime_82_fps`
  - `base_grasp_lift_final_test`
  - `steel_spec_material`
  - `candidate_direct_fn_ft`
  - `sensorcontact_alignment`
  - `normal_and_area_proxy_overlay`
  - `candidate_gel_marker_render`
  - `reference_comparison_assets`
  - `channel_semantic_layout_audit`
  - `semantic_reference_matrix_available`
- Failed checks:
  - `univtac_official_reference_sanity`
  - `tacauchy_official_reference_sanity`
- Gate status:
  - Gate 00D: `open_reference_semantics_blocked`
  - Gate 00E: `open_tactile_validation_blocked`
  - Gate 00F: `open_official_semantic_validation_blocked`
  - curiosity training allowed: `false`
- Classification:
  positive gate-enforcement milestone and blocker evidence. The project now
  prevents the candidate marker/layout evidence from being mistaken for
  official tactile semantic validation.

### Official Reference Environment Plan

- Date: 2026-07-01.
- Report:
  `experiments/reports/phase00/ref_tactile/reference_environment_plan.md`
- Source inspection:
  - UniVTAC official installer requires a heavy all-in-one environment build:
    conda env, PyTorch, Isaac Sim 4.5, Isaac Lab 2.1.1, cuRobo, modified TacEx,
    UIPC/libuipc, tests, and data collection.
  - TaCauchy requires Isaac Sim/Lab, UIPC/libuipc, GCC/CMake/CUDA toolchain
    settings, and separate TacEx tactile assets.
- Local environment audit:
  no approved executable `envs/univtac/.venv` or `envs/tacauchy/.venv` exists,
  and metadata-only scans found no reusable installed Isaac/TacEx/UIPC package
  directories in current Curiosity envs.
- Decision:
  prepare official reference environments as controlled local shared-filesystem
  envs before compute sanity. Do not install dependencies on compute nodes and
  do not start curiosity training while Gate 00F is blocked.

### Official Reference Environment Dry-Run Stage Commands

- Date: 2026-07-01.
- Config manifest:
  `experiments/configs/phase00/ref_tactile/envprep/reference_env_manifest_v1.json`
- Stage runner:
  `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh`
- Scope:
  dry-run command generation only. No conda environment creation, dependency
  install, Isaac launch, TacEx/UIPC build, asset download, simulation, data
  collection, training, or official sanity was executed.
- Generated targets:
  - `univtac`
  - `tacauchy`
- Generated stages for each target:
  - `preflight`
  - `create_env`
  - `install_isaac`
  - `install_isaaclab`
  - `install_curobo_or_assets`
  - `install_tacex_core`
  - `build_uipc`
  - `setup_assets`
  - `official_sanity`
- Output command/status root:
  `experiments/outputs/phase00/ref_tactile/envprep/`
- Report root:
  `experiments/reports/phase00/ref_tactile/envprep/`
- Preflight dry-run status:
  - UniVTAC expected and observed commit matched
    `05bcd3edb92237107efa40105292a24f1a9fd761`.
  - TaCauchy expected and observed commit matched
    `c228cfe9050904cd5d71d64f6eb5104768d4cbda`.
- Classification:
  controlled environment-preparation planning. This does not close Gate 00F.
  The real blocker remains missing approved executable prebuilt environments
  for official UniVTAC/TaCauchy sanity.

### Official Reference Environment Toolchain Preflight

- Date: 2026-07-01.
- Report:
  `experiments/reports/phase00/ref_tactile/envprep/toolchain_preflight.md`
- Scope:
  lightweight executable/version inspection only on the login node.
- Available:
  - Curiosity local conda at `envs/taccel/miniforge/bin/conda`
  - conda `26.3.2`
  - `/usr/bin/gcc-11` and `/usr/bin/g++-11`, version `11.4.0`
  - existing local package/cache directories under `envs/taccel/`
- Missing from login environment:
  - `cmake`
  - `git-lfs`
  - `nvcc`
  - `nvidia-smi`
  - shell-level `conda` on `PATH` by default
- Classification:
  environment-preparation risk evidence. This does not close Gate 00F and does
  not permit curiosity training.

### Semantic Bridge Spec

- Date: 2026-07-01.
- Config:
  `experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json`
- Report:
  `experiments/reports/phase00/ref_tactile/semantic_bridge_spec.md`
- Candidate source:
  `p00_mjw_marker_v1_20260701_074200`
- Official references:
  - UniVTAC fields: tactile `rgb`, `rgb_marker`, `depth`, `marker`, and pose.
  - TaCauchy fields: Cauchy stress, traction, normal pressure, tangential
    traction, nodal/tributary contact area, and pressure-normalized force.
- Bridge items:
  - `candidate.newton_mjw.Fn`
  - `candidate.newton_mjw.Ft`
  - `candidate.newton_mjw.marker_flow`
  - `candidate.newton_mjw.area_proxy`
  - `candidate.newton_mjw.contact_normal`
  - `candidate.newton_mjw.scene_rgb`
- Classification:
  concrete Gate 00F validation spec. It is source/document-level mapping only,
  not official UniVTAC/TaCauchy sanity and not a Gate 00F pass.

### Gate Review Bridge-Spec Enforcement

- Date: 2026-07-01.
- Updated source:
  `src/newton_tactile_curiosity/phase00_gate_review.py`
- Updated runners:
  - `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`
  - `experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`
- New gate-review input:
  `--semantic-bridge-spec`
- New check:
  `semantic_bridge_spec_available`
- Verification:
  lightweight `python3 -m py_compile`, `bash -n`, `jq empty`, and
  `git diff --check` passed on the edited files.
- Classification:
  gate-enforcement improvement only. The gate review was not rerun after this
  code change because no Curiosity compute allocation was active, and running
  validation builders on the login node is disallowed.

### Gate Review V5 With Bridge-Spec Enforcement

- Successful run tag: `p00_gate_review_v5_20260701_060100`.
- Slurm job: `160454`
- Host: `server02`
- Launcher:
  `experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`
- Runner:
  `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`
- Summary:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v5_20260701_060100/phase00_gate_review_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v5_20260701_060100/phase00_gate_review.md`
- Added passed check:
  `semantic_bridge_spec_available`
- Still failed checks:
  - `univtac_official_reference_sanity`
  - `tacauchy_official_reference_sanity`
- Gate status:
  - Gate 00D: `open_reference_semantics_blocked`
  - Gate 00E: `open_tactile_validation_blocked`
  - Gate 00F: `open_official_semantic_validation_blocked`
  - curiosity training allowed: `false`
- Resource handling:
  Slurm job `160454` was cancelled after the validation report completed to
  avoid idle GPU allocation.
- Classification:
  positive gate-enforcement verification. It proves the bridge-spec check is
  wired into compute-side Gate review, but it does not close Gate 00F.

### Latest 2026 Tactile Reference Code Recheck

- Date: 2026-07-01.
- Report:
  `experiments/reports/phase00/ref_tactile/latest_reference_code_recheck.md`
- Web/source findings:
  - Tacmap remains code-unavailable in the current audit. ArXiv/html has no
    GitHub/code link; common GitHub remote probes returned `Repository not
    found` or timed out.
  - ControlTac remains code-unavailable in the current audit. The project page
    has no code/GitHub link; common GitHub remote probes returned `Repository
    not found`.
  - FreeTacMan official repository was cloned to `external/FreeTacMan` at
    `9285740a5d33385d3a9cf5ccdb185e3387b547bd`.
  - DiffTactile official repository was cloned to `external/DiffTactile` at
    `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`.
  - DiffTactile source audit found FEM tactile sensor models, marker
    extraction, contact-rich tasks (`grasp_elastic`, `object_repose`,
    `surface_follow`), and baseline code for CMA-ES/PPO/SAC/RNN.
  - DiffTactile `requirements.txt` is UTF-16 little-endian text with CRLF line
    endings, so it must not be blindly passed to a normal installer without
    encoding-aware review.
- Classification:
  source-audit improvement only. DiffTactile is a secondary comparison path.
  Mandatory Gate 00F remains UniVTAC plus TaCauchy, and curiosity training
  remains disallowed.

### Policy And Photometric Reference Audit

- Cloned Reactive Diffusion Policy:
  `external/reactive_diffusion_policy`,
  commit `824c5e8de1fd1811106907a04b5f0186e0138c0b`.
- Cloned ImplicitRDP:
  `external/ImplicitRDP`,
  commit `4c90646df17787e31c88838106c4a0323ddefb4a`.
- Cloned Tactile Diffusion:
  `external/Tactile-Diffusion`,
  commit `16868fb96d19d93dc5837600c26b48415632e4f6`.
- Checked Action Conditioned Tactile Prediction:
  remote HEAD `085d2ab82d2e0574f39a359dd2c445b8f7f7a3b3`; local clone failed
  with `fetch-pack: unexpected disconnect while reading sideband packet`.
- Wrote:
  `experiments/reports/phase00/ref_tactile/policy_reference_audit.md`.
- Updated:
  `experiments/configs/phase00/ref_tactile/curiosity_reference_matrix_v1.json`
  and
  `experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`.
- Classification:
  source/text audit only. No checkpoint download, model loading, training,
  evaluation, rendering, simulation, or dataset conversion was run. These are
  future comparison/design references, not current Phase 00 gate completion.

### Phase 00 Requirement-Status Audit

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
- Updated active tracking:
  - `PLAN/00_ref_tactile_env/plan.md`
  - `TODO/00_ref_tactile_env/todo.md`
  - `experiments/reports/phase00/ref_tactile/active_evidence_index.md`
- Classification:
  status audit only. This is not training, not Gate 00D/00E/00F completion,
  not a base-model success claim, and not a curiosity-learning success claim.
- Current conclusion:
  latest source audits and the Newton main steel-spec MJWarp direct-force
  candidate are partial positive evidence; the 92.6 FPS official Panda hydro
  base is partial positive base evidence; Gate 00F remains blocked by missing
  approved UniVTAC/TaCauchy prebuilt environments; curiosity training remains
  disallowed.

### Reference Env Location Audit

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_location_audit.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/envprep/reference_env_location_audit_v1.json`.
- Refreshed availability preflight:
  `experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh`.
- Latest availability output:
  `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`.
- Classification:
  lightweight login-node file-location audit only. No simulation, rendering,
  training, model loading, package import, dependency installation, dataset
  conversion, or Slurm allocation was run.
- Result:
  no approved prebuilt UniVTAC or TaCauchy Python environment was found in
  project `envs/` or common home conda/env locations. Current shell also lacks
  `conda`, `mamba`, `micromamba`, `module`, `cmake`, `git-lfs`, and `nvcc`.
  Project-local `envs/taccel/miniforge/bin/conda` exists (`conda 26.3.2`),
  but target UniVTAC/TaCauchy envs are absent.
- Interpretation:
  Gate 00F remains blocked. Do not install dependencies on compute nodes and
  do not start curiosity training.

### Reference Env Stage Checklist

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_stage_checklist.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/envprep/reference_env_stage_checklist_v1.json`.
- Source documents inspected:
  - `external/UniVTAC/docs/Installation.md`
  - `external/UniVTAC/scripts/install.sh`
  - `external/UniVTAC/collect_data.sh`
  - `external/UniVTAC/policy/task_settings.json`
  - `external/TaCauchy/README.md`
  - `external/TaCauchy/REPRODUCTION.md`
  - `external/TaCauchy/ASSETS.md`
  - `external/TaCauchy/docs/source/installation/Local-Installation.md`
  - `external/TaCauchy/TACTILE_VISUALIZATION_GUIDE.md`
- Classification:
  planned environment-preparation checklist only. No dependency installation,
  build, official demo, simulation, rendering, data collection, training, model
  loading, or Slurm allocation was run.
- Result:
  UniVTAC and TaCauchy should remain separate env targets because official docs
  specify different Isaac Sim / Isaac Lab versions. UniVTAC also requires its
  modified bundled TacEx, while TaCauchy requires large tactile assets and
  UIPC/libuipc stress/contact extraction. The project-local Miniforge conda is
  available as an env-creator candidate, but heavy env construction has not
  been started.

### Reference Env Stage Runner Guard Refresh

- Date: 2026-07-01.
- Updated:
  `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh`.
- Refreshed dry-run status for both `univtac` and `tacauchy` stages:
  - `preflight`: `dry_run_preflight_ready`
  - `create_env`: `dry_run_create_env_ready_not_executed`
  - all later stages: `blocked_missing_target_env`
- Status roots:
  - `experiments/outputs/phase00/ref_tactile/envprep/univtac/`
  - `experiments/outputs/phase00/ref_tactile/envprep/tacauchy/`
- Classification:
  guard and evidence-refresh only. No environment creation, dependency
  installation, build, official demo, simulation, rendering, data collection,
  training, model loading, or Slurm allocation was run.

### Reference Asset Availability Audit

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/envprep/reference_asset_availability.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json`.
- Classification:
  lightweight file-presence audit only. No asset setup, Git LFS download,
  package import, simulation, rendering, data collection, training, model
  loading, or Slurm allocation was run.
- Result:
  UniVTAC bundled TacEx has useful GelSight/GF225/shape assets present.
  TaCauchy has partial placeholder assets only: GelSight/DIGIT are mostly docs
  and `params.json`, 9dtact has `.before_*` USD backups, Franka has a broken
  USD, and `Props/tactile_test_shapes` is missing.
- Interpretation:
  Gate 00F remains blocked by both missing target environments and incomplete
  TaCauchy official tactile assets.

### Reference Asset Reuse Plan

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/envprep/reference_asset_reuse_plan.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json`.
- Classification:
  candidate plan only. No assets were copied, downloaded, modified, or
  installed.
- Observation:
  UniVTAC bundled TacEx data is about `410M`; TaCauchy current data is about
  `1.8M`. The UniVTAC bundled tree has useful GelSight/GF225/Franka/test-shape
  assets that may serve as a local asset source if exact provenance is accepted.
- Risk:
  UniVTAC uses a modified bundled TacEx asset/source path and may not exactly
  match TaCauchy expected upstream assets. Reuse also does not solve missing
  target envs or UIPC/libuipc build.

### Reference Asset Stage Runner Guard

- Date: 2026-07-01.
- Wrote:
  `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_asset_stage.sh`.
- Refreshed dry-run status for TaCauchy asset stages:
  - `audit`: `dry_run_asset_audit_ready`
  - `reuse_copy`: `dry_run_asset_reuse_copy_not_executed`
  - `verify`: `blocked_missing_reused_assets`
- Status root:
  `experiments/outputs/phase00/ref_tactile/envprep/assets/tacauchy/`
- Classification:
  guard and evidence-refresh only. No assets were copied, downloaded, modified,
  or installed.

### Gate Review Asset-Blocker Enforcement

- Date: 2026-07-01.
- Updated:
  `src/newton_tactile_curiosity/phase00_gate_review.py`.
- Updated:
  `experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh`.
- Updated:
  `experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh`.
- New gate-review inputs:
  - `--reference-asset-availability-summary`
  - `--reference-asset-reuse-plan`
- New checks:
  - `reference_asset_availability`
  - `reference_asset_reuse_plan_available`
- Verification:
  lightweight `python3 -m py_compile`, `bash -n`, `jq empty`, and
  `git diff --check` passed on the edited files.
- Classification:
  gate-enforcement improvement only. The Gate review was not rerun after this
  code change because running validation/report builders on the login node is
  disallowed and no Curiosity compute allocation was active.

### Latest Source Remote Refresh

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/latest_source_remote_refresh.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/latest_source_remote_refresh_v1.json`.
- Method:
  lightweight web/source search plus `git ls-remote` only. No external repo
  checkout was modified.
- Key result:
  Newton upstream main is now
  `d58e70266be0db803261f3e46a2f7d923a43db37`, while active evidence
  `external/newton_main` remains at
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`.
- Other matching local/remote heads:
  Taccel, UniVTAC, TaCauchy, HydroShear, FreeTacMan, DiffTactile, APPLE,
  Tactile MNIST, Reactive Diffusion Policy, ImplicitRDP, and Tactile
  Diffusion.
- Gaps:
  T-Rex local is behind remote main and has existing dirty state that must not
  be overwritten silently. The probed `yanglh14/IsaacLabTactile` URL returned
  repository not found.
- Classification:
  source-refresh evidence only. This is not a code update, not official sanity,
  not training, and not Gate completion.

### Latest Newton d58 Worktree Preparation

- Date: 2026-07-01.
- Added worktree:
  `external/newton_d58`.
- Commit:
  `d58e70266be0db803261f3e46a2f7d923a43db37`.
- Preserved active evidence worktree:
  `external/newton_main` at
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`.
- Wrote:
  `experiments/reports/phase00/ref_tactile/newton_d58_worktree_status.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/newton_d58_worktree_status_v1.json`.
- Classification:
  code preparation only. No Newton run, Python experiment, simulation,
  rendering, benchmark, training, evaluation, or Slurm allocation was run.
- Next required action:
  run official Newton hydro benchmark/sanity in a Curiosity tmux-held H200
  allocation with `NEWTON_ROOT` set to `external/newton_d58`.

### Newton d58 Allocation Request

- Date: 2026-07-01.
- Requested through:
  `experiments/configs/phase00/ref_tactile/launch_phase00_ref_allocation_tmux.sh`.
- tmux session/window:
  `curiosity_phase00_ref_tactile:alloc_d58`.
- Job name:
  `curiosity_p00_d58_1gpu_1day`.
- Observed Slurm job id:
  `160467`.
- Observed state:
  queued and waiting for resources.
- Wrote:
  `experiments/reports/phase00/ref_tactile/newton_d58_allocation_request.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/newton_d58_allocation_request_v1.json`.
- Classification:
  allocation request only. No benchmark, simulation, rendering, training,
  evaluation, or Gate completion.

### Newton d58 Official Hydro Benchmark

- Date: 2026-07-01.
- Worktree:
  `external/newton_d58`.
- Commit:
  `d58e70266be0db803261f3e46a2f7d923a43db37`.
- Slurm job/host:
  `160467` / `server02`.
- First run:
  `p00_bench_d58_v1_20260701_070459`, `70.8 FPS`, execution pass, below the
  82 FPS target.
- Hot/longer run:
  `p00_bench_d58_hot_v1_20260701_070611`, `82.7 FPS`, execution pass, meets
  the 82 FPS target.
- Wrote:
  `experiments/reports/phase00/ref_tactile/newton_d58_benchmark_status.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/newton_d58_benchmark_status_v1.json`.
- Classification:
  latest-upstream Newton runtime sanity only. This is not tactile export, not
  training, not curiosity success, and not Gate completion.

### Gate 00F TacSL Source Compatibility Record

- Date: 2026-07-01.
- Validator:
  `src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py`.
- Handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_handoff.md`.
- Current report:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`.
- Current summary:
  `experiments/outputs/phase00/ref_tactile/tacsl_source_compat/p00_tacsl_src_compat_20260701/tacsl_source_compat_summary.json`.
- Status:
  `pass_tacsl_source_compat`.
- Scope:
  static source compatibility for `external/IsaacLab_official` VERSION
  `2.3.2` and candidate image ref `nvcr.io/nvidia/isaac-lab:2.3.2`.
- Verified source features:
  required TacSL data fields, demo flags, and import strings are present.
- Updated active records:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`,
  `PLAN/00_ref_tactile_env/plan.md`,
  `TODO/00_ref_tactile_env/todo.md`,
  `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`,
  and
  `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
- Verification:
  lightweight `python3 -m py_compile`, `jq empty`, and `git diff --check`
  passed on the relevant files.
- Classification:
  source compatibility only. No container was pulled or built, no Isaac/TacSL
  module was imported, no Isaac Sim process ran, no runtime was registered,
  no official sanity was run, Gate 00F remains open, and curiosity training
  remains disallowed.

### Gate 00F Runtime Registration Handoff

- Date: 2026-07-01.
- Added:
  `src/newton_tactile_curiosity/gate00f_runtime_register.py`.
- Handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registration_handoff.md`.
- Machine-readable handoff:
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registration_handoff_v1.json`.
- Purpose:
  write a copied candidate runtime registry when a real dependency-complete
  Python environment, local Docker image ID, or shared container artifact is
  available.
- Guardrails:
  rejects placeholder values, rejects excluded resource-zone paths, requires
  existing provenance paths and expected modules, requires executable Python
  paths for `python_env`, and requires a local `image_id` or existing
  `artifact_path` for `container`.
- Container registration tightening:
  `gate00f_runtime_register.py` now requires container registrations to include
  `--container-provenance-summary` with status
  `pass_gate00f_container_provenance` and matching target before it will write
  a candidate registry.
- Negative control:
  a registration attempt using dummy `image_id=sha256:negative-control-dummy`
  and the failed IsaacLab remote-image-only provenance summary exited with code
  `1`, rejected the `fail_gate00f_container_provenance` status, and did not
  write
  `experiments/outputs/phase00/ref_tactile/runtime_registry/p00_register_negative_failed_provenance_20260701/candidate_registry.json`.
- Validator update:
  `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py` now
  records `provenance_results` and fails missing or placeholder provenance.
  The refreshed current registry summary still reports
  `fail_gate00f_runtime_registry`; all current provenance paths exist, and the
  remaining failures are base-env/missing-runtime status and resolution-path
  issues.
- Updated active records:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`,
  `PLAN/00_ref_tactile_env/plan.md`,
  `TODO/00_ref_tactile_env/todo.md`,
  `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`,
  and
  `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
- Classification:
  metadata handoff only. It does not pull/build images, run containers, import
  Isaac/TacSL modules, install dependencies, run official sanity, clear Gate
  00F, or allow curiosity training.

### Gate 00F Scoped Project Artifact Probe

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/gate00f_project_artifact_probe_20260701.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/gate00f_project_artifact_probe_20260701_v1.json`.
- Scope:
  bounded project-local search under `envs`, `experiments/configs`,
  `experiments/outputs`, and `external`.
- Result:
  no `.sif`, `.sqsh`, `.tar`, `.tar.gz`, or `.img` container artifact was
  found at max depth `5`.
- Env tool result:
  no `cmake`, `git-lfs`, `singularity`, `apptainer`, or `docker` file was found
  under `envs` at max depth `4`; only `envs/taccel/cuda-toolkit/bin/nvcc` was
  found.
- Classification:
  scoped blocker evidence only. No image was pulled or built, no dependency
  was installed, no runtime was registered, Gate 00F remains open, and
  curiosity training remains disallowed.

### Gate 00F Container Provenance Contract

- Date: 2026-07-01.
- Added:
  `src/newton_tactile_curiosity/gate00f_container_provenance_validate.py`.
- Contract:
  `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`.
- Machine-readable contract:
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_contract_v1.json`.
- Negative-control packet:
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_isaaclab_ref_only_20260701_v1.json`.
- Negative-control summary:
  `experiments/outputs/phase00/ref_tactile/container_provenance/p00_isaaclab_ref_only_20260701/container_provenance_validation_summary.json`.
- Result:
  `fail_gate00f_container_provenance` because the packet contains only remote
  `image_ref=nvcr.io/nvidia/isaac-lab:2.3.2`, with no local `image_id` and no
  existing shared `artifact_path`.
- Source metadata recorded:
  IsaacLab official `b4c321024792976150ca55fddb26fa34480d974e`, UniVTAC
  `05bcd3edb92237107efa40105292a24f1a9fd761`, TacEx
  `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`, and TaCauchy
  `c228cfe9050904cd5d71d64f6eb5104768d4cbda`.
- Classification:
  guard evidence only. No container was pulled or built, no runtime was
  registered, no official sanity was run, Gate 00F remains open, and curiosity
  training remains disallowed.

### Gate 00F Runtime Intake Chain

- Date: 2026-07-01.
- Added:
  `src/newton_tactile_curiosity/gate00f_runtime_intake_chain.py`.
- Handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff.md`.
- Machine-readable handoff:
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff_v1.json`.
- Negative-control summary:
  `experiments/outputs/phase00/ref_tactile/runtime_intake/p00_isaaclab_ref_only_20260701/runtime_intake_summary.json`.
- Result:
  remote-image-only IsaacLab packet stops at `fail_container_provenance`
  before registry registration, and no `candidate_registry.json` is written.
- Classification:
  metadata-only chain guard. A future pass would only make a copied candidate
  registry ready for runtime preflight; it would not clear Gate 00F or allow
  curiosity training.

### Latest Source Freshness V4

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/latest_source_freshness_20260701_v4.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/latest_source_freshness_20260701_v4.json`.
- Method:
  lightweight `git ls-remote` plus local `git rev-parse` only.
- Result:
  tracked official refs for Newton, Taccel, T-Rex main/full-pipeline,
  IsaacLab, TacEx, TaCauchy, UniVTAC, FTP-1, AnyTouch2, and HydroShear match
  the current source records.
- Newton decision:
  `external/newton_8c501` matches latest remote main but remains negative
  runtime-target evidence from the `80.1 FPS` and `80.8 FPS` H200 runs;
  `external/newton_d58` remains the current strongest runtime/tactile
  candidate.
- Classification:
  source freshness only. No checkout was modified, no dependency was installed,
  no official sanity was run, Gate 00F remains open, and curiosity training
  remains disallowed.

### Gate 00E Base Evidence Audit

- Date: 2026-07-01.
- Added:
  `src/newton_tactile_curiosity/gate00e_base_evidence_audit.py`.
- Wrote:
  `experiments/reports/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit.md`.
- Wrote:
  `experiments/outputs/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit_summary.json`.
- Result:
  `partial_positive_gate00e_base_candidate_tactile_validation_blocked`.
- Positive evidence:
  d58 meets the 82 FPS target, passes the official final lift test, uses
  steel-spec candidate material settings, exports candidate Fn/Ft tactile
  mechanics, has contact/tactile density, and has nonblank reference
  comparison/channel audit assets.
- Remaining blocker:
  Gate 00E remains open because tactile semantics, real contact-area semantics,
  and official UniVTAC/TaCauchy reference sanity remain unresolved.
- Classification:
  base evidence audit only. It does not clear Gate 00E or Gate 00F and does
  not allow curiosity training.

### Gate 00D Environment Evidence Audit

- Date: 2026-07-01.
- Wrote:
  `experiments/reports/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701.md`.
- Wrote:
  `experiments/configs/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701_v1.json`.
- Result:
  `partial_positive_environment_candidate_reference_semantics_blocked`.
- Positive evidence:
  d58 has real scene views, left/right candidate Fn/Ft maps, shear arrows,
  contact-normal overlay, steel-spec candidate material settings,
  reference-comparison assets, and time-series mechanics.
- Remaining blocker:
  contact area is still proxy-only, dense penetration/compression semantics
  are not validated, and official UniVTAC/TaCauchy/IsaacLab TacSL semantic
  sanity remains blocked.
- Classification:
  environment evidence audit only. It does not clear Gate 00D or allow
  curiosity training.

### Supplementary Codebase Audit

- Date: 2026-07-01.
- Added source-only audit:
  `experiments/reports/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701.md`.
- Added machine-readable record:
  `experiments/configs/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701_v1.json`.
- Recorded `external/TactSim-IsaacLab` at
  `4f92257177cd0ee18928de720b880505ec7f7638` as a secondary photometric
  GelSight/DIGIT-style IsaacLab tactile reference.
- Recorded `external/newton-actuators` at
  `134dacb0912f4b8ce0465ecebf564479f2e62315` as deprecated Newton actuator
  background only; active Newton actuator work should remain inside current
  Newton.
- Recorded UniT remote HEAD
  `52a286520b09708934b25c77aa826360d72c79db` as remote-only future tactile
  representation evidence.
- Classification:
  source audit only. No runtime was registered, no official sanity was run, no
  checkpoint was downloaded or loaded, no simulation or training ran, and Gate
  00D/00E/00F/G remain unchanged.

### Gate 00F Container Guard Hardening

- Date: 2026-07-01.
- Updated:
  `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`.
- Updated:
  `src/newton_tactile_curiosity/gate00f_container_provenance_validate.py`.
- Updated:
  `src/newton_tactile_curiosity/gate00f_runtime_register.py`.
- Updated documentation and machine-readable contracts:
  `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_contract_v1.json`,
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701.md`,
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701_v1.json`.
- New guard:
  future container `image_id` values must look like immutable local digests or
  IDs, must not equal `image_ref`, and container `artifact_path` values must
  exist as files with `.sif`, `.sqsh`, `.tar`, `.tar.gz`, or `.img` suffixes.
- Classification:
  metadata guard only. This does not register a runtime, run a container, run
  official sanity, clear Gate 00F, or allow curiosity training.

### Runtime Gate Correction

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/runtime_gate_correction_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/runtime_gate_correction_20260701_v1.json`.
- Corrected active policy:
  `82 FPS` is historical reference only, not a hard blocker. Runtime around
  `80 FPS` is acceptable for continuing dense tactile export and Gate checks.
- Newton 8c501 effect:
  the `80.1 FPS` and `80.8 FPS` H200 benchmark runs are acceptable continuation
  evidence and must not block 8c501 dense tactile export.
- Remaining real blockers:
  tactile semantic completeness, base grasp/lift/hold tactile evidence, and
  official UniVTAC/TaCauchy/IsaacLab TacSL sanity or faithful blocker evidence.

### Newton 8c501 Continuation Chain

- Date: 2026-07-01.
- Slurm job: `160924`.
- Host: `server30`.
- Added:
  `experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`.
- Dense tactile export:
  `p00_mjw_8c501_cont_20260701_1924`, status
  `pass_candidate_direct_force_export`, `240` frames, `147` contact frames,
  max lift `0.22243839502334595 m`, max Fn `40.09991455078125`, max Ft
  `12.027889251708984`.
- Reference comparison:
  `p00_refcmp_8c501_cont_20260701_1925`, status
  `pass_reference_comparison_assets`.
- Channel audit:
  `p00_chan_8c501_cont_20260701_1926`, status
  `pass_channel_audit_open_validation`.
- Gate review:
  `p00_gate_8c501_cont_20260701_1927`, status
  `open_not_curiosity_ready`; passed runtime-around-80, base grasp/lift,
  steel material, Fn/Ft, alignment, reference comparison, and channel layout
  checks.
- Remaining blockers:
  official UniVTAC/TaCauchy/IsaacLab TacSL sanity, validated photometric
  marker semantics, validated deformation marker tracking, and real contact
  area semantics.
- Classification:
  latest-source positive candidate tactile evidence only. It does not clear
  Gate 00D/00E/00F and does not allow curiosity training.

### Gate 00F Post-8c501 Runtime Acceptance Lock

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff_v1.json`.
- Updated:
  `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`.
- Updated:
  `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`.
- Updated:
  `src/newton_tactile_curiosity/gate00f_runtime_register.py`.
- Updated active records:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`,
  `PLAN/00_ref_tactile_env/plan.md`, and
  `TODO/00_ref_tactile_env/todo.md`.
- Runtime policy:
  the latest 8c501 candidate evidence should be reused for future Gate 00F
  bundle attempts; do not rerun Newton candidate export just to chase the old
  82 FPS reference.
- Preflight guard:
  runtime preflight now reads accepted registry entries as authoritative. It
  supports registered Python envs, docker local image IDs, and
  singularity/apptainer/sif artifact paths for module-spec checks only. Enroot,
  sqsh, and tar still require explicit runners.
- Container registration:
  container entries may record `container_python`, defaulting to `python3`.
- Validation:
  `bash -n` passed for the runtime preflight and bundle launch scripts; `jq
  empty` passed for the updated JSON handoffs; `git diff --check` passed for
  the touched files; safety search found only the expected "82 FPS is not a
  hard gate" wording.
- Classification:
  guard/handoff only. This does not register a runtime, run a container, run
  official sanity, clear Gate 00F, or allow curiosity training.

### Gate 00F TacSL Container Documentation Refresh

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701_v1.json`.
- Checked official web sources:
  NVIDIA NGC Isaac Lab container catalog, Isaac Lab visuo-tactile sensor docs,
  Isaac Lab Docker guide, Isaac Lab Docker example guide, and IsaacLab issue
  `4528`.
- Local static check:
  TacSL source paths exist under `external/IsaacLab_official`, but no `bg.jpg`
  or obvious GelSight R15 background asset was found.
- Runtime risk:
  the official TacSL sanity with `--use_tactile_rgb` may fail if the GelSight
  background asset is absent. Do not silently drop tactile RGB to make Gate 00F
  pass.
- Classification:
  documentation/static-source risk record only. No container was pulled or run,
  no module was imported, no simulation ran, and Gate 00F remains open.

### Gate 00F Runtime Preflight Login Refuse After Container Support

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701_v1.json`.
- Command:
  `RUN_TAG=p00_runtime_preflight_login_refuse_after_container_support_20260701 bash experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`.
- Result:
  exit code `2`; stderr `ERROR: must run inside a Slurm allocation.`
- Classification:
  login-node safety check only. No registry validation, container command,
  module import, simulation, official sanity, or training ran.

### Gate 00F IsaacLab Upstream Freshness

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701_v1.json`.
- Local source:
  `external/IsaacLab_official` at
  `b4c321024792976150ca55fddb26fa34480d974e`, VERSION `2.3.2`.
- Upstream probe:
  official `HEAD` and `refs/heads/main` also point to
  `b4c321024792976150ca55fddb26fa34480d974e`; `v3.0.0-beta` and
  `v3.0.0-beta2` tags are visible as release context.
- Classification:
  source freshness only. No checkout was modified, no container was pulled or
  run, no module was imported, no TacSL sanity ran, and Gate 00F remains open.

### Gate 00F Reference Repository Freshness

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701_v1.json`.
- UniVTAC:
  local and upstream main both at
  `05bcd3edb92237107efa40105292a24f1a9fd761`.
- TaCauchy:
  local and upstream main both at
  `c228cfe9050904cd5d71d64f6eb5104768d4cbda`.
- TacEx:
  local and upstream main both at
  `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`.
- Classification:
  source freshness only. No checkout was modified, no dependency-complete
  runtime was found or registered, no official sanity ran, and Gate 00F
  remains open.

### Gate 00F Bundle Runtime Registry Forwarding

- Date: 2026-07-01.
- Updated:
  `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`.
- Updated:
  `experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`.
- Updated:
  `experiments/configs/phase00/ref_tactile/gate00f_reference_bundle_handoff_v1.json`.
- Updated:
  `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`.
- Updated:
  `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff_v1.json`.
- Updated active records:
  `experiments/reports/phase00/ref_tactile/active_evidence_index.md`,
  `PLAN/00_ref_tactile_env/plan.md`, and
  `TODO/00_ref_tactile_env/todo.md`.
- Behavior:
  the Gate 00F bundle now defines `RUNTIME_REGISTRY` and forwards it to
  runtime preflight and all official sanity sub-scripts. Container-aware
  official sanity runners are wired for UniVTAC, TaCauchy, and IsaacLab TacSL
  through the accepted registry and shared container helper.
- Classification:
  guard/handoff only. No runtime was registered, no container was run, no
  official sanity ran, and Gate 00F remains open.

### Gate 00F Container-Aware Official Sanity Dispatch

- Date: 2026-07-01.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_common.sh`.
- Updated:
  `experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh`.
- Updated:
  `experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh`.
- Behavior:
  UniVTAC, TaCauchy, and IsaacLab TacSL official sanity scripts can now read
  accepted `RUNTIME_REGISTRY` entries and dispatch registered
  docker/singularity/apptainer/sif runtimes through the shared helper.
- Failure recording:
  UniVTAC/TaCauchy schema probe failures and IsaacLab TacSL official demo
  failures now write blocker summaries instead of exiting without evidence.
  TacSL preserves `--use_tactile_rgb`; missing RGB assets must be recorded as
  runtime/asset blockers rather than removed from the command.
- Classification:
  glue only. No runtime was registered, no container was run, no official
  sanity ran, and Gate 00F remains open.

### Gate 00F Container Dispatch Login Refuse

- Date: 2026-07-01.
- Added:
  `experiments/reports/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701.md`.
- Added:
  `experiments/configs/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701_v1.json`.
- Result:
  UniVTAC/TaCauchy sanity, IsaacLab TacSL sanity, and Gate 00F bundle entry
  scripts all exit with code `2` and `ERROR: must run inside a Slurm
  allocation.` when `SLURM_JOB_ID` is missing.
- Classification:
  login-node safety check only. No registry validation, container command,
  module import, official sanity, simulation, rendering, or training ran.
