# Phase 00 Reference-Video Tactile Environment Plan

## Status

Status: active. This is the current starting phase after the 2026-07-01
reference-video reset.

The previous contact-count Phase 00/01 plans are archived under
`PLAN/legacy_20260630_contact_proxy_stopgate/`. They remain useful negative
evidence, but they are no longer the active implementation target.

Current high-signal evidence index:
`experiments/reports/phase00/ref_tactile/active_evidence_index.md`.

Current requirement-status audit:
`experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`.
This audit is not a completion claim; it records which user requirements are
positive, partial, blocked, or not started so later work cannot confuse current
candidate tactile assets with final base-model or curiosity-training success.

Latest source remote refresh:
`experiments/reports/phase00/ref_tactile/latest_source_remote_refresh.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/latest_source_remote_refresh_v1.json`.
This refresh records that Newton upstream `main` has moved to
`d58e70266be0db803261f3e46a2f7d923a43db37`, while the current active evidence
worktree `external/newton_main` remains at
`a217e55fab3d373a08fba374cc5cafc1826cf27f`.

Latest source recheck V3:
`experiments/reports/phase00/ref_tactile/latest_reference_recheck_20260701_v3.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/latest_reference_recheck_20260701_v3.json`.
This recheck records that Newton upstream `main` has advanced again to
`8c501b47847569fecdda97a9f7f01205c6f7964f`; `external/newton_8c501` is now a
source-only worktree at that commit. It has not passed H200 runtime sanity,
dense tactile export, reference comparison, channel audit, or Gate review.
TacEx was cloned at `external/TacEx`, commit
`adceed41afb7cb48f9ec1f66a662fb8e5a06627f`. IsaacLabTactile was acquired at
`external/IsaacLabTactile`, commit
`21bcb476b27ceedccccd63afef6bbd822adc2b2b`, with LFS skipped; it is source
evidence only, not asset-complete or official sanity evidence.

Latest Newton 8c501 compute handoff:
`experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md`
with machine-readable commands in
`experiments/configs/phase00/ref_tactile/newton_8c501_sanity_handoff_v1.json`.
This records the exact tmux-held Slurm command sequence for H200 runtime
benchmark, candidate dense tactile export, reference-video comparison, channel
audit, and Gate review. The runtime benchmark stage has now executed around
80 FPS, which is acceptable for continuing. Downstream 8c501 dense
tactile/reference/Gate stages should proceed when compute is available.

Latest Newton 8c501 allocation request:
`experiments/reports/phase00/ref_tactile/newton_8c501_allocation_request.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/newton_8c501_allocation_request_v1.json`.
Job `160854` was requested as `curiosity_p00_8c501_1gpu_1day` in tmux window
`curiosity_phase00_ref_tactile:alloc_8c501`; initial state was `PENDING`
because of `Priority`.

Latest Newton 8c501 benchmark status:
`experiments/reports/phase00/ref_tactile/newton_8c501_benchmark_status.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/newton_8c501_benchmark_status_v1.json`.
Job `160854` ran two official Panda hydro null-viewer benchmarks on H200:
`80.1 FPS` over `30.01s` and `80.8 FPS` over `60.00s`. Both executed
successfully and are acceptable around 80 FPS. The old `82 FPS` number is a
historical reference only and must not block 8c501 dense tactile export.

Runtime gate correction:
`experiments/reports/phase00/ref_tactile/runtime_gate_correction_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/runtime_gate_correction_20260701_v1.json`.
This correction records the active policy: do not optimize for 82 FPS or use it
as a blocker; around 80 FPS is acceptable, and the next 8c501 step is dense
tactile export/reference comparison/channel audit/Gate review.

Latest Newton 8c501 continuation chain:
`experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`.
This run completed dense tactile export, reference comparison, channel audit,
and Gate review in tmux-held Slurm job `160924`. It is latest-source positive
candidate tactile evidence, not Gate completion. Gate 00D/00E/00F remain open
on official reference sanity and validated photometric/real-area semantics.

Latest Gate 00F readiness refresh:
`experiments/reports/phase00/ref_tactile/gate00f_readiness_refresh_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00f_readiness_refresh_20260701_v1.json`.
It confirms base UniVTAC/TaCauchy env pythons and copied assets are present,
but `gate00f_ready=false`; effective failed checks remain
`univtac_official_reference_sanity` and `tacauchy_official_reference_sanity`.
It also records that IsaacLabTactile source is cloned but LFS asset completeness
is not verified.

Latest Gate 00F tool lookup:
`experiments/reports/phase00/ref_tactile/gate00f_tool_lookup_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00f_tool_lookup_20260701_v1.json`.
PATH exposes no `git-lfs`, `cmake`, `nvcc`, or `nvidia-smi`; project-local
lookup found only `envs/taccel/cuda-toolkit/bin/nvcc`, and no prebuilt
Isaac/Lab/TacEx/UIPC directories were found under `envs` at max depth 4.

Latest Gate 00F static source audit:
`experiments/reports/phase00/ref_tactile/gate00f_static_source_audit_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00f_static_source_audit_20260701_v1.json`.
This audit records the exact official UniVTAC and TaCauchy sanity/schema
entrypoints, confirms the required left/right tactile fields and stress/
pressure/traction semantics, and records that the local IsaacLabTactile clone
is generic Isaac Lab source with no obvious TacSL/GelSight/TacEx entrypoint or
verified LFS asset completeness. It is source audit only, not official sanity
or Gate completion.

Latest Gate 00F module/env probe:
`experiments/reports/phase00/ref_tactile/gate00f_module_env_probe_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00f_module_env_probe_20260701_v1.json`.
The current login shell exposes no `module`/`ml` command, so module-based
lookup for `cmake`, `git-lfs`, CUDA, or Isaac is unavailable from this shell.
Shallow target-env file-name probing found no Isaac/TacEx/UIPC/cuRobo/Torch
component names under the existing UniVTAC/TaCauchy base env prefixes.

Latest Gate 00F container path audit:
`experiments/reports/phase00/ref_tactile/gate00f_container_path_audit_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00f_container_path_audit_20260701_v1.json`.
TacEx/TaCauchy and IsaacLabTactile provide Docker/Singularity build/helper
paths, but no approved prebuilt Curiosity image/SIF/tar artifact was found.
The discovered paths require image builds, NGC/Isaac base image setup, or
placeholder cluster SIF configuration, so they do not clear Gate 00F.

Latest 2026-07-01 web/codebase refresh:
`experiments/reports/phase00/ref_tactile/latest_20260701_web_codebase_refresh.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/latest_20260701_web_codebase_refresh_v1.json`.
This adds official Isaac Lab main TacSL source at `external/IsaacLab_official`
commit `b4c321024792976150ca55fddb26fa34480d974e`, FTP-1 at
`external/ftp1-policy` commit `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`,
and AnyTouch2 at `external/AnyTouch2` commit
`82c5677d9cf0176d97a1fe04745f63cd02dd6f54`. No checkpoint download, model
load, simulation, training, or official sanity was run.

Latest source freshness V4:
`experiments/reports/phase00/ref_tactile/latest_source_freshness_20260701_v4.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/latest_source_freshness_20260701_v4.json`.
This lightweight `git ls-remote` audit confirms the tracked official refs for
Newton, Taccel, T-Rex, IsaacLab, TacEx, TaCauchy, UniVTAC, FTP-1, AnyTouch2,
and HydroShear still match the current source records. It does not modify
checkouts, run official sanity, or clear Gate 00F. Newton decision remains:
`external/newton_8c501` is latest source and runtime around 80 FPS is
acceptable for continuing dense tactile export, while d58 remains the strongest
complete runtime/tactile evidence chain until 8c501 downstream evidence exists.

Latest supplementary codebase audit:
`experiments/reports/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701_v1.json`.
This records `external/TactSim-IsaacLab` at
`4f92257177cd0ee18928de720b880505ec7f7638` as a secondary photometric
GelSight/DIGIT-style IsaacLab tactile reference, `external/newton-actuators`
at `134dacb0912f4b8ce0465ecebf564479f2e62315` as deprecated Newton actuator
background only, and UniT remote HEAD
`52a286520b09708934b25c77aa826360d72c79db` as remote-only future tactile
representation evidence. This audit does not register a runtime, run official
sanity, load a checkpoint, or clear any gate.

Latest Newton worktree status:
`experiments/reports/phase00/ref_tactile/newton_d58_worktree_status.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/newton_d58_worktree_status_v1.json`.
`external/newton_d58` is prepared at upstream `d58e702...`, but official
compute-side sanity has not run yet.

Latest Newton d58 benchmark status:
`experiments/reports/phase00/ref_tactile/newton_d58_benchmark_status.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/newton_d58_benchmark_status_v1.json`.
The hot/longer official hydro benchmark on H200 reached `82.7 FPS`, but this is
runtime sanity only, not dense tactile export or Gate completion.

Latest Gate 00E d58 base evidence audit:
`experiments/reports/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit.md`
with machine-readable status at
`experiments/outputs/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit_summary.json`.
The audit classifies d58 as
`partial_positive_gate00e_base_candidate_tactile_validation_blocked`: it meets
the 82 FPS target, lifts the object, exports steel-spec candidate Fn/Ft tactile
mechanics, and has nonblank reference-comparison/channel-audit assets. It does
not clear Gate 00E because tactile semantics and official reference sanity
remain blocked.

Latest Gate 00D environment evidence audit:
`experiments/reports/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701.md`
with machine-readable status at
`experiments/configs/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701_v1.json`.
The audit classifies d58 as
`partial_positive_environment_candidate_reference_semantics_blocked`: it has
real scene views, candidate Fn/Ft maps, shear arrows, contact-normal overlay,
steel-spec candidate material settings, and reference-comparison assets. Gate
00D remains open because contact area is still proxy-only, dense
penetration/compression semantics are not validated, and official tactile
semantic sanity remains blocked.

## Objective

Build a Newton/Taccel-based tactile simulation environment that matches the
diagnostic richness of `0780e5ec3fdb26b63ae63de0f49f07c4.mp4`, then prepare a
basic grasping base model/controller that can grasp while exporting dense
tactile/mechanics evidence. Curiosity training restarts only after those gates
pass.

## Source-Backed Codebase Decision

Latest source audit on 2026-07-01:

- Newton is the primary physics/runtime target. Official repository:
  `https://github.com/newton-physics/newton`. Latest observed upstream main
  after the source refresh is
  `d58e70266be0db803261f3e46a2f7d923a43db37`; current active evidence worktree
  `external/newton_main` remains at
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`; latest observed stable tag:
  `v1.3.0` at `ce11136b3a28390944f7fe5a32801b31d8aa5670`. Local checkout is
  now behind upstream main and must get fresh compute-side sanity before any
  future "latest Newton main" evidence claim.
- Taccel is the primary tactile simulation reference. Official repository:
  `https://github.com/Taccel-Simulator/Taccel`. Latest observed upstream main:
  `cb23bc251b531ba6908a3788c2f91423cd543149`. Local checkout already matches
  this commit.
- T-Rex is the policy/model reference for high-frequency tactile-reactive
  control, temporal tactile VQ-VAE, and tactile expert design. Official
  repository: `https://github.com/ZhuoyangLiu2005/T-Rex`. Latest observed
  upstream main: `43ff632259d76f08373c085c53111825060d029b`. Local checkout
  has unrelated dirty changes and must not be overwritten silently. Clean
  source snapshots now exist at `external/T-Rex_43ff` for main and
  `external/T-Rex_full_b23` for the `full-pipeline` branch. The official
  released checkpoints are `miniFranka/T-Rex_pretrain_mecka22k_epoch1` and
  `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`; the midtrain
  checkpoint embeds the tactile VQ-VAE and is the future strongest
  tactile-reactive policy starting point after Gate 00F passes and a faithful
  Newton-to-T-Rex data contract exists.
- HydroShear is a reference for hydroelastic shear simulation and tactile
  shear policy training. Official repository:
  `https://github.com/MMintLab/hydroshear`. Latest observed HEAD:
  `a53a51cb74f0608ca53839415d7f1964a99f1db0`.
- IsaacLab TacSL/IsaacLabTactile is a reference path for force-field tactile
  outputs and visuo-tactile sensor integration. Latest observed
  IsaacLabTactile tag: `v2.2.1`.
- Official Isaac Lab main TacSL is now sparsely cloned at
  `external/IsaacLab_official`, commit
  `b4c321024792976150ca55fddb26fa34480d974e`. It provides official
  visuo-tactile data fields (`tactile_depth_image`, `tactile_rgb_image`,
  `penetration_depth`, `tactile_normal_force`, `tactile_shear_force`) and
  force-field config (`normal_contact_stiffness`, `friction_coefficient`,
  `tangential_stiffness`, tactile array size, SDF contact-object path). This is
  a stronger Gate 00F semantic-validation candidate than the generic
  `external/IsaacLabTactile` clone, but still needs official environment/assets
  and compute-side sanity.
- UniVTAC is now a local official visuo-tactile manipulation benchmark
  reference at `external/UniVTAC`, commit
  `05bcd3edb92237107efa40105292a24f1a9fd761`. It provides Isaac Lab/TacEx
  tactile tasks, data collection, ACT/ViTAL baselines, modality ablations, and
  left/right optical tactile fields (`rgb`, `rgb_marker`, `depth`, `marker`).
- TaCauchy is now a local official FEM tactile semantic reference at
  `external/TaCauchy`, commit
  `c228cfe9050904cd5d71d64f6eb5104768d4cbda`. It provides Cauchy stress,
  normal pressure, tangential traction, mesh refinement, force-field
  visualization, and tactile RGB reference semantics.
- FreeTacMan is now a local secondary official real visuo-tactile
  data/pretraining reference at `external/FreeTacMan`, commit
  `9285740a5d33385d3a9cf5ccdb185e3387b547bd`. It provides tactile encoder
  pretraining and ACT tactile-image policy references, but it is not a
  simulator and not a Gate 00F replacement.
- APPLE is now a local secondary official active-perception/curiosity
  reference at `external/APPLE`, commit
  `4b1d71fadb786d865d4ee29a184ab408b9605083`. It provides SAC/CrossQ/PPO,
  random/grid exploration baselines, ViT tactile-image configurations, and
  sequence/memory model references. It is not a Newton-native grasping infant
  checkpoint and not Gate 00D/00E/00F evidence.
- Tactile MNIST is now a local secondary official active tactile benchmark
  reference at `external/tactile-mnist`, commit
  `9e4e59139e9349ab361a3b9297f4815724ad6387`. It provides GelSight Mini active
  tactile perception environments, tactile-only observation/action schemas,
  train/test/holdout mesh splits, and real/synthetic tactile image datasets.
  It is not a grasp/lift/hold base controller and not a Gate 00F replacement.
- DiffTactile is now a local secondary differentiable tactile simulator
  reference at `external/DiffTactile`, commit
  `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`. It provides FEM tactile sensor
  models, marker extraction, soft/rigid/multi-material object models,
  contact-rich manipulation tasks, and CMA-ES/PPO/SAC/RNN baselines. Its
  `requirements.txt` is UTF-16 little-endian text, so it must not be blindly
  passed to a normal installer without encoding-aware review.
- Reactive Diffusion Policy is now a local secondary serious visual-tactile
  policy reference at `external/reactive_diffusion_policy`, commit
  `824c5e8de1fd1811106907a04b5f0186e0138c0b`. It provides tactile marker
  embedding, real visual-tactile datasets, diffusion/RDP training scripts, and
  official dataset/checkpoint links. It is not a Newton-native infant
  checkpoint.
- ImplicitRDP is now a local secondary visual-force diffusion policy reference
  at `external/ImplicitRDP`, commit
  `4c90646df17787e31c88838106c4a0323ddefb4a`. It provides visual-force
  diffusion policy configs, force RNN components, and official dataset/
  checkpoint links. It is not dense tactile semantic validation.
- Tactile Diffusion is now a local secondary photometric tactile-image
  generation reference at `external/Tactile-Diffusion`, commit
  `16868fb96d19d93dc5837600c26b48415632e4f6`. It is useful for the future
  gel/marker photometric gap but not for mechanics validation.
- FTP-1 is now a local future policy/checkpoint reference at
  `external/ftp1-policy`, commit
  `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`. It provides a 2026 generalist
  foundation tactile policy codebase with published pretrain and
  UniVTAC-finetune checkpoint links, but it is not a Newton-native infant
  checkpoint and cannot be used before checkpoint/environment/schema sanity.
- AnyTouch2 is now a local future tactile representation reference at
  `external/AnyTouch2`, commit
  `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`. It provides an ICLR 2026 optical
  tactile representation codebase with checkpoint links, but it is not a
  controller and not Gate 00F official simulation sanity.
- Tacmap and ControlTac remain code-unavailable comparison gaps after web and
  common GitHub remote probes.

The mainline is Newton plus Taccel. HydroShear and IsaacLab tactile are
comparison/reference sources unless Newton/Taccel reach a recorded blocker.
UniVTAC and TaCauchy are mandatory semantic-validation references for the
current Gate 00D/00E blockers, not silent replacements for the Newton base.
APPLE and Tactile MNIST are Gate 00G curiosity-design references only.
Reactive Diffusion Policy, ImplicitRDP, and Tactile Diffusion are future
comparison/design references only until official environment, checkpoint, and
schema sanity are proven. FTP-1 and AnyTouch2 join this future serious-method
pool for policy/encoder comparison after dense tactile/base gates pass.

## Required Tactile Environment Outputs

Each valid Phase 00 environment run must export synchronized evidence for:

- visual scene frames;
- left and right tactile pad fields;
- pressure or compression heatmaps;
- `Fn` normal force per pad;
- `Ft` tangential/shear force per pad;
- shear direction vectors;
- contact area and center/proxy;
- mean/max penetration or compression;
- material labels and physical parameters;
- rigid steel/metal stress/contact response;
- friction and tangential force statistics;
- time-series plots for grip force, shear, contact area, penetration, and
  compliance;
- MP4 rollout video plus dense diagnostic contact sheet;
- source arrays under explicit namespaces such as `newton.*`, `taccel.*`, and
  `candidate.*`.

Scalar contact count is allowed only as an auxiliary summary, not as tactile.

## Phase Gates

### Gate 00A: Codebase Audit

Record local and upstream commits, dirty state, available examples, required
dependencies, and whether official demos can be run later inside a compute
allocation. No simulation is run on the login node.

### Gate 00B: Environment Build Spec

Write the steel/metal tactile scene spec:

- Panda or selected robot;
- two tactile pads/fingers;
- steel/metal rigid object first;
- contact normal, friction, stress/pressure/compression, and shear output
  fields;
- target frame rate and decimation plan;
- expected output schema.

### Gate 00C: Official Sanity In Compute Allocation

Run official Newton and Taccel sanity/examples in a Curiosity-owned tmux-held
Slurm allocation using prebuilt local environments only. Record command, host,
job ID, exact commit, expected output, observed output, and pass/fail.

### Gate 00D: Reference-Style Diagnostic Render

Generate one reference-style diagnostic video/contact sheet for rigid metal:
visual scene, tactile maps, force/shear plots, and material statistics. This is
environment evidence only, not training.

Current status on 2026-07-01: partial positive. Official Newton Panda hydro now
has a USD rollout plus hydro-derived left/right tactile map videos and NPZ
source arrays under `phase00/ref_tactile/newton_hydro/`. The evidence is not a
Gate 00D pass yet because the scene rollout and tactile maps are not fused into
one synchronized diagnostic video, `Fn`/`Ft`/shear fields are not complete, and
the enhanced tactile sheet shows sparse lower-pad contact rather than
reference-video-level tactile richness.

Update on 2026-07-01: `p00_sync_hydro_20260701_025818` now provides a
synchronized diagnostic AVI with scene schematic, left/right tactile maps,
object-z, contact area, `hydro_proxy.Fn`, and `hydro_proxy.shear_motion`.
Gate 00D still remains open because the scene panel is a schematic from
Newton `body_q`, not a USD/photoreal render, and `Ft` plus pad-resolved shear
vectors are proxy-only.

Further update on 2026-07-01: `p00_base_mech_20260701_030544` extends the
synchronized diagnostic with `hydro_proxy.stress`, force-weighted contact
normal, tangential-capacity proxy, force balance, lift/hold/drop/slip/safety
metrics, and synchronized stress/Ft-capacity curve panels. It is stronger
base-mechanics evidence, but Gate 00D still remains open because tactile maps
are sparse/lower-edge concentrated, the scene panel is still schematic,
direct solver `Ft` and pad-resolved shear vectors are missing, and observed
official material parameters do not match the steel-first spec.

Direct-force probe update on 2026-07-01: `p00_force_probe_20260701_032310`
attempted to request `Contacts.force` and call `SolverMuJoCo.update_contacts()`
after each official Panda hydro step. The run failed with CUDA illegal memory
access and no valid direct-force arrays. Direct `Ft` is therefore a recorded
blocker for the current official Panda hydro Newton-contacts path; use only
explicitly labeled `hydro_proxy.*` force/shear fields until a faithful official
force-export path is found.

Steel-spec update on 2026-07-01: `p00_steel_v1_20260701_032709` applies the
candidate steel material override `mu=0.3`, `kh=1e12` without modifying the
official Newton repository. The run verifies observed arrays
`shape_material_mu=[0.30000001192092896]` and
`shape_material_kh=[999999995904.0]`, exports the synchronized mechanics/tactile
video and source arrays, and preserves lift/hold success. Gate 00D remains open
because the fields are still `hydro_proxy.*`, tactile maps remain sparse, and
the scene panel is schematic rather than USD/photoreal.

Grid-tactile update on 2026-07-01: `p00_grid_v1_20260701_033556` adds
HydroShear-style Gaussian grid fields for left/right `Fn`, stress,
deformation, shear-vector, and shear-magnitude maps, while preserving explicit
`hydro_proxy.*` provenance. It verifies nonzero grid `Fn` and shear fields under
the same steel-spec candidate material and exports a synchronized six-panel
tactile video/contact sheet. Gate 00D still remains open because direct solver
`Ft`, direct pad-resolved shear force, USD/photoreal scene fusion, and
reference-video-level tactile density are not solved yet.

F6 proxy update on 2026-07-01: `p00_f6_v1_20260701_034033` adds T-Rex-aligned
per-pad F6 proxy arrays (`normal`, `Ft_capacity`, and `combined` wrench
proxies) to the Newton hydro export. It preserves steel-spec material evidence
and lift/hold success while producing nonzero left/right F6 proxy norms. This
does not mean T-Rex checkpoint compatibility or direct tactile force success;
it is a schema bridge only.

Direct-force comparison update on 2026-07-01: `p00_mjc_sensor_v1_20260701_034541`
confirms that Newton's official `SensorContact` path can produce nonzero direct
force/friction on a related Panda MuJoCo-contact variant with
`SolverMuJoCo(use_mujoco_contacts=True)`. The variant also lifts the object, but
it does not use the Newton hydro collision pipeline and therefore cannot
replace the active hydro tactile base or close Gate 00D.

### Gate 00E: Base Grasp Controller/Model

Prepare the basic "infant" base controller/model that can grasp/lift/hold in
the dense tactile environment and exports the same tactile/mechanics evidence.
This may start from the official Newton Panda hydro prior or another serious
existing controller. It must not be a toy model represented as a base policy.

Current status on 2026-07-01: selected base is official Newton
`newton.examples.robot.example_robot_panda_hydro`. It has passed the official
lift test on H200 and produced positive tactile export evidence with max object
lift `0.22351960837841034` m. Gate 00E remains open until hold/slip/drop/safety
metrics and complete force/shear tactile fields are exported.

Update on 2026-07-01: synchronized diagnostic `p00_sync_hydro_20260701_025818`
raised the base evidence to max object lift `0.2235533893108368` m, max contact
area `0.0034376755356788635` m^2, max `hydro_proxy.Fn` `2931.3955078125`, and
max `hydro_proxy.shear_motion` `0.03598056361079216` m. These are still
hydro-derived proxies and do not close the base gate without direct `Ft`,
pad-resolved shear vectors, and hold/slip/drop/safety metrics.

Further update on 2026-07-01: `p00_base_mech_20260701_030544` adds hold/drop/
slip/safety metrics. It shows lift success over `0.15` m, first lift at frame
`169`, `71` hold frames above threshold, no detected post-lift drop, max object
lift `0.22364932298660278` m, max `hydro_proxy.Fn` `2936.611083984375`, max
stress proxy `1489732.5`, max tangential-capacity proxy `2936.611083984375`,
and max object acceleration `1.310336709022522` m/s^2. It also reveals a
calibration gap: observed official `shape_material_mu` is `[1.0]` and observed
`shape_material_kh` is `[10000000000.0, 99999997952.0]`, not the steel-first
spec target `mu=0.3`, `kh=1e12`.

### Gate 00F: Official Tactile Semantic Validation

Validate the current candidate Newton tactile channels against official tactile
reference semantics before restarting curiosity. This gate is open until:

- UniVTAC official schema/sanity is checked in a compute allocation or a
  concrete environment blocker is recorded;
- TaCauchy official schema/sanity is checked in a compute allocation or a
  concrete environment blocker is recorded;
- candidate `Fn`, `Ft`, marker/deformation, contact-normal, and area fields are
  mapped to official reference semantics without renaming proxies into official
  keys;
- Gate 00D/00E review consumes this mapping and either passes it or records a
  faithful blocker.

Current status on 2026-07-01: source/document audit complete only. Local
official checkouts exist for UniVTAC and TaCauchy, and the reference matrix is
recorded in
`experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`.
No official UniVTAC or TaCauchy compute-side sanity has been run yet, so
Gate 00F remains open and curiosity training remains disallowed.

Update on 2026-07-01: official reference sanity/blocker probes were launched in
Curiosity held Slurm job `160450` on `server02` using
`run_tactile_reference_sanity_in_alloc.sh`. UniVTAC and TaCauchy both matched
their expected official commits, but both returned
`blocked_missing_prebuilt_environment` because no approved prebuilt
`envs/univtac/conda`, `envs/univtac/.venv`, `envs/tacauchy/conda`,
`envs/tacauchy/.venv`, `UNIVTAC_PYTHON`, or `TACAUCHY_PYTHON` exists.
`p00_gate_review_v4_20260701_055100` now consumes
those blocker summaries and keeps Gate 00F
`open_official_semantic_validation_blocked`. The next faithful action is to
prepare or locate approved local shared-filesystem environments for official
UniVTAC/TaCauchy sanity, not to run installs on compute nodes and not to start
curiosity training.

Requirement-status update on 2026-07-01:
`phase00_requirement_status_v1.json` records the current state across the
standing user requirements. It classifies the Newton main steel-spec MJWarp
direct-force candidate and 92.6 FPS official Panda hydro base as partial
positive evidence, keeps official semantic validation blocked by missing
UniVTAC/TaCauchy prebuilt environments, and explicitly leaves curiosity
training disallowed until Gate 00D/00E/00F pass or a faithful blocker is
accepted.

Environment-location update on 2026-07-01:
`reference_env_location_audit_v1.json` and
`reference_env_location_audit.md` record a lightweight login-node search for
existing reference environments. No approved UniVTAC/TaCauchy target Python was
found in project `envs/` or common home conda/env locations; existing
Newton/Taccel/T-Rex/autoresearch venvs are not acceptable substitutes. Gate 00F
therefore remains blocked by environment availability. Project-local
`envs/taccel/miniforge/bin/conda` exists and can be treated as an env-creator
candidate, but the target environments do not exist and heavy construction has
not started.

Environment-staging update on 2026-07-01:
`reference_env_stage_checklist_v1.json` and
`reference_env_stage_checklist.md` split the official UniVTAC/TaCauchy
environment work into audited stages. UniVTAC requires the modified bundled
TacEx path with Isaac Sim 4.5 / Isaac Lab 2.1.1, while TaCauchy requires Isaac
Sim 5.0 / Isaac Lab 2.2.1, UIPC/libuipc, large tactile assets, and stress/
traction visualization outputs. The checklist explicitly forbids running the
official all-in-one installers or dependency builds as untracked shortcuts.

Asset-availability update on 2026-07-01:
`reference_asset_availability_v1.json` and
`reference_asset_availability.md` record that UniVTAC bundled TacEx contains
useful GelSight/GF225/shape assets, but TaCauchy has only partial placeholder
assets and is missing required full sensor USD/calibration, valid Franka UIPC
assets, and tactile test shapes. Gate 00F therefore remains blocked by both
missing target environments and incomplete TaCauchy assets.

Asset-reuse planning update on 2026-07-01:
`reference_asset_reuse_plan_v1.json` and
`reference_asset_reuse_plan.md` record a candidate local asset reuse path from
the UniVTAC bundled TacEx asset tree into TaCauchy. This was not executed. It
is only a future option if exact asset provenance is accepted or if the
official Git LFS asset path remains blocked.

Asset-stage guard update on 2026-07-01:
`prepare_reference_asset_stage.sh` now records dry-run asset commands for
`audit`, `reuse_copy`, and `verify` without copying files. Current dry-run
status is audit ready, reuse-copy not executed, and verify blocked because
TaCauchy still lacks `Sensors/GelSight_Mini/Sensor.usd`.

Gate-review asset enforcement update on 2026-07-01:
`phase00_gate_review.py` and its Slurm/tmux runners now accept the asset
availability audit and asset reuse plan as inputs. Future Gate 00F reviews must
include `reference_asset_availability` and `reference_asset_reuse_plan_available`
checks so missing TaCauchy assets cannot be hidden behind environment-only
blockers. The updated code has passed syntax checks only; it has not been
rerun in compute after this change.

Latest-source refresh update on 2026-07-01:
`latest_source_remote_refresh_v1.json` records that Newton upstream main has
moved from the active evidence commit `a217e55...` to `d58e702...`. Taccel,
UniVTAC, TaCauchy, HydroShear, FreeTacMan, DiffTactile, APPLE, Tactile MNIST,
Reactive Diffusion Policy, ImplicitRDP, and Tactile Diffusion local checkouts
match their observed remote HEADs. T-Rex remains behind remote main and has
existing dirty state that must not be overwritten silently. IsaacLabTactile
probe at `yanglh14/IsaacLabTactile` returned repository not found.

Latest-source recheck V3 update on 2026-07-01:
`latest_reference_recheck_20260701_v3.json` supersedes the stale
IsaacLabTactile URL probe and records the current source truth. Newton upstream
main is now `8c501b47847569fecdda97a9f7f01205c6f7964f`; a detached source-only
worktree exists at `external/newton_8c501`. Taccel official source is
`https://github.com/Taccel-Simulator/Taccel.git`; TaCauchy official source is
`https://github.com/figsama/TaCauchy.git`; HydroShear local official source is
`https://github.com/MMintLab/hydroshear.git`; IsaacLabTactile official source is
`https://github.com/UM-ARM-Lab/IsaacLabTactile.git`; TacEx official source is
`https://github.com/DH-Ng/TacEx.git`. `external/TacEx` is cloned at
`adceed41afb7cb48f9ec1f66a662fb8e5a06627f`. `external/IsaacLabTactile` is
cloned at `21bcb476b27ceedccccd63afef6bbd822adc2b2b` using
`GIT_LFS_SKIP_SMUDGE=1` and blob filtering; `git-lfs` is unavailable on the
current PATH, so LFS asset completeness is not verified.
No dependency installation, official sanity, rendering, or training was run for
this recheck, so it changes source readiness only and does not close any gate.

Newton 8c501 handoff update on 2026-07-01:
`newton_8c501_sanity_handoff_v1.json` defined the faithful compute-side
sequence: runtime benchmark on `external/newton_8c501`, then dense tactile
export, reference compare, channel audit, and Gate review. The runtime stage
has now been executed around 80 FPS, so the downstream 8c501 dense
tactile/reference/Gate stages should proceed when compute is available.

Newton 8c501 allocation update on 2026-07-01:
`newton_8c501_allocation_request_v1.json` records tmux-held H200 allocation
request job `160854` for the 8c501 sanity path. It is allocation evidence only,
not benchmark, tactile, Gate, or training evidence.

Newton 8c501 benchmark update on 2026-07-01:
`newton_8c501_benchmark_status_v1.json` records two successful executions on
H200 around 80 FPS: `80.1 FPS` and `80.8 FPS`. The old `82 FPS` number is a
historical reference only, so this is not a runtime blocker. Launch 8c501 dense
tactile export when a Curiosity tmux-held Slurm allocation is available.

Gate 00F readiness refresh on 2026-07-01:
`gate00f_readiness_refresh_20260701_v1.json` records the latest lightweight
readiness state. File-level env and asset availability are positive; official
UniVTAC/TaCauchy sanity is still missing. Curiosity training remains
disallowed.

Gate 00F tool lookup on 2026-07-01:
`gate00f_tool_lookup_20260701_v1.json` records that no existing `git-lfs` or
`cmake` executable was found in PATH or project-local lookup. This supports the
current dependency-readiness blocker; no dependencies were installed.

Gate 00F static source audit on 2026-07-01:
`gate00f_static_source_audit_20260701_v1.json` records the source-level
requirements for the remaining official semantic validation. UniVTAC provides
the left/right GelSight Mini tactile schema (`rgb_marker`, `marker`, `depth`,
`rgb`, and `pose`) and manipulation benchmark baselines, but its official path
requires dependency-complete Isaac Lab/TacEx/curobo readiness and sanity.
TaCauchy provides the Cauchy-stress, normal-pressure, tangential-traction, and
GelSight optical/marker semantic reference, but requires Isaac Sim 5.0,
Isaac Lab 2.2.1, CMake/vcpkg/UIPC readiness, assets, and official demo sanity.
The local IsaacLabTactile clone is not a Gate 00F replacement because static
audit found generic Isaac Lab/contact-sensor source rather than an asset-
complete TacSL/GelSight/TacEx tactile entrypoint.

Gate 00F module/env probe on 2026-07-01:
`gate00f_module_env_probe_20260701_v1.json` records that the current shell has
no `module`/`ml` command and that shallow file-name inspection of
`envs/univtac/conda` and `envs/tacauchy/conda` found no Isaac, TacEx, UIPC,
cuRobo, or Torch component names. This reinforces the dependency-complete
environment blocker and does not run imports or official sanity.

Gate 00F container path audit on 2026-07-01:
`gate00f_container_path_audit_20260701_v1.json` records that `docker` exists on
PATH, while `singularity`, `apptainer`, `enroot`, and `podman` do not. The
official TacEx/TaCauchy Docker paths are build recipes requiring an Isaac Lab
base image, build tools, CMake, vcpkg, CUDA, and container setup. The
IsaacLabTactile Singularity helper expects a pre-existing tarred SIF and
configured cluster paths, but the local config is placeholder-only and no
project-local SIF/tar artifact was found.

Latest codebase refresh on 2026-07-01:
`latest_20260701_web_codebase_refresh_v1.json` adds official IsaacLab TacSL,
FTP-1, and AnyTouch2 to the active source-backed reference set. The Gate 00F
semantic bridge now includes official IsaacLab TacSL fields for tactile RGB,
depth, penetration, normal force, and shear force. The next faithful TacSL
action is not to run the demo immediately; it is to locate or provide an
approved dependency-complete IsaacLab/TacSL environment or prebuilt container,
then run `scripts/demos/sensors/tacsl_sensor.py` inside Curiosity tmux-held
Slurm.

Latest policy/checkpoint refresh on 2026-07-01:
`latest_policy_checkpoint_refresh_20260701_v1.json` and
`latest_policy_checkpoint_refresh_20260701.md` record the current serious
policy and representation checkpoint landscape. Clean T-Rex source snapshots
now exist at `external/T-Rex_43ff` (`main`) and `external/T-Rex_full_b23`
(`full-pipeline`), preserving the older dirty `external/T-Rex` checkout.
Official T-Rex released checkpoints are
`miniFranka/T-Rex_pretrain_mecka22k_epoch1` and
`miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`; the midtrain
checkpoint embeds the tactile VQ-VAE and should be the future strongest
tactile-reactive policy starting point. FTP-1 provides a future 4B general
tactile policy checkpoint, while AnyTouch2 and Sparsh provide future tactile
representation/force-field baselines. None of these replace Gate 00F: they can
only be used after dense tactile semantic validation and a faithful data
contract exist.

Post-Gate 00F policy bridge checklist on 2026-07-01:
`post_gate00f_policy_bridge_checklist_v1.json` and
`post_gate00f_policy_bridge_checklist.md` define the exact preconditions before
future T-Rex/FTP-1/AnyTouch2/Sparsh work. T-Rex promotion requires slow/fast
RGB streams, eef-62 compatibility or a validated adapter, high-frequency
hand/finger F6, deformation/tactile image alignment, timing metadata,
leak-free splits, and normalization compatibility. Required ablations are
vision+tactile, tactile-only masked vision, vision-only, and noisy/mismatched
tactile. This checklist is planning only and does not authorize checkpoint
loading or training before Gate 00D/00E/00F close.

T-Rex data-contract extraction on 2026-07-01:
`trex_data_contract_v1.json`, `trex_data_contract.md`, and
`src/newton_tactile_curiosity/trex_contract_validate.py` extract the concrete
official T-Rex metadata gate from `external/T-Rex_43ff`: head/wrist videos,
`observation.state [62]`, `action [16,62]`, `action_abs [62]`,
`observation.tactile_f6 [10,6]`, ten tactile-deform video streams, and
normalization stats for action/state/tactile_f6. This is a schema guard only;
it confirms current Newton Panda evidence cannot be renamed into T-Rex data.

IsaacLab TacSL sanity handoff on 2026-07-01:
`isaaclab_tacsl_sanity_handoff_v1.json` and
`isaaclab_tacsl_sanity_handoff.md` define the official TacSL sanity command and
tmux-held Slurm launch scripts. The handoff is not run. It remains blocked by
the absence of an approved dependency-complete IsaacLab/TacSL environment or
prebuilt Curiosity-owned container.

Gate 00F TacSL review wiring on 2026-07-01:
`src/newton_tactile_curiosity/phase00_gate_review.py` now treats official
IsaacLab TacSL as a real Gate 00F condition, not only a planning note. Future
gate reviews require `OfficialIsaacLabTacSL` in the semantic matrix, require
`candidate.newton_mjw.penetration_or_compression` in the bridge spec, and keep
Gate 00F blocked unless an IsaacLab TacSL sanity summary reports
`pass_official_isaaclab_tacsl_demo_exited_zero`. The allocation launcher now
accepts `ISAACLAB_TACSL_SANITY_SUMMARY` so a compute-side official TacSL sanity
can be consumed without changing the gate script.

TacSL env/container blocker refresh on 2026-07-01:
`isaaclab_tacsl_env_blocker_refresh_20260701_v1.json` and
`isaaclab_tacsl_env_blocker_refresh_20260701.md` record that the currently
running Slurm job `160860` belongs to the Reflex exclusion zone
(`WorkDir=/public/home/yanhongru/ICLR2027/Reflex`) and cannot be reused. No
`envs/isaaclab_tacsl` prefix and no Curiosity-local TacSL/Isaac/TacEx prebuilt
container archive were found in the limited checks. Gate 00F therefore remains
blocked until a Curiosity-owned dependency-complete IsaacLab/TacSL env or
prebuilt container exists.

Unified Gate 00F reference bundle handoff on 2026-07-01:
`gate00f_reference_bundle_handoff_v1.json` and
`gate00f_reference_bundle_handoff.md` define a single Curiosity allocation
workflow that runs UniVTAC sanity, TaCauchy sanity, IsaacLab TacSL sanity, and
then Gate review with fixed summary paths. The launcher
`launch_gate00f_reference_bundle_tmux.sh` refuses non-Curiosity allocations by
checking Slurm workdir. The safety check
`gate00f_bundle_launcher_reflex_refuse_check_20260701.md` confirms it refused
job `160860` because the workdir is `/public/home/yanhongru/ICLR2027/Reflex`.
The bundle forwards `RUNTIME_REGISTRY` into runtime preflight and all official
sanity sub-scripts, and those runners can dispatch registered docker/
singularity/apptainer/sif runtimes through the shared container helper. This
does not clear Gate 00F; it only makes the future official sanity path less
error-prone once valid envs/containers and a Curiosity-owned allocation exist.

Gate 00F bundle acceptance checker on 2026-07-01:
`gate00f_bundle_acceptance_handoff_v1.json`,
`gate00f_bundle_acceptance_handoff.md`, and
`src/newton_tactile_curiosity/gate00f_bundle_acceptance.py` define the strict
post-run acceptance gate. A bundle is accepted only if UniVTAC and TaCauchy
return `pass_official_schema_probe`, IsaacLab TacSL returns
`pass_official_isaaclab_tacsl_demo_exited_zero`, the Gate review returns
`pass_official_semantic_reference_sanity`, blocker sanity is disabled, and the
Gate review has no failed checks or hard blockers. This prevents a completed
bundle or blocker-only run from being mistaken for Gate 00F completion.

Latest Newton worktree update on 2026-07-01:
`external/newton_d58` was added as a detached worktree at
`d58e70266be0db803261f3e46a2f7d923a43db37`, preserving `external/newton_main`
at the existing active evidence commit. Runtime and candidate tactile evidence
now exist, but reference-video comparison and Gate review still decide whether
it can replace the older active evidence chain.

Newton d58 allocation update on 2026-07-01:
`newton_d58_allocation_request_v1.json` records a Curiosity tmux-held H200
allocation request for the d58 sanity path. Job `160467` was granted on
`server02` under tmux window `curiosity_phase00_ref_tactile:alloc_d58` and was
reused for benchmark, tactile export, and reference-video comparison work.

Newton d58 benchmark update on 2026-07-01:
`p00_bench_d58_v1_20260701_070459` passed execution but measured only
`70.8 FPS`. The follow-up hot/longer run
`p00_bench_d58_hot_v1_20260701_070611` measured `82.7 FPS` over `2482` frames
in `30.01` seconds on H200, meeting the 82 FPS target for the latest upstream
Newton worktree. This upgrades d58 to runtime-positive evidence only. It still
needs dense tactile export, reference diagnostic video, and Gate review before
it can replace the active d58-pending base evidence.

Newton d58 tactile export update on 2026-07-01:
`p00_mjw_d58_marker_v1_20260701_071248` produced candidate direct-force
dense tactile/mechanics evidence on latest upstream Newton `d58e702...`.
Evidence:
`experiments/reports/phase00/ref_tactile/newton_d58_tactile_export_status.md`,
`experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_summary.json`,
`experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile.avi`,
`experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_sheet.jpg`, and
`experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_timeseries.npz`.
Manual inspection shows nonblank scene-plus-tactile panels and tactile response
during grasp/lift/hold. This upgrades d58 from runtime-only to candidate
tactile evidence, but `direct_tactile_claim_allowed=false`; reference-video
comparison, validated tactile semantics, and Gate review remain open.

Newton d58 reference/Gate update on 2026-07-01:
`p00_refcmp_d58_marker_v1_20260701_071521` produced reference-vs-candidate
comparison assets and passed `pass_reference_comparison_assets`.
`p00_chan_d58_marker_v1_20260701_071757` passed layout/channel audit with no
failed checks. `p00_gate_d58_marker_v1_20260701_071843` then consumed the d58
benchmark, tactile export, reference comparison, channel audit, semantic matrix,
bridge spec, env availability, asset availability, and official sanity
summaries. Gate result remains `open_not_curiosity_ready`: 12 checks passed,
but `reference_env_availability`, `reference_asset_availability`,
`univtac_official_reference_sanity`, and `tacauchy_official_reference_sanity`
failed. Gate 00D/00E/00F are still open, and curiosity training remains
disallowed.

TaCauchy asset blocker audit on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/tacauchy_asset_blocker_audit.md`
records that official `external/TaCauchy/scripts/setup_assets.sh` requires
`git-lfs` to sparse-checkout and LFS-pull TacEx assets, but current login PATH
does not expose `git-lfs`. Local TaCauchy target assets are only `1.8M` and
lack `Sensors/GelSight_Mini/Sensor.usd` and tactile test shapes. Local UniVTAC
bundled TacEx assets are `410M` and include GelSight Mini USDs plus 21 test
shape USDs, but copying them into `external/TaCauchy` is a material official
repo mutation and is not executed without explicit approval or a cleaner
official asset path.

Reference environment blocker audit on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/reference_env_blocker_audit.md`
now records that `envs/univtac/conda/bin/python` and
`envs/tacauchy/conda/bin/python` are present as base Python env prefixes.
Official dependency installation and official sanity are still blocked.
Current PATH lacks `module`, `cmake`, `git-lfs`, `nvcc`, and `nvidia-smi`;
project-local CUDA/conda tools exist but do not by themselves satisfy official
UniVTAC/TaCauchy sanity.

Gate 00F decision packet on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/gate00f_decision_packet.md`
records the exact approval boundary. The project-local CUDA toolkit contains
`envs/taccel/cuda-toolkit/bin/nvcc` (`12.8`), but `git-lfs`, executable
`cmake`, and the target UniVTAC/TaCauchy env pythons are still missing.
Non-Curiosity OmniWorld/ICLR2027 environment/resource hits were observed but
not inspected or reused. The two approval choices are: official `git-lfs`/asset
bundle vs. approved UniVTAC-to-TaCauchy asset copy, and approved prebuilt envs
vs. controlled local env construction.

Runtime update on 2026-07-01: official Newton null-viewer benchmark
`p00_hydro_bench_20260701_030813` measured `67.5 FPS` on H200 for
`newton.examples.robot.example_robot_panda_hydro --scene cube --world-count 1`.
This meets the current `60 FPS` minimum but does not meet the user's `82 FPS`
target. Gate 00E remains open until the 82 FPS target is either met by a
faithful optimized configuration or recorded as a concrete performance blocker.

Hot-cache runtime update on 2026-07-01:
`p00_bench_hot_20260701_034952` reran the same official Newton Panda hydro
null-viewer benchmark for `30` seconds after kernels were warm and measured
`79.2 FPS` (`2377` frames in `30.00` s). This is much closer to the user's
`82 FPS` target, but still does not meet it. Gate 00E remains open.

Longer runtime update on 2026-07-01:
`p00_bench_60_20260701_035208` reran the same official benchmark for `60`
seconds and measured `79.1 FPS` (`4749` frames in `60.01` s). The current
faithful official hydro base appears stable around `79 FPS` on this H200
allocation, below the `82 FPS` target.

Latest-main runtime update on 2026-07-01:
`external/newton_main` was created as an independent official upstream-main
worktree at `a217e55fab3d373a08fba374cc5cafc1826cf27f`. The official Panda
hydro null-viewer benchmark `p00_bench_main_20260701_035529` measured
`92.6 FPS` (`2597` frames in `28.04` s), meeting the user's `82 FPS` target.
The follow-up synchronized diagnostic `p00_main_f6_v1_20260701_035926` also
passed on the same main worktree with steel-spec material override, grid tactile
maps, F6 proxy arrays, lift/hold metrics, AVI, and contact sheet. Gate 00E is
now performance-positive on latest main, but still open overall because direct
hydro `Ft`, reference-grade tactile density, and USD/photoreal fusion remain
unsolved.

Calibrated-view tactile update on 2026-07-01:
`p00_calib_view_v1_20260701_040715` adds calibrated tactile visualization maps
on latest Newton main. It keeps the raw grid/F6 arrays and adds
`*_calibrated_view_*` arrays whose tactile plane uses the rollout's 1%-99%
contact local-yz window. This improves the visible Fn-map nonzero cell ratio
from `0.03515625` raw to `0.236328125` calibrated for both pads, while
preserving lift/hold success and steel-spec material evidence. Gate 00D remains
open because this is a visualization-window correction, not direct hydro `Ft`,
not photoreal scene fusion, and not validated gel/marker tactile rendering.

Official scene/USD update on 2026-07-01:
`p00_main_usd_v1_20260701_041900` exports the official Newton main Panda hydro
cube rollout through `ViewerUSD` from commit
`a217e55fab3d373a08fba374cc5cafc1826cf27f`. The 6.9 MB USD verifies that the
latest-main base can still produce official scene/geometry/rollout evidence
after the runtime and calibrated tactile updates. Gate 00D remains open because
this is not yet rasterized/fused with the dense tactile diagnostic and it does
not add direct `Ft`, pad-resolved shear force, or gel/marker tactile rendering.

Scene-frame probe update on 2026-07-01:
`p00_usd_probe_v2_20260701_042430` shows that the exported USD stage opens
correctly (`220` prims, time codes `0-239` at `60 Hz`) but current environment
does not provide `usdrecord`, `usdview`, `usdcat`, `ffmpeg`,
`pxr.UsdAppUtils`, or `pxr.UsdImagingGL`. The direct USD raster path is
therefore blocked in the current prebuilt environment. As a faithful fallback
within official Newton, `p00_scene_cam_v3_20260701_043330` uses
`SensorTiledCamera` on the latest-main Panda hydro model to render real
head/right-wrist/left-wrist scene frames. The probe is nonblank
(`pixel_std=97.0467646595397`), exports an AVI and contact sheet, and records
object lift `0.19237708300352097` m. Gate 00D remains open until these real
scene frames are fused into the calibrated tactile/mechanics diagnostic and
direct `Ft` or a faithful force path is available.

Fused scene+tactile update on 2026-07-01:
`p00_fused_cam_v1_20260701_043900` fuses official Newton main
`SensorTiledCamera` head/right-wrist/left-wrist scene frames with calibrated
left/right `Fn`, shear-vector, deformation tactile panels and mechanics curves
in one synchronized AVI/contact sheet. The run preserves steel-spec material
evidence (`mu=0.3`, `kh=1e12`), lift success, and calibrated tactile density:
scene camera nonblank `true`, `scene_camera_pixel_std=96.05477790619898`,
max object lift `0.22351396083831787` m, max `hydro_proxy.Fn`
`22550.27734375`, and calibrated Fn nonzero cell ratio `0.2470703125` for both
pads. This is a major Gate 00D visual-fusion improvement over the schematic
scene panel. Gate 00D remains open only on the remaining physics/sensor gaps:
direct solver `Ft`, direct pad-resolved shear force, and validated gel/marker
tactile rendering comparable to the reference video.

MJWarp direct-force array audit update on 2026-07-01:
`p00_mjw_force_audit_v1_20260701_045000` safely read official
`SolverMuJoCo.mjw_data` contact and EFC force arrays for 90 frames without
calling the previously crashing `SolverMuJoCo.update_contacts()` path. It
proved nonzero bottom-level EFC normal/tangent force arrays exist, but the
short horizon only covered nonzero hand/world contacts and showed no
pad-object force. The longer audit `p00_mjw_force_audit_v2_20260701_045700`
ran 240 frames on the same official Newton main Panda hydro base and observed
pad-object force for `128` frames, max pad-object EFC abs sum
`253.05938720703125`, max pad-object tangent EFC abs sum
`141.14100646972656`, max `nacon=107`, and zero read errors. This is positive
force-path evidence: direct `Ft` is likely blocked by the official
`update_contacts()` writeback/conversion path rather than by missing solver
constraint force data. Gate 00D/00E still remain open because MJWarp EFC
export is only a candidate direct-force path until it is validated against
official `SensorContact` on a compatible MuJoCo-contact scene and then fused
into the dense pad tactile maps.

Candidate MJWarp direct-force tactile export update on 2026-07-01:
`p00_mjw_direct_v1_20260701_052900` converts the audited bottom-level MJWarp
EFC normal/tangent components into left/right pad-local dense tactile maps and
renders a synchronized Newton `SensorTiledCamera` scene + candidate direct
force tactile video. The run completed 240 frames on official Newton main,
passed the official Panda hydro final lift test, produced nonblank scene views,
and exported candidate maps with `127` pad-object contact frames, max
pad-object candidate `Fn` sum `48.28089141845703`, max pad-object candidate
`Ft` sum `48.28089141845703`, max left/right candidate `Fn` maps
`13.648624420166016` / `11.802962303161621`, and max left/right nonzero cell
ratios `0.3154296875` / `0.31640625`. Manual sheet inspection confirms
head/right-wrist/left-wrist scene frames plus left/right `Fn`/`Ft` heatmaps and
shear arrows appear together after the grasp contact window. This is a major
candidate direct-force tactile milestone, but Gate 00D/00E still remain open
until the candidate MJWarp mapping is validated against official
`SensorContact`/`update_contacts` on a compatible MuJoCo-contact scene and the
result is merged with the steel-spec calibrated hydro diagnostic.

SensorContact alignment update on 2026-07-01:
`p00_mjw_align_v1_20260701_055200` validates the candidate MJWarp EFC frame
mapping against official Newton `SensorContact.force_matrix` and
`force_matrix_friction` on the compatible MuJoCo-contact Panda variant. The
best sign convention is `shape0_negative` for both force and friction. Force
relative RMSE is `3.2491620810680347e-08`, friction relative RMSE is
`2.0018143688320552e-07`, both mean cosine values are `1.0`, and both norm
correlations are effectively `1.0`. This closes the compatible-scene
validation gap for the candidate EFC mapping. Gate 00D/00E still remain open
for active hydro because the validated mapping must now be run in the
steel-spec hydro fused diagnostic and the reference-video gel/marker tactile
comparison is still missing.

Steel-spec validated-sign direct-force tactile update on 2026-07-01:
`p00_mjw_direct_steel_v1_20260701_060500` reruns the direct-force tactile
export on official Newton main with steel-spec material override
(`mu=0.3`, `kh=1e12`), successful `notify_model_changed`, and the validated
`shape0_negative` sign convention from `p00_mjw_align_v1_20260701_055200`.
It passed the official 240-frame Panda hydro final test, enabled real
`SensorTiledCamera` scene views, and exported synchronized candidate direct
`Fn`/`Ft` tactile maps: `146` pad-object contact frames, max object lift
`0.2225421965122223` m, max pad-object candidate `Fn` sum
`40.099632263183594`, max pad-object candidate `Ft` sum `12.027974128723145`,
max left/right `Fn` map `8.926953315734863` / `9.650286674499512`, and max
left/right `Ft` map `2.6780858039855957` / `2.8950860500335693`. Manual sheet
inspection confirms nonblank scene views and synchronized `Fn`/`Ft` maps; the
`Ft` magnitude is consistent with the `mu=0.3` steel-spec friction setting.
Gate 00D/00E now has a validated-sign, steel-spec candidate direct force
asset, but final completion still requires reference-video-level gel/marker
tactile comparison and a final gate review.

Reference-video comparison update on 2026-07-01:
`p00_refcmp_v3_20260701_065300` compares the user reference video
`0780e5ec3fdb26b63ae63de0f49f07c4.mp4` with the steel-spec candidate
direct-force tactile video from `p00_mjw_direct_steel_v1_20260701_060500`.
The comparison decoded the reference MP4 with `imageio_ffmpeg` and decoded the
candidate uncompressed DIB AVI with the local built-in decoder. It produced
sample sheets and a side-by-side sheet:
`experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_v3_20260701_065300/reference_vs_candidate_sheet.jpg`.
Both videos are nonblank. Reference metrics include pixel-std mean
`92.9096450805664`, edge-density mean `0.09730502218008041`, and colorfulness
mean `32.0333366394043`; candidate metrics include pixel-std mean
`87.54474639892578`, edge-density mean `0.06210779771208763`, and colorfulness
mean `27.778425216674805`. Manual inspection confirms the candidate now has
real scene views plus synchronized candidate direct `Fn`/`Ft` maps and shear
arrows, but the reference remains richer because it includes gel/marker-style
tactile camera diagnostics, denser multi-panel tactile fields, and more
complete tactile/mechanics overlays. Gate 00D/00E remain open; this comparison
is a positive gap-defining asset, not completion.

Normal/area overlay update on 2026-07-01:
`p00_mjw_normarea_v1_20260701_071900` extends the steel-spec direct-force
candidate with contact-normal overlays from MJWarp `contact.frame` and
contact-area proxy overlays from pad-object point-contact density. It preserves
official Newton main, the official Panda hydro base, `SensorContact`-validated
`shape0_negative` force sign, and steel-spec material override. The run passed
the official final test, had zero read errors, `147` pad-object contact frames,
max object lift `0.22243636846542358` m, max candidate `Fn` sum
`40.0997428894043`, max candidate `Ft` sum `12.027881622314453`, left/right
area-proxy cell ratios `0.2900390625` / `0.279296875`, and left/right normal-yz
norm maxima `9.213287353515625` / `8.88884162902832`. The follow-up reference
comparison `p00_refcmp_normarea_v2_20260701_073000` marks normal overlay and
area-proxy overlay as current candidate channels. Gate 00D/00E remain open
because the area is still a point-contact-density proxy and the reference
video's gel/marker tactile rendering plus channel-by-channel semantic match are
still missing.

Candidate marker-render update on 2026-07-01:
`p00_mjw_marker_v1_20260701_074200` adds a blue gel-like marker/deformation
rendering derived from the candidate direct `Fn`, `Ft`, normal, and
contact-area-proxy fields. It keeps the official Newton main Panda hydro base,
steel-spec material override, and validated `shape0_negative` force sign. The
run passed the official final test, had zero read errors, `146` pad-object
contact frames, max object lift `0.2225111573934555` m, max candidate `Fn` sum
`41.90861511230469`, max candidate `Ft` sum `12.294239044189453`, and nonzero
left/right marker-flow norms `4.690944671630859` /
`3.1349213123321533`. The follow-up comparison
`p00_refcmp_marker_v1_20260701_074900` lists candidate gel/marker-style
rendering as a current channel. Gate 00D/00E remain open because this is a
candidate rendering derived from force fields, not validated Taccel/hardware
photometric marker output, and the reference-video channel semantics are still
not matched.

Gate review update on 2026-07-01:
`p00_gate_review_v2_20260701_080800` is the current strict Gate 00D/00E review.
It passes all current evidence checks: official Newton runtime `92.6 FPS`,
base grasp/lift final test, steel-spec material settings, candidate direct
`Fn`/`Ft`, `SensorContact` alignment, normal/area proxy overlay, candidate
gel/marker render, and nonblank reference-comparison assets. The review result
is intentionally conservative: status `open_not_curiosity_ready`, Gate 00D
`open_reference_semantics_blocked`, Gate 00E
`open_tactile_validation_blocked`, and `curiosity_training_allowed=false`.
The active blockers are validated gel/marker photometric semantics, validated
deformation-marker tracking, validated real contact-area semantics beyond the
point-contact proxy, and channel-by-channel semantic matching against the
reference video.

Channel semantic audit update on 2026-07-01:
`p00_chan_audit_v1_20260701_082100` performs the first channel-level visual
layout audit between the reference video and the current marker candidate. It
passes candidate scene, marker-render, force-heatmap, area-proxy, mechanics-
curve, and reference scene/tactile/mechanics layout checks, and exports
`experiments/visuals/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_sheet.jpg`.
`p00_gate_review_v3_20260701_082600` consumes that audit and adds
`channel_semantic_layout_audit` to the passed checks. Gate 00D/00E still remain
open because this is a layout/channel audit, not validated photometric marker
semantics, validated real contact-area semantics, or physical semantic
equivalence.

Latest 2026 reference-code scan on 2026-07-01:
UniVTAC, Tacmap, TaCauchy, and ControlTac are now recorded as relevant
comparison directions for the remaining semantic-validation blockers. UniVTAC
has official code and is built on Isaac Lab/TacEx for multiple optical tactile
sensors; Tacmap targets geometry-consistent penetration/deform maps; TaCauchy
targets FEM/Cauchy-stress tactile ground truth in Isaac Sim; ControlTac targets
force/position-conditioned tactile image generation. These should be treated as
reference/comparison paths until their official code paths pass sanity checks.

Steel-spec base update on 2026-07-01: `p00_steel_v1_20260701_032709` shows lift
success over `0.15` m, first lift at frame `169`, `71` hold frames above
threshold, no detected post-lift drop, max object lift `0.2235182225704193` m,
max `hydro_proxy.Fn` `22572.54296875`, max stress proxy `6979607.5`, max
tangential-capacity proxy `6771.763671875`, and max object acceleration
`1.2139862775802612` m/s^2. Gate 00E remains open because the 82 FPS target,
direct `Ft`, pad-resolved shear vectors, and reference-video-level tactile
density are still missing.

Grid-tactile base update on 2026-07-01: `p00_grid_v1_20260701_033556` preserves
lift success over `0.15` m and `71` hold frames while exporting dense grid
source arrays: max left/right `Fn` maps `3512.72900390625` /
`1440.114013671875`, max left/right shear-magnitude maps
`165.96604919433594` / `42.3233757019043`, and active grid shear frames
`165` / `167`. This is a positive base-export improvement, but it is still
proxy tactile evidence and does not close Gate 00E.

F6 proxy base update on 2026-07-01: `p00_f6_v1_20260701_034033` preserves lift
success over `0.15` m, `71` hold frames, no detected post-lift drop, max object
lift `0.2235504388809204` m, max `hydro_proxy.Fn` `22694.48046875`, and exports
nonzero F6 combined proxy norms: left `1477.2451171875`, right
`2114.81494140625`. Gate 00E remains open because these are proxy wrenches, not
direct `SensorContact`/hardware-like tactile force.

Direct-force comparison base update on 2026-07-01:
`p00_mjc_sensor_v1_20260701_034541` has direct `SensorContact` friction
evidence with max total friction norm `8.532435417175293`, max matrix friction
norm `4.5348076820373535`, lift success over `0.15` m, and max object lift
`0.21197126805782318` m. Gate 00E remains open because this is a separate
MuJoCo-contact variant, not the dense hydro tactile base.

Gate 00F update on 2026-07-01:
`p00_gate_review_v4_20260701_055100` consumes the official reference blocker
probes. It keeps mechanical/runtime/candidate-layout evidence positive, but
fails `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity` because approved prebuilt reference
environments are missing. Gate 00D remains `open_reference_semantics_blocked`,
Gate 00E remains `open_tactile_validation_blocked`, Gate 00F remains
`open_official_semantic_validation_blocked`, and curiosity training remains
disallowed.

Gate 00F bridge-spec update on 2026-07-01:
`semantic_bridge_spec_v1.json` makes the required candidate-to-official mapping
explicit for `Fn`, `Ft`, marker flow, area proxy, contact normal, and scene RGB.
`p00_gate_review_v5_20260701_060100` ran inside Curiosity Slurm job `160454`
on `server02` and verifies that Gate review now consumes this bridge spec. The
new `semantic_bridge_spec_available` check passes. Gate 00F still remains open
because `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity` remain blocked by missing approved
prebuilt environments.

Gate 00F environment-availability update on 2026-07-01:
`experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh`
now provides a repeatable lightweight guard for the official reference
environments. The current status file
`experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`
reports both UniVTAC candidate Python paths missing
(`envs/univtac/conda/bin/python` and `envs/univtac/.venv/bin/python`) and both
TaCauchy candidate Python paths missing (`envs/tacauchy/conda/bin/python` and
`envs/tacauchy/.venv/bin/python`); `git-lfs`, `cmake`, and `nvcc` are also
missing from the current login PATH. This confirms Gate 00F is still blocked
before compute-side official sanity can pass. Gate review code and launchers
now accept
`experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`
as explicit Gate 00F evidence, so future reports include the missing-env
failure directly.

Gate 00F repeatable readiness check on 2026-07-01:
`experiments/configs/phase00/ref_tactile/envprep/check_gate00f_readiness.sh`
now provides a conservative guard that combines target env, toolchain, asset,
and latest Gate-review checks without importing packages or running simulation.
It writes
`experiments/outputs/phase00/ref_tactile/envprep/gate00f_readiness/gate00f_readiness_status.json`
and
`experiments/reports/phase00/ref_tactile/envprep/gate00f_readiness.md`.
Current result is `gate00f_ready=false` with
`reason=blocked_official_sanity_or_gate_review_not_passed`. It records
project-local conda and CUDA 12.8 `nvcc` as present. After approved env prep,
`envs/univtac/conda/bin/python` and `envs/tacauchy/conda/bin/python` are both
present. After approved asset reuse, TaCauchy `Sensor.usd` and `21` tactile
test shape USDs are also present. The effective remaining failed checks are
`univtac_official_reference_sanity` and `tacauchy_official_reference_sanity`.

Approved TaCauchy asset reuse on 2026-07-01:
After the user explicitly allowed continuation, the candidate UniVTAC bundled
TacEx asset reuse path was executed:
`experiments/reports/phase00/ref_tactile/envprep/approved_asset_reuse_execution.md`.
The copy created `273` files, transferred `429244198` bytes, and changed the
TaCauchy asset tree from `1.8M` to `412M`. `Sensors/GelSight_Mini/Sensor.usd`
is now present and TaCauchy has `21` tactile test shape USD files. This clears
the asset file-presence blocker, but not official UniVTAC/TaCauchy sanity. A
fresh Gate review still needs to consume the post-copy asset and env
availability.

Approved local env creation on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`
records successful base Python env creation for UniVTAC (`Python 3.10.20`,
`140M`) and TaCauchy (`Python 3.11.15`, `166M`) using `--no-lock --solver
classic`. This is only base env availability, not official dependency
installation or sanity.

UniVTAC lock-error audit on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/univtac_env_create_attempts.md`
records three failed conda-lock attempts followed by one successful `--no-lock`
retry. This keeps the failed-path evidence without misclassifying the current
env state.

Official dependency stage dry-run on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/reference_dependency_stage_plan.md`
records generated dry-run command evidence for UniVTAC and TaCauchy official
dependency stages (`install_isaac`, `install_isaaclab`,
`install_curobo_or_assets`, `install_tacex_core`, `build_uipc`,
`setup_assets`, and `official_sanity`). No heavy dependency installation or
official sanity was executed.

Reference dependency install blocker on 2026-07-01:
`experiments/reports/phase00/ref_tactile/envprep/reference_dependency_install_blocker.md`
records the remaining hard blocker. Official readiness now requires heavy
Isaac/TacEx/UIPC dependency installation or builds. Project rules forbid heavy
work on the login node and also forbid dependency installation/builds on
compute nodes, so continuing requires an approved prebuilt Curiosity
shared-filesystem reference env or a compliant non-login env-prep workflow.

Gate 00F dependency resolution packet on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_dependency_resolution_packet.md`
and
`experiments/configs/phase00/ref_tactile/gate00f_dependency_resolution_packet_v1.json`
make the remaining dependency gap explicit. UniVTAC requires the official
Isaac Sim 4.5 / Isaac Lab 2.1.1 / TacEx / cuRobo / UIPC path. TaCauchy requires
the Isaac Sim 5.0 / Isaac Lab 2.2.1 / TacEx assets / UIPC-lib build path with
vcpkg, CMake, GCC, and CUDA readiness. IsaacLab TacSL requires a
dependency-complete official Isaac Lab/TacSL environment. Acceptable next
paths are only existing dependency-complete envs, existing prebuilt containers,
or a compliant non-login/non-experiment env-prep workflow followed by bundle
acceptance. Login-node UIPC builds and compute-allocation dependency installs
remain disallowed.

Gate 00F runtime locator probe on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701.md`
and
`experiments/configs/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701_v1.json`
record the refreshed lightweight path state. The shell exposes `/usr/bin/docker`
but no `module`/`ml`, `singularity`/`apptainer`/`enroot`, `git-lfs`, `cmake`,
`nvcc`, or `nvidia-smi`. Project env prefixes include UniVTAC and TaCauchy base
Python paths, but no dependency-complete UniVTAC/TaCauchy/IsaacLab TacSL
runtime was found. Gate 00F remains blocked.

Gate 00F shared runtime locator and preflight handoff on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701.md`
records that common shared software/container top-level paths and Docker image
names did not contain an existing Isaac/TacEx/TaCauchy/UniVTAC/TacSL/UIPC
runtime or container. A slower project-local artifact search was interrupted to
avoid login-node waste. The next compute-side handoff is
`experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`,
documented by
`experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_handoff.md`.
It checks only Python executability and module specs for UniVTAC, TaCauchy, and
IsaacLab TacSL before the Gate 00F bundle, but only after runtime registry
validation passes. It is preflight evidence only and does not clear Gate 00F.
The login-node refuse check
`experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701.md`
confirms the preflight exits with code `2` when `SLURM_JOB_ID` is missing.

Gate 00F runtime registry on 2026-07-01:
`experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`
and validator
`src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py` add a
required registration layer before runtime preflight. UniVTAC, TaCauchy, and
IsaacLab TacSL must each be registered as `dependency_complete_registered`
through an allowed resolution path before any runtime is used. Current
validation
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_current_20260701.md`
fails as expected: UniVTAC/TaCauchy are base Python envs only, and IsaacLab
TacSL has no runtime path. This failure prevents accidental direct promotion
from base envs into runtime preflight or Gate 00F bundle execution.

Gate 00F runtime registration handoff on 2026-07-01:
`src/newton_tactile_curiosity/gate00f_runtime_register.py` and
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registration_handoff.md`
provide a controlled metadata-only way to write a copied candidate registry
when a real dependency-complete Python env, local Docker image ID, or shared
container artifact becomes available. It rejects placeholders and excluded
resource-zone paths, and it does not pull/build images, run containers, import
Isaac/TacSL modules, or modify environments. Any candidate registry must still
pass the registry validator, runtime preflight, Gate 00F bundle, and strict
bundle acceptance. Container registrations additionally require a provenance
summary with status `pass_gate00f_container_provenance` and matching target
before a copied candidate registry can be written.

Gate 00F post-8c501 runtime acceptance handoff on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`
records that the latest Newton `8c501...` candidate evidence should be reused
for future Gate 00F bundle attempts. The current blocker is official reference
runtime readiness, not FPS or another Newton candidate export. The required
order is copied runtime registration, registry validation, runtime preflight,
Gate 00F bundle against the 8c501 evidence, then strict bundle acceptance. The
runtime preflight now reads registered `python_env` paths from the accepted
registry and supports container module-spec checks for registered docker local
image IDs and singularity/apptainer/sif artifact paths. Enroot, sqsh, and tar
still require explicit runners.

Gate 00F scoped project artifact probe on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_project_artifact_probe_20260701.md`
records a bounded project-local search under `envs`, `experiments/configs`,
`experiments/outputs`, and `external`. No `.sif`, `.sqsh`, `.tar`,
`.tar.gz`, or `.img` container artifact was found, and no `cmake`, `git-lfs`,
`singularity`, `apptainer`, or `docker` file was found under `envs` at max
depth `4`. The only relevant tool hit was `envs/taccel/cuda-toolkit/bin/nvcc`.
This is additional blocker evidence only; it does not clear Gate 00F.

Gate 00F bundle preflight gate update on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701.md`
records that `run_gate00f_reference_bundle_in_alloc.sh` now runs runtime
preflight before official UniVTAC/TaCauchy/TacSL sanity commands. If preflight
does not report `pass_gate00f_runtime_preflight`, the bundle writes
`fail_gate00f_bundle_runtime_preflight_not_passed` and exits before official
sanity. This makes the active order registry validation -> runtime preflight ->
official sanity -> Gate review -> bundle acceptance.

Post-container-preflight update: the bundle now forwards `RUNTIME_REGISTRY` to
runtime preflight and official sanity sub-scripts. Supported container runtimes can now feed the
container-aware official sanity runners after registration and preflight, but a
real registered runtime and compute-side execution are still required before
any Gate 00F pass claim.

Gate 00F container acquisition plan on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701.md`
records the latest official container route. NVIDIA provides official Isaac
Sim containers and Isaac Lab documents pre-built NGC Isaac Lab containers such
as `nvcr.io/nvidia/isaac-lab:2.3.2`; this is the strongest current candidate
for the IsaacLab TacSL path after compatibility checks. UniVTAC/TacEx and
TaCauchy still require project container layers over an Isaac Lab base image:
local TacEx/TaCauchy Docker configs use `ISAACLAB_BASE_IMAGE=isaac-lab-base`
and build `isaac-lab-tacex`. Therefore the faithful container path requires an
existing prebuilt project image or a compliant external build, then runtime
registry registration and validation before preflight.

Gate 00F runtime registry container support update on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701.md`
extends the registry validator for strict container registration. A container
entry must provide `container_runtime` plus either a local `image_id` or an
existing shared `artifact_path`; remote `image_ref` alone is only acquisition
evidence. The validator also requires local image IDs to look like immutable
digests or IDs and requires artifact paths to exist as files with container
archive suffixes; the runtime registration helper applies the same checks
before writing a copied registry. The current registry still fails, as
intended, because no real container runtime is registered.

Gate 00F container provenance contract on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`
and validator
`src/newton_tactile_curiosity/gate00f_container_provenance_validate.py` define
the minimum evidence required before any future prebuilt container can be
written into a copied runtime registry. The contract records the official
source commits for IsaacLab TacSL, UniVTAC/TacEx, and TaCauchy, the local
Dockerfile/compose metadata for TacEx/TaCauchy, the expected module families,
and the requirement for real provenance paths plus local `image_id` or
existing `artifact_path`. The provenance validator now applies the same hard
local runtime-reference checks as the registry validator. The negative-control
packet
`gate00f_container_provenance_isaaclab_ref_only_20260701_v1.json` fails as
expected because a remote `image_ref` alone is not runtime evidence. The
runtime registration helper now requires this validator to pass before any
future container registration can write a copied candidate registry.

Gate 00F runtime intake chain on 2026-07-01:
`src/newton_tactile_curiosity/gate00f_runtime_intake_chain.py` composes
container provenance validation, copied runtime registry registration, and
copied runtime registry validation. It stops on the first failure and does not
run containers, imports, installs, simulation, rendering, training, evaluation,
or Slurm jobs. The IsaacLab remote-image-only negative control stops at
`fail_container_provenance` and does not write a candidate registry.

Gate 00F TacSL source compatibility on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`
records a static source check of `external/IsaacLab_official` against the
candidate Isaac Lab container ref `nvcr.io/nvidia/isaac-lab:2.3.2`. The check
passes for local VERSION `2.3.2`, required TacSL data fields, required demo
flags, and required imports. This is useful positive evidence for the
IsaacLab TacSL container path, but it is not runtime evidence: no container was
pulled, no image was built, no TacSL/Isaac module was imported, no Isaac Sim
process ran, and no runtime was registered as dependency-complete. Gate 00F
remains open.

Gate 00F TacSL container/documentation refresh on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701.md`
records the official Isaac Lab container and TacSL documentation route plus a
runtime risk for tactile RGB: an upstream issue reports that
`tacsl_sensor.py --use_tactile_rgb` can fail when the GelSight R15 `bg.jpg`
asset is missing, and local static checks found no `bg.jpg` under
`external/IsaacLab_official`. This does not clear Gate 00F; it prevents
silently dropping tactile RGB from the official sanity path.

Gate 00F IsaacLab upstream freshness on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701.md`
records that local `external/IsaacLab_official` matches upstream `main`/`HEAD`
at `b4c321024792976150ca55fddb26fa34480d974e`, with `v3.0.0-beta*` tags
visible as release context. This is source freshness only; it does not
register a runtime or clear Gate 00F.

Gate 00F reference repository freshness on 2026-07-01:
`experiments/reports/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701.md`
records that local UniVTAC, TaCauchy, and TacEx checkouts match upstream main.
This keeps the official reference source path current, but Gate 00F remains
blocked until dependency-complete runtimes and official sanity pass.

### Gate 00G: Curiosity Readiness

Only after Gate 00D, Gate 00E, and Gate 00F pass or have faithful blockers
accepted for comparison gaps, design the new curiosity training:

- dense visuo-tactile forward/world model;
- active probing objective;
- tactile learning-progress reward;
- safety constraints;
- baselines and ablations;
- tactile-mask training.

Current Gate 00G source-design update on 2026-07-01: APPLE and Tactile MNIST
have been cloned and source-audited as secondary official active tactile
perception references. The reference matrix is
`experiments/configs/phase00/ref_tactile/curiosity_reference_matrix_v1.json`,
and the audit report is
`experiments/reports/phase00/ref_tactile/curiosity_reference_audit.md`. These
references constrain the future curiosity design toward closed-loop active
probing, sequential tactile memory, random/grid/scripted baselines, and
tactile-only/masked-vision ablations. They do not allow curiosity training to
start while Gate 00D/00E/00F remain open.

## Completion Criteria

Phase 00 completes only when:

- the codebase audit is written and source-backed;
- official code paths are preserved or blockers are documented;
- the dense tactile schema exists;
- at least one rigid metal reference diagnostic is generated in compute;
- the base grasp controller/model produces complete dense tactile/mechanics
  evidence;
- official tactile semantic validation against UniVTAC/TaCauchy passes or
  records faithful blockers that the user accepts;
- no login-node simulation/rendering/training/data conversion was used;
- all outputs use short grouped paths under `phase00/ref_tactile/`.

Current transition decision on 2026-07-01: the user explicitly challenged
continuing to block on official reference runtimes because the Newton
simulation platform now has the needed engineering tactile evidence for a
Newton-only training track. Therefore Phase 01 may proceed as Newton-only
dense tactile curiosity training using the accepted blocker boundary:
Gate 00F remains open, official UniVTAC/TaCauchy/IsaacLab TacSL validation is a
pending comparison gap, and Phase 01 results must not be reported as final
reference-video tactile validation.

Priority update on 2026-07-01: the user then explicitly downgraded Gate 00F.
Gate 00F must be treated as a low-priority final validation/comparison-gap
track, not a high-priority current experiment and not the active blocker.
Current work is paused/blocked by user request pending the next instruction.
