# Phase 00 Active Evidence Index

Date: 2026-07-01

Purpose: keep the current reference-video tactile reset evidence easy to find.
This is an index, not a success claim. Gate 00D/00E/00F remain open and
curiosity training remains disallowed.

## Active Objective

Build a Newton/Taccel-based tactile infant that can grasp/lift/hold while
exporting dense visual+tactile mechanics comparable to the user reference
video. Restart curiosity only after dense tactile semantics and base evidence
pass.

## Requirement Status Audit

- Human-readable audit:
  `experiments/reports/phase00/ref_tactile/phase00_requirement_status.md`
- Machine-readable audit:
  `experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`
- Classification:
  status audit only; not training, not Gate completion, and not curiosity
  success.
- Current conclusion:
  latest source audit, Newton main steel-spec direct-force candidate, and
  92.6 FPS official Panda hydro base are partial positive evidence. Official
  UniVTAC/TaCauchy semantic validation is blocked by missing approved
  prebuilt environments. Curiosity training remains disallowed.
- Latest source recheck V3:
  `experiments/reports/phase00/ref_tactile/latest_reference_recheck_20260701_v3.md`
  and
  `experiments/configs/phase00/ref_tactile/latest_reference_recheck_20260701_v3.json`
  record the latest source truth: Newton upstream main is
  `8c501b47847569fecdda97a9f7f01205c6f7964f`, `external/newton_8c501` is a
  source-only worktree, `external/TacEx` is cloned at
  `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`, and
  `external/IsaacLabTactile` is cloned at
  `21bcb476b27ceedccccd63afef6bbd822adc2b2b` with LFS skipped. This is not
  compute sanity or Gate completion evidence.
- Newton 8c501 compute handoff:
  `experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md`
  and
  `experiments/configs/phase00/ref_tactile/newton_8c501_sanity_handoff_v1.json`
  record the exact tmux-held Slurm command sequence for H200 runtime benchmark
  and downstream tactile/Gate checks. The runtime stage has run around 80 FPS
  and is acceptable for continuing; the old 82 FPS number is not a hard gate.
- Newton 8c501 allocation request:
  `experiments/reports/phase00/ref_tactile/newton_8c501_allocation_request.md`
  and
  `experiments/configs/phase00/ref_tactile/newton_8c501_allocation_request_v1.json`
  record job `160854` in tmux window
  `curiosity_phase00_ref_tactile:alloc_8c501`, initially `PENDING (Priority)`.
  This is allocation evidence only, not benchmark or tactile evidence.
- Newton 8c501 benchmark status:
  `experiments/reports/phase00/ref_tactile/newton_8c501_benchmark_status.md`
  and
  `experiments/configs/phase00/ref_tactile/newton_8c501_benchmark_status_v1.json`
  record two H200 runs on job `160854`: `80.1 FPS` and `80.8 FPS`. Both ran
  successfully and are acceptable around 80 FPS. They should not block
  `8c501...` dense tactile export; the remaining blockers are tactile semantic
  completeness and official reference sanity/blocker evidence.
- Runtime gate correction:
  `experiments/reports/phase00/ref_tactile/runtime_gate_correction_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/runtime_gate_correction_20260701_v1.json`
  record that `82 FPS` is historical reference only; around `80 FPS` is
  acceptable for continuing dense tactile export and Gate checks.
- Newton 8c501 continuation allocation request:
  `experiments/reports/phase00/ref_tactile/newton_8c501_cont_allocation_request.md`
  and
  `experiments/configs/phase00/ref_tactile/newton_8c501_cont_allocation_request_v1.json`
  record tmux-held Slurm job `160924`, window `alloc_8c501_cont`, initially
  `PENDING (Priority)`, intended for the 8c501 dense tactile export ->
  reference compare -> channel audit -> Gate review chain.
- Newton 8c501 continuation chain status:
  `experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`
  and
  `experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`
  record latest-source positive candidate tactile evidence on job `160924`:
  dense tactile export passed, reference comparison passed, channel audit
  passed, and Gate review remains `open_not_curiosity_ready` only on official
  reference sanity/semantic blockers.
- Gate 00F post-8c501 readiness:
  `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_readiness_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_readiness_20260701_v1.json`
  record that the Gate 00F bundle now defaults to the 8c501 chain, but runtime
  registry still fails because UniVTAC/TaCauchy are base envs only and
  IsaacLab TacSL has no dependency-complete runtime.
- Gate 00F post-8c501 runtime acceptance handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_post_8c501_runtime_acceptance_handoff_v1.json`
  record the current strict order after the successful 8c501 candidate chain:
  register dependency-complete official runtimes, validate the copied runtime
  registry, run runtime preflight, run the Gate 00F bundle against the latest
  8c501 candidate evidence, then run strict bundle acceptance. This handoff
  also records the supported container preflight path for future registered
  docker/singularity/apptainer/sif runtimes.
- Gate 00F readiness refresh:
  `experiments/reports/phase00/ref_tactile/gate00f_readiness_refresh_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_readiness_refresh_20260701_v1.json`
  record the latest lightweight state: env pythons/assets are present,
  `gate00f_ready=false`, and effective failed checks remain
  `univtac_official_reference_sanity` and
  `tacauchy_official_reference_sanity`.
- Gate 00F tool lookup:
  `experiments/reports/phase00/ref_tactile/gate00f_tool_lookup_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_tool_lookup_20260701_v1.json`
  record that PATH lacks `git-lfs`, `cmake`, `nvcc`, and `nvidia-smi`;
  project-local lookup found only `envs/taccel/cuda-toolkit/bin/nvcc`; no
  prebuilt Isaac/Lab/TacEx/UIPC env directories were found under `envs`.
- Gate 00F static source audit:
  `experiments/reports/phase00/ref_tactile/gate00f_static_source_audit_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_static_source_audit_20260701_v1.json`
  record the source-level official sanity/schema requirements. UniVTAC is the
  left/right tactile RGB/marker/depth/pose schema and manipulation benchmark
  reference; TaCauchy is the Cauchy-stress, normal-pressure, tangential-
  traction, optical/marker tactile semantic reference; local IsaacLabTactile is
  currently a source gap/generic contact-sensor reference, not a Gate 00F
  replacement.
- Gate 00F module/env probe:
  `experiments/reports/phase00/ref_tactile/gate00f_module_env_probe_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_module_env_probe_20260701_v1.json`
  record that the current shell has no `module`/`ml` command and that shallow
  target-env probing found no Isaac/TacEx/UIPC/cuRobo/Torch component names in
  the existing UniVTAC/TaCauchy base env prefixes.
- Gate 00F container path audit:
  `experiments/reports/phase00/ref_tactile/gate00f_container_path_audit_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_container_path_audit_20260701_v1.json`
  record that Docker build/helper paths exist in TacEx/TaCauchy/IsaacLabTactile,
  but no approved prebuilt Curiosity image/SIF/tar artifact was found.
- Latest 2026-07-01 web/codebase refresh:
  `experiments/reports/phase00/ref_tactile/latest_20260701_web_codebase_refresh.md`
  and
  `experiments/configs/phase00/ref_tactile/latest_20260701_web_codebase_refresh_v1.json`
  record new official sparse sources: `external/IsaacLab_official` for TacSL,
  `external/ftp1-policy` for FTP-1, and `external/AnyTouch2` for tactile
  representation. This is source evidence only, not checkpoint/model/sanity or
  training evidence.
- Latest supplementary codebase audit:
  `experiments/reports/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701_v1.json`
  record `external/TactSim-IsaacLab` as a secondary photometric
  GelSight/DIGIT-style IsaacLab tactile reference, `external/newton-actuators`
  as deprecated Newton actuator background only, and UniT as remote-head-only
  future representation evidence. This is supplementary source evidence only;
  it does not clear Gate 00D/00E/00F and does not authorize curiosity
  training.
- Latest source freshness V4:
  `experiments/reports/phase00/ref_tactile/latest_source_freshness_20260701_v4.md`
  and
  `experiments/configs/phase00/ref_tactile/latest_source_freshness_20260701_v4.json`
  record that tracked official refs for Newton, Taccel, T-Rex, IsaacLab,
  TacEx, TaCauchy, UniVTAC, FTP-1, AnyTouch2, and HydroShear match current
  source records. This is source freshness only and does not clear any gate.
- Latest policy/checkpoint refresh:
  `experiments/reports/phase00/ref_tactile/latest_policy_checkpoint_refresh_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/latest_policy_checkpoint_refresh_20260701_v1.json`
  record clean official T-Rex source snapshots at `external/T-Rex_43ff` and
  `external/T-Rex_full_b23`, plus released T-Rex, FTP-1, AnyTouch2, and Sparsh
  checkpoint/reference availability. This is source/checkpoint availability
  evidence only; no checkpoint was downloaded, loaded, or trained.
- Post-Gate 00F policy bridge checklist:
  `experiments/reports/phase00/ref_tactile/post_gate00f_policy_bridge_checklist.md`
  and
  `experiments/configs/phase00/ref_tactile/post_gate00f_policy_bridge_checklist_v1.json`
  define future T-Rex/FTP-1/AnyTouch2/Sparsh preconditions, schema contracts,
  required ablations, and forbidden shortcuts.
- T-Rex data-contract extraction:
  `experiments/reports/phase00/ref_tactile/trex_data_contract.md`,
  `experiments/configs/phase00/ref_tactile/trex_data_contract_v1.json`, and
  `src/newton_tactile_curiosity/trex_contract_validate.py` define the future
  metadata gate for Newton-to-T-Rex conversion. This is a schema guard only,
  not T-Rex compatibility evidence for the current Newton Panda candidate.
- IsaacLab TacSL sanity handoff:
  `experiments/reports/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff.md`
  and
  `experiments/configs/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff_v1.json`
  prepare the official TacSL demo sanity path and launch scripts. This handoff
  is not run and does not clear Gate 00F.
- Gate 00F TacSL review wiring:
  `src/newton_tactile_curiosity/phase00_gate_review.py` now requires
  `OfficialIsaacLabTacSL` semantic matrix coverage,
  `candidate.newton_mjw.penetration_or_compression` bridge coverage, and a
  compute-side sanity summary with status
  `pass_official_isaaclab_tacsl_demo_exited_zero` before TacSL can count for
  official reference sanity. The gate-review launchers now accept
  `ISAACLAB_TACSL_SANITY_SUMMARY`.
- TacSL env/container blocker refresh:
  `experiments/reports/phase00/ref_tactile/isaaclab_tacsl_env_blocker_refresh_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/isaaclab_tacsl_env_blocker_refresh_20260701_v1.json`
  record that Slurm job `160860` is Reflex-owned by `WorkDir` and cannot be
  reused, while no `envs/isaaclab_tacsl` prefix or Curiosity-local TacSL/
  Isaac/TacEx prebuilt container archive was found in limited checks.
- Unified Gate 00F reference bundle handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_reference_bundle_handoff_v1.json`
  define a single allocation workflow for UniVTAC, TaCauchy, IsaacLab TacSL,
  and Gate review. The bundle forwards `RUNTIME_REGISTRY` to runtime preflight
  and to the official sanity sub-scripts; those runners can dispatch registered
  docker/singularity/apptainer/sif runtimes through the shared container
  helper. The safety check
  `experiments/reports/phase00/ref_tactile/gate00f_bundle_launcher_reflex_refuse_check_20260701.md`
  confirms the launcher refused Reflex-owned Slurm job `160860`.
- Gate 00F bundle acceptance checker:
  `experiments/reports/phase00/ref_tactile/gate00f_bundle_acceptance_handoff.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_bundle_acceptance_handoff_v1.json`,
  and `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py` define the
  strict JSON-only acceptance gate for any future bundle summary.
- Gate 00F container acquisition plan:
  `experiments/reports/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701_v1.json`
  record official Isaac Sim/Isaac Lab container availability, TacEx/TaCauchy
  project-container build requirements, and the required registration route for
  any future prebuilt container before runtime preflight.
- Gate 00F runtime registry container support update:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701_v1.json`,
  `experiments/reports/phase00/ref_tactile/gate00f_container_runtime_registration_examples.md`,
  and
  `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_registration_examples_v1.json`.
  Container registration now requires a local `image_id` or existing shared
  `artifact_path`; remote `image_ref` alone is acquisition evidence only. The
  registration helper and validators also require image IDs to look like
  immutable local digests/IDs and artifact paths to exist as files with
  accepted container archive suffixes.
- Gate 00F container provenance contract:
  `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_contract_v1.json`,
  and
  `src/newton_tactile_curiosity/gate00f_container_provenance_validate.py`.
  This defines the minimum evidence required before any future prebuilt
  container/image can be written into a copied runtime registry: target source
  commit, expected modules, real provenance paths, and local `image_id` or
  existing `artifact_path`. The same strict local runtime-reference checks are
  enforced here before any container can be registered.
- Gate 00F container provenance negative control:
  `experiments/reports/phase00/ref_tactile/gate00f_container_provenance_negative_control_20260701.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_container_provenance_isaaclab_ref_only_20260701_v1.json`,
  and
  `experiments/outputs/phase00/ref_tactile/container_provenance/p00_isaaclab_ref_only_20260701/container_provenance_validation_summary.json`.
  The validator correctly rejects `nvcr.io/nvidia/isaac-lab:2.3.2` when it is
  only a remote `image_ref` with no local image ID or shared artifact path.
- Gate 00F runtime intake chain:
  `src/newton_tactile_curiosity/gate00f_runtime_intake_chain.py`,
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff_v1.json`,
  and
  `experiments/outputs/phase00/ref_tactile/runtime_intake/p00_isaaclab_ref_only_20260701/runtime_intake_summary.json`.
  The chain composes provenance validation, copied-registry registration, and
  copied-registry validation. The remote-image-only negative control stops at
  `fail_container_provenance` and writes no candidate registry.
- Gate 00F TacSL source compatibility:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_handoff.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_tacsl_source_compat_handoff_v1.json`,
  `src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py`,
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`,
  and
  `experiments/outputs/phase00/ref_tactile/tacsl_source_compat/p00_tacsl_src_compat_20260701/tacsl_source_compat_summary.json`
  record that local `external/IsaacLab_official` VERSION `2.3.2` is source-
  compatible with candidate image ref `nvcr.io/nvidia/isaac-lab:2.3.2` for
  required TacSL data fields, demo flags, and imports. This strengthens the
  IsaacLab TacSL container candidate only; it does not register a runtime,
  run Isaac Sim, import TacSL modules, or clear Gate 00F.
- Gate 00F TacSL container/doc refresh:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701_v1.json`
  record that official Isaac Lab docs and the NGC catalog support the
  TacSL/IsaacLab container route, while an upstream IsaacLab issue and local
  static checks flag a `--use_tactile_rgb` risk from a missing GelSight R15
  `bg.jpg` background asset. Do not silently drop tactile RGB to pass Gate
  00F.
- Gate 00F IsaacLab upstream freshness:
  `experiments/reports/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701_v1.json`
  record that local `external/IsaacLab_official` matches upstream
  `main`/`HEAD` at `b4c321024792976150ca55fddb26fa34480d974e`; visible
  `v3.0.0-beta*` tags are release context, not evidence that the main checkout
  is stale.
- Gate 00F reference repository freshness:
  `experiments/reports/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701_v1.json`
  record that local UniVTAC, TaCauchy, and TacEx checkouts match upstream
  `main`/`HEAD`. This is source freshness only, not runtime or official sanity
  evidence.

## Core Base Evidence

- Latest Newton main worktree:
  `external/newton_main`
- Active evidence Newton commit:
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`
- Latest observed upstream Newton main:
  `8c501b47847569fecdda97a9f7f01205c6f7964f`
- Source refresh:
  `experiments/reports/phase00/ref_tactile/latest_source_remote_refresh.md`
- Latest source recheck V3:
  `experiments/reports/phase00/ref_tactile/latest_reference_recheck_20260701_v3.md`
- Newton 8c501 compute handoff:
  `experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md`
- Newton 8c501 allocation request:
  `experiments/reports/phase00/ref_tactile/newton_8c501_allocation_request.md`
- Newton 8c501 benchmark status:
  `experiments/reports/phase00/ref_tactile/newton_8c501_benchmark_status.md`
- Gate 00F readiness refresh:
  `experiments/reports/phase00/ref_tactile/gate00f_readiness_refresh_20260701.md`
- Gate 00F tool lookup:
  `experiments/reports/phase00/ref_tactile/gate00f_tool_lookup_20260701.md`
- Gate 00F static source audit:
  `experiments/reports/phase00/ref_tactile/gate00f_static_source_audit_20260701.md`
- Gate 00F module/env probe:
  `experiments/reports/phase00/ref_tactile/gate00f_module_env_probe_20260701.md`
- Gate 00F container path audit:
  `experiments/reports/phase00/ref_tactile/gate00f_container_path_audit_20260701.md`
- Latest web/codebase refresh:
  `experiments/reports/phase00/ref_tactile/latest_20260701_web_codebase_refresh.md`
- IsaacLab TacSL sanity handoff:
  `experiments/reports/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff.md`
- Latest Newton source-only worktree:
  `external/newton_8c501`
- Latest Newton prepared worktree:
  `external/newton_d58`
- Latest Newton continuation evidence worktree:
  `external/newton_8c501`
- Latest Newton prepared-worktree status:
  `experiments/reports/phase00/ref_tactile/newton_d58_worktree_status.md`
- Newton d58 allocation request:
  `experiments/reports/phase00/ref_tactile/newton_d58_allocation_request.md`
- Newton d58 benchmark status:
  `experiments/reports/phase00/ref_tactile/newton_d58_benchmark_status.md`
- Newton d58 tactile export status:
  `experiments/reports/phase00/ref_tactile/newton_d58_tactile_export_status.md`
- Gate 00E d58 base evidence audit:
  `experiments/reports/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit.md`
  and
  `experiments/outputs/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit_summary.json`.
  Current status is
  `partial_positive_gate00e_base_candidate_tactile_validation_blocked`: d58 is
  the strongest base candidate, but Gate 00E remains open.
- Status:
  latest upstream `8c501...` has source evidence and acceptable runtime around
  80 FPS on two H200 benchmark runs (`80.1 FPS` and `80.8 FPS`), so FPS should
  not block dense tactile export. The previous d58 worktree has runtime-positive
  evidence from
  `p00_bench_d58_hot_v1_20260701_070611` at `82.7 FPS` and candidate dense
  tactile/mechanics export from `p00_mjw_d58_marker_v1_20260701_071248`.
  The d58 export is nonblank and physically responsive, but
  `direct_tactile_claim_allowed=false`; reference-video comparison and Gate
  review are still required before d58 can replace the older active evidence
  chain.
- Newton d58 reference comparison:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_d58_marker_v1_20260701_071521/reference_video_compare_summary.json`
- Newton d58 channel audit:
  `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_d58_marker_v1_20260701_071757/channel_semantic_audit_summary.json`
- Newton d58 Gate review:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_d58_marker_v1_20260701_071843/phase00_gate_review_summary.json`
- Gate 00D environment evidence audit:
  `experiments/reports/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701_v1.json`.
  Current status is
  `partial_positive_environment_candidate_reference_semantics_blocked`: d58 has
  real environment/tactile candidate evidence, but contact area is proxy-only
  and dense penetration/compression semantics are not validated.
- Gate result:
  `open_not_curiosity_ready`. d58 is currently the strongest candidate evidence
  chain, with 12 passed checks, but Gate 00D/00E/00F remain open due official
  reference environment/assets and UniVTAC/TaCauchy sanity blockers plus
  unvalidated photometric marker and real contact-area semantics.
- TaCauchy asset blocker audit:
  `experiments/reports/phase00/ref_tactile/envprep/tacauchy_asset_blocker_audit.md`
  records the pre-copy blocker. After user approval, the UniVTAC bundled TacEx
  assets were copied into `external/TaCauchy`; the current post-copy asset
  status is recorded in `approved_asset_reuse_execution.md` and
  `reference_asset_availability.md`.
- Reference environment blocker audit:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_blocker_audit.md`
  records that the UniVTAC/TaCauchy target envs are missing and heavy env
  construction is not allowed as a silent login-node shortcut or as compute-node
  dependency installation.
- Gate 00F decision packet:
  `experiments/reports/phase00/ref_tactile/envprep/gate00f_decision_packet.md`
  records that project-local `nvcc` exists, target base env pythons are now
  present, but `git-lfs`, executable `cmake`, official dependency readiness,
  and official sanity are still missing. It also records that non-Curiosity
  OmniWorld/ICLR2027 hits must not be inspected or reused.
- Approved asset reuse execution:
  `experiments/reports/phase00/ref_tactile/envprep/approved_asset_reuse_execution.md`
  records the user-approved UniVTAC bundled TacEx to TaCauchy copy. TaCauchy
  now has `Sensors/GelSight_Mini/Sensor.usd` and `21` tactile test shape USDs;
  a fresh Gate review still needs to consume this post-copy availability.
- UniVTAC env create attempts:
  `experiments/reports/phase00/ref_tactile/envprep/univtac_env_create_attempts.md`
  records three approved local conda create attempts that failed with conda
  lock errors, followed by one successful `--no-lock` retry.
- Reference env create execution:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`
  records successful base Python env creation for UniVTAC (`Python 3.10.20`)
  and TaCauchy (`Python 3.11.15`). This is not official dependency readiness
  or official sanity.
- Reference dependency stage plan:
  `experiments/reports/phase00/ref_tactile/envprep/reference_dependency_stage_plan.md`
  records dry-run official dependency/sanity command staging for both UniVTAC
  and TaCauchy. No dependency installation or official sanity was run.
- Reference dependency install blocker:
  `experiments/reports/phase00/ref_tactile/envprep/reference_dependency_install_blocker.md`
  records the current hard blocker: official dependency installation/builds are
  required, but login-node heavy work and compute-node dependency installation
  are both forbidden by project rules.
- 82 FPS target evidence:
  `p00_bench_main_20260701_035529`, measured `92.6 FPS`
- Base official controller:
  `newton.examples.robot.example_robot_panda_hydro`

## Current Best Tactile Candidate

- Candidate marker/direct-force run:
  `p00_mjw_marker_v1_20260701_074200`
- Video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.avi`
- Contact sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_sheet.jpg`
- Source arrays:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_timeseries.npz`
- Summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json`
- Positive evidence:
  zero read errors, `146` pad-object contact frames, max lift
  `0.2225111573934555` m, max candidate `Fn` sum `41.90861511230469`, max
  candidate `Ft` sum `12.294239044189453`, nonzero marker-flow norms.
- Limitation:
  marker rendering is still candidate force-derived rendering, not validated
  Taccel/hardware photometric marker semantics.

## Direct-Force Validation Evidence

- MJWarp EFC force audit:
  `p00_mjw_force_audit_v2_20260701_045700`
- SensorContact alignment:
  `p00_mjw_align_v1_20260701_055200`
- Alignment result:
  best sign `shape0_negative`; force relative RMSE
  `3.2491620810680347e-08`; friction relative RMSE
  `2.0018143688320552e-07`; mean cosine `1.0`.
- Steel-spec direct-force run:
  `p00_mjw_direct_steel_v1_20260701_060500`
- Steel-spec material:
  `mu=0.3`, `kh=1e12`.

## Reference-Video Comparison Evidence

- Reference video:
  `0780e5ec3fdb26b63ae63de0f49f07c4.mp4`
- Best current comparison:
  `p00_refcmp_marker_v1_20260701_074900`
- Side-by-side sheet:
  `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_vs_candidate_sheet.jpg`
- Channel audit:
  `p00_chan_audit_v1_20260701_082100`
- Channel audit sheet:
  `experiments/visuals/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_sheet.jpg`
- Status:
  layout/channel presence is positive, but physical/photometric semantic
  equivalence is not validated.

## Gate Reviews

- Current strict gate review:
  `p00_gate_review_v5_20260701_060100`
- Summary:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v5_20260701_060100/phase00_gate_review_summary.json`
- Report:
  `experiments/reports/phase00/ref_tactile/gate_review/p00_gate_review_v5_20260701_060100/phase00_gate_review.md`
- Current gate status:
  Gate 00D open, Gate 00E open, Gate 00F open, curiosity training allowed
  `false`.
- Pending gate-review code update:
  `src/newton_tactile_curiosity/phase00_gate_review.py` now accepts
  `--reference-asset-availability-summary` and
  `--reference-asset-reuse-plan`, and the Slurm runners pass those paths.
  Syntax checks passed, but Gate review has not been rerun after this update.

## Official Semantic References

- UniVTAC:
  `external/UniVTAC`, commit
  `05bcd3edb92237107efa40105292a24f1a9fd761`
- TaCauchy:
  `external/TaCauchy`, commit
  `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
- Semantic reference matrix:
  `experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`
- Semantic bridge spec:
  `experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json`
- Gate 00F blocker:
  approved prebuilt UniVTAC/TaCauchy environments are missing.

## Environment Blocker Evidence

- Environment plan:
  `experiments/reports/phase00/ref_tactile/reference_environment_plan.md`
- Toolchain preflight:
  `experiments/reports/phase00/ref_tactile/envprep/toolchain_preflight.md`
- Location audit:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_location_audit.md`
- Location audit JSON:
  `experiments/configs/phase00/ref_tactile/envprep/reference_env_location_audit_v1.json`
- Staged environment checklist:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_stage_checklist.md`
- Staged environment checklist JSON:
  `experiments/configs/phase00/ref_tactile/envprep/reference_env_stage_checklist_v1.json`
- Asset availability audit:
  `experiments/reports/phase00/ref_tactile/envprep/reference_asset_availability.md`
- Asset availability audit JSON:
  `experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json`
- Candidate asset reuse plan:
  `experiments/reports/phase00/ref_tactile/envprep/reference_asset_reuse_plan.md`
- Candidate asset reuse plan JSON:
  `experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json`
- Asset stage runner:
  `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_asset_stage.sh`
- Asset stage status:
  `experiments/outputs/phase00/ref_tactile/envprep/assets/tacauchy/`
- Stage runner:
  `experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh`
- Stage status files:
  `experiments/outputs/phase00/ref_tactile/envprep/univtac/` and
  `experiments/outputs/phase00/ref_tactile/envprep/tacauchy/`
- Availability checker:
  `experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh`
- Latest availability status:
  `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`
- Gate 00F readiness checker:
  `experiments/configs/phase00/ref_tactile/envprep/check_gate00f_readiness.sh`
- Latest Gate 00F readiness:
  `experiments/outputs/phase00/ref_tactile/envprep/gate00f_readiness/gate00f_readiness_status.json`
- Current result:
  UniVTAC conda Python and TaCauchy conda Python are present. TaCauchy now
  contains `Sensors/GelSight_Mini/Sensor.usd` and `21` tactile test shape USD
  files. The file-presence asset/env blockers are cleared, but official
  dependency readiness and official sanity remain unresolved. The Gate 00F
  readiness checker reports `gate00f_ready=false` and
  `reason=blocked_official_sanity_or_gate_review_not_passed`; effective failed
  checks are `univtac_official_reference_sanity` and
  `tacauchy_official_reference_sanity`.
- Gate 00F dependency resolution packet:
  `experiments/reports/phase00/ref_tactile/gate00f_dependency_resolution_packet.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_dependency_resolution_packet_v1.json`.
  This records the official UniVTAC, TaCauchy/TacEx/UIPC, and IsaacLab TacSL
  dependency requirements, the allowed env/container resolution paths, and the
  disallowed login-node or compute-allocation install/build paths.
- Gate 00F runtime locator probe:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701_v1.json`.
  This lightweight probe found `/usr/bin/docker` but no `module`/`ml`,
  `singularity`/`apptainer`/`enroot`, `git-lfs`, `cmake`, `nvcc`, or
  `nvidia-smi` on the refreshed shell PATH, and it found no dependency-complete
  UniVTAC/TaCauchy/IsaacLab TacSL runtime under the shallow approved checks.
- Gate 00F shared runtime locator:
  `experiments/reports/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701_v1.json`.
  This checked common shared software/container top-level paths and Docker
  image names, found no existing dependency-complete Isaac/TacEx/TaCauchy/
  UniVTAC/TacSL/UIPC runtime or container, and stopped a slower project-local
  artifact search to avoid login-node waste.
- Gate 00F runtime preflight handoff:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_handoff.md`,
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_handoff_v1.json`,
  and
  `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`.
  This future compute-side preflight first requires the runtime registry to
  pass, then reads registered `python_env` paths or supported container
  references from the accepted registry and checks module specs for UniVTAC,
  TaCauchy, and IsaacLab TacSL before the Gate 00F bundle. Container module
  preflight supports `docker` local image IDs and
  `singularity`/`apptainer`/`sif` artifact paths; `enroot`, `sqsh`, and `tar`
  remain unsupported until explicit runners exist. It does not clear Gate 00F
  by itself.
- Gate 00F container-aware official sanity dispatch:
  `experiments/configs/phase00/ref_tactile/gate00f_container_runtime_common.sh`,
  `experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh`,
  and
  `experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh`.
  These scripts can read accepted `RUNTIME_REGISTRY` entries and dispatch
  registered docker/singularity/apptainer/sif runtimes for UniVTAC, TaCauchy,
  and IsaacLab TacSL official sanity. Failed UniVTAC/TaCauchy schema probes
  and failed TacSL official demos now write blocker summaries; TacSL keeps
  `--use_tactile_rgb` and records runtime/asset failures rather than weakening
  the command. This is glue only; a real registered runtime is still required.
- Gate 00F container dispatch login refusal:
  `experiments/reports/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_container_dispatch_login_refuse_20260701_v1.json`.
  UniVTAC/TaCauchy sanity, IsaacLab TacSL sanity, and the Gate 00F bundle still
  exit with code `2` on the login node after container dispatch support was
  added.
- Runtime preflight login-node refuse check:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701_v1.json`.
  The preflight exits with code `2` when `SLURM_JOB_ID` is missing.
- Runtime preflight login-node refuse check after container support:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_after_container_support_20260701_v1.json`.
  The preflight still exits with code `2` before registry validation, container
  commands, or module imports when `SLURM_JOB_ID` is missing.
- Gate 00F runtime registry:
  `experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`,
  `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`,
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_handoff.md`,
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registry_handoff_v1.json`.
  The registry requires UniVTAC, TaCauchy, and IsaacLab TacSL runtimes to be
  explicitly registered as `dependency_complete_registered` through allowed
  resolution paths before runtime preflight.
- Current runtime registry validation:
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_current_20260701.md`
  and
  `experiments/outputs/phase00/ref_tactile/runtime_registry/p00_registry_current_20260701/gate00f_runtime_registry_validation_summary.json`.
  Current status is `fail_gate00f_runtime_registry`: UniVTAC/TaCauchy are only
  base Python envs and IsaacLab TacSL has no registered runtime.
- Gate 00F runtime registration handoff:
  `src/newton_tactile_curiosity/gate00f_runtime_register.py`,
  `experiments/reports/phase00/ref_tactile/gate00f_runtime_registration_handoff.md`,
  and
  `experiments/configs/phase00/ref_tactile/gate00f_runtime_registration_handoff_v1.json`.
  This metadata-only helper writes a copied candidate registry for a future
  dependency-complete Python env or prebuilt container. It does not pull/build
  images, run containers, import Isaac/TacSL modules, or clear Gate 00F.
- Gate 00F scoped project artifact probe:
  `experiments/reports/phase00/ref_tactile/gate00f_project_artifact_probe_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_project_artifact_probe_20260701_v1.json`.
  Bounded project-local search found no `.sif`, `.sqsh`, `.tar`, `.tar.gz`, or
  `.img` container artifact and no dependency-complete runtime tool path under
  `envs`; only `envs/taccel/cuda-toolkit/bin/nvcc` was found.
- Gate 00F bundle preflight gate update:
  `experiments/reports/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701_v1.json`.
  The bundle now requires runtime preflight before official UniVTAC/TaCauchy/
  TacSL sanity commands run; runtime preflight itself requires the runtime
  registry to pass. Python-env and supported container runtimes both flow
  through the accepted registry; a real registered runtime is still required.
- Gate 00F TacSL source compatibility:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`
  and
  `experiments/outputs/phase00/ref_tactile/tacsl_source_compat/p00_tacsl_src_compat_20260701/tacsl_source_compat_summary.json`.
  Current status is `pass_tacsl_source_compat` for source VERSION `2.3.2` and
  candidate image ref `nvcr.io/nvidia/isaac-lab:2.3.2`; gate effect remains
  `source_compat_only_does_not_register_runtime_or_clear_gate00f`.
- Gate 00F TacSL container/doc refresh:
  `experiments/reports/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_tacsl_container_doc_refresh_20260701_v1.json`.
  Official Isaac Lab docs and NGC catalog support the TacSL/IsaacLab container
  route, while a public IsaacLab issue and local static check flag a
  `--use_tactile_rgb` risk from a missing GelSight R15 `bg.jpg` background
  asset. Do not silently drop tactile RGB to pass Gate 00F.
- Gate 00F IsaacLab upstream freshness:
  `experiments/reports/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_isaaclab_upstream_freshness_20260701_v1.json`.
  Local `external/IsaacLab_official` matches upstream `main`/`HEAD` at
  `b4c321024792976150ca55fddb26fa34480d974e`; this is source freshness only,
  not TacSL runtime evidence.
- Gate 00F reference repository freshness:
  `experiments/reports/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701.md`
  and
  `experiments/configs/phase00/ref_tactile/gate00f_reference_repo_freshness_20260701_v1.json`.
  UniVTAC, TaCauchy, and TacEx all match upstream main. Gate 00F remains
  blocked by dependency-complete runtime readiness and official sanity.

## Secondary References

- FreeTacMan:
  `external/FreeTacMan`, commit
  `9285740a5d33385d3a9cf5ccdb185e3387b547bd`
- DiffTactile:
  `external/DiffTactile`, commit
  `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
- APPLE:
  `external/APPLE`, commit
  `4b1d71fadb786d865d4ee29a184ab408b9605083`
- Tactile MNIST:
  `external/tactile-mnist`, commit
  `9e4e59139e9349ab361a3b9297f4815724ad6387`
- Reactive Diffusion Policy:
  `external/reactive_diffusion_policy`, commit
  `824c5e8de1fd1811106907a04b5f0186e0138c0b`
- ImplicitRDP:
  `external/ImplicitRDP`, commit
  `4c90646df17787e31c88838106c4a0323ddefb4a`
- Tactile Diffusion:
  `external/Tactile-Diffusion`, commit
  `16868fb96d19d93dc5837600c26b48415632e4f6`
- Curiosity reference matrix:
  `experiments/configs/phase00/ref_tactile/curiosity_reference_matrix_v1.json`
- Policy/photometric reference audit:
  `experiments/reports/phase00/ref_tactile/policy_reference_audit.md`
- Rule:
  these are secondary reference/design sources. They do not close Gate 00D,
  Gate 00E, or Gate 00F.

## Next Faithful Action

Prepare or locate approved dependency-complete shared-filesystem environments
or prebuilt containers for UniVTAC, TaCauchy, and IsaacLab TacSL. For the
container route, follow the acquisition plan, then register exact runtime paths
in the runtime registry, validate the registry, run runtime preflight, run the
Gate 00F reference bundle, and run strict acceptance in a Curiosity tmux-held
Slurm allocation. Do not install dependencies on compute nodes. Do not start
curiosity training before Gate 00D/00E/00F pass or the user accepts faithful
blockers.
