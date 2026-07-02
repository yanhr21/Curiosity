# Phase 00 Requirement Status

Date: 2026-07-01

This is a requirement audit, not a completion report. It exists to prevent the
current candidate tactile assets from being mistaken for final simulator,
base-model, or curiosity-training success.

Machine-readable status:
`experiments/configs/phase00/ref_tactile/phase00_requirement_status_v1.json`

## Current Gate State

- Gate 00D reference diagnostic: open, reference semantics blocked.
- Gate 00E base model: open, tactile validation blocked.
- Gate 00F official semantic validation: open, official reference environments
  missing.
- Gate 00G curiosity readiness: design references only; training disallowed.

## Requirement Summary

- Idea/agent/plan/todo persistence: active records exist, must stay updated.
- Latest codebase audit: expanded and source-backed, with some acquisition
  gaps still recorded. Latest source refresh shows Newton upstream main has
  advanced beyond the old active evidence worktree; `external/newton_d58` now
  has compute-side runtime, candidate tactile, reference-comparison, channel
  audit, and Gate-review evidence. A later source-only recheck shows Newton
  upstream main is now `8c501b47847569fecdda97a9f7f01205c6f7964f` with
  `external/newton_8c501` prepared and tested twice on H200. Both runs are
  acceptable around 80 FPS, so FPS is not a blocker for 8c501 dense tactile
  export. The same recheck now has IsaacLabTactile source acquired
  locally, but LFS asset completeness and official runtime sanity are not
  verified.
- Steel contact mechanics: positive candidate direct-force evidence exists,
  but semantics are not validated.
  The Gate 00D environment evidence audit
  `experiments/reports/phase00/ref_tactile/gate00d_environment_evidence_audit_20260701.md`
  classifies the d58 environment as
  `partial_positive_environment_candidate_reference_semantics_blocked`;
  contact area is proxy-only and dense penetration/compression semantics are
  not validated.
- Base model/controller: official Newton Panda hydro on latest Newton main has
  older 92.6 FPS evidence on the `a217e55...` chain and latest-upstream d58 now
  has `82.7 FPS` runtime evidence plus candidate grasp/lift tactile export,
  reference comparison, channel audit, and Gate review. The base gate is not
  closed because tactile semantics are still unvalidated.
  The Gate 00E d58 base evidence audit
  `experiments/reports/phase00/ref_tactile/base_evidence/p00_gate00e_d58_audit_20260701/gate00e_base_evidence_audit.md`
  classifies d58 as
  `partial_positive_gate00e_base_candidate_tactile_validation_blocked`, not as
  base completion.
- Official semantic validation: base UniVTAC/TaCauchy env prefixes exist, but
  official dependency readiness and official sanity are still blocked.
- Curiosity training: intentionally not started.

## Current Hard Blocker

The active blocker is Gate 00F. `reference_env_availability_status.json`
currently shows UniVTAC and TaCauchy conda Python paths present, with
`gate_00f_ready=candidate_envs_present_pending_compute_sanity`. The current
login PATH still lacks `git-lfs`, `cmake`, and `nvcc`.

The location audit
`experiments/reports/phase00/ref_tactile/envprep/reference_env_location_audit.md`
also found no approved prebuilt target environment in project `envs/` or common
home conda/env locations. Existing non-target Newton/Taccel/T-Rex and
autoresearch venvs are not accepted as UniVTAC/TaCauchy official reference
environments.

The staged environment checklist
`experiments/reports/phase00/ref_tactile/envprep/reference_env_stage_checklist.md`
records the controlled preparation order. It keeps UniVTAC and TaCauchy
separate because UniVTAC targets Isaac Sim 4.5 / Isaac Lab 2.1.1 while
TaCauchy targets Isaac Sim 5.0 / Isaac Lab 2.2.1. Project-local
`envs/taccel/miniforge/bin/conda` exists. The target base envs now also exist,
but official dependency installation and sanity remain blocked.
The focused environment blocker audit
`experiments/reports/phase00/ref_tactile/envprep/reference_env_blocker_audit.md`
records the current blocker after base env creation: target env pythons are
present, but official UniVTAC/TaCauchy dependencies are not installed and
official sanity has not passed.
The Gate 00F decision packet
`experiments/reports/phase00/ref_tactile/envprep/gate00f_decision_packet.md`
adds the current decision boundary: project-local `nvcc` exists at
`envs/taccel/cuda-toolkit/bin/nvcc` with CUDA `12.8`; target reference env
pythons are present; `git-lfs` and executable `cmake` are still missing from
the checked PATH.
Non-Curiosity OmniWorld/ICLR2027 hits were observed but not inspected or reused.
The repeatable readiness checker
`experiments/configs/phase00/ref_tactile/envprep/check_gate00f_readiness.sh`
now writes
`experiments/outputs/phase00/ref_tactile/envprep/gate00f_readiness/gate00f_readiness_status.json`
and
`experiments/reports/phase00/ref_tactile/envprep/gate00f_readiness.md`.
Current result is `gate00f_ready=false` with
`reason=blocked_official_sanity_or_gate_review_not_passed`.
After the user said `全都允许继续`, the approved asset reuse path was executed.
`experiments/reports/phase00/ref_tactile/envprep/approved_asset_reuse_execution.md`
records that `273` files were created, `429244198` bytes were transferred, the
TaCauchy asset tree is now `412M`, `Sensors/GelSight_Mini/Sensor.usd` is
present, and the tactile test shape USD count is `21`. This clears the asset
file-presence blocker, but official sanity remains blocked.

The asset audit
`experiments/reports/phase00/ref_tactile/envprep/reference_asset_availability.md`
now records post-copy file presence: UniVTAC bundled TacEx assets are present,
TaCauchy `Sensors/GelSight_Mini/Sensor.usd` is present, and TaCauchy tactile
test shape USD count is `21`. This is not official TaCauchy sanity, but it
does remove the previous file-presence asset blocker.
The reuse plan
`experiments/reports/phase00/ref_tactile/envprep/reference_asset_reuse_plan.md`
records that the local copy path from UniVTAC bundled TacEx to TaCauchy was
approved and executed.
The guarded asset runner
`experiments/configs/phase00/ref_tactile/envprep/prepare_reference_asset_stage.sh`
now verifies the post-copy file-presence state in dry-run mode.
The focused asset blocker audit
`experiments/reports/phase00/ref_tactile/envprep/tacauchy_asset_blocker_audit.md`
remains useful as pre-copy negative evidence: official TaCauchy setup required
`git-lfs`, which is not on the current PATH. Its old missing-file counts have
been superseded by `approved_asset_reuse_execution.md` and
`reference_asset_availability.md`.
The reference env creation execution record
`experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`
records successful base Python env creation for UniVTAC (`Python 3.10.20`) and
TaCauchy (`Python 3.11.15`). This clears file-level env availability, but not
official dependency readiness or official sanity.
The reference dependency stage plan
`experiments/reports/phase00/ref_tactile/envprep/reference_dependency_stage_plan.md`
records dry-run official dependency and sanity commands for both reference
repos. No dependency installation or official sanity was run.
The reference dependency install blocker
`experiments/reports/phase00/ref_tactile/envprep/reference_dependency_install_blocker.md`
records the current hard blocker: official dependency installation/builds are
required, but login-node heavy work and compute-node dependency installation
are both forbidden by project rules.
The UniVTAC env creation audit
`experiments/reports/phase00/ref_tactile/envprep/univtac_env_create_attempts.md`
records three approved local conda create attempts that failed with conda lock
errors, followed by one successful `--no-lock` retry.
The latest source recheck V3
`experiments/reports/phase00/ref_tactile/latest_reference_recheck_20260701_v3.md`
records that Newton upstream main has advanced to
`8c501b47847569fecdda97a9f7f01205c6f7964f`; `external/newton_8c501` is a
source-only worktree and has not passed H200 runtime sanity, dense tactile
export, reference comparison, channel audit, or Gate review. The same recheck
records the corrected official source URLs for Taccel, TaCauchy, HydroShear,
TacEx, and IsaacLabTactile; `external/TacEx` is cloned, and
`external/IsaacLabTactile` is now cloned with LFS skipped. `git-lfs` is not
available on the checked PATH, so IsaacLabTactile asset completeness and
official sanity remain unverified.
The Newton 8c501 compute handoff
`experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md`
records exact tmux-held Slurm commands for runtime benchmark, dense tactile
export, reference comparison, channel audit, and Gate review. This is a
handoff only and has not been executed.
The Newton 8c501 allocation request
`experiments/reports/phase00/ref_tactile/newton_8c501_allocation_request.md`
records job `160854` for the 8c501 sanity path. Its initial state was
`PENDING (Priority)` before the benchmark windows were launched.
The Newton 8c501 benchmark status
`experiments/reports/phase00/ref_tactile/newton_8c501_benchmark_status.md`
records two successful H200 executions on job `160854`: `80.1 FPS` and
`80.8 FPS`. Both are acceptable around 80 FPS, so `8c501...` should advance to
dense tactile export/reference comparison/channel audit/Gate review when a
Curiosity tmux-held Slurm allocation is available.
The Newton 8c501 continuation chain status
`experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`
records that this continuation chain has now run on job `160924`: dense tactile
export passed, reference comparison passed, channel audit passed, and Gate
review remains `open_not_curiosity_ready` only because official UniVTAC,
TaCauchy, and IsaacLab TacSL sanity plus validated photometric/real-area
semantics are still missing.
The Gate 00F readiness refresh
`experiments/reports/phase00/ref_tactile/gate00f_readiness_refresh_20260701.md`
records the latest lightweight readiness state: candidate env pythons and
copied assets are present, but `gate00f_ready=false`; effective failed checks
remain `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity`. IsaacLabTactile source is cloned, but LFS
asset completeness is not verified.
The Gate 00F tool lookup
`experiments/reports/phase00/ref_tactile/gate00f_tool_lookup_20260701.md`
records that PATH still lacks `git-lfs`, `cmake`, `nvcc`, and `nvidia-smi`.
Project-local lookup found only `envs/taccel/cuda-toolkit/bin/nvcc`, and no
prebuilt Isaac/Lab/TacEx/UIPC env directories were found under `envs` at max
depth 4.
The Gate 00F static source audit
`experiments/reports/phase00/ref_tactile/gate00f_static_source_audit_20260701.md`
records the exact source-level reason the official semantic gate is still
open. UniVTAC supplies the official left/right GelSight Mini tactile schema
(`rgb_marker`, `marker`, `depth`, `rgb`, and `pose`) and manipulation benchmark
baseline context, but requires dependency-complete official sanity. TaCauchy
supplies the required Cauchy-stress, normal-pressure, tangential-traction, and
GelSight optical/marker semantics, but requires Isaac Sim/Lab, CMake/vcpkg,
UIPC/libuipc, asset, and demo sanity readiness. The local IsaacLabTactile clone
currently appears to be generic Isaac Lab/contact-sensor source without an
identified asset-complete TacSL/GelSight/TacEx entrypoint, so it cannot replace
UniVTAC/TaCauchy validation.
The Gate 00F module/env probe
`experiments/reports/phase00/ref_tactile/gate00f_module_env_probe_20260701.md`
records that the current shell has no `module` or `ml` command and that shallow
file-name probing under the existing UniVTAC/TaCauchy base env prefixes found
no Isaac, TacEx, UIPC, cuRobo, or Torch component names. This does not prove no
cluster module exists under every shell setup, but it reinforces that the
current base env prefixes are not dependency-complete official reference envs.
The Gate 00F container path audit
`experiments/reports/phase00/ref_tactile/gate00f_container_path_audit_20260701.md`
records that Docker build/helper paths exist for TacEx/TaCauchy/IsaacLabTactile,
but no approved prebuilt Curiosity image/SIF/tar artifact was found. `docker`
exists on current PATH, while `singularity`, `apptainer`, `enroot`, and
`podman` do not. The discovered official container paths still require image
build/setup or placeholder cluster SIF configuration, and a read-only
`docker images` query returned no Isaac/TacEx/TaCauchy/UniVTAC-related image
names, so they do not clear Gate 00F.
The latest 2026-07-01 web/codebase refresh
`experiments/reports/phase00/ref_tactile/latest_20260701_web_codebase_refresh.md`
adds official Isaac Lab main TacSL source at `external/IsaacLab_official`,
FTP-1 at `external/ftp1-policy`, and AnyTouch2 at `external/AnyTouch2`.
Official IsaacLab TacSL is now a Gate 00F candidate because it exposes tactile
RGB/depth, penetration, normal force, and shear force fields, but it still
requires official environment/assets and compute-side sanity. FTP-1 and
AnyTouch2 are future serious policy/representation references, not current
Gate completion evidence.
The latest supplementary codebase audit
`experiments/reports/phase00/ref_tactile/latest_supplementary_codebase_audit_20260701.md`
adds `external/TactSim-IsaacLab` as a secondary photometric
GelSight/DIGIT-style tactile simulation reference, `external/newton-actuators`
as deprecated Newton actuator background only, and UniT as remote-head-only
future tactile representation evidence. This is source audit only: no runtime
was registered, no official demo was run, no model/checkpoint was loaded, and
Gate 00D/00E/00F remain open.
The latest source freshness V4 audit
`experiments/reports/phase00/ref_tactile/latest_source_freshness_20260701_v4.md`
confirms with `git ls-remote` that the tracked official refs for Newton,
Taccel, T-Rex, IsaacLab, TacEx, TaCauchy, UniVTAC, FTP-1, AnyTouch2, and
HydroShear still match current records. It does not change the runtime
decision: Newton 8c501 is latest source and its around-80-FPS runtime is
acceptable for continuing dense tactile export; d58 remains stronger historical
runtime/tactile evidence until the 8c501 downstream evidence chain exists.
The latest policy/checkpoint refresh
`experiments/reports/phase00/ref_tactile/latest_policy_checkpoint_refresh_20260701.md`
adds clean official T-Rex snapshots at `external/T-Rex_43ff` and
`external/T-Rex_full_b23`, records the official T-Rex pretrain/midtrain
checkpoint names, and records FTP-1, AnyTouch2, and Sparsh as future serious
checkpoint/representation references. The post-Gate checklist
`experiments/reports/phase00/ref_tactile/post_gate00f_policy_bridge_checklist.md`
defines the required T-Rex data contract and ablations. This is planning and
source availability evidence only; no checkpoint was downloaded, loaded, or
trained.
The T-Rex data-contract extraction
`experiments/reports/phase00/ref_tactile/trex_data_contract.md` and validator
`src/newton_tactile_curiosity/trex_contract_validate.py` define the future
metadata gate for Newton-to-T-Rex conversion. It requires slow/fast RGB streams,
`observation.state [62]`, `action [16,62]`, `action_abs [62]`,
`observation.tactile_f6 [10,6]`, ten tactile-deform video streams, and
normalization stats. This confirms current Newton Panda evidence is not yet
T-Rex-compatible.
The IsaacLab TacSL sanity handoff
`experiments/reports/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff.md`
prepares the official `tacsl_sensor.py` sanity command and tmux-held Slurm
launch scripts. It has not been run and remains blocked by the missing
approved dependency-complete IsaacLab/TacSL environment or prebuilt container.
The unified Gate 00F reference bundle handoff
`experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`
now provides one allocation workflow for UniVTAC sanity, TaCauchy sanity,
IsaacLab TacSL sanity, and Gate review with fixed summary paths. The launcher
safety check
`experiments/reports/phase00/ref_tactile/gate00f_bundle_launcher_reflex_refuse_check_20260701.md`
confirmed it refuses Slurm job `160860` because the job workdir is
`/public/home/yanhongru/ICLR2027/Reflex`.
The Gate 00F bundle acceptance handoff
`experiments/reports/phase00/ref_tactile/gate00f_bundle_acceptance_handoff.md`
and validator `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py`
require the three official sanity pass statuses plus
`pass_official_semantic_reference_sanity`, blocker sanity disabled, and no
failed checks or hard blockers before any future bundle can be accepted.
The Gate 00F dependency resolution packet
`experiments/reports/phase00/ref_tactile/gate00f_dependency_resolution_packet.md`
records the concrete dependency gap. UniVTAC needs the Isaac Sim 4.5 /
Isaac Lab 2.1.1 / TacEx / cuRobo / UIPC path. TaCauchy needs the Isaac Sim
5.0 / Isaac Lab 2.2.1 / TacEx assets / UIPC-lib path with vcpkg, CMake, GCC,
and CUDA readiness. IsaacLab TacSL needs a dependency-complete official Isaac
Lab/TacSL environment. The only accepted resolution paths are existing
dependency-complete envs, existing prebuilt containers, or a compliant
non-login/non-experiment env-prep workflow followed by bundle acceptance.
The Gate 00F runtime locator probe
`experiments/reports/phase00/ref_tactile/gate00f_runtime_locator_probe_20260701.md`
found `/usr/bin/docker` but no `module`/`ml`, `singularity`/`apptainer`/
`enroot`, `git-lfs`, `cmake`, `nvcc`, or `nvidia-smi` on the refreshed shell
PATH. It found UniVTAC and TaCauchy base Python prefixes, but no
dependency-complete UniVTAC/TaCauchy/IsaacLab TacSL runtime.
The Gate 00F shared runtime locator
`experiments/reports/phase00/ref_tactile/gate00f_shared_runtime_locator_20260701.md`
checked common shared software/container top-level paths and Docker image names
and found no existing dependency-complete Isaac/TacEx/TaCauchy/UniVTAC/TacSL/
UIPC runtime or container. The runtime preflight handoff
`experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_handoff.md`
adds a future compute-side metadata check that first requires runtime registry
acceptance, then checks Python executability and module specs before the Gate
00F bundle. It does not clear Gate 00F by itself.
The login-node refuse check
`experiments/reports/phase00/ref_tactile/gate00f_runtime_preflight_login_refuse_20260701.md`
confirms this preflight exits with code `2` when `SLURM_JOB_ID` is missing.
The Gate 00F runtime registry
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_handoff.md`
and validator `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`
add a required registration layer before runtime preflight. Current validation
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_current_20260701.md`
fails as expected because UniVTAC/TaCauchy are only base Python envs and
IsaacLab TacSL has no registered runtime path.
The runtime registration handoff
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registration_handoff.md`
adds a metadata-only helper for future real runtime registration into a copied
candidate registry. It rejects placeholder values and excluded resource-zone
paths, but it does not pull/build images, run containers, import Isaac/TacSL
modules, install dependencies, or clear Gate 00F. Container registrations now
also require a matching `pass_gate00f_container_provenance` summary before a
candidate registry can be written.
The scoped project artifact probe
`experiments/reports/phase00/ref_tactile/gate00f_project_artifact_probe_20260701.md`
found no `.sif`, `.sqsh`, `.tar`, `.tar.gz`, or `.img` container artifact
under bounded project-local paths. It also found no `cmake`, `git-lfs`,
`singularity`, `apptainer`, or `docker` file under `envs` at max depth `4`;
only `envs/taccel/cuda-toolkit/bin/nvcc` was present.
The Gate 00F bundle preflight gate update
`experiments/reports/phase00/ref_tactile/gate00f_bundle_preflight_gate_update_20260701.md`
now requires runtime preflight to pass before official UniVTAC/TaCauchy/TacSL
sanity commands run. If preflight does not pass, the bundle writes
`fail_gate00f_bundle_runtime_preflight_not_passed` and exits before official
sanity.
The Gate 00F container acquisition plan
`experiments/reports/phase00/ref_tactile/gate00f_container_acquisition_plan_20260701.md`
records that official Isaac Sim and Isaac Lab containers exist, including
`nvcr.io/nvidia/isaac-lab:2.3.2` as a current IsaacLab candidate after
compatibility checks. TacEx/UniVTAC and TaCauchy still require project image
layers over an Isaac Lab base image or an existing prebuilt project image, so
the plan does not clear Gate 00F.
The runtime registry container-support update
`experiments/reports/phase00/ref_tactile/gate00f_runtime_registry_container_support_update_20260701.md`
now requires container registrations to include a supported runtime plus a
local `image_id` or existing shared `artifact_path`. Remote `image_ref` alone
is acquisition evidence only and cannot pass registry validation.
The container provenance contract
`experiments/reports/phase00/ref_tactile/gate00f_container_provenance_contract.md`
now defines the minimum evidence before a future container can enter even a
copied candidate runtime registry: official source commit, expected modules,
real provenance paths, and local `image_id` or existing `artifact_path`.
The negative-control summary
`experiments/outputs/phase00/ref_tactile/container_provenance/p00_isaaclab_ref_only_20260701/container_provenance_validation_summary.json`
confirms that `nvcr.io/nvidia/isaac-lab:2.3.2` as a remote `image_ref` alone
fails validation and cannot be treated as a runtime.
The runtime intake chain
`experiments/reports/phase00/ref_tactile/gate00f_runtime_intake_chain_handoff.md`
now composes provenance validation, copied-registry registration, and
copied-registry validation. The remote-image-only negative control stops at
`fail_container_provenance` and writes no candidate registry.
The TacSL source compatibility check
`experiments/reports/phase00/ref_tactile/gate00f_tacsl_source_compat_current_20260701.md`
passes for local IsaacLab VERSION `2.3.2`, candidate image ref
`nvcr.io/nvidia/isaac-lab:2.3.2`, required TacSL data fields, demo flags, and
imports. This strengthens the IsaacLab TacSL container candidate, but it is
source-only evidence: no container was pulled, no image was built, no module
was imported, no Isaac Sim process ran, no runtime was registered, and Gate
00F remains open.
The Gate 00F review code now also treats official IsaacLab TacSL as a hard
semantic-sanity condition: future reviews require `OfficialIsaacLabTacSL` in
the semantic matrix, require
`candidate.newton_mjw.penetration_or_compression` in the bridge spec, and
require a compute-side TacSL summary with status
`pass_official_isaaclab_tacsl_demo_exited_zero` before TacSL can clear official
reference sanity.
The TacSL env/container blocker refresh
`experiments/reports/phase00/ref_tactile/isaaclab_tacsl_env_blocker_refresh_20260701.md`
records that Slurm job `160860` is Reflex-owned by
`WorkDir=/public/home/yanhongru/ICLR2027/Reflex` and cannot be reused. It also
records that no `envs/isaaclab_tacsl` prefix or Curiosity-local TacSL/Isaac/
TacEx prebuilt container archive was found in limited checks.
The Gate review code now consumes asset availability and reuse-plan evidence,
and `p00_gate_d58_marker_v1_20260701_071843` reran it on the d58 evidence
chain. The result is still `open_not_curiosity_ready`: passed checks include
runtime 82 FPS, base grasp/lift final test, steel material, Fn/Ft,
SensorContact alignment, normal/area proxy, marker-style render, reference
comparison assets, channel layout audit, semantic matrix, bridge spec, and
asset reuse plan availability. The raw pre-copy/pre-env failed checks included
`reference_asset_availability` and `reference_env_availability`, but the
current readiness checker removes both from the effective file-presence
failures. Remaining effective failed checks are
`univtac_official_reference_sanity` and `tacauchy_official_reference_sanity`.

Until this is solved or accepted as a faithful blocker, no curiosity-training
success claim is allowed.

## Next Action

Prepare or locate approved local shared-filesystem environments for UniVTAC,
TaCauchy, and IsaacLab TacSL, or approved prebuilt Curiosity-owned containers,
then rerun official reference sanity in a Curiosity tmux-held Slurm allocation.
Do not install dependencies on compute nodes. Do not load T-Rex/FTP-1/AnyTouch2
checkpoints until Gate 00D/00E/00F and the post-Gate data-contract
preconditions are satisfied.
