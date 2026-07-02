# Gate 00F Post-8c501 Runtime Acceptance Handoff

- Date: `2026-07-01`
- Classification: `runtime_acceptance_handoff_not_training_not_gate_completion`

## Current State

The latest Newton `8c501...` candidate chain has produced enough candidate
evidence for future Gate 00F bundle attempts. Do not rerun Newton candidate
export just to chase the old 82 FPS reference. Around 80 FPS is accepted for
continuation, and the active blocker is official reference runtime readiness.

Current runtime registry validation remains:
`fail_gate00f_runtime_registry`.

Current Gate review remains:
`open_not_curiosity_ready`, with Gate 00F blocked by:

- `univtac_official_reference_sanity`
- `tacauchy_official_reference_sanity`
- `isaaclab_tacsl_official_reference_sanity`

## Latest Candidate Evidence To Reuse

- Newton benchmark:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_8c501_hot_r2_v1_20260701_162800/newton_hydro_benchmark_summary.json`
- Dense candidate tactile export:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_summary.json`
- Reference comparison:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_video_compare_summary.json`
- Channel audit:
  `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_8c501_cont_20260701_1926/channel_semantic_audit_summary.json`
- Current Gate review:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_8c501_cont_20260701_1927/phase00_gate_review_summary.json`

## Required Acceptance Order

1. Register real dependency-complete UniVTAC, TaCauchy, and IsaacLab TacSL
   runtimes into a copied runtime registry.
2. Validate the copied registry with
   `src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py`.
3. Run `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`
   inside a Curiosity-owned tmux-held Slurm allocation.
4. Run
   `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`
   against the latest 8c501 candidate evidence.
5. Run strict acceptance with
   `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py`.

When using a copied candidate runtime registry, pass it to both preflight and
bundle with `RUNTIME_REGISTRY=/path/to/accepted_candidate_runtime_registry.json`.
Container-aware official sanity runners are now wired for UniVTAC, TaCauchy,
and IsaacLab TacSL through the same accepted registry, and the bundle forwards
that registry to every stage. A real registered runtime and compute-side
execution are still required before any pass claim.

## Forbidden Shortcuts

- Do not treat base UniVTAC/TaCauchy Python envs as dependency-complete.
- Do not use `ALLOW_BLOCKER_SANITY=1` for an accepted bundle.
- Do not treat a remote container `image_ref` as a local runtime.
- Do not use non-Curiosity Reflex/OpenPI/Cosmos sessions, allocations, logs,
  or images.
- Do not install or build dependencies on login nodes or inside experiment
  compute allocations.
- Do not start curiosity training until Gate 00D, Gate 00E, and Gate 00F pass
  or faithful blockers are explicitly accepted by the user.

## Container Runtime Note

The registry can record future prebuilt containers only after provenance
validation passes. Runtime preflight now supports registered `docker`
containers via local `image_id` and registered `singularity`/`apptainer`/`sif`
containers via existing `artifact_path`, using module-spec checks only.
Registered `enroot`, `sqsh`, or `tar` runtimes still require an explicit
preflight runner. Docker, singularity, apptainer, and sif runtimes can feed the
container-aware official sanity runners after they are registered and pass
runtime preflight.
