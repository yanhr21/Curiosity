# Gate 00F Reference Bundle Handoff

- created_at: `2026-07-01`
- classification: `handoff_not_training_not_gate_completion`

This handoff creates a single orchestration path for official Gate 00F sanity:

1. Runtime registry validation.
2. Runtime preflight.
3. UniVTAC official reference sanity.
4. TaCauchy official reference sanity.
5. IsaacLab TacSL official demo sanity.
6. Gate 00F review using the three generated summary paths.

The Gate review stage now defaults to the latest positive Newton `8c501`
candidate tactile evidence chain:

- `p00_bench_8c501_hot_r2_v1_20260701_162800`
- `p00_mjw_8c501_cont_20260701_1924`
- `p00_refcmp_8c501_cont_20260701_1925`
- `p00_chan_8c501_cont_20260701_1926`

These defaults can still be overridden with `BENCHMARK_SUMMARY`,
`CANDIDATE_SUMMARY`, `REFERENCE_COMPARE_SUMMARY`, `CHANNEL_AUDIT_SUMMARY`, and
`ALIGNMENT_SUMMARY`.

Scripts:

- `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`
- `experiments/configs/phase00/ref_tactile/launch_gate00f_reference_bundle_tmux.sh`

The launcher refuses a Slurm job whose workdir is not under
`/public/home/yanhongru/Curiosity`, so Reflex-owned allocations cannot be
reused by accident.

The run script now requires runtime preflight to pass before official reference
sanity commands run. Runtime preflight itself requires the runtime registry
`experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json`
to pass unless `REQUIRE_RUNTIME_PREFLIGHT=0` is explicitly set for a documented
diagnostic path.

When using a copied candidate runtime registry, pass
`RUNTIME_REGISTRY=/path/to/accepted_candidate_runtime_registry.json` to the
bundle. The bundle forwards the same registry to runtime preflight and to the
UniVTAC, TaCauchy, and IsaacLab TacSL official sanity sub-scripts.

Container-aware official sanity runners are wired for UniVTAC, TaCauchy, and
IsaacLab TacSL. They use the same accepted `RUNTIME_REGISTRY` as runtime
preflight and dispatch registered `docker`, `singularity`, `apptainer`, or
`sif` runtimes through the shared container helper. This still does not make a
container available; a real registered runtime and compute-side execution are
required before any pass claim.

Expected pass statuses:

- UniVTAC: `pass_official_schema_probe`
- TaCauchy: `pass_official_schema_probe`
- IsaacLab TacSL: `pass_official_isaaclab_tacsl_demo_exited_zero`

This is not training and not curiosity success. The bundle does not complete
Gate 00F unless the generated summaries contain the required pass statuses and
the downstream Gate review passes.
