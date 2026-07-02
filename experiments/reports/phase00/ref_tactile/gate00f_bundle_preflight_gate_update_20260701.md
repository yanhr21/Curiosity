# Gate 00F Bundle Preflight Gate Update

- Date: `2026-07-01`
- Classification: `bundle_guard_update_not_training_not_gate_completion`

The Gate 00F bundle now requires runtime preflight before official reference
sanity commands run. Runtime preflight itself requires the runtime registry to
pass, so the current default execution order is:

1. Runtime registry validation.
2. Runtime preflight.
3. UniVTAC official reference sanity.
4. TaCauchy official reference sanity.
5. IsaacLab TacSL official demo sanity.
6. Gate 00F review.
7. Strict bundle acceptance.

Updated files:

- `experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh`
- `experiments/configs/phase00/ref_tactile/launch_gate00f_reference_bundle_tmux.sh`
- `experiments/configs/phase00/ref_tactile/gate00f_reference_bundle_handoff_v1.json`
- `experiments/reports/phase00/ref_tactile/gate00f_reference_bundle_handoff.md`

If runtime preflight does not report `pass_gate00f_runtime_preflight`, the
bundle writes `fail_gate00f_bundle_runtime_preflight_not_passed` and exits
before UniVTAC/TaCauchy/TacSL sanity commands run.

`REQUIRE_RUNTIME_PREFLIGHT=0` remains available only as an explicit diagnostic
escape hatch. It must not be treated as Gate 00F acceptance evidence.

This update does not clear Gate 00F. It prevents direct bundle execution from
bypassing the registry and runtime preflight gates.
