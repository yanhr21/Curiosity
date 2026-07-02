# Gate 00F TacSL Source Compatibility Handoff

- Date: `2026-07-01`
- Classification: `source_compat_handoff_not_runtime_not_gate_completion`
- Validator:
  `src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py`
- Source repo: `external/IsaacLab_official`
- Expected version: `2.3.2`
- Candidate image ref: `nvcr.io/nvidia/isaac-lab:2.3.2`

Run the static source compatibility check with:

```bash
python src/newton_tactile_curiosity/gate00f_tacsl_source_compat_validate.py \
  --repo external/IsaacLab_official \
  --expected-version 2.3.2 \
  --expected-image-ref nvcr.io/nvidia/isaac-lab:2.3.2 \
  --output-json experiments/outputs/phase00/ref_tactile/tacsl_source_compat/<RUN_TAG>/tacsl_source_compat_summary.json
```

This check validates only source metadata and required TacSL fields/flags. It
does not run Isaac Sim, import Isaac Lab, launch containers, load models, run
simulation, or install dependencies.

Passing this check means the local source is compatible with the candidate
version metadata. It does not register a runtime and does not clear Gate 00F.
