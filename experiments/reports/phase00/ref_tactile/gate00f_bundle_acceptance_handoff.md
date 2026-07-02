# Gate 00F Bundle Acceptance Handoff

- created_at: `2026-07-01`
- classification: `acceptance_handoff_not_training_not_gate_completion`
- validator: `src/newton_tactile_curiosity/gate00f_bundle_acceptance.py`

After the Gate 00F reference bundle runs, validate the generated bundle summary
with:

```bash
python src/newton_tactile_curiosity/gate00f_bundle_acceptance.py \
  --bundle-summary experiments/outputs/phase00/ref_tactile/gate00f_bundle/<RUN_TAG>/gate00f_reference_bundle_summary.json \
  --output-json experiments/outputs/phase00/ref_tactile/gate00f_bundle/<RUN_TAG>/gate00f_bundle_acceptance_summary.json
```

Required statuses:

- `univtac_status`: `pass_official_schema_probe`
- `tacauchy_status`: `pass_official_schema_probe`
- `isaaclab_tacsl_status`: `pass_official_isaaclab_tacsl_demo_exited_zero`
- `gate00f_status`: `pass_official_semantic_reference_sanity`

Additional acceptance requirements:

- `ALLOW_BLOCKER_SANITY` must not be enabled.
- `gate_review_summary.failed_checks` must be empty.
- `gate_review_summary.hard_blockers` must be empty.
- Gate review must not directly enable curiosity training.

This validator only reads JSON summaries. It does not run training, simulation,
rendering, model loading, or dataset conversion.
