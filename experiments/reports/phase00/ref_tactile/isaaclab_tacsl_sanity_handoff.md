# IsaacLab TacSL Sanity Handoff

Date: 2026-07-01

Classification: handoff only. This has not been run, does not load a model,
does not simulate, does not train, and does not close Gate 00F.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/isaaclab_tacsl_sanity_handoff_v1.json`

## Source

- Official repo: `https://github.com/isaac-sim/IsaacLab.git`
- Local sparse checkout: `external/IsaacLab_official`
- Expected commit: `b4c321024792976150ca55fddb26fa34480d974e`
- Demo entrypoint: `scripts/demos/sensors/tacsl_sensor.py`

## Scripts

- Run inside allocation:
  `experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh`
- Launch in tmux-held allocation:
  `experiments/configs/phase00/ref_tactile/launch_isaaclab_tacsl_sanity_tmux.sh`

Both scripts pass `bash -n`.

## Required Environment

The launcher looks for:

- `ISAACLAB_TACSL_PYTHON`
- `envs/isaaclab_tacsl/conda/bin/python`
- `envs/isaaclab_tacsl/.venv/bin/python`

Current status: no approved dependency-complete IsaacLab/TacSL environment is
known. The launcher refuses to consume a GPU allocation when the environment is
missing unless `ALLOW_MISSING_TACSL_ENV_BLOCKER_RUN=1` is explicitly set to
record a compute-side blocker.

## Official Command

The prepared official sanity command is:

```bash
scripts/demos/sensors/tacsl_sensor.py \
  --headless \
  --enable_cameras \
  --use_tactile_rgb \
  --use_tactile_ff \
  --normal_contact_stiffness 1.0 \
  --tangential_stiffness 0.1 \
  --friction_coefficient 2.0 \
  --contact_object_type nut \
  --num_envs 1 \
  --save_viz
```

Expected fields are `tactile_rgb_image`, `tactile_depth_image`,
`penetration_depth`, `tactile_normal_force`, and `tactile_shear_force`.

## Gate Effect

This prepares the official TacSL sanity path. It does not clear Gate 00F.
Curiosity training remains disallowed until official semantic validation runs
or a faithful blocker is accepted.
