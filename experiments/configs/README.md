# Experiment Configs

Post-pivot configs and launch scripts for Newton-native adaptation experiments.

Phase 02 no-adaptation baseline entry points:

```text
experiments/configs/lift_hold_no_adaptation_baseline_v1.json
experiments/configs/lift_hold_metrics_schema_v1.json
experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
experiments/configs/extract_lift_hold_metrics.py
experiments/configs/launch_lift_hold_metrics_tmux.sh
```

The baseline launcher must be run only from a lightweight login-node shell and
must target an existing Curiosity-specific tmux-held Slurm allocation via
`JOB_ID` and `TMUX_SESSION`. The actual Newton simulation/rendering runs inside
the allocation through `srun --jobid ... --overlap`.

Historical configs are in:

```text
legacy/2026-06-26_pre_pivot_archive/experiments/configs/
```
