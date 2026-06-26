# Phase 02 No-Adaptation Nominal Baseline

Run tag:

```text
lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210
```

## Purpose

Run the first Phase 02 short-term infant-prior baseline:

```text
official Newton Panda hydro scripted grasp/lift, no adaptation
```

This is not a learned policy, not curiosity, not T-Rex schema promotion, and
not a pretrained checkpoint result.

## Command

Launched from the login node as a lightweight tmux/srun submission into the
existing Curiosity allocation:

```text
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210 \
WINDOW_NAME=phase02_noadapt_baseline \
bash experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

The actual Newton simulation/rendering ran inside Slurm allocation `154023` on
`server56`.

## Configuration

```text
experiments/configs/lift_hold_no_adaptation_baseline_v1.json
experiments/configs/lift_hold_metrics_schema_v1.json
experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh
```

Runtime settings:

```text
scene=cube
tracked_object=official_object
controller=official_newton_panda_hydro_scripted_no_adaptation
num_steps=240
sample_steps=0,30,60,90,120,150,180,210,239
```

## Evidence

```text
logs/newton/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210.log
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_summary.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_visual_validation.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_manual_visual_inspection.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_downstream_gate_cleared.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210.npz
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_metrics.json
experiments/outputs/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210_metrics.csv
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210/contact_sheet.png
experiments/visuals/lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210/frame_browser.html
```

## Result

- Fresh official Newton `sensor_contact` sanity: pass.
- SensorTiledCamera export: pass.
- Visual validation: pass.
- Manual visual inspection: pass with metric limitations.
- Downstream gate: pass for nominal baseline evidence.
- Metrics extractor: pass as a tool run, with baseline result `fail`.

The summary reports:

```text
initial_object_z=0.1200004369020462
final_object_z=0.33630338311195374
max_object_z=0.3435905873775482
max_lift=0.22359015047550201
```

The full metrics report says:

```text
status=fail
lift_height_m=0.22359015047550201
hold_duration_s=1.3166654109954834
max_slip_m=0.09295262564260072
object_not_dropped=true
drop_height_loss_m=0.007287204265594482
contact_loss_frames=0
max_contact_proxy=110
max_object_accel_m_s2=7.545902702324915
failure_reasons=[
  hold_duration_below_threshold,
  slip_above_threshold
]
```

The lift threshold in the Phase 02 metrics schema is `0.12m`, so this run
clears lift-height evidence. It fails the full stable-grasp baseline metric
because the continuous hold duration is below `2.0s` and lateral slip is above
`0.025m`.

## Limitations

This is a valid no-adaptation baseline failure case for the official cube
object. It does not block progress; it gives the next adaptive baselines a
concrete failure mode to improve: hold duration and slip control. It does not
claim nominal cup success, mass/fill generalization, curiosity, or learned
policy behavior.

## Non-Claims

- no learned policy;
- no curiosity result;
- no mass/fill generalization result;
- no tactile dominance result;
- no T-Rex schema fields;
- no pretrained checkpoint result.
