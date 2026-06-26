# Phase 04 Residual Label Source Runner V1

## Scope

This report records the first formal residual-label source runner and two
additional ordinary-cell source candidates. It is not learned-adapter training
and does not create a model.

## Allocation

- tmux session: `curiosity_residual_source_alloc_20260627_034021`.
- Slurm job: `154142`.
- Node: `server56`.
- Request type: tmux-held one GPU, one day, no `sbatch`.
- Environment: prebuilt local `envs/newton/.venv` activated on compute node.

## Runner

Files:

- Config: `experiments/configs/residual_label_source_runner_v1.json`.
- Builder: `experiments/configs/build_residual_label_source_runner.py`.
- Compute runner:
  `experiments/configs/run_residual_label_source_runner_in_alloc.sh`.
- tmux launcher:
  `experiments/configs/launch_residual_label_source_runner_tmux.sh`.

Final command:

```bash
RUN_TAG=residual_label_source_runner_v1_20260627_0420 \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
WINDOW_NAME=phase04_residual_source_runner4_final \
bash experiments/configs/launch_residual_label_source_runner_tmux.sh
```

Final output:

- Manifest:
  `data/processed/residual_label_source_runner_v1_20260627/manifest.json`.
- Records CSV:
  `data/processed/residual_label_source_runner_v1_20260627/residual_label_records.csv`.
- Fresh sanity:
  `experiments/outputs/residual_label_source_runner_v1_20260627_0420_fresh_newton_sensor_contact_sanity.json`.
- Log:
  `logs/newton/residual_label_source_runner_v1_20260627_0420.log`.

Result:

- fresh official Newton sanity: pass;
- status: pass;
- source run count: 4;
- record count: 1440;
- total feedback trigger count: 963;
- total feedback-active frames: 963;
- contact count range: 44 to 62;
- failures: [];
- generated T-Rex fields: [];
- schema promotion: blocked;
- training started: false.

## Source Candidates

Promoted ordinary cells:

- `half_low`:
  `residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`,
  360 records, feedback trigger count 241, strict metrics pass.
- `empty_low`:
  `residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345`,
  360 records, feedback trigger count 240, strict metrics pass.
- `half_medium`:
  `residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352`,
  360 records, feedback trigger count 241, strict metrics pass.
- `full_high`:
  `residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410`,
  360 records, feedback trigger count 241, strict metrics pass.

Held-out cells `full_low` and `empty_high` were not used for label collection.

## Additional Visual Gates

`empty_low` command:

```bash
RUN_TAG=residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345 \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
OBJECT_MASS_KG=0.08 \
OBJECT_FRICTION_MU=0.35 \
PRE_RECORD_WARMUP_STEPS=15 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Evidence:

- summary:
  `experiments/outputs/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345_summary.json`;
- visual validation:
  `experiments/outputs/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345_manual_visual_inspection.json`;
- metrics:
  `experiments/outputs/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345_metrics.json`;
- peak analysis:
  `experiments/outputs/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345_accel_peak_analysis.json`;
- contact sheet:
  `experiments/visuals/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345/contact_sheet.png`;
- frame browser:
  `experiments/visuals/residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345/frame_browser.html`.

`empty_low` metrics: lift height `0.16150899231433868` m, hold duration
`2.566664218902588` s, max slip `0.003694874589167713` m, max object
acceleration `0.5811279828844691` m/s^2, contact loss frames `0`.

`half_medium` command:

```bash
RUN_TAG=residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352 \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
OBJECT_MASS_KG=0.20 \
OBJECT_FRICTION_MU=0.80 \
PRE_RECORD_WARMUP_STEPS=15 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Evidence:

- summary:
  `experiments/outputs/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352_summary.json`;
- visual validation:
  `experiments/outputs/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352_manual_visual_inspection.json`;
- metrics:
  `experiments/outputs/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352_metrics.json`;
- peak analysis:
  `experiments/outputs/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352_accel_peak_analysis.json`;
- contact sheet:
  `experiments/visuals/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352/contact_sheet.png`;
- frame browser:
  `experiments/visuals/residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352/frame_browser.html`.

`half_medium` metrics: lift height `0.1581486612558365` m, hold duration
`2.5333309173583984` s, max slip `0.0030355661021019874` m, max object
acceleration `0.476706469800432` m/s^2, contact loss frames `0`.

`full_high` command:

```bash
RUN_TAG=residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410 \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
OBJECT_MASS_KG=0.35 \
OBJECT_FRICTION_MU=1.20 \
PRE_RECORD_WARMUP_STEPS=15 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Evidence:

- summary:
  `experiments/outputs/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410_summary.json`;
- visual validation:
  `experiments/outputs/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410_manual_visual_inspection.json`;
- metrics:
  `experiments/outputs/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410_metrics.json`;
- peak analysis:
  `experiments/outputs/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410_accel_peak_analysis.json`;
- contact sheet:
  `experiments/visuals/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410/contact_sheet.png`;
- frame browser:
  `experiments/visuals/residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410/frame_browser.html`.

`full_high` metrics: lift height `0.15532247722148895` m, hold duration
`2.499997615814209` s, max slip `0.0033865464497656577` m, max object
acceleration `1.3344599392207466` m/s^2, contact loss frames `0`.

Manual visual inspection opened the contact sheets plus `frame_0225.png` and
`frame_0359.png` for each new cell. All were nonblank and showed the cup held
at the final sampled frame.

## Interpretation

The source-gate path is now functional and has four promoted ordinary-cell
sources. This clears the previous runner blocker but does not authorize a
learned-adapter claim. The next aligned step is either to collect any remaining
ordinary source coverage or implement a reviewed learned residual-adapter
runner with the same fresh-sanity/source-gate/held-out split checks.
