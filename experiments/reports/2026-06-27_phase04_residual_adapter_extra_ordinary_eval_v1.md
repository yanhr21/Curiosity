# Phase 04 Residual Adapter Extra Ordinary Evaluation V1

## Scope

This report records two additional ordinary-cell evaluations of the trained
Newton-native residual controller adapter. These runs broaden mass/friction
coverage beyond the ordinary `empty_medium` validation cell and the held-out
`full_low` / `empty_high` gate.

This is not a T-Rex result, not a tactile F6 result, and not broad object-family
generalization. `generated_trex_fields=[]` and `schema_promotion=blocked`
remain required.

## Shared Setup

- Allocation: tmux-held Slurm job `154142` on `server56`.
- Controller mode: `lift_hold_learned_residual`.
- Checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`.
- Evaluation config:
  `experiments/configs/residual_adapter_evaluation_v1.json`.
- Launcher:
  `experiments/configs/launch_residual_adapter_evaluation_tmux.sh`.
- Metric launcher:
  `experiments/configs/launch_lift_hold_metrics_tmux.sh`.
- Official Newton sanity: fresh pass before each rollout.
- Downstream rule: each rollout stayed blocked until automated visual
  validation and manual frame inspection passed.

## Ordinary Half-High

- Cell: `half_high`.
- Run tag: `residual_adapter_eval_v1_half_high_ordinary_20260627_0631`.
- Object mass: `0.20` kg.
- Object friction: `1.2`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection:
  `experiments/outputs/residual_adapter_eval_v1_half_high_ordinary_20260627_0631_manual_visual_inspection.json`.
- Contact sheet:
  `experiments/visuals/residual_adapter_eval_v1_half_high_ordinary_20260627_0631/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_adapter_eval_v1_half_high_ordinary_20260627_0631/frame_browser.html`.
- Metrics:
  `experiments/outputs/residual_adapter_eval_v1_half_high_ordinary_20260627_0631_metrics.json`.

Metric result:

- Status: pass.
- Lift height: `0.16001036763191223` m.
- Hold duration: `2.549997568130493` s.
- Max slip: `0.003629093007284586` m.
- Contact-loss frames: `0`.
- Max contact proxy: `62`.
- Max object acceleration: `0.4709508074000259` m/s^2.
- Object not dropped: `true`.
- Model evaluation: `true`.
- `no_model_or_training`: `false`.
- `generated_trex_fields`: `[]`.
- `schema_promotion`: `blocked`.

## Ordinary Full-Medium

- Cell: `full_medium`.
- Run tag: `residual_adapter_eval_v1_full_medium_ordinary_20260627_0638`.
- Object mass: `0.35` kg.
- Object friction: `0.8`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection:
  `experiments/outputs/residual_adapter_eval_v1_full_medium_ordinary_20260627_0638_manual_visual_inspection.json`.
- Contact sheet:
  `experiments/visuals/residual_adapter_eval_v1_full_medium_ordinary_20260627_0638/contact_sheet.png`.
- Frame browser:
  `experiments/visuals/residual_adapter_eval_v1_full_medium_ordinary_20260627_0638/frame_browser.html`.
- Metrics:
  `experiments/outputs/residual_adapter_eval_v1_full_medium_ordinary_20260627_0638_metrics.json`.

Metric result:

- Status: pass.
- Lift height: `0.1546410769224167` m.
- Hold duration: `2.499997615814209` s.
- Max slip: `0.0034754009349139145` m.
- Contact-loss frames: `0`.
- Max contact proxy: `61`.
- Max object acceleration: `2.6287727996680594` m/s^2.
- Object not dropped: `true`.
- Model evaluation: `true`.
- `no_model_or_training`: `false`.
- `generated_trex_fields`: `[]`.
- `schema_promotion`: `blocked`.

## Interpretation

The trained residual adapter now passes visual and metric gates on four
evaluation cells after training:

- ordinary validation: `empty_medium`;
- held-out evaluation: `full_low`, `empty_high`;
- extra ordinary evaluation: `half_high`, `full_medium`.

The two extra ordinary cells reduce the chance that the held-out pass was an
isolated two-cell artifact. They do not by themselves prove adaptation speed,
multi-seed stability, new object-family generalization, or tactile/T-Rex
equivalence.

Next work should compare adaptation speed and failure modes across repetitions,
including baseline-vs-learned timing of residual activation, contact proxy,
acceleration peaks, slip, and hold recovery.
