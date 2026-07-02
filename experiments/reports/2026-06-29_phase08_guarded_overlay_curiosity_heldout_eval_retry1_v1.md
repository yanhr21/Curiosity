# Phase08 Curiosity-Weighted Held-Out Eval V1

Evaluation only. Not training and not a final curiosity success claim.

- status: `open_not_satisfied`
- active threshold: `0.5`
- curiosity-weighted beats strongest baseline all cells without safety regression: `False`
- curiosity-weighted beats advantage-gated residual all cells without safety regression: `True`
- manual visual inspection:
  `experiments/outputs/phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1_20260629_manual_visual_inspection.json`
- visual conclusion: all three candidate held-out contact sheets and rollout
  GIFs are nonblank with robot/object visible. Center and high-y are visually
  consistent with the success metrics; low-x is valid negative evidence
  consistent with zero hold duration and large slip. This is not a final
  curiosity success claim.

## Cells

- `pen_end_bias_heldout_center`: strongest baseline `guarded_feedback`, curiosity beats strongest `True`
  - `no_adaptation`: status `fail`, hold `0.0`, lift `0.1000232845544815`, accel `66.82017377787496`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_center_no_adaptation_20260628/rollout_video.gif`
  - `guarded_feedback`: status `fail`, hold `0.21666646003723145`, lift `0.14409343153238297`, accel `65.00249754885135`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_center_guarded_feedback_20260628/rollout_video.gif`
  - `advantage_gated_residual`: status `fail`, hold `0.0`, lift `0.08747351169586182`, accel `83.50906004973507`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_center_advantage_gated_residual_20260628/rollout_video.gif`
  - `curiosity_weighted_residual`: status `success`, hold `2.999997138977051`, lift `0.16922999918460846`, accel `3.7091188570093196`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_guarded_overlay_curiosity_retry1_eval_pen_end_bias_heldout_center_curiosity_weighted_residual_20260629/rollout_video.gif`
- `pen_end_bias_heldout_high_y`: strongest baseline `guarded_feedback`, curiosity beats strongest `True`
  - `no_adaptation`: status `fail`, hold `0.0`, lift `0.07911207526922226`, accel `64.05148283246749`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_high_y_no_adaptation_20260628/rollout_video.gif`
  - `guarded_feedback`: status `fail`, hold `0.33333301544189453`, lift `0.16264109313488007`, accel `70.09238925631959`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_high_y_guarded_feedback_20260628/rollout_video.gif`
  - `advantage_gated_residual`: status `fail`, hold `0.21666646003723145`, lift `0.14220823347568512`, accel `54.32288660879546`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_high_y_advantage_gated_residual_20260628/rollout_video.gif`
  - `curiosity_weighted_residual`: status `success`, hold `3.049997091293335`, lift `0.17296354472637177`, accel `2.8257651367175596`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_guarded_overlay_curiosity_retry1_eval_pen_end_bias_heldout_high_y_curiosity_weighted_residual_20260629/rollout_video.gif`
- `pen_end_bias_heldout_low_x`: strongest baseline `guarded_feedback`, curiosity beats strongest `False`
  - `no_adaptation`: status `fail`, hold `0.0`, lift `0.051910437643527985`, accel `41.79061979793733`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_low_x_no_adaptation_20260628/rollout_video.gif`
  - `guarded_feedback`: status `fail`, hold `0.0`, lift `0.05482320487499237`, accel `36.871479299759905`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_low_x_guarded_feedback_20260628/rollout_video.gif`
  - `advantage_gated_residual`: status `fail`, hold `0.0`, lift `0.049674391746520996`, accel `39.895279880954234`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_advantage_retry1_eval_pen_end_bias_heldout_low_x_advantage_gated_residual_20260628/rollout_video.gif`
  - `curiosity_weighted_residual`: status `fail`, hold `0.0`, lift `0.05279061198234558`, accel `35.61625832814527`, video `/public/home/yanhongru/Curiosity/experiments/visuals/phase08_guarded_overlay_curiosity_retry1_eval_pen_end_bias_heldout_low_x_curiosity_weighted_residual_20260629/rollout_video.gif`
