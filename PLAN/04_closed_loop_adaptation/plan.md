# Phase 04: Closed-Loop Adaptation

## Goal

Test the cup-mass example directly.

## Experiment

1. Start from a basic grasping prior.
2. Evaluate on nominal cup.
3. Evaluate on varied mass/fill-level cups.
4. Observe mismatch between expected and actual lift/contact response.
5. Adapt grip force, lift speed, stabilization, or regrasp timing.
6. Compare against no-adaptation and scripted feedback baselines.

## Residual Adaptation Policy

The first learned policy should output residual controller parameters rather
than full low-level torques:

- gripper closure target;
- lift velocity scale;
- hold height target;
- regrasp trigger threshold;
- stabilization duration.

This keeps the problem focused on contact-rich adaptation while the official
Newton Panda hydro scripted prior handles basic approach and grasp structure.
A pretrained checkpoint must not replace this short-term prior unless a
separate checkpoint audit proves code, weights, embodiment, action semantics,
and visual/metric behavior inside Newton.

User-approved short-term route as of 2026-06-27: begin closed-loop adaptation
from the official Newton scripted infant prior. The first adaptation policies
should tune controller parameters around that prior, not learn end-to-end
grasping from scratch and not depend on an unverified checkpoint.

## Dataset And Evaluation

Training rollouts should cover nominal and randomized cup properties. Held-out
mass/friction cells are reserved for testing whether the policy learned a
physical adaptation rule rather than memorizing the grid.

Evaluation must report:

- lift success;
- slip/drop rate;
- excessive-force rate;
- adaptation speed after mismatch;
- success per contact-proxy integral;
- visual success and failure cases with direct paths.

## Completion Criteria

- Adaptation improves at least one key metric without hiding safety failures.
- Results include success, under-grip/drop, over-force, wrong-mass expectation,
  and corrected-adaptation visual cases.
- Direct image paths are recorded in the report.

## Completed Scripted Feedback Nominal Gate

2026-06-27: configured and ran the first scripted feedback baseline on the
nominal existing-cup lift-hold task.

- Config: `experiments/configs/lift_hold_scripted_feedback_baseline_v1.json`.
- Launcher: `experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh`.
- Controller mode: `lift_hold_feedback`.
- Run tag: `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`.
- Output NPZ:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545.npz`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_nominal_cup.md`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- visual lift-hold behavior: pass;
- strict metric status: fail, only `object_accel_above_threshold`;
- feedback trigger count: 0.

Interpretation: nominal cup verifies the feedback path and logged controller
fields, but it does not yet demonstrate adaptation because no feedback event
was triggered. The next Phase 04 step is to run the scripted feedback baseline
across the same mass/friction grid used by Phase 02, preserving held-out cells.

## Completed Scripted Feedback Ordinary Grid Cells

2026-06-27: ran the first scripted feedback ordinary grid cell.

- Cell: `empty_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_low_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: the feedback grid path is runnable on a real mass/friction
variant. This cell still does not demonstrate adaptation because the current
feedback thresholds did not trigger. Continue through the grid before drawing
adaptation claims.

2026-06-27: validated the second scripted feedback ordinary grid cell.

- Cell: `empty_medium`.
- Canonical run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635`.
- Duplicate run kept as noncanonical output:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1630`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this second ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the third scripted feedback ordinary grid cell.

- Cell: `half_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_low_prefinalize_20260627_1700`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_low_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this third ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the fourth scripted feedback ordinary grid cell.

- Cell: `half_medium`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this fourth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the fifth scripted feedback ordinary grid cell.

- Cell: `half_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_high_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this fifth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the sixth scripted feedback ordinary grid cell.

- Cell: `full_medium`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this sixth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the seventh and final scripted feedback ordinary grid
cell.

- Cell: `full_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_high_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: the ordinary scripted feedback grid is now complete. All cells
are visually valid and correctly parameterized, but no cell triggered the
current feedback rule, so no adaptation-improvement claim is valid yet.

## Held-Out Scripted Feedback Evaluation

2026-06-27: evaluated the first held-out scripted feedback cell.

- Cell: `full_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845`.
- Held-out generalization cell: true.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_low_heldout.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Lift height: `0.15313686430454254` m.
- Hold duration: `2.7833306789398193` s.
- Max slip: `0.0034078387381632435` m.
- Contact-loss frames: `0`.
- Max contact proxy: `62.0`.
- Max object acceleration: `8.308707788010144` m/s^2.
- Feedback trigger count: `0`.

Interpretation: `full_low` remains held-out evidence. It is visually valid and
correctly parameterized, but still does not demonstrate feedback adaptation
because the feedback rule did not trigger. `empty_high` remains the last
held-out scripted feedback evaluation cell.

2026-06-27: evaluated the second and final held-out scripted feedback cell.

- Cell: `empty_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955`.
- Held-out generalization cell: true.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_high_heldout.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Lift height: `0.16016103327274323` m.
- Hold duration: `2.8333306312561035` s.
- Max slip: `0.0035689078921667837` m.
- Contact-loss frames: `0`.
- Max contact proxy: `62.0`.
- Max object acceleration: `8.308498000056417` m/s^2.
- Feedback trigger count: `0`.

Interpretation: scripted feedback evaluation now covers the nominal cup, seven
ordinary cells, and two held-out cells. All are visually valid and correctly
parameterized, but no cell triggered the current feedback rule, so no
adaptation-improvement claim is valid. The next adaptation step must either
revise the feedback trigger with documented rationale or move to the planned
residual learned controller-parameter adapter.

## Contact-Aware Curiosity Diagnostic Gate

2026-06-27: Phase 03 contact-aware curiosity replay diagnostic already covers
the mass/friction variant grid needed by Phase 04.

- Config: `experiments/configs/curiosity_reward_baseline_replay_v1.json`.
- Evaluator: `experiments/configs/evaluate_curiosity_reward_baseline_replay.py`.
- Output JSON:
  `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- Output CSV:
  `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv`.
- Report:
  `experiments/reports/2026-06-27_phase03_curiosity_reward_replay_v1.md`.
- Log: `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`.
- Status: pass.
- Rollouts evaluated: 9.
- Held-out cells included: `full_low`, `empty_high`.
- Tactile source: `newton.contact_proxy_only`.

This completes the Phase 04 contact-aware curiosity diagnostic across
mass/friction variants, but only as replay reward-shape evidence. It does not
train a world model, does not update a policy, does not promote data to the
T-Rex schema, and does not use real tactile-marker evidence yet. The learned
forward-model target path remains a separate Phase 03/04 requirement before
curiosity can be treated as a learned adaptation mechanism.

## Residual Adapter Training Contract V1

2026-06-27: defined the first residual controller-parameter adapter contract
without starting training.

- Spec: `docs/residual_adapter_forward_model_contract_v1.md`.
- Config: `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- Status: `target_contract_ready_training_not_started`.

Allowed residual outputs:

- gripper closure target delta;
- lift velocity scale delta;
- hold height target delta;
- regrasp trigger threshold delta;
- stabilization duration delta.

Forbidden outputs remain full low-level torque control, T-Rex `action`,
T-Rex `action_abs`, and promoted `observation.*` fields. Training remains a
separate unfinished Phase 04 item and must run only after source gates, official
sanity checks, held-out split preservation, visual paths, and ablations are
ready.

## Residual Adapter Training Readiness V1

2026-06-27: audited whether the first learned residual controller-parameter
adapter can be trained now.

- Config:
  `experiments/configs/residual_adapter_training_readiness_v1.json`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_readiness_v1.md`.
- Status: `blocked_training_not_started`.

Ready:

- official Newton scripted infant prior;
- real mass/friction grid;
- held-out `full_low` and `empty_high`;
- Phase 05 Newton contact proxy source manifest;
- Phase 03 curiosity replay diagnostics;
- residual adapter and forward-model target contract.

Blocking:

- all scripted-feedback runs have `feedback_trigger_count=0`;
- current data has no nonzero residual controller-parameter corrections;
- training on current labels would produce a no-op residual adapter;
- no approved residual-adapter training implementation/runner exists yet.

Interpretation: Phase 04 should not start learned-adapter training until
nonzero residual demonstrations or another approved serious objective exists.
The next aligned action is to revise the scripted feedback trigger and collect
real residual correction evidence, or ask for approval before switching to a
different serious policy method.

## Residual Correction Collection Diagnostic V1

2026-06-27: converted the no-nonzero-residual blocker into an executed
ordinary-cell diagnostic.

- Plan/config:
  `experiments/configs/residual_correction_collection_plan_v1.json`.
- Plan report:
  `experiments/reports/2026-06-27_phase04_residual_correction_collection_plan_v1.md`.
- Executed diagnostic report:
  `experiments/reports/2026-06-27_phase04_residual_label_source_sensitive_feedback_half_low.md`.
- Run tag:
  `residual_label_source_sensitive_feedback_half_low_20260627_030145`.
- Allocation: `154023`.
- Tmux session: `curiosity_next_source_alloc_20260626_232937`.
- Cell: ordinary `half_low`, not held-out.

Result:

- official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: `pass_nonblank_but_task_failure`;
- feedback reason: `low_contact_count`;
- final feedback trigger count: 241;
- feedback-active frames: 241;
- object dropped: false;
- contact-loss frames: 0;
- metrics status: fail;
- failure reasons: `hold_duration_below_threshold`,
  `object_accel_above_threshold`.

Interpretation: the official Newton rollout path can produce nonzero residual
controller-parameter labels under `candidate.controller.*`, so the project is
not stuck on schema mismatch or an impossible data path. This specific
threshold is too aggressive and is not promoted to a training-label source.
Two additional sweep facts are now recorded in
`experiments/configs/residual_correction_collection_plan_v1.json`:

- `residual_label_sweep_half_low_contact58_20260627_0310`: nonzero labels, but
  hold still fails.
- `residual_label_source_accel_sensitive_half_low_20260627_030748`: lift/hold/
  drop/contact behavior preserved, but `feedback_trigger_count=0`.
- `residual_label_sweep_half_low_contact58_gentle_20260627_0345`: nonzero
  labels plus preserved lift/hold/drop/contact/visual/manual gates; strict
  metrics still fail only on `object_accel_above_threshold`.

The next Phase 04 action should use `contact58_gentle` as the best current
candidate and reduce object acceleration while preserving nonzero feedback. Do
not start learned-adapter training until the candidate passes the strict gate
or the object-acceleration threshold change is explicitly approved and
documented.

## Short-Term Stable Residual Route

2026-06-27: user approved the short-term stable method. The active route is:

1. Keep the official Newton Panda hydro scripted controller as the infant
   grasp/lift prior.
2. Do not wait for an unverified Newton-native pretrained grasp checkpoint.
3. Collect residual controller-parameter labels only from ordinary cells that
   pass official Newton sanity, automated/manual visual checks, lift, hold,
   drop, and contact gates.
4. Preserve `full_low` and `empty_high` as held-out generalization cells.
5. Train the learned residual adapter only after at least one nonzero residual
   label source is promoted by those gates.

The first threshold-sweep follow-up was run:

- Run tag: `residual_label_sweep_half_low_contact58_20260627_0310`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58.md`.
- Cell: ordinary `half_low`, not held-out.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_but_task_failure`.
- Feedback trigger count: `241`.
- Feedback-active frames: `241`.
- Lift height: `0.12752966582775116` m.
- Longest hold: `0.9833333333333333` s.
- Drop from max: `0.0` m.
- Promotion decision: not promoted to training-label source.

Interpretation: lowering the contact trigger from 64 to 58 still produced
nonzero residual labels but did not recover the formal 2s hold gate. The next
run should use a less disruptive ordinary-cell trigger strategy, preferably
acceleration-sensitive or milder contact-sensitive, instead of starting
adapter training.

2026-06-27 follow-up: the less disruptive contact58 gentle route was tested.

- Run tag: `residual_label_sweep_half_low_contact58_gentle_20260627_0345`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle.md`.
- Cell: ordinary `half_low`, not held-out.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Feedback trigger count: `241`.
- Feedback-active frames: `241`.
- Lift height: `0.1518997997045517` m.
- Hold duration: `2.7166640758514404` s.
- Max slip: `0.002728142855700976` m.
- Contact-loss frames: `0`.
- Max object acceleration: `8.308707788010144` m/s^2.
- Strict metrics status: fail, only `object_accel_above_threshold`.

This is the best current residual-label candidate because it combines nonzero
labels with preserved lift/hold/drop/contact/visual gates, but it is not fully
promoted under the strict metrics gate.

2026-06-27 second follow-up: increasing the initial lift duration scale was
tested to reduce the repeated object-acceleration failure.

- Run tag:
  `residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_smooth.md`.
- Initial lift duration scale: `1.8`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Feedback trigger count: `241`.
- Lift height: `0.1519654542207718` m.
- Hold duration: `2.3499977588653564` s.
- Max slip: `0.0026201330580321184` m.
- Contact-loss frames: `0`.
- Max object acceleration: `8.308972018193668` m/s^2.
- Strict metrics status: fail, only `object_accel_above_threshold`.

Interpretation at that point: the repeated acceleration blocker was not solved
by simply reducing feedback amplitude or stretching the initial lift waypoint.
The next diagnostic therefore analyzed peak timing instead of continuing blind
threshold sweeps.

2026-06-27 third follow-up: peak analysis identified the strict acceleration
failure as a recorded initial settling artifact, and a pre-record warmup source
candidate was tested.

- Run tag:
  `residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_lift165_warmup15.md`.
- Source manifest:
  `experiments/configs/residual_label_source_manifest_v1.json`.
- Cell: ordinary `half_low`, not held-out.
- Pre-record warmup steps: `15`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: `pass_nonblank_success_with_feedback`.
- Feedback trigger count: `241`.
- Feedback-active frames: `241`.
- Lift height: `0.15815936028957367` m.
- Hold duration: `2.5333309173583984` s.
- Max slip: `0.0030417809728431086` m.
- Contact-loss frames: `0`.
- Max object acceleration: `0.5063306543767194` m/s^2.
- Strict metrics status: pass.

Peak analysis showed the earlier non-warmup acceleration maximum occurred at
step 2, phase 0, before feedback was active. The warmup15 source candidate
removes that artifact from the recorded metric window without changing to a
toy model or starting training. This is now the first promoted residual-label
source candidate for runner input.

Next action at that point was to build the formal residual-label source runner
and source-gate checks around
`experiments/configs/residual_label_source_manifest_v1.json`. That runner is
recorded in the following follow-up.

2026-06-27 fourth follow-up: the formal residual-label source runner was
implemented and run inside tmux-held allocation `154142`.

- Runner config:
  `experiments/configs/residual_label_source_runner_v1.json`.
- Builder:
  `experiments/configs/build_residual_label_source_runner.py`.
- Launcher:
  `experiments/configs/launch_residual_label_source_runner_tmux.sh`.
- Compute runner:
  `experiments/configs/run_residual_label_source_runner_in_alloc.sh`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_label_source_runner_v1.md`.
- Final runner tag: `residual_label_source_runner_v1_20260627_0455`.
- Output manifest:
  `data/processed/residual_label_source_runner_v1_20260627/manifest.json`.
- Output records:
  `data/processed/residual_label_source_runner_v1_20260627/residual_label_records.csv`.
- Fresh official Newton sanity: pass.
- Source run count: `5`.
- Record count: `1800`.
- Total feedback trigger count: `1203`.
- Total feedback-active frames: `1203`.
- Contact count range: `44..62`.
- Failures: `[]`.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.
- Training started: `false`.

Promoted source cells are ordinary `half_low`, `empty_low`, `half_medium`,
`full_high`, and `empty_medium`. Held-out `full_low` and `empty_high` remain
unused for label collection.

2026-06-27 fifth follow-up: the residual-adapter training preflight runner was
implemented and run inside tmux-held allocation `154142`.

- Config:
  `experiments/configs/residual_adapter_training_preflight_v1.json`.
- Builder:
  `experiments/configs/build_residual_adapter_training_preflight.py`.
- Launcher:
  `experiments/configs/launch_residual_adapter_training_preflight_tmux.sh`.
- Compute runner:
  `experiments/configs/run_residual_adapter_training_preflight_in_alloc.sh`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_preflight_v1.md`.
- Final preflight tag:
  `residual_adapter_training_preflight_v1_20260627_0523`.
- Output manifest:
  `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
- Train split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_train_records.csv`.
- Validation split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_validation_records.csv`.
- Fresh official Newton sanity: pass.
- Source record count: `1800`.
- Train records: `1440`.
- Validation records: `360`.
- Failures: `[]`.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.
- Training started: `false`.
- Model created: `false`.

Train cells are ordinary `half_low`, `empty_low`, `half_medium`, and
`full_high`. Validation cell is ordinary `empty_medium`. Held-out `full_low`
and `empty_high` remain unused for labels and training.

Next action: implement the actual learned residual-adapter trainer that
consumes the preflight manifest, reruns fresh official Newton sanity, preserves
held-out split checks, trains for the required GPU duration, and reports
checkpoint/metrics/visual evidence. No learned-adapter result exists yet.

2026-06-27 sixth follow-up: the Newton-native residual-adapter trainer smoke
was implemented and run inside tmux-held allocation `154142`.

- Trainer config:
  `experiments/configs/residual_adapter_trainer_v1.json`.
- Trainer script:
  `experiments/configs/train_residual_adapter_v1.py`.
- Launcher:
  `experiments/configs/launch_residual_adapter_trainer_tmux.sh`.
- Compute runner:
  `experiments/configs/run_residual_adapter_trainer_in_alloc.sh`.
- Environment note:
  `docs/residual_adapter_environment_v1.md`.
- Report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_trainer_smoke_v1.md`.
- Final smoke tag:
  `residual_adapter_trainer_v1_smoke_20260627_0539`.
- Summary:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_smoke_20260627_0539_summary.json`.
- Fresh official Newton sanity: pass.
- Trainer venv: `envs/residual_adapter/.venv`.
- Torch version: `2.6.0+cu124`.
- Device: `cuda:0`, NVIDIA H200.
- Optimizer steps: `3`.
- Validation loss: `1.626072883605957`.
- Checkpoint written: `false`.
- Real training result: `false`.
- Generated T-Rex fields: `[]`.
- Schema promotion: `blocked`.
- Failures: `[]`.

Next action: run real `RUN_MODE=train` only with a GPU-utilization plan that
satisfies the one-GPU one-hour rule, then evaluate the checkpoint on held-out
`full_low` and `empty_high` with visual and metric gates before making any
learned-adaptation claim.
