# Phase 01 Closed-Loop Curiosity Plan

Phase 01 starts after the accepted Phase 00 asset/data set. Its job is to turn
the fixed 15-cell Newton asset set into real closed-loop curiosity training
evidence, while preserving the hard training contract and the five-attempt
stop rule.

## Status

Status: active repair after negative evidence; one valid one-hour forward-model
component exists, one learned no-curiosity residual baseline exists,
source-matched curiosity scores are available, and four real one-hour
curiosity-weighted residual candidates have completed with negative held-out
comparisons.

Phase 00 provides the fixed asset basis. Phase 01 must not claim curiosity
success until closed-loop training improves over the declared strongest
available baseline on held-out metrics without safety regression.

Current training evidence: `p01_fwd_a1r2_20260630_002030` completed a
one-hour H200 Newton-native forward-model training component with checkpoint,
fresh official Newton sanity, and passing GPU utilization. This is only a
forward-model/learning-progress component. It is not a policy update, not
held-out closed-loop evaluation, and not a positive curiosity result.

Current learned-baseline evidence:
`p01_resid_base_a1_20260630_0307` completed one-hour H200 training for the
learned no-curiosity residual-controller baseline from the gated train-only
corrective source manifest. It wrote
`checkpoints/phase01/core/resid/base/p01_resid_base_a1_20260630_0307.pt`,
trained for 3600.239 seconds, reached optimizer step 14460, and passed GPU
utilization with 90.322% mean utilization. This is a required non-curiosity
baseline component. It is not curiosity training success and still needs
held-out evaluation.

Current learned-baseline held-out evidence:
`p01_resid_eval_a1_20260630_0342` evaluated the no-curiosity residual checkpoint
on all four locked held-out cells and succeeded on all four. The comparison
does not show a clean improvement over the existing strongest baseline set:
some slip/acceleration metrics improve, but other hold/lift/acceleration
metrics regress, especially versus scripted feedback on the cylinder held-out
cell. This is valid baseline evidence, not a positive curiosity result.

Current learning-progress evidence:
`experiments/outputs/phase01/core/lp/curiosity_learning_progress_summary.json`
passed from the valid forward-model checkpoint pair and wrote 19789 train/
validation scores. It records `policy_updated=false`; therefore it is a
curiosity scoring component only, not closed-loop policy success.

Current source-matched learning-progress evidence:
`p01_src_lp_a1_20260630_0405` ran inside Curiosity H200 Slurm job `157999` on
`server64`. It built `data/processed/phase01/src_lp/manifest.json` from the
same gated corrective source rollout keys used by residual training, then
computed matched scores in
`experiments/outputs/phase01/core/src_lp/curiosity_learning_progress_summary.json`.
The manifest passed with 8995 train transitions and 1799 validation
transitions, expected score coverage 0.999444 for train residual rows and
0.999444 for validation residual rows, and all held-out cells excluded. The
score summary passed with 10794 scores, mean learning progress
`0.5819649252997756`, and `policy_updated=false`. This is matched curiosity
scoring evidence only; it is not a policy update and not a success claim.

Current curiosity-weighted candidate evidence:
`p01_resid_cur_a1_20260630_0407` completed the first real one-hour
curiosity-weighted residual candidate in Curiosity H200 Slurm job `157999` on
`server64`. It trained for 3600.2106466293335 seconds, wrote
`checkpoints/phase01/core/resid/curiosity/p01_resid_cur_a1_20260630_0407.pt`,
and passed GPU utilization with 87.53781512605042% mean utilization. Held-out
evaluation `p01_resid_cur_eval_a1r1_20260630_0511` succeeded on all four
locked held-out cells and MP4 export
`p01_resid_cur_mp4_a1_20260630_0519` passed for 4/4 videos. However, strongest
baseline comparison `p01_resid_cur_cmp_a1_20260630_0518` classified the result
as `negative_or_incomplete_candidate`: `positive_curiosity_result=false`,
`safety_regression_cell_count=4`, and only one useful improvement. This is one
negative real one-hour curiosity candidate, not success.

Second curiosity repair evidence:
`p01_resid_cur_sa_a2_20260630_0521` completed the second real one-hour
safety-anchor curiosity-weighted residual candidate in Curiosity H200 Slurm
job `157999` on `server64`. It trained for 3600.1226358413696 seconds, wrote
`checkpoints/phase01/core/resid/curiosity_sa/p01_resid_cur_sa_a2_20260630_0521.pt`,
and passed GPU utilization with 88.14285714285714% mean utilization. Held-out
evaluation `p01_resid_cur_sa_eval_a2_20260630_0622` succeeded on all four
locked held-out cells, and MP4 export
`p01_resid_cur_sa_mp4_a2_20260630_0639` passed for 4/4 videos. However,
strongest-baseline comparison
`p01_resid_cur_sa_cmp_a2_20260630_0630` classified the result as
`negative_or_incomplete_candidate`: `positive_curiosity_result=false`,
`safety_regression_cell_count=4`, and `useful_improvement_count=0`. This is
negative evidence, not success. The intended baseline-preservation anchor did
not actually activate (`train_anchor_weight_mean=0.0`,
`validation_anchor_weight_mean=0.0`) because the trainer checked
`newton.contact.rigid_contact_count` while the Phase 01 rows provide
`newton.panda.rigid_contact_count`.

Third curiosity repair evidence:
`p01_resid_cur_sa2_a3_20260630_0641` completed the contact-fallback
safety-anchor curiosity-weighted residual candidate in Curiosity H200 Slurm
job `157999` on `server64`. It trained for 3600.0938968658447 seconds, wrote
`checkpoints/phase01/core/resid/curiosity_sa2/p01_resid_cur_sa2_a3_20260630_0641.pt`,
and passed GPU utilization with 87.19327731092437% mean utilization. The
trainer repair worked: `anchor_contact_count_columns` includes
`newton.panda.rigid_contact_count`, `train_anchor_weight_mean=0.33101025223731995`,
and `validation_anchor_weight_mean=0.38709700107574463`. Held-out evaluation
`p01_resid_cur_sa2_eval_a3_20260630_0739` succeeded on all four locked
held-out cells, and MP4 export `p01_resid_cur_sa2_mp4_a3_20260630_0748`
passed for 4/4 videos. However, strongest-baseline comparison
`p01_resid_cur_sa2_cmp_a3_20260630_0747` classified the result as
`negative_or_incomplete_candidate`: `positive_curiosity_result=false`,
`safety_regression_cell_count=4`, and `useful_improvement_count=1`. This is
negative evidence, not success. The active neutral anchor also damaged
validation behavior (`validation_loss=4.01609468460083`,
`validation_active_accuracy=0.5355555415153503`).

Fourth curiosity repair evidence:
`p01_resid_cur_distill_a4_20260630_0752` completed the base-policy
distillation/trust-region curiosity-weighted residual candidate in Curiosity
H200 Slurm job `157999` on `server64`. It trained for 3600.3141434192657
seconds, wrote
`checkpoints/phase01/core/resid/curiosity_distill/p01_resid_cur_distill_a4_20260630_0752.pt`,
and passed GPU utilization with 89.89830508474576% mean utilization. The
base-policy distillation anchor stayed active
(`train_anchor_weight_mean=0.41376280784606934`,
`validation_anchor_weight_mean=0.4838712811470032`) and preserved validation
behavior better than the neutral-anchor repair (`validation_loss=0.3044162690639496`,
`validation_active_accuracy=0.9661111235618591`). However, held-out evaluation
`p01_resid_cur_distill_eval_a4_20260630_0853` succeeded on only 3/4 held-out
cells, with the contact-only heavy cylinder failing and reaching
`max_object_accel_m_s2=9.535886263570077`. MP4 export
`p01_resid_cur_distill_mp4_a4_20260630_0903` passed for 4/4 videos. Strongest
baseline comparison `p01_resid_cur_distill_cmp_a4_20260630_0902` classified
the result as `negative_or_incomplete_candidate`:
`positive_curiosity_result=false`, `safety_regression_cell_count=4`, and
`useful_improvement_count=2`. This is negative evidence, not success. The
five-attempt stop-gate count is now 4/5.

Current baseline evidence:
`p01_base_heldout_r1_20260630_0120` completed all 8 held-out non-curiosity
baseline evaluations, covering 4 held-out cells by `no_adaptation` and
`scripted_feedback`. `p01_base_mp4_20260630_0132` then exported 8/8 real MP4
videos. This defines the current held-out comparison target; it is not a
curiosity result.

Current final-attempt data-repair evidence:
After four negative real one-hour curiosity policy candidates, Phase 01 ran
two train-only source/objective repair gates before allowing any fifth
candidate. Strict reuse of the existing corrective source was blocked:
`data/processed/phase01/src_strict/manifest.json` reports `status=blocked`,
0 admitted sources, 6 rejected sources, 0 train rows, 0 validation rows, and
`final_one_hour_attempt_allowed_from_this_preflight=false`. The stricter gate
rejected the prior source because the paired metrics traded away lift/hold.

The follow-up gentle strict source collection
`p01_src_gentle_a1_20260630_0913` also failed the final-attempt gate. It wrote
`data/processed/phase01/src_gentle/manifest.json` and
`experiments/reports/phase01/core/src_gentle_collect.md`, admitted only
`train_cylinder_heavy_low`, rejected 7 cells, produced 1800 train rows, and
produced 0 validation rows. Its failures are
`admitted_cells_below_min:1` and `no_validation_rows_from_admitted_sources`.
This is useful train-only repair evidence, but it is not sufficient to launch
the fifth and final allowed real one-hour curiosity policy candidate.

Two additional train-only source repairs were then tested without changing the
strict final-attempt gate. `p01_src_bal_a1r1_20260630_0945` used a balanced
feedback source with initial waypoint adjustment correctly disabled after a
runner boolean-normalization fix. It wrote
`data/processed/phase01/src_bal/manifest.json`, failed with only 1 admitted
source, 7 rejected sources, 1800 train rows, 0 validation rows,
`admitted_cells_below_min:1`, and `no_validation_rows_from_admitted_sources`.
`p01_src_box_a1_20260630_1006` focused on train-only box/light-cylinder cells
after the balanced gate showed the box cells were close to passing. It wrote
`data/processed/phase01/src_box/manifest.json`, but also failed with only 1
admitted source, 2 rejected sources, 1800 train rows, 0 validation rows,
`admitted_cells_below_min:1`, and `no_validation_rows_from_admitted_sources`.

The next repair changed the objective from source-level admission to
local-advantage segment masking without lowering the source safety contract.
`p01_local_adv_a1_20260630_1024` ran in Curiosity H200 Slurm job `158247` on
`server29`. It built `data/processed/phase01/local_adv/manifest.json` with
3 train segments, 1 validation segment, 576 train rows, 192 validation rows,
58 train active feedback labels, and 29 validation active feedback labels. All
held-out cells remain excluded. It then computed learning-progress scores in
`experiments/outputs/phase01/core/local_adv_lp/curiosity_learning_progress_summary.json`
with 768 scores, `policy_updated=false`, and no fake score fields. Smoke
diagnostic `p01_resid_cur_local_adv_smoke_a1_20260630_1026` passed with score
coverage 1.0 and no checkpoint. Therefore the fifth real one-hour curiosity
policy candidate is now allowed from this repaired local segment source.

Fifth-attempt evidence and stop-gate state:
`p01_resid_cur_local_adv_a5_20260630_1028` completed as the fifth real
one-hour curiosity policy candidate in Curiosity H200 Slurm job `158247` on
`server29`. It trained for 3600.0268936157227 seconds, wrote
`checkpoints/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028.pt`,
and passed GPU utilization with 99.18333333333334% mean utilization. Held-out
evaluation `p01_resid_cur_local_adv_eval_a5_20260630_1323` succeeded on all
four held-out cells, and MP4 export
`p01_resid_cur_local_adv_mp4_a5_20260630_1343` passed for 4/4 videos with
601 encoded frames each. However, strongest-baseline comparison
`p01_resid_cur_local_adv_cmp_a5_20260630_1340` classified the result as
`negative_or_incomplete_candidate`: `positive_curiosity_result=false`,
`safety_regression_cell_count=4`, and `useful_improvement_count=2`.
The five-attempt stop gate is now triggered. Do not start a sixth real
one-hour curiosity policy training attempt without explicit user instruction.

Current residual-training blocker:
`p01_resid_manifest_20260630_0138` tried to build residual-controller train/
validation CSVs from real Phase 00 Newton feedback fields. It failed the data
gate because the accepted source rows contain no active feedback labels:
`train_active_feedback_count=0` and `validation_active_feedback_count=0`.
Residual training must not start from these empty labels. The next faithful
step is to repair the training data/objective, for example by collecting
train-only corrective source rollouts that pass an advantage gate, or by moving
to a harder official task/object family where the no-adaptation baseline leaves
measured room for improvement. This blocker is not a negative one-hour training
attempt.

Current data-repair step:
Phase 01 now has a train-only corrective source collection path under
`experiments/configs/phase01/src_collect.json`,
`experiments/configs/phase01/run_src_collect_in_alloc.sh`,
`experiments/configs/phase01/launch_src_collect_tmux.sh`, and
`experiments/configs/phase01/build_src_gate.py`. It pairs each train cell with
`no_adaptation` and sensitive official scripted feedback, then admits a source
only if it has real active feedback labels and beats the paired no-adaptation
rollout without safety regression. This is data repair/preflight only; it is
not training, not a policy update, and not curiosity success.

Current data-repair evidence:
`p01_src_a3r2_20260630_0221` ran inside the Curiosity H200 allocation and
passed the source gate. It admitted 6 train-only corrective sources, rejected 2
weak/unsafe box sources, wrote `data/processed/phase01/src/train.csv` and
`data/processed/phase01/src/validation.csv`, and recorded 4061 train active
feedback labels plus 1616 validation active feedback labels. Held-out cells
remain excluded. This unlocks no-curiosity residual baseline training, but it
is still not residual training and not curiosity success.

## Inputs

- Phase 00 final MP4/asset evidence:
  `experiments/outputs/phase00_video_mp4_export_h200_20260629_203527_phase00_video_mp4_summary.json`
- Phase 00 generated row files:
  `experiments/outputs/phase00_core_asset_generation_h200_long_20260629_182052_phase00_cell_rows.jsonl`
  and
  `experiments/outputs/phase00_core_asset_generation_h200_long_repair2_20260629_183216_phase00_cell_rows.jsonl`
- Train split: 8 Phase 00 train cells.
- Validation split: 3 Phase 00 validation cells.
- Held-out split: 4 Phase 00 held-out cells, evaluation only.

Held-out cells remain locked. They must not be used for training, threshold
tuning, hyperparameter selection, label construction, source selection, or
controller repair.

## Phase 01 Objective

The core question is whether curiosity-driven interaction improves a basic
Newton Panda hydro grasp/lift/hold prior beyond no-adaptation and
no-curiosity/scripted baselines on harder held-out Phase 00 cells.

Phase 01 must produce:

- a compute-built transition manifest from Phase 00 Newton `.npz` records;
- baseline metrics for no-adaptation and strongest available non-curiosity
  controller on the same held-out cells;
- real one-hour training attempts for the curiosity model and residual policy
  update;
- closed-loop evaluation with full MP4 videos and strict metrics;
- an attempt ledger enforcing the five negative real-training attempt stop
  rule.

## Declared Baselines

The active Phase 01 baseline set is declared in
`experiments/configs/phase01/baselines.json`. It is not a result and does not
claim success. It fixes the comparison contract before any curiosity-weighted
policy result is interpreted:

- `no_adaptation`: official Newton Panda hydro scripted grasp/lift/hold prior
  with no feedback, no learned residual, and no curiosity reward;
- `scripted_feedback`: the same official prior with observation-triggered
  scripted contact/transition feedback, but no learning and no curiosity;
- `no_curiosity_residual`: learned residual adaptation without curiosity or
  learning-progress weighting;
- `curiosity_weighted_residual`: the candidate method, using bounded learning
  progress from the Phase 01 Newton-native forward model.

The curiosity candidate must beat the strongest valid non-curiosity baseline on
the held-out cells without safety regression before any positive claim is
allowed. Loss reduction, a checkpoint, or a forward-model-only result is not a
positive curiosity result.

## Training Design

Phase 01 is allowed to use Newton-native components as intermediate tools:

- transition prediction over documented Newton/contact/camera/mask fields;
- bounded learning-progress curiosity scores;
- curiosity-weighted residual controller adaptation.

These are not official T-Rex, not VQ-VAE, not T-Rex schema promotion, and not a
general world-model success claim. They are valid only as Newton-native
intermediate components until held-out closed-loop evidence proves policy
improvement.

## Evidence Gates

Before training:

- run inside a Curiosity-owned tmux-held H200 Slurm allocation;
- pass fresh official Newton SensorContact sanity in the same allocation;
- build the Phase 01 transition manifest from Phase 00 train/validation cells
  only;
- prove held-out rows are excluded from training artifacts;
- record exact commands, configs, logs, Slurm job, host, GPU, and env paths.

For each real one-hour training attempt:

- duration must be at least 3600 seconds;
- GPU utilization evidence is required;
- config, command, log, output summary, checkpoint/failure path, and metrics
  must be recorded;
- classify the attempt as positive, negative, invalid, or blocked.

Positive means improvement over the declared strongest available baseline on
the declared held-out metric without safety regression. Training loss
improvement alone is not positive.

Before residual policy training:

- do not use the failed zero-label residual manifest;
- collect corrective source only from Phase 01 train cells, never held-out;
- require paired no-adaptation vs scripted-feedback metrics for every candidate
  source cell;
- require active feedback labels and an advantage gate before writing a
  residual-controller train/validation CSV;
- compute source-matched learning-progress scores on the same
  `run_tag,timestep_index` keys as the gated residual source rows before
  curiosity-weighted residual training;
- if the gate admits too few source cells, record the blocker and do not run
  one-hour residual training from empty or unsafe labels.

If five real one-hour attempts are negative, stop before attempt six and report
to the user.

Current stop-gate state: four real one-hour curiosity candidates are negative.
Only one real one-hour negative candidate remains before the mandatory user
stop gate. Do not claim completion.

Current repair attempt:
The next step should be a more substantive data/objective repair focused on the
held-out failure pattern without using held-out data for training or tuning.
The current evidence suggests that train-only corrective imitation still
overfits or destabilizes harder held-out safety metrics, especially the
contact-only heavy cylinder. The four-negative repair audit is recorded at
`experiments/reports/phase01/core/resid/p01_four_negative_repair_audit.md`.
A fifth real one-hour candidate is allowed by the user stop gate, but it must
first repair the harmful train-only source objective or document a blocker. If
the fifth candidate is negative, stop and report before any sixth real training
attempt.

## Short Directory Layout

New Phase 01 artifacts must use compact grouped paths:

- configs: `experiments/configs/phase01/`
- processed data: `data/processed/phase01/core/`
- outputs: `experiments/outputs/phase01/core/`
- reports: `experiments/reports/phase01/core/`
- visuals: `experiments/visuals/phase01/core/`
- checkpoints: `checkpoints/phase01/core/`
- logs: `logs/newton/phase01/core/`

Long run tags may remain inside metadata, but human-facing directories must
stay grouped and short.

## Phase 02+ Direction

Phase 02 should only start after Phase 01 has either positive held-out evidence
or a documented blocker/negative-attempt report. Phase 02 is reserved for
expanding beyond the fixed Phase 00 asset family, such as distractors, broader
object families, or faithful mainstream-method comparisons. It must not be
used to escape a negative Phase 01 result.
