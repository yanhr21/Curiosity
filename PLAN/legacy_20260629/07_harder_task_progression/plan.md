# Phase 07: Harder Task Progression

## Goal

Move beyond the first easy cup benchmark once the residual adapter and
curiosity-training gates are under control. The harder tasks should test
whether the agent can actively learn physical properties, not merely replay a
single easy lift-hold routine.

This phase is now a hard continuation requirement, not an optional extension.
The previous Phase 03 Newton-native curiosity V1 result only proved that a
training pipeline can run and pass the original held-out cup cells. It did not
prove improvement over the no-curiosity residual baseline and must not be
treated as final success.

The required end state is a complete harder-task closed-loop curiosity result:

- train or adapt with curiosity in a real closed loop, not only offline replay
  scoring or a one-off supervised weighting diagnostic;
- evaluate on harder held-out tasks beyond the original easy cup cells;
- produce full rollout videos and sampled visual browsers/contact sheets;
- compare against no-adaptation, scripted feedback, no-curiosity residual
  adaptation, curiosity ablations, and serious/mainstream reference methods or
  official checkpoints when available;
- show that curiosity is better than the declared baseline on success and
  safety metrics before any improvement claim is allowed.

If the curiosity path only matches baseline, improves one metric while
worsening safety, lacks videos, lacks harder held-out tests, or lacks a
mainstream-method comparison, Phase 07 remains incomplete.

## No-Downgrade Hard Training Protocol

Before each Phase 07 training attempt, write down the exact task cells,
training budget, held-out cells, baseline policies, ablations, success
metrics, safety metrics, checkpoint paths, log paths, and video evidence that
will be used to judge the run.

The training attempt must stay on the harder task family unless a blocker is
documented. It must not be quietly simplified into:

- the original easy cup benchmark;
- a single cherry-picked task cell;
- offline replay scoring only;
- supervised curiosity-weighted reweighting only;
- a code-path smoke test;
- a sparse contact-sheet-only visual check;
- a small homemade replacement for a missing serious method.

Those activities are allowed only when explicitly labeled as diagnostics,
preflight, source collection, or blocker investigation. They cannot close the
Phase 07 objective and cannot be described as completed curiosity training.

If a run fails or curiosity does not beat the declared baseline, keep the phase
open. The next action should be either a faithful fix, a stronger baseline
audit, a better closed-loop training run, or a clearly documented blocker.

This is a hard stop against quick-exit behavior. Do not close Phase 07 after
writing a checkpoint, computing curiosity scores, producing a single rollout,
or obtaining a negative comparison. Those outputs are evidence, not completion.
Completion requires the declared harder-task contract to pass and the
curiosity-trained/adapted policy to beat the strongest declared baseline
without hiding safety regressions. If that does not happen, continue with the
next faithful training or evaluation step.

User reaffirmation on 2026-06-27: the harder-training requirement must remain
visible in the plan and must prevent future downgrade or fast-exit behavior.
Phase 07 cannot be reframed as complete because a diagnostic ran, a checkpoint
was written, an ablation finished, or queue/resource pressure made the next run
hard. Negative evidence must trigger continued faithful work or an explicit
blocker, not a reduced success claim.

Continuation lock: after every Phase 07 run, classify the evidence before any
completion claim. Missing closed-loop curiosity updates, harder held-out
evaluation, full videos, strict safety metrics, required ablations, or faithful
serious-method comparison means the phase is incomplete. Curiosity failing to
beat the strongest declared baseline without safety regression means the result
is negative evidence. The required follow-up is a faithful next training or
evaluation step, a baseline/ablation audit, objective repair, stronger data
collection, or a documented blocker.

Memory-file persistence lock from 2026-06-27: this harder-training requirement
has to stay written in the idea, agent, plan, and todo records. Future work
must treat it as an active gate, not a historical note. Do not reduce Phase 07
to a toy pipeline, an easy original-cup rerun, offline score computation,
single-cell video evidence, a missing-checkpoint excuse, or a small homemade
replacement for a serious method. If the current result is weak, negative,
resource-constrained, or inconvenient, the plan remains open and the next step
is faithful continuation: objective repair, stronger data collection, real
training/evaluation, ablation, baseline audit, official-method setup, or a
documented blocker.

User reaffirmation on 2026-06-28: the harder-training requirement is written
here, in `IDEA/idea.md`, in `AGENTS.md`, and in the active TODO to prevent
another downgrade or fast exit. Phase 07 must not be closed by a diagnostic,
ablation, queue completion, checkpoint, single video, or negative comparison.
Completion requires the declared harder-task contract to pass, including
closed-loop curiosity updates, harder held-out evaluation, full videos, strict
safety metrics, ablations, and faithful serious-method comparison or blocker
evidence. If curiosity does not beat the strongest declared baseline without
safety regression, continue faithful training/evaluation or record a concrete
blocker.

Latest enforcement lock on 2026-06-28: this plan must be interpreted as an
active no-downgrade gate. Do not simplify the hard objective into a convenient
easy-cell rerun, toy model, offline score, smoke test, validation-only repair,
queue/script completion, checkpoint-exists claim, single rendered video, or a
negative result framed as completion. Every run must be judged against the
declared hard-training contract and strongest-baseline comparison. If the
contract is not satisfied, Phase 07 remains open and the next planned action is
faithful repair, stronger data collection, real training/evaluation, ablation,
baseline audit, official-method setup, or a documented blocker.

Evidence challenge reaffirmation on 2026-06-28: any future "curiosity training
complete" statement must cite a complete evidence chain: training command/log,
checkpoint path, official sanity check, held-out harder-task metrics, strongest
baseline comparison, safety comparison, full rollout videos, manual visual
inspection, and serious-method comparison or faithful blocker. A script exit,
validation-only threshold repair, diagnostic rerun, checkpoint artifact,
rendered video, or negative comparison cannot close this phase.

Evidence-gate update on 2026-06-27: the project now has an executable hard
training classifier at
`experiments/configs/audit_phase07_hard_training_evidence_gate_v1.py`. The
latest output is
`experiments/outputs/phase07_hard_training_evidence_gate_v1_20260627.json` and
the report is
`experiments/reports/2026-06-27_phase07_hard_training_evidence_gate_v1.md`.
Current gate status is `open_not_satisfied` with
`final_curiosity_success_allowed=false`: remaining ablation training is
missing, existing older held-out NPZ files lack the candidate action bridge,
curiosity-weighted residual does not beat the strongest baseline without
safety regression, and the serious/mainstream comparison gate is still open.

Evidence-gate refresh on 2026-06-28: allocation-only refresh wrote
`experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json` and
`experiments/reports/2026-06-28_phase07_evidence_refresh_v1.md`. Training and
evaluation evidence are no longer stale, but
`final_curiosity_success_allowed=false` remains open because the current
curiosity path still does not beat the strongest baseline on held-out harder
cells and official serious-method readiness is not satisfied.

Closed-loop threshold-repair diagnostic on 2026-06-28: validation-only Newton
interaction repair wrote
`experiments/outputs/phase07_closed_loop_threshold_repair_v1_20260628_summary.json`
and selected the safety-first threshold `0.65` after fixing a selector bug.
This is objective repair and threshold selection only; it is not training and
not a success claim.

Corrected 0.65 held-out diagnostic on 2026-06-28: the rerun wrote
`experiments/outputs/phase07_threshold065_heldout_eval_retry_v1_20260628_summary.json`
and full videos under
`experiments/visuals/phase07_eval_*_curiosity_threshold065_repaired_20260628/`.
Status remains `open_not_satisfied`: all three held-out cells were visually
valid and had no listed safety regression, but none beat the strongest
`no_adaptation` baseline on the ordered comparison, mainly because hold
duration and lift remain lower. This is negative evidence requiring continued
faithful training/evaluation, not completion.

V2 objective-repair start on 2026-06-28: audit of the corrected 0.65 held-out
failure found a structural action-range issue. Current Phase07 source labels
and learned-residual evaluation clamp
`candidate.controller.feedback_stabilization_extension_s` to `0.3s`, while the
strongest `no_adaptation` held-out baseline holds for about `3.1s`. The learned
residual therefore cannot express a hold policy that beats the baseline on the
primary ordered comparison. The next faithful step is not another held-out
threshold tweak; it is a train/validation-only V2 source collection and
retraining pass with a wider but still bounded stabilization action range.
Added configs:
`experiments/configs/phase07_v2_stabilization_source_collection_v1.json`,
`experiments/configs/phase07_v2_stabilization_residual_label_source_runner_v1.json`,
`experiments/configs/phase07_v2_residual_adapter_training_preflight_v1.json`,
`experiments/configs/phase07_v2_residual_adapter_trainer_v1.json`,
`experiments/configs/phase07_v2_curiosity_forward_model_preflight_v1.json`,
`experiments/configs/phase07_v2_curiosity_forward_model_trainer_v1.json`,
`experiments/configs/phase07_v2_curiosity_learning_progress_v1.json`, and
`experiments/configs/phase07_v2_curiosity_weighted_residual_adapter_trainer_v1.json`.
The source collector is allocation-only and writes a manual-visual-pending
manifest first; it cannot feed training until direct visual checks pass.

V2 source collection allocation on 2026-06-28: started tmux-held Slurm job
`155749` in session `curiosity_phase07_v2_source_alloc_20260628`, running
`experiments/configs/run_phase07_v2_stabilization_source_collection_in_alloc.sh`
through launcher
`experiments/configs/launch_phase07_v2_stabilization_source_collection_tmux.sh`.
Log:
`logs/newton/phase07_v2_stabilization_source_collection_v1_20260628.log`.
This is source collection only, not training and not a success claim.

V2 source/preflight progress on 2026-06-28: source collection finished with
exit `0`, wrote
`experiments/outputs/phase07_v2_stabilization_source_collection_v1_20260628_summary.json`,
and generated eight 420-frame train/validation rollout videos. Direct manual
inspection passed for all eight contact sheets, then
`experiments/configs/phase07_v2_stabilization_source_manifest_v1.json` was
promoted to
`phase07_v2_stabilization_source_candidates_complete_training_not_started`.
The V2 source runner then passed, writing
`data/processed/phase07_v2_stabilization_residual_label_source_runner_v1_20260628/manifest.json`
with `3360` records, eight source runs, and `2405` feedback-trigger frames.
The V2 residual-adapter preflight passed, writing
`data/processed/phase07_v2_residual_adapter_training_preflight_v1_20260628/manifest.json`
with `2520` train records and `840` validation records. These are still data
and preflight gates, not curiosity success.

V2 no-curiosity baseline training on 2026-06-28: started real training for
`phase07_v2_residual_adapter_v1_train_20260628` via
`experiments/configs/phase07_v2_residual_adapter_trainer_v1.json` in the same
tmux-held Slurm job `155749`. Log:
`logs/newton/phase07_v2_residual_adapter_v1_train_20260628.log`. This baseline
must complete at least one GPU-hour and produce a checkpoint before the V2
curiosity-weighted policy can be judged.

V2 no-curiosity baseline result on 2026-06-28: training completed with exit
`0`, `real_training_result=true`, `elapsed_seconds=3600.183328151703`,
`optimizer_steps=18668`, checkpoint
`checkpoints/phase07_v2_residual_adapter_trainer_v1_20260628/phase07_v2_residual_adapter_v1_train_20260628.pt`,
and mean GPU utilization `99.07964601769912%`. This is the required V2
no-curiosity learned baseline, not curiosity success.

V2 curiosity forward model progress on 2026-06-28: forward-model preflight
passed, writing
`data/processed/phase07_v2_curiosity_forward_model_preflight_v1_20260628/manifest.json`
with `3352` transition records (`2514` train, `838` validation). Real V2
forward-model training then started as
`phase07_v2_curiosity_forward_model_v1_train_20260628` in Slurm job `155749`
using `experiments/configs/phase07_v2_curiosity_forward_model_trainer_v1.json`.
Log:
`logs/newton/phase07_v2_curiosity_forward_model_v1_train_20260628.log`.
It must complete at least one GPU-hour before learning-progress curiosity can
be computed.

V2 forward-model and curiosity-policy progress on 2026-06-28: forward-model
training completed with `real_training_result=true`,
`elapsed_seconds=3600.023451566696`, `optimizer_steps=17970`, checkpoint
`checkpoints/phase07_v2_curiosity_forward_model_v1_20260628/phase07_v2_curiosity_forward_model_v1_train_20260628.pt`,
initial snapshot
`checkpoints/phase07_v2_curiosity_forward_model_v1_20260628/phase07_v2_curiosity_forward_model_v1_train_20260628_initial_snapshot.pt`,
and mean GPU utilization `99.130081300813%`. V2 learning-progress scoring then
passed, writing
`experiments/outputs/phase07_v2_curiosity_learning_progress_v1_20260628/curiosity_learning_progress_summary.json`
and `curiosity_learning_progress_scores.csv` for `3352` records. Mean bounded
curiosity reward was `0.643422921641614` overall, `0.6859650810873743` on
train, and `0.5157964433043273` on validation. Real V2 curiosity-weighted
residual training then started as
`phase07_v2_curiosity_weighted_residual_adapter_v1_train_20260628` in Slurm
job `155749`. This is a policy update, but still not a success claim until
held-out videos and comparisons pass.

V2 held-out evaluation result on 2026-06-28: the matched 420-frame held-out
evaluation completed in tmux-held Slurm job `155749` and wrote
`experiments/outputs/phase07_v2_heldout_eval_v1_20260628_summary.json`,
`experiments/reports/2026-06-28_phase07_v2_heldout_eval_v1.md`, nine full
rollout GIFs, and manual visual evidence at
`experiments/outputs/phase07_v2_heldout_eval_v1_20260628_manual_visual_inspection.json`.
Manual inspection passed only as render evidence: all contact sheets were
nonblank and showed the robot, object, and multi-camera views. The performance
result is negative/incomplete, not success. Status is `open_not_satisfied`:
curiosity-weighted residual did not beat `no_adaptation` on all three held-out
cells and did not beat `no_curiosity_residual` without safety regression. This
means the V2 wider stabilization range did not solve the hard-training gate.
The next faithful direction is objective repair: preserve the strong
no-adaptation hold/lift behavior while adding curiosity-driven safety gains,
rather than reporting threshold tuning or checkpoint existence as completion.
Every later queue completion must rerun this classifier before any completion
language is used.

Evidence-refresh update on 2026-06-28 04:15-04:43 CST: a compute-side
allocation-only refresh path was added and run:
`experiments/configs/run_phase07_evidence_refresh_in_alloc.sh` with launcher
`experiments/configs/launch_phase07_evidence_refresh_tmux.sh`. It ran in CPU
Slurm job `155732` on `server13` through tmux session
`curiosity_phase07_evidence_refresh_alloc_20260628`, log
`logs/newton/phase07_evidence_refresh_v1_20260628.log`, and summary
`experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json`.
This refresh did not train, render, run inference, or claim success. It
refreshed action-bridge backfill, mainstream adapter conversion preflight,
stage-1 dataset indices, no-held-out-leakage audit, official-method readiness,
held-out comparison, and the hard evidence gate inside the allocation. The
stale missing manual-visual issue is now cleared:
`evaluation_evidence_status=pass` and
`heldout_missing_or_failed_entry_count=0`. The final hard gate remains
`open_not_satisfied` with `final_curiosity_success_allowed=false` because the
curiosity-weighted policy still does not beat the strongest declared baseline
without safety regression, the held-out comparison report is still not passing,
and official/mainstream method readiness remains open.

Action-bridge backfill update: the allocation-only backfill now covers the
train/validation source NPZs and every existing held-out method that already
has Phase07 NPZ evidence (`no_adaptation`, `scripted_feedback`,
`residual_baseline`, `curiosity_weighted`, `random_intrinsic`, and
`object_only`). The evidence gate can accept those provenance-preserving
backfilled NPZs for old evaluations, while newly generated ablation rollouts
must contain `candidate.action.*` directly from the exporter.

Mainstream dataset-index update: an allocation-only stage-1 index builder now
exists at
`experiments/configs/build_phase07_mainstream_stage1_dataset_index_v1.py` with
runner
`experiments/configs/run_phase07_mainstream_stage1_dataset_index_in_alloc.sh`.
It runs after action-bridge backfill and before the remaining ablation queue.
It writes method-specific index/config files for OpenPI LeRobot, GR00T
LeRobot-v2/modality, Diffusion Policy `shape_meta`, and RT-X RGB/task/7D-action
mapping. This is not full official dataset materialization, not inference, not
training, and not a mainstream comparison pass; it is a required bridge toward
faithful official-method comparison without leaking held-out episodes into
training or normalization.
The queue now runs a stage-1 leakage audit immediately after stage-1 index
creation:
`experiments/configs/audit_phase07_stage1_no_heldout_leakage_v1.py` through
`experiments/configs/run_phase07_stage1_no_heldout_leakage_in_alloc.sh`.
It requires held-out episodes to remain `held_out_eval_only` and marked
training-forbidden across the main index and all method-specific indices. A
failed leakage audit stops the allocation queue before ablation training.
The stage-1 builder also writes explicit split files under `splits/train.jsonl`,
`splits/validation.jsonl`, and `splits/held_out_eval_only.jsonl`; later
official dataset materializers should use those split files instead of the
combined index when computing training data or normalization statistics.

Official-method readiness update: a read-only gate now exists at
`experiments/configs/audit_phase07_official_method_readiness_v1.py`, with
output `experiments/outputs/phase07_official_method_readiness_v1_20260627.json`
and report
`experiments/reports/2026-06-27_phase07_official_method_readiness_v1.md`.
This gate checks each mainstream candidate for an official repo checkout,
prepared environment under `envs/`, official checkpoint or documented blocker,
stage-1 dataset index, and a closed-loop Phase07 comparison runner. A repo
clone, bridge spec, or dataset index alone cannot satisfy the mainstream gate.
The allocation runner
`experiments/configs/run_phase07_official_method_readiness_in_alloc.sh` is now
attached immediately before the hard evidence gate in the remaining-ablation
queue, so the final queue audit uses the latest stage-1 dataset-index status
instead of stale login-node readiness output.
The structured environment/checkpoint preparation plan is
`experiments/configs/phase07_official_method_env_checkpoint_plan_v1.json`.
It explicitly keeps environment creation under `envs/` on the shared
filesystem before compute use and records that official checkpoint access or a
faithful documented blocker is required before any mainstream comparison claim.
The local environment preparation entry point is
`experiments/configs/prepare_phase07_official_method_envs_local.sh`; it is
dry-run by default, refuses to run inside Slurm, and requires
`RUN_ENV_INSTALL=1` before creating official-method environments under `envs/`.
Checkpoint blocker templates are generated by
`experiments/configs/build_phase07_official_checkpoint_blocker_templates_v1.py`;
unfilled templates do not count as blockers or mainstream evidence.
Official checkpoint remote-entry access is probed by
`experiments/configs/audit_phase07_official_checkpoint_access_v1.py`, which
uses `gsutil ls` or HTTP HEAD/GET probes without downloading checkpoint files.
Passing this probe only shows that a remote entry point is reachable; actual
checkpoint files under `checkpoints/` or a filled blocker are still required.
Official comparison runner gates now exist for OpenPI, GR00T, Diffusion Policy,
and RT-X under `experiments/configs/run_phase07_*_official_comparison_in_alloc.sh`.
They are hard guards, not toy evaluators: they require Slurm allocation,
official environment, checkpoint or filled blocker, stage-1 indices, and
no-held-out-leakage proof before method-specific official inference/fine-tune
code can run.

Held-out comparison aggregation update: the queue now writes
`experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json` and
`experiments/reports/2026-06-27_phase07_heldout_comparison_report_v1.md` via
`experiments/configs/build_phase07_heldout_comparison_report_v1.py`. This
aggregates per-cell metrics, full-video evidence paths, visual inspection
status, missing ablations, strongest baseline selection, and curiosity safety
regressions before the hard evidence gate runs.
The 2026-06-28 evidence refresh reran this report after manual visual JSONs
were added for the remaining ablations. It now reports
`missing_or_failed_entry_count=0`, but still reports
`curiosity_beats_all_strongest_baselines_without_safety_regression=false`, so
this is clean negative comparison evidence, not a success result.

Closed-loop repair setup on 2026-06-28: after the refreshed hard gate showed
clean negative evidence, a validation-only threshold-repair loop was added at
`experiments/configs/phase07_closed_loop_threshold_repair_v1.json` with runner
`experiments/configs/run_phase07_closed_loop_threshold_repair_in_alloc.sh` and
launcher
`experiments/configs/launch_phase07_closed_loop_threshold_repair_tmux.sh`.
This is not a training result and not a success claim. It is an interaction
repair step: run the current Phase07 curiosity-weighted checkpoint on the two
validation cells (`empty_medium_hidden` and `full_medium_misleading`) at active
thresholds `0.5`, `0.65`, `0.8`, and `0.95`, with full 360-frame rollout
videos, then select a safer threshold using validation evidence only. The held
out cells remain forbidden for threshold selection. GPU allocation job `155734`
was submitted in tmux session
`curiosity_phase07_threshold_repair_alloc_20260628` and is pending before the
repair runner starts.

Closed-loop repair result on 2026-06-28: job `155734` later ran on `server37`
and completed the validation threshold sweep plus an initial held-out
re-evaluation. Eight validation rollouts were generated with complete
360-frame videos; the initial summary selected threshold `0.8`. A selector
audit then found a bug: the code ranked lift before acceleration even though
the config stated safety-first. The scripts were patched to rank acceleration
safety before lift after success/status/hold ties. The `0.8` held-out result is
therefore retained as diagnostic evidence only. It remained
`open_not_satisfied`: it did not beat no-adaptation on any held-out cell and
`empty_high_misleading` still had `max_object_accel_m_s2_regression`. A
corrected safety-first held-out re-evaluation with threshold `0.65` later ran
as Slurm job `155742` in tmux session
`curiosity_phase07_threshold065_eval_alloc_20260628`; the allocation was
released after completion. The corrected retry summary is
`experiments/outputs/phase07_threshold065_heldout_eval_retry_v1_20260628_summary.json`.
It remains `open_not_satisfied`: all three held-out cells completed with full
videos and no listed safety regression, but all three failed to beat
`no_adaptation` on the ordered held-out comparison.

## Promotion Rule From Simple Tasks

Do not advance a task family from diagnostic to training unless the current
task has:

- fresh official Newton sanity;
- visual validation and manual inspection;
- full rollout video or dense-frame video-equivalent evidence;
- strict lift/hold/slip/drop/contact/acceleration metrics;
- no held-out leakage;
- direct visual paths;
- clear comparison against no-adaptation, scripted feedback, residual adapter
  without curiosity, curiosity-trained policy, curiosity ablations, and
  serious/mainstream reference methods when available.

## Harder Task Families

### Variable Water-Cup Weight And Fill

This is the first harder task requested by the user.

The task should include cups with:

- empty, quarter, half, three-quarter, and full mass settings;
- low, medium, and high friction;
- randomized initial pose and handle orientation when handles are available;
- visual fill-level cues that can be misleading or hidden;
- held-out mass/friction/fill combinations.

The agent must infer physical response from lift/contact/tactile evidence and
adapt grip, lift speed, stabilization, or regrasp timing. Success requires
more than lifting one mass setting: it must handle unseen weight/fill
combinations without excessive slip, drop, or force.

This family is the first mandatory harder-task target for the next training
round. It must include ordinary train/validation cells and held-out cells with
unseen mass/friction/fill combinations. Held-out cells must not be used for
label construction, training, hyperparameter selection, threshold tuning, or
controller-gate tuning.

Current source-collection status as of 2026-06-27: four train cells have
passed the Phase07 scripted-feedback source gate:
`quarter_low_truthful`, `quarter_medium_hidden`, `half_low_hidden`, and
`half_medium_truthful`. The gated source-runner output is
`data/processed/phase07_residual_label_source_runner_v1_20260627/manifest.json`
with `source_run_count=4`, `record_count=1440`, `failures=[]`, and held-out
cells still reserved. This is not a training result. It is the source-data
foundation for later harder-task residual and curiosity training. Visual fill
cue conditions are currently recorded under `candidate.task.*` provenance only;
they are not rendered as physical liquid, hidden occluders, or misleading
visual evidence yet, so the visual-cue implementation remains a blocker for a
full Phase07 claim.

Later 2026-06-27 update: the planned train/validation source set now has all
eight promoted source candidates:
`quarter_low_truthful`, `quarter_medium_hidden`, `half_low_hidden`,
`half_medium_truthful`, `three_quarter_medium_misleading`,
`three_quarter_high_truthful`, `empty_medium_hidden`, and
`full_medium_misleading`. The Phase07 source runner passed after fresh
official Newton sanity with `source_run_count=8`, `record_count=2880`,
`total_feedback_active_frames=1927`, `failures=[]`, and held-out cells still
reserved as `empty_high_misleading`, `full_low_hidden`, and
`three_quarter_low_misleading`. This clears the train/validation source-data
gate for the first hard-training contract. It does not clear the final Phase07
claim because the visual fill cue is still metadata-only and no complete
closed-loop curiosity training has run on the harder task.

Training-contract update: `experiments/configs/phase07_hard_training_contract_v1.json`
now defines the no-downgrade contract for this harder-task run. It fixes the
six train cells, two validation cells, three held-out cells, baseline and
ablation requirements, one-GPU-hour minimum training budget, video evidence
requirements, and forbidden success claims. Phase07-specific configs were
added for residual-adapter preflight/training, curiosity-forward preflight/
training, learning-progress scoring, and curiosity-weighted residual training.

Current execution state: Phase07 residual-adapter preflight passed after fresh
official Newton sanity with 2160 train rows and 720 validation rows at
`data/processed/phase07_residual_adapter_training_preflight_v1_20260627/manifest.json`.
The no-curiosity Phase07 residual baseline training run
`phase07_residual_adapter_v1_train_20260627` has started from
`experiments/configs/phase07_residual_adapter_trainer_v1.json`. This is a
required learned baseline for later curiosity comparison, not a curiosity
success claim.

Held-out evaluation update: the Phase07 no-curiosity residual baseline,
curiosity forward model, learning-progress scoring, and curiosity-weighted
residual trainer all completed real one-GPU-hour training/scoring gates. The
held-out cells `empty_high_misleading`, `full_low_hidden`, and
`three_quarter_low_misleading` were evaluated with no-adaptation,
scripted-feedback, no-curiosity residual, and curiosity-weighted residual
policies, each with a 360-frame rollout GIF. The current result is negative:
curiosity-weighted residual does not beat the strongest baseline because the
no-adaptation scripted prior has higher final lift and longer hold on all
three held-out cells. Details are recorded in
`experiments/reports/2026-06-27_phase07_heldout_curiosity_weighted_eval_v1.md`.
Manual visual JSONs, standard metrics CSV/JSON, and acceleration peak analyses
have now been added for all 12 held-out policy videos. The added safety metrics
confirm the negative interpretation: curiosity-weighted residual does not
recover the no-adaptation baseline's lift/hold advantage and has the highest
acceleration peak on `full_low_hidden`. Phase07 remains incomplete; the next
work is proper ablation runs, serious/mainstream reference comparison, visual
fill cue repair or explicit blocker documentation, and a stronger closed-loop
curiosity update rather than claiming this run as success.

Random-intrinsic ablation update: six ablation score CSV variants were generated
from real Phase07 Newton transition records under
`experiments/outputs/phase07_curiosity_ablation_scores_v1_20260627/` with a
fresh official Newton sanity pass and `policy_updated=false`. The
`random_intrinsic` ablation then completed a real one-GPU-hour residual adapter
training run:
`phase07_random_intrinsic_residual_adapter_v1_train_20260627`, checkpoint
`checkpoints/phase07_random_intrinsic_residual_adapter_trainer_v1_20260627/phase07_random_intrinsic_residual_adapter_v1_train_20260627.pt`,
`optimizer_steps=21642`, mean GPU utilization `98.88333333333334%`. It was
evaluated on the three held-out cells with full 360-frame videos, manual visual
JSONs, standard metrics, and acceleration peak analysis. It also does not beat
the no-adaptation baseline, so this strengthens the failure diagnosis rather
than closing Phase07.

Object-only ablation update: `phase07_object_only_residual_adapter_v1_train_20260627`
completed a real one-GPU-hour residual adapter training run with checkpoint
`checkpoints/phase07_object_only_residual_adapter_trainer_v1_20260627/phase07_object_only_residual_adapter_v1_train_20260627.pt`,
`optimizer_steps=21634`, and mean GPU utilization `98.91666666666667%`. It was
evaluated on the same three held-out cells with full 360-frame videos, manual
visual JSONs, standard metrics, and acceleration peak analysis. It also does
not beat no-adaptation and is below curiosity-weighted residual on two of the
three held-out lifts, so object-motion-only intrinsic weighting is insufficient
and Phase07 remains open.

Contact-only retry update: `experiments/configs/phase07_contact_only_residual_adapter_trainer_v1.json`
is prepared. The first training attempt
`phase07_contact_only_residual_adapter_v1_train_20260627` passed fresh official
Newton sanity but failed before a valid training result with CUDA OOM because
allocation `154290` was occupied by an OpenPI process using about 129393 MiB.
OpenPI is outside Curiosity resources and must not be stopped or modified. The
resource conflict is recorded in
`experiments/outputs/phase07_contact_only_residual_adapter_v1_train_20260627_resource_conflict.json`.
A new Curiosity-dedicated tmux-held allocation was requested in session
`curiosity_contact_ablation_alloc_20260627_222339`, Slurm job `155039`.
Live update 2026-06-27 23:54 CST: job `155039` is running on `server07`, the
remaining-ablation queue passed the fresh official Newton sensor/contact
sanity check, action-bridge backfill, mainstream adapter conversion preflight,
stage-1 dataset indexing, and no-held-out-leakage audit. It is currently in
`phase07_contact_only_residual_adapter_v1_train_retry_20260627`; GPU
utilization is 99-100% after startup, but no valid retry summary/checkpoint has
been produced yet. This is in-progress training evidence only, not curiosity
completion.

Contact-only completion update 2026-06-28 00:51 CST: the retry completed a real
one-GPU-hour training run with summary
`experiments/outputs/phase07_contact_only_residual_adapter_trainer_v1_20260627/phase07_contact_only_residual_adapter_v1_train_retry_20260627_summary.json`,
checkpoint
`checkpoints/phase07_contact_only_residual_adapter_trainer_v1_20260627/phase07_contact_only_residual_adapter_v1_train_retry_20260627.pt`,
`elapsed_seconds=3600.155266523361`, `optimizer_steps=21494`,
`real_training_result=true`, `checkpoint_written=true`, and mean GPU
utilization `98.80165289256199%`. The queue then evaluated all three held-out
cells with 360-frame video export, metrics, acceleration analysis, and
candidate action-bridge validation. Metrics pass on `empty_high_misleading`,
`full_low_hidden`, and `three_quarter_low_misleading`, but manual visual
inspection is still `pending_direct_agent_check` and contact-only remains an
ablation result. It does not close Phase07 and does not prove curiosity beats
the strongest baseline.
Manual visual update 2026-06-28 00:54 CST: direct contact-sheet inspection for
the three contact-only held-out videos found nonblank three-camera triptychs,
visible gripper/object, start/middle/final frames, and no obvious drop or
render failure. Manual inspection JSONs were added for all three cells under
`experiments/outputs/phase07_eval_*_contact_only_20260627_manual_visual_inspection.json`.
They are marked as contact-only ablation evidence with
`curiosity_success_claim_valid=false`.

Shuffled-contact completion update 2026-06-28 01:59 CST: the shuffled-contact
ablation completed a real one-GPU-hour training run with summary
`experiments/outputs/phase07_shuffled_contact_residual_adapter_trainer_v1_20260627/phase07_shuffled_contact_residual_adapter_v1_train_20260627_summary.json`,
checkpoint
`checkpoints/phase07_shuffled_contact_residual_adapter_trainer_v1_20260627/phase07_shuffled_contact_residual_adapter_v1_train_20260627.pt`,
`elapsed_seconds=3600.129097223282`, `optimizer_steps=21651`,
`real_training_result=true`, `checkpoint_written=true`, and mean GPU
utilization `98.875%`. The three held-out evaluations wrote passing metrics,
360-frame videos, acceleration analysis, candidate action-bridge validation,
and manual visual inspection JSONs. This is ablation evidence only and still
does not prove the final curiosity claim.

Delayed-contact completion update 2026-06-28 03:02 CST: the delayed-contact
ablation completed a real one-GPU-hour training run with summary
`experiments/outputs/phase07_delayed_contact_residual_adapter_trainer_v1_20260627/phase07_delayed_contact_residual_adapter_v1_train_20260627_summary.json`,
checkpoint
`checkpoints/phase07_delayed_contact_residual_adapter_trainer_v1_20260627/phase07_delayed_contact_residual_adapter_v1_train_20260627.pt`,
`elapsed_seconds=3600.1080799102783`, `optimizer_steps=21745`,
`real_training_result=true`, `checkpoint_written=true`, and mean GPU
utilization `98.825%`. The three held-out evaluations wrote passing metrics,
360-frame videos, acceleration analysis, candidate action-bridge validation,
and manual visual inspection JSONs. This remains ablation evidence only.

No-learning-progress completion update 2026-06-28 04:06 CST: the
no-learning-progress ablation completed a real one-GPU-hour training run with
summary
`experiments/outputs/phase07_no_learning_progress_residual_adapter_trainer_v1_20260627/phase07_no_learning_progress_residual_adapter_v1_train_20260627_summary.json`,
checkpoint
`checkpoints/phase07_no_learning_progress_residual_adapter_trainer_v1_20260627/phase07_no_learning_progress_residual_adapter_v1_train_20260627.pt`,
`elapsed_seconds=3600.085962533951`, `optimizer_steps=21607`,
`real_training_result=true`, `checkpoint_written=true`, and mean GPU
utilization `99.0%`. The three held-out evaluations wrote passing metrics,
360-frame videos, acceleration analysis, candidate action-bridge validation,
and manual visual inspection JSONs. The allocation queue also reran official
method readiness, held-out comparison, and hard evidence gate, but that gate
still reports `final_curiosity_success_allowed=false` because curiosity does
not beat the strongest baseline and the official/mainstream comparison gate is
not ready. The gate run happened before the no-learning-progress manual JSONs
were added, so its missing-manual item is stale; the negative curiosity and
official-method blockers remain valid.

Remaining-ablation config update: training configs are now prepared and
syntax-checked for `shuffled_contact`, `delayed_contact`, and
`no_learning_progress`:
`experiments/configs/phase07_shuffled_contact_residual_adapter_trainer_v1.json`,
`experiments/configs/phase07_delayed_contact_residual_adapter_trainer_v1.json`,
and `experiments/configs/phase07_no_learning_progress_residual_adapter_trainer_v1.json`.
Their score summaries already exist under
`experiments/outputs/phase07_curiosity_ablation_scores_v1_20260627/` with
`policy_updated=false`. They still require real one-GPU-hour training and
held-out video evaluation before they count as completed ablations.

Remaining-ablation queue update: the allocation-internal runner
`experiments/configs/run_phase07_remaining_ablation_queue_in_alloc.sh` and
launcher `experiments/configs/launch_phase07_remaining_ablation_queue_tmux.sh`
now encode the next faithful step. Once dedicated Curiosity allocation `155039`
is running, the queue will run contact-only retry plus shuffled-contact,
delayed-contact, and no-learning-progress ablation training, then evaluate each
checkpoint on `empty_high_misleading`, `full_low_hidden`, and
`three_quarter_low_misleading` with 360-frame rollout videos, metrics, and
acceleration analysis. Manual visual inspection and report updates remain
required after the queue; this queue is not a final curiosity-success claim.
The lightweight watcher
`experiments/configs/watch_phase07_remaining_ablation_queue_autolaunch.sh`
exists only to wait for `155039` and submit the queue into the held allocation;
it performs no training or rendering on the login node.
The queue also writes `logs/newton/*_env.sh` files for each ablation training
and held-out evaluation so the exact config, checkpoint, held-out cell
parameters, and video settings remain reproducible.

Mainstream comparison gate update: `experiments/configs/phase07_mainstream_comparison_gate_v1.json`
records the current serious-method comparison requirement as of 2026-06-27.
The gate names OpenPI/pi0, Diffusion Policy, Open X/RT-X, and NVIDIA Isaac
GR00T as candidates that must be either faithfully compared through official
code/checkpoints or explicitly blocked by compatibility, licensing,
dependency, embodiment, or action-space evidence. Phase07 cannot claim final
curiosity success while this gate remains unsatisfied, and no toy substitute
may be used to fill the mainstream-method slot.

Mainstream audit update: `experiments/outputs/phase07_mainstream_comparison_audit_v1_20260627.json`
and `experiments/reports/2026-06-27_phase07_mainstream_comparison_audit_v1.md`
record that the official OpenPI/pi0, Diffusion Policy, Open X/RT-X, and
NVIDIA Isaac GR00T repositories are reachable, but no faithful Phase07
comparison or concrete official incompatibility blocker has been completed.
This strengthens the open gate: repository reachability is not comparison.
The repeatable lightweight checker
`experiments/configs/audit_phase07_mainstream_repos_v1.py` writes
`experiments/outputs/phase07_mainstream_repo_reachability_audit_v1_20260627.json`
and is used to track official local clone and checkpoint availability inside
the Curiosity workspace.

Official-code compatibility update: the official OpenPI/pi0, Diffusion Policy,
Open X/RT-X, and NVIDIA Isaac GR00T repositories are now shallow-cloned under
`external/` using `GIT_LFS_SKIP_SMUDGE=1`; no large model weights were
downloaded. The matrix
`experiments/configs/phase07_mainstream_official_code_compatibility_matrix_v1.json`
and report
`experiments/reports/2026-06-27_phase07_mainstream_official_code_compatibility_matrix_v1.md`
record commits, submodule state, official checkpoint identifiers, and the
Phase07 observation/action compatibility gaps. This still does not satisfy the
mainstream comparison gate because no official checkpoint has been run and no
concrete incompatibility blocker has been completed.

Adapter-bridge update: `experiments/configs/phase07_mainstream_adapter_bridge_spec_v1.json`
and `experiments/reports/2026-06-27_phase07_mainstream_adapter_bridge_spec_v1.md`
define the required Phase07 mappings for OpenPI/pi0, GR00T, Diffusion Policy,
and RT-X. The key decision is that a faithful mainstream comparison should use
a low-level or 7D relative EEF/gripper action bridge where possible. Training a
mainstream codebase to imitate the current 4D residual controller parameters
would be diagnostic only and cannot satisfy the mainstream gate.

Bridge readiness audit update: `experiments/configs/audit_phase07_mainstream_bridge_readiness_v1.py`
ran a lightweight schema/file-presence audit and wrote
`experiments/outputs/phase07_mainstream_bridge_readiness_audit_v1_20260627.json`.
It found all required Phase07 source/context columns and existing held-out
summary/NPZ/video artifacts for current no-adaptation and curiosity-weighted
runs, but the source CSV lacks the preferred 7D EEF/gripper action columns:
`candidate.action.eef_delta_x/y/z/roll/pitch/yaw` and
`candidate.action.gripper`. Therefore OpenPI, GR00T, Diffusion Policy, and RT-X
remain blocked on a provenance-preserving Newton Panda EEF/gripper action
bridge or a concrete official incompatibility blocker.

Action-bridge implementation update: `experiments/configs/newton_panda_hydro_tiled_camera_export.py`
now emits `candidate.action.eef_delta_x/y/z/roll/pitch/yaw`,
`candidate.action.gripper`, and `candidate.action.eef_delta_xyzrpy_gripper`
for future rollout NPZs by finite-differencing `newton.panda.ee_body_q` and
using `candidate.controller.commanded_gripper_target`. This does not retrofit
older NPZs and does not complete the mainstream gate, but it means the next
Phase07 reruns can produce bridge-bearing artifacts for official mainstream
adapter conversion.
The remaining-ablation queue now validates these fields after each new held-out
evaluation and writes `experiments/outputs/<run_tag>_candidate_action_bridge_validation.json`.
If any bridge field is missing, the queue fails instead of producing a silent
mainstream-incompatible artifact.

Existing-artifact backfill update: `experiments/configs/backfill_phase07_candidate_action_bridge_v1.py`
and `experiments/configs/run_phase07_candidate_action_bridge_backfill_in_alloc.sh`
prepare allocation-only backfill for the six existing no-adaptation and
curiosity-weighted held-out NPZs. The backfill writes bridge-bearing copies
under `experiments/outputs/phase07_action_bridge_backfill_v1_20260627/` and
preserves the original NPZs. This backfill is now the first step in the
remaining-ablation queue, so the existing baseline artifacts can later feed
mainstream adapter conversion without rerendering videos.

Mainstream conversion preflight update:
`experiments/configs/build_phase07_mainstream_adapter_conversion_preflight_v1.py`
and `experiments/configs/run_phase07_mainstream_adapter_conversion_preflight_in_alloc.sh`
are prepared as the next allocation-only step after backfill. They validate the
bridge-bearing NPZ shapes and write
`experiments/outputs/phase07_mainstream_adapter_conversion_preflight_v1_20260627/manifest.json`
with OpenPI, GR00T, Diffusion Policy, and RT-X mapping specs. This is not full
dataset conversion, training, inference, or a success claim; it is the required
preflight before faithful official comparisons.

### Slippery And Low-Contact Objects

Examples: smooth cup, metal cylinder, laminated card, thin box, and object
with small contact patch.

Key failures:

- insufficient grip;
- slip after lift;
- delayed contact loss;
- over-squeezing to compensate.

### Deformable Or Compliant Objects

Examples: pouch, sponge block, soft bottle, partially filled container.

Key failures:

- visual shape does not predict contact response;
- deformation changes grasp stability;
- excessive force damages the object.

### Handled And Off-Center Objects

Examples: mug with handle, small basket, object with asymmetric center of
mass, object with handle not aligned to camera.

Key failures:

- wrong grasp point;
- torque-induced slip;
- lift succeeds but hold fails.

### Fragile Or Safety-Constrained Objects

Examples: crushable cup, thin shell, object with force limit.

Key failures:

- task success by brute force;
- excessive contact proxy;
- visually successful lift with unacceptable safety cost.

## Required Comparisons

Each harder task family should compare:

- no-adaptation scripted prior;
- scripted feedback baseline;
- residual adapter without curiosity;
- curiosity-trained policy;
- random intrinsic reward;
- object-only curiosity;
- contact-only curiosity;
- vision/contact curiosity;
- tactile-aware curiosity once real tactile evidence exists.
- current serious/mainstream method or official checkpoint baseline when a
  faithful compatible implementation is available. If no faithful compatible
  implementation exists, record the audit and blocker explicitly instead of
  replacing it with a toy model.

## Completion Criteria

Phase 07 is not complete until at least one harder task family has:

- generated source data under explicit Newton/Taccel namespaces;
- trained or adapted the curiosity path in a complete closed loop without
  held-out leakage;
- passed visual and strict metric gates on held-out variants;
- produced full rollout videos for the evaluated policies;
- produced a failure-mode comparison report;
- documented whether curiosity helps beyond the non-curiosity residual
  adapter;
- compared against serious/mainstream reference methods or recorded an
  explicit faithful-comparison blocker;
- shown improvement over the declared baseline without hiding safety failures.

Do not claim broad manipulation generalization from the current cup benchmark.
Do not claim Phase 07 completion if the result is only a pipeline smoke test,
only a contact-sheet visualization, only a supervised reweighting run, or only
performance parity with the baseline.

## 2026-06-28 V2 Anchor Objective Repair Result

The neutral-residual baseline-preservation anchor was implemented only as an
objective repair attempt after the negative V2 held-out result. Its smoke check
confirmed the anchor was active with `train_anchor_weight_mean=0.3271425664424896`.
The real anchor training ran for one GPU-hour in Slurm job `155749` and wrote:

- checkpoint:
  `checkpoints/phase07_v2_curiosity_weighted_residual_adapter_anchor_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_anchor_v1_train_20260628.pt`;
- summary:
  `experiments/outputs/phase07_v2_curiosity_weighted_residual_adapter_anchor_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_anchor_v1_train_20260628_summary.json`;
- log:
  `logs/newton/phase07_v2_curiosity_weighted_residual_adapter_anchor_v1_train_20260628.log`.

The training artifact is valid, but its validation metrics are poor
(`active_accuracy=0.6595238447189331`, `continuous_mse=1.8517640829086304`,
`loss=4.608899116516113`). The held-out anchor evaluation
`phase07_v2_anchor_heldout_eval_v1_20260628` also failed:

- status: `open_not_satisfied`;
- curiosity beats no-adaptation all cells without safety regression: `false`;
- curiosity beats no-curiosity residual all cells without safety regression:
  `false`;
- manual visual inspection: all nine contact sheets passed as nonblank
  multi-camera robot/object rollouts;
- report:
  `experiments/reports/2026-06-28_phase07_v2_anchor_heldout_eval_v1.md`;
- manual inspection:
  `experiments/outputs/phase07_v2_anchor_heldout_eval_v1_20260628_manual_visual_inspection.json`.

This is negative evidence. It must not be reported as completed curiosity
training. The next repair should preserve baseline behavior more selectively,
for example by reducing or separating the continuous neutral-anchor term and
adding explicit safety/hold-duration objectives instead of forcing all stable
contact frames toward a neutral residual.

Immediate follow-up: the trainer now supports separated
`active_loss_weight` and `continuous_loss_weights` for the baseline-preservation
anchor. The next real attempt is
`phase07_v2_curiosity_weighted_residual_adapter_active_anchor_trainer_v1`,
which keeps the same Newton-native GRU residual adapter, same V2 data, same
held-out cells, and same video/metric gate, but uses a softer active-only
anchor (`anchor_strength=0.35`, `inverse_curiosity_power=2.0`) instead of
forcing all continuous residual outputs toward neutral on stable-contact
frames.

Execution request: one-day tmux-held GPU allocation requested as Slurm job
`155785` in session
`curiosity_phase07_active_anchor_alloc_20260628_101625` for the active-anchor
smoke, real training, and held-out video evaluation sequence. This is not a
one-shot `sbatch` run and is not a success claim.

Execution progress: active-anchor smoke
`phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_smoke_20260628`
passed after fresh official Newton sensor-contact sanity, with
`smoke_diagnostic_only=true`, `checkpoint_written=false`, and validation
`active_accuracy=0.997619092464447`. Real one-hour training then started as
`phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628`
in the same tmux-held allocation. This is in progress and is not a success
claim until the checkpoint, GPU-utilization gate, held-out videos, metrics, and
manual visual inspection all exist.

Training result: active-anchor real training completed with
`real_training_result=true`, `elapsed_seconds=3600.1757838726044`,
`optimizer_steps=18649`, checkpoint
`checkpoints/phase07_v2_curiosity_weighted_residual_adapter_active_anchor_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.pt`,
and mean GPU utilization `98.79166666666667%`. Validation remained weak
(`active_accuracy=0.663095235824585`, `active_bce=0.6922045350074768`), so this
is only a valid training artifact, not evidence that curiosity succeeds.
Held-out evaluation started as
`phase07_v2_active_anchor_heldout_eval_v1_20260628` using the fixed V2 harder
cells, full rollout videos, and the same no-adaptation/no-curiosity baselines.

Held-out result: `phase07_v2_active_anchor_heldout_eval_v1_20260628` completed
9/9 videos and was directly visually inspected. It is still
`open_not_satisfied`: curiosity did not beat no-adaptation across all cells and
did not beat no-curiosity residual without safety regression. The failure is
not a visualization artifact. Representative failures:

- `empty_high_misleading`: curiosity hold `3.3166635036468506` vs
  no-adaptation `4.133329391479492`, with acceleration regression
  `1.2795535523490018` vs `0.8367652074157562`;
- `full_low_hidden`: curiosity hold `3.233330249786377` vs no-adaptation
  `4.099996089935303`, and worse than no-curiosity on hold/lift/accel;
- `three_quarter_low_misleading`: curiosity hold `3.2666635513305664` vs
  no-adaptation `4.099996089935303`, with acceleration regression
  `2.7942004374800127` vs no-curiosity `1.854889805196864`.

This makes another anchor-only objective tweak weakly motivated. The next
faithful repair must inspect the learned-residual controller baseline
transition and ensure neutral/no-feedback learned-residual mode can reproduce
the official scripted no-adaptation waypoint/contact behavior before spending
another one-hour training run.

Controller repair in progress: `lift_hold_learned_residual` previously reused
`_configure_lift_hold_feedback_waypoints`, which changed the official lift/hold
trajectory before any learned residual activation by applying
`FEEDBACK_INITIAL_LIFT_DURATION_SCALE` and an initial stabilization extension.
This likely explains the systematic hold-duration gap versus no-adaptation.
The export code now initializes learned-residual mode from
`_configure_lift_hold_waypoints` and only modifies waypoints when the learned
adapter is active. Syntax check passed. A neutral parity eval was launched as
`phase07_v2_learned_neutral_parity_eval_v1_20260628` with
`ACTIVE_THRESHOLD=2.0` to verify that learned-residual neutral/no-feedback mode
can reproduce no-adaptation on the same harder held-out cells before another
training run is trusted.

Neutral parity result: the controller repair restored neutral hold-duration
parity. With `ACTIVE_THRESHOLD=2.0`, learned-residual/no-feedback hold matched
no-adaptation on the three harder cells (`4.133329391479492`,
`4.099996089935303`, `4.099996089935303`). This is a controller sanity result,
not curiosity success. A post-repair held-out eval using the active-anchor
checkpoint and `ACTIVE_THRESHOLD=0.5` was launched as
`phase07_v2_fixed_controller_active_anchor_heldout_eval_v1_20260628` to test
whether the trained residual helps after the neutral controller bias is removed.

Post-repair held-out result: `phase07_v2_fixed_controller_active_anchor_heldout_eval_v1_20260628`
completed 9/9 videos and direct visual inspection, but remains
`open_not_satisfied`. The controller repair improved hold relative to the old
learned-residual path, but the active checkpoint still fails the hard gate:

- `empty_high_misleading`: curiosity hold `3.9333295822143555` vs
  no-adaptation `4.133329391479492`, with acceleration regression
  `2.2832677871850904` vs `0.8367811268019772`;
- `full_low_hidden`: curiosity hold `3.8833296298980713` vs no-adaptation
  `4.099996089935303`, and safety regression vs no-curiosity residual;
- `three_quarter_low_misleading`: curiosity hold `3.8833296298980713` vs
  no-adaptation `4.099996089935303`, plus drop/acceleration regressions.

The next training run should not reuse the old residual source labels as-is.
Regenerate source labels from the repaired official no-adaptation base and
contact-trigger residual transition, then retrain curiosity from that aligned
source. This remains incomplete/negative evidence.

Repaired-base source collection started: added
`experiments/configs/phase07_v3_repaired_base_source_collection_v1.json`, which
keeps the V2 train/validation cells but disables initial feedback waypoint
adjustment (`feedback_apply_initial_waypoint_adjustment=false`). The runner now
passes this through to the Newton export script. Started source collection as
`phase07_v3_repaired_base_source_collection_v1_20260628` in Slurm job `155785`,
tmux window `phase07_v3_repaired_source`. This is data collection only and must
not be called training or success.

Repaired-base source collection completed with 8/8 rollouts. All eight contact
sheets were directly inspected and promoted to
`pass_nonblank_success_with_feedback`. The repaired-base source manifest is now
`experiments/configs/phase07_v3_repaired_base_source_manifest_v1.json` with
status `phase07_v3_repaired_base_source_candidates_complete_training_not_started`.
Generated V3 repaired-base source runner/preflight configs and launched source
runner `phase07_v3_repaired_base_residual_label_source_runner_v1_20260628`.

V3 repaired-base source runner result: source runner
`phase07_v3_repaired_base_residual_label_source_runner_v1_20260628` passed
after a fresh official Newton sensor-contact sanity check. It produced
`3360` source records from 8 source runs with
`total_feedback_active_frames=2405`, kept `schema_promotion=blocked`, and did
not start training. The output manifest is
`data/processed/phase07_v3_repaired_base_residual_label_source_runner_v1_20260628/manifest.json`.

V3 residual-adapter preflight result:
`phase07_v3_repaired_base_residual_adapter_training_preflight_v1_20260628`
passed after a fresh official Newton sensor-contact sanity check. The split is
`2520` train records across
`half_low_hidden`, `half_medium_truthful`, `quarter_low_truthful`,
`quarter_medium_hidden`, `three_quarter_high_truthful`, and
`three_quarter_medium_misleading`, plus `840` validation records across
`empty_medium_hidden` and `full_medium_misleading`. The harder held-out cells
`empty_high_misleading`, `full_low_hidden`, and
`three_quarter_low_misleading` remain reserved for evaluation. This is an input
audit only, not a model or curiosity result.

V3 repaired-base training configs added: no-curiosity residual baseline,
curiosity forward-model preflight, curiosity forward-model trainer,
learning-progress scoring, and active-anchor curiosity-weighted residual
fine-tune now have dedicated V3 configs under `experiments/configs/`. They all
point to repaired-base V3 manifests and preserve the one-hour real-training
gate where training occurs. The next running job is the V3 no-curiosity
residual baseline:
`phase07_v3_repaired_base_residual_adapter_v1_train_20260628`, launched in
Slurm job `155785`, tmux window `phase07_v3_residual_train`, with log
`logs/newton/phase07_v3_repaired_base_residual_adapter_v1_train_20260628.log`.
This baseline checkpoint is required before V3 curiosity fine-tuning and is
not a curiosity success claim.

V3 no-curiosity baseline training result:
`phase07_v3_repaired_base_residual_adapter_v1_train_20260628` passed as a real
one-hour training run with `elapsed_seconds=3600.1102674007416`,
`optimizer_steps=18651`, `real_training_result=true`, checkpoint
`checkpoints/phase07_v3_repaired_base_residual_adapter_trainer_v1_20260628/phase07_v3_repaired_base_residual_adapter_v1_train_20260628.pt`,
and mean GPU utilization `98.96666666666667%`. This is a repaired-base
no-curiosity baseline artifact only. It is not curiosity learning evidence.

V3 curiosity forward preflight result:
`phase07_v3_repaired_base_curiosity_forward_model_preflight_v1_20260628`
passed after fresh official Newton sanity. It produced `3352` transition
records (`2514` train, `838` validation), kept the harder held-out cells
reserved, did not create a model, and kept `schema_promotion=blocked`.
V3 forward-model real training started as
`phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628` in Slurm
job `155785`, tmux window `phase07_v3_forward_train`, with log
`logs/newton/phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628.log`.
This forward model is only the prerequisite for learning-progress scoring; it
is not a policy update or curiosity success claim.

V3 curiosity forward-model training result:
`phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628` passed as
a real one-hour training run with `elapsed_seconds=3600.0523829460144`,
`optimizer_steps=17989`, initial snapshot
`checkpoints/phase07_v3_repaired_base_curiosity_forward_model_v1_20260628/phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628_initial_snapshot.pt`,
trained checkpoint
`checkpoints/phase07_v3_repaired_base_curiosity_forward_model_v1_20260628/phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628.pt`,
and mean GPU utilization `98.9%`. This is a dynamics checkpoint only, not a
policy update.

V3 learning-progress scoring result:
`phase07_v3_repaired_base_curiosity_learning_progress_v1_20260628` passed
after fresh official Newton sanity and produced `3352` curiosity scores in
`experiments/outputs/phase07_v3_repaired_base_curiosity_learning_progress_v1_20260628/curiosity_learning_progress_scores.csv`.
Mean bounded curiosity reward was `0.7738659704284149`; train mean was
`0.8107618392954183` over `2514` scores and validation mean was
`0.6631783638273964` over `838` scores. `policy_updated=false`, so this still
is not policy training success. The V3 curiosity-weighted residual fine-tune
started as
`phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628`
in Slurm job `155785`, tmux window `phase07_v3_curiosity_residual_train`, with
log
`logs/newton/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.log`.
This is the first V3 repaired-base policy-update run, but it remains incomplete
until the checkpoint passes the one-hour/GPU gates and then beats the strongest
declared baselines on harder held-out video evaluation without safety
regression.

V3 curiosity-weighted residual policy fine-tune result:
`phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628`
passed as a real one-hour policy-update run with
`elapsed_seconds=3600.164398908615`, `optimizer_steps=18549`,
`train_score_coverage=0.9976190476190476`, checkpoint
`checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.pt`,
and mean GPU utilization `98.76666666666667%`. This is now a valid trained
curiosity-weighted policy artifact, but it is still not success evidence until
held-out comparison passes.

V3 repaired-base held-out evaluation started:
`phase07_v3_repaired_base_curiosity_heldout_eval_v1_20260628`, using eval tag
prefix `phase07_v3_repaired_base_eval`, active threshold `0.5`, no-curiosity
checkpoint
`checkpoints/phase07_v3_repaired_base_residual_adapter_trainer_v1_20260628/phase07_v3_repaired_base_residual_adapter_v1_train_20260628.pt`,
and curiosity checkpoint
`checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.pt`.
The output report path is
`experiments/reports/2026-06-28_phase07_v3_repaired_base_curiosity_heldout_eval_v1.md`.
This evaluation must still produce rollout videos, metrics, direct manual
visual inspection, and strongest-baseline comparison before any success claim.

V3 repaired-base held-out result:
`phase07_v3_repaired_base_curiosity_heldout_eval_v1_20260628` completed all
9/9 rollouts and direct visual inspection. Manual inspection JSON:
`experiments/outputs/phase07_v3_repaired_base_curiosity_heldout_eval_v1_20260628_manual_visual_inspection.json`.
All contact sheets were nonblank multi-camera robot/object rollouts. The
aggregate summary remains `open_not_satisfied`; both hard gates are false:
curiosity did not beat no-adaptation across all cells without safety regression
and did not beat the no-curiosity residual baseline across all cells without
safety regression.

Representative metrics:

- `empty_high_misleading`: curiosity hold `4.133329391479492`, lift
  `0.16140423715114594`, accel `1.3371639166010236`; no-adaptation hold
  `4.133329391479492`, lift `0.16634894907474518`, accel
  `0.8366925363338481`. Curiosity matches hold but has lower lift and an
  acceleration safety regression.
- `full_low_hidden`: curiosity hold `3.8666629791259766`, lift
  `0.15421180427074432`; no-adaptation hold `4.099996089935303`, lift
  `0.15921781957149506`; no-curiosity hold `3.8833296298980713`, lift
  `0.15448197722434998`. Curiosity is below both baselines on hold/lift.
- `three_quarter_low_misleading`: curiosity hold `3.8833296298980713`, lift
  `0.1553822010755539`; no-adaptation hold `4.099996089935303`, lift
  `0.16033704578876495`; no-curiosity residual has similar hold but lower slip
  (`0.0034037721706038906` vs curiosity `0.0034419198726162374`).

This is valid negative evidence, not completion. The next faithful step should
diagnose why repaired-base curiosity scoring/fine-tuning remains less selective
than the strongest baseline: inspect active frequency, residual magnitudes,
score distribution, and threshold sensitivity on validation cells before
another held-out run.

Immediate V3 diagnostic: learning-progress scores are poorly selective
(`41.795942720763724%` of all scores are `>=0.95`, median
`0.8553736656904221`), while the trained residual active validation accuracy is
only `0.6571428775787354` with active BCE `1.1270439624786377`. Held-out
rollouts show unstable threshold behavior: on `empty_high_misleading`,
curiosity active triggered for only `0.2380952380952381%` of frames, while on
`full_low_hidden` and `three_quarter_low_misleading` it triggered for about
`71.19047619047619%`. This supports a validation-only threshold/score
selectivity diagnostic before any further held-out tuning.

Started V3 validation-only threshold repair:
`phase07_v3_repaired_base_closed_loop_threshold_repair_v1_20260628`, using
config
`experiments/configs/phase07_v3_repaired_base_closed_loop_threshold_repair_v1.json`
and the V3 curiosity checkpoint. This runs only the validation cells
`empty_medium_hidden` and `full_medium_misleading` over thresholds
`0.5`, `0.65`, `0.8`, and `0.95`; held-out cells remain forbidden for threshold
selection. This is diagnostic/selection work only, not training and not a
success claim.

Validation-only threshold repair result:
`phase07_v3_repaired_base_closed_loop_threshold_repair_v1_20260628` completed
8/8 validation rollouts and selected threshold `0.95`. All validation rollouts
had `status_ok=true` and `success=true`, but the selected high threshold shows
that the current curiosity policy is safest when mostly suppressed rather than
more actively exploratory. This is not success evidence. A post-selection
held-out evaluation was launched as
`phase07_v3_repaired_base_threshold095_heldout_eval_v1_20260628` with eval tag
prefix `phase07_v3_repaired_base_thr095_eval` and active threshold `0.95`.
Because the threshold was selected on validation cells, this is allowed as a
held-out re-evaluation, but it still must beat no-adaptation and no-curiosity
baselines without safety regression before any success claim.

Threshold `0.95` held-out re-evaluation result:
`phase07_v3_repaired_base_threshold095_heldout_eval_v1_20260628` completed
9/9 rollouts and direct contact-sheet inspection. Manual inspection JSON:
`experiments/outputs/phase07_v3_repaired_base_threshold095_heldout_eval_v1_20260628_manual_visual_inspection.json`.
It remains `open_not_satisfied`. Threshold `0.95` made
`empty_high_misleading` nearly match no-adaptation, but `full_low_hidden` and
`three_quarter_low_misleading` remain below no-adaptation on hold/lift and add
acceleration regressions. This confirms threshold repair alone is not enough.

Next training-side repair started: the saturated V3 learning-progress scores
were rank-calibrated into
`experiments/outputs/phase07_v3_repaired_base_curiosity_learning_progress_rank_calibrated_v1_20260628/curiosity_learning_progress_summary.json`,
with `rank_calibrated_curiosity_reward` distributed from `0` to `1` within
train/validation splits. Added
`experiments/configs/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_trainer_v1.json`
and launched real one-hour training
`phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628`
in Slurm job `155785`, tmux window `phase07_v3_rank_residual_train`. This is a
faithful config-level repair of weighting/anchor behavior, not a new model and
not a success claim.

Rank-calibrated residual training result:
`phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628`
completed a real one-hour run with `elapsed_seconds=3600.0904426574707`,
`optimizer_steps=18576`, `train_score_coverage=0.9976190476190476`, checkpoint
`checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628.pt`,
and mean GPU utilization `98.73333333333333%`. Validation active behavior
improved substantially (`active_accuracy=0.9297619462013245`,
`active_bce=0.18072140216827393`), but this is still not success evidence
until held-out comparison passes. Held-out evaluation started as
`phase07_v3_repaired_base_rank_curiosity_heldout_eval_v1_20260628`, eval tag
prefix `phase07_v3_repaired_base_rank_eval`, active threshold `0.5`, comparing
no-adaptation, V3 no-curiosity residual, and the rank-calibrated curiosity
checkpoint.

Rank held-out eval first attempt failed before completion due to a loader
allow-list mismatch: the Newton export script did not yet accept checkpoint
classification
`newton_native_rank_calibrated_curiosity_weighted_residual_controller_adapter_v1_checkpoint`.
This was a glue compatibility issue, not held-out performance evidence. The
loader allow-list in
`experiments/configs/newton_panda_hydro_tiled_camera_export.py` was updated to
accept this classification, syntax check passed, and the evaluation was
restarted as
`phase07_v3_repaired_base_rank_curiosity_heldout_eval_retry_v1_20260628` with
eval tag prefix `phase07_v3_repaired_base_rank_retry_eval`.

Rank-calibrated held-out retry result:
`phase07_v3_repaired_base_rank_curiosity_heldout_eval_retry_v1_20260628`
completed all 9/9 harder held-out rollouts and direct contact-sheet inspection.
Manual inspection JSON:
`experiments/outputs/phase07_v3_repaired_base_rank_curiosity_heldout_eval_retry_v1_20260628_manual_visual_inspection.json`.
All contact sheets were nonblank multi-camera robot/object rollouts, so the
visual evidence is valid. The result remains `open_not_satisfied`, with both
hard all-cell gates false. This is valid negative evidence, not completion.

Representative rank-calibrated retry metrics:

- `empty_high_misleading`: rank curiosity hold `3.9333295822143555`, lift
  `0.16174010932445526`, accel `2.803110015763228`; no-adaptation hold
  `4.133329391479492`, lift `0.16636569797992706`, accel
  `0.8368612521882356`; no-curiosity residual hold `3.9333295822143555`,
  lift `0.1614381968975067`, accel `0.6390070039916543`.
- `full_low_hidden`: rank curiosity hold `3.8666629791259766`, lift
  `0.15421772003173828`; no-adaptation hold `4.099996089935303`, lift
  `0.159222811460495`; no-curiosity residual hold `3.8833296298980713`,
  lift `0.15442362427711487`.
- `three_quarter_low_misleading`: rank curiosity hold `3.8833296298980713`,
  lift `0.15547941625118256`, accel `1.3453099476479045`; no-adaptation hold
  `4.099996089935303`, lift `0.16082791984081268`, accel
  `2.3245204220789333`; no-curiosity residual hold `3.8833296298980713`,
  lift `0.1554972380399704`, accel `1.2168714296341352`.

Conclusion: rank calibration repaired validation active prediction quality, but
it did not produce harder held-out improvement over the strongest baseline.
The next faithful step is not another success claim or quick threshold tweak;
it is an objective/data repair that changes the residual target or active
intervention source so curiosity can improve no-adaptation hold/lift without
adding acceleration/safety regressions.

V3 closed-loop teacher repair started on 2026-06-28: added a DAgger-style
closed-loop data path rather than another offline score-reweighting pass. The
Newton export now supports `--record-scripted-teacher-labels`, which keeps the
learned residual checkpoint in control while recording non-applied scripted
corrective labels under `candidate.teacher.*`. Added preflight builder
`experiments/configs/build_phase07_closed_loop_teacher_preflight_v1.py`,
preflight config
`experiments/configs/phase07_v3_closed_loop_teacher_preflight_v1.json`,
trainer config
`experiments/configs/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1.json`,
and chain runner
`experiments/configs/run_phase07_v3_closed_loop_teacher_training_in_alloc.sh`.
The chain will collect eight train/validation on-policy rollouts, fail if
teacher labels are zero or held-out cells leak into training, train for at
least one GPU-hour, then run the same three-cell harder held-out video
evaluation. Static checks passed:
`py_compile` for the modified export and preflight builder, `bash -n` for the
runner/launcher scripts, JSON validation for the new configs, and both source
policy/no-curiosity checkpoints exist. Slurm job `156649` was requested in
tmux session `curiosity_phase07_closed_loop_teacher_alloc_20260628` for one
GPU/one day and is currently pending due to priority. This is active training
work in progress, not success evidence.

V3 closed-loop teacher source/preflight progress on 2026-06-28: Slurm job
`156649` started on `server57` and launched
`phase07_v3_closed_loop_teacher_chain_v1_20260628` in tmux window
`phase07_v3_closed_loop_teacher_chain`. All eight train/validation on-policy
source rollouts completed with fresh official Newton sanity, learned-residual
control, `scripted_teacher_labels.enabled=true`, non-applied
`candidate.teacher.*` labels, and video export pass. The preflight manifest
`data/processed/phase07_v3_closed_loop_teacher_preflight_v1_20260628/manifest.json`
passed with `2520` train rows, `840` validation rows,
`total_teacher_active_frames=2407`, and `failures=[]`. Training then entered
`phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628` after a
fresh official Newton sanity pass. This is now a real closed-loop-source policy
update in progress, not a completed curiosity result.

V3 closed-loop teacher training result on 2026-06-28:
`phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628` completed
a real one-hour policy update with `elapsed_seconds=3600.0265502929688`,
`optimizer_steps=18576`, checkpoint
`checkpoints/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1_20260628/phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628.pt`,
mean GPU utilization `98.56198347107438%`, and validation metrics
`active_accuracy=0.9952381253242493`, `active_bce=0.039532799273729324`,
`continuous_mse=0.0038401642814278603`. This clears the closed-loop-source
training gate but still is not a curiosity success claim. The chain has moved
into harder held-out full-video evaluation
`phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628` against
no-adaptation and V3 no-curiosity residual baselines.

V3 closed-loop teacher held-out result on 2026-06-28:
`phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628` completed all 9/9
harder held-out rollouts with full rollout GIFs and direct contact-sheet
inspection. Manual inspection JSON:
`experiments/outputs/phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628_manual_visual_inspection.json`.
All contact sheets are nonblank multi-camera robot/object rollouts. The result
remains `open_not_satisfied` and is negative evidence, not completion:

- `empty_high_misleading`: closed-loop teacher hold `3.9333295822143555`,
  lift `0.16152004897594452`, accel `0.6474675065114583`; no-adaptation hold
  `4.133329391479492`, lift `0.1663660854101181`, accel
  `0.8368541546954719`.
- `full_low_hidden`: closed-loop teacher hold `3.8666629791259766`, lift
  `0.1542869359254837`, accel `1.1031449367739388`; no-adaptation hold
  `4.099996089935303`, lift `0.15918205678462982`, accel
  `1.0745576969847066`.
- `three_quarter_low_misleading`: closed-loop teacher hold
  `3.8833296298980713`, lift `0.15595264732837677`, accel
  `2.132696179270316`; no-adaptation hold `4.099996089935303`, lift
  `0.16022272408008575`, accel `0.9597340451215278`.

Post-hoc diagnostics in
`experiments/outputs/phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628_diagnostics.json`
show the learned checkpoint still activates on about `72%` of held-out frames,
nearly the same as the no-curiosity residual. Therefore the failure mode is
over-dense corrective labels/over-intervention, not missing video evidence or
missing training time. The next faithful repair must avoid another dense
teacher imitation pass: use an advantage-gated intervention target that only
labels residual action where paired train/validation evidence shows improvement
over no-adaptation, or expand to genuinely harder object/task families where
no-adaptation fails and residual intervention has measurable room to improve.

Phase08 harder-candidate probe on 2026-06-28: to test whether simply moving to
ultra-low friction creates a useful improvement gap, ran full-video diagnostic
`phase08_harder_candidate_probe_v1_20260628` on
`full_ultralow_hidden` and `three_quarter_ultralow_misleading`, comparing
no-adaptation and scripted feedback. Manual inspection JSON:
`experiments/outputs/phase08_harder_candidate_probe_v1_20260628_manual_visual_inspection.json`.
All four contact sheets were nonblank. The probe is negative for source
selection: no-adaptation still holds/lifts better than scripted feedback.
`full_ultralow_hidden` no-adaptation hold/lift is `4.1`/`0.1592235416173935`
versus scripted feedback `2.966666666666667`/`0.15194369852542877`;
`three_quarter_ultralow_misleading` no-adaptation hold/lift is
`4.1`/`0.1602979302406311` versus scripted feedback
`2.9833333333333334`/`0.1528860181570053`. This rules out "lower friction
alone" as the next training distribution. The next real repair should either
change object geometry/contact patch/deformability/off-center torque demands
or build advantage-gated labels from paired evidence rather than dense
feedback triggers.

Phase08 advantage-gated/object-family repair setup on 2026-06-28: added
`experiments/configs/build_advantage_gated_residual_preflight_v1.py` and
`experiments/configs/phase08_advantage_gated_residual_preflight_v1.json`. This
preflight is intentionally strict: it compares paired no-adaptation and
intervention rollouts, admits labels only from cells where the intervention
beats the paired baseline with nonnegative hold/lift gain and no acceleration
regression, and fails rather than writing training data if no accepted
train/validation evidence exists. Added diagnostic script
`experiments/configs/run_phase08_object_family_probe_in_alloc.sh` to probe the
official Newton `pen` scene with no-adaptation versus scripted feedback. Static
checks passed for the advantage-gated Python preflight, its JSON config, and
the object-family shell script. Slurm job `156688` has been requested in tmux
session `curiosity_phase08_object_probe_alloc_20260628` for one GPU/one day and
is pending due to priority. This is source-selection and data-gate repair work,
not a training result or success claim.

Phase08 official pen/object-family probe result on 2026-06-28:
`phase08_object_family_probe_v1_20260628` completed four full-video diagnostic
rollouts on the official Newton `pen` scene with fresh official sensor/contact
sanity checks. Manual visual inspection JSON:
`experiments/outputs/phase08_object_family_probe_v1_20260628_manual_visual_inspection.json`.
All four contact sheets are nonblank multi-camera robot/object rollouts, but
the source-selection result is negative and not training. In both paired cells,
scripted feedback triggered `0` times and had lower hold duration than
no-adaptation: `pen_nominal_medium` no-adaptation hold/lift
`4.25`/`0.21550706028938293` versus scripted feedback
`3.8666666666666667`/`0.21685871481895447`; `pen_low_misleading`
no-adaptation hold/lift `4.25`/`0.21560673415660858` versus scripted feedback
`3.8666666666666667`/`0.21685360372066498`. These pen cells must not be
accepted into the advantage-gated residual dataset. The next source-selection
step needs a task change that creates real no-adaptation failure or a better
intervention source, such as off-center torque, altered grasp pose/contact
patch, handle/asymmetric COM, or compliant/deformable objects with explicit
provenance.

Phase08 contact-patch/off-center repair setup on 2026-06-28: added an official
Newton Panda hydro grasp-target perturbation adapter to
`experiments/configs/newton_panda_hydro_tiled_camera_export.py`. The adapter
exposes `--grasp-offset-delta-xyz`, applies it before IK waypoint capture,
does not change the official object geometry or body state, and records
provenance in `grasp_perturbation_adapter` plus
`candidate.task.grasp_offset_delta_xyz`. The allocation runner now passes
`GRASP_OFFSET_DELTA_XYZ`. Added diagnostic runner
`experiments/configs/run_phase08_contact_patch_probe_in_alloc.sh` for paired
no-adaptation versus guarded scripted feedback on `cube_edge_x`,
`cube_corner_xy`, and `pen_end_bias`. Static checks passed for Python syntax
and shell syntax. This is source-selection for later advantage-gated training,
not training and not a success claim.

Phase08 contact-patch source-selection result on 2026-06-28: the retry2
contact-patch probe found a real paired improvement region on the official
Newton `pen` scene. `pen_end_bias` no-adaptation visibly failed to maintain
the object and had hold/lift about `0.0`/`0.1073`, while guarded feedback held
the object for about `3.05s` with lift about `0.1736` and much lower
acceleration/slip. Follow-up pair collection accepted
`pen_end_bias_train_d`; validation refinement accepted `pen_end_bias_val_c`
and `pen_end_bias_val_e` while rejecting the failing `val_d` intervention.
Direct contact-sheet inspection passed for accepted paired evidence. This is
valid source-selection evidence for advantage-gated residual repair, not a
curiosity success claim.

Phase08 strict advantage-gated preflight on 2026-06-28: the preflight passed
at
`data/processed/phase08_advantage_gated_residual_preflight_v1_20260628/manifest.json`
with `train_record_count=900`, `validation_record_count=900`,
`accepted_cell_count=4`, `accepted_active_frames=1313`, and `failures=[]`.
The strict gate requires the intervention rollout itself to succeed, not just
outscore a failed baseline. Rejected cells remain rejected and must not be
renamed into training positives.

Phase08 advantage-gated residual training and evaluation preparation on
2026-06-28: started real one-hour training
`phase08_advantage_gated_residual_adapter_v1_train_20260628` in tmux-held
Slurm job `156696` on `server02` after fresh official Newton sensor/contact
sanity passed. The training process is using GPU with high utilization; it is
not complete until the one-hour trainer summary and checkpoint exist and pass
the real-training gates. To prepare the next closed-loop held-out evaluation,
the learned-residual exporter now feeds
`candidate.task.grasp_offset_delta_x/y/z` into checkpoint feature lookup, and
`experiments/configs/run_phase08_advantage_gated_heldout_eval_in_alloc.sh` was
added. That runner evaluates held-out `pen_end_bias_heldout_center`,
`pen_end_bias_heldout_high_y`, and `pen_end_bias_heldout_low_x` against
`no_adaptation`, `guarded_feedback`, and the trained
`advantage_gated_residual` checkpoint with full 450-frame videos, metrics,
acceleration analysis, NPZ field validation, and strongest-baseline comparison.
This evaluation is still not final curiosity training and cannot close the
project without later curiosity-specific training, ablations, and
mainstream/official comparison gates.

Phase08 advantage-gated residual result on 2026-06-28: training passed the real
one-hour gate with `elapsed_seconds=3600.1261126995087`,
`optimizer_steps=25629`, checkpoint
`checkpoints/phase08_advantage_gated_residual_adapter_trainer_v1_20260628/phase08_advantage_gated_residual_adapter_v1_train_20260628.pt`,
validation `active_accuracy=0.9933333396911621`, and mean GPU utilization
`97.14876033057851%`. Held-out retry1 evaluation then completed all 9/9
full-video rollouts under
`phase08_advantage_gated_heldout_eval_retry1_v1_20260628`, with manual visual
inspection pass at
`experiments/outputs/phase08_advantage_gated_heldout_eval_retry1_v1_20260628_manual_visual_inspection.json`.
The performance result is negative: status is `open_not_satisfied`; the
trained residual did not beat the strongest baseline on any of the three
held-out cells. The strongest baseline was `guarded_feedback` for
`pen_end_bias_heldout_center`, `pen_end_bias_heldout_high_y`, and
`pen_end_bias_heldout_low_x`. Representative trained residual metrics were
center hold/lift/slip/accel `0.0`/`0.08747`/`1.98288`/`83.509`,
high_y `0.2167`/`0.14221`/`0.09050`/`54.323`, and low_x
`0.0`/`0.04967`/`2.07521`/`39.895`. This is valid negative evidence for the
advantage-gated residual repair, not a final curiosity result. The next
faithful step must not repeat the same residual imitation target; it needs a
curiosity-specific closed-loop objective or stronger intervention/source
distribution that can improve held-out hold/lift without safety regression.

Phase08 curiosity-specific continuation on 2026-06-28: after the negative
advantage-gated residual held-out result, the next step moved into an actual
curiosity chain rather than stopping. Added source-compat glue
`experiments/configs/build_phase08_advantage_source_compat_v1.py` and
`experiments/configs/phase08_advantage_source_compat_v1.json` to convert the
strict advantage-gated split CSVs into a source-runner-compatible manifest
without changing held-out splits. Source compat passed with `1800` rows and
four accepted source runs. Added Phase08 curiosity configs
`experiments/configs/phase08_curiosity_forward_model_preflight_v1.json`,
`experiments/configs/phase08_curiosity_forward_model_trainer_v1.json`,
`experiments/configs/phase08_curiosity_learning_progress_v1.json`, and
`experiments/configs/phase08_curiosity_weighted_residual_adapter_trainer_v1.json`.
The forward-model preflight passed with `1796` transition records, `898`
train and `898` validation transitions, and nonzero/nonconstant physical
targets. Real one-hour training
`phase08_curiosity_forward_model_v1_train_20260628` has started in the held
Slurm allocation after fresh official Newton sanity and is using the GPU. This
is curiosity-model training progress, not a policy success claim; it still
requires learning-progress scoring, curiosity-weighted policy update, held-out
full-video evaluation, ablations, and mainstream/official comparison gates.

Phase08 curiosity forward-model and policy-update progress on 2026-06-28:
`phase08_curiosity_forward_model_v1_train_20260628` completed a real one-hour
training run with `elapsed_seconds=3600.0301122665405`,
`optimizer_steps=24798`, checkpoint
`checkpoints/phase08_curiosity_forward_model_v1_20260628/phase08_curiosity_forward_model_v1_train_20260628.pt`,
initial snapshot
`checkpoints/phase08_curiosity_forward_model_v1_20260628/phase08_curiosity_forward_model_v1_train_20260628_initial_snapshot.pt`,
and mean GPU utilization `97.34166666666667%`. Learning-progress scoring
then passed with `1796` scores, mean learning progress
`0.5845009502902926`, train mean bounded curiosity reward
`0.6406456770669913`, and validation mean bounded curiosity reward
`0.528628062855096`. Real curiosity-weighted residual policy training
`phase08_curiosity_weighted_residual_adapter_v1_train_20260628` has started in
the same held allocation after fresh official Newton sanity and high GPU
utilization. This is the policy-update stage, but still not a success claim
until the trained checkpoint passes held-out full-video comparison against
no-adaptation, guarded feedback, the advantage-gated residual, and required
serious-method/ablation gates.

Phase08 curiosity-weighted held-out result on 2026-06-29: the policy checkpoint
`checkpoints/phase08_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase08_curiosity_weighted_residual_adapter_v1_train_20260628.pt`
was evaluated through tmux-held Slurm allocation `156696` using GPU `srun`
after the first non-srun attempt failed the official Newton CUDA sanity. The
valid rerun passed fresh official Newton sanity, completed all three
`pen_end_bias` held-out cells, wrote full rollout GIFs and contact sheets, and
exited with `TMUX_PHASE08_CURIOSITY_WEIGHTED_EVAL_EXIT=0`. Summary:
`experiments/outputs/phase08_curiosity_weighted_heldout_eval_v1_20260628_summary.json`.
Manual visual inspection:
`experiments/outputs/phase08_curiosity_weighted_heldout_eval_v1_20260628_manual_visual_inspection.json`.
The result is negative: status is `open_not_satisfied`, and
`curiosity_weighted_beats_strongest_baseline_all_cells_without_safety_regression=false`.
The strongest baseline remains `guarded_feedback` on all three held-out cells.
Curiosity-weighted residual only improves over the advantage-gated residual on
the center cell, but still loses to the strongest baseline and has safety
regressions versus no-adaptation/guarded feedback. High-y and low-x both fail
the strongest-baseline gate. This is real closed-loop held-out evaluation
evidence with complete videos, but it is not a completed curiosity result.
The next faithful step must repair the policy objective/source distribution:
preserve no-adaptation/guarded hold behavior while allowing curiosity-driven
contact/slip corrections, rather than repeating the same supervised residual
reweighting target.

Phase08 guarded-anchor repair result on 2026-06-29: after the negative
curiosity-weighted held-out result, added
`experiments/configs/phase08_guarded_anchor_curiosity_weighted_residual_adapter_trainer_v1.json`
and trained `phase08_guarded_anchor_curiosity_repair_v1_train_20260629` for a
real one GPU-hour in Slurm allocation `156696`. Training passed with
`elapsed_seconds=3600.1154675483704`, `optimizer_steps=25422`, checkpoint
`checkpoints/phase08_guarded_anchor_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_guarded_anchor_curiosity_repair_v1_train_20260629.pt`,
and mean GPU utilization `97.26666666666667%`. The first held-out eval attempt
failed only because the exporter checkpoint-classification allowlist lacked
the new guarded-anchor classification; this was patched in
`experiments/configs/newton_panda_hydro_tiled_camera_export.py`. Retry1 then
completed all three full-video held-out cells and wrote
`experiments/outputs/phase08_guarded_anchor_heldout_eval_retry1_v1_20260629_summary.json`
plus manual visual inspection
`experiments/outputs/phase08_guarded_anchor_heldout_eval_retry1_v1_20260629_manual_visual_inspection.json`.
The result is still negative: status `open_not_satisfied`, strongest baseline
`guarded_feedback`, and guarded-anchor does not beat the strongest baseline on
all cells. It reduced acceleration and some slip, especially on high_y/low_x,
but sacrificed hold/lift, with high_y lift dropping to about `0.0662` versus
guarded `0.1626`. This means the strong global neutral anchor is too blunt.
The next repair is a selective anchor that preserves lift velocity capacity and
only weakly constrains hold-height/stabilization residuals in high-contact
stable phases.

Phase08 selective-anchor repair result on 2026-06-29: added
`experiments/configs/phase08_selective_anchor_curiosity_weighted_residual_adapter_trainer_v1.json`
and trained `phase08_selective_anchor_curiosity_repair_v1_train_20260629` from
the previous curiosity-weighted checkpoint, not from the over-anchored guarded
checkpoint. Real training passed with `elapsed_seconds=3600.115711927414`,
`optimizer_steps=25592`, checkpoint
`checkpoints/phase08_selective_anchor_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_selective_anchor_curiosity_repair_v1_train_20260629.pt`,
validation `active_accuracy=0.9977778196334839`,
`continuous_mse=0.00719881895929575`, and mean GPU utilization
`98.13333333333334%`. Held-out eval
`experiments/outputs/phase08_selective_anchor_heldout_eval_v1_20260629_summary.json`
completed all three full-video cells and manual visual inspection at
`experiments/outputs/phase08_selective_anchor_heldout_eval_v1_20260629_manual_visual_inspection.json`.
The result remains `open_not_satisfied`: it does not beat strongest
`guarded_feedback` on all cells without safety regression. Selective anchor
recovered high_y hold/lift relative to guarded-anchor
(`0.1833s`/`0.1383m`), but slip was far worse than guarded feedback
(`1.5792m` versus `0.0709m`). Low_x reduced slip to `0.8224m` but introduced an
acceleration regression (`46.208` versus guarded `36.871`). Center remained
below guarded feedback on hold/lift and above no-adaptation on slip. This
shows anchor tuning alone is insufficient. The next faithful repair must
change the data/control structure: use guarded feedback as the stable hold
prior and train curiosity only as a local slip/contact correction overlay, or
collect paired source rollouts where that overlay beats the guarded baseline
before another one-hour policy update.

Phase08 guarded-overlay source probe on 2026-06-29: implemented direct
controller mode `lift_hold_feedback_residual_overlay` and direct allocation
runner paths so the learned residual can only act as a local overlay on top of
the guarded feedback prior. This keeps the stable scripted waypoint/contact
transition as the base controller instead of replacing it with another global
learned residual. The direct probe
`phase08_guarded_overlay_probe_direct_v1_20260629` ran in tmux-held Slurm
allocation `156696` and wrote
`experiments/outputs/phase08_guarded_overlay_probe_direct_v1_20260629_summary.json`
plus report
`experiments/reports/2026-06-29_phase08_guarded_overlay_probe_direct_v1.md`.
It is source selection only, not training and not a success claim. It found
one accepted train source cell, `pen_end_bias_train_c`, where guarded overlay
improved hold by `2.3667s`, lift by `0.00444m`, and did not regress slip or
acceleration versus guarded feedback. It rejected `pen_end_bias_train_d`,
`pen_end_bias_val_c`, and `pen_end_bias_val_e`; both validation overlay cells
lost hold/lift and had acceleration regressions. Therefore this evidence is
not sufficient for the next real training run: a single accepted train cell
with zero accepted validation cells would be another narrow overfit source.

Phase08 guarded-overlay expanded source probe started on 2026-06-29: added
`experiments/configs/run_phase08_guarded_overlay_expanded_probe_direct_in_alloc.sh`
and launched it in the same tmux-held Slurm allocation `156696`, tmux window
`phase08_overlay_expanded_direct`, log
`logs/newton/phase08_guarded_overlay_expanded_probe_direct_v1_20260629.srun.log`.
The probe stays train/validation only, uses no held-out cells, and compares
guarded feedback against guarded overlay over eight near-neighbor
`pen_end_bias` offsets. It must produce multiple accepted train cells and at
least one accepted validation cell before any follow-up training preflight is
allowed. If it fails that gate, the next faithful step is source/control
repair, not one-cell training.

Phase08 guarded-overlay expanded source probe result on 2026-06-29: the
expanded probe completed with exit `0` and wrote
`experiments/outputs/phase08_guarded_overlay_expanded_probe_direct_v1_20260629_summary.json`
and
`experiments/reports/2026-06-29_phase08_guarded_overlay_expanded_probe_direct_v1.md`.
It found one accepted validation source, `pen_end_bias_overlay_val_c0`, where
guarded overlay improved hold by `3.0s`, lift by `0.08422m`, and did not
regress slip or acceleration. It found zero accepted train sources under the
current strict hold/lift/slip/acceleration non-regression gate. Several train
cells (`train_c0`, `train_c2`) repaired hold, slip, and acceleration but lost
a small amount of peak lift versus a failed guarded-feedback baseline, so they
remain rejected under the current gate. This confirms the overlay mechanism can
repair a real validation failure, but the source set is still too narrow for
training. The next faithful step is a train-focused probe around the successful
validation offset and the previous direct accepted `pen_end_bias_train_c`
region; do not train until there is enough accepted train coverage.

Phase08 guarded-overlay train-focused source probe started on 2026-06-29:
added
`experiments/configs/run_phase08_guarded_overlay_train_focus_probe_direct_in_alloc.sh`
and launched it in tmux-held Slurm allocation `156696`, tmux window
`phase08_overlay_train_focus`, log
`logs/newton/phase08_guarded_overlay_train_focus_probe_direct_v1_20260629.srun.log`.
It probes five train-only near-neighbor offsets around the accepted validation
offset. It is not training, not held-out evaluation, and not a success claim.
Its only purpose is to decide whether there is enough accepted train source
coverage to build the next overlay-training preflight.

Phase08 guarded-overlay train-focused source probe result on 2026-06-29: the
train-focused probe completed and wrote
`experiments/outputs/phase08_guarded_overlay_train_focus_probe_direct_v1_20260629_summary.json`
and
`experiments/reports/2026-06-29_phase08_guarded_overlay_train_focus_probe_direct_v1.md`.
Status is `open_no_train_overlay_source_candidates`: none of the five
train-only offsets passed the old strict hold/lift/slip/accel non-regression
gate. This confirmed that another narrow direct training run would be invalid.
However, analysis of the direct and expanded probes showed a gate pathology:
when the guarded-feedback baseline itself fails by throwing the object upward
and then dropping/slipping, `max_lift` can be misleadingly high. Rejecting an
overlay that stabilizes the object for a tiny peak-lift loss is too brittle.

Phase08 guarded-overlay failure-repair preflight on 2026-06-29: added a new
source gate rather than weakening the old one:
`experiments/configs/build_phase08_guarded_overlay_failure_repair_preflight_v1.py`,
`experiments/configs/phase08_guarded_overlay_failure_repair_preflight_v1.json`,
and
`experiments/configs/run_phase08_guarded_overlay_failure_repair_preflight_in_alloc.sh`.
The gate keeps the old strict non-regression rule when the baseline succeeds.
When the baseline fails, it accepts an overlay only if the overlay itself is a
successful repair with absolute hold/lift/drop/slip/accel/contact-loss safety
thresholds and no safety regression. It forbids held-out source use and writes
`not_training=true`, `not_success_claim=true`,
`generated_trex_fields=[]`, and `schema_promotion=blocked`.
The allocation run in Slurm job `156696` passed with exit `0` and wrote
`data/processed/phase08_guarded_overlay_failure_repair_preflight_v1_20260629/manifest.json`.
Accepted cells: 4 total, 3 train and 1 validation. Record counts: 1350 train,
450 validation, with 1318 accepted active frames. This is preflight evidence
only; it is not policy training and not held-out success.

Phase08 guarded-overlay curiosity preflight chain on 2026-06-29: added
`experiments/configs/phase08_guarded_overlay_failure_repair_source_compat_v1.json`
and
`experiments/configs/phase08_guarded_overlay_curiosity_forward_model_preflight_v1.json`,
then ran the existing allocation-only preflight chain in Slurm job `156696`.
It wrote
`data/processed/phase08_guarded_overlay_failure_repair_source_compat_v1_20260629/manifest.json`
and
`data/processed/phase08_guarded_overlay_curiosity_forward_model_preflight_v1_20260629/manifest.json`.
Forward preflight status is `pass` with 1796 transition records: 1347 train
records from `pen_end_bias_train_c`, `pen_end_bias_overlay_train_c0`, and
`pen_end_bias_overlay_train_c2`, and 449 validation records from
`pen_end_bias_overlay_val_c0`. Held-out cells remain reserved and are not used
for source, target construction, threshold selection, or preflight tuning.

Phase08 guarded-overlay training chain started on 2026-06-29: added
`experiments/configs/phase08_guarded_overlay_curiosity_forward_model_trainer_v1.json`,
`experiments/configs/phase08_guarded_overlay_curiosity_learning_progress_v1.json`,
and
`experiments/configs/phase08_guarded_overlay_curiosity_weighted_residual_adapter_trainer_v1.json`.
Started a sequential allocation run in Slurm job `156696`, tmux window
`phase08_overlay_train_chain`, log
`logs/newton/phase08_guarded_overlay_training_chain_v1_20260629.srun.log`.
The chain is: real one-hour guarded-overlay curiosity forward-model training,
learning-progress scoring, then real one-hour curiosity-weighted residual
policy training from the selective-anchor checkpoint. This is in progress and
must not be described as success until the training summaries, checkpoint
paths, held-out full-video evaluation, strict metrics, strongest-baseline
comparison, and mainstream-method comparison/blocker evidence all pass.

Phase08 guarded-overlay training chain result on 2026-06-29: the sequential
training chain completed with exit `0`. Forward-model training
`phase08_guarded_overlay_curiosity_forward_model_v1_train_20260629` passed as
a real one-hour training result with `elapsed_seconds=3600.0973856449127`,
`optimizer_steps=17503`, checkpoint
`checkpoints/phase08_guarded_overlay_curiosity_forward_model_v1_20260629/phase08_guarded_overlay_curiosity_forward_model_v1_train_20260629.pt`,
validation loss `0.16814851760864258`, and mean GPU utilization
`98.25833333333334%`. Learning-progress scoring then passed with 1796 scores,
mean bounded curiosity reward `0.7238376709891858`, train mean
`0.7662746050818906`, validation mean `0.5965268687110703`, and
`not_raw_prediction_error_only=true`. Policy training
`phase08_guarded_overlay_curiosity_weighted_residual_adapter_v1_train_20260629`
also passed as a real one-hour training result with
`elapsed_seconds=3600.164947986603`, `optimizer_steps=17665`, checkpoint
`checkpoints/phase08_guarded_overlay_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_guarded_overlay_curiosity_weighted_residual_adapter_v1_train_20260629.pt`,
and mean GPU utilization `98.24166666666666%`. The policy validation metrics
were weak (`active_accuracy=0.5822222232818604`,
`continuous_mse=0.5252048373222351`, `loss=1.5460398197174072`), so the
training artifact is valid but risky and not success evidence by itself.

Phase08 guarded-overlay held-out retry1 evaluation on 2026-06-29: patched
`experiments/configs/run_phase08_curiosity_weighted_heldout_eval_in_alloc.sh`
to allow `CURIOSITY_CONTROLLER_MODE` and then to use the direct controller
runner for residual-overlay modes, because the first eval attempt failed when
the v2 wrapper did not pass `--residual-adapter-checkpoint` to
`lift_hold_feedback_residual_overlay`. Retry1 ran in Slurm job `156696` with
full 450-frame video export for all three held-out cells and wrote
`experiments/outputs/phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1_20260629_summary.json`,
report
`experiments/reports/2026-06-29_phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1.md`,
and manual visual inspection
`experiments/outputs/phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1_20260629_manual_visual_inspection.json`.
All candidate contact sheets/videos are nonblank with robot/object visible.
The result remains `open_not_satisfied`: guarded-overlay curiosity beats the
strongest guarded-feedback baseline on `pen_end_bias_heldout_center` and
`pen_end_bias_heldout_high_y` without listed safety regression, and beats the
advantage-gated residual on all cells, but fails `pen_end_bias_heldout_low_x`
with `hold_duration_s=0.0`, `lift_height_m=0.05279061198234558`, and
`max_slip_m=1.6643761054070036`. This is meaningful progress over previous
Phase08 runs, but it is still negative/incomplete for the final all-cell
harder-task gate. The next faithful step is targeted low-x source/control
repair or source expansion, not completion language.

Phase08 guarded-overlay repair-coverage probe and v2 source gate on
2026-06-29: after the low-x held-out failure, ran a train-only repair-coverage
probe using the latest guarded-overlay checkpoint, not the old selective-anchor
checkpoint. The probe used no held-out cells and wrote full 450-frame videos
for five paired train offsets under
`phase08_guarded_overlay_repair_coverage_probe_direct_v1_20260629`. It found
mixed evidence: `pen_end_bias_overlay_train_focus_a` and
`pen_end_bias_overlay_train_focus_d` were useful failed-baseline repairs under
the absolute repair gate, while `train_focus_b`, `train_focus_c`, and
`train_focus_e` were rejected because the overlay either failed, damaged a
successful baseline, or introduced safety regression. Added v2 configs
`experiments/configs/phase08_guarded_overlay_failure_repair_preflight_v2.json`,
`phase08_guarded_overlay_failure_repair_source_compat_v2.json`, and
`phase08_guarded_overlay_curiosity_forward_model_preflight_v2.json`.
The allocation-only v2 failure-repair preflight passed with 6 accepted cells
(5 train, 1 validation), 2250 train records, 450 validation records, and 1973
accepted active frames at
`data/processed/phase08_guarded_overlay_failure_repair_preflight_v2_20260629/manifest.json`.
The v2 source-compat and forward preflight also passed, producing 2694
transition records at
`data/processed/phase08_guarded_overlay_curiosity_forward_model_preflight_v2_20260629/manifest.json`.
Held-out cells remain reserved and were not used for source, target
construction, threshold selection, or preflight tuning.

Phase08 guarded-overlay repair-coverage training chain started on 2026-06-29:
added
`experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_trainer_v2.json`,
`experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_learning_progress_v2.json`,
and
`experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_trainer_v2.json`.
Started the sequential allocation run in Slurm job `156696`, tmux window
`phase08_overlay_train_v2`, log
`logs/newton/phase08_guarded_overlay_repair_coverage_training_chain_v2_20260629.srun.log`.
The policy v2 training continues from the current guarded-overlay v1
checkpoint and uses a slightly stronger active preservation anchor to reduce
over-activation observed in rejected train cells. This is in progress and is
not a success claim until the forward model, learning-progress score, policy
checkpoint, held-out full-video evaluation, strongest-baseline comparison, and
serious-method comparison/blocker evidence all pass.

Phase08 guarded-overlay repair-coverage training progress on 2026-06-29:
the v2 forward model completed as a real one-hour training result in Slurm job
`156696`. Summary:
`experiments/outputs/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_train_20260629_summary.json`.
Checkpoint:
`checkpoints/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_train_20260629.pt`.
It passed with `elapsed_seconds=3600.151699781418`, `optimizer_steps=10537`,
and validation loss `0.21219861507415771`. The v2 learning-progress scorer
then passed with 2694 scores, mean bounded curiosity reward
`0.5845975945337932`, train mean `0.6020250130443103`, validation mean
`0.49746050198120656`, and `not_raw_prediction_error_only=true`; output:
`experiments/outputs/phase08_guarded_overlay_repair_coverage_curiosity_learning_progress_v2_20260629/curiosity_learning_progress_summary.json`.
The v2 policy training then started in the same allocation as
`phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_v2_train_20260629`.
An automatic held-out eval waiter was added and launched via
`experiments/configs/launch_phase08_guarded_overlay_repair_coverage_heldout_eval_v2_wait_tmux.sh`
in tmux window `phase08_overlay_eval_v2_wait`; it waits for the policy summary
to pass before running
`experiments/configs/run_phase08_curiosity_weighted_heldout_eval_in_alloc.sh`
with full-video held-out evaluation. This remains in progress and is not a
completion or improvement claim.
