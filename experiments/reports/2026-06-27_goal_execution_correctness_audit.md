# Goal Execution Correctness Audit

## Scope

This audit checks the active execution goal:

1. after downloading checkpoints, run sanity checks;
2. run visualizations on compute nodes and manually inspect browser/frame
   outputs before downstream/innovation claims;
3. periodically reread `AGENTS.md`;
4. prepare separate local shared-filesystem venvs under `envs/` and only
   activate them on compute nodes.

This is not a new experiment and not a new scientific claim. It records whether
the current workspace evidence satisfies the execution-correctness goal.

## Current Workspace State

- Git status at audit start: clean.
- Active non-legacy TODO items: none remaining.
- Current reusable Curiosity allocation:
  `154142|curiosity_residual_source_1gpu_1day|RUNNING|server56`.
- Current allocation step state: only allocation/extern steps remain; no
  unfinished experiment step was active during the final audit.

## Requirement 1: Checkpoints Have Sanity Checks

Status: pass.

Evidence:

- T-Rex reference checkpoint sanity:
  `experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
  Integrity and official midtrain model-load sanity both passed. This remains
  reference-only and does not promote Newton data into T-Rex schema.
- Residual adapter real training checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`.
  Training report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_v1.md`.
  The run passed fresh official Newton sanity, trained for
  `3600.0302035808563` seconds, wrote the checkpoint, and had
  `generated_trex_fields=[]` / `schema_promotion=blocked`.
- Residual adapter evaluation reports:
  `experiments/reports/2026-06-27_phase04_residual_adapter_eval_empty_medium_validation_v1.md`,
  `experiments/reports/2026-06-27_phase04_residual_adapter_heldout_eval_v1.md`,
  and
  `experiments/reports/2026-06-27_phase04_residual_adapter_extra_ordinary_eval_v1.md`.
  Each evaluation recorded fresh official Newton sanity before camera export
  and metrics.

Conclusion: all active checkpoint-dependent routes have recorded sanity checks
before downstream use.

## Requirement 2: Compute Visualizations And Manual Browser/Frame Inspection

Status: pass.

Evidence:

- Visualizations were generated inside tmux-held compute allocations on
  `server56`, with reports recording tmux allocation/session usage.
- Automated visual validation passed for trained residual adapter evaluations:
  each of `empty_medium`, `full_low`, `empty_high`, `half_high`, and
  `full_medium` has 9 nonblank sampled frames with `576x200` dimensions and a
  frame browser.
- Manual visual inspection JSONs:
  - `experiments/outputs/residual_adapter_eval_v1_empty_medium_validation_20260627_0605_manual_visual_inspection.json`
  - `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_manual_visual_inspection.json`
  - `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_manual_visual_inspection.json`
  - `experiments/outputs/residual_adapter_eval_v1_half_high_ordinary_20260627_0631_manual_visual_inspection.json`
  - `experiments/outputs/residual_adapter_eval_v1_full_medium_ordinary_20260627_0638_manual_visual_inspection.json`
- Manual inspection status for all listed trained residual adapter evaluations:
  `pass_nonblank_success_learned_residual`.
- Reports that use these outputs keep downstream claims narrow:
  `experiments/reports/2026-06-27_phase04_residual_adapter_heldout_eval_v1.md`,
  `experiments/reports/2026-06-27_phase04_residual_adapter_extra_ordinary_eval_v1.md`,
  and
  `experiments/reports/2026-06-27_phase04_residual_adapter_failure_mode_comparison_v1.md`.

Conclusion: visual outputs were generated on compute resources, browser/frame
artifacts were manually inspected, and downstream claims were made only after
visual gates passed.

## Requirement 3: Periodic `AGENTS.md` Rereads

Status: pass.

Evidence:

- Current audit reread `AGENTS.md` before checking state.
- Compute logs contain repeated `AGENTS_REREAD_HEAD` /
  `AGENTS_REREAD_END` blocks before experiment runs and metric extraction.
- Examples include residual adapter training, evaluation, held-out metrics,
  ordinary extra evaluation metrics, failure-mode comparison, and earlier
  baseline logs under `logs/newton/`.

Conclusion: the workspace records repeated rule rereads during the run sequence
and the final audit also reread `AGENTS.md`.

## Requirement 4: Separate Local Venvs And Compute-Only Activation

Status: pass.

Evidence:

- Local shared-filesystem env folders exist under `envs/`:
  `envs/newton`, `envs/taccel`, `envs/trex`, `envs/residual_adapter`, and
  `envs/trex_dataset`.
- Residual adapter environment note:
  `docs/residual_adapter_environment_v1.md`.
  It records `envs/residual_adapter/.venv`, local installation with the
  Tsinghua mirror, and the runtime split:
  `NEWTON_VENV=envs/newton/.venv` for official Newton sanity and
  `TRAINER_VENV=envs/residual_adapter/.venv` for PyTorch trainer execution.
- Compute runners activate existing venvs and do not perform dependency
  installation on compute nodes.

Conclusion: environments are separated by experiment family and prepared under
the shared workspace `envs/` hierarchy before compute use.

## Additional Correctness Checks

Status: pass.

Evidence:

- Real residual adapter training obeyed the one-GPU one-hour rule:
  elapsed `3600.0302035808563` seconds.
- GPU utilization monitoring passed:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_gpu_utilization.json`
  reports mean utilization `99.08333333333333%`.
- No toy T-Rex/VQ-VAE/world-model substitution was used. Newton-native
  residual adapter work remains explicitly labeled as not official T-Rex and
  not T-Rex schema.
- T-Rex bridge reassessment concluded no-go for strict promotion rather than
  padding or fabricating missing source fields:
  `experiments/reports/2026-06-27_phase06_trex_bridge_source_reassessment_v1.md`.

## Completion Decision

The execution-correctness goal is satisfied for the current staged workspace:

- checkpoint-dependent routes have sanity checks;
- compute visualizations have automated and manual browser/frame gates;
- `AGENTS.md` rereads are recorded;
- separate local venvs are maintained and compute nodes only activate them;
- active non-legacy TODO items are complete;
- the worktree is clean after commits.

Future scientific work remains possible, especially broader object-family
generalization and true tactile/F6 source acquisition, but those are new
research directions beyond this execution-correctness goal.
