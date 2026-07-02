# Phase07 Evidence Refresh V1

This report records a compute-side audit refresh. It is not training, not
rendering, not inference, and not a curiosity success claim.

## Command And Allocation

- Slurm job: `155732`
- Partition/node: `cpu` / `server13`
- tmux session: `curiosity_phase07_evidence_refresh_alloc_20260628`
- launcher: `JOB_ID=155732 TMUX_SESSION=curiosity_phase07_evidence_refresh_alloc_20260628 bash experiments/configs/launch_phase07_evidence_refresh_tmux.sh`
- runner: `experiments/configs/run_phase07_evidence_refresh_in_alloc.sh`
- log: `logs/newton/phase07_evidence_refresh_v1_20260628.log`
- exit marker: `TMUX_PHASE07_EVIDENCE_REFRESH_EXIT=0`
- allocation cleanup: `scancel 155732`; `squeue` showed `COMPLETI`

## Refreshed Evidence

- action bridge backfill refreshed;
- mainstream adapter conversion preflight refreshed;
- stage-1 mainstream dataset indices refreshed;
- stage-1 no-held-out-leakage audit refreshed and passed;
- official-method readiness refreshed;
- held-out comparison report refreshed;
- hard training evidence gate refreshed.

Summary file:
`experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json`.

Key refreshed status:

- `hard_gate_status=open_not_satisfied`
- `final_curiosity_success_allowed=false`
- `heldout_missing_or_failed_entry_count=0`
- `heldout_curiosity_beats_all_strongest_baselines_without_safety_regression=false`
- `official_method_readiness_status=open_not_ready`
- `official_method_comparison_ready=false`

The stale no-learning-progress manual-visual gap is now cleared, and the hard
gate reports `training_evidence_status=pass` plus
`evaluation_evidence_status=pass`. The remaining result is still negative or
incomplete evidence because curiosity does not beat the strongest declared
baseline without safety regression and the serious/mainstream comparison gate
remains open.

## Refreshed Failure Details

Held-out comparison:

- `empty_high_misleading`: curiosity does not beat `no_adaptation`;
  safety regression list is empty, but hold/lift score is lower.
- `full_low_hidden`: curiosity does not beat `no_adaptation`; safety regression
  includes `max_object_accel_m_s2_regression`.
- `three_quarter_low_misleading`: curiosity does not beat `no_adaptation`;
  safety regression includes `max_object_accel_m_s2_regression`.

Official-method readiness:

- `openpi_pi0`: missing prepared env under `envs/` and missing official
  checkpoint or filled checkpoint blocker.
- `gr00t`: missing prepared env under `envs/` and missing official checkpoint
  or filled checkpoint blocker.
- `diffusion_policy`: missing prepared env under `envs/` and missing official
  checkpoint or filled checkpoint blocker.
- `rtx`: missing prepared env under `envs/` and missing official checkpoint or
  filled checkpoint blocker.

## Next Required Work

Continue with faithful objective repair or stronger closed-loop curiosity
training on the harder Phase07 task, and continue official-method environment
and checkpoint/blocker preparation. Do not claim curiosity success from this
refresh.
