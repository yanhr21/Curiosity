# Phase 05 Newton Contact Source Manifest V1

## Scope

This run builds the first Phase 05 tactile/contact source manifest from real
Newton lift-hold scripted-feedback rollouts. It is namespace-preserving source
evidence only.

This run does not create a T-Rex dataset, does not create calibrated F6 tactile
force fields, does not create dense tactile deformation fields, and does not
train a policy or world model.

## Files

- Config: `experiments/configs/newton_lift_hold_contact_source_manifest_v1.json`
- Builder: `experiments/configs/build_newton_lift_hold_contact_source_manifest.py`
- Launcher: `experiments/configs/launch_newton_lift_hold_contact_source_manifest_tmux.sh`
- Compute runner: `experiments/configs/run_newton_lift_hold_contact_source_manifest_in_alloc.sh`
- Manifest: `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
- Records CSV: `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/contact_source_records.csv`
- Log: `logs/newton/newton_lift_hold_contact_source_manifest_v1_20260627.log`

## Command

Executed inside the existing Curiosity tmux-held Slurm allocation:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
  bash experiments/configs/launch_newton_lift_hold_contact_source_manifest_tmux.sh
```

The compute-side runner uses the prebuilt local venv at `envs/newton/.venv` and
now fails immediately if `SLURM_JOB_ID` is not set.

## Result

Status: pass.

- Source runs: 10.
- Records: 3600.
- Contact count range: 29 to 63.
- Total feedback trigger count: 0.
- Schema promotion: blocked.
- Generated T-Rex fields: none.
- Failures: none.

Dataset fields:

- `newton.panda.step`
- `newton.panda.sim_time`
- `newton.contact.rigid_contact_count`
- `newton.object.body_q.z`
- `candidate.controller.phase_index`
- `candidate.controller.feedback_trigger_count`
- `candidate.controller.commanded_gripper_target`
- `candidate.controller.commanded_lift_target`

## Source Runs

The manifest includes nominal, ordinary-grid, and held-out scripted-feedback
rollouts:

- nominal: `nominal_cup`
- ordinary: `empty_low`, `empty_medium`, `half_low`, `half_medium`,
  `half_high`, `full_medium`, `full_high`
- held-out: `full_low`, `empty_high`

Each source run keeps direct contact-sheet and frame-browser paths in the
manifest under `source_runs`.

## Interpretation

Newton contact proxy is the first post-pivot tactile/contact source. It is
valid for Newton-native contact-aware diagnostics and future residual adapter
input audits.

It is not valid as T-Rex tactile F6, not valid as dense tactile deformation,
and not sufficient for exact T-Rex schema promotion. Real Taccel marker or FTac
evidence must remain under `taccel.marker.*` or `taccel.ftac.*` when added.

The shared strict lift-hold metrics for these scripted-feedback runs still fail
on the object-acceleration threshold, while visual/lift/hold/slip/drop/contact
gates pass. This manifest should therefore be treated as contact-source
evidence, not as an adaptation-success claim.
