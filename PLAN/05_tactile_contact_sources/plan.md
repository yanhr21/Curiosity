# Phase 05: Tactile And Contact Sources

## Goal

Add touch-like evidence only where it is real, useful, and clearly named.

## Allowed Namespaces

```text
newton.contact.*
newton.camera.*
newton.object.*
taccel.marker.*
taccel.ftac.*
candidate.*
```

## Forbidden Promotions

Do not create these fields unless their real contracts are satisfied:

```text
observation.tactile_f6
observation.tactile_deform.*
observation.images.*
observation.state
action
action_abs
```

## Completion Criteria

- Contact/tactile source improves or clarifies adaptation behavior.
- Every tactile visual has direct image paths.
- No partial source is renamed into T-Rex schema.

## Fusion Plan

Touch should enter both the policy and the curiosity forward model:

```text
vision_encoder(rgb, depth) -> z_v
tactile_encoder(contact_proxy, marker_flow, deform) -> z_t
proprio_encoder(joint, ee, gripper, phase) -> z_p
action_encoder(controller_params) -> z_a

fusion(z_v, z_t, z_p, z_a, masks) -> z
policy_head(z) -> residual controller params
forward_model(z, action) -> next object/contact/tactile prediction
```

The first tactile source can be Newton contact proxies. Taccel marker evidence
can be added only under `taccel.marker.*` after real nonzero data and visual
inspection exist. Dense tactile deformation can be added only after a real
nonuniform deformation stream passes its own gate.

## Modality Masking

Training must prevent the policy from collapsing into pure vision or pure
touch. Required masking modes:

- both vision and touch visible;
- vision masked, touch visible;
- touch masked, vision visible;
- partial vision mask;
- partial tactile mask.

After contact, include pure tactile windows:

```text
p(mask_vision | post_contact) = 0.3 -> 0.6 curriculum
p(mask_tactile | post_contact) = 0.1 -> 0.2
p(both_visible) remains nonzero
```

This is intended to make the robot stabilize and detect slip by touch once
contact has been established, while still requiring vision for approach and
global scene context.

## Balance Tests

Report vision-only, tactile-only, vision+tactile, shuffled tactile, and delayed
tactile results. Touch is considered causally useful only if vision+tactile
outperforms single-modality baselines and corrupted tactile hurts performance.

## Completed Newton Contact Source Conversion

2026-06-27: converted the Phase 04 scripted feedback lift-hold rollouts into a
namespace-preserving Newton contact/tactile source manifest.

- Config:
  `experiments/configs/newton_lift_hold_contact_source_manifest_v1.json`.
- Builder:
  `experiments/configs/build_newton_lift_hold_contact_source_manifest.py`.
- Runner:
  `experiments/configs/run_newton_lift_hold_contact_source_manifest_in_alloc.sh`.
- Launcher:
  `experiments/configs/launch_newton_lift_hold_contact_source_manifest_tmux.sh`.
- Slurm job: `154023`.
- Tmux session: `curiosity_next_source_alloc_20260626_232937`.
- Log:
  `logs/newton/newton_lift_hold_contact_source_manifest_v1_20260627.log`.
- Manifest:
  `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`.
- Records CSV:
  `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/contact_source_records.csv`.
- Report:
  `experiments/reports/2026-06-27_phase05_newton_contact_source_manifest_v1.md`.

Result:

- status: pass;
- source runs: `10` real Newton rollouts;
- timestep records: `3600`;
- contact proxy range: `29..63`;
- total feedback trigger count: `0`;
- generated T-Rex fields: `[]`;
- schema promotion: `blocked`.

The compute-side runner now fails if `SLURM_JOB_ID` is not set, so this NumPy
manifest build cannot be accidentally run as a login-node data-processing job.

Dataset/schema mismatch was not used as a stop gate. The conversion preserves
real source evidence under `newton.*` and `candidate.*` namespaces and does not
create `observation.*`, `action`, `action_abs`, calibrated F6, or dense tactile
deformation fields.

## Tactile/Contact Stream And Mask Contract

2026-06-27: added the active Newton contact proxy to the residual-adapter and
forward-model target contract.

- Spec: `docs/residual_adapter_forward_model_contract_v1.md`.
- Config: `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- Active contact stream: `newton.contact.rigid_contact_count`.
- Active forward target: `contact_proxy_next_step`.
- Blocked future tactile streams: `taccel.marker.*` and `taccel.ftac.*` until
  real nonzero, visually inspected evidence exists.

The contract defines both vision/contact visible, vision masked, contact
masked, partial vision mask, partial contact mask, and post-contact pure-touch
windows. It does not train a model and does not prove tactile benefit; those
claims still require the required ablations and held-out evaluation.

## Contact-Proxy Ablation Report V1

2026-06-27: reported the current contact-proxy ablation evidence from the
validated Phase 03 replay output.

- Report:
  `experiments/reports/2026-06-27_phase05_contact_proxy_ablation_report_v1.md`.
- Source replay:
  `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- Source manifest:
  `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`.
- Rollouts: 9.
- Held-out cells: `full_low`, `empty_high`.
- Tactile/contact source: `newton.contact_proxy_only`.

Reported diagnostic ablations:

- object-motion-only proxy for current vision-only reporting;
- Newton contact-proxy tactile/contact-only;
- object-motion plus contact proxy;
- shuffled contact proxy;
- delayed contact proxy.

This completes the current Phase 05 ablation-reporting item, but only as
replay diagnostics. It does not show a trained policy using tactile/contact
information and does not provide calibrated tactile F6 or dense deformation
evidence.
