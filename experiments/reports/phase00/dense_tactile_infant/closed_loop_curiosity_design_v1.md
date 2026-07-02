# Closed-Loop Curiosity Design

Date: 2026-07-01

Machine-readable record:
`experiments/configs/phase00/dense_tactile_infant/closed_loop_curiosity_design_v1.json`

Status: design ready, training not started.

## Prerequisite Evidence

The design starts only after active dense tactile/base evidence exists:

`experiments/configs/phase00/dense_tactile_infant/active_evidence_manifest_20260701_v1.json`

This evidence includes base grasp/lift/hold, scene video, left/right candidate
`Fn/Ft`, shear, contact-area proxy, candidate CoP proxy, hydro pressure/deform
proxy maps, and reference-video comparison assets.

## Forward Model

The forward model must predict dense tactile/contact/mechanics, not only object
height or contact count.

Required target families:

- object pose delta, velocity, lift height, acceleration, slip/drop precursor;
- left/right pressure or `Fn` maps;
- left/right compression/deform proxy maps;
- left/right shear fields;
- contact masks;
- candidate `Ft`, area proxy, and CoP proxy;
- grip balance, stress proxy, shear proxy, and force/contact cost.

## Policy Actions

The policy must be able to actively probe and repair contact, not only adjust
small lift/hold parameters.

Required residual action families:

- grip-force delta;
- gripper-width delta;
- lift-velocity scale;
- hold-height delta;
- wrist micro-probing;
- left/right pressure balancing;
- regrasp trigger;
- shear-minimizing lateral adjustment.

## Intrinsic Reward

The intrinsic reward must enter online policy optimization and change future
rollout data. Sample reweighting alone is not closed-loop curiosity.

Reward terms:

```text
learning_progress_dense_tactile_prediction
+ learning_progress_object_mechanics_prediction
+ bounded_novel_contact_state
+ controllable_disagreement
- drop/slip/contact-loss penalty
- force/contact cost
- no-op penalty
```

## Masking And Ablations

Tactile must stay online and causal. Required modes:

- vision+tactile;
- tactile-only after masking vision post-contact;
- vision-only;
- noisy tactile;
- delayed tactile;
- shuffled or mismatched tactile.

The success condition is not just task success. Vision+tactile must beat
vision-only, and corrupted tactile must hurt without hiding safety regression.

## Boundary

This is not training, not a checkpoint, and not curiosity success. It is the
active design contract for the next training stage.
