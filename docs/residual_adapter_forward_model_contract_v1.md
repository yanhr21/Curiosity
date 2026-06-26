# Residual Adapter And Forward-Model Target Contract V1

## Purpose

This contract defines the first learned-adaptation target surface for the
Newton lift-hold task. It is a training-preparation artifact, not a trained
model and not a placeholder world model.

The learned policy must remain a residual controller-parameter adapter around
the official Newton Panda hydro scripted infant prior. It must not replace the
prior with full low-level torque control.

## Source Evidence

Current allowed source evidence:

- Phase 02 no-adaptation Newton rollouts;
- Phase 04 scripted-feedback Newton rollouts;
- Phase 05 Newton contact source manifest;
- Phase 03 curiosity replay reward diagnostics.

Required source gates before any training run:

- fresh official Newton sanity JSON;
- camera/export summary JSON;
- automated visual validation JSON;
- manual visual inspection JSON with pass status;
- lift-hold metrics JSON;
- namespace-preserving source manifest with `schema_promotion=blocked`;
- no generated T-Rex fields.

## Inputs

The first residual adapter may use only namespace-preserving Newton and
candidate fields:

- `newton.panda.sim_time`
- `newton.panda.object_body_q`
- `newton.panda.rigid_contact_count`
- `newton.camera.color_rgba`
- `newton.camera.depth`
- `candidate.controller.phase_index`
- `candidate.controller.commanded_gripper_target`
- `candidate.controller.commanded_lift_target`
- `candidate.controller.feedback_trigger_count`
- `candidate.physics.body_mass_scale`
- `candidate.physics.shape_friction_scale`

The first tactile/contact stream is the Newton contact proxy:

- `newton.contact.rigid_contact_count`

Taccel fields may be added later only under explicit namespaces such as
`taccel.marker.*` or `taccel.ftac.*`.

## Residual Policy Outputs

The first learned adapter may output residuals for controller parameters only:

- gripper closure target delta;
- lift velocity scale delta;
- hold height target delta;
- regrasp trigger threshold delta;
- stabilization duration delta.

The adapter must not output full joint torques, low-level actions, T-Rex
`action`, or T-Rex `action_abs`.

## Forward-Model Targets

The learned forward-model target set is:

- object pose delta;
- object velocity;
- contact proxy at the next step;
- slip risk;
- contact-loss risk;
- lift-response residual under the current controller command;
- success/failure risk;
- tactile-marker response only when real `taccel.marker.*` evidence exists.

For the current Newton-only source, tactile-marker response is blocked and the
active tactile/contact target is `newton.contact.rigid_contact_count`.

## Modality Masks

Training must include these modality modes:

- both vision and touch visible;
- vision masked, touch/contact visible;
- touch/contact masked, vision visible;
- partial vision mask;
- partial touch/contact mask;
- post-contact pure-touch windows.

Initial post-contact mask curriculum:

```text
p(mask_vision | post_contact) = 0.3 -> 0.6
p(mask_tactile | post_contact) = 0.1 -> 0.2
p(both_visible) remains nonzero
```

## Required Ablations

Before claiming tactile or curiosity benefit, report:

- no adapter;
- scripted feedback;
- residual adapter without curiosity;
- residual adapter with curiosity;
- vision-only;
- tactile/contact-only;
- vision+tactile/contact;
- shuffled tactile/contact;
- delayed tactile/contact.

## Success Criteria

The learned adapter can be promoted only if it improves at least one held-out
metric without hiding failures:

- lift success;
- slip/drop rate;
- contact-loss rate;
- excessive-force/contact-proxy rate;
- object acceleration;
- adaptation speed after mismatch;
- success per contact-proxy integral.

Held-out `full_low` and `empty_high` must stay held-out for the first
generalization comparison.

## Forbidden Claims

This contract does not prove:

- learned world-model performance;
- learned residual-adapter performance;
- real tactile F6 availability;
- dense tactile deformation availability;
- exact T-Rex schema compatibility.

Any future implementation must record the official sanity check, source gates,
training command, environment, checkpoint path, metrics, visual paths, and
failure modes before reporting a learned result.
