# Curiosity Reward Spec V1

## Purpose

Phase 03 introduces curiosity as a physical-learning signal for the Newton
lift-hold cup task. The first implementation is a replay evaluator on validated
Phase 02 baseline rollouts. It does not train a policy, does not train a world
model, and does not create a placeholder T-Rex/VQ-VAE model.

The goal is to verify that the reward has the right shape before it is used for
closed-loop adaptation:

```text
intrinsic_reward =
  learning_progress(object/contact/tactile prediction)
+ controllable_disagreement
+ bounded_useful_change
- safety_penalty
- no_op_penalty
- excessive_force_penalty
```

## Required Gates

Each replay-evaluated rollout must already have:

- a fresh official Newton sanity JSON;
- camera export summary;
- automated visual validation JSON;
- manual visual inspection JSON with `status == pass`;
- Phase 02 lift-hold metrics JSON.

If any gate is missing or failed, Phase 03 replay evaluation for that rollout is
invalid.

## Signals

The available Newton-native Phase 02 rollout fields are:

- `newton.panda.object_body_q`: object pose, used for object position, velocity,
  acceleration, lift, and slip;
- `newton.panda.rigid_contact_count`: high-frequency contact/tactile proxy;
- `candidate.controller.phase_index`: scripted controller phase;
- `candidate.controller.commanded_gripper_target`: command context;
- `candidate.controller.commanded_lift_target`: command context;
- `candidate.physics.body_mass_scale`: requested mass provenance;
- `candidate.physics.shape_friction_scale`: requested friction provenance.

No official tactile marker field is present yet. The evaluator therefore logs
`newton.contact_proxy_only` as the tactile source and blocks any exact T-Rex
schema promotion.

## Diagnostic Predictors

V1 uses diagnostic replay predictors only:

- object motion: constant-velocity one-step prediction;
- contact response: contact-persistence one-step prediction;
- delayed tactile: one-step delayed contact proxy;
- shuffled tactile: deterministic replay shuffle for ablation only.

These are not learned world models and must not be described as final T-Rex,
VQ-VAE, or world-model progress. They only validate reward wiring on existing
rollouts before heavier model-based curiosity is attempted.

## Components

- `object_motion_prediction_error`: one-step object position prediction error.
- `contact_prediction_error`: one-step contact proxy prediction error.
- `tactile_proxy_prediction_error`: contact-proxy prediction error when tactile
  marker fields are unavailable.
- `controllable_disagreement`: mismatch between commanded lift change and
  observed object height change during active lift/hold phases.
- `bounded_useful_change`: clipped useful lift progress while contact is present.
- `safety_penalty`: unstable acceleration, drop, and contact-force proxy excess.
- `excessive_force_penalty`: contact proxy above the configured force threshold.
- `no_op_penalty`: active command frames with no contact and no useful object
  motion.
- `learning_progress_proxy`: replay-window decrease in prediction error; this is
  logged as a proxy, not as policy learning progress.

## Ablations

The replay evaluator reports all required Phase 03 ablation rewards from the
same validated rollout:

- `no_curiosity`;
- `random_intrinsic`;
- `object_motion_only`;
- `contact_only`;
- `tactile_only`;
- `vision_tactile`;
- `shuffled_tactile`;
- `delayed_tactile`.

The ablation outputs are diagnostic rewards for analysis. They are not yet used
to update policy parameters.
