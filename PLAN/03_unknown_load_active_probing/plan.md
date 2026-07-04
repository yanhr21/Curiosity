# Plan 03: Unknown Load And Active Probing

## Purpose

Turn a known box manipulation task into unknown-load carrying. The robot must
infer object mechanics through action-conditioned feedback, not hidden
privileged labels.

## Environment Extensions

Randomize at reset:

- box mass;
- center of mass;
- dimensions and aspect ratio;
- friction;
- contact affordances or handles;
- initial pose;
- target distance and duration;
- mild external perturbations.

Privileged values may be used for training critics or diagnostics only if the
final policy and success claim do not depend on them.

## Probing Action Space

The policy must be able to choose:

- micro-lift;
- push-pull;
- grip-force ramp;
- regrasp;
- stance widening;
- squat-depth adjustment;
- hold-height adjustment;
- torso/forearm/chest support;
- slow one-step carry test;
- abort-and-reset behavior.

## Belief State

The method should maintain an online estimate or latent belief over:

- mass range;
- center-of-mass offset;
- friction/slip risk;
- required support force;
- contact stability;
- whether continuing is unsafe.

## Baselines

- no probing;
- scripted probing;
- privileged oracle load;
- fixed posture;
- active probing without video.

## Exit Criteria

- Active probing improves held-out unknown-load carrying over no-probing and
  fixed-posture baselines.
- Safety metrics do not regress.
- The report shows what the robot did before committing to carry.

