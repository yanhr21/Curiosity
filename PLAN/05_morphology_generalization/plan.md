# Plan 05: Morphology Generalization

## Purpose

Show that the same task/video prior leads to different feasible carrying
strategies for different robot bodies and limits.

## Morphology Variations

Start with simulator-controlled perturbations:

- arm length;
- torso height;
- total mass;
- mass distribution;
- torque limits;
- joint ranges;
- hand/forearm/chest contact geometry;
- foot size or support margin.

If supported by the platform, later compare different humanoid embodiments
instead of only scaling G1.

## Strategy Diversity

The policy should be allowed to choose:

- front carry;
- low carry;
- chest-supported carry;
- forearm-supported carry;
- asymmetric carry;
- regrasp;
- widened stance;
- slower gait;
- abort if unsafe.

## Baselines

- one fixed carry posture for all bodies;
- retargeted human posture;
- no morphology randomization;
- oracle morphology but no probing;
- probing with morphology hidden or corrupted.

## Exit Criteria

- The same reference video does not force the same posture across bodies.
- Strategy differences are explained by metrics: torque cost, balance margin,
  slip/drop risk, and carry duration.
- Held-out morphologies improve over fixed-pose and retargeting baselines.

