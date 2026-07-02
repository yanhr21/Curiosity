# Video-Guided Active Carrying

## Core Claim

As of 2026-07-02, the target is not solved by existing robotics systems:

```text
A bipedal or humanoid robot with its own body proportions, mass, torque limits,
arm reach, and carrying capacity faces a box of unknown weight and shape,
actively probes it, chooses a stable low-cost posture for its own body, carries
it for a long duration, and learns with human/robot/simulation video as a
non-retargeting reference signal for RL.
```

The research gap is:

```text
video reference + active probing of unknown object dynamics
+ morphology-aware whole-body posture selection
+ harder held-out carrying evaluation
```

## Working Title

Video-guided, embodiment-aware active loco-manipulation for unknown-load
carrying.

## Key Principle

Video is a weak prior, not a motion target.

Allowed video information:

- task semantics;
- phase progress;
- object motion;
- coarse contact affordances;
- visual success/failure cues.

Forbidden use:

- human joint retargeting;
- robot joint retargeting;
- end-effector trajectory cloning;
- teleoperation replay;
- posture copying as the main claim.

## Why Active Probing Is Required

RGB or RGB-D video cannot reliably reveal mass, center of mass, friction,
internal fill, stiffness, or required grip force. Therefore video conditioning
must be paired with probing actions such as micro-lift, push-pull, grip ramp,
stance adjustment, hold-height adjustment, regrasp, contact redistribution, and
slow one-step carry tests.

## What The Policy Must Learn

The policy must choose carrying strategies that fit the robot's own body:

- front carry;
- low carry;
- chest or torso support;
- forearm support;
- asymmetric carry;
- squat-depth adjustment;
- stance widening;
- walking-speed reduction;
- regrasp before walking;
- abort when unsafe.

The same reference video should be allowed to produce different robot postures
for different morphologies and loads.

## Success Gate

No success claim is allowed unless the method:

- beats the strongest baseline on harder held-out objects and robot bodies;
- improves carry distance, carry duration, and efficiency;
- avoids safety regression in falls, drops, slip, contact loss, excessive
  torque, and object acceleration;
- proves video helps beyond no-video RL;
- proves active probing helps beyond video-only reward;
- proves it is not merely retargeting or behavior cloning.

## Active References

- Main survey: `docs/2026-07-02_research_overview.md`
- Robot carrying review: `docs/robot_carrying_capability_review.md`
- Video-conditioned learning review:
  `docs/video_conditioned_rl_review.md`
- Research program design: `docs/research_program_design.md`
