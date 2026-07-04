# Plan 04: Non-Retargeting Video Prior

## Purpose

Add video conditioning without turning the method into retargeting. Video
should guide task progress and object-contact semantics while RL and probing
choose the robot-specific strategy.

## First Video Source

Use simulation-generated videos first. This avoids real dataset contamination
and gives controllable labels for debugging reward failure modes.

Initial references:

- successful simulated carry;
- failed simulated carry;
- different morphology carrying the same object;
- different object with same task semantics.

## Candidate Components

### XIRL-Style Progress Reward

Learn a task-progress embedding from video and use distance to goal/progress
as reward. This is the first non-retargeting baseline because it is explicitly
cross-embodiment imitation from observation.

### GraphIRL-Style Object Graph Reward

Represent robot, box, support surfaces, and goal as object-centric graph nodes.
Learn task progress from relations rather than body pose.

### Optional Later Components

- VIP-like visual reward if real video pretraining becomes allowed later.
- Vid2Robot-like prompt encoder only as an architectural comparison, not as
  behavior cloning.
- SUGAR as a video-driven humanoid baseline, clearly labeled and not promoted
  as the core method.

## Forbidden Uses

- copying joint angles;
- copying end-effector paths;
- copying footstep timing;
- copying a human carrying posture;
- using video-derived pseudo-force labels as truth;
- using hidden box mass as video-conditioned input.

## Required Ablations

- no video;
- wrong video;
- mismatched embodiment video;
- mismatched object video;
- video-only no probing;
- active probing no video;
- retargeting baseline;
- behavior-cloning baseline.

## Exit Criteria

- Video improves sample efficiency or held-out success over active probing
  without video.
- Wrong or mismatched videos degrade predictably rather than silently
  succeeding through hidden signals.
- The chosen posture remains morphology-dependent.

