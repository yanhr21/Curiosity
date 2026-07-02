# Research Program Design

## Working Title

Video-guided, embodiment-aware active loco-manipulation for unknown-load
carrying.

## Core Problem

A robot can walk and balance. A box appears. The box has unknown mass, shape,
center of mass, friction, and possibly internal fill. Different robot bodies
have different mass, limb lengths, torque limits, hand geometry, and carrying
capacity. The robot may receive a video of a human, another robot, or a
simulation carrying the object, but it must not retarget that motion. It must
actively try the object and choose a stable, efficient posture for itself.

## System Decomposition

### 1. Base Whole-Body Capability

The robot must already be able to:

- walk;
- stop and balance;
- squat or bend;
- reach toward the object;
- establish hand/forearm/torso contact;
- lift lightly loaded objects;
- recover from small disturbances.

This is not the research contribution. It is the base controller.

### 2. Video Prior

The video module should output weak reference variables:

- task phase: approach, contact, test, lift, stabilize, walk, place;
- object progress: lifted, translated, lowered, dropped;
- candidate contact regions;
- coarse object orientation and size;
- success/failure visual state;
- phase transition constraints.

It should not output:

- human joint angles;
- robot joint angles;
- full pose targets;
- end-effector trajectory to copy;
- fixed footstep plan.

### 3. Active Load Belief

The robot should maintain a belief over:

- mass;
- center of mass;
- friction;
- object compliance;
- stable contact patches;
- slip probability;
- required normal force;
- expected torque/energy cost for candidate postures.

The belief must update from probing:

- micro-lift;
- push-pull;
- tilt;
- grip ramp;
- lateral shear test;
- hold-height perturbation;
- one-step carry test;
- contact redistribution.

### 4. Morphology-Aware Posture Search

Candidate strategies should include:

- high front carry;
- low front carry;
- chest-supported carry;
- forearm-supported carry;
- asymmetric carry;
- wide-stance lift;
- squat-dominant lift;
- hip-hinge lift;
- slow gait with low hold height;
- regrasp before walking;
- abort and request different strategy if unsafe.

The policy should adapt these choices to:

- arm length;
- torso height;
- center of mass;
- foot support polygon;
- torque limits;
- hand contact area;
- actuator thermal/effort budget;
- balance controller.

### 5. RL Objective

The reward should include:

Positive terms:

- task progress from video-derived phase prior;
- object lift and carry distance;
- stable hold duration;
- successful placement;
- controllable information gain about load dynamics;
- posture efficiency improvement after probing.

Negative terms:

- falls;
- drops;
- slip;
- contact loss;
- excessive object acceleration;
- excessive joint torque;
- high energy cost;
- low balance margin;
- unsafe contact force;
- unnecessary probing after confidence is high;
- copying reference posture when it is inefficient or infeasible.

## Training Curriculum

### Stage 0: Diagnostics Only

- Fixed robot, known lightweight box.
- Validate base controller, sensors, object pose tracking, and metrics.
- No success claims.

### Stage 1: Unknown Load Probing

- Same robot, varied mass and center of mass.
- No video or simple video prior.
- Goal: learn active load belief and safe probing.

### Stage 2: Video Prior

- Add human/robot/simulation reference videos.
- Reward uses video progress/contact priors.
- Compare correct, wrong, and mismatched videos.

### Stage 3: Morphology Variation

- Vary robot body proportions and torque limits in simulation.
- Require posture diversity and robot-specific strategy choice.

### Stage 4: Held-Out Carrying

- Held-out boxes, shapes, friction, centers of mass, and videos.
- Strongest-baseline comparison required.

### Stage 5: Sim-to-Real Or High-Fidelity Transfer

- Only after Stage 4 shows stable held-out improvement.
- Document exact simulator, controller, assets, and safety constraints.

## Baselines

Required baselines:

- fixed posture base controller;
- scripted probing plus WBC;
- no-video RL;
- video reward only, no active probing;
- active probing only, no video;
- behavior cloning from video-conditioned data;
- retargeting baseline;
- oracle mass and center-of-mass baseline;
- strongest available humanoid loco-manipulation method or documented blocker.

## Metrics

Task metrics:

- carry success;
- carry distance;
- carry duration;
- placement success;
- attempts to successful lift.

Safety metrics:

- fall rate;
- drop rate;
- slip;
- contact loss;
- object acceleration;
- collision or unsafe contact;
- peak torque;
- balance margin.

Efficiency metrics:

- energy;
- torque integral;
- peak joint effort;
- hold height versus effort;
- gait speed versus stability;
- probing cost.

Generalization metrics:

- held-out mass;
- held-out center of mass;
- held-out friction;
- held-out shape;
- held-out robot morphology;
- held-out video embodiment.

Interpretability metrics:

- posterior uncertainty reduction after probing;
- posture changes as load/morphology changes;
- strategy distribution across robots;
- failure mode classification.

## Success Condition

The project can only claim success if:

```text
On harder held-out object and robot settings, the method beats the strongest
baseline in carry duration/distance and efficiency, without safety regression,
and ablations show that both video priors and active probing causally matter.
```

## Early Red Flags

- Success only on one robot body.
- Success only on one mass/friction cell.
- Policy copies human posture even when torque cost is poor.
- Video-only reward improves appearance but not carrying metrics.
- Probing does not change the belief or future action distribution.
- Retargeting baseline performs the same as the proposed method.
- Oracle load baseline is dramatically better and the proposed method never
  closes the gap.
- Wrong video performs as well as correct video.

## First Concrete Milestone

Build a simulator task where:

- a walking/balancing humanoid has a basic carry controller;
- boxes vary in mass, center of mass, size, and friction;
- reference videos are stored as phase/object-motion observations, not
  retargeting targets;
- the policy can perform at least three probing actions before full lift;
- metrics distinguish raw lift from stable long-duration carry;
- a no-video and fixed-posture baseline are already strong enough to be fair.
