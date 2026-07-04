# Execution Path: Video-Guided Active Unknown-Load Carrying

Date: 2026-07-02.

This document turns the survey into an executable path. It deliberately does
not treat the current repository as a successful prior system. The current
target is:

```text
video reference as weak prior
+ active probing of unknown object dynamics
+ embodiment-aware whole-body posture selection
+ long-duration box carrying in simulation first
```

## Non-Negotiable Boundaries

- Do not use T-Rex, real-world-only robot datasets, or real-scene-only models
  in the first execution path.
- Do not download real datasets at this stage.
- Do not turn the project into retargeting, teleoperation replay, motion
  shadowing, or end-effector trajectory cloning.
- Do not claim success from a known-load box, a single robot body, a copied
  pose, a table-top arm task, or a video-only reward.
- On login nodes, only do lightweight git, text inspection, and documentation.
  Simulation, rendering, model loading, training, evaluation, dataset
  conversion, and visualization generation must run inside a Curiosity-owned
  tmux-held Slurm allocation.

## Primary Simulation Platform

### Choice

Use Isaac Lab / Isaac Lab-Arena as the first simulation platform.

Reason:

- Isaac Lab-Arena is built on Isaac Lab and explicitly includes a G1
  loco-manipulation pick-and-place task where a G1 humanoid navigates, picks
  up a box, and places it in a bin.
- Its environment model separates scene, embodiment, and task, which is useful
  for morphology and object randomization.
- It is the closest open platform I found to "humanoid + walking + box
  manipulation" without starting from real data.

Important caveat:

- Isaac Lab-Arena is alpha software. The main branch targets Isaac Sim 6.0,
  Isaac Lab 3.0, Python >= 3.12, and Docker-only source installation. It may
  be unstable.

Local source:

- `external/IsaacLab-Arena` at commit `8a74e79`.

Primary external references:

- https://github.com/isaac-sim/IsaacLab-Arena
- https://isaac-sim.github.io/IsaacLab-Arena/main/index.html
- https://huggingface.co/nvidia/isaaclab-arena-envs

## Secondary Execution Stacks

### VIRAL / GR00T-VisualSim2Real

Use as a fallback/control stack for simulation-trained G1 loco-manipulation
and teacher-student visual policy infrastructure.

Reason:

- It trains a privileged PPO teacher and a vision student for Unitree G1
  loco-manipulation in Isaac Sim 5.1 / Isaac Lab.
- It is simulation-first and does not require the first phase to touch real
  datasets.

Limitation:

- It is not specifically unknown-load active carrying and not video-guided
  non-retargeting RL.

Local source:

- `external/GR00T-VisualSim2Real` at commit `92bf086`.

### WBC-AGILE

Use as a whole-body controller and evaluation workflow reference, especially
if the primary Arena path needs a stronger locomotion/control base.

Reason:

- It provides Isaac-Lab-based humanoid whole-body RL workflows, G1 support,
  teacher-student distillation, evaluation reporting, and sim-to-MuJoCo
  validation.

Limitation:

- It is not itself a box-carrying unknown-load video-conditioned system.

Local source:

- `external/WBC-AGILE` at commit `7259792`.

### SUGAR

Use as a video-driven humanoid loco-manipulation reference and later baseline,
not as the main claim.

Reason:

- It is built on IsaacLab and includes tasks such as `CarryBox`.
- It is one of the closest open codebases to human-video-driven humanoid
  loco-manipulation.

Limitation:

- It converts third-person human-object videos into physically refined motion
  priors and hierarchical policies. This is close to retargeting/motion-prior
  territory and must not become the project's core claim.
- The README requires processed data and demo checkpoints for ordinary use.
  Those downloads are intentionally deferred.
- The RGB-D human-video processing pipeline is not released as of the checked
  README.

Local source:

- `external/SUGAR` at commit `01fe123`.

Primary external references:

- https://github.com/tianshuwu/SUGAR
- https://arxiv.org/html/2605.20373v1

## Video Reward / Non-Retargeting Components

### XIRL

Role:

- First baseline for non-retargeting video reward learning.
- Use the idea of task-progress embeddings from cross-embodiment videos.

Limitation:

- Original tasks are small relative to whole-body carrying.
- Do not download X-MAGICAL or real datasets yet; first use simulation-generated
  reference videos if video reward experiments start.

Local source:

- `external/google-research-xirl` at commit `62457e1`.

References:

- https://arxiv.org/abs/2106.03911
- https://x-irl.github.io/

### GraphIRL

Role:

- Reference for object-centric graph reward learning from diverse third-person
  videos.
- More aligned with this project than raw pose imitation because the box and
  robot-object relation can be the primary abstraction.

Limitation:

- Original tasks are reach, push, and peg-in-box scale, not humanoid carrying.
- Do not download its dataset or trained models yet.

Local source:

- `external/graph-inverse-rl` at commit `7d06634`.

References:

- https://arxiv.org/abs/2207.14299
- https://sateeshkumar21.github.io/GraphIRL/

### VIP And Vid2Robot

Role:

- VIP is a reference for dense visual rewards from unlabeled videos.
- Vid2Robot is a reference for video-conditioned policy architectures.

Limitations:

- VIP commonly relies on large human-video pretraining such as Ego4D; this is
  deferred because the first path avoids real datasets.
- Vid2Robot is mainly supervised video-conditioned policy learning, not the
  target RL-with-active-probing setup.
- Neither should be used as the first executable core.

References:

- https://arxiv.org/abs/2210.00030
- https://sites.google.com/view/vip-rl
- https://arxiv.org/abs/2403.12943
- https://vid2robot.github.io/

## Load Adaptation References

### FALCON

Role:

- Reference for force-adaptive humanoid loco-manipulation under unknown
  end-effector forces.
- Its force curriculum and lower/upper-body decomposition are directly useful
  for robust carrying under load disturbance.

Limitation:

- It is not video-guided and not autonomous unknown-box posture discovery.
- Local clone was not kept because checkout stalled; use sparse checkout later
  if needed.

References:

- https://arxiv.org/abs/2505.06776
- https://lecar-lab.github.io/falcon-humanoid/
- https://github.com/LeCAR-Lab/FALCON

### SplitAdapter

Role:

- Strong conceptual reference for separating object/load context from robot
  dynamics context.
- Particularly relevant to unknown mass and height variation.

Limitation:

- Treat as paper/reference unless official code becomes available and is
  verified.
- It adapts a frozen box manipulation policy; it does not solve video-guided
  active posture discovery by itself.

Reference:

- https://arxiv.org/abs/2606.03297

## Actual Build Path

### Stage 0: Source And Version Inventory

Goal:

- Freeze the list of usable open sources, commits, version constraints, and
  forbidden downloads.

Exit condition:

- A written source inventory exists and names the primary platform, fallback
  stacks, video-reward components, and explicit non-goals.

### Stage 1: Compute-Node Platform Preflight

Goal:

- In a Curiosity-owned tmux-held Slurm allocation, choose exactly one Isaac
  stack to execute first.

Preferred order:

1. Isaac Lab-Arena release/0.2.x or main with Isaac Sim 6.0 / Isaac Lab 3.0.
2. If Arena fails on cluster compatibility, use GR00T-VisualSim2Real or
   WBC-AGILE on Isaac Sim 5.1 / Isaac Lab 2.3.x.

Exit condition:

- One stack imports and runs its own official smoke test in a compute
  allocation. This is only a platform smoke test, not an experiment.

### Stage 2: Reproduce A Simulated Box Loco-Manipulation Task

Goal:

- Run an existing simulated G1 box pick/place or carry-like task without any
  new method claim.

Initial target:

- Isaac Lab-Arena `g1_locomanip_pnp`: pick up the brown box from a shelf and
  place it into a bin/tray.

Exit condition:

- One MP4 rollout, one log, exact command, commit, environment version, and
  a short report. Label it "base task reproduction", not success.

### Stage 3: Convert To Unknown-Load Carrying

Goal:

- Extend the task from pick/place to carrying under randomized object
  mechanics.

Randomize:

- mass;
- center of mass;
- box dimensions;
- friction;
- handle/contact affordance;
- initial pose;
- target carry distance;
- required carry duration.

Metrics:

- carry distance and duration;
- drop/slip/contact-loss;
- falls and recovery;
- object acceleration;
- torque/energy cost;
- peak torque and balance margin.

Exit condition:

- A no-video baseline environment with mechanics randomization and metrics.

### Stage 4: Add Active Probing

Goal:

- Make unknown-load inference an action-conditioned online process, not a
  privileged input.

Required probing behaviors:

- micro-lift;
- push-pull;
- grip-force ramp;
- stance widening;
- hold-height adjustment;
- arm/torso contact redistribution;
- slow one-step carry test;
- abort or regrasp when unsafe.

Exit condition:

- A probing policy or controller improves over no-probing baseline on held-out
  loads without safety regression.

### Stage 5: Add Non-Retargeting Video Prior

Goal:

- Use video only for task progress, object motion, and contact-affordance
  priors.

First video source:

- Simulation-generated reference videos from the same platform, because real
  datasets are intentionally deferred.

Candidate methods:

- XIRL-style progress embedding reward.
- GraphIRL-style object-centric graph progress reward.
- Later: VIP-like visual reward or Vid2Robot-like prompt encoder only if
  useful and clearly not trajectory cloning.

Forbidden:

- joint retargeting;
- footstep timing imitation;
- end-effector trajectory tracking as the main objective;
- hidden weight or force labels from video.

Exit condition:

- Video prior improves over no-video active probing and fails gracefully on
  wrong or mismatched videos.

### Stage 6: Morphology Generalization

Goal:

- Prove the policy selects different feasible strategies for different robot
  bodies and limits.

Variations:

- arm length;
- torso height;
- mass distribution;
- torque limits;
- hand/forearm/chest contact geometry;
- balance margin constraints.

Exit condition:

- Same reference video produces different stable carrying strategies when
  morphology changes, with metric improvement over fixed-pose baselines.

### Stage 7: Claim Gate

Goal:

- Decide whether the method is a real result, negative result, or only an
  engineering milestone.

Required ablations:

- no-video;
- wrong-video;
- mismatched-embodiment video;
- retargeting baseline;
- behavior-cloning baseline;
- no-probing;
- active probing only;
- oracle load;
- corrupted/delayed contact or force feedback;
- fixed posture.

Success requires:

- stronger held-out success than the best baseline;
- no safety regression;
- evidence that video and probing both causally help;
- morphology-dependent posture selection;
- full reporting and MP4 rollout evidence.

