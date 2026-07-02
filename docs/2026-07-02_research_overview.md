# 2026-07-02 Research Overview: Video-Guided Active Carrying

## Bottom Line

As of 2026-07-02, I did not find a complete system that solves the target
problem:

> A bipedal or humanoid robot with body-specific proportions, mass, arm reach,
> torque limits, and carrying capacity faces a box of unknown weight and shape;
> it actively tries the object, chooses an energy-efficient and stable posture
> suitable for its own body, carries it for a long duration, and learns with
> human/robot/simulation video as a non-retargeting reference signal for RL.

The important positive finding is that the ingredients now exist:

1. Real humanoids can already perform constrained box carrying, material
   handling, forceful loco-manipulation, and balance under external load.
2. Video-conditioned policies, visual reward learning, and cross-embodiment
   imitation-from-observation exist.
3. Most video methods remain table-top manipulation, behavior cloning,
   video-reward pretraining, or retargeting/teleoperation/shadowing.
4. The open research gap is the combination of video reference, active
   probing of unknown object dynamics, and morphology-aware whole-body posture
   selection for long-duration carrying.

The project should therefore not be framed as "imitate a human carrying a
box." The stronger and more defensible framing is:

```text
Video-guided, embodiment-aware active loco-manipulation
for unknown-load carrying.
```

## Strict Interpretation

The reference video should provide only weak, cross-embodiment information:

- task semantics;
- task phase or progress;
- object-motion direction and endpoints;
- coarse contact-affordance priors;
- success/failure visual cues;
- maybe phase-level style priors such as "lift before walking" or "hold close
  to body."

The reference video must not provide:

- human joint targets;
- robot joint targets;
- end-effector trajectory to track;
- footstep timing to clone;
- a fixed body posture to copy;
- hidden load parameters.

The robot must choose posture and carrying strategy using its own morphology,
balance controller, torque limits, contact feedback, and online probing data.

## Why Video Is Not Enough

RGB or RGB-D video can show that an object moved, how a demonstrator arranged
the body, and which surface was contacted. It cannot reliably reveal:

- object mass;
- center of mass;
- internal fill distribution;
- surface friction;
- compliance or stiffness;
- required grip force;
- how the same posture maps onto another robot's limb lengths and torque
  limits.

Therefore video conditioning cannot replace active probing. A valid system
needs behaviors such as micro-lift, push-pull, grip ramping, body repositioning,
stance widening, hold-height adjustment, arm/torso contact redistribution, and
gait speed modulation.

## Capability Map

### Humanoid Carrying And Loco-Manipulation

The closest academic systems demonstrate that humanoid robots can learn or
execute pieces of box carrying and load-bearing loco-manipulation:

- Digit box loco-manipulation demonstrates walking to a box, pickup, carry,
  and place-down with sim-to-real RL, but the real system still uses a modular
  skill pipeline and does not solve autonomous posture discovery for unknown
  load properties.
- BHR10 hybrid RL visuomotor loco-manipulation handles load-carrying and door
  opening with depth and proprioception, but it is still structured by a
  hand-designed finite-state process and does not use video reference.
- FALCON improves robustness to unknown end-effector forces on humanoids, but
  it addresses force-adaptive tracking and loco-manipulation, not
  video-guided posture choice for unknown boxes.
- JAXON work on unknown mass/friction carrying shows that explicit mechanics
  reasoning and alternative support strategies are useful, but it is a
  manually staged controller rather than learned, video-conditioned active
  exploration.

### Video-Conditioned Or Video-Reward Learning

The closest video-learning systems demonstrate relevant mechanisms:

- XIRL learns cross-embodiment visual reward functions from videos and then
  trains policies with RL.
- GraphIRL learns graph-abstraction rewards from diverse third-person videos.
- VIP learns visual rewards and representations from unlabeled human videos
  such as Ego4D and can support downstream robot learning.
- Vid2Robot conditions a robot policy on human video demonstrations, but the
  method is primarily supervised policy learning for manipulation, not
  whole-body RL with active unknown-load probing.

These methods are valuable sources of design ideas, but none directly solves
the target system.

## Research Hypothesis

The central hypothesis should be:

> A video-derived progress/contact prior plus active dynamics probing can
> outperform no-video RL, retargeting, fixed-posture baselines, and scripted
> probing on held-out unknown-load carrying tasks, while adapting posture to
> robot morphology without safety regression.

This hypothesis is narrow enough to test and strong enough to be meaningful.

## Minimal Valid Experiment Design

The evaluation must vary both object and robot:

- object weight;
- object center of mass;
- shape and size;
- friction;
- handle availability;
- internal fill or compliance;
- robot height;
- arm length;
- torso length;
- mass distribution;
- torque limits;
- hand/forearm/chest contact geometry.

The method should be compared against:

- fixed carrying posture;
- scripted probing plus hand-designed controller;
- no-video RL;
- video reward only, no active probing;
- active probing only, no video;
- behavior cloning or video-conditioned supervised policy;
- retargeting baseline;
- oracle load/center-of-mass baseline;
- wrong-video or mismatched-video baseline.

Required metrics:

- carry distance;
- carry duration;
- drop rate;
- slip;
- contact loss;
- fall rate;
- recovery after perturbation;
- object acceleration;
- energy or torque cost;
- peak joint torque;
- balance margin;
- number and type of probing actions;
- strategy diversity across bodies and load distributions.

## Non-Claim Boundaries

Do not claim the target is solved when:

- a humanoid carries a known box in a structured demo;
- a robot follows a retargeted human motion;
- a policy imitates a video without active probing;
- a table-top arm uses video conditioning;
- a single morphology succeeds on a single load distribution;
- lower training loss or prettier rollout videos improve without held-out
  carrying metrics;
- force or load parameters are secretly provided as privileged inputs.

## Source Map

Primary and near-primary sources used for the survey:

- Digit box loco-manipulation: https://arxiv.org/abs/2310.03191
- Digit paper HTML: https://arxiv.org/html/2310.03191v1
- Robust Visuomotor Control for Humanoid Loco-Manipulation Using Hybrid RL:
  https://www.mdpi.com/2313-7673/10/7/469
- FALCON: https://arxiv.org/abs/2505.06776
- FALCON project: https://lecar-lab.github.io/falcon-humanoid/
- JAXON unknown mass/friction carrying:
  https://crlab.cs.columbia.edu/humanoids_2018_proceedings/media/files/0188.pdf
- XIRL: https://arxiv.org/abs/2106.03911
- XIRL project: https://x-irl.github.io/
- GraphIRL: https://arxiv.org/abs/2207.14299
- GraphIRL project: https://sateeshkumar21.github.io/GraphIRL/
- VIP: https://arxiv.org/abs/2210.00030
- VIP project: https://sites.google.com/view/vip-rl
- Vid2Robot: https://arxiv.org/html/2403.12943
- Vid2Robot project: https://vid2robot.github.io/
- Figure BMW production note:
  https://www.figure.ai/news/production-at-bmw
- Agility Digit solutions:
  https://www.agilityrobotics.com/solutions
- Boston Dynamics Atlas LBM blog:
  https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- TRI/Boston Dynamics LBM release:
  https://www.tri.global/news/ai-powered-robot-boston-dynamics-and-toyota-research-institute-takes-key-step-towards-general
