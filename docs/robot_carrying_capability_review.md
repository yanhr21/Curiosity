# Robot Carrying Capability Review

## Question

Can current virtual or real humanoid robots carry objects and autonomously
select suitable carrying posture for unknown weight, shape, and robot body?

## Short Answer

They can carry objects in constrained settings. They cannot yet be credited
with the full target: unknown box properties, active probing, morphology-aware
posture selection, long-duration carrying, and non-retargeting video-guided RL.

## Digit: Sim-to-Real Humanoid Box Loco-Manipulation

Source: https://arxiv.org/abs/2310.03191

This is one of the closest academic systems. It trains box pickup and carrying
skills for Agility Digit and demonstrates real-robot transfer. The paper
targets walking to a box, stopping, picking it up, carrying it to another
table, and placing it down. The simulation setup randomizes box properties and
reports strong pickup performance.

What it supports:

- humanoid box pickup and carrying on real hardware;
- sim-to-real RL for several whole-body skills;
- varying box sizes, weights, poses, and locations in simulation;
- real hardware demonstrations with boxes in a constrained setup.

Limitations for this project:

- The full system is composed from separate skills rather than a single
  autonomous active probing policy.
- The hardware setup uses engineered perception and task structure.
- Real-world trials do not prove self-selected posture for unknown load.
- There is no non-retargeting reference-video learning component.
- The goal is successful transport, not long-duration energy-efficient posture
  discovery across robot morphologies.

Use in this project:

- Strong baseline for humanoid box loco-manipulation.
- Evidence that base carrying skill is feasible.
- Not sufficient as the target method.

## BHR10 Hybrid RL Visuomotor Loco-Manipulation

Source: https://www.mdpi.com/2313-7673/10/7/469

This work combines depth/proprioception, model-free RL in task space, and
model-based whole-body control for humanoid load carrying and door opening. It
reports an 83% overall success rate across real tasks and is closer to unknown
pose/size/weight handling than many demonstrations.

What it supports:

- real humanoid load-carrying experiments;
- visuomotor control with depth input;
- task-space policy outputs executed through whole-body control;
- reaction to environmental variation.

Limitations for this project:

- The task is not long-duration unknown-load carrying.
- It still relies on a structured state machine/task process.
- It does not use video reference.
- It does not demonstrate cross-morphology posture selection.
- It does not optimize "省力" posture under robot-specific body limits as the
  central objective.

Use in this project:

- Useful reference for hybrid RL plus WBC architecture.
- Useful baseline for task-space loco-manipulation.
- Not a solution to video-guided active posture discovery.

## FALCON: Force-Adaptive Humanoid Loco-Manipulation

Sources:

- https://arxiv.org/abs/2505.06776
- https://lecar-lab.github.io/falcon-humanoid/

FALCON trains humanoid policies robust to external end-effector forces using a
dual-agent RL framework and torque-limit-aware force curriculum. It deploys
across humanoids for tasks such as payload transport, cart pulling, and door
opening.

What it supports:

- robust locomotion under external load;
- force-adaptive upper-body control;
- deployment on multiple humanoid platforms;
- explicit concern for torque limits and force disturbances.

Limitations for this project:

- It is not video-conditioned.
- It does not focus on active estimation of object mass/center of mass.
- It does not choose carrying posture from reference video.
- Payload transport is not the same as unknown box pickup, posture selection,
  and long-duration carrying.

Use in this project:

- Important component or baseline for force-adaptive locomotion.
- Strong evidence that unknown external forces are becoming tractable.
- Not enough for unknown-load carrying with video priors.

## JAXON Unknown Mass/Friction Carrying On The Head

Source:
https://crlab.cs.columbia.edu/humanoids_2018_proceedings/media/files/0188.pdf

This work proposes lifting and carrying an object of unknown mass properties
and friction on the head by a humanoid robot. It uses a support strategy in
which the object is stabilized by both hands and the head.

What it supports:

- explicit handling of unknown object mass/friction;
- alternative whole-body support strategy;
- real humanoid implementation;
- recognition that carrying posture can change task feasibility.

Limitations for this project:

- It is a hand-designed multi-stage system.
- It is not learned active exploration.
- It is not video-conditioned.
- It does not evaluate morphology-dependent policy learning across robots.

Use in this project:

- Strong mechanical inspiration: carrying is not only a hand grasp problem.
- Supports including torso/head/forearm/body contacts in strategy space.

## Commercial Humanoid Evidence

### Figure 02 At BMW

Source: https://www.figure.ai/news/production-at-bmw

Figure reports 90,000+ parts loaded and 1,250+ runtime hours at BMW. This is
important evidence that humanoid material handling is becoming operational.

However, this is not proof of autonomous unknown-box posture discovery. It is
industrial part handling in a constrained production task.

### Agility Digit

Source: https://www.agilityrobotics.com/solutions

Agility lists Digit with 35 lb carrying capacity and 4 hour battery life. This
supports practical payload capacity and warehouse relevance.

It does not by itself prove unknown weight/shape probing or morphology-aware
video-guided posture selection.

### Boston Dynamics / TRI Atlas Large Behavior Models

Sources:

- https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- https://www.tri.global/news/ai-powered-robot-boston-dynamics-and-toyota-research-institute-takes-key-step-towards-general

Boston Dynamics and TRI show autonomous whole-body manipulation and locomotion
behaviors using Large Behavior Models. The Atlas blog states that policies map
images, proprioception, and language prompts to full-robot actions at 30 Hz.

This is a strong sign of the direction of the field. It is not open evidence of
the target task because benchmark details, held-out unknown-load evaluations,
energy/posture metrics, and video-conditioned active probing are not available.

## Design Implication

The project should start from a strong locomotion/carrying base, not from
scratch. The new contribution should be the missing layer:

```text
video prior + active load probing + morphology-aware posture optimization
```

The right claim is not "humanoids can carry boxes." That is increasingly true.
The claim should be whether a robot can use video as weak guidance, infer
unknown load dynamics through probing, and choose its own stable low-cost
carrying posture across bodies and objects.
