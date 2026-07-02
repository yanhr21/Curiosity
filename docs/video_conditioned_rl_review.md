# Video-Conditioned And Non-Retargeting Learning Review

## Question

Can video be used as a reference for RL without retargeting and without relying
on detailed tactile annotations?

## Short Answer

Yes, but not yet for the full unknown-load humanoid carrying problem. Existing
methods provide useful machinery for visual reward learning, task-progress
embedding, graph abstraction, and video-conditioned policies. Most evidence is
still table-top manipulation, small simulated tasks, supervised policy
learning, or retargeting/teleoperation-adjacent imitation.

## XIRL

Sources:

- https://arxiv.org/abs/2106.03911
- https://x-irl.github.io/

XIRL learns visual reward functions from cross-embodiment demonstration videos.
It explicitly targets settings where agents differ in shape, action space, and
end-effector dynamics. It learns a task-progress embedding and uses distance
to the goal in embedding space as reward for RL.

Why it matters:

- It is genuinely relevant to non-retargeting video guidance.
- It does not require action labels.
- It is designed for embodiment mismatch.
- It is closer to "video as reward/progress prior" than "copy trajectory."

Limitations for carrying:

- Demonstrated tasks are much smaller than whole-body unknown-load carrying.
- It does not infer hidden physical properties such as mass, friction, or
  center of mass.
- It does not solve balance, torque, long-horizon carry, or posture cost.

Use:

- Candidate reward-learning backbone.
- Strong baseline for cross-embodiment video reward.

## GraphIRL

Sources:

- https://arxiv.org/abs/2207.14299
- https://sateeshkumar21.github.io/GraphIRL/

GraphIRL learns visually invariant rewards from diverse third-person videos
using object-centric graph abstraction and temporal matching. It is useful
because carrying can be described partly through object and body-contact
relations rather than pixel appearance.

Why it matters:

- It aims to scale IRL from diverse videos.
- It abstracts away irrelevant texture and appearance.
- It can use human demonstration videos for real-robot manipulation.

Limitations for carrying:

- Evidence is still small manipulation and X-MAGICAL-style tasks.
- Whole-body contact, balance, and load uncertainty are outside its core
  demonstrated scope.
- It does not directly produce active probing behaviors.

Use:

- Object-centric reward abstraction for reference videos.
- Good source for "video should encode relations and progress, not joints."

## VIP

Sources:

- https://arxiv.org/abs/2210.00030
- https://sites.google.com/view/vip-rl

VIP learns visual rewards and representations from large unlabeled human video
datasets. It casts representation learning as an offline goal-conditioned RL
problem without actions and can provide dense visual rewards for downstream
robot tasks.

Why it matters:

- It can use large-scale human video without tactile labels.
- It is action-free and therefore not retargeting.
- It can provide dense reward for unseen tasks.

Limitations for carrying:

- The reward is visual; hidden dynamics remain unobserved.
- It typically uses goal-image style supervision rather than a full reference
  video policy for humanoid carrying.
- It does not solve active load identification.

Use:

- Pretrained visual representation or progress reward.
- Baseline for video-only visual reward.

## Vid2Robot

Sources:

- https://arxiv.org/html/2403.12943
- https://vid2robot.github.io/

Vid2Robot is an end-to-end video-conditioned robot policy. It takes a human
demonstration video and the robot's current observation and outputs robot
actions. The project reports real-robot evaluation and performance gains over
other video-conditioned policies.

Why it matters:

- It directly tests video as task condition.
- It does not simply use text instructions.
- It shows that human video can help specify robot tasks.

Limitations for carrying:

- It is mainly supervised policy learning/behavior cloning, not RL with
  active probing.
- It targets manipulation tasks, not whole-body humanoid carrying.
- It does not solve unknown mass, friction, center of mass, or energy-optimal
  posture selection.
- It requires paired video/robot trajectory style data for its main training
  setup.

Use:

- Strong baseline for video-conditioned policy learning.
- Not the main target if the project requires active RL and no retargeting.

## Retargeting And Teleoperation-Adjacent Methods

Many humanoid-video papers are useful but must be labeled carefully. If a
method's core contribution is human-to-humanoid retargeting, teleoperation,
shadowing, or object-aware motion transfer, it should not be presented as
solving this project.

Allowed uses:

- baseline;
- data collection;
- initialization source;
- negative comparison showing why direct copying is insufficient.

Forbidden claim:

```text
Retargeted human carrying motion = robot has learned its own efficient
carrying posture.
```

## Required Video Ablations

A credible experiment should include:

- no-video policy;
- correct video;
- wrong-task video;
- same-task different embodiment video;
- same-task human video;
- same-task robot video;
- simulation video;
- video reward only, no probing;
- probing only, no video;
- retargeting baseline;
- behavior-cloning baseline.

## Key Design Rule

The video should be a prior over "what task is happening" and "how progress
looks." The robot's own interaction data must decide "how this body should
carry this object."
