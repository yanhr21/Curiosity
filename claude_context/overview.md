# Robot Baby — Overview & Core Story

*The long-form pitch. This mirrors the top-level `../context.md` and is what the "Motivation" and "Framework"
chapters of `index.html` are drawn from.*

## The problem: the wrong cost curve

Robot learning today buys skill at a steep price. The dominant recipes are:

- **Teleoperation → behavior cloning.** A human puppeteers the robot to produce action-labeled data, then a
  policy is fit by supervised regression. No embodiment gap, but the data costs **human hours per skill**, and
  the policy cannot improve past its demonstrations.
- **Imitation from human video via reconstruction.** Start from video (like us), then pay a heavy tax:
  estimate accurate **2D/3D hand-object trajectories**, build a **dense 3D reconstruction**, **retarget** to
  the robot, and **solve contact and forces** so the motion is physical. Fragile, compute-heavy, object-specific,
  and quality-capped by reconstruction accuracy.
- **Pure RL / reward engineering.** Hand-design a dense reward and learn from scratch. Reward design is its own
  research problem; exploration is sample-inefficient on long-horizon, contact-rich tasks.

All three pay a tax that nature never charges.

## The insight: two signals are enough

A child receives **no** labeled trajectories, and nobody solves contact forces on their behalf. Hands-on
teaching — physically moving a learner's limbs — is **rare** for humans and **almost absent** in animals.
Skill flows through two channels:

1. **Demonstration** (third-person, approximate): *watch a capable adult and imitate the intent.*
2. **Reinforcement** (interactive, corrective): *try it, feel the outcome, adjust toward a similar goal.*

If two signals suffice for a child, they suffice for a robot. The trajectories, reconstruction, and contact
solving that the expensive pipeline precomputes are either unnecessary or are exactly what the robot should
**discover by acting** in a rich-enough world.

## The bet, stated precisely

> A **vision-tactile** policy, **conditioned on a third-person demonstration video**, can reach **similar task
> success** to the demonstrator by **reinforcement** in a physically-rich simulator — without any 2D/3D
> trajectory labels, 3D reconstruction, solved contact, or teleoperation.

Corollaries we expect to demonstrate:
- RL lets the robot **improve on suboptimal demonstrations** (the demo need not be expert or on-embodiment).
- **Tactile sensing closes the contact gap** that vision-only imitation cannot see.
- The demo supplies the **reward** (goal/video similarity), so no per-task reward engineering is needed.

## Why this is an NVIDIA story

The recipe is only affordable because of the simulation substrate:

- **Newton** — open-source GPU physics on Warp + OpenUSD (NVIDIA · Google DeepMind · Disney Research), with
  DeepMind's MuJoCo-Warp giving 70× humanoid / 100× in-hand speedups — enables thousands of parallel,
  contact-rich rollouts.
- **Isaac Sim / Isaac Lab** — the multi-modal RL/IL training framework.
- **REST3D** — casual images → physically-stable, penetration-free, simulation-ready scenes, arranged like
  real human environments.

Downstream, the same synthetic data and demonstration-conditioned policies feed NVIDIA's humanoid/robot
foundation-model efforts (GR00T, Cosmos). Robot Baby is **SDG in service of a cheaper learning recipe**.

## The framework in two parts

- **Part 1 — SDG engine (Shengze):** Newton/Isaac world factory + REST3D assets + human-arranged scene sampler,
  emitting randomized vision+tactile episodes. See `architecture_sdg_engine.md`.
- **Part 2 — Demonstration-conditioned RL (Hongru & Shengze):** a video-conditioned, vision-tactile policy that
  closes the gap to the demonstrated goal by RL. See `architecture_dcrl.md`.

## Positioning vs. prior art

The name is deliberate: **demonstration-conditioned reinforcement learning** (DCRL) is an existing formulation
(Dance et al., ICML 2021) in which a policy takes demonstrations as input and is trained to maximize reward
across tasks — with the exact properties we want (improve on suboptimal demos, use state-only demos, cope with
demonstrator↔agent domain shift). Prior DCRL and few-shot imitation work is mostly **state-based, low-DoF, and
vision-only**. Robot Baby pushes the idea to **third-person video demonstrations**, **vision + tactile**
manipulation, and **human-arranged simulated worlds**. See `related_works.md`.
