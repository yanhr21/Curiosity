# Part 2 — Demonstration-Conditioned RL for Vision-Tactile Robots

*Owners: Hongru & Shengze. Status: design / scoping. Target architecture; expect revision as we experiment.*

## Purpose

A policy that **watches a third-person demonstration video** and **reaches a similar goal through
reinforcement** — using **vision + tactile** observation, in the Part-1 SDG world. No 2D/3D trajectory labels,
no 3D reconstruction of the demo, no solved contact, no teleoperation.

## The formulation

We adopt **demonstration-conditioned reinforcement learning (DCRL)**: the policy takes the demonstration as an
*input* and is trained to maximize reward across a distribution of tasks/scenes.

```
π( a_t | o_t^vision , o_t^tactile , z_demo )      z_demo = Encode(third-person demo video)
```

- **`z_demo`** — a conditioning vector/goal from the demo, ideally invariant to **viewpoint** and to the
  **human↔robot embodiment** gap.
- **Reward** `r_t = sim(rollout, demo)` — similarity to the demonstrated outcome (goal/video-based), optionally
  combined with a sparse success bonus. **No per-task hand-shaped reward.**
- **RL** optimizes `π` in the Part-1 environments, so the policy **improves on the demo** rather than merely
  copying it.

Why DCRL (Dance et al., ICML 2021) is the right frame: relative to BC or IRL few-shot imitation, it can
(a) **improve on suboptimal demonstrations**, (b) operate from **state-only / observation-only** demos, and
(c) **cope with domain shift** between demonstrator and agent — the three properties a third-person, human
demo forces on us.

## Components (and candidate building blocks)

### 1. Demo encoder → `z_demo`
Requirements: viewpoint-invariant, embodiment-tolerant, temporally aware.
Candidates to evaluate (see `related_works.md`):
- **R3M** — Ego4D-pretrained manipulation representation (time-contrastive + video-language).
- **TCN** — self-supervised, viewpoint-invariant time-contrastive embedding.
- **VIP / LIV** — goal-conditioned value representations from action-free video.
- **RoboCLIP** — video-language similarity from a single demo.
- **Vid2Robot** — end-to-end video-conditioned policy (architectural reference for the conditioning path).

### 2. Reward from demonstration
Requirements: dense enough to guide RL, robust to gaming, no per-task shaping.
Candidates:
- **VIP** zero-shot goal-conditioned value as a dense reward.
- **RoboCLIP** video/text-similarity reward (one demo is enough).
- **TCN** embedding distance to demo frames.
- **Diffusion Reward** — conditional video-diffusion likelihood/entropy.
- Robustness plan: pair the learned reward with a **sparse success** term to prevent degenerate solutions.

### 3. Policy & RL backbone
- π consumes **vision + tactile + `z_demo`**; tactile is first-class because contact is where video imitation
  is blindest.
- RL candidates: **DAPG**-style demo-augmented policy gradient, **RLPD** (online RL with offline/demo data),
  **DemoStart**-style demonstration-led **auto-curriculum** for hard-exploration, contact-rich tasks.

### 4. Embodiment-gap handling
- The demonstrator (human/other robot) ≠ the agent. Rely on RL to absorb the gap and on
  embodiment-tolerant encoders; **measure** transfer explicitly (human-video demo → robot policy).

### 5. Sim-to-real
- Domain randomization over the Part-1 SDG distribution; transfer a first skill to hardware (M4).
- Tactile sim-to-real is a known risk (tracked in `TODOs.md`).

## Evaluation plan

- **Primary:** task success on held-out scenes from a **single third-person demo**, vs.:
  - demo-free RL (does the demo help?),
  - BC-from-video (does RL help beyond copying?),
  - vision-only ablation (does tactile help?).
- **Improve-on-bad-demos:** RL success should exceed a deliberately suboptimal demonstration.
- **Generalization:** across REST3D-arranged scene variations from one demo.

## What this is *not*

- Not teleoperation + behavior cloning (no action-labeled on-robot data).
- Not video→reconstruction→retarget→track (no 2D/3D trajectory, no 3D recon, no solved contact).
- Not pure RL with a hand-designed reward (the demo supplies the objective).

## Open questions

- Reward-gaming robustness of video-similarity rewards.
- Best demo source (human hand vs. another robot vs. rendered) and the resulting embodiment gap.
- How much the tactile channel actually buys on contact-rich tasks — the headline ablation (M2).
