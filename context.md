# Robot Baby — Human-Like Learning for Vision-Tactile Robots

**One line.** Teach a robot the way a child learns: **watch a third-person demonstration video, then reach a
similar goal through reinforcement** — with no teleoperation, no hand-labeled 2D/3D trajectories, no per-object
3D reconstruction, and no physically-solved contact. Two signals only: **demonstration** and **reinforcement**.

> The rendered version of this document (with the paradigm map, comparison tables, roadmap, and an interactive
> **Related Work** tab) lives at `claude_context/index.html`. Serve it with `python3 claude_context/serve.py`
> and open `http://localhost:8090/`.

---

## The core story (what we pitch to the research community and to NVIDIA)

Today's robot-learning data-curation and training pipeline is **too expensive**. To make a robot learn a
manipulation skill from human video, the field first converts the video into something a controller can
consume:

- **accurate 2D and 3D trajectories** of hands and objects,
- often a **dense 3D reconstruction** of the scene,
- and a **physically-correct pass** for contact, forces, and stability so the retargeted motion is executable.

Every one of these stages is fragile, compute-heavy, and object-specific, and the final policy quality is
capped by the weakest link in the reconstruction chain. Teleoperation avoids reconstruction but replaces it
with an even scarcer resource — **human hands-on time**, one skill at a time.

**Humans and animals do none of this.** Hands-on teaching (physically guiding a learner's limbs) is *rare*
for people and *almost absent* in animals. Skill instead comes from two signals:

1. **Demonstration** — watch a capable adult and imitate the *intent*. The demonstration is **third-person**
   and approximate; it defines *what* success looks like, not *how* to move.
2. **Reinforcement** — try it, feel the outcome, and adjust toward a *similar goal*. The correction comes from
   **acting in the world**, not from a pre-solved plan.

**Robot Baby builds exactly this:** a robot that **imitates a third-person demonstration video** and reaches
**similar task success through RL** in a physically-rich simulator — for **vision-tactile** manipulation.
Everything the expensive pipeline computes (trajectories, reconstruction, contact) is either unnecessary or is
something the robot should **discover through its own interaction** in a rich enough world.

**Why now / why NVIDIA.** The affordability of this bet rests on NVIDIA's simulation stack: **Newton**
(GPU physics on Warp + OpenUSD, with DeepMind's MuJoCo-Warp) and **Isaac Sim / Isaac Lab** make
massively-parallel, contact-rich rollouts cheap, and **REST3D**-style reconstruction turns casual images into
simulation-ready, physically-stable assets arranged like real human environments. This is synthetic data
generation in service of a fundamentally cheaper learning recipe — and it feeds directly into NVIDIA's
humanoid/robot foundation-model efforts (e.g. GR00T, Cosmos).

---

## Two parts, one framework

### Part 1 — SDG engine (owner: Shengze)
A **Newton / Isaac Sim**-based synthetic-data-generation engine that manufactures the world the learner lives in:

- **Assets:** REST3D-style single-image → **physically-stable, penetration-free, simulation-ready** 3D assets.
- **Scenes:** those assets **arranged like real human environments** (kitchens, desks, workbenches) via a
  procedural, randomized layout sampler.
- **Physics:** Newton + Isaac Lab for massively-parallel, contact-accurate rollouts — the playground where
  reinforcement actually happens.
- **Output:** unlimited randomized episodes carrying synchronized **RGB, depth, proprioception, and tactile**
  streams for Part 2.

See `claude_context/architecture_sdg_engine.md`.

### Part 2 — Demonstration-conditioned RL (owners: Hongru & Shengze)
A **vision-tactile policy conditioned on a third-person demonstration video**:

- **Conditioning:** encode the demo into a viewpoint- and embodiment-invariant goal/intent the policy consumes
  at every step.
- **Learning:** RL in Part-1's world **closes the gap** to the demonstrated goal — improving on *suboptimal*
  demos and absorbing the *human→robot embodiment gap*.
- **Contact is learned, not solved** — discovered through interaction and sensed by **touch**.
- **Reward = similarity to the demonstrated outcome** (video/goal-based), not a bespoke per-task reward.

See `claude_context/architecture_dcrl.md`. The nearest prior art is literally named **demonstration-conditioned
reinforcement learning** (Dance et al., ICML 2021); we scale the idea to third-person *video* demos,
*vision+tactile* sensing, and human-arranged scenes.

---

## What we deliberately drop vs. keep

| Drop — the curation tax | Keep — the two cheap signals |
|---|---|
| Accurate 2D/3D hand-object trajectories | **Demonstration:** an approximate third-person video of the goal |
| Per-object dense 3D reconstruction of the demo | **Reinforcement:** act, observe vision+tactile outcome, adjust |
| Physically-correct contact/force solving of the demo | **Contact learned by touch**, not solved offline |
| Teleoperated, action-labeled data collection | **Reward = demo similarity**, not hand-shaped |
| Per-task hand-engineered dense reward | |

---

## Status & where to look

- **Status:** project bring-up — scaffolding, scoping, and literature review.
- **Roadmap / task list:** `claude_context/TODOs.md` (also the default tab in the rendered page).
- **Architecture:** `claude_context/architecture_sdg_engine.md` (Part 1),
  `claude_context/architecture_dcrl.md` (Part 2).
- **Literature:** `claude_context/related_works.md` — and the interactive, rankable **Related Work** tab in
  `claude_context/index.html`.
- **Infra quick-reference:** `claude_context/experiment_context.md`.

**People.** Shengze Wang (SDG engine + co-lead on the learner), Hongru (co-lead on the learner).
