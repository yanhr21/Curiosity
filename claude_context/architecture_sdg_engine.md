# Part 1 — SDG Engine (Newton / Isaac Sim + REST3D)

*Owner: Shengze. Status: design / bring-up. This document is the target architecture; it will diverge from
reality as we build — keep it honest.*

## Purpose

Manufacture the **world** the demonstration-conditioned learner lives in: an endless stream of
**physically-enriched, human-arranged 3D scenes** with **contact-rich** interaction and **vision + tactile**
sensing. The engine exists so that reinforcement is *cheap* — thousands of parallel rollouts — and so that the
scenes look like the environments where the demonstrations were filmed.

## Design goals

- **Human-plausible arrangement.** Objects placed the way people place them (reachable, supported, cluttered),
  not random scatter — this is what makes third-person demos transferable.
- **Physical validity.** Assets are stable, penetration-free, and simulation-ready; contact behaves.
- **Massively parallel.** GPU physics, thousands of envs, high episodes/sec.
- **Multi-modal sensing.** Synchronized RGB, depth, proprioception, and **tactile** per step.
- **Randomizable.** Domain randomization over materials, lighting, poses, clutter, and physics params for
  sim-to-real.

## Pipeline

```
casual images ──► REST3D ──► physically-stable, sim-ready assets (OpenUSD)
                                     │
                                     ▼
                        human-arranged scene sampler ──► randomized scenes
                                     │
                                     ▼
                 Newton (GPU physics, Warp+OpenUSD) / Isaac Sim
                                     │
             ┌───────────────┬───────────────┬───────────────┐
             ▼               ▼               ▼               ▼
           RGB            depth          proprio          tactile
             └───────────────┴───────────────┴───────────────┘
                                     │
                                     ▼
                     episode + sensor API  ──►  Part 2 (learner)
```

## Components

### 1. Assets — REST3D
- **REST3D** ("Reconstructing Physically Stable 3D Scenes from a Single Image") builds an agentic scene-tree
  from a single RGB image (object physical states + gravity-support relationships), initializes with
  image-to-3D models, then runs scene-tree-guided alignment + **physics-constrained optimization** to remove
  floating and penetration.
- Output: **simulation-ready, physically-stable** assets/scenes we can drop straight into the physics engine.
- Why it matters: it converts *casual images of real human environments* into sim assets **without** the
  expensive manual asset/scene authoring pipeline — the asset-side analog of our cheap-signal thesis.

### 2. Scene arrangement — human-like layouts
- A procedural sampler that places REST3D assets into **plausible human layouts** (kitchen, desk, workbench),
  respecting support, reachability, and typical co-occurrence.
- Domain randomization: object identities/poses, clutter density, materials, lighting.
- Complementary layout priors to survey: RoboCasa, InternScenes, PhyScene, Infinigen (see `related_works.md`).

### 3. Physics — Newton / Isaac
- **Newton:** open-source GPU physics on **Warp + OpenUSD** (NVIDIA · Google DeepMind · Disney Research;
  Linux Foundation; Apache-2.0). DeepMind's **MuJoCo-Warp** contributes 70× humanoid / 100× in-hand speedups.
- **Isaac Sim / Isaac Lab:** the multi-modal RL/IL training framework and renderer we build the env API on.
- We need contact-accurate manipulation and stable grasps at high env counts.

### 4. Tactile sensing
- Vision-tactile is first-class, so the SDG engine must emit **touch**, not just pixels.
- Evaluate **TacSL** (GPU visuotactile sensor simulation + learning in Isaac) as the tactile channel.
- Open risk: tactile sim-to-real fidelity (tracked in `TODOs.md`).

### 5. Episode + sensor API
- Stable schema: per-step synchronized `{rgb, depth, proprio, tactile, reward_info, success}`, plus the
  scene/task metadata Part 2 needs to attach a demonstration.
- Must be fast to iterate on and decoupled from the learner.

## Interfaces to Part 2

- Provides the **environments** (reset/step, randomized), the **observation streams** (vision + tactile), and
  the **rollout throughput** that demonstration-conditioned RL consumes.
- Does **not** provide the demonstration or the reward model — those live in Part 2
  (`architecture_dcrl.md`). The engine only needs to expose enough state for a demo-similarity reward to be
  computed.

## Open questions

- Newton feature/API maturity for our contact + tactile needs on the project timeline.
- REST3D fidelity vs. throughput trade-off (scene realism vs. episodes/sec).
- How much scene realism is actually required for third-person-demo transfer — measure, don't assume.
