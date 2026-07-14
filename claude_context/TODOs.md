# Robot Baby — Roadmap & TODOs

**Status:** project bring-up (scaffolding · scoping · literature review). Nothing trained yet — this is the
plan of record. Keep this file the single source of truth for "what's next"; the rendered page opens it by
default.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` open question / decision needed.

---

## Now (bring-up)

- [x] Create project context scaffold (`context.md`, `claude_context/`, `index.html`, `serve.py`).
- [x] Seed the Related Work tab with an initial literature map.
- [~] Literature review: demonstration-conditioned RL, third-person/video imitation, video→reward, tactile RL,
  simulation/SDG. (Living in `related_works.md` + the Related Work tab.)
- [ ] Write a 1-page problem statement + success criteria to align with Hongru and stakeholders.
- [?] **Decision — target embodiment(s):** dexterous hand vs. parallel gripper vs. humanoid arm+hand. Drives
  the tactile sensor choice and the sim setup.
- [?] **Decision — first task family:** which contact-rich task is milestone M1 (e.g. pick-place, pour, insert,
  wipe, open-drawer)? Pick one where tactile clearly matters.

## Part 1 — SDG engine (Shengze)

- [ ] Stand up **Newton** + **Isaac Lab** on the cluster; run a parallel-rollout smoke test.
- [ ] Wire **tactile sensing** into the sim (evaluate TacSL / Isaac tactile) alongside RGB, depth, proprio.
- [ ] **REST3D asset path:** reconstruct a first batch of physically-stable, sim-ready assets; verify no
  float/penetration under simulation; export to OpenUSD.
- [ ] **Human-arranged scene sampler:** procedurally place assets into plausible human layouts
  (kitchen/desk/workbench) with domain randomization (materials, lighting, poses, clutter).
- [ ] **Episode + sensor API:** stable schema emitting synchronized RGB / depth / proprio / tactile per step,
  consumable by Part 2.
- [ ] Throughput target: quantify episodes/sec/GPU and scene-variety knobs.

## Part 2 — Demonstration-conditioned RL (Hongru & Shengze)

- [ ] **Demo encoder:** choose/adapt a viewpoint- & embodiment-invariant video representation for third-person
  conditioning. Shortlist: R3M, TCN, VIP/LIV features, RoboCLIP video-text. (`related_works.md`)
- [ ] **Reward from demo:** define a demo-similarity reward (goal/video-based) requiring no per-task shaping.
  Candidates: VIP zero-shot value, RoboCLIP similarity, TCN embedding distance, Diffusion Reward.
- [ ] **Demo-conditioned policy** π(a | vision, tactile, demo): pick RL backbone (candidates: DAPG-style
  demo-augmented PG, RLPD, DemoStart-style auto-curriculum).
- [ ] **Baselines to beat:** (a) demo-free RL, (b) BC-from-video, (c) vision-only ablation.
- [ ] **Improve-on-bad-demos test:** show RL exceeds a deliberately suboptimal demonstration.
- [ ] **Embodiment gap:** quantify human→robot transfer from third-person video.
- [ ] **Tactile ablation:** vision+tactile vs. vision-only on the contact-rich M1 task.
- [ ] **Sim-to-real:** domain-randomized transfer of the first skill to hardware.

## Story milestones (for the community & NVIDIA)

- [ ] **M1** — "one third-person video → sim task success" on a single contact-rich task, **no reconstruction
  anywhere** in the loop.
- [ ] **M2** — tactile ablation showing touch closes the contact gap vision-only imitation misses.
- [ ] **M3** — generalization across REST3D-arranged scenes from a single demonstration.
- [ ] **M4** — sim-to-real transfer of the demonstration-conditioned policy.

## Open questions / risks

- [?] **Reward fidelity vs. gaming:** does a video-similarity reward admit degenerate solutions? Need a
  robustness plan (e.g. combine with sparse success).
- [?] **Demo source realism:** human-hand demos vs. another-robot demos vs. rendered demos — how large is the
  embodiment gap and does RL absorb it?
- [?] **Tactile sim-to-real:** simulated touch is notoriously hard to transfer; is TacSL-grade sim enough?
- [?] **Scene realism vs. throughput:** REST3D fidelity vs. episodes/sec — find the sweet spot.
- [?] **Newton maturity:** feature/API stability for our contact + tactile needs at project timescale.

## Deferred / later

- [ ] Language-conditioned variant (demo video + text goal), leveraging LIV / VLA backbones.
- [ ] Multi-demo aggregation and demo-quality weighting.
- [ ] Connection to GR00T/Cosmos as downstream consumers of the SDG data + policies.
