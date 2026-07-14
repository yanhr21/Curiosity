# Related Work — Robot Baby

*A living map. The interactive, rankable version is the **Related Work** tab in `index.html` (star / pin /
demote, persisted in your browser). This file is the plain-text mirror — add papers here and we fold them into
the tab's `WORKS` array. Stars below are default relevance (1–5), not consensus.*

Categories: **DCRL** (demonstration-conditioned / few-shot imitation + RL) · **Human-Video** (3rd-person /
human-video imitation) · **Video-Reward** (video → reward / representation) · **Tactile** (vision-tactile /
tactile sim & RL) · **Sim-SDG** (simulation / SDG / scene & asset generation) · **Foundation** (robot
foundation / world models).

---

## DCRL — demonstration-conditioned & few-shot imitation + RL

- **★★★★★ Demonstration-Conditioned RL for Few-Shot Imitation** — Dance, Perez, Cachet (NAVER LABS), ICML 2021.
  <https://proceedings.mlr.press/v139/dance21a.html>
  *The namesake.* Policy takes demos as input, trained to maximize reward across tasks. **Key:** can improve on
  suboptimal demos, use state-only demos, and cope with demonstrator↔agent domain shift — our exact needs.
- **★★★★★ DemoStart** — Google DeepMind, 2024. <https://arxiv.org/abs/2409.06613>
  Sparse reward + a handful of sim demos → auto-curriculum on a multi-fingered hand. **Key:** 98%+ sim success,
  ~100× fewer demos, zero-shot sim-to-real via domain randomization. Template for our RL half.
- **★★★★ DAPG — Dexterous Manipulation with Deep RL and Demonstrations** — Rajeswaran et al., RSS 2018.
  <https://arxiv.org/abs/1709.10087> A few demos slash dexterous-RL sample complexity. Demo+RL > either alone.
- **★★★★ One-Shot Imitation Learning** — Duan et al., NeurIPS 2017. <https://arxiv.org/abs/1703.07326>
  Single demo (attention over it) conditions a policy to a new task instance. The conditioning primitive we
  scale to video.
- **★★★ RLPD — Efficient Online RL with Offline Data** — Ball et al., ICML 2023.
  <https://arxiv.org/abs/2302.02948> Symmetric demo/online sampling; robust way to fold demos into online RL.

## Human-Video — third-person / human-video imitation

- **★★★★ Third-Person Imitation Learning** — Stadie, Abbeel, Sutskever, ICLR 2017.
  <https://arxiv.org/abs/1703.01703> Domain-adversarial features → imitate from 3rd-person obs w/o actions.
  Foundational to our premise.
- **★★★★ Time-Contrastive Networks (TCN)** — Sermanet et al., ICRA 2018. <https://arxiv.org/abs/1704.06888>
  Self-supervised viewpoint-invariant embedding from multi-view video → imitation reward. Demo-encoder + reward
  candidate.
- **★★★★ DexMV** — Qin et al., ECCV 2022. <https://arxiv.org/abs/2108.05877> Human hand video → 3D pose →
  retarget → demo-augmented RL. Clean exemplar of the **expensive recon path we avoid** — our baseline/foil.
- **★★★★ MimicPlay** — Wang et al., CoRL 2023 (best paper). <https://arxiv.org/abs/2302.12422> Human play video
  → latent plan → low-level policy. Unstructured human video supplies a plan without teleop/recon.
- **★★★ VideoDex** — Shaw, Bahl, Pathak, CoRL 2022. <https://arxiv.org/abs/2212.04498> Internet human-video
  priors (hand pose, gaze) → robot hand.
- **★★★★ Vid2Robot** — Jain et al., 2024. <https://arxiv.org/abs/2403.12943> Cross-attention policy prompted by
  a human demo video. **Closest architectural analog** to our demo-conditioning — but BC-only, no RL, no tactile.

## Video-Reward — video → reward / representation

- **★★★★ R3M** — Nair et al., CoRL 2022. <https://arxiv.org/abs/2203.12601> Ego4D-pretrained manipulation
  representation (time-contrastive + video-language). Candidate encoder for demo + observation.
- **★★★★ VIP — Value-Implicit Pre-training** — Ma et al., ICLR 2023. <https://arxiv.org/abs/2210.00030>
  Goal-conditioned value from action-free video → **zero-shot dense reward**. Leading reward candidate.
- **★★★ LIV — Language-Image Value** — Ma et al., ICML 2023. <https://arxiv.org/abs/2306.00958> VIP + language
  grounding; demo by video and/or text.
- **★★★★ RoboCLIP** — Sontakke et al., NeurIPS 2023. <https://arxiv.org/abs/2310.07899> VLM video-text
  similarity → reward from **one** demo, no fine-tuning. Strong simple baseline.
- **★★★ Diffusion Reward** — Huang et al., ECCV 2024. <https://arxiv.org/abs/2312.14134> Conditional
  video-diffusion likelihood/entropy as reward for expert-like behavior.

## Tactile — vision-tactile / tactile sim & RL

- **★★★★★ TacSL** — Akinola et al. (NVIDIA), 2024. <https://arxiv.org/abs/2408.06506> GPU visuotactile sensor
  simulation + tactile RL/sim-to-real in Isaac. **The touch channel for our SDG engine.**
- **★★★★ T-Dex — Dexterity from Touch** — Guzey et al., CoRL 2023. <https://arxiv.org/abs/2303.12076>
  Self-supervised tactile pretraining + NN policies. Quantifies where touch beats vision.
- **★★★★ General In-Hand Rotation with Vision and Touch** — Qi et al., CoRL 2023.
  <https://arxiv.org/abs/2309.09979> Multi-fingered in-hand rotation from vision+touch, sim→real. The regime we
  target.
- **★★★ Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids** — Lin et al., 2025.
  <https://arxiv.org/abs/2502.20396> Vision dexterous sim→real at humanoid scale.

## Sim-SDG — simulation / SDG / scene & asset generation

- **★★★★★ Newton** — NVIDIA · Google DeepMind · Disney Research, 2025. <https://developer.nvidia.com/newton-physics>
  GPU physics on Warp + OpenUSD; MuJoCo-Warp 70× humanoid / 100× in-hand. **Part-1 substrate.**
- **★★★★ Isaac Lab** — NVIDIA, 2025. <https://arxiv.org/abs/2511.04831> GPU RL/IL framework on Isaac Sim.
  Part-2 training harness.
- **★★★★★ REST3D** — 2026. <https://shirleymaxx.github.io/REST3D/> Single image → physically-stable,
  penetration-free, **simulation-ready** scene via agentic scene-tree + physics-constrained optimization.
  **Named asset source for Part 1.**
- **★★★★ RoboCasa** — Nasiriany et al., RSS 2024. <https://arxiv.org/abs/2406.02523> 120+ kitchen scenes,
  generative assets, task/scene SDG. Human-environment SDG reference.
- **★★★ InternScenes** — InternRobotics, NeurIPS 2025. <https://arxiv.org/abs/2509.10813> 40k+ simulatable
  scenes, realistic human layouts. Layout prior.
- **★★★ PhyScene** — Yang et al., CVPR 2024. <https://arxiv.org/abs/2404.09465> Physically-interactable,
  reachable scene synthesis via diffusion.
- **★★★ Infinigen** — Raistrick et al., CVPR 2023. <https://arxiv.org/abs/2306.09310> Procedural photorealistic
  worlds; unlimited asset/scene variety + domain randomization.

## Foundation — robot foundation & world models (context / downstream)

- **★★★★ GR00T N1** — NVIDIA, 2025. <https://arxiv.org/abs/2503.14734> Open humanoid VLA foundation model,
  trained partly on sim/neural-generated data. Downstream consumer of our SDG data.
- **★★★ Cosmos** — NVIDIA, 2025. <https://arxiv.org/abs/2501.03575> Video world-foundation models + data
  pipeline for physical AI. Demo generation / augmentation / eval.
- **★★★ RT-2** — Google DeepMind, 2023. <https://arxiv.org/abs/2307.15818> VLA transferring web knowledge to
  control. Scaled BC frontier; no online RL — contrast point.
- **★★★ π₀** — Physical Intelligence, 2024. <https://arxiv.org/abs/2410.24164> Flow-matching VLA over large
  teleop corpora. Quantifies the teleop-hours cost curve we sidestep.

---

## To triage / add later

*(Drop candidates here; we'll rank and fold them into the tab.)*

- AVID (pixel-level human→robot video translation), Concept2Robot, DVD (Domain-agnostic Video Discriminator),
  Rank2Reward, XSkill, MimicGen (data amplification), DexMimicGen, RialTo (real-to-sim), Eureka (LLM reward
  design — as a *contrast* to demo-derived reward), Holodeck / ProcTHOR / Architect (scene generation),
  GenSim, RoboGen, Genie/world-model demo synthesis.
