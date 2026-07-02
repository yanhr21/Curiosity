# Latest Tactile Reference Code Recheck

Date: 2026-07-01

Scope: lightweight web/source and `git ls-remote` checks only. No simulation,
rendering, training, dependency installation, model loading, dataset conversion,
or Python-heavy validation was run on the login node.

## Tacmap

- Paper: `https://arxiv.org/abs/2602.21625`
- Web evidence:
  - arXiv/html describes Tacmap as a geometry-consistent penetration/deform map
    representation for tactile sim-to-real.
  - The page does not expose a GitHub/code link.
- Remote probes attempted:
  - `https://github.com/LeiSu-Tacmap/Tacmap.git`
  - `https://github.com/Tacmap/Tacmap.git`
  - `https://github.com/tacmap/tacmap.git`
  - `https://github.com/sulei1025/Tacmap.git`
  - `https://github.com/zzp15/Tacmap.git`
- Probe result:
  all probed repository names returned `Repository not found` or were
  unavailable within the short timeout.
- Current project status:
  code-unavailable comparison gap. Tacmap remains conceptually important for
  deform-map/penetration-depth semantics, but it cannot be used as an official
  code path until a real repository/config is found.

## ControlTac

- Project page: `https://dongyuluo.github.io/controltac/`
- Paper: `https://arxiv.org/abs/2505.20498`
- Web evidence:
  - the project page describes force- and pose-conditioned tactile image
    generation from a single reference image;
  - it reports force-control and pose-control stages, downstream force/pose
    estimation, tactile augmentation, and imitation-learning results;
  - the page does not expose a code/GitHub link.
- Remote probes attempted:
  - `https://github.com/dongyuluo/ControlTac.git`
  - `https://github.com/DongyuLuo/ControlTac.git`
  - `https://github.com/controltac/controltac.git`
  - `https://github.com/ControlTac/ControlTac.git`
- Probe result:
  all probed repository names returned `Repository not found`.
- Current project status:
  code-unavailable comparison gap. ControlTac remains a useful photometric
  tactile image generation target, but it cannot satisfy official code sanity
  without released code/checkpoints/configs.

## FreeTacMan

- Official repository: `https://github.com/OpenDriveLab/FreeTacMan`
- Local path: `external/FreeTacMan`
- Local commit: `9285740a5d33385d3a9cf5ccdb185e3387b547bd`
- Branch: `main`
- Role:
  secondary official 2026 real visuo-tactile data and tactile-pretraining
  reference.
- Relevant upstream claims/paths:
  - robot-free visuo-tactile data collection system;
  - more than 3000k visuo-tactile image pairs, more than 10k trajectories, and
    50 tasks;
  - tactile encoder pretraining under `pretrain/`;
  - ACT policy training with tactile images under `policy/act/`;
  - tactile normalization metadata in `pretrain/tactile_data.json`.
- Current project status:
  useful for future real-data representation, tactile encoder pretraining, and
  policy baseline design. It is not a simulator, not a Newton-native infant
  checkpoint, and not a Gate 00F replacement for UniVTAC/TaCauchy.

## DiffTactile

- Official repository remote checked:
  `https://github.com/Genesis-Embodied-AI/DiffTactile`
- Observed remote HEAD:
  `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
- Local path:
  `external/DiffTactile`
- Local commit:
  `c4bf43d44071758aea68a5c7ae125fc8257bb8e1`
- Role:
  secondary differentiable tactile simulator/reference for soft tactile
  physics and tactile optimization tasks.
- Relevant observed paths:
  - `difftactile/sensor_model/fem_sensor.py`
  - `difftactile/sensor_model/gripper_fem.py`
  - `difftactile/tasks/grasp_elastic.py`
  - `difftactile/tasks/object_repose.py`
  - `difftactile/tasks/surface_follow.py`
  - `difftactile/baseline/training_rl.py`
  - `difftactile/baseline/tactile_env.py`
- Observed design:
  differentiable tactile physics with FEM tactile sensor models, marker
  extraction, soft/rigid/multi-material object models, gradient-based skill
  learning tasks, and CMA-ES/PPO/SAC/RNN baselines.
- Environment note:
  `external/DiffTactile/requirements.txt` is UTF-16 little-endian text with
  CRLF line endings. Do not blindly pass it to a normal installer without an
  encoding-aware review.
- Current project status:
  cloned for source audit only. It remains a secondary comparison/reference
  path and does not replace UniVTAC/TaCauchy Gate 00F semantic validation.

## Integration Decision

Mandatory Gate 00F remains UniVTAC plus TaCauchy. FreeTacMan and DiffTactile are
secondary references. Tacmap and ControlTac remain comparison gaps until
official code becomes available.

## Reactive Diffusion Policy / ImplicitRDP / Tactile Diffusion Follow-up

Additional source-level check on 2026-07-01:

- Reactive Diffusion Policy:
  - official repository: `https://github.com/xiaoxiaoxh/reactive_diffusion_policy`
  - local path: `external/reactive_diffusion_policy`
  - local commit: `824c5e8de1fd1811106907a04b5f0186e0138c0b`
  - role: secondary serious visual-tactile policy reference with official
    data/checkpoint links, tactile marker embeddings, diffusion/RDP training
    scripts, and real visual-tactile dataset code.
- ImplicitRDP:
  - official repository: `https://github.com/Chen-Wendi/ImplicitRDP`
  - local path: `external/ImplicitRDP`
  - local commit: `4c90646df17787e31c88838106c4a0323ddefb4a`
  - role: secondary visual-force diffusion policy reference linked from RDP,
    with official data/checkpoint links and force/wrench policy configs.
- Tactile Diffusion:
  - official repository: `https://github.com/carolinahiguera/Tactile-Diffusion`
  - local path: `external/Tactile-Diffusion`
  - local commit: `16868fb96d19d93dc5837600c26b48415632e4f6`
  - role: secondary tactile photometric generation reference for DIGIT/TACTO
    style sim-to-real image generation.
- Action Conditioned Tactile Prediction:
  - official remote: `https://github.com/imanlab/action_conditioned_tactile_prediction`
  - remote HEAD: `085d2ab82d2e0574f39a359dd2c445b8f7f7a3b3`
  - local status: clone failed with `fetch-pack: unexpected disconnect while
    reading sideband packet`; record as acquisition blocker, not local source
    evidence.

Detailed audit:
`experiments/reports/phase00/ref_tactile/policy_reference_audit.md`.

These references do not change Gate 00D/00E/00F. They are future baselines or
photometric references only.
