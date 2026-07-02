# Curiosity Active-Tactile Reference Audit

Date: 2026-07-01

Scope: lightweight web/source and git audit only. No simulation, rendering,
training, dependency installation, model loading, dataset conversion, or
Python-heavy validation was run on the login node.

## APPLE

- Official repository: `https://github.com/TimSchneider42/apple`
- Local path: `external/APPLE`
- Local commit: `4b1d71fadb786d865d4ee29a184ab408b9605083`
- Project page: `https://timschneider42.github.io/apple/`
- Paper status from repository/project page: ICLR 2026 active perception via
  reinforcement learning.

Relevant observed code paths:

- `external/APPLE/config/`
- `external/APPLE/python/algorithm/sac.py`
- `external/APPLE/python/algorithm/ppo.py`
- `external/APPLE/python/models/vision_transformer.py`
- `external/APPLE/python/models/multi_modal_sequence_encoder.py`
- `external/APPLE/python/envs/ham_tactile_classification.py`
- `external/APPLE/run_experiment.bash`

Observed design:

- APPLE trains active perception policies with RL algorithms such as SAC,
  CrossQ, PPO, and random-action baselines.
- Vision-based tactile configurations use a ViT image encoder for environments
  such as TactileMNIST, Toolbox, and TactileMNISTVolume.
- The code includes sequence/memory modules and supports action observation in
  memory models for some configurations.
- The documented TactileMNIST example uses:
  `./run_experiment.bash vit/sac=sac tactile_mnist:TactileMNIST-v0 tactile_mnist:TactileMNIST-test-v0`.

Phase 00 role:

- Secondary curiosity/active-probing reference.
- Useful for the later closed-loop design of sequential tactile exploration,
  exploration baselines, action-conditioned memory, and tactile-only active
  perception evaluation.
- Not a Newton-native grasping infant checkpoint.
- Not a tactile semantic-validation replacement for UniVTAC/TaCauchy.
- Not evidence that current Curiosity training has succeeded.

## Tactile MNIST Benchmark Suite

- Official repository: `https://github.com/TimSchneider42/tactile-mnist`
- Local path: `external/tactile-mnist`
- Local commit: `9e4e59139e9349ab361a3b9297f4815724ad6387`
- Project page: `https://timschneider42.github.io/tactile-mnist/`
- Latest release observed from repository web metadata: `v0.12.0` on
  2026-05-11.

Relevant observed code/docs:

- `external/tactile-mnist/docs/TactilePerceptionEnv.md`
- `external/tactile-mnist/docs/TactilePerceptionConfig.md`
- `external/tactile-mnist/docs/datasets.md`
- `external/tactile-mnist/tactile_mnist/tactile_perception_vector_env.py`
- `external/tactile-mnist/tactile_mnist/tactile_renderer/`
- `external/tactile-mnist/tactile_mnist/resources/gelsight_mini.obj`
- `external/tactile-mnist/tactile_mnist/resources/cycle_gan_tactile_mnist_v0.pth`

Observed environment/data design:

- The active tactile environments use one simulated GelSight Mini over a
  platform with hidden objects.
- The agent controls `sensor_target_pos_rel` and, when enabled,
  `sensor_target_rot_rel`.
- The observation includes tactile image `sensor_img`, sensor pose, optional
  orientation, and time step; no visual scene input is provided in those
  benchmark environments.
- Implemented tasks include TactileMNIST, Starstruck, Toolbox,
  ABCCenterOfMass, TactileMNISTVolume, and ABCVolume.
- Static datasets include 3D mesh datasets and real/synthetic touch datasets.
  Real touch data includes GelSight image sequences and single tactile images
  with pose/timing metadata.

Phase 00 role:

- Reference for designing tactile-only and tactile-masked active probing after
  the Newton/Taccel base environment passes.
- Reference for tactile image representation and task splits, especially
  train/test/holdout separation.
- Useful for future optional sanity/example runs only after an approved
  prebuilt environment exists.
- Not a grasp/lift/hold base controller.
- Not a direct replacement for the user's reference-video tactile mechanics
  requirement, because it is an active perception benchmark, not the current
  Newton grasping environment.

## Integration Decision

APPLE plus Tactile MNIST should shape Gate 00G only:

1. Use APPLE as the active tactile exploration/RL reference when designing the
   closed-loop curiosity policy.
2. Use Tactile MNIST as a reference for tactile-only observation, sequential
   touch actions, holdout splits, and tactile image benchmark baselines.
3. Keep Gate 00D/00E/00F unchanged: Newton/Taccel base evidence and
   UniVTAC/TaCauchy semantic validation remain mandatory before curiosity
   training restarts.
4. Do not claim an APPLE/Tactile MNIST result until their official examples run
   inside an approved Curiosity compute allocation with a prebuilt environment.

## Design Implications For Future Curiosity

- The future curiosity loop should be closed-loop active probing, not offline
  scalar-contact reweighting.
- The policy should choose probing/grasp adjustment actions that reduce
  uncertainty over dense tactile/mechanics predictions.
- Intrinsic reward should be bounded learning progress over dense tactile
  fields and mechanics, not raw prediction error alone.
- Baselines must include no-curiosity policy, scripted probing, random probing,
  no-tactile or vision-only policy, tactile-only masked-vision policy, and
  noisy/mismatched tactile ablations.
- Tactile-mask training should be a first-class protocol: train/evaluate
  vision+tactile, tactile-only masked vision, vision-only, and corrupted
  tactile settings.

## Current Blockers

- Gate 00D/00E/00F are still open, so no APPLE-style curiosity training should
  start.
- UniVTAC and TaCauchy official sanity remain blocked by missing approved
  prebuilt environments.
- APPLE/Tactile MNIST environments are not installed or sanity-checked in a
  Curiosity allocation; they are source references only for now.
