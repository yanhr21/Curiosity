# Plan 02: Base Simulated Box Loco-Manipulation Environment

## Purpose

Build the base simulated box-carrying environment directly in Isaac before
adding unknown-load probing or video priors.

The official Isaac Lab-Arena G1 loco-manipulation task remains a useful
reference and possible baseline, but it is no longer the first blocking target.
The first target is a controlled scene that we own and can extend:

```text
floor + carry box + target marker + configurable mass/shape/pose
+ later robot embodiment + probing/carry objectives
```

Current local entry points:

- `scripts/isaac/build_minimal_carry_scene.py`
- `scripts/isaac/run_minimal_carry_scene.sh`

Useful Arena reference files, not blockers:

- `external/IsaacLab-Arena/isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/lerobot/config/g1_locomanip_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_g1/`

## What Counts

This phase counts only as reproduction/base environment setup. It does not
claim unknown-load carrying, active probing, or video-guided learning.

Required evidence:

- exact command;
- environment version;
- commit;
- output directory;
- log;
- one MP4 rollout or documented reason video is unavailable;
- scene/object description;
- whether the box mass/shape/load was known or randomized.

## Exit Criteria

- A base Isaac box scene runs in compute allocation with verified rigid-body
  gravity/collision.
- A robot embodiment can be inserted without breaking the box scene.
- It records object pose, robot state, contact/drop/fall status if available.
- A short report states which parts are inherited and which parts are missing
  for the target research problem.
