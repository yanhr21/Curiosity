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
- `scripts/isaac/run_official_policy_locomotion_smoke.py`
- `scripts/isaac/run_official_policy_locomotion_smoke.sh`

Useful Arena reference files, not blockers:

- `external/IsaacLab-Arena/isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/lerobot/config/g1_locomanip_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml`
- `external/IsaacLab-Arena/isaaclab_arena_g1/`

## 2026-07-04 Active Isaac Route

Do not wait for extra external models before building the scene. The active
route is now:

1. Verify a known official Isaac locomotion controller in this cluster using
   the installed `isaacsim.robot.policy.examples` Go2/H1 policies and local
   mirrored assets.
2. Run the same controller with `PAYLOAD_MODE=fixed_base`, where a physical
   rigid box is fixed to the Go2 base link. This is a fixed-payload balance
   diagnostic, not unknown-object grasping.
3. Replace the fixed joint with a contact or constraint sequence that starts
   from a free box in the scene.
4. Add active probing actions and unknown mass/shape/COM randomization only
   after the robot-control and object-dynamics path shows nonzero measured
   motion.

The current official-policy script deliberately avoids hand-written toy gait
controllers. It uses NVIDIA's installed flat-terrain policy wrappers and local
policy checkpoints. If this route fails, record the exact Isaac/Slurm failure
and move to another Isaac-native control path; do not wait on unrelated model
downloads.

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
