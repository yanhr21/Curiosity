# Curiosity

This branch resets the project to a new research direction:

```text
Video-guided, embodiment-aware active loco-manipulation
for unknown-load carrying.
```

The old dense-tactile Curiosity workspace has been moved out of the repository
to:

```text
/public/home/yanhongru/Curiosity_archive_20260702_pre_video_guided_carrying/
```

## Research Question

A walking and balancing humanoid sees a box with unknown weight, shape, center
of mass, and friction. The robot may receive a video of a human, another robot,
or a simulation carrying the object, but it must not retarget that motion. It
must actively probe the object and choose a stable, efficient posture suitable
for its own body.

## Current Conclusion

As of 2026-07-02, existing systems provide important components but do not
solve the full target. Humanoids can carry objects in constrained settings.
Video reward and cross-embodiment imitation from observation exist. The missing
piece is their combination with active unknown-load probing and
morphology-aware posture selection.

## Documents

- `IDEA/idea.md`: concise active idea and success gate.
- `docs/execution_path_2026-07-02.md`: concrete executable path, source
  inventory, platform choice, and phase gates.
- `docs/reference_clone_inventory_2026-07-02.md`: reference code/model clones
  for Digit/MuJoCo and mc_rtc loco-manipulation lineage.
- `docs/2026-07-02_research_overview.md`: main survey and conclusion.
- `docs/robot_carrying_capability_review.md`: humanoid carrying capability
  review.
- `docs/video_conditioned_rl_review.md`: video-conditioned/non-retargeting
  learning review.
- `docs/research_program_design.md`: proposed experiment and evaluation
  design.
- `PLAN/`: staged execution plans.
- `TODO/`: staged task lists.
- `src/carrying_visualization/`: browser diagnostic visualization for
  box-carrying posture adjustment.
- `experiments/reports/carrying_visualization_completion_audit_2026-07-02.md`:
  requirement-by-requirement audit for the first kinematic visualization step.
- `scripts/isaac/`: strict Isaac Lab-Arena execution scripts for the real G1
  brown-box loco-manipulation path plus the current direct carrying-task scene
  diagnostic.
- `scripts/isaac/run_official_policy_locomotion_smoke.py`: current
  Isaac-native robot-control route using NVIDIA's installed Go2/H1 locomotion
  policy examples and local official assets. It supports an optional Go2
  `PAYLOAD_MODE=fixed_base` diagnostic for walking with a rigid box fixed to
  the base link; this is not unknown-box grasping or final carrying success.
- `scripts/isaac/build_adaptive_probe_carry_scene.py`: current direct Isaac
  scaffold for active probing and morphology-dependent posture selection.
- `scripts/isaac/run_adaptive_probe_carry_sweep.sh`: current direct Isaac
  sweep runner. The 2026-07-04 sweep
  `adaptive_probe_sweep_20260704_adaptive_direct_sweep1` completed 5/5
  diagnostic scaffold cases with 0 drops and strategy diversity across
  `front_carry`, `low_front_carry`, and `chest_supported_slow`. This is not a
  dynamic robot-carrying success claim.
- `experiments/reports/isaac_arena_g1_locomanip_preflight_2026-07-02.md`:
  Isaac preflight status and current blockers.
- `experiments/reports/direct_isaac_g1_wbc_carry_progress_2026-07-04.md`:
  latest Isaac execution status, including the runnable
  `direct_carry_task_scene` diagnostic and the still-blocked G1 articulation
  path.
- `AGENTS.md`: active project rules for future agents.

## Non-Goal

This project is not human-to-robot retargeting, teleoperation replay, motion
shadowing, or table-top video-conditioned behavior cloning. Those methods may
be baselines, but they are not the target claim.

## Success Requirement

A valid success claim must beat the strongest baseline on harder held-out
object and robot settings without safety regression, and ablations must show
that both video priors and active probing causally improve carrying.
