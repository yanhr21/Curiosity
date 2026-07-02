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
- `docs/2026-07-02_research_overview.md`: main survey and conclusion.
- `docs/robot_carrying_capability_review.md`: humanoid carrying capability
  review.
- `docs/video_conditioned_rl_review.md`: video-conditioned/non-retargeting
  learning review.
- `docs/research_program_design.md`: proposed experiment and evaluation
  design.
- `AGENTS.md`: active project rules for future agents.

## Non-Goal

This project is not human-to-robot retargeting, teleoperation replay, motion
shadowing, or table-top video-conditioned behavior cloning. Those methods may
be baselines, but they are not the target claim.

## Success Requirement

A valid success claim must beat the strongest baseline on harder held-out
object and robot settings without safety regression, and ablations must show
that both video priors and active probing causally improve carrying.
