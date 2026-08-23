# Plan index

## Current priority: demo following

The first causal selected-demo experiment is complete:

1. CarryBox45 teacher-only, zero-residual evaluation passes bilateral contact and 5 cm lift in all
   20 profiles;
2. both learned arms use the same fixed CarryBox45 teacher, seeds, initialization, physics,
   optimizer, reward weights and 64-update budget;
3. only the selected reward demo differs: CarryBox45 versus unrelated KickBox21;
4. frozen outcomes are `16/20` versus `18/20` success, with two physical root-height falls per arm;
5. selected-demo reward changes the checkpoint and rollout, but correct-demo superiority and
   semantic obedience are not established.

Do not extend the existing seed automatically. The next valid experiment requires a direct,
predictor-independent behavior-level adherence metric and at least three matched training seeds.
Only after that should teacher authority be changed identically in both arms.

The official selected-demo TinyMDM gate is also complete. Exact selected-clip identity passes, but
the independent CarryBox96/KickBox22 semantic extension fails. Policy integration is not
authorized, and an arbitrary Transformer hidden state must not be called an official SMP latent.

Training, frozen evaluation, evidence paths and exact commands are consolidated in
[`DOCS/reproducibility.md`](../DOCS/reproducibility.md). The active entrypoint is
`scripts/sugar/demo_following/run_matched_state_predictor.py`; its default design is the current
same-teacher 64-update experiment.

## Frozen historical plan

[`15_online_patch_tactile_mass_adaptation/plan.md`](15_online_patch_tactile_mass_adaptation/plan.md)
retains the exact online 54-patch tactile and sudden-mass protocol. All audited source defects were
fixed, but no valid corrected matched Z/P/PS comparison was completed. The line is frozen and is
not the current execution queue.
