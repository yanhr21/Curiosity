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

The direct predictor-independent behavior audit is complete. It uses only robot/object state and
rigid hand-contact force from the existing traces; predictor loss, demo reward and training loss are
excluded. CarryBox45 lifts the box by `0.7639 m` and completes `81.41%` of its horizontal box path
while lifted, whereas KickBox21 never crosses the `0.05 m` lift threshold and performs all box
motion at ground level.

Under the common CarryBox45 teacher, both learned arms remain Carry-like. Compared with the correct
arm, the KickBox21-reward arm has `+0.0350` lifted-frame fraction, `+0.0323` lifted-transport
fraction, `-0.0323` ground-transport fraction and `-0.0050 rad/s` orbit rate. Thus none of the four
declared semantic directions is observed. The two transport fractions are complementary views, not
independent tests. The reward changes the Carry solution, but semantic demo following is not
established.

The next experiment is a serial three-seed repeat of the same 64-update design. Training/action
seed pairs are `161581/161582` (existing), `161583/161584`, and `161585/161586`; frozen evaluation
seeds are `171581`, `171583`, and `171585`. Every seed must use the same teacher, initialization,
physics profiles, update budget and reward weights across the two arms. Do not launch it without
explicit approval. Only after the seed-level behavior result is stable should teacher authority be
changed identically in both arms.

### Expected behavior, not just reward score

For the correct Carry demo, the expected interaction is: approach the box, establish bilateral hand
contact, lift above `0.05 m`, transport it predominantly while lifted, and keep the robot/object
geometry coupled. For the unrelated Kick demo, genuine conditioning must instead cause an
observable shift toward ground-level object motion, more motion around the box, and a different
body/object contact mode, even though the external task still asks for CarryBox completion. Task
success and demo adherence are reported separately.

The exact source timelines make this concrete. CarryBox45 has one continuous binary hand-contact
proxy interval from frame `245` to `541` (`4.90--10.82 s`), crosses the 5 cm lift threshold at frame
`286` (`5.72 s`), peaks at frame `350` (`7.00 s`) and remains lifted through frame `508`
(`10.16 s`). KickBox21 has 14 intermittent binary foot-contact proxy intervals, its closest named
end-effector is the right ankle at frame `190` (`3.80 s`), and it never crosses 5 cm. These binary
labels describe reference contact roles only; they are not tactile force. The learned unrelated
arm must measurably move toward this event structure, not merely change predictor score.

This standard follows the behavior-level structure used in physics-based interaction imitation:
[DeepMimic](https://arxiv.org/abs/1804.02717) separates imitation from the task objective;
[PhysHOI](https://arxiv.org/abs/2312.04393) explicitly evaluates body-object contact topology;
[InterMimic](https://arxiv.org/abs/2502.20390) checks object deviation, joint-object relations and
required-contact duration; and [CHORD](https://nvidia-isaac.github.io/video_to_data/chord/) argues
that object-centric contact wrench should measure how contact moves the object. Therefore policy
loss, predicted reward or task success alone is not a demo-following verdict.

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
