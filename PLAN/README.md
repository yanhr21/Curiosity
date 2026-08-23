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

The serial three-seed repeat is complete. Training/action seed pairs are `161581/161582`,
`161583/161584`, and `161585/161586`; frozen evaluation seeds are `171581`, `171583`, and `171585`.
For unrelated minus correct, lifted-frame deltas are `+0.0350/+0.0179/-0.0058`, lifted-transport
deltas are `+0.0323/+0.0132/-0.0277`, and orbit-rate deltas are
`-0.0050/-0.0294/-0.0115 rad/s`. Thus the Kick-like direction occurs in only `1/3` seeds for
lift/transport and `0/3` for orbit. Seed161585's partial `3/4` shift does not replicate. The final
multiseed verdict is `stable_semantic_following=false`.

### Teacher-authority learnability diagnostic：已完成

The fixed-physics overfit pair resumed both seed161581 update-64 endpoints and executed exactly 64
new updates. The common official CarryBox45 teacher was annealed from `1.0` to a nonzero `0.25`
floor in both arms; task, initialization, optimizer, physics and frozen evaluation stayed matched.
Both endpoint proofs and frozen evaluations passed their execution checks.

The behavioral gate failed decisively. In 20 frozen profiles per arm, correct and unrelated both
have zero bilateral-contact fraction, zero lifted fraction and zero lifted-transport fraction;
both have zero foot-to-box contact, and `0/4` Kick-like directions are observed. Episodes terminate
after about `0.88 s`. This is behavioral collapse after reducing teacher authority, not semantic
separation, so the schedule is not repeated across seeds.

### Current branch: contact/event internal reward redesign

The official reference-corpus audit now covers 100 CarryBox and 99 KickBox motions. Binary source
contact is retained only as a reference-event proxy. Carry contact frames select a hand as the
nearest named effector in `95.46%` on average; Kick contact frames select a foot in `99.78%`.
Carry median lifted-moving fraction is `40.85%`; Kick is exactly zero. Thus the desired semantic
targets exist in the reference data.

The actual-rollout target corpus and predictor admission gate are now complete. The corpus contains
100 CarryBox and 99 KickBox source motions, each with 700 same-clock frames. Actual left/right
hand/foot contact is the exact 0.1 N threshold of named body-to-box filtered force; event duration
is reset-bounded and motion regime is episode-relative. No reference binary label is used as an
actual target. Carry median bilateral contact/longest hand event/lift are `0.3293`, `4.60 s` and
`0.490 m`; Kick median foot contact/longest foot event/lift are `0.0414`, `0.22 s` and `0.0066 m`.

The first 510-D predictor passed ordinary held-out gates but failed deployment and reward semantics.
The goal actor exposes a 121-D causal core, and the target builder independently minimized over 32
demo windows at every frame. That free alignment allowed a trajectory to keep matching an easy
static phase. Fixed Carry45/Kick21 scoring consequently preferred Carry45 even on held-out Kick
rollouts. Removing uncertainty did not fix it; directly recomputed targets isolated phase alignment
as the cause. This version is rejected evidence, not the active reward.

The corrected dataset binds every target to causal normalized episode phase. With that single
scientific change, direct validation/test targets prefer Carry45 for Carry and Kick21 for Kick. The
active V3 remains a 6-layer, 384-D serious Transformer and has `11,386,010` parameters. It reads the
past `10 x 121` deployable core, a fixed numeric selected demo and one normalized phase scalar; no
future event, task ID or motion ID is an input. Seed271303 freezes epoch 20. Validation/test MAE is
`0.1771/0.1560` against constant `0.2803/0.2566`; zero-demo is `0.2945/0.2766`, permuted-demo is
`0.2018/0.1761`, median Spearman is `0.677/0.694`, and all 12 model gates pass.

Validation-only calibration gives nominal-90% coverage of `97.13%/97.77%` on validation/test, with
`91.86%` minimum test target coverage. The fixed Carry45/Kick21 scale audit passes all ten checks:
both held-out splits prefer their matching task, and held-out Carry receives positive mean feedback
under Carry45 and negative mean feedback under Kick21. Dense feedback is
`eta * (exp(-calibrated_event_risk) - train_baseline)`, with `eta=0.2427623309` and clip
`0.1431077421`; it is not potential-difference shaping because the purpose is to change the policy
objective toward the selected demonstration.

This establishes a causal deployable reward before optimization, not policy obedience. The next
experiment is one explicitly authorized matched fixed-teacher policy comparison. Teacher,
initialization, physics, task reward, seeds, budget and evaluation remain fixed; only the selected
demo changes. Inspect updates 32 and 64, and judge only predictor-independent frozen behavior.

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
scientifically supported by this result, and an arbitrary Transformer hidden state must not be
called an official SMP latent.

Training, frozen evaluation, evidence paths and exact commands are consolidated in
[`DOCS/reproducibility.md`](../DOCS/reproducibility.md). The policy entrypoint remains
`scripts/sugar/demo_following/run_matched_state_predictor.py`; reference-event feasibility is
reproduced with `scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py`.

## Frozen historical plan

[`15_online_patch_tactile_mass_adaptation/plan.md`](15_online_patch_tactile_mass_adaptation/plan.md)
retains the exact online 54-patch tactile and sudden-mass protocol. All audited source defects were
fixed, but no valid corrected matched Z/P/PS comparison was completed. The line is frozen and is
not the current execution queue.
