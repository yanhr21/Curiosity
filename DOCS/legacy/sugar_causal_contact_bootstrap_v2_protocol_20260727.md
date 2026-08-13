# SUGAR causal contact bootstrap V2 protocol

Date: 2026-07-27

## Why the old posture-adaptive runs are not behavior evidence

The posture-adaptive V1 runner restored official motion-45 robot, object, and
joint state at selected source frame 103, but it did not restore the official
Refiner `last_action` observation and initialized the four-frame direct-TacSL
history by repeating only frame 103. The official 890-D Refiner observation
contains `mdp.last_action`. In the old grid trace, the first alleged
zero-residual teacher action differs from recorded official action 103 by
L2 `12.2222967` and maximum absolute error `9.0012836`. Therefore the V1
posture-adaptive formal checkpoints and their grid fragments are withdrawn as
causal-bootstrap and behavior evidence. They remain local negative diagnostic
records and cannot complete a project gate.

## Corrected bootstrap

The V2 runner must use the unchanged audited official direct-TacSL motion-45
source and:

1. select source frame 103 / official reference frame 299 by maximum bilateral
   integrated normal force;
2. restore robot root, joint position/velocity, object root, command state, and
   process exact recorded `policy_actions_unclipped[102]`, then require both
   official Refiner `mdp.last_action` to equal that raw actor action and the
   goal policy's custom previous-applied-action view to equal
   `applied_actions_policy_units[102]` bitwise;
3. restore real spatial direct-TacSL frames 100--103, preserving
   `2 x 3 x 20 x 25` pressure and signed shear per frame;
4. advance the frozen H2R1 stress runtime causally over all four real frames so
   the latency role receives its real preceding frame;
5. require the first live frozen-Refiner actor output to match recorded
   `policy_actions_unclipped[103]` within L2 `5e-6`.

The source's actor-output and inverse-scaled physically applied action fields
are distinct provenance fields. Their full-sequence float32 round-trip maximum
absolute difference is `9.536743e-7`, so V2 audits that bounded conversion
instead of falsely requiring the two arrays to be bitwise equal.

The maximum-absolute action gate is decomposed rather than hidden behind one
number. The hash-bound previously passed one-environment live observation must
still reproduce source action 103 within `2e-6` maximum absolute error. The
newly restored 80-environment observation may differ from that live
observation by at most `2e-6`; only body/reference/object position terms may
carry the observed fresh-PhysX reconstruction drift. Its resulting live
teacher action must remain within `3e-6` maximum absolute error of source
action 103. This revision follows two explicitly failed V2 diagnostics: the
strict `2e-6` live-action gate observed `2.384e-6` and `2.623e-6`, while the
passed reference itself remains at `9.537e-7`, restored observations remain
within `1.461e-6`, and the old incomplete reset was wrong by `9.001`.

The bootstrap must not synthesize missing history, hand-write a controller, or
retroactively repair an old checkpoint.

## Two-update admission gate

The first retained-GPU execution is fixed at 80 environments, 24 steps per
environment, two updates, seed 105781, and action seed 105782. It uses the same
serious SUGAR-native residual actor, frozen official Refiner, official
TinyMDM/SMP, original ICM learner, and direct official TacSL R15 path as the
formal run.

An independent audit must reconstruct source frame selection and action
indices, bind all source hashes, and require:

- previous raw actor action reaches official `last_action` bitwise and its
  inverse-scaled physical target reaches the goal previous-applied-action view
  bitwise;
- all four causal TacSL fields reach the policy history exactly;
- the first frozen teacher action matches source action 103;
- all five H2R1 direct-TacSL roles are present and self-audited;
- `v16_tactile_slip_belief` is a policy input and `tactile_slip` remains a
  separately logged external constraint, absent from original ICM;
- the demo-conditioned trajectory predictor is loaded and frozen in the
  no-demo preflight but contributes zero reward;
- frozen model and two-update checkpoint reload checks pass.

A passing preflight is interface evidence only. It is not learned recovery,
posture selection, stable lifting, alternative strategy, visual evidence, or
completion. Formal no-demo/demo training must restart from new initialization
and later pass frozen multi-physics, cross-seed, and synchronized visual gates.
