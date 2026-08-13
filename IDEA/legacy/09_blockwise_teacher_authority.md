# Blockwise Official-Teacher Authority for Post-Failure Carrying

## Research Question

Can a SUGAR residual policy explore new arm/contact strategies after a
direct-TacSL-confirmed nominal clamp failure without simultaneously losing the
official Refiner's leg and waist support?

The current residual adapter uses one scalar teacher coefficient for all 29
actions. When tactile failure closes, the same four-step schedule releases
hips, knees, ankles, waist, shoulders, elbows, and wrists together. The
post-drop exposure experiment increased zero-teacher training actions by
`41.9%`, but fresh no-grace evaluation still failed original-ICM discovery
and the rendered robot fell after unilateral contact. This idea tests whether
that failure is caused by coupling manipulation freedom to whole-body support
loss.

This is a focused child of
`IDEA/06_sugar_smp_tactile_strategy_exploration.md` and consumes the negative
result in
`DOCS/sugar_official_refiner_residual_postfailure_exposure_result_20260725.md`.

## Core Mechanism

Resolve the official SUGAR 29-D action term to named joints at runtime:

- support block: every hip, knee, ankle, and waist joint;
- manipulation block: every left/right shoulder, elbow, and wrist joint.

Before tactile-confirmed failure, both blocks use coefficient one. After the
existing causal four-step release:

```text
support coefficient       = 1
manipulation coefficient  -> 0

a_executed[j] =
    alpha_teacher[j] * a_official_teacher[j]
    + 0.05 * a_residual[j]
```

PPO still samples and log-scores the same 29-D residual. Original ICM still
reads the actual 29-D applied action and rewards not-yet-predicted
action-conditioned transitions. No task outcome, failure flag, teacher
coefficient, or reward is added to ICM.

## Why This Is Not a Shortcut

The official Refiner remains a nominal support prior, not the final policy.
The arm joints must still discover a recontact/topology change through the
causal goal/direct-TacSL policy. The comparison holds the official checkpoint,
physics, state source, action scale, policy architecture, SMP, ICM, reward
weights, optimizer, stress roles, and update budget fixed.

This first blockwise control cannot demonstrate a lower-squat strategy because
the leg/waist teacher remains anchored. It can only establish whether stable
support enables asymmetric arm recontact, bottom/side support, or another arm
contact topology. A later lower-body release phase is allowed only after a
repeatable direct-TacSL arm-recovery boundary exists.

## Falsifiable Predictions

Compared with scalar whole-body release under common sources and evaluation:

1. post-failure reset/fall frequency should decrease;
2. zero-manipulation-teacher actions should increase without weakening safety;
3. later direct-TacSL recontact or topology-change candidates should increase;
4. the learned residual mean should become less dominated by its Gaussian
   sample; and
5. frozen original-ICM forward error and support coverage may increase even if
   the task still fails.

Items 1--4 do not define curiosity. Increased discovery is admitted only by
the exact frozen-ICM attribution gate. Recovery still requires same-episode
direct-TacSL recontact plus physical lift/carry evidence.

## Claim Boundary

- High-fidelity simulated tactile only; no physical GelSight calibration or
  sim-to-real claim.
- Arm-only release is an action-prior intervention, not ICM, SMP, imitation
  reward, or success shaping.
- A stable upright rollout is not recovery.
- A new arm pose without taxel-resolved pressure/shear contact is not a
  strategy.
- A lower-squat claim is forbidden while the support block stays anchored.
- Recovery/alternative-strategy counts remain `0/0` until the existing hard
  gates pass.

## First Executed Result

The complete fixed comparison is recorded in
`DOCS/sugar_blockwise_teacher_authority_result_20260725.md`.

The named partition and exact per-joint routing pass. Arm-zero-teacher exposure
increases from `979/30,720` (`3.19%`) in the scalar parent to
`1,996/30,720` (`6.50%`) while all 15 support columns remain exactly one.
However, fresh update-64 evaluation changes reset pairs from `7` to `8`,
bilateral-contact frames remain `23`, and paired mean final box height changes
by `-0.02083 m`. Frozen original-ICM forward-error CI crosses zero and
frozen-feature coverage decreases, so discovery, recovery, and alternative
counts remain `0/0`.

This rejects the strong version of the idea: retaining an advancing official
support teacher is not sufficient. The next bounded hypothesis is that support
authority must be **failure-latched** rather than continuing to advance along a
nominal reference that no longer matches the post-failure box/contact state.

## Failure-Latched Successor Diagnostic

The earlier successor diagnostic is recorded in
`DOCS/sugar_failure_latched_support_hold_result_20260725.md`.

The no-learning routing gate and fixed 64-update endpoint both pass exact
audits. Across 20 fresh common-random streams, however, update-64 final height
changes by `-0.02360 m`, reset pairs worsen `6 -> 7`, and bilateral-contact
frames stay `23 -> 23`. Frozen original-ICM feature coverage increases, but
the paired forward-error 95% interval `[-0.02438, 0.04016]` crosses zero.
Rendering shows unilateral contact followed by reset and no regrasp.

This diagnostic does not reject or complete the hypothesis. Its implementation
and experiment must be re-audited and rerun before it can support a conclusion.

## Supported Post-Drop Successor Diagnostic

The withdrawn first-attempt record is retained in
`DOCS/sugar_supported_postdrop_exposure_result_20260725.md`.

Its producer and independent audit pass exact 15-D support/14-D arm routing,
raw/effective drop accounting, non-drop termination preservation, direct
TacSL provenance, original-ICM causality, and checkpoint/source identities.
Arm-zero-teacher exposure rises to `3,383/30,720` (`11.01%`).

The first attempt reported a negative outcome. Fresh no-grace reset pairs
worsen `7 -> 9`, paired final box height changes by `-0.04458 m`, and bilateral
frames stay `23 -> 23`. Frozen original-ICM feature coverage increases by
`+0.06566`, but the paired forward-error interval
`[-0.09782, 0.02427]` crosses zero. Rendering shows bilateral seed contact,
left-only contact after release, and a no-contact endpoint without regrasp.

This does **not** close the blockwise-authority hypothesis and does not complete
the idea. All prior completion language is withdrawn. The training contract,
evaluation baseline, ICM attribution, and visual/recovery gates must be
re-audited from source and rerun with new seeds before any scientific
conclusion. Recovery/alternative-strategy counts remain `0/0` only as the
current unproven state, not as task completion.
