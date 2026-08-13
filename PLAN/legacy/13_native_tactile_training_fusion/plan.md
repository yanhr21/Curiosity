# Plan 13: Native Whole-Hand Tactile Training and Fusion

## Question

Does the current 54-patch native TacSL representation help a CarryBox policy
when RGB and measured object state are unavailable, and how should that
representation enter the serious SUGAR training stack?

## First matched experiment

Both actor arms preserve the same deployable SUGAR observation contract: the
official five-frame histories of base angular velocity, relative joint
position/velocity, previous action, and projected gravity; current base linear
velocity; normalized motion phase; and the official `35-D` Tracker command. The
command is exactly `29-D` reference joint position plus `3-D` reference root
linear and `3-D` reference root angular velocity. The official Generator's
36th `contact_label` output is excluded because it is a non-tactile proxy.
Neither actor receives RGB, measured or future object state, mass, friction,
rigid-contact labels, or simulator contact velocity. The privileged critic and
frozen official Refiner teacher remain training-only.

The resulting non-tactile actor input is `504-D`: `35-D` command, `465-D`
official history, `3-D` current base linear velocity, and `1-D` phase. This
retains the released Tracker's temporal structure. The withdrawn current-only
`129-D` diagnostic reached only 33--38 deterministic steps and zero tactile
contact, so it cannot initialize the tactile experiment.

The common base initialization uses the released CarryBox `tracker.pt`.
Command and history columns plus all later actor layers transfer directly;
contact-label and measured-object columns are omitted, while base linear
velocity, phase, and tactile columns begin with zero or low-gain authority.
Both matched arms then use the same saved base checkpoint. The released
Refiner remains the frozen BCPPO teacher rather than an actor input. Actor and
critic empirical observation normalization remain disabled, as in the
released SUGAR Tracker/Refiner BCPPO configuration.

Every physical-skin episode starts at official motion frame zero and evolves
continuously.  The ordinary SUGAR mid-trajectory reference-state reset is not
used here: teleporting the added elastomer collision bodies into a contact
pose skips the preceding dynamics and creates broad transient penetration.

- `tactile`: four causal 50 Hz frames from both hands, 27 physical patches per
  hand, and native normal plus signed-XY shear on every `20 x 25` patch;
- `zero`: identical model, optimizer, task, physics, teacher, seed, and tensor
  width, but an exact-zero observation function that does not read a sensor.

The tactile tensor is serialized as
`[hand,history,patch,channel,row,column]`. Each hand therefore supplies 324
spatial channels. The existing serious SUGAR `SpatialTactileEncoder`
(`32/64/64` convolutions and 128-D embedding per hand) feeds the unchanged
`512/256/128` SUGAR actor MLP. The two student arms use the same initialization
and zero tactile maps to an exact-zero embedding. The existing spatial encoder
and its fusion columns are the tactile extension. Repository-native SUGAR
BCPPO retains the released Refiner as frozen teacher and uses the privileged
critic only during training; no new toy network is introduced.

## Execution order

1. Run a live one-update preflight for the tactile arm and confirm the exact
   deployable `504-D` command/history/proprioception/phase contract,
   expected tactile dimensions, real contact signal, encoder gradients, and
   checkpoint creation.
2. Run the matched one-update exact-zero arm and confirm identical non-tactile
   inputs with zero sensor observation.
3. Run matched tactile/zero endpoints serially, first on the known motion-45
   learnability case and then with mass/friction/contact disturbances.
4. Freeze both policies and compare teacher action error, task return, lift,
   hold duration, drop/fall rate, and disturbance recovery on common rollouts.
5. Only after this no-RGB comparison is resolved, add the existing official
   ResNet18 RGB feature branch and compare RGB, tactile, and RGB+tactile.

Tactile benefit requires a matched frozen-policy difference on physical task
behavior. A nonzero encoder gradient or lower training loss alone is not a
positive result.

## Active status on 2026-08-11

The reusable physical sensor, tensor serialization, and full CarryBox videos
are complete. The active Tracker-command tactile task registration and serious
spatial fusion code now exist. The history-preserving official-Tracker
warm-start runtime gate and then the live/zero one-update preflights are the
next work; the withdrawn `129-D` diagnostic and older `890-D` route below
cannot substitute for them.

## Historical `890-D` diagnostic record

Everything in this section used the earlier reference-only actor containing
full future reference state. It is retained to explain what was learned about
the tactile representation and late-fusion failure, not as completion of the
active deployable-input experiment. Any sentence describing a “next” gate
records the historical sequence and is superseded by the active status above.

The pure-model official warm-start audit passes: zero tactile reproduces the
released Refiner actor and critic exactly, the official base columns receive
zero gradient, and gradients reach the serious spatial encoder and appended
tactile columns. This resolves model fusion. The uninterrupted live gate also
now passes on `server13`: the tactile arm reaches contact continuously at
update 9, signal is present in four updates, gradients reach only the declared
spatial adapter, and the matched exact-zero arm stays zero without reading a
sensor. Both arms start from exactly equal policy states and retain the
official checkpoint learning rate.

The first frozen update-15 evaluation is deliberately unresolved. Tactile
improves maximum lift and final object-position error but worsens episode
length, return, and student/teacher action MAE. It establishes online policy
influence after only four contact-bearing updates, not benefit.

The fresh matched 64-update pair is now complete. It began from 31/31 exactly
equal policy tensors; only the eight declared tactile-adapter tensors differ
at update 63. The tactile arm saw physical signal in 17 updates and 235 rollout
frames, while the exact-zero arm remained identically zero. Under the nominal
0.5 kg condition, tactile reaches 0.8485 m maximum lift versus 0.6755 m and has
higher cumulative reward, but it terminates 60 steps earlier, has 0.1131 m
more final position error, and has higher teacher-action error. Under the
predeclared 1.0 kg, static-friction 0.25, dynamic-friction 0.20 condition,
tactile reaches 0.1752 m versus 0.2067 m and has slightly lower reward, while
its final position error is 0.0110 m lower. Both disturbed arms terminate at
nearly the same step for object orientation. This is policy influence with a
mixed single-seed outcome, not tactile usefulness or generalization.

The immediate next evidence is synchronized world-plus-anatomical behavior
video and tactile-input dependence on these frozen endpoints. The dependence
test is complete: live, exact-zero, and fixed anatomical-patch-permuted inputs
are action-identical before contact and diverge exactly when tactile support
begins at step 243. Same-state maximum action changes are `1.1834` against
zero and `0.6082` against the permutation, proving that the policy reads both
the modality and its spatial patch order. The live arm lifts more than zero in
the nominal rollout, but its common-horizon position tracking is worse and its
reward advantage over zero is only `0.3715`; the disturbed condition also
remains mixed. Cross-seed and additional disturbance profiles are required
before a benefit claim.

The synchronized camera-enabled videos are a distinct presentation cohort,
not a numerical replacement for the camera-free matched evaluation. Each
shows the same-step CarryBox world state and both complete anatomical hand
maps, with an explicit label for the tensor actually fed to the actor. The
live, exact-zero, and fixed-permutation H.264 files now fully decode
`363/363`, `361/361`, and `329/329` frames. They share one readable physical
active-taxel scale; no sensor value or rollout was altered for display.

## Frozen tactile-authority response curve

Before changing the serious fusion architecture or training longer, evaluate
the existing update-63 tactile checkpoint at fixed tactile-column authority
scales `0`, `0.25`, `0.5`, `0.75`, and `1.0`. This is a no-learning nominal
CarryBox diagnostic: checkpoint, initial physical state, raw tactile, actor
input mode, reference, seed, physics, and every non-tactile weight remain
fixed. Only the appended tactile-feature columns of `actor.0` are multiplied
after checkpoint loading. Scale zero must exactly reproduce the same actor
mapping as a zeroed tactile input.

Judge every scale over its common rollout horizon using task reward, reference
position error, lift, termination, and same-state live-versus-zero action
change. Select no scale from a single favorable endpoint: the curve is used to
decide whether the present late fusion is over-authoritative. If an interior
scale consistently improves the lift/tracking tradeoff, the next trainable
route will make that authority explicit and bounded. If no interior scale does
so, prioritize contact-balanced training/data coverage rather than adding more
late-fusion gain.

The complete curve now passes every matched check. Tactile support begins at
step 243 for all scales, every pre-contact action is exact, and scale zero is
bitwise identical to the independently evaluated zeroed-input rollout. Over
the common 340 steps, final lift rises approximately monotonically from
`0.6126` at zero authority to `0.8485 m` at full authority, while mean object
position error also worsens monotonically from `0.02168` to `0.03378 m`.
Scale `0.5` has only `+0.4054` common-horizon reward relative to zero while
increasing mean position error to `0.02623 m`; scale `1.0` exits earliest by
object-position failure. No nonzero scale jointly improves reward, tracking,
and lift.

This resolves the architecture decision: the learned tactile branch contains
a lift-promoting correction, but ordinary late concatenation has learned
increasingly aggressive authority instead of stable task correction. Do not
select a post-hoc scalar as the next model. The next matched training route is
an explicit bounded first-layer tactile correction plus contact-balanced
distillation, while preserving the official SUGAR base path and exact-zero
mapping. The `0.15` per-hidden-unit preactivation cap is declared from current
telemetry before the new run: the learned update-63 branch has active-frame
median/p95/max first-layer corrections `0.270/0.323/0.397`, so the cap is near
the empirically moderate `0.5` authority without allowing weights to restore
full authority by simple growth.

The first bounded/contact-balanced 16-update pair is a negative stability
result. It passes the live-contact and matched-checkpoint gates: both arms
start with all `31/31` policy tensors equal, the tactile arm sees `108`
tactile-bearing rollout frames in updates 9--12, the zero arm remains exact
zero, and only the eight declared adapter tensors differ at update 15. In the
frozen nominal rollout tactile raises maximum lift by `0.1819 m`, but it
terminates 24 steps earlier and increases teacher-action MAE by `0.03105`.
Over the matched 376-step common horizon, reward is lower by `11.5903` and
mean/final position error are worse by `0.01296/0.06287 m`. On the same 133
physically supported states, live tactile has
teacher MAE `0.16977` versus `0.11450` when the same frozen actor receives
zero tactile. The contact-only reduction amplified the first 4/48 supported
rollout by 12x, and update-12 distillation loss reached `10.0469`. Therefore
the hidden-unit cap and contact-only mean do not solve stable fusion.

The next route is fixed before observing its behavior. It retains the same
official actor, serious spatial encoder, `0.15` hidden preactivation cap,
continuous CarryBox physics, teacher, seed, and tactile-only gradient gate. It
returns to official all-sample distillation, because zero tactile rows already
give the adapter zero gradient, and adds a direct per-action tactile residual
bound of `0.1` relative to the exact official zero-tactile action. The static
audit drives the raw residual to `0.6821`, limits the applied residual to
`0.1000`, preserves exact zero-tactile behavior, and keeps all base-column
gradients zero. Live training and frozen behavior remain separate gates.

The action-residual 16-update matched gate is now complete. Both arms again
start from exactly equal policy tensors, only the eight tactile adapter tensors
change, and the tactile arm reaches contact in updates 9--12. Restoring the
official all-sample mean limits the largest observed distillation loss to
`1.8421` rather than the preceding route's `10.0469`. In frozen evaluation,
the actual same-state live-versus-zero action difference is capped at
`0.09997`. Over the common 365 steps, tactile raises lift by `0.15059 m` and
lowers mean/final object-position error by `0.00409/0.01274 m`, but reward is
lower by `4.7322` and the tactile arm terminates 35 steps earlier for object
orientation. Contact-state teacher MAE is still `0.03087` worse with live
tactile than with zeroed tactile. This is a materially improved but still
mixed single-seed result. Extend this exact bounded-action route before
changing architecture again; do not call the 16-update gate tactile benefit.

The fresh 64-update action-residual pair is now complete. Both arms start with
all `31/31` policy tensors exactly equal; only the same eight declared tactile
adapter tensors differ at update 63. The tactile arm receives real signal in
19 updates and 606 rollout frames, from update 9 through update 63. In the
camera-free frozen evaluation it terminates at step 356 versus 400 for zero,
both on object orientation. Over the matched first 356 steps, tactile changes
reward by `-1.4274`, mean/final object-position error by
`+0.00176/+0.06442 m`, and maximum/final lift by `-0.00277/-0.02997 m`. On the
same 113 tactile-supported states, live tactile teacher MAE is `0.01810` worse
than the zeroed counterfactual. The 0.1 action bound controls authority but
does not make the learned correction useful at this endpoint. Do not extend
this exact PPO route to more updates or call it tactile benefit.

Before another policy run, the next gate is held-out contact-state
predictability with the unchanged serious spatial encoder and official base
actor. Train only the bounded tactile residual against the official privileged
teacher residual on a declared training split, and require lower validation
error than the exact-zero residual baseline on physically supported held-out
states. This is a supervised fusion diagnostic, not policy success. Only a
positive held-out result can authorize another matched closed-loop policy
experiment; a negative result means the current tactile channels/target do not
support this fusion objective.

The gate is fixed before its test outcomes. Every sample comes from a
continuous motion-45 frame-zero rollout driven by the exact-zero official base
actor; the physical 54-patch TacSL history is recorded pre-action but cannot
change the collection trajectory. Only rows with nonzero current TacSL contact
are retained, and the complete four-frame `324000-D` tensor is stored without
pooling as exact sparse indices and float32 values. Training uses the task's
default `0.5 kg` object and its `0.75 kg` mass-scaled condition. Checkpoint
selection uses only `0.625 kg` with static/dynamic friction `0.40/0.30`.
Untouched tests are `1.0 kg` at default friction and `0.5 kg` with
static/dynamic friction `0.25/0.20`. These masses are the training-task object
conditions; the canonical representation video deliberately used the separate
`0.3023376 kg` successful-carry condition and must not be mislabeled as this
gate's nominal mass.

The model is the existing `32/64/64` per-hand spatial encoder, `128-D` per-hand
embedding, frozen official `512/256/128` actor, `0.15` hidden tactile cap, and
`0.1` normalized-action residual cap. Adam runs for exactly 400 contact-only
minibatches at learning rate `1e-3`, batch size 16, and seed 13011. The lowest
selection MAE at 25-step intervals selects the checkpoint. The gate passes
only if a post-step-zero checkpoint beats the stored exact-zero official action
on the selection condition, on each of the two untouched test conditions, and
on their aggregate, while the official actor base columns remain bitwise
unchanged. Patch-permuted error is reported as a spatial diagnostic but is not
substituted for the exact-zero primary comparison.

If the predictability gate passes, do not jump directly to PPO. Load its
selected adapter as one frozen policy checkpoint and evaluate the two untouched
conditions again with either live tactile or the exact-zero/no-sensor-read
observation. The live and zero arms otherwise share the checkpoint, source
state, seed, physics, and task. On each condition, live tactile must have
higher common-horizon task reward, lower common-horizon mean object-position
error, and at least as many completed steps as exact zero. Only a pass on all
three checks in both conditions authorizes a later matched PPO experiment.

The completed result separates information from control. The predictability
test passes with a `26.26%` aggregate held-out teacher-action MAE reduction,
but the frozen behavior comparison fails on both conditions. At `1.0 kg`, live
tactile has better mean tracking and lift but `2.19394` lower common-horizon
reward and terminates eight steps earlier. At low friction, live tactile has
`0.35444` higher reward and equal duration but `0.000745 m` worse mean tracking
and `0.03916 m` lower lift. This blocks PPO under the frozen rule. The next
question must distinguish a teacher-action target mismatch from closed-loop
distribution shift before changing the fusion architecture or collecting more
policy updates.

The offline canonical-trace check constrained the historical fusion choice.
That route used late concatenation of the two `128-D` per-hand embeddings
before `actor.0`; it preserves the official policy exactly at zero tactile and
introduces only a small initial action perturbation on real CarryBox contact.
Its later matched closed-loop result was negative, as recorded above.

## Withdrawn diagnostics

The first 64-update pair used `init_with_ref=true`, which is appropriate for
the original unsensorized SUGAR geometry but not for the added physical skin.
Its first tactile rollout had `37.46%` exact-nonzero taxels, versus `0.744%` in
the complete continuously simulated canonical CarryBox trace, and its direct
frame-230 frozen restores reached thousands of active taxels per hand versus
the canonical `275/271` maxima.  That pair remains archived as a reset-design
diagnostic and cannot decide tactile usefulness.  Fresh training uses only the
continuous frame-zero route above.

The next random compressed-student route did use the correct continuous reset,
but all 4,464 sampled environment-frames through update 92 remained zero
tactile and mean episode length reached only 39.44 frames.  It never reached
the contact interval and was stopped. The later historical official-width
warm-start removed that contact-exposure failure without reintroducing
teleportation.
