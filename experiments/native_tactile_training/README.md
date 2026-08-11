# Native whole-hand tactile policy experiment

This directory contains the active Plan-13 comparison.  The question is
whether the current physical 54-patch TacSL signal helps a serious SUGAR
CarryBox student when RGB and current measured object state are unavailable.

## Matched arms

- `tactile`: four causal 50 Hz frames of normal and signed-XY shear from both
  hands, 27 patches per hand and `20 x 25` taxels per patch;
- `zero`: the same `324000-D` input width and the same trainable spatial
  encoder, but an exact-zero observation function that never reads a sensor.

Both actors otherwise use the existing official-width SUGAR reference-only
observation contract: robot state and the commanded motion plan remain, while
every measured current-object channel is replaced by its reference-plan
counterpart.  The released Refiner checkpoint initializes the unchanged
`512/256/128` actor and privileged critic exactly.  Zero tactile therefore
recovers the official SUGAR mapping, while only the serious spatial encoder
and the first-layer tactile columns are allowed to change the actor.  Neither
actor receives RGB, measured current object state, mass, friction, contact
labels, or simulator contact velocity.

## Withdrawn reset diagnostic

The first 64-update pair completed normally and began from model states that
were exactly equal for all 39 tensors.  It is withdrawn from the active result,
because it used ordinary SUGAR random mid-trajectory reference-state resets.
With added physical elastomer collision bodies, those resets skip the preceding
contact dynamics and produce broad transient penetration: `37.46%` of training
taxels were exact nonzero, compared with `0.744%` in the continuously simulated
canonical CarryBox trace.  Direct frame-230 evaluation likewise activated
thousands of taxels per hand versus canonical maxima of `275/271`.

Those checkpoints and evaluations live only in the project archive as a reset
design diagnostic.  They cannot establish tactile benefit or harm.  The active
configuration now starts every episode at official frame zero and reaches
contact only through continuous simulation.

The old one-update preflights are archived with that pair.  Their model-shape,
no-camera, and exact-zero observations remain implementation diagnostics, but
their teleported nonzero tactile values are not physical-contact evidence.

## Withdrawn compressed-student diagnostic

A later continuous-start route removed the reset burst but randomly
initialized a compressed 129-D student.  Through 93 updates its 4,464 sampled
environment-frames remained exact-zero tactile, while mean episode length
plateaued at 39.44 frames, far before contact near frame 230.  It was stopped
because it could only study pre-contact imitation.  The run and its stop
record are archived under
`withdrawn_compressed_reference_student_20260811/`; they are not a tactile
training result.

The active replacement combines the clean frame-zero physical rollout with
the existing exact official Refiner warm-start and tactile-only actor gradient
gate.  Its first runtime gate must reconstruct the official action at zero
tactile and then reach the real contact interval before matched training is
admitted.

The model-only warm-start gate is complete. It confirms the official `890-D`
actor and critic contract, `324000-D` raw tactile input, `256-D` tactile
embedding, exact zero embedding, and exactly zero action/critic error relative
to the released Refiner at zero tactile. Backpropagation leaves the official
base input columns unchanged while reaching both the tactile input columns and
the spatial encoder. The machine-readable result is
`preflight_official_warmstart_static_20260811/report.json`. This is structural
evidence only; the live continuous-contact gate and matched experiments are
completed below.

The same serious late-fusion path has also consumed the retained canonical
CarryBox taxel trace without Isaac Sim. Across all `660` frames, current
physical contact is present on `272` frames and the causal four-frame input is
nonzero on `275`. Every all-zero history maps to an exact-zero embedding and
zero action perturbation; every nonzero history reaches the `256-D` spatial
encoder. At the initial `0.01` tactile gain, the standardized zero-base action
delta has median maximum component `0.00152` and maximum `0.00274`. This shows
that the current late concatenation is correctly ordered, zero preserving, and
initially small—not that tactile improves the policy. The record is
`canonical_trace_fusion_20260811/report.json`. The synchronized H.264
`canonical_trace_fusion_20260811/canonical_carrybox_late_fusion.mp4` places the
actual CarryBox/whole-hand video beside active taxels, input RMS, encoder
features, first-layer contribution, and the explicitly diagnostic standardized
action delta; it fully decodes `430/430` frames.

An actual-contact backward probe uses canonical source frames `244:252` and
the same tactile-only finetune gate. The official actor base-column gradient is
exactly `0`; the appended tactile columns have gradient L2 `0.14260`, and the
spatial encoder has gradient L2 `0.002116`. Thus recorded physical contact can
train the intended adapter without updating the accepted non-tactile mapping.
The probe is still an offline actor-loss check, not online PPO evidence.

## Live matched endpoints

The continuous frame-zero live gate now passes on `server13`. Both 16-update
arms start from model states that are exactly equal across all 31 comparable
policy tensors and retain the official checkpoint learning rate
`5.0625e-5`; optimizer moments are not loaded. The tactile arm first reaches
physical contact in update 9. Updates 9--12 contain respectively 2, 24, 24,
and 4 tactile-bearing rollout frames, and the spatial encoder plus appended
first-layer tactile columns receive nonzero gradients. Every update in the
exact-zero arm remains exactly zero, and its observation function does not
read a sensor.

At update 15, only the eight declared tactile-adapter tensors differ between
the arms (`L2=0.87715`); the accepted actor mapping is unchanged in the zero
arm. A deterministic no-learning nominal evaluation is mixed rather than a
positive result. Tactile raises maximum box lift from `0.6523` to `0.7008 m`
and lowers final object-position error from `0.1696` to `0.1351 m`, but it
terminates six steps earlier, has lower cumulative reward, and increases
student/teacher action MAE from `0.0303` to `0.0409`. Only four updates contain
contact, so this endpoint proves online signal/learning entry but neither
tactile benefit nor harm. This motivated the separately initialized matched
64-update pair below; it was not a continuation selected from this outcome.

The fresh 64-update pair is also complete. All 31 initial policy tensors are
exactly equal; only the eight declared tactile-adapter tensors differ at
update 63 (`L2=1.79687`). The tactile arm has 17 contact-bearing updates and
235 tactile-bearing rollout frames; the zero arm stays bitwise zero throughout.

The frozen single-seed result is mixed. In the nominal 0.5 kg condition,
tactile reaches `0.8485 m` maximum lift versus `0.6755 m` and has higher
cumulative reward, but terminates 60 steps earlier, has `0.1131 m` more final
position error, and has higher student/teacher action error. In the explicit
1.0 kg, static-friction `0.25`, dynamic-friction `0.20` condition, tactile
reaches `0.1752 m` versus `0.2067 m`, has slightly lower reward, and has
`0.0110 m` lower final position error. Both disturbed arms terminate at nearly
the same time for object orientation. The checked pair summaries are
`matched_64u_frozen_eval_20260811/nominal_comparison.json` and
`heavy_low_friction_comparison.json`. This proves policy influence, not stable
tactile benefit. Synchronized behavior video, tactile dependence, more seeds,
and more disturbance profiles remain.

Input dependence is now resolved on the same frozen tactile checkpoint. Three
camera-free nominal rollouts feed live tactile, exact-zero tactile, or a fixed
within-hand permutation of the 27 anatomical patches. Their initial physical
states are exact, and every pre-contact action is exact through step 242. At
the first tactile-supported step 243, the selected actions diverge. On the
same physical states, live-versus-zero action difference reaches `1.1834` in
one action component, while live-versus-permuted reaches `0.6082`. Closed-loop
object trajectories differ by as much as `0.4156 m` and `0.3050 m`.

This proves that the learned policy reads the tactile input and depends on its
anatomical patch order. It still does not prove stable benefit. Over the common
340-step live/zero horizon, live reward is only `0.3715` higher and mean object
position error is worse (`0.03378` versus `0.02168 m`), although final-common
lift is higher (`0.8485` versus `0.6126 m`). Against the corrupted patch order,
live reward over the common 325 steps is `14.6136` higher and mean position
error is lower. The checked record is
`matched_64u_tactile_dependence_20260811/summary.json`.

The synchronized camera-enabled cohort is kept separate from these numerical
rollouts because enabling rendering can perturb GPU simulation numerics and
termination timing. Its videos show actual world behavior above both complete
27-patch anatomical hand maps. See `REPRODUCE.md` for the single serial entry
point and the exact distinction between displayed physical sensor values and
the tensor fed to the actor.

The complete presentation cohort is now available under
`matched_64u_tactile_dependence_videos_20260811/`:

- `live_policy_world_and_bilateral_tactile.mp4`: physical tactile enters the
  actor (`363` fully decoded frames);
- `zeroed_policy_world_and_bilateral_tactile.mp4`: the actor receives exact
  zeros while the lower maps continue to show the real physical sensors (`361`
  fully decoded frames);
- `patch_permuted_policy_world_and_bilateral_tactile.mp4`: the actor receives a
  fixed within-hand permutation while the display remains anatomical (`329`
  fully decoded frames).

All three share one active-taxel 95th-percentile display scale (`0.02117 N`
normal magnitude and `0.02650 N` XY shear magnitude), written directly in the
video. This fixes the misleading near-white first render without changing any
raw tactile value, action, trajectory, or evaluation metric. The camera cohort
passes its matched-state/input audit; its metric summary is the adjacent
`summary.json`. `MANIFEST.json` is the compact inventory of the complete active
training package.

## Frozen tactile-authority curve

The update-63 tactile checkpoint was also evaluated without learning at fixed
`actor.0` tactile-column scales `0`, `0.25`, `0.5`, `0.75`, and `1.0`. All
initial states and pre-contact actions are exact, tactile begins at step 243 in
every arm, and scale zero exactly reproduces the independent zeroed-input
trajectory. The checked record is
`matched_64u_tactile_authority_curve_20260811/summary.json`.

Over the common 340-step horizon, lift rises from `0.6126` to `0.8485 m` as
authority rises, but mean object-position error worsens from `0.02168` to
`0.03378 m`. No nonzero scale improves reward, tracking, and lift together.
The learned branch therefore carries useful lift-direction information but is
not a stable correction channel. The next architecture uses an explicit
bounded first-layer correction and contact-balanced distillation; it does not
call a favorable post-hoc scale a training result.

The bounded route's CPU structural audit is now complete at
`bounded_fusion_static_audit_20260811/report.json`. It retains the same
`324000-D` raw tensor, `256-D` serious spatial embedding, official `890-D`
base actor, and 29 actions. Zero tactile remains exactly equal to the official
actor. A tanh correction caps each of the 512 first-layer tactile
preactivations at `0.15`; the audit drives it into saturation, keeps all base
column gradients exactly zero, reaches both tactile columns and the encoder,
and verifies that contact-balanced distillation averages the declared
supported rows rather than all-zero tactile rows. This is structural evidence,
not yet live training or policy benefit.

The subsequent bounded/contact-balanced 16-update pair is complete and is a
negative stability result. Both arms start from all `31/31` policy tensors
exactly equal. The tactile arm receives `108` tactile-bearing rollout frames
in updates 9--12; the zero arm stays exact zero. Only the eight declared
adapter tensors differ at update 15. Frozen tactile raises maximum lift by
`0.1819 m`, but ends 24 steps earlier and worsens teacher-action MAE by
`0.03105`. On the matched 376-step common horizon, reward is lower by
`11.5903` and mean/final object-position errors are worse by
`0.01296/0.06287 m`. On the same 133 contact-supported states, live tactile has teacher
MAE `0.16977`; counterfactually zeroing tactile lowers it to `0.11450`.
Therefore this route changes behavior but does not make a stable or
teacher-consistent correction. The checked records are
`bounded_contact_balanced_16u_endpoint_comparison_20260811.json` and
`bounded_contact_balanced_16u_frozen_eval_20260811/`.

The next route is deliberately narrower. It retains the `0.15` hidden cap but
returns to official all-sample distillation and additionally limits every
normalized tactile action residual to `0.1` relative to the exact official
zero-tactile action. Its static audit is
`action_residual_fusion_static_audit_20260811/report.json`: a strong probe's
raw `0.6821` action residual is limited to `0.1000`, zero tactile is still
exact, and gradients reach only the declared tactile adapter. This is not yet
a live policy result.

The action-residual 16-update pair is now complete. Initialization is exact,
only the same eight adapter tensors differ, and the tactile arm reaches real
contact in updates 9--12. Official all-sample distillation lowers the largest
observed contact-update distillation loss from `10.0469` to `1.8421`. The
frozen actor's maximum same-state live-versus-zero action difference is
`0.09997`, so the declared output bound is active. Over the common 365 steps,
tactile raises lift by `0.15059 m` and lowers mean/final position error by
`0.00409/0.01274 m`, but reward is lower by `4.7322` and the tactile arm exits
35 steps earlier for object orientation. On its 122 tactile-supported states,
live tactile teacher MAE remains `0.03087` worse than zeroed tactile. This is
better controlled and promising for tracking, but it is still mixed and not a
tactile-benefit claim. Records are under
`action_residual_16u_frozen_eval_20260811/`.

The fresh action-residual 64-update pair is also complete and closes this exact
route as a negative single-seed result. The tactile and zero arms begin with
all `31/31` tensors exactly equal; only the eight declared adapter tensors
differ at update 63 (`L2=1.90453`). The tactile arm has 19 contact-bearing
updates and 606 tactile-bearing rollout frames; zero remains exact throughout.

In camera-free frozen evaluation, tactile ends at step 356 and zero at step
400, both for object orientation. Over the common first 356 steps, tactile has
`1.4274` lower reward, `0.00176 m` higher mean position error, `0.06442 m`
higher final position error, and `0.02997 m` lower final lift. On the same 113
physically tactile-supported states, live tactile teacher MAE is `0.07741`
versus `0.05931` when the same frozen actor receives zeros. The direct 0.1
action cap is active—the largest same-state live/zero difference is
`0.09919`—but bounded authority alone does not make the correction useful.
The numerical records are under
`action_residual_64u_frozen_eval_20260811/`.

Human-review evidence is under
`action_residual_64u_policy_visualization_20260811/`. The most direct file is
`tactile_trained_vs_zero_trained_side_by_side.mp4`: tactile-trained is left,
zero-trained is right, each panel contains the actual CarryBox world rollout
and both complete 27-patch hand maps, both use the same physical-taxel P95
scale, and the paired H.264 fully decodes 348/348 frames. The separate full
resolution videos are retained beside it. This camera-enabled cohort is for
visual behavior/contact review; the camera-free pair above supplies the
matched numerical comparison.

The next gate is not a longer repeat of this PPO run. It must first test whether
the unchanged serious spatial encoder can predict the official teacher's
action residual better than an exact-zero residual on held-out physically
supported states. A positive held-out result is required before another policy
experiment.

The earlier runtime diagnosis remains relevant. On 2026-08-11 `server56` and
`server38` reported Vulkan `ERROR_DEVICE_LOST` during scene startup. A
direct 120-frame launch of the already successful canonical CarryBox collector
also failed at startup on `server38`. One launcher defect was then removed:
the force-only training scripts no longer request Isaac Sim's rendering
experience when every RGB/depth camera is disabled. With true force-only
headless mode and an isolated process cache, server51 created the physical
scene and completed updates 0--2 (72 frames), then hit the same driver loss.
All 72 frames were legitimately pre-contact and exact zero, so this interrupted
run ended before the source-frame-244 contact interval and cannot test the
sensor or policy. Its logs and partial checkpoints are archived rather than
kept as active evidence. The working `server13` allocation remains alive.

One real scene-config defect was fixed during this diagnosis: the original
unsensorized SUGAR material event targeted every robot body and therefore also
randomized the 54 newly added elastomer bodies. Training now reuses the
sensorized CarryBox event definition, which randomizes the original G1 bodies
but preserves the declared tactile-patch materials and friction.

## Reproduce

The complete current route is in [`REPRODUCE.md`](REPRODUCE.md). The commands
below are the training subset.

Run inside the retained allocation; each command is one independently
recorded child process and does not release the allocation:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tactile experiments/native_tactile_training/reproduced/tactile 512 2 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  zero experiments/native_tactile_training/reproduced/zero 512 2 13011
```

The arms must run serially.  Compare their initialization and endpoint with
`scripts/sugar/native_tactile/compare_native_tactile_training_endpoints.py`.
Frozen nominal evaluation uses
`scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh`.

Each run writes a compact tactile row for every PPO update. The first PPO
update also writes `training_signal_update0.json`; the first later update that
actually contains physical tactile contact writes
`training_signal_first_contact.json`, including the live spatial-encoder
gradient and parameter change from that same update. This distinguishes a
working model connection from a policy that never reaches contact.
