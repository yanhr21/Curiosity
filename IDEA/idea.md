# SUGAR + High-Fidelity Tactile CarryBox Foundation

> Superseded for new research execution on 2026-07-23 by
> `IDEA/06_sugar_smp_tactile_strategy_exploration.md`. This file remains the
> accepted direct-tactile foundation and historical claim boundary.
> `IDEA/07_demo_conditioned_internal_reward.md` is the active companion
> hypothesis for a causal learned demonstration-compatibility reward. It does
> not replace original ICM or the frozen SMP prior.
> The first frozen tactile-only slip evaluation is recorded in
> `DOCS/legacy/sugar_direct_tacsl_slip_calibration_result_20260723.md`; it is a
> controlled, rendered negative generalization result and does not pass the
> new strategy mainline's Stage E.

## Current Mainline

The active research mainline is now **SUGAR + high-fidelity tactile sensing**
for humanoid CarryBox. The accepted official SUGAR reproduction remains the
frozen baseline; tactile sensing is added on top of that baseline rather than
replacing its code, assets, task definitions, checkpoints, or stage order.

The central question is whether spatially resolved tactile feedback lets a
SUGAR policy maintain a safer, more stable grasp when payload mass, surface
friction, geometry, center of mass, and external disturbances differ from the
human-video demonstration.

## Working Title

Tactile-SUGAR: video-guided humanoid loco-manipulation with dense pressure,
shear, and visuotactile feedback for robust unknown-load carrying.

## Non-Negotiable Definition of Tactile

For this project, tactile sensing must be a direct, spatially resolved sensor
modality. A valid simulated tactile observation must contain:

- a taxel-aligned normal-force field, converted to pressure using taxel area;
- a two-axis tangential/shear-force field that represents frictional loading;
- a spatial contact patch rather than only a body-level resultant wrench;
- GelSight-style tactile RGB and/or depth deformation images; and
- time-resolved measurements sufficient to observe stick, incipient slip, and
  sliding.

The following do **not** count as tactile input:

- binary contact labels or force-threshold flags;
- SUGAR's existing hand-contact label history;
- a summed contact force, force matrix, or six-axis body wrench by itself;
- object pose, hand-object distance, penetration flags, reward terms, or other
  state-derived proxies presented as touch;
- a learned latent target generated from privileged simulator state without a
  corresponding tactile sensor measurement.

PhysX contact reports may be used only as an independent calibration check,
not as the policy's claimed tactile modality.

## Selected Technical Path

The primary implementation target is the official IsaacLab `v2.3.2`
TacSL-based `VisuoTactileSensor` in `isaaclab_contrib`. It directly exposes
per-taxel normal and two-axis shear forces, tactile RGB/depth images, tactile
point poses, and penetration depth. Its force field uses an SDF contact query
with normal stiffness, tangential stiffness, and Coulomb friction, while its
camera path renders GelSight-style images from deformation depth.

Primary sources:

- [IsaacLab v2.3.2 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2)
- [official visuo-tactile sensor documentation](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/visuo_tactile_sensor.html)
- [official `isaaclab_contrib` tactile API](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab_contrib/isaaclab_contrib.sensors.html)
- [TacSL paper](https://arxiv.org/abs/2408.06506)

The current faithful SUGAR environment uses IsaacLab `v2.3.0`, which predates
this official tactile integration. The tactile branch must first test an
auditable upgrade to official `v2.3.2` (commit
`37ddf626871758333d6ed89cf64ad702aef127d0`). If SUGAR compatibility prevents
that upgrade, only a minimal, source-preserving backport of the official
`v2.3.2` sensor, assets, and configs is allowed. A hand-written toy tactile
array is not an acceptable fallback.

IsaacLab `v2.3.2` also adds friction-force reporting to the ordinary
`ContactSensor`, but that remains contact-report data without an elastomer or
taxel-resolved pressure/shear field. It is useful as a non-tactile proxy control
or validation reference, not as the selected tactile modality.

TacEx is a secondary small-scale validation option because it couples an
explicit soft-body gel simulation to GelSight rendering in Isaac Sim. It is not
the first large-scale training backend because its dependency and parallel
stability risks are materially higher. It may be used to test whether the
faster TacSL penalty model preserves the relevant pressure footprint and
deformation trends.

## Hardware and Sensor Target

The first embodiment target is a pair of GelSight-style elastomer patches on
the left and right G1 palm contact surfaces, using official GelSight Mini or
R15 assets and configs. Finger-pad coverage is a later extension after the
dual-palm path passes fidelity and throughput gates.

The CarryBox object must have a valid SDF collision representation for the
force-field query. Tactile data must remain separated by hand and retain its
taxel/image layout; reducing it to a contact bit or a single scalar before the
policy is forbidden.

## Learning Hypothesis

Video supplies global task intent and nominal whole-body motion. Dense tactile
feedback supplies local information that video and proprioception cannot
reliably infer under occlusion: contact-patch migration, pressure imbalance,
frictional shear, incipient slip, and load redistribution. The intended policy
therefore fuses SUGAR's existing motion/proprioceptive observations with a
spatial tactile encoder, while retaining the original SUGAR baseline as the
control.

The first tactile experiment changes observations only. Tactile-shaped rewards
are deferred so that any gain can be attributed to sensing rather than reward
engineering.

An implementation audit found that the official SUGAR Refiner actor and critic
currently share the same privileged observation group. The actor directly sees
exact object pose, orientation, linear/angular velocity, and future reference
terms computed relative to the exact current object state. The first matched
tactile experiment deliberately preserves this official baseline, but it is an
exact-state simulation upper bound in which touch may be informationally
redundant. If that branch does not improve the frozen control, a separately
predeclared matched branch uses a reference-only actor: actual object state is
replaced by the reference plan and future object commands are recomputed in the
current reference frame, while the critic remains privileged and all
full/zero/pressure budgets stay identical. Object state must never be relabeled
as tactile evidence. The fixed protocol is recorded in
`DOCS/legacy/sugar_tactile_reference_only_actor_protocol.md`.

A pre-result optimizer audit of the running reference-only branch found that
actor-KL adaptation controls one shared actor/critic Adam. The frozen zero actor
therefore drives its critic rate to `1e-2`, while live tactile roles train their
critics near `1e-4`. That branch was stopped after its common model-1000
diagnostic and cannot cleanly attribute performance to tactile input. Its
formal follow-up is
the fixed, separately predeclared critic/adapter learning-rate contract in
`DOCS/legacy/sugar_tactile_optimizer_deconfounding_protocol.md`. Its official Isaac
App `1 env x 1 update` admission smoke now passes all source, checkpoint,
optimizer-group, frozen-actor, and metric checks; this admits matched formal
training but is not itself tactile-performance evidence.

If the optimizer-clean reference-only actor still fails despite real taxel
exposure and closed-loop action dependence, the next branch is not a
result-selected easier reset or proxy task. A separately predeclared
latent-contact-dynamics protocol trains matched full/zero/pressure policies
across actor-hidden payload mass, friction, COM, and contact-phase disturbances
so the policy can learn the meaning of pressure/shear under slip. Its fixed
fairness and claim boundary is in
`DOCS/legacy/sugar_tactile_latent_contact_dynamics_protocol.md`.

## Claim Gates

Before training claims, the sensor must pass:

- zero/noise behavior with no contact;
- monotonic pressure response under a controlled normal-load sweep;
- correct shear direction and magnitude trend under tangential motion;
- a Coulomb-consistent transition from sticking to sliding;
- spatial contact-patch motion under known box translations/rotations;
- left/right consistency and stable 60 Hz temporal output;
- integrated taxel-force agreement with an independently measured simulator
  wrench within a declared tolerance; and
- calibration against a physical GelSight-class sensor before any
  real-tactile or sim-to-real claim.

Without physical calibration data, results must be labeled
**high-fidelity simulated tactile**, not physically validated tactile.

## Required Ablations

- frozen official SUGAR baseline;
- SUGAR plus body-level contact force or contact labels, reported only as a
  non-tactile proxy control;
- SUGAR plus normal pressure map;
- SUGAR plus normal pressure and shear maps;
- SUGAR plus GelSight RGB/depth;
- full pressure + shear + GelSight model;
- tactile dropout/noise and sensor-parameter randomization;
- matched-training-budget no-tactile control.

Evaluation must include held-out mass, friction, geometry, center-of-mass
offset, and lateral disturbance. Report SUGAR task success/error together with
drop/fall rate, slip distance, pressure imbalance, peak pressure, shear ratio,
and contact-patch stability.

## Success Standard

The tactile model succeeds only if it improves the faithfully reproduced SUGAR
CarryBox baseline on held-out carrying conditions without hiding failures
behind privileged state, proxy contact labels, reward shortcuts, or a larger
training budget. All code changes must remain auditable against official SUGAR
and official IsaacLab/TacSL sources.
