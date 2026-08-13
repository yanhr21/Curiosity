# SUGAR tactile latent-contact-dynamics protocol

## Status and activation boundary

This protocol is predeclared on 2026-07-15 before the active reference-only
seed-42 final checkpoint or performance result exists. It does not modify the
running reference-only training, its frozen evaluation, or its statistical
gate. It is a separate follow-up branch:

- finish and report the current reference-only seed42 branch unchanged, but do
  not replicate its discovered shared-adaptive-LR optimizer confound;
- first run the current-distribution branch under the separately predeclared
  `sugar_tactile_optimizer_deconfounding_protocol.md` contract;
- follow the predeclared optimizer-clean replication order 42, 43, 44; activate
  this latent-dynamics branch only on the first fully admissible negative
  single-seed report, with every earlier attempted seed required to have a
  complete positive report, and only after this branch's implementation/source
  manifest and reduced official-code diagnostics are audited;
- never pool its reports with exact-state or reference-only reports.

All claims remain `high-fidelity simulated tactile`. Nothing in this protocol
supports a physical GelSight or sim-to-real claim.

### Prepared implementation status (2026-07-15)

The following implementation is present only as an unregistered, inactive
candidate while optimizer-clean seed42 is still training:

- `SUGAR/source/sugar_rl/sugar_rl/assets/latent_contact_dynamics.py` wraps the
  existing official SUGAR/G1/R15 spawners and binds an explicit
  average-combine physics material to the unchanged CarryBox and all possible
  palm interface bodies (the two official R15 elastomers and the two backing
  rubber-hand colliders). Because IsaacLab's nested binding helper
  deliberately skips instanced prims, the wrapper also authors the same
  physics-purpose material as a stronger inherited binding on each editable
  R15 reference root; it does not edit the referenced R15 geometry;
- `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/latent_contact_visuotactile_sensor.py`
  selects a per-environment coefficient for the current lazy-update subset and
  calls the existing official TacSL force implementation via `super()`; it
  does not copy or rewrite the SDF/penalty-force equation. The TacSL Coulomb
  cap is matched to PhysX **dynamic** friction; PhysX static friction is logged
  separately and constrained to be no smaller than dynamic friction;
- `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/latent_contact_dynamics_events.py`
  predeclares a fixed stratified mass/friction/COM tuple and one hidden
  world-XY pulse per episode, eligible only during the official human-video
  reference contact schedule;
- `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/carry_box_tactile_latent_contact_dynamics_env_cfg.py`
  composes full, zero, and pressure-only configs without adding the tuple or
  reference contact label to actor observations.
- `SUGAR/scripts/sugar_rl/audit_latent_contact_dynamics_preflight.py` and
  `scripts/sugar/run_sugar_latent_contact_dynamics_preflight.sh` implement the
  finite admission diagnostic; the runner additionally requires a valid
  optimizer-clean negative report and an explicit activation environment flag.
  Both the login-node watcher and compute runner require evaluation seed 42, a
  passing suite-manifest audit, and exact SHA binding to the current
  optimizer-clean single-seed executor in addition to the locked analyzer and
  unchanged statistical-gate declarations. The runner exports the admitted
  report/executor paths and SHA256 values into the Isaac process; the preflight
  independently reloads and revalidates the complete negative gate, records
  those bindings in its result, and includes the executor in its source
  manifest. For a seed43/44 negative it also reloads, rehashes, and requires
  the exact sequential seed42[/43] positive chain, preventing selective use of
  a later negative report.

No task registration, training entry point, candidate-specific allocation,
simulation, or result exists for this candidate. Syntax-only inspection is not
a preflight pass.

The conditional watcher intentionally stops after writing and validating a
passing preflight report. That stop is the activation boundary, not a missing
automatic experiment step. Only after such a pass may a separate change add
the following, in this order:

1. a process-local full/zero/pressure task registration and guarded training
   entry point that require the exact preflight path and SHA256, independently
   reload `overall_pass=true`, and rehash every source recorded by the
   preflight;
2. a latent-specific training wrapper that reuses the unchanged
   `ReferenceOnlyTactileActorCritic`, fixed-group
   `ReferenceOnlyTactileOptimizerDeconfoundedPPO`, official PPO update, frozen
   accepted-SUGAR actor boundary, v3 mounts, and official R15 asset;
3. a retained-compute-node `1 env x 1 update` full-role runtime smoke, followed
   by bitwise-equal full/zero/pressure pre-update checkpoint, source/config,
   optimizer-group, and actor-freeze audits;
4. only after those diagnostics pass, three fresh matched formal roles at 512
   environments and 7098 uninterrupted updates for latent training seed 42,
   followed sequentially by independent latent seeds 43 and 44 only when the
   preceding latent single-seed result passes;
5. a latent-specific paired evaluator that records the hidden dynamics tuple
   for audit without exposing it to the actor, uses evaluation seed 42, keeps
   initial state and tuple indices exact across roles/interventions, and tests
   both the predeclared held-out latent distribution and the existing
   six-condition comparison suite; and
6. a distinct latent source manifest, identity/weight audit, single-seed
   executor, and three-seed executor. Latent reports may not be analyzed by
   relabeling current-distribution outputs or pooled with them.

The optimizer-clean seed that admits the preflight selects whether this branch
may be considered; it does not replace the latent branch's independent
42/43/44 training-seed sequence. Until the preflight passes, the guarded task
registration, training entry, runtime smoke, formal allocation, and evaluator
above remain intentionally absent. This prevents an unvalidated physical
configuration from becoming an executable RL branch merely because a source
file imports.

An attempted compute-node **no-App** import stopped in the standard IsaacLab
import chain at `ModuleNotFoundError: omni`; candidate config construction was
therefore never reached. This records the normal AppLauncher bootstrap
requirement, not a pass or a candidate-runtime failure, and is preserved at
`experiments/sugar_reproduction/logs/20260716_latent_contact_dynamics_static_import.log`.
Before activation, a retained compute allocation must still verify config
construction, material binding/combine modes, exact CarryBox mass/inertia/COM
readback, exact object and four-palm-interface static/dynamic material plus
zero-restitution
readback, matched per-environment TacSL friction, partial-environment sensor
refresh, one-pulse-per-episode behavior, actor-side invariance, and spatial
pressure/shear exposure. The preflight now records the resolved palm shape IDs
and checks these quantities independently of the event's own fail-closed
readback. During live exposure it additionally checks every environment's
actual taxel shear against that environment's `mu_dynamic * normal_force`
Coulomb cap and records the largest excess. Its USD audit traverses instance
proxies, resolves inherited
physics-purpose materials, and requires at least one collision shape under
each of the five declared roots, preventing a missing R15 binding from being
hidden by valid box or rubber-hand records. Any failure keeps the branch
blocked and cannot be bypassed with proxy observations or a reconstructed
sensor.

The pulse admission gate checks more than the event counter: for every
reference-contact-eligible environment, the first world-frame linear-velocity
delta must equal its fixed stratified tuple, angular velocity must remain
unchanged, the event audit state must report the same vector, and a second call
must be exactly idempotent.

The guarded runner also fixes the already validated v3 mount offsets
(`-0.004606,-0.041890,0.005119` left and
`-0.005480,0.063320,0.025027` right) and requires the official R15 USD SHA256
`92139f53c8cff8d70ee7668dddc4912b6b15b549cf2eaacf0f85e635ae93aa43`.
The preflight independently records those inputs and requires the corrected
`33.351139 kg` tactile G1, both `1e-6 kg` virtual tips, 29 actions, 500 taxels
per hand, 50 Hz force updates, and the existing normal/tangential stiffness
contract. This prevents the fallback from silently validating an older mount
or a different sensor asset/configuration. Its source manifest covers not only
the isolated candidate modules but also the guarded runner/watcher, local
TacSL sensor, official-R15 G1 spawner, SDF object, reference-only observation
boundary, and inherited tactile/reference environment configs.

## Hypothesis

The active reference-only actor removes actual object state and already trains
across an official SUGAR startup distribution of object mass and friction.
However, each parallel environment keeps that startup draw, the Refiner does
not train with object-COM offsets or contact-phase impulses, and TacSL shear
uses a fixed sensor-level friction coefficient. Held-out COM and impulse tests
can therefore ask the policy to use a pressure/shear pattern whose corrective
meaning was not represented during learning. The follow-up hypothesis is
narrower and causal:

> When payload/contact dynamics are hidden from the actor and contact-phase
> load redistribution varies during training, taxel-resolved normal pressure
> and signed two-axis shear allow the policy to infer slip early enough to
> improve carrying success over an otherwise identical zero-tactile policy.

The branch must learn this mapping online from the official TacSL signal. It
may not receive sampled dynamics, object state, contact reports, reward terms,
or penetration flags at the actor boundary.

## Fixed implementation boundary

Keep all of the following from the accepted reference-only branch:

- official SUGAR CarryBox code, data, robot/object descriptions, checkpoint,
  PPO, reward, termination, and stage order;
- official IsaacLab v2.3.2 TacSL R15 dual-palm sensor path, locked mounts,
  corrected robot/tip masses, and separate `20 x 25` maps per hand;
- reference-plan actor observation with no actual object pose/orientation or
  velocity, and a privileged critic;
- frozen accepted SUGAR actor tensors, trainable spatial tactile encoder and
  only the declared tactile columns of the actor input layer;
- full, zero, and pressure-only roles initialized from bitwise-identical model
  states and trained with identical environments, updates, reference sampling,
  rewards, and terminations;
- the fixed critic/adapter parameter groups and learning rates in
  `DOCS/sugar_tactile_optimizer_deconfounding_protocol.md`; the role-dependent
  shared actor-KL/critic learning rate may not be reused.

Only official IsaacLab/SUGAR configuration, event, observation, and adapter
glue may be added. No binary contact, body-wrench proxy, hand-written taxel
array, toy controller, or simplified model may substitute for TacSL.

## Audited starting distribution

The current `BaseEventCfg` and all three serialized seed42 `env.yaml` files
establish the actual starting point:

- object mass scale is already sampled log-uniformly from `0.5` to `2.0`;
- object static and dynamic friction are each sampled from `0.2` to `0.8` using
  64 material buckets;
- both terms use event mode `startup`, so the draw varies across the 512
  parallel environments but is not resampled on every episode reset;
- the official config does not set `make_consistent=True`, so its independently
  sampled dynamic coefficient is not guaranteed to be below static friction;
- both R15 force fields use fixed `friction_coefficient=2.0` and
  `tangential_stiffness=0.1`; the official force calculation reads those scalar
  config values directly when applying the Coulomb cap;
- no Refiner startup/reset event currently randomizes object COM or applies a
  contact-phase lateral impulse.

These facts mean the active branch already tests hidden mass/friction learning.
They also expose a physical-coherence limitation, not a reason to invalidate or
retrofit the frozen run: PhysX material draws and the TacSL shear coefficient
are not represented by one serialized per-environment interface parameter.

## Follow-up contact-dynamics distribution

The first follow-up should preserve the current mass `0.5–2.0` and object
friction `0.2–0.8` envelopes for comparability, then add the missing
contact-phase variation:

- local-Y COM offset: `-0.04 m` to `+0.04 m`, stratified across environments;
- contact-phase lateral impulse: randomized sign/direction, application time,
  and magnitude up to the existing `0.8 m/s` evaluation pulse;
- physically consistent material buckets with dynamic friction no greater than
  static friction;
- an audited decision, made from a reset-throughput diagnostic before formal
  training, between per-reset material/mass reassignment and a fixed stratified
  512-environment startup distribution.

Exact distributions and event frequency must be locked in a source/config
manifest before the first formal update. The training held-out boundary remains
mass scale `2.5`, friction `0.1/0.05`, COM `+0.05 m`, and unseen impulse timing
or direction; those values may not leak into the training RNG stream.

The rigid-body interface friction and TacSL frictional/shear configuration must
be made coherent or their distinction must be explicitly modelled and logged.
The current official sensor API exposes only a scalar config coefficient. A
minimal auditable extension may add a per-environment force-field coefficient
or setter while retaining the exact official TacSL force equation. If this
cannot be done without reconstructing the sensor, record the compatibility
blocker and use a predeclared finite set of coherent scene variants; do not
claim coherent friction randomization while changing only PhysX.

The critic may receive the sampled tuple only as an explicitly serialized
asymmetric-training ablation shared by all three roles. The preferred first run
keeps the critic observation identical to the current reference-only critic so
that the only experimental change is the environment distribution.

## Matched roles and causal controls

- `full`: live normal pressure plus signed two-axis shear;
- `zero`: both official sensors continue to execute, but all actor taxels are
  zero at the same policy boundary;
- `pressure_only`: live normal pressure with both shear axes zeroed;
- same-policy interventions at evaluation: zero, pressure-only, shear-only,
  swapped hands, wrong-environment tactile, one-step lag, and live.

All roles must have equal formal budgets. Any capacity-matched contact-label or
body-force branch remains a separately labelled non-tactile proxy control and
cannot replace zero or pressure-only.

## Training and evaluation gates

Before formal training, require compute-node diagnostics for:

1. exact distribution readback and actor-side invariance to the sampled hidden
   tuple when tactile input is zero;
2. coherent PhysX/TacSL friction plus mass/inertia/COM application after reset,
   including explicit static/dynamic readback on the object and every selected
   palm-interface shape;
3. nonzero, spatially varying pressure/shear exposure across the distribution;
4. bitwise-identical full/zero/pressure pre-update checkpoints and matching
   source/config manifests;
5. the same actor-freeze audit used by the active branch.

Formal training remains 512 environments and 7098 uninterrupted updates for
each role and policy-training seed. Final evaluation fixes evaluation seed 42,
uses paired initial states, and contains:

- nominal accepted-SUGAR dynamics for noninferiority;
- in-distribution hidden-dynamics tuples not reused from training RNG streams;
- held-out mass/friction/COM combinations beyond the training envelope where
  physically stable;
- unseen impulse timing/direction and partial sensor failure;
- the existing six-condition suite for direct comparison, without merging its
  statistics into the new distribution-level primary endpoint.

The primary single-seed claim requires all existing provenance, exposure,
action-dependence, actor-freeze, nominal-noninferiority, matched pressure, and
complete-intervention gates, plus both:

- full-live versus matched-zero success-gain 95% paired-bootstrap lower bound
  greater than zero on held-out latent dynamics;
- full-policy live versus zero-intervention success-gain 95% paired-bootstrap
  lower bound greater than zero on the same held-out tuples.

Failure reduction, episode progress, reward, slip distance, pressure balance,
and shear ratio remain diagnostics and cannot replace the success endpoint. A
research-level advantage claim still requires three independent matched
training seeds, a positive effect in every seed, and positive seed-bootstrap
lower bounds for both success comparisons.

## Failure interpretation

- No taxel exposure: geometry/contact-distribution failure; do not tune PPO to
  hide it.
- Exposure but no action dependence: optimization/adapter failure.
- Action dependence but worse success: learned tactile response is harmful;
  inspect slip/contact-phase decomposition before changing architecture.
- Full ties zero while pressure beats zero: pressure advantage only; do not
  claim shear advantage.
- Full beats zero but not same-policy zeroing: training/control confound; no
  tactile advantage claim.
- Any provenance or physical-coherence gate failure: invalid run, not a
  negative or positive performance result.
