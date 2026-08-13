# SUGAR-SMP-ICM: Tactile Curiosity for Alternative CarryBox Strategies

## Research Question

Can a G1 policy retain the natural whole-body action manifold learned from
official SUGAR CarryBox motions, while using tactile slip evidence and
curiosity-driven exploration to discover materially different ways to lift and
carry a box when the nominal symmetric two-palm side clamp slips or cannot lift
the payload?

The desired behavior is not a small correction to the original SUGAR pose. A
heavy or low-friction box should be allowed to trigger a different whole-body
strategy, for example a lower squat and hold height, asymmetric hand heights,
one hand supporting the bottom while the other stabilizes a side, a controlled
regrasp, or safe torso/forearm support. The policy must not receive the hidden
mass, friction, or center of mass as actor input; it should infer that the
current attempt is failing from proprioception and real spatial tactile
history.

## Frozen Tactile Meaning (2026-07-29)

The tactile observation is no longer allowed to mean “some force near the
hand.” It is the exact anatomical sensor defined by
`DOCS/sugar_whole_hand_tactile_non_degradation_standard_20260729.md`:

- 27 physically load-bearing elastomer patches per hand;
- twelve palm patches plus proximal/middle/distal patches on thumb, index,
  middle, ring, and little finger;
- one real `20 x 25` normal-force map and one channel-last
  `20 x 25 x 2` signed-shear map per patch;
- one fixed, symmetric official R15 RGB/depth palm module per hand; and
- raw full-resolution temporal history retained before any learned policy
  encoder.

This is the information from which the future causal slip belief must be
learned. Contact labels, hand wrenches, object state, merged hull heat maps,
projected rigid contacts, shadow sensors, generated taxels, and per-frame
replay are not alternate representations of the same idea; they are different
and inadmissible inputs.

The tactile chain must first prove that its physical palm/finger patches are
the surfaces supporting the box, that their spatial and temporal fields agree
with independent same-frame contact audits, that calibrated summed forces
close gravity and object dynamics, and that the two real R15 optical streams
agree with the corresponding palm force footprints. The nominal full-lift
video must then show the world, left hand, right hand, palm optics, and force
balance in separately inspectable synchronized H.264 videos. Until all gates
and explicit human review pass, tactile policy use, slip detection, internal
reward, SMP/ICM integration, recovery, and alternative-strategy claims remain
blocked.

The earlier `0.729171 m` 660-step one-hull run is retained only as a movement
baseline. Its merged field is not evidence for the tactile premise of this
idea, and no former tactile/slip checkpoint is inherited.

## Why SMP Fits, and What It Does Not Solve by Itself

[SMP](https://yxmu.foo/smp-page/) trains a motion diffusion model once and
freezes it as a score-matching motion prior. During downstream policy training,
score distillation sampling (SDS) compares noise added to a simulated motion
window with noise predicted by the diffusion model. The resulting residual is
converted into a dense motion-naturalness reward. SMP does not require the
policy to track one reference frame-by-frame, so a task reward can select a
new motion while the SDS reward keeps it near a learned motion manifold.

The official SMP implementation remains PPO. Curiosity, however, must not be
reduced to another hand-authored outcome term inside a PPO reward table. The
original [Intrinsic Curiosity Module
(ICM)](https://arxiv.org/abs/1705.05363) is a second learned subsystem: an
inverse dynamics model learns a controllable feature space, and a forward
dynamics model tries to predict the next feature from the current feature and
the executed action. Its prediction error is the intrinsic signal for a new,
not-yet-understood discovery. The ICM model is updated jointly from the
agent's own experience, including experience collected without any task
success reward.

This project therefore separates four mechanisms:

1. a frozen official-architecture SMP/SDS prior reward;
2. a learned ICM intrinsic discovery signal;
3. goal, slip, repeated-failed-strategy, and safety objectives/constraints; and
4. a policy optimizer that consumes the resulting experience.

PPO may remain the numerical optimizer required by official SMP, just as the
original ICM paper used A3C as its policy optimizer. That does not make ICM a
“PPO reward trick”: the important research component is the independently
learned forward/inverse knowledge model and the exploration distribution it
creates. A run with only novelty-shaped hand-written rewards is not ICM, and a
plain PPO run with more entropy is not curiosity.

SMP alone is not sufficient for the requested diversity. The paper explicitly
notes mode collapse as a limitation. The accepted SUGAR CarryBox data has a
large dominant contact-geometry region but also whole distinct motion
families; one global “original clamp” center rejects motions 43–47 completely.
A strong unmodified SUGAR interaction prior could therefore reinforce common
behavior while a fixed pose gate could incorrectly punish valid official
modes. The proposed method needs an explicit diversity mechanism and a
phase/body-part prior schedule that loosens hand-object geometry after a
failed attempt while retaining whole-body naturalness.

## Faithful Source Boundary

The implementation must use:

- official MimicKit source at commit
  `2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69`;
- its official `TinyMDMModel`, EMA, noise scheduler, ensemble SDS calculation,
  adaptive per-diffusion-step normalization, and GSI implementation where the
  state representation permits it;
- the official SUGAR G1, CarryBox assets, 29-DoF joint-position action, accepted
  `model_10000.pt`, and 922 accepted Refiner rollouts; and
- the official IsaacLab/TacSL dual-palm R15 spatial pressure/shear path.

No locally invented small diffusion model, discriminator, VAE, or “SMP-like”
scalar score may stand in for these components. Adapter code may convert SUGAR
trajectories and live IsaacLab state into the exact input contract required by
official TinyMDM. If the official architecture or weights cannot be made
compatible, that is a blocker, not permission to substitute a toy model.

The public MimicKit repository does not currently contain the paper's Object
Carry environment, its human-object interaction dataset, or an Object Carry
checkpoint. Its documented pretrained LaFAN1 prior also targets a different
character and state dimension. It cannot be applied directly to the G1 policy.
The first valid SUGAR prior must therefore be trained with the official
TinyMDM training code and architecture on exported SUGAR G1+box motion
features.

## SUGAR-to-SMP Motion Contract

SUGAR and SMP are action-compatible at the controller boundary: both issue
target joint positions to PD-controlled joints. SUGAR's action remains the
original 29-DoF `JointPositionActionCfg`; SDS never directly outputs actions.

Each prior window will be built at one declared control frequency and contain,
in a pelvis/anchor local frame:

- root linear and angular velocity;
- local 6D joint rotations for the G1 joints;
- local 3D positions of both hands and both feet;
- box local position and 6D rotation;
- box local linear and angular velocity; and
- an explicit validity mask for padding, if any.

This is a joint robot-object prior, following the SMP paper's human-object
interaction formulation. Tactile fields, hidden mass/friction/COM, task goal,
reward, and contact-label proxies are not part of the motion-prior training
target. The actor may use tactile and goal observations, but the frozen prior
evaluates motion naturalness only.

The accepted processed SUGAR dataset already preserves joint state, body pose
and velocity, and object pose and velocity. A compute-node schema audit must
prove exact body ordering, quaternion convention, dimensions, frequency, and
window coverage before prior training.

## Why the Existing Refiner Task Must Change

The current SUGAR Refiner rewards joint pose, anchor pose, every tracked body,
object pose/velocity, and object-to-body transforms against a specific
reference trajectory. It also terminates when the wrists, anchor, or object
deviate roughly 0.3 m from that reference. Those terms make a bottom-support
grasp or a lower squat a failure even if the box is lifted successfully.

The exploration task must therefore preserve SUGAR's robot, actuator scales,
box, physics, simulation frequency, reset safety, and regularization, but
replace exact trajectory tracking with a goal-based state machine:

1. approach the box;
2. establish a controlled contact;
3. perform a micro-lift/probe;
4. lift and stabilize;
5. transport toward a local goal;
6. recover or regrasp after detected slip; and
7. terminate on task success, unrecoverable drop, unsafe fall, or time limit.

SUGAR reference poses may seed reset states and train the prior, but they may
not remain a hard per-frame target during alternative-strategy discovery.

## Direct Tactile Slip Detection

Slip detection must use the time series of the official per-hand R15 fields:

- non-negative taxel normal load/pressure `p[t, i, j]`;
- signed two-axis shear `s[t, i, j, 2]`;
- active contact footprint and pressure-weighted centroid;
- integrated normal load `F_n` and tangential load `F_t`;
- friction utilization `rho = ||F_t|| / (mu * F_n + eps)`;
- footprint overlap and centroid motion between consecutive tactile frames;
- pressure loss or load transfer while shear and patch motion persist; and
- a persistence/hysteresis state separating transient noise from incipient and
  gross slip.

The online detector exposed to the actor and reward must be computed from
these tactile fields and their history. A simulator-state oracle may calculate
hand-box relative tangential velocity at active contacts only to label and
calibrate detector precision, recall, latency, and false positives. The oracle
must not be passed to the actor or called tactile.

Intentional release and regrasp must not be punished as uncontrolled slip. A
release is treated as a regrasp transition only when it is followed by a new
stable contact and maintains support or improves task progress within a fixed
window. Otherwise contact loss, downward box motion, or persistent tactile
sliding is penalized.

All results remain **high-fidelity simulated tactile** until the detector is
calibrated against a physical GelSight-class sensor.

### Current Slip Evidence and Boundary (2026-07-23)

The tactile-only estimator and exact evaluation oracle are implemented. The
deployed path reads only direct R15 pressure/shear history and reset mask; the
exact SDF contact normal and contact-conditioned relative tangential velocity
remain calibration/evaluation-only. Release and regrasp are exposed as contact-
topology transitions rather than zero-slip negatives.

The frozen v9 and v13 tests remain important negative evidence. v9 exposed
right-palm recall and latency failure. v13 passed its expanded development
groups but failed the single-use motion-45 test on severity precision, a short
right-hand contact episode, and delayed regrasp state. Motions 2, 6, and 45
remain opened development evidence and are permanently excluded from later
held-out claims.

The subsequent v14/v15 development work did not retune those opened tests. It
replaced the global threshold family with a causal spatial-temporal model that
preserves taxel geometry and separates two questions:

1. whether a direct sliding frame contains sufficient raw tactile evidence to
   be observable; and
2. how the frozen detector classifies observable stick, incipient, gross,
   release, and regrasp transitions.

The v16 protocol locked that distinction, the v15 model and thresholds
`0.20/0.825`, the direct-contact floor
`6.463075578722055e-6` (`0.25` of the development reference), the new
`normalize(2x+5y)` direction, stress seeds, source-selection manifest, code
hashes, and exact gates before opening any detector output from the new source
motions. A detector-blind scan selected left-palm motion 43 and right-palm
motion 18; neither was used by v9/v13. Six new official-TacSL controlled probes
cover nominal, `3x` mass, lower friction, four load states, stick, incipient,
gross, release, regrasp, no-contact, noise, one-step latency, dead taxels, and
combined stress. Every probe passed its 17 direct-sensor/data checks and
retains native world RGB plus `20 x 25` pressure and signed 2-D shear.

The single-use fresh v16 evaluation **passes Stage E**. Every source,
condition, load group, palm, and all five stress roles pass. Raw observable
sliding-frame coverage is `1.0` for nominal, noise, latency, dead-taxel, and
combined roles; no sliding sequence is unobservable. Slip and gross precision
are `1.0` throughout, incipient-as-gross and negative false-positive rates are
`0`, and no aligned release/regrasp sequence carries false slip. The worst
overall gross recall is `0.9838` under dead taxels; nominal incipient/gross
recall is `0.9943/0.9882`; delayed input gives `0.9947/0.9890` and at most one
wall-clock step of latency. Both selected palms and every locked subgroup pass.

Motions 43 and 18 were opened exactly once and are now permanently unavailable
for retuning or reuse as held-out evidence. The immutable result SHA256 is
`b1d6d2ceed26d8bd1175fdd0cfbd2547f3d944390ff7839f5006c9c252116f66`;
the curated evidence manifest is
`experiments/sugar_reproduction/outputs/final/smp_tactile/stage_e_v16_20260723/CURATION_MANIFEST.json`.
The full protocol/result boundary is recorded in
`DOCS/sugar_direct_tacsl_spatiotemporal_v16_fresh_result_20260723.md`.

This admits the tactile-only slip component for the next policy-interface
stage. It does not establish physical GelSight calibration, real-hardware
tactile, sim-to-real, on-policy ICM exploration, or an alternative carrying
strategy. Slip remains an external observation/objective; it is not the ICM
intrinsic signal.

## Preventing Repeated Two-Hand Side Clamping

The project must not punish every two-hand contact. A valid one-hand-bottom +
one-hand-side strategy can still use both hands. Instead, define a
box-coordinate, viewpoint-invariant `original_clamp_similarity` from:

- both palms contacting opposing lateral faces;
- approximately equal palm heights;
- lateral palm normals pointing toward one another;
- left/right symmetry about the box center;
- similarity to the original SUGAR palm-to-box transforms; and
- lack of bottom, forearm, or torso support.

The penalty becomes active only after the nominal strategy has produced a
persistent slip, failed micro-lift, or repeated no-progress attempt. It then
penalizes returning to the same high-similarity strategy even if the robot
approaches from another world-space angle. A short nominal contact during
approach is allowed.

## Curiosity Rewards New Discoveries, Not Successful Results

Let `o_t` be the non-privileged observation available at time `t`, including
proprioception, previous action, local task context, and the dual-palm spatial
tactile history. The ICM encoder produces `phi(o_t)`. Its inverse model predicts
the actually executed 29-D SUGAR action from `phi(o_t), phi(o_t+1)`, forcing
the feature space to retain changes that the robot can cause or that affect the
robot. Its forward model predicts `phi(o_t+1)` from `phi(o_t), a_t`.

The intrinsic discovery signal is the forward feature-prediction error:

```text
r_icm(t) = eta / 2 * ||f(phi(o_t), a_t) - phi(o_t+1)||^2
```

It is high when the robot encounters a controllable transition it has not yet
learned: a new load transfer, contact topology, regrasp response, squat/contact
combination, or slip/recovery dynamic. As ICM learns that transition, its
prediction error should fall. This learning-dependent decay is what makes the
signal about discovery rather than a permanent bonus for a manually named
result.

The intrinsic signal must **not** be multiplied by lift height, success,
progress, reduced slip, or distance from a hand-authored target strategy. A
failed new experiment can be a real discovery and must still train ICM. Task
success, slip punishment, repeated-failed-strategy punishment, falls, energy,
and safety remain separate external objectives/constraints and are reported
separately.

The original ICM insight also protects against a “noisy tactile television”:
the inverse-dynamics feature objective should suppress sensor variation that
neither influences nor is influenced by the G1 action. This must be tested with
TacSL noise, uncontrolled disturbances, and repeated identical actions. Raw
pixel/taxel prediction error without the inverse model is not accepted as ICM.
The paper's actual robustness experiment keeps a fresh white-noise region in
40% of the observation during training and evaluation. Therefore the primary
test is invariance between independent nuisance realizations in the same sensor
domain, not equality between a clean checkpoint and a newly introduced noisy
domain.

The original code targets discrete actions. The SUGAR adapter must faithfully
extend the inverse head to the continuous 29-D applied joint-target action and
predeclare its likelihood/regression loss. This is an adapter to the original
ICM formulation, not permission to replace ICM with a generic world model.

### Current ICM Evidence and Boundary (2026-07-23)

The first direct-R15 semantics audit has passed without a policy-training or
task-result input. It uses 135 identical transition indices from official
SUGAR motion 6, start frame 129, under nominal physics, 3x payload mass, and
lower `0.4/0.3` static/dynamic friction. The actor/predictor never receives the
mass or friction value. Each condition independently retains 98/79/83 real
direct-contact transitions in the paired contact-union.

After 1,000 nominal ICM updates, mean nominal forward-error signal falls from
`0.138091` to `0.000515084` (`0.00373x`). First exposure to heavy and
lower-friction dynamics restores error to `10.46x` and `4.64x` the familiar
nominal level. Five hundred updates on each new condition reduce them to
`0.127x` and `0.261x` their respective first-exposure values. Shuffling the
next transition or executed action raises error to `467.9x` and `35.8x` the
familiar level. The transition API contains observation, direct tactile
history, executed action, next observation, and a reset-validity mask only;
there is no task reward, success, lift, slip result, mass, or friction field.

This result establishes the learned familiar/new/repeat semantics needed for
curiosity. It does **not** show that a policy has explored because of ICM,
detected slip, switched grasp, lifted a held-out payload, or found an
alternative posture. Those remain downstream gates.

A second paper-aligned nuisance diagnostic uses fresh synthetic direct-R15
noise on every update and four unseen noise holdouts. Full inverse+forward ICM
reduces unseen-noise intrinsic signal by `86.63%` and same-transition
independent-noise feature sensitivity by `57.49%`, while inverse action RMSE
improves by `13.37%`. With the same noise realizations, heavy-mass and
low-friction transitions remain `5.108x` and `2.797x` more novel than nominal.
No result variable enters this diagnostic.

The diagnostic also makes two boundaries explicit. First, its strict
predeclared composite result stays failed because a forward-only ablation
collapses nuisance variation even more while damaging inverse action structure;
lower variation in that degenerate branch is not stronger curiosity. Second,
noisy-only adaptation causes `59.30x` clean-domain signal drift through the
running normalizer. The final tactile noise/latency regime must be present from
ICM training start, or clean/noisy replay must be deliberately mixed. Neither
issue is repaired by task-success gating, because result labels still do not
define discovery.

## Strategy Memory Is a Separate Constraint and Measurement

At every stable contact attempt, construct a strategy descriptor containing:

- contacted box faces for each hand;
- left/right contact height and height asymmetry;
- palm-to-box orientation and symmetry;
- bottom/side/top support indicators;
- forearm/torso support indicators;
- pelvis height, knee flexion, stance width, box hold height;
- tactile load split, footprint locations, and shear utilization; and
- temporal outcome: lift progress, slip duration, drop, recovery, and energy.

Store descriptors of failed attempts in per-episode memory. The descriptor is
used to detect and penalize returning to an actually failed episode-specific
contact geometry, measure strategy coverage, and verify whether ICM actually
discovered new behavior. It does not define `r_icm`. A descriptor-distance
bonus, if studied at all, is a separate count/episodic-novelty ablation and may
not be labeled ICM.

The all-922 audit rules out one universal original-clamp classifier:
frame-weighted acceptance is `0.82384`, but official motions 43–47 have zero
accepted frames. The deployed anti-repeat interface therefore uses a fixed
robust box-frame scale and pairwise similarity to the episode's stored failed
attempts. A ten-frame no-contact lockout prevents one persistent slip from
creating repeated false attempts. Unit tests and a 350-frame direct-R15 replay
pass this interface logic; this remains state-memory evidence, not learned
strategy switching.

Flailing and deliberate falls are controlled by the independent SMP manifold,
safety constraints, terminations, and external penalties—not by erasing ICM's
intrinsic reward whenever the experiment fails. Historical offline
contact-count reweighting is explicit negative evidence and must not be
repeated.

The actor needs temporal memory or an explicit non-privileged attempt-history
observation so that it can condition the next action on “the previous clamp
slipped.” ICM also needs transition-aligned observation/action pairs and its
own optimizer/checkpoint state. Hidden physical parameters remain critic-only
diagnostics.

## C3 Causal Failure-Induction Boundary

The original bilateral clamp must exist as physics-integrated behavior before
we are allowed to induce or name its failure. The frozen motion-45 direct-TacSL
source contains one six-frame bilateral run at source frames 102--107. C3 first
restores frame 102 exactly once and then advances only through Isaac physics
and the recorded official SUGAR actor actions. Rewriting the robot or box state
on later frames would be state replay, not a sustained clamp.

The fixed C3-P0 gate requires all 20 coherent mass/friction/COM environments to
show at least three consecutive same-episode observations with valid direct
R15 pressure and signed shear on both hands. It performs no policy inference
or update. Only after that gate passes may C3-P1 cross the five tactile roles
with no-op, mass, friction, and combined failure interventions.

The intervention is external curriculum, not discovery. Clamp establishment,
the intervention transition, and fixed settling steps stay out of PPO and
original-ICM storage so injected dynamics surprise cannot inflate `r_icm`.
Failure still needs the locked direct-TacSL slip and strategy-memory closure;
C4 recovery still needs release/re-arm, a later changed-topology attempt,
`>=0.10 m` lift, and post-lift goal stability.

Authoritative protocol:
`DOCS/sugar_stage_i_c3_bilateral_failure_protocol_20260723.md`.

C3-P0 now passes this physical boundary in all 20 coherent environments. The
observed run is three-to-four frames (`60--80 ms`), so it is enough to place a
causal intervention after observation 2 but not enough to claim a durable
carry. C3-P1 is correspondingly fixed to no-op, current mass/inertia `3x`,
matched PhysX/TacSL friction `0.15/0.10`, and combined columns. The result
remains zero learning and zero recovery:
`DOCS/sugar_stage_i_c3_p0_bilateral_clamp_result_20260723.md`.

C3-P1 has now run once without outcome-driven retuning. The strict failure
counts for `[no-op, mass/inertia 3x, matched low friction, combined]` are
`[0, 1, 0, 1]`, so the full four-column coverage claim is a no-result.
Mass-only environment 1 and combined environment 3 independently satisfy the
unchanged pre-clamp, post-event slip, and failure-closure predicates and are
retained as fixed C4 starting evidence. Low friction alone remains an explicit
coverage gap. There is still no recovery, alternative strategy, policy
learning, predictor fit, or ICM result:
`DOCS/sugar_stage_i_c3_p1_bilateral_failure_result_20260724.md`.

## C4-P0 Same-Episode Handoff Result

C4-P0 now closes the first causal handoff boundary. V1 and V2 show that a
strict tactile failure cannot be replicated merely by copying rigid state,
properties, sensor role, and seed into new parallel slots. V4 therefore
replays the exact original C3 twenty-environment identity matrix. It
reconstructs strict failure only in environments 1 and 3, preserves the full
original observation-7 failed-strategy memory pattern, excludes all
`7 x 20` prelude transitions from learned storage, and completes one official
SUGAR/RSL-PPO + SMP + original-ICM update from global 73 to 74.

This proves the mechanism can begin learned post-failure exploration without a
reset or proxy tactile substitution. It does not prove adaptive carrying.
Only environment 3 releases and re-arms during the 24-step endpoint; neither
eligible branch makes a later changed-topology contact and successful lift.
Strict recovery, successful alternative strategy, and predictor-positive
counts remain zero. The next scientific experiment is a fixed-budget C4-P1
continuation, not a relaxation of the recovery predicate.

Authoritative result:
`DOCS/sugar_stage_i_c4_p0_same_episode_handoff_result_20260724.md`.

The C4-P1 boundary is now frozen before execution: resume learned/RNG state
only from curated global 74, recreate the exact C3 physical prelude, and run
16 complete 24-step updates to global 90. The fixed 384-step post-failure
horizon runs regardless of recovery; checkpoints at local `[1,4,8,12,16]`
are audit artifacts and cannot be selected from behavior. Only environments
`[1,3]` remain recovery-eligible, original ICM remains ungated by outcomes,
and a numeric recovery candidate still requires independent rendered
RGB/direct-TacSL/topology verification.

Authoritative frozen protocol:
`DOCS/sugar_stage_i_c4_p1_fixed_postfailure_exploration_protocol_20260724.md`.

## C4-P1 Fixed-Endpoint Result

The complete global-74 to global-90 endpoint passes every engineering,
accounting, numerical, independent-audit, and rendered-evidence gate, but it
is a scientific no-result. Environment 3 re-arms at stored step 9;
environment 1 does not. Neither makes a later same-episode direct-TacSL
contact, reaches attempt index 2, changes topology, or recovers.

The decisive new finding is that the recovery-eligible episode lasts only nine
stored policy steps in environment 1 and eleven in environment 3. The rest of
the fixed 384-step horizon is after reset and therefore cannot recover the
original C3 failure. More global updates are not the next solution. C4-P2 must
first capture the exact raw termination term and test whether a bounded,
failure-aware recovery grace period is required, while keeping safety
constraints and original ICM semantics unchanged.

## C4-P2 Negative and C4-P3 Design Consequence

C4-P2 falsifies the assumption that a fresh Isaac/PhysX/TacSL restart can
bitwise replay C4-P1: 31/57 shared scientific fields differ. Its pre-reset
replication still shows the same near-immediate reset topology and a sole
`dropped_after_lift` cause for envs 1/3, while fall and workspace conditions
remain safe. This is replicated blocker evidence, not exact historical raw
cause.

The resulting idea refinement is important: a policy cannot discover a
release/regrasp strategy if episode design treats the recoverable release/drop
state itself as terminal. C4-P3 therefore adds no new reward and does not
change ICM. It gives only the original failed branches a fixed 256-step window
where raw drop is observed but not effective, with fall/workspace/timeout
fully active. The scientific question becomes whether additional coherent
control time creates a later direct-contact attempt; success remains a
separate, much stricter gate.

C4-P3 answers that question negatively. The window extends the original
episodes by 27/41 valid transitions, but both branches only release and then
fall; neither recontacts. This changes the main hypothesis again: the immediate
drop reset was necessary to fix, but alternative strategy discovery is now
limited by stable recovery-motion support, not merely episode length. The
project needs a faithful official-compatible release-to-recontact motion
source or multi-strategy demonstration coverage. Curiosity alone should not be
redefined into a directed recovery reward to hide that data/control gap.

Authoritative result:
`DOCS/sugar_stage_i_c4_p1_fixed_postfailure_exploration_result_20260724.md`.
The 101 MiB final package retains all sixteen event shards, all native RGB and
spatial TacSL evidence, and only the global-90 checkpoint. Its curation
manifest is
`120ea17140b78b819db852c93f65949e79e857197fe6d78afabec7b3ab8ecf8e`;
all twelve curated verification checks pass.

## Official Source Search and Motion-37 Falsification

An exhaustive first source screen does not yet close that data gap. Across all
922 accepted processed CarryBox trajectories, only two proxy
release/recontact sequences appear and neither lifts by `0.10 m`. Across the
77 frozen direct-TacSL train motions, motion 37 is the sole stored sequence
that changes from right-only to left-only contact, later lifts, and reaches
official trajectory completion.

That stored ordering is not a reproducible positive. In the preregistered
fresh four-environment replay, predeclared env 0 has no stable right-hand
precursor. Env 3 comes closest, but a transient right-hand touch inside the
required ten-observation release invalidates the transition; selecting env 3
after seeing the result is forbidden. All four executions still complete the
official reference, so the negative concerns stable contact-transition
support rather than the accepted SUGAR baseline.

The research consequence is stronger than “mine more nominal rollouts.”
Neither ICM nor a learned semantic reward may promote an accidental contact
ordering into a demonstrated strategy. Before fitting a recovery-conditioned
predictor, the project needs a faithful and repeatable source containing an
intentional hand switch, bottom support, or release-to-recontact event. The
next source gate therefore audits the original human-video bank and any other
official-compatible demonstration source; if those assets are not released or
lack the behavior, that limitation is recorded rather than filled with a toy
expert.

Authoritative results:
`DOCS/sugar_official_release_recontact_source_audit_result_20260724.md` and
`DOCS/sugar_motion37_release_recontact_replay_result_20260724.md`.

The follow-on source search also closes two public paths without a positive.
SUGAR does not publicly release its original RGB-D human-video bank or
video-to-motion pipeline, and its public CarryBox robot videos show only the
nominal bilateral side grasp. All 13 publicly accessible official OmniContact
CarryBox trajectories likewise contain only bilateral/no-hand-contact labels.
Motion 12 repeats five nominal bilateral carry cycles; none of its four
release/recontact boundaries changes active hands or grasp geometry. These
sources cannot teach the proposed reward model that hand switching or bottom
support is semantically valid, and OmniContact's binary labels are not tactile.

Results:
`DOCS/sugar_public_human_video_source_audit_result_20260724.md` and
`DOCS/sugar_omnicontact_source_compatibility_result_20260724.md`.

The subsequent official GRAIL boxed-pickup G0 audit also adds no candidate,
but for a different reason. All 155 fixed public metadata files rehash and
decode safely, yet their schema exposes neither time-series hand contact nor
trajectory length/source identity. The frozen protocol therefore stops before
robot/object/video expansion. This is a provenance limitation, not a
behavioral claim about unseen GRAIL trajectories, and it authorizes neither
SUGAR/TacSL replay nor a reward-predictor row.

Result:
`DOCS/sugar_grail_boxed_pickup_source_compatibility_result_20260724.md`.
The independently verified final package has curation manifest
`94cf8387d1d64bd5fba5e26282a3f6ef2bf5ad8c481a727ff8aa347e34f884fa`.

## Prior Scheduling for Alternative Poses

The initial nominal approach may use the full robot-object SMP reward. After a
failed clamp, the method will compare three declared prior variants:

1. full joint robot-object prior;
2. task-clipped prior, following SMP Object Carry's style-reward clipping; and
3. body-part/feature-composed prior that keeps lower-body and torso motion on
   the SUGAR manifold while reducing the weight of original palm-box relative
   geometry during regrasp.

Only official MimicKit diffusion predictions and official SMP reward equations
may supply these prior scores. Feature masking/composition is an auditable
adapter around the official model, not a new learned substitute.

If the SUGAR-only prior makes every bottom-support candidate irrecoverably
out-of-distribution, the next allowed expansion is additional **official** G1
motion/interaction data or a declared composition of official priors. Hand
authored fake demonstrations may be used only as explicitly labeled
kinematic/safety diagnostics, never as proof of learned SMP behavior.

## Training and Evaluation Contract

Actor-visible observations:

- G1 proprioception and previous action;
- local task goal and phase;
- root-relative box pose/velocity only if the selected task contract declares
  it as vision/state input;
- dual-palm pressure and signed shear history;
- tactile slip belief and non-privileged attempt-history state.

Actor-hidden variables:

- exact mass, friction coefficient, COM offset, randomization seed;
- simulator slip oracle and future reference trajectory;
- privileged success/failure labels beyond observations physically available
  at that step.

The critic may use hidden physics during asymmetric training, but every such
field must be listed and ablated. Training randomizes mass, friction, COM,
geometry, and disturbances. Held-out evaluation uses disjoint combinations,
including payloads and friction levels outside the nominal SUGAR range.

Required controls:

- frozen accepted SUGAR reference-tracking policy;
- goal-based policy optimization without an SMP prior or ICM;
- goal-based policy optimization + official SMP only;
- pure ICM discovery with task-success reward disabled;
- SMP + ICM discovery with task-success reward disabled;
- SMP + tactile slip penalty without ICM;
- SMP + ICM with tactile zeroed in the ICM/policy inputs;
- full SMP + ICM + tactile slip + anti-repeat constraint;
- full method with failed-strategy repeat penalty disabled; and
- full method with simulator oracle used only as a detector upper-bound,
  clearly labeled non-deployable.

## Success Criteria

The research claim requires more than task reward improvement. On held-out
mass/friction/COM conditions, the full method must show:

- higher lift-and-carry success than the frozen SUGAR control and matched SMP
  without ICM after the discovery policy is evaluated on the task;
- lower gross-slip duration and drop rate;
- a lower rate of returning to episode-specific failed contact strategies;
- statistically supported strategy switching after detected slip;
- independently verified ICM novelty: high forward error on first encounters,
  declining error after the same controllable transition is learned, and no
  task-outcome term in the intrinsic calculation;
- at least one reproducible non-nominal successful strategy cluster, verified
  from contact topology and rendered motion rather than a subjective label;
- no increase in unsafe falls beyond the declared bound; and
- synchronized world, pressure, shear, slip-belief, strategy-descriptor, and
  reward-component visualizations for every claimed behavior.

Until those gates pass, the honest status is “SMP/SUGAR migration and
alternative-strategy exploration in progress,” not learned adaptive carrying.

## Current Execution Boundary (2026-07-23)

The first real Stage-H one-update integration diagnostic passes all 25
interface checks. A four-environment, 32-step rollout uses direct dual-R15
signals, live `10 x 216` windows, the frozen admitted TinyMDM/ESM prior, and the
independent original-ICM learner. This closes only the one-update interface
gate.

The predeclared strict-MimicKit H1 then **fails**. Across eight fixed updates,
every update exceeds the clip gate; update 1 has clip fraction `0.9703125` and
epoch approximate-KL means up to `79.3931`. The action-bound loss becomes
nonzero at update 2 because the adapter omitted MimicKit's normalized-action
to environment-action normalizer and sent policy space directly to SUGAR's
unbounded policy units. Semantic invariants, frozen TinyMDM, original ICM, and
checkpoints remain correct, so this is a policy/action-boundary failure rather
than a failed-discovery relabeling.

The strict MimicKit PPO branch is rejected as the primary optimizer and its H2
is canceled. This does not remove SMP: official TinyMDM/EMA, ESM/SDS, and their
normalization remain the motion prior. The next named preflight uses official
SUGAR/RSL PPO on native SUGAR actions while original ICM remains an independent
learner. A MimicKit-action-normalized bridge is only a future ablation. No
multi-update exploration, low squat, bottom support, regrasp, or alternative
successful carrying strategy has yet been learned.

That native path has now been resolved through a predeclared evidence chain.
The untouched BasePPO `1e-3` start fails HN1; an adaptive floor-LR run exposes
non-causal tactile bias amplification; freezing exactly 14 additive tactile
encoder biases restores `zero taxel -> zero feature`; and the upstream adaptive
schedule then overshoots. The final named contract keeps that causal repair and
uses upstream PPO's supported fixed schedule at `1e-5`.

ZF0 and ZF1 pass. ZF1 starts from scratch with 20 real environments, completes
eight updates and 160 upstream Adam steps, retains direct TacSL in every
rollout, keeps the TinyMDM frozen and ICM independent, reloads checkpoints
exactly, and preserves exact zero-taxel actor/critic features. Its maximum
update clip fraction is `0.0141667`; maximum epoch clip is `0.05`; maximum
epoch sampled approximate-KL mean is `0.00398737`. This admits the nominal
policy optimizer and capacity preflight, not formal exploration.

The predeclared five-role tactile gate is now resolved. Locked H2 v1 completed
but was rejected because it exposed two scientifically invalid
`nonzero-every-rollout` predicates and an off-by-one initialization-cache
predicate. That negative remains preserved. Separately locked H2R1 keeps the
same learning and stress contracts, records raw TacSL provenance before noise,
accepts physically valid exact-zero no-contact windows, and locks the actual
shared-consumer topology.

H2R1 starts nominal, fresh-noise, one-step-latency, five-percent-dead-taxel,
and combined roles before the first ICM bootstrap transition. It passes eight
updates, 160 upstream Adam steps, 3,708 valid original-ICM transitions, raw
TacSL provenance in every role, exact `193/385/192`
generated-frame/apply-call/cache-hit accounting, fixed `1e-5`, numerical
stability, zero-taxel causality, frozen TinyMDM, zero outcome ledger, and exact
checkpoint continuity. This admits the final Stage-H robustness/interface
preflight.

No formal long-horizon exploration, low squat, bottom support, regrasp, or
alternative successful carrying strategy has yet been learned. That remains
the next scientific phase rather than something implied by the H2R1 pass.

The authoritative local evidence is
`experiments/sugar_reproduction/outputs/final/smp_icm/stage_h_integration_20260723/`.
The H1 negative package is
`experiments/sugar_reproduction/outputs/final/smp_icm/stage_h_h1_failed_20260723/`.
The complete native tactile-optimizer stabilization package is
`experiments/sugar_reproduction/outputs/final/smp_icm/stage_h_tactile_optimizer_stabilization_20260723/`.
The complete five-role negative/admitted package is
`experiments/sugar_reproduction/outputs/final/smp_icm/stage_h_h2_five_role_20260723/`.
Both remain ignored by Git.

## Humanoid Everyday Auxiliary Semantic Result

The frozen official Humanoid Everyday real-G1 screen finds one bilateral
football transport and one unilateral soft-toy transport. This is evidence
that an auxiliary semantic representation test can require the same broad
object-transport meaning across two hand topologies.

It is not alternative CarryBox evidence: there is no common object/load
condition, failure/recovery, hand switch, bottom support, or direct TacSL, and
the public LeRobot schema contains no pressure. The screen therefore adds zero
strategy positives and leaves SUGAR/direct-TacSL alternative-success
collection as the active gate. See
`DOCS/sugar_humanoid_everyday_carry_source_compatibility_result_20260724.md`.
The independently verified package has curation manifest
`58f7c3f2563ed5ab9e7e75d8da2563ca06d33dfb4ecd5b3bfcbba7e87cc79d7e`.

## OmniRetarget Alternative-Strategy Source Gate

The fixed public OmniRetarget release provides about three hours of
OMOMO-derived G1+box qpos and advertises diverse box-carrying styles. This is
substantially closer to the missing alternative-strategy source than task-name
metadata or nominal project videos.

It remains external kinematic data with no declared contact, force, action,
material, mass intervention, or TacSL. The complete 1,826-file audit and exact
asset screen now pass, 57 originals are fully rendered, and manual
full-timeline review retains three physical-replay candidates covering
bottom-plus-side, unilateral bottom, and long bottom-dominant transport.
Apparent lower-posture motion 39 is correctly rejected as terminal placement,
not sustained low carrying.

All `15/15` official augmented siblings of the three retained families also
preserve their required roles, with exact original reproduction. This is
derived geometric stability rather than independent strategy diversity.

This closes source discovery only. The source uses a `0.1 kg` largebox with
different geometry from official SUGAR's `0.5 kg` smallbox, and the two
projects use different 29-joint column orders. It demonstrates neither
weight-conditioned choice nor SUGAR physics success. Exact official-Holosoma
conversion, official-smallbox retargeting, frozen SUGAR policy/action, and
direct-TacSL replay remain mandatory. Protocols:
`DOCS/sugar_omniretarget_boxcarry_source_compatibility_protocol_20260724.md`
and
`DOCS/sugar_omniretarget_or4_sugar_tacsl_bridge_protocol_20260724.md`.
The target-asset subgate was frozen before inspection and now passes. The
SUGAR-smallbox is one 50,004-vertex collision-tagged composed mesh with exact
local dimensions `0.40004499 x 0.54605001 x 0.53614101 m`; both runtime
configs use unit scale and `0.5 kg`, and its three-view render passes. Its raw
vertices are still not official retargeter surface points.
Official-source inspection also finds that demo/current object points are the
same 100 seeded same-mesh samples before/after `smpl_scale`; the optimizer
assumes indexed correspondence. No official largebox-to-SUGAR-smallbox
correspondence method has been found, so independent sampling or normalized
box scaling is prohibited. Audit:
`DOCS/sugar_omniretarget_official_object_point_correspondence_audit_20260724.md`.
The official `holosoma-retargeting==0.1.0` wheel is now cached and verified
byte-for-byte against main, but it remains uninstalled and is not a complete
dependency environment. All 66 visible refs and the release code still expose
no cross-mesh mapping. Release/runtime audit:
`DOCS/sugar_omniretarget_holosoma_release_runtime_audit_20260724.md`.
The wheel's own raw `sub3_largebox_003` fixture and exact released target now
define an adapter-free official round trip before selected-candidate work:
`DOCS/sugar_omniretarget_holosoma_largebox_roundtrip_protocol_20260724.md`.
Asset result:
`DOCS/sugar_omniretarget_or4_m1_smallbox_asset_result_20260724.md`.
OR0--OR3 result:
`DOCS/sugar_omniretarget_boxcarry_source_compatibility_result_20260724.md`.
The compact independently verified evidence package has curation manifest
`0dbb267962c4b78d359b8d941ac86023c6cd960106f285227a6facda0de4416a`.

## Fresh Strategy-Support Confirmation Boundary

The fresh evidence path is now implemented and runtime validated. An unregistered
one-environment coherent-task config preserves the official SUGAR action,
physics, goal task, and direct dual-R15 TacSL while adding official GelSight
RGB/depth and a world camera. A guarded single-pass runner loads only an
admitted frozen tactile policy and uses deterministic inference; it does not
train PPO, SMP, ICM, or any reward model. It records complete causal
goal/safety/action/TacSL fields, one `960 x 540` world frame per policy step,
raw-frame hashes, and event-window bilateral GelSight RGB/depth.

This runner is a confirmation/visual-evidence collector, not curiosity and
not a replacement for ICM discovery. Current C4 cannot be backfilled into the
new schema. The first authorized candidate passes its independent same-pass
collection audit but fails manual strategy-support review: zero direct
contact, no failure/later attempt/goal success, and unsafe termination after
23 observations. Current counts remain `0/0`. Record:
`DOCS/sugar_demo_reward_strategy_support_collection_protocol_20260724.md`.

The deeper source-interface audit closes the first-draft runtime assumptions.
The goal task uses its exact fifteen
state/direct-TacSL policy terms rather than reference-Refiner terms; terminal
goal stability must be causally reconstructed because reset precedes command
counter update; and native shear `[environment, taxel, xy]` must be
transposed before channel-first archival. The frozen generic Stage-I C4
archive did not perform that transpose, so its generic signed-shear maps are
not evidence. Its valid normal channel still proves zero later contact, and
the correctly transposed official policy/ICM path is unaffected. The guarded
collector now also hash-pins the seven official R15/TAXIM runtime assets and
verifies the declared mass/friction/COM event readback. The independent audit
now recursively binds the collector run record, original request, frozen
checkpoint, fourteen source hashes, all outputs, and the 16-observation
minimum. Exact camera replay remains only a frozen future contract until its
producer and complete provenance verifier exist.

Runtime result:
`DOCS/sugar_demo_reward_strategy_support_collection_result_20260724.md`.
The collector now self-binds the unchanged global-v3 mounts before robot
import. Exact motion-45 contact states verify bilateral taxel-resolved normal
force and two-axis signed shear. The official v2.3.2 compliant-contact
stiffness/damping `10/1` is now faithfully bound to both elastomers, and an
independently audited `0.25--4 mm` per-hand normal-press sweep validates
bilateral simulated pressure/shear plus GelSight RGB/depth response. The
camera-fresh exact nominal replay remains optically static at only
`0.005138395 N` maximum integrated normal force, and the fresh Stage-H policy
still has zero direct contact. Contact-seeded evaluation now sharpens the
strategy blocker: both H2R1 and C4-P3 global-90 begin from a verified bilateral
motion-45 state, discard one hand on their first action, and subsequently
produce only unilateral force-coupled optical response. The next learned
curriculum must first retain coordinated two-hand contact/load; original ICM
remains independent and may only explore alternatives after a stable nominal
clamp can actually fail, release, and recontact. The mount may not be tuned to
manufacture deformation. Recovery/alternative counts stay `0/0`, and camera
images remain audit-only unless a separate ablation is explicitly authorized.
Record:
`DOCS/sugar_tacsl_gelsight_contact_load_policy_preflight_result_20260724.md`.

The first contact-retention integration is now concrete rather than
conceptual. A separate external term scores the weaker hand's integrated
official TacSL normal load before the first failed strategy; it never enters
or gates original ICM. The canonical one-update run also hash-locks the
bilateral motion-45 reset state on every episode and self-binds the unchanged
v3 sensor mounts. Exact reward reconstruction and independent ICM separation
pass, but positive retention occurs on only `3/480` valid environment-steps.
This is sufficient to admit the mechanism, not to claim learned retention.
The next scientific comparison is a bounded reward-on/reward-off pair,
followed by held-out direct-TacSL plus GelSight validation. Record:
`DOCS/sugar_contact_topology_retention_smoke_result_20260724.md`.

That comparison is now complete and negative. After excluding 101 automatic
reset observations from each arm, the eight-update reward-on run has only 26
valid bilateral-contact transitions versus 34 for the reward-disabled
control, and closes two more failures. Separate fresh PhysX processes also
fail the predeclared first-rollout bitwise gate, so those training differences
remain descriptive rather than causal. The decisive frozen held-out result is
unambiguous: both final policies lose left-hand direct TacSL on their first
action, preserve bilateral direct contact on `0/32` post-action frames, and
pass independent trace/visual audits with the candidate gate false. Manual
inspection shows the same lost-box behavior in both synchronized
pressure/GelSight/world dashboards.

This rejects the current `0.04 N`, `0.02`-scaled, pre-first-failure objective;
it must not be extended. The next idea is narrower: first learn a separately
named nominal contact foundation whose continuous weaker-hand TacSL score is
normalized to frozen source-scale statistics and remains active across its
short retention horizon. That external foundation is not curiosity and must
turn off before post-failure discovery. Original forward/inverse ICM remains
learned, independently logged, and outcome-ungated, including on novel failed
transitions. Record:
`DOCS/sugar_contact_topology_retention_pair_result_20260725.md`.

The narrower nominal foundation has now also been implemented and rejected at
this parent/reset boundary. Its frozen scale is the motion-45 source weak-hand
load `0.0010956876212731004 N`; a reward-manager weight of `5` gives a maximum
per-transition ledger contribution of `0.1`. The one-update wiring audit
passes, including direct-taxel reconstruction and nonzero original ICM on all
`475` valid zero-foundation transitions. But the eight-update reward-on trace
has only `33` valid bilateral transitions versus `31` for control and closes
three more failures; the strict fresh-PhysX first-rollout bitwise gate still
fails. Frozen seed-`4253` evaluation is decisive: both final policies lose
left direct TacSL on their first action, have `0/32` bilateral direct-contact
frames, and retain right contact on only `2/32`. Both synchronized
pressure/GelSight/world artifacts pass independent audit and were manually
inspected.

The common first-action failure across both reward designs and both controls
changes the next idea. Before another PPO segment, audit the causal
reset/action bootstrap: determine whether the restored motion-45 physical
state is paired with the policy/action history expected at source frame 103,
and compare the frozen action against the official SUGAR reference
continuation plus a declared no-learning hold diagnostic. This is an audit of
a faithful bootstrap boundary, not permission to hand-code a controller.
Only if official-action continuation preserves bilateral direct TacSL should
a new nominal bootstrap curriculum be considered. Recovery/alternative counts
remain `0/0`. Record:
`DOCS/sugar_nominal_contact_foundation_pair_result_20260725.md`.

That causal bootstrap audit now passes and resolves the boundary. In one
synchronous four-branch PhysX step, cold H2R1, H2R1 with the correct action-102
plus TacSL-100--103 history, the recorded official SUGAR action 103, and a
current-joint-target hold all begin from the same bilateral frame-103 state.
Only the official SUGAR action preserves bilateral direct TacSL: it produces
left/right loads `0.004663/0.042392 N` across `31/150` active taxels with
nonzero signed shear on both hands. Both frozen H2R1 branches still lose the
left hand; the hold loses both. The causal-history policy remains L2 `14.36`
from the official action. Independent array audit and manual render inspection
pass.

The active idea is therefore no longer another scalar contact reward. Use the
accepted official SUGAR Refiner as a separately declared nominal action
teacher, with an exact teacher-observation mapping and zero-change audit.
Test a frozen multi-step teacher prefix, then an official-teacher residual or
behavior-cloned goal-policy bootstrap. The teacher prior must relax after a
tactile-confirmed nominal failure so it cannot suppress asymmetric,
bottom-support, lower-squat, regrasp, or other valid alternatives. Original
ICM remains independently learned and outcome ungated. This is an action
prior, not curiosity and not permission to restore exact reference tracking
as the final exploration objective. Record:
`DOCS/sugar_contact_bootstrap_action_probe_result_20260725.md`.

The frozen four-step follow-up strengthens this direction. Exact recorded
official actions 103--106, generated by the accepted Refiner, preserve
bilateral direct TacSL on every goal-scene step and grow left/right loads from
`0.00110/0.00314 N` to `0.06139/0.22731 N` with `146/293` active taxels.
Independent audit and manual pressure/shear/GelSight/world inspection pass.
The resulting state is not an exact original-trajectory reproduction, and the
probe does not perform live Refiner inference. The remaining nominal-teacher
idea gate is therefore exact online reconstruction of the accepted Refiner
reference observation and live frozen action semantics before any residual or
behavior-cloning adapter.

That live-teacher gate now passes. The unchanged goal scene computes the exact
uncorrupted official 890-D Refiner observation through a second official
`ObservationManager`, and the accepted frozen `model_10000.pt` is loaded
through the official RSL-RL `ActorCritic`. At the same restored source state
103 and the source-declared reference frame 299, the first live teacher action
matches the recorded official actor output to L2 `1.94e-6` and maximum
absolute error `9.54e-7`. Four live closed-loop steps preserve bilateral
direct pressure and two-axis signed shear on every seed/post record, ending at
left/right `0.02515/0.09650 N` over `87/203` active taxels with no reset.
Independent audit and manual synchronized render inspection pass.

This admits the official Refiner only as a nominal action teacher. The active
idea is now a matched causal goal-policy comparison: a zero-residual official-
teacher action prior versus an offline behavior-cloned bootstrap. Neither is
untouched official SUGAR, neither defines curiosity, and both must relax
teacher imitation after tactile-confirmed nominal failure. SMP, original ICM,
slip, task outcome, safety, and teacher imitation remain separate. Record:
`DOCS/sugar_live_official_refiner_teacher_result_20260725.md`.

That comparison has now selected the live-teacher residual mechanism. The
offline BC actor fits held-out official actions but shows no tactile advantage
and violates the fresh four-step contact-load/footprint envelope. The selected
residual path instead passes exact-zero initialization and one real integrated
upstream-PPO update: PPO stores/log-scores only the causal residual, the
environment executes the declared teacher-plus-scaled-residual action, the
native joint target reconstructs exactly, and a real direct-TacSL-confirmed
failure releases the teacher. Original ICM remains an independently learned
transition-novelty model, reads the actual applied action, and stays active
through release. The frozen checkpoint world/direct-TacSL/GelSight RGB/depth
render makes the present limitation clear: contact is lost and the rollout
resets. This closes an interface gate, not discovery; counts remain `0/0`.
Record:
`DOCS/sugar_official_refiner_residual_ppo_integration_result_20260725.md`.

The subsequent 64-update residual test resolves the immediate “just train
longer” hypothesis negatively. All serious integration checks remain sound,
but deterministic update 8 and 64 both fail/reset at steps 3/17 with no
bilateral recontact. A common-random-number stochastic control appears to lift
the box one-handed by about `0.48 m`, yet the same trajectory is already
present at update 1: sampled residual L2 is about `5.306`, versus only
`0.0383` for the update-64 learned mean, and the nominal teacher never
releases. This is a valuable exploration sample but not learned ICM policy
progress or post-failure recovery.

The paired 20-stream follow-up now confirms that boundary: update 64 adds two
failures/resets, avoids none, and lowers mean final box height by `0.04317 m`,
while sampled residual L2 remains about `5.35` versus learned-mean L2
`0.03359`. A separate exact original-ICM attribution audit cross-scores both
rollout endpoints with both checkpointed ICM models. Under frozen ICM update
1, the per-stream source-episode update64-minus-update1 forward-error
difference is positive in 13/20 streams but has mean `-0.06389` and paired-
bootstrap 95% interval `[-0.21107,0.04297]`. Frozen-feature support distance
increases only `0.02498`. Thus current policy training has not demonstrated
increased discovery, even though ICM itself learns visited transitions. This
conclusion does not use task result and does not call novel failures bad
curiosity. The next gate is a matched policy-credit/reward-mechanism ablation,
not more of the same unqualified training. Records:

- `DOCS/sugar_official_refiner_residual_h2r1_64_update_result_20260725.md`;
- `DOCS/sugar_official_refiner_residual_multiseed_distribution_result_20260725.md`;
- `DOCS/sugar_official_refiner_residual_icm_discovery_attribution_result_20260725.md`.

## 2026-07-25 Policy-Credit and Exposure Finding

The matched full/SMP-credit-zero/ICM-credit-zero comparison does not admit a
policy-credit separation. All three exact original-ICM frozen-forward-error
intervals cross zero, even though every ICM learner continues to learn its
visited transitions. A larger learned residual after removing SMP is therefore
not itself a new discovery.

The more actionable observation is distributional: only `2.36%--3.23%` of
the fixed training budget is executed with a zero teacher coefficient. The
next hypothesis changes post-failure exposure, not the curiosity definition
or reward mix. A recurring 64-action window suppresses only effective
`dropped_after_lift` after its first raw trigger in each episode. Raw drop and
all safety terms stay visible; original ICM remains outcome-ungated. Frozen
evaluation uses the original termination. Recovery/alternative counts remain
`0/0`.

That exposure test is now complete. It raises zero-teacher actions by `41.9%`
but does not increase frozen-feature coverage, and its forward-error interval
still crosses zero. The physical replay exposes why: one scalar release turns
off arms, waist, and legs together, so post-failure arm exploration is coupled
to loss of humanoid support. The next idea is blockwise teacher authority:
keep the official leg/waist block during an arm-release comparison, then only
consider lower-body relaxation after stable recontact. This is action-prior
structure, not ICM shaping.

## CHORD-Inspired Functional Contact Equivalence (2026-07-27)

The official CHORD paper strengthens one part of this idea: demonstrations
should constrain what contacts can do to the object, not require the robot to
copy human contact positions. CHORD compares friction-cone contact sets
through object-centric force/torque support. That representation can regard a
side clamp and a bottom-support-plus-side-brace strategy as mechanically
similar even when hand roles and contact locations differ.

CHORD is not yet an implementation dependency. Its project page says
`Code soon`, its reported whole-body policy removes contact observation, and
its real deployment is state/Vicon based rather than TacSL based. Exact CHORD
also keeps object/robot reference tracking and per-hand contact-mismatch
penalties that may suppress the alternative strategies sought here.

The admitted hypothesis is therefore a future, explicitly new
`r_object_wrench_guidance`: retain the complete spatial TacSL history, derive
an object-frame mechanics audit, aggregate across both hands, and reward only
sufficient support in task-required force/torque directions. It remains an
external mechanics objective, never ICM and never a replacement for spatial
tactile input. It may not enter policy training until its source/mathematical
contract and alternative-strategy-preserving ablations are frozen.

Full audit:
`DOCS/sugar_chord_contact_wrench_guidance_audit_20260727.md`.

## Whole-Hand Tactile Topology Clarification (2026-07-28)

One R15-sized palm patch is not equivalent to whole-hand tactile sensing. The
active candidate topology now separates 12 palm regions and three regions on
each of five rigid digits per hand. Its research value is not merely “more
taxels”: it preserves where load transfers between palm, individual fingers,
and proximal/distal segments, which is necessary for distinguishing a side
clamp from a bottom-support or brace strategy and for localizing slip.

The repeated-54-R15 rollout is now rejected rather than force-valid.  A later
same-rollout raw-PhysX audit measured the actual randomized box mass
`0.989127576 kg` and weight `9.70334194 N`.  Raw rigid contact closes
`m(a-g)` with median vertical error `0.40512 N`, while the former
`7.57885 mm` TacSL field has median physical/TacSL normal cosine `-0.99225`,
reconstructs `-2.78117 N` rather than the required `+9.00794 N` vertical
support, and has only `1.51%/4.66%` left/right active-taxel proximity to a raw
contact under the corrected `5 mm` radius.  Its earlier bilateral
palm/thumb counts were ghost-SDF activity, not validated physical load, and
are withdrawn.

Same-state `0--3 mm` scans repair the normal sign but do not admit a shallow
mount: raw-point-to-active coverage reaches only `0.50--0.76`, while only
`0.13--0.40` of active taxels have a raw point within `5 mm`.  Official TacSL
shear remains directionally wrong at every standoff (median cosine
approximately `-0.54`) because its velocity-only penalty does not retain
static tangential displacement.  No common scaling, sign flip, negative
stiffness, or hand-written shear memory is allowed.

Optical and force admission must remain separate. A controlled official R15
with a hand integration window produces force-correlated RGB/depth and tracks
`1/2 mm` presses with approximately `0.979/1.979 mm` deformation. Enabling the
same compliant-contact elastomers inside the live CarryBox articulation,
however, adds collision response and changes the carrying trajectory. That
negative cannot be repaired by relabeling camera noise or synthesizing
deformation from taxel force.

The first predeclared shadow/sensor replay passes an interface/provenance
`30/30` audit.
It hash-locks the accepted 660-frame collision-neutral rollout, reconstructs
the CarryBox pose in the selected left/right elastomer frames, and drives two
isolated official compliant R15 articulations with the official GelSight
renderer. It does not map force to an image. All 19 bilateral source-palm
contact frames have bilateral optical response; left/right source-load versus
peak-deformation correlations are `0.994/0.964`.

User video inspection rejects the apparent visual/taxel correspondence and
shadow optical behavior. The `30/30` audit did not test box weight/acceleration
balance, repeated-patch overlap, visual hand-mesh distance to sampled taxels,
or valid R15 deformation/relaxation. It is therefore interface evidence and a
negative diagnostic, not an optical result. The offline optical-rendering gate
is reopened; human review still gates topology acceptance and any
policy/slip/ICM migration.

The replacement sensor idea is therefore an isolated official
Tactile-Genesis study, not another repeated-R15 offset search.  Its official
`ElastomerTaxel` attaches arbitrary probes to real collision geometry and
retains a contact/release-gated shear anchor, yielding full three-axis marker
displacement.  It must first pass the untouched official sandbox and a
static-hold/release bench in a separate Genesis environment, then use the
official G1 hand/box geometry.  Marker displacement must not be relabeled as
newtons or GelSight RGB, and no SUGAR observation integration is allowed
without an explicit cross-backend state/action/time synchronization gate.

Full negative result:
`DOCS/sugar_whole_hand_tacsl_physics_correspondence_negative_result_20260728.md`.

## Primary Sources

- [SMP project page](https://yxmu.foo/smp-page/)
- [SMP paper](https://arxiv.org/abs/2512.03028)
- [official MimicKit repository](https://github.com/xbpeng/MimicKit)
- [official MimicKit SMP instructions](https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md)
- [original ICM paper](https://arxiv.org/abs/1705.05363)
- [official ICM code](https://github.com/pathak22/noreward-rl)
- [CHORD project page](https://nvidia-isaac.github.io/video_to_data/chord/)
- [CHORD paper](https://arxiv.org/abs/2607.00033)
