# SUGAR Tactile Reference-Only Actor Protocol

Status: predeclared on 2026-07-15. The leak-free observation functions,
unregistered task configs, and asymmetric non-tactile-critic actor-critic
adapter have been prepared as isolated new files. Architecture, deterministic
transform, and direct-config one-environment Manager/PhysX audits pass, but the
task has not been permanently registered, trained, or performance-evaluated.
This protocol becomes the next mainline only if the completed seed-42
exact-state branch fails its final tactile-advantage gate. It must remain a
separate experiment; its results may not be merged with or retrofitted into the
exact-state run.

The isolated actor-critic architecture and official checkpoint warm start have
passed a compute-node preflight. The official actor/critic widths are 890; the
reference-only actor has 890 base plus 256 spatial tactile features, while the
critic remains exactly 890. Zero actor tactile and the non-tactile critic both
reproduce official `model_10000.pt` with maximum error zero; changing tactile
changes actor output (`4.9891e-3` maximum in the deterministic probe) but leaves
critic output bitwise unchanged. All tactile encoder biases are frozen and all
seven non-bias spatial encoder weights are trainable. The local-only report is
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_preflight/actor_critic_architecture_audit.json`
(SHA256
`43ce1b5ec155fa3dece7de25b011952a41a493f7961c5866ad154b59ee99905a`).
This is architecture evidence only; simulator observation-leak and performance
gates remain open.

The six reference-plan transforms also pass a deterministic IsaacLab-math
audit. At a perfectly tracked frame they are bitwise identical to the six
official exact-state terms and retain the official widths. After changing only
the actual box by `(0.08, -0.05, 0.03) m`, `0.35 rad`, `(-0.7, 0.4, 0.2) m/s`,
and `(0.8, -0.6, 0.3) rad/s`, all six reference-only outputs remain bitwise
unchanged. The official exact-state terms respond in every field, with maximum
absolute changes from `0.0705` to `0.8` in their native representations. The
local report is
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_preflight/observation_transform_audit.json`
(SHA256
`0c05745679209e6aa29b3c86394fb0b40ab3e495eda45fab02a15a22bb07679a`).
That deterministic transform result is not used as a substitute for the
separate live Manager/PhysX perturbation audit below.

The direct-config live Manager/PhysX audit now passes as well without adding a
Gym registration. In one official CarryBox environment at fixed reference
frame 425, the manager serialized policy/critic/tactile widths of 890/890/3000.
Writing the requested actual object pose and linear/angular velocity was
verified in both the IsaacLab buffers and PhysX view. The complete policy group
remained bitwise identical (`0.0` maximum delta), while the complete privileged
critic group changed by `0.60784167`; the reference timestep remained 425. The
tactile field was intentionally unchanged because no simulation step occurred.
All 16 provenance, mapping, shape, readback, and invariance checks pass. The
local report is
`experiments/sugar_reproduction/outputs/CarryBox_20260715_tactile_reference_only_preflight/runtime_observation_audit.json`
(SHA256
`2b62c4ba8f4257007eaaf5793159c2bde872d9afac7b900ce323059d766381ff`).
This is observation wiring evidence only, not closed-loop task advantage.

An isolated matched-training wrapper is also prepared at
`scripts/sugar/run_official_sugar_carrybox_reference_only_tactile_refiner.sh`.
It fixes the three role-specific task IDs, accepted checkpoint, v3 mounts,
official dual R15 assets, 512 environments, 7098 uninterrupted updates, and an
18-file source manifest. It remains deliberately non-runnable until the
exact-state final result is frozen: the Gym registrations and the two minimal
official train/evaluator class injections are not yet active. This ordering
prevents preparation of the fallback branch from changing exact-state final
provenance. The formal wrapper hard-requires `512` environments and `7098`
updates. Its explicit diagnostic mode permits only `1-8` environments and
`1-2` updates, forces `diagnostic` or `smoke` into the run label, records a
no-performance claim boundary, and requires the official pre-update/model-0,
warm-start, and YAML artifacts. A reduced run can therefore validate the
process-local entry point but cannot be serialized as a matched run.

The activation glue is now implemented without editing either shared official
entry point or the exact task registry. Guarded process-local shims register
the three reference-only task IDs and expose
`ReferenceOnlyTactileActorCritic` to the unchanged RSL-RL class lookup before
delegating to the official training/evaluator scripts. A compute-node registry
test proves registration is rejected without
`CURIOSITY_ENABLE_REFERENCE_ONLY_TACTILE=1` and that exactly the three expected
configs appear when enabled. The formal training wrapper still requires the
completed exact-state audit to serialize `advantage_proven=false`, so this
prepared glue cannot activate the fallback branch before its predeclared gate.

The post-training identity audit is also prepared at
`scripts/sugar/audit_sugar_reference_only_training_identity.py`. It does not
infer reference-only semantics from a task name: it inspects the serialized
three-role environment and agent YAML files, requires the six actor mappings
to be the reference-plan functions and the six critic mappings to be official
exact-state functions, verifies the asymmetric observation groups and warm
start, normalizes only the declared tactile boundary, and binds all 18 source
manifest entries to current bytes. It additionally requires both the live
Manager/PhysX leak audit and the negative exact-state final gate. This audit
will run only on real training artifacts; its existence is not evidence that
training or advantage has occurred.

The formal identity audit now consumes all three preflight reports together:
actor-critic/checkpoint architecture, deterministic reference transform, and
live Manager/PhysX perturbation. Every reported check must remain true, and
the recorded exact task, reference observations/config, asymmetric
actor-critic, and accepted `model_10000.pt` SHA values are recomputed from
current files. A stale preflight can no longer authorize formal evaluation.

The downstream final-role and suite entry points are prepared separately at
`scripts/sugar/run_sugar_reference_only_final_evaluation_role.sh` and
`scripts/sugar/run_sugar_reference_only_closed_loop_eval_suite.sh`. They refuse
to start without real final checkpoints, a negative exact-state report, a
passing reference-only training-identity report, and all 18 matching source
digests. The suite fixes the process-local reference-only full task, 256
environments, 1501 steps, natural starts, six conditions, and the same seven
full-policy interventions plus matched zero/pressure policies. Its manifests
add the guarded evaluator shim, task registration, reference observations,
and asymmetric actor-critic hashes. A compute-node pretraining guard returned
status 3 before application launch, as intended; no reference-only evaluation
has run.

Both branches deliberately share the statistical implementation in
`scripts/sugar/analyze_sugar_tactile_advantage.py`. Its default `exact_state`
profile preserves the active final watcher behavior; the isolated
`reference_only` runner selects the second provenance profile. That profile
changes only the expected task, process-local entry points, source manifests,
and prerequisite negative exact-state audit, while tightening performance
admission as described below. It does not relax or fork the paired bootstrap,
nominal noninferiority, taxel/action exposure, or same-policy zeroing evidence.

The code now enforces the protocol's stronger success semantics rather than
merely documenting them. Under the `reference_only` profile, neither failure
reduction nor normalized progress can pass the performance gate: held-out
full-versus-matched-zero success and live-versus-zero success for the same full
policy must each have a positive paired 95% confidence lower bound. Those
secondary fields remain visible for diagnosis. The frozen exact-state profile
retains its already predeclared secondary path, but it cannot be pooled into
the reference-only multi-seed result.

Thin compute-only runners now connect the three real training directories to
the reference-only identity audit and the existing strict pre-update-to-final
weight audit. Their fixed outputs are the mandatory inputs of the final role
and reference-only analysis runner. They do not weaken the common checkpoint
contract or synthesize missing reports; before real training artifacts exist,
they fail closed.

The independent-training-seed audit now also binds an explicit protocol
profile, preventing exact-state and reference-only reports from being pooled.
Its reference-only runner fixes seed 42/43/44 reports, requires every
single-seed suite-manifest/weight/training-identity/advantage gate to pass, and
then requires both matched-control and same-policy zeroing success gains to be
positive in every seed with positive seed-bootstrap lower bounds. Environment
episodes are still not counted as independent training seeds. At aggregation
time it also requires report contents in exact 42/43/44 order and rehashes the
single-seed analyzer, both audit files, and every recorded shard manifest, so
renamed or post-hoc-mutated evidence fails closed.

The independent policy-training seeds are 42/43/44, but all three final suites
fix evaluation seed 42. Training and evaluation seeds are serialized
separately, and the multi-seed analyzer requires a common evaluation seed; this
prevents initial-state sampling differences from being counted as policy-seed
evidence. The training-identity audit independently reads both environment and
agent YAML seeds and requires them to equal the requested policy seed; final
evaluation rejects a directory/report seed mismatch.

Because the palm transform is injected while composing the USD and is not a
reliable `env.yaml` field, each formal run also writes a runtime metadata
record. The identity audit requires three distinct numeric Slurm jobs on
compute hosts, the exact full/zero/pressure tasks, 512 environments, 7098
updates, both locked v3 offsets, matching source-manifest and negative
exact-state-audit hashes, start/finish timestamps, and terminal status zero.
The mutable Unitree USD workspace and IsaacLab converter cache are isolated as
`/tmp/IsaacLab/reference_train_<job>_<run>/{unitree,converters}`; the three
runtime records must contain distinct paths matching their actual jobs and run
labels.

If seed42 passes while the replacement allocations remain alive, a generic
follow-up worker may use them for seed43. Each of the three roles must first
prove at least fourteen hours remain before the Slurm deadline and create its
own ready sentinel; the measured 7098-update runs project to roughly 11 hours,
leaving the remaining margin for audits and the 54-run suite. Only the
three-way barrier publishes the activation. This prevents a wall-clock edge
from starting an unmatched subset. Unactivated
readiness is removed after a 20-minute barrier timeout, safely below the
cluster's three-hour low-utilization eviction window. A login-node-only
fallback gives retained workers ten minutes after a passing seed42 report; if
they decline for insufficient remaining time, it waits for any partial barrier
to drain and then requests three fresh one-day seed43 allocations. The worker
uses the same formal wrapper, identity/weight audits, 54-run suite, and
single-seed gate. A second login-node-only watcher remains inert until that
strict seed43 report passes, then requests three separately retained one-day
seed44 allocations. The generic worker requires its prerequisite report to be
the immediately preceding training seed; after seed44's single-seed report,
role A automatically invokes the fixed seed42/43/44 audit. No seed43/44
allocation is requested after a negative preceding result, and seed44 remains
mandatory before a multi-seed claim.

## Hypothesis and boundary

The accepted SUGAR Refiner actor currently observes exact simulator object
pose, orientation, linear velocity, and angular velocity. Its future object
position and orientation commands are also expressed relative to the exact
current object pose. This is a useful exact-state upper bound, but it can make
touch informationally redundant.

The reference-only actor branch tests a narrower deployment hypothesis: when
the actor receives the intended CarryBox motion plan but no measurement of the
box's actual state, spatial pressure and shear can close the feedback loop
after slip or disturbance. The critic remains asymmetric and privileged. The
reference trajectory is a task command, not tactile input, and must never be
reported as such.

## Fixed actor observation transformation

All tensor widths, ordering, robot observations, action history, and future
robot-motion commands stay unchanged. Only the object-state terms below are
transformed. The same transformation is used by full, zero, and pressure-only
policies.

| Actor term | Exact-state source | Reference-only source |
| --- | --- | --- |
| `obj_pos_b` | actual box pose in current robot-anchor frame | current reference box pose in current robot-anchor frame |
| `obj_ori_b` | actual box orientation in current robot-anchor frame | current reference box orientation in current robot-anchor frame |
| `obj_lin_vel_b` | actual box linear velocity | current reference box linear velocity |
| `obj_ang_vel_b` | actual box angular velocity | current reference box angular velocity |
| `ref_obj_pos_b_future` | future reference poses relative to actual current box pose | future reference poses relative to current reference box pose |
| `ref_obj_ori_b_future` | future reference orientations relative to actual current box orientation | future reference orientations relative to current reference box orientation |

The two future reference velocity terms already use the current robot-anchor
frame and do not read the actual box state; they remain unchanged. At an
exactly tracked trajectory frame, the transformed object pose/orientation terms
match the original terms. After the physical box deviates, the actor receives
no pose-error shortcut from these fields.

Implementation must fail closed if any actor observation function reads
`command.obj_pos_w`, `command.obj_quat_w`, `command.obj_lin_vel_w`, or
`command.obj_ang_vel_w`. A compute-node observation audit must perturb the
actual box while freezing the reference index and prove:

- the actor's non-tactile object fields are bitwise unchanged;
- the critic's actual object fields change by the requested perturbation;
- the live tactile field changes only when the perturbation produces physical
  R15 contact; and
- full, zero, and pressure-only initial robot/object states and reference
  indices remain bitwise paired.

## Learning boundary

The branch keeps the accepted official-code `model_10000.pt`, official SUGAR
PPO, official CarryBox data/task/rewards/terminations, official v3 dual-R15
TacSL geometry, and the corrected robot/tip masses. It retains the frozen
accepted actor plus trainable spatial tactile adapter used by the corrected
exact-state branch. Tactile-encoder biases stay fixed at zero, so the zero
policy remains the reference-only accepted SUGAR actor. The critic is
trainable, receives the exact privileged SUGAR observation group, and receives
no role-dependent tactile input; this keeps the full/zero/pressure treatment at
the actor boundary.

The three seed-42 roles use identical initial parameter tensors, 512
environments, 7098 uninterrupted updates, optimizer/PPO settings, task
randomization, and reference sampling. They differ only in actor tactile mode:

- full: live taxel-resolved normal pressure plus signed two-axis shear;
- zero: all actor tactile taxels zero while both official sensors still run;
- pressure-only: live normal pressure with both shear axes zeroed at the actor
  boundary.

No contact label, aggregate wrench, object-state residual, penetration flag,
or reward quantity may enter the actor tactile path. Noise and latency are
held out as later declared sensor ablations; they are not tunable alternatives
for selecting the first positive result.

## Admission gate

The first seed uses the same natural trajectory starts, 256 paired evaluation
environments, six conditions, and seven full-policy tactile interventions as
the exact-state final protocol. A task advantage requires all of the following:

- nominal full is noninferior to matched zero;
- held-out full success exceeds matched zero with a positive predeclared paired
  bootstrap lower bound;
- the same trained full policy degrades when its tactile input is zeroed, also
  with a positive paired lower bound for live-minus-zero success;
- the gain is present in taxel-exposed/action-dependent environments and is not
  explained by unmatched starts, checkpoint drift, or critic inputs;
- pressure-only and the remaining shear/hand/lag interventions identify which
  spatial tactile content carries the gain; and
- the strict initial-state, source, checkpoint, and allowed-weight-change
  audits pass.

Object RMSE, reward, or progress alone cannot override worse official task
success. If seed 42 passes, repeat matched training with at least seeds 43 and
44 and require the existing multi-training-seed gate. If seed 42 fails, record
the negative result; do not sweep alternate occlusion/noise settings until one
is positive.

All results remain `high-fidelity simulated tactile` until physical
GelSight-class calibration is complete.
