# Curiosity: Native Tactile Training and Fusion

## Active objective

The active work is
[`PLAN/13_native_tactile_training_fusion/plan.md`](PLAN/13_native_tactile_training_fusion/plan.md)
with its executable
[`TODO`](TODO/13_native_tactile_training_fusion/todo.md).

Plan 12 established the reusable IsaacLab-native whole-hand representation and
synchronized CarryBox evidence. Plan 13 now tests whether that representation
helps a serious policy and how to fuse it into training. The active first
comparison uses no RGB and no measured object state at the actor. Both arms
receive the official five-frame Tracker histories of base angular velocity,
joint position/velocity, previous action, and projected gravity; current base
linear velocity; normalized motion phase; and the deployable `35-D` official
Tracker command (`29-D` reference joint position plus `3-D` reference root
linear and `3-D` angular velocity). This is a `504-D` non-tactile actor input.
The tactile arm additionally receives the four-frame bilateral native TacSL
tensor; the matched control receives an exact-zero tensor and does not call the
sensor. The privileged critic and released Refiner are training-only.

The completed policy experiments below predate this corrected contract and
used an `890-D` reference-only actor containing full future reference state.
They remain reproducible diagnostic evidence, but they are not the active
deployable-input comparison and must not be used to claim completion of Plan
13. The next runnable milestone is the live one-update `504-D` tactile preflight,
followed serially by its exact-zero/no-read preflight.

Only the official IsaacLab v2.3.2
`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor` and official
`GELSIGHT_R15_CFG` provide tactile data. Rigid-contact labels, aggregate body
wrenches, object state, and generated or thresholded taxels are not tactile
inputs.

## Representation foundation

The active result is the articulated SUGAR CarryBox scene with 27 physical
TacSL patches on each hand. Every patch preserves its raw `20 x 25` signed
normal and signed-XY shear fields; the two center-palm patches also preserve
official R15 RGB/depth. The three main videos use one representation and one
anatomical layout:

- [successful grasp and carry](experiments/native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/successful_carrybox_whole_hand_tactile.mp4)
- [grasp followed by physical release](experiments/native_tactile_representation/whole_hand_carrybox_v3/failed_grasp/failed_carrybox_whole_hand_tactile.mp4)
- [left-only failed closure](experiments/native_tactile_representation/whole_hand_carrybox_v3/failed_closure/failed_closure_carrybox_whole_hand_tactile.mp4)

The successful grasp is physically sparse: all recorded support force is on
distal finger patches. During the lifted bilateral interval the thumb is only
intermittently active, while the four non-thumb fingertips dominate. Palm,
middle, and proximal patches remain visible but unloaded. TacSL patch activity
agrees with the corresponding PhysX patch contact on `99.797%` of patch-frame
pairs, and the sensorized-patch wrench equals the all-robot wrench to numerical
precision; the box is not secretly supported by uninstrumented robot bodies.

The clock-correct force audit is here:

- [complete force, kinematics, and friction video](experiments/native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/force_kinematics_friction_complete.mp4)
- [machine-readable values](experiments/native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/force_kinematics_friction_complete.audit.json)

At the native `5 ms` physics clock, the PhysX force on the box and
`m(a-g)` have a median full-vector residual of `8.78e-7` box weights. Friction
provides a median `2.568 N` of vertical support, with about `8.7%` global use
against the patch coefficient `0.5`. The old control-sampled force plot mixed a
last-substep contact force with a `20 ms` acceleration and is withdrawn as a
balance judgment.

The official TacSL penalty law itself reconstructs from the raw arrays, and
its contact localization is correct, but its raw taxel wrench is not calibrated
to the PhysX box wrench: median magnitude is `32.754 N` and median residual is
`11.533` box weights. These results are therefore high-fidelity simulated
contact-location/field evidence, not calibrated hardware force, FEM soft-body,
sim-to-real, policy-benefit, or real-robot evidence. Explicit user acceptance
of the videos remains the final completion gate.

## Active workspace

- `SUGAR/`: official SUGAR source and task assets.
- `IsaacLab/`: matching IsaacLab source tree.
- `scripts/sugar/native_tactile/`: native collector, representation, renderer,
  and the one-command complete CarryBox reproduction entry point.
- `experiments/native_tactile_representation/`: curated active raw data,
  videos, calibration, and checks.
- `experiments/native_tactile_training/`: matched tactile/zero checkpoints,
  camera-free evaluations, and synchronized policy videos from the historical
  `890-D` diagnostic route. Its 64-update action-residual pair is negative at
  one seed. A later held-out
  information gate is positive: the same serious adapter reduces aggregate
  teacher-action MAE by `26.26%` on two untouched mass/friction conditions.
  Its frozen live-versus-zero behavior gate is negative: the heavy condition
  loses reward and terminates earlier, while the low-friction condition has
  worse tracking and lift. The result is useful tactile information without
  established closed-loop benefit. These results do not authorize another PPO
  run and do not satisfy the active `35-D` actor contract. Start from the
  package `REPRODUCE.md` rather than individual runtime files.
- `PLAN/13_native_tactile_training_fusion/`: the active training/fusion plan.
- `TODO/13_native_tactile_training_fusion/`: the active training/fusion TODO.
- `PLAN/12_isaaclab_native_tactile_representation/` and its TODO: completed
  representation foundation plus pending explicit visual acceptance record.

The active collector/renderer/training code is indexed in
[`scripts/sugar/native_tactile/README.md`](scripts/sugar/native_tactile/README.md).
The exact output tree and retained-allocation command are recorded in
[`REPRODUCE.md`](experiments/native_tactile_representation/whole_hand_carrybox_v3/REPRODUCE.md).
The sensor is online in simulation but is not currently wall-clock real-time;
its force field is geometry-reusable, while the present task, object SDF, and
official controller remain CarryBox-specific.

The latest historical-policy human-review files are local experiment artifacts
and therefore intentionally absent from Git:

- `experiments/native_tactile_training/action_residual_64u_policy_visualization_20260811/tactile_trained_vs_zero_trained_side_by_side.mp4`
- `experiments/native_tactile_training/heldout_contact_residual_policy_visualizations_v1_20260811/heldout_heavy_1p0kg_live_vs_zero.mp4`
- `experiments/native_tactile_training/heldout_contact_residual_policy_visualizations_v1_20260811/heldout_low_friction_0p5kg_live_vs_zero.mp4`

Each held-out comparison shows the CarryBox world state above both complete
27-patch anatomical hand maps. The physical tactile field remains visible in
the zero condition, while the header states that exact zero/no-read entered the
actor. These camera-enabled videos are presentation evidence; the matched
camera-free JSON/NPZ results supply the numerical comparison.

## Reproduce from this branch

Use Python 3.11 with Isaac Sim 5.1 and the matching IsaacLab tree included in
this repository. Obtain the official SUGAR CarryBox data and released Refiner
checkpoints as described in [`SUGAR/README.md`](SUGAR/README.md). The active
entry points expect these three local-only files/directories:

```text
SUGAR/data/CarryBox/data_045/
SUGAR/demo_ckpts/CarryBox/tracker.pt
experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt
```

Inside a retained Slurm GPU allocation, reproduce the complete 660-frame
CarryBox sensor record and all five synchronized H.264 views with:

```bash
export CURIOSITY_ISAAC_PYTHON=/absolute/path/to/python

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/native_tactile_representation/runtime/reproduce.process \
  --status experiments/native_tactile_representation/runtime/reproduce.status \
  --log experiments/native_tactile_representation/runtime/reproduce.log \
  --tag reproduce_complete_carrybox \
  -- bash scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh \
    experiments/native_tactile_representation/reproduced_complete_carrybox \
    successful_grasp
```

The command returns zero only after collection, all five renders, complete
H.264 decoding, and source-frame-count checks pass. For the completed
historical policy diagnostics and matched tactile-versus-zero evaluation,
follow
[`experiments/native_tactile_training/REPRODUCE.md`](experiments/native_tactile_training/REPRODUCE.md).
That guide records the exact `890-D` inputs, warm-start export, serial training,
five-condition held-out teacher-residual test, frozen closed-loop comparison,
and synchronized live-versus-zero H.264 commands. It intentionally labels the
route historical. The active Tracker-command entry points and their exact
warm-start/resume variables are listed in
[`scripts/sugar/native_tactile/README.md`](scripts/sugar/native_tactile/README.md);
the live input-shape preflight remains the admission gate. Every long command
can be launched through `launch_retained_child.sh`; the retained allocation
stays alive when its recorded child completes.
All generated artifacts remain below the ignored `experiments/` tree and must
not be committed.

### Active `504-D` Tracker-command experiment

The commands below are the complete active no-RGB route. Run them serially
inside one retained GPU allocation from the repository root. Each output path
must be new. The released Tracker is used only to initialize the common actor;
the released Refiner remains the frozen BCPPO teacher and the `890-D` critic is
training-only. Official Tracker observation normalization remains disabled,
matching the released policy scale.

First create one common proxy-free base from the released Tracker:

```bash
export CURIOSITY_ISAAC_PYTHON=/absolute/path/to/python
export CURIOSITY_TRACKER_WARM_START_CHECKPOINT="$PWD/SUGAR/demo_ckpts/CarryBox/tracker.pt"

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/native_tactile_training/runtime/tracker_base.process \
  --status experiments/native_tactile_training/runtime/tracker_base.status \
  --log experiments/native_tactile_training/runtime/tracker_base.log \
  --tag tracker_base \
  -- bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
    tracker_zero \
    experiments/native_tactile_training/reproduced_tracker504/base \
    128 1 13011

unset CURIOSITY_TRACKER_WARM_START_CHECKPOINT
```

Freeze-evaluate `base/model_127.pt` before continuing:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh \
  tracker_zero \
  experiments/native_tactile_training/reproduced_tracker504/base/model_127.pt \
  experiments/native_tactile_training/reproduced_tracker504/base_frozen.json
```

The common base is admitted only if this continuous frame-zero rollout reaches
the physical CarryBox grasp/contact interval. If it does not, resume that same
base checkpoint and optimizer; do not replace the architecture, add object
state, or start the tactile comparison. Once admitted, run the live arm first
and the exact-zero/no-read arm second from the identical checkpoint:

```bash
export CURIOSITY_TRACKER_BASE_CHECKPOINT="$PWD/experiments/native_tactile_training/reproduced_tracker504/base/model_127.pt"

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tracker_preflight_tactile \
  experiments/native_tactile_training/reproduced_tracker504/preflight_tactile \
  1 1 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tracker_preflight_zero \
  experiments/native_tactile_training/reproduced_tracker504/preflight_zero \
  1 1 13011

python scripts/sugar/native_tactile/summarize_tracker_command_preflights.py \
  --tactile experiments/native_tactile_training/reproduced_tracker504/preflight_tactile \
  --zero experiments/native_tactile_training/reproduced_tracker504/preflight_zero \
  --output experiments/native_tactile_training/reproduced_tracker504/preflight_pair.json
```

The summary must pass real native signal and encoder optimization in the live
arm, exact-zero/no-read behavior with zero encoder update in the control, equal
pre-learning policy tensors, and the `504/324000/890-D` runtime shapes. Only
then run the matched endpoints, still serially:

```bash
bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tracker_tactile \
  experiments/native_tactile_training/reproduced_tracker504/tactile \
  512 1 13011

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_training.sh \
  tracker_zero \
  experiments/native_tactile_training/reproduced_tracker504/zero \
  512 1 13011

unset CURIOSITY_TRACKER_BASE_CHECKPOINT

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh \
  tracker_tactile \
  experiments/native_tactile_training/reproduced_tracker504/tactile/model_511.pt \
  experiments/native_tactile_training/reproduced_tracker504/tactile_frozen.json

bash scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh \
  tracker_zero \
  experiments/native_tactile_training/reproduced_tracker504/zero/model_511.pt \
  experiments/native_tactile_training/reproduced_tracker504/zero_frozen.json
```

These commands define a reproducible experiment, not a pre-claimed result.
Tactile benefit requires better matched frozen CarryBox behavior; a changed
gradient, loss, action, or checkpoint alone is insufficient.

## Legacy and experiment curation

Plans and TODOs 04--11 are read-only history under
[`PLAN/legacy/`](PLAN/legacy/) and [`TODO/legacy/`](TODO/legacy/). They are not
execution queues.

Historical experiments for (1) tactile effect on training, (2) demo following,
and (3) original ICM/Curiosity are curated to at most five retained packages in
the current workspace. Other historical and superseded artifacts are under the
single `/public/home/yanhongru/Curiosity_archive` tree. The active native
tactile representation is outside that historical five-package quota but is
also pruned to distinct scientific cases rather than version ladders. See
[`DOCS/curiosity_workspace_cleanup_20260810.md`](DOCS/curiosity_workspace_cleanup_20260810.md)
and the experiment-local
[`README`](experiments/native_tactile_representation/README.md).
