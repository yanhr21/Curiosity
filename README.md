# Curiosity: Native Tactile Training and Fusion

## Active objective

The active work is
[`PLAN/13_native_tactile_training_fusion/plan.md`](PLAN/13_native_tactile_training_fusion/plan.md)
with its executable
[`TODO`](TODO/13_native_tactile_training_fusion/todo.md).

Plan 12 established the reusable IsaacLab-native whole-hand representation and
synchronized CarryBox evidence. Plan 13 now tests whether that representation
helps a serious policy and how to fuse it into training. The first matched
experiment warm-starts the official-width reference-only actor from the
released Refiner, freezes its non-tactile mapping, and adds either current
TacSL or exact-zero tactile through a spatial adapter. Neither arm receives RGB
or measured current object state; the privileged critic and official Refiner
teacher exist only during training.

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
  camera-free evaluations, and synchronized policy videos. The current
  64-update action-residual pair is negative at one seed; its tactile-trained
  versus zero-trained video is indexed in the package README. Start from its
  `REPRODUCE.md` rather than individual runtime files.
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

## Reproduce from this branch

Use Python 3.11 with Isaac Sim 5.1 and the matching IsaacLab tree included in
this repository. Obtain the official SUGAR CarryBox data and released Refiner
checkpoint as described in [`SUGAR/README.md`](SUGAR/README.md). The active
entry points expect these two local-only files/directories:

```text
SUGAR/data/CarryBox/data_045/
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
H.264 decoding, and source-frame-count checks pass. For policy training and
matched tactile-versus-zero evaluation, follow
[`experiments/native_tactile_training/REPRODUCE.md`](experiments/native_tactile_training/REPRODUCE.md).
All generated artifacts remain below the ignored `experiments/` tree and must
not be committed.

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
