# Curiosity: Universal Native Tactile and Slip

## Active objective

The active work is
[`PLAN/14_newton_isaaclab_universal_tactile/plan.md`](PLAN/14_newton_isaaclab_universal_tactile/plan.md)
with its executable
[`TODO`](TODO/14_newton_isaaclab_universal_tactile/todo.md). Policy training is
paused; Plan 14 builds one causal tactile and slip interface for IsaacLab and
Newton and demonstrates it on box and non-box scenes.

IsaacLab uses only the official v2.3.2
`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`. Newton uses the
public `newton.sensors.SensorTactile` added on the nested
`Newton/` branch `yanhongru/universal-tactile`, which spatially serializes
solved `Contacts.force` without replacing it by `kh * depth`. Both adapters
expose signed local normal force, signed two-axis shear, penetration, patch
geometry and source time. Official GelSight RGB/depth remains available only
where the backend actually produces it; Newton reports optical unavailable.

The slip detector consumes only current and past tactile fields plus source
timestamps. Simulator relative contact velocity is recorded separately and is
used only after a run as a held-out evaluation label. Object pose, reward,
success labels, aggregate rigid-contact wrenches and thresholded contact labels
never enter the deployable tactile frame.

## Representation foundation

The Plan-14 IsaacLab common adapter has completed a fresh full 660-frame
official SUGAR G1 CarryBox cycle using the current detector and patch geometry.
Contact begins near frame 250, remains bilateral through the lift/carry
interval, and returns to zero after release. The box reaches `0.836 m` above
its initial height; 266 frames contain bilateral contact and 192 frames are
both lifted and bilateral. The run keeps the raw `[660,2,27,20,25]` normal
field, signed two-axis shear, real per-patch metric sizes, per-taxel poses,
center-palm R15 RGB/depth and tactile-only slip evidence. Its ignored runtime
package is
`experiments/newton_universal_tactile/isaaclab_carrybox_universal_current/`.

Against held-out simulator tangential velocity, the corrected tactile-only
detector's binary incipient-or-gross decision has `99.66%` precision and
`82.84%` recall. Exact four-state agreement on contact is `59.64%`. The
detector no longer treats a high static shear/normal ratio alone as slip; it
requires temporal redistribution of the tactile field, which removes curved-
surface static false positives at the cost of missed velocity-defined slip.
Relative velocity is used only by the post-run evaluator, never by the
detector.

The unchanged official R15 adapter also runs on a non-box swept-capsule scene.
The 240-frame record contains 227 contact frames, a zero-speed `STICK` phase,
a slow `0.006 m/s` sweep dominated by `INCIPIENT`, and a fast `0.020 m/s`
sweep classified `GROSS` on all 20 frames. Against held-out velocity its
binary precision is `98.39%` and recall is `96.83%`. The video and raw trace
are under
`experiments/newton_universal_tactile/isaaclab_r15_capsule_slip/`; the signal
is the official R15 per-taxel force/shear/RGB/depth stream, not rigid-contact
force or a binary label.

The Newton backend now completes the same generalization check with one public
`SensorTactile` implementation and no scene-specific force formula. In the
600-frame Panda cube run, both pads have native contact for `356/357` frames,
the cube is lifted `0.224 m`, and the largest raw-force-to-grid conservation
residual is `7.63e-6 N`. In the unchanged 600-frame pen run, both pads have
contact for `351/349` frames, the pen is lifted `0.254 m`, and the maximum
conservation residual is `5.72e-6 N`. Newton has no GelSight optical stream, so
both records explicitly report optical unavailable rather than displaying a
fabricated image.

The controlled Newton plate/cube sequence physically separates stationary,
slow-stick, incipient and fast gross-slip intervals at friction coefficient
`0.005`. Against held-out relative tangential velocity, the tactile-only
detector has `79.82%` precision, `95.79%` recall and `89.67%` exact ordinal
accuracy. Incipient and gross onset are each detected two 60-Hz frames
(`33.3 ms`) after the first matching held-out frame. The instantaneous binary
comparison contains `23` false positives and `4` false negatives; most false
positives are expected hysteresis around sinusoidal turning points. This is a
simulator check, not hardware slip validation.

The five retained Plan-14 videos are:

- `experiments/newton_universal_tactile/isaaclab_carrybox_universal_current/carrybox_native_tactile_slip.mp4`
- `experiments/newton_universal_tactile/isaaclab_r15_capsule_slip/isaaclab_r15_capsule_tactile_slip.mp4`
- `experiments/newton_universal_tactile/newton_cube/native_tactile.mp4`
- `experiments/newton_universal_tactile/newton_pen/native_tactile.mp4`
- `experiments/newton_universal_tactile/newton_slip_control/native_tactile_slip.mp4`

The first four are the required box/non-box scene evidence across both engines;
the fifth isolates Newton slip classification. All are local ignored artifacts
and are intentionally absent from Git.

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
- `Newton/`: nested Newton clone on `yanhongru/universal-tactile`, including
  the public solved-force tactile sensor, tests and Panda box/pen evidence
  entry point.
- `scripts/sugar/native_tactile/`: native collector, representation, renderer,
  common IsaacLab/Newton adapters, tactile-only slip detector, held-out slip
  evaluator and CarryBox renderers.
- `experiments/newton_universal_tactile/`: ignored Plan-14 runtime traces,
  videos and concise numerical records.
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
- `PLAN/14_newton_isaaclab_universal_tactile/`: the active no-training plan.
- `TODO/14_newton_isaaclab_universal_tactile/`: the active executable queue.
- Plan 13 is a read-only record of the paused training investigation.
- `PLAN/12_isaaclab_native_tactile_representation/` and its TODO: completed
  representation foundation plus pending explicit visual acceptance record.

The active collector/renderer/training code is indexed in
[`scripts/sugar/native_tactile/README.md`](scripts/sugar/native_tactile/README.md).
The exact output tree and retained-allocation command are recorded in
[`REPRODUCE.md`](experiments/native_tactile_representation/whole_hand_carrybox_v3/REPRODUCE.md).
The sensors are online and causal in simulation but are not currently
wall-clock real-time. Their field contracts are geometry-reusable; the SUGAR
controller and its declared contact-object SDF remain CarryBox-specific, while
the unchanged adapters have separately run on the IsaacLab capsule and Newton
cube/pen scenes.

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

IsaacLab requires Python 3.11, Isaac Sim 5.1, the included IsaacLab tree, the
official SUGAR motion and released Refiner checkpoint. The active CarryBox
entry point expects:

```text
SUGAR/data/CarryBox/data_045/
experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt
```

Inside a retained GPU allocation, collect the common 660-frame IsaacLab frame
and render the full bilateral hand map with tactile-only slip state:

```bash
export CURIOSITY_ISAAC_PYTHON=/absolute/path/to/isaac-python

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/isaaclab.process \
  --status experiments/newton_universal_tactile/runtime/isaaclab.status \
  --log experiments/newton_universal_tactile/runtime/isaaclab.log \
  --tag isaaclab-universal-carrybox \
  -- env PYTHONPATH="$PWD:$PWD/IsaacLab/source/isaaclab:$PWD/IsaacLab/source/isaaclab_assets:$PWD/IsaacLab/source/isaaclab_contrib:$PWD/SUGAR/source/sugar_rl" \
    "$CURIOSITY_ISAAC_PYTHON" \
    scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
    --output-root experiments/newton_universal_tactile/isaaclab_carrybox \
    --scenario successful_grasp --max-steps 660 --headless \
    --enable_cameras --device cuda:0

"$CURIOSITY_ISAAC_PYTHON" \
  scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  --run-root experiments/newton_universal_tactile/isaaclab_carrybox \
  --output experiments/newton_universal_tactile/isaaclab_carrybox/carrybox_tactile_slip.mp4 \
  --title "IsaacLab native TacSL | SUGAR G1 CarryBox" --fps 50

"$CURIOSITY_ISAAC_PYTHON" \
  scripts/sugar/native_tactile/evaluate_tactile_only_slip.py \
  --run-root experiments/newton_universal_tactile/isaaclab_carrybox \
  --output experiments/newton_universal_tactile/isaaclab_carrybox/slip_evaluation.json

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/isaaclab_capsule.process \
  --status experiments/newton_universal_tactile/runtime/isaaclab_capsule.status \
  --log experiments/newton_universal_tactile/runtime/isaaclab_capsule.log \
  --tag isaaclab-r15-capsule-slip \
  -- env PYTHONPATH="$PWD:$PWD/IsaacLab/source/isaaclab:$PWD/IsaacLab/source/isaaclab_assets:$PWD/IsaacLab/source/isaaclab_contrib:$PWD/SUGAR/source/sugar_rl" \
    "$CURIOSITY_ISAAC_PYTHON" \
    scripts/sugar/native_tactile/run_isaaclab_r15_capsule_slip.py \
    --output-root experiments/newton_universal_tactile/isaaclab_r15_capsule_slip \
    --frames 240 --fps 50 --headless --enable_cameras --device cuda:0
```

Newton uses the nested clone and its environment with Warp, MuJoCo/MJWarp,
USD and headless GL available:

```bash
git clone --branch yanhongru/universal-tactile --single-branch \
  https://github.com/yanhr21/Curiosity.git Newton

export CURIOSITY_NEWTON_PYTHON=/absolute/path/to/newton-python

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/newton_cube.process \
  --status experiments/newton_universal_tactile/runtime/newton_cube.status \
  --log experiments/newton_universal_tactile/runtime/newton_cube.log \
  --tag newton-native-tactile-cube \
  -- env PYTHONPATH="$PWD:$PWD/Newton" "$CURIOSITY_NEWTON_PYTHON" \
    Newton/tactile_video.py --scene cube --frames 600 --normal-scale-n 5 \
    --output experiments/newton_universal_tactile/newton_cube/native_tactile.mp4 \
    --device cuda:0
```

Use `--scene pen` without changing the sensor or adapter for the non-box case.
Every displayed Newton tactile cell comes from `Contacts.force` after
`solver.update_contacts()`. The upper world panel uses synchronized top and
side projections of the same Newton `State.body_q` frame, including the Panda
joint chain, both finger bodies, object, table and cup; it is a state view, not
a replayed or reconstructed trajectory. Generated traces and videos remain
below the ignored `experiments/` tree and must not be committed. The retained
allocation shell remains alive after each recorded child exits.

Run the controlled Newton slip sequence with the same native sensor and common
detector:

```bash
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/newton_slip.process \
  --status experiments/newton_universal_tactile/runtime/newton_slip.status \
  --log experiments/newton_universal_tactile/runtime/newton_slip.log \
  --tag newton-native-tactile-slip \
  -- env PYTHONPATH="$PWD:$PWD/Newton" "$CURIOSITY_NEWTON_PYTHON" \
    Newton/tactile_slip_demo.py --frames 300 \
    --output experiments/newton_universal_tactile/newton_slip_control/native_tactile_slip.mp4 \
    --device cuda:0
```

Its JSON records the actual plate/object state, native tactile load, detector
state and held-out relative velocity for every frame. That held-out velocity
is computed only after the tactile detector update and never enters the common
frame or detector.

### Paused historical `504-D` Tracker-command experiment

The section below is retained only to reproduce the paused Plan-13
investigation. It is not an active queue and does not authorize training.

The commands below reproduce the frozen historical no-RGB route. Run them serially
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
