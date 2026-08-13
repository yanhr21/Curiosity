# Curiosity: Universal Native Tactile and Slip

## Active objective

The active work is
[`PLAN/14_newton_isaaclab_universal_tactile/plan.md`](PLAN/14_newton_isaaclab_universal_tactile/plan.md)
with its executable
[`TODO`](TODO/14_newton_isaaclab_universal_tactile/todo.md). Policy training is
paused. As of 2026-08-13, all newly requested tactile demonstrations run only
in IsaacLab with the complete SUGAR G1 physically moving the object and exactly
27 anatomical TacSL patches on each hand. The existing G1 CarryBox run is the
foundation. Detached two-R15 cup/block/soft-body runs are diagnostics only and
do not count as completed object demos.
Newton results below are retained historical contract evidence, and Newton
assets may supply geometry, but no new demo uses Newton as its simulator. The
active serial order is rigid pickups and physical failures first, followed by
soft-object compatibility only after the rigid cases are complete.

The first non-box route is implemented around official PickBottle
`data_017` and its released Tracker, with the same full-G1 anatomical-54 TacSL
contract. H200 is a proven runtime here: retained job `231928` on `server13`
completed the full 660-frame IsaacLab CarryBox collection and all six videos.
On 2026-08-13 both explicit system-ICD and Vulkan-loader-default starts failed
before scene construction on `server13`, including the official IsaacLab camera
demo and the previously successful CarryBox entry point; a loader-default
formal start then failed identically on `server53`. This is a current cross-node
Kit/Vulkan runtime failure, not evidence against H200, PickBottle, or TacSL.
The failed starts are diagnostics, not tactile results. A separate clean Isaac
Sim 5.1 runtime is prepared, and retained job `237668` is queued for the final
H200 canary before formal PickBottle collection.

Inside that retained allocation, the complete PickBottle route is one command:

```bash
CURIOSITY_ISAAC_PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510_clean/bin/python \
  bash scripts/sugar/native_tactile/run_pickbottle_whole_hand_visualization.sh \
  experiments/isaaclab_g1_anatomical27_object_demos/pickbottle_success_native_v1
```

The script sets empty `DISPLAY`, selects the cluster NVIDIA Vulkan ICD, runs
the released PickBottle Tracker for all 270 official motion frames, records
native bilateral 27-patch TacSL fields and world RGB, and renders the combined
H.264. It does not allocate or release the retained GPU.

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

## IsaacLab-only rigid demo expansion

Four new IsaacLab/PhysX runs now exercise the same dual-official-R15 collector
on two rigid geometries and two distinct failure mechanisms:

- [cup stable bottom-support lift](experiments/isaaclab_rigid_tactile_demos/cup_bottom_support_success/isaaclab_cup_stable_lift_world_tactile.mp4):
  `0.1667 m` lift, `380/400` bilateral-contact frames and `204` lifted
  bilateral-contact frames;
- [cup post-lift physical drop](experiments/isaaclab_rigid_tactile_demos/cup_postlift_drop/isaaclab_cup_postlift_drop_world_tactile.mp4):
  the cup reaches `0.1753 m`, support is withdrawn physically, both tactile
  fields become zero, and the cup falls to the IsaacLab ground;
- [flat-block stable bottom-support lift](experiments/isaaclab_rigid_tactile_demos/flat_block_bottom_support_success/isaaclab_flat_block_bottom_support_success.mp4):
  `0.1667 m` lift with bilateral contact on `400/400` frames and median lifted
  active areas of `126/500` and `128/500` taxels;
- [flat-block dense side-pinch failure](experiments/isaaclab_rigid_tactile_demos/flat_block_side_pinch_no_pickup/isaaclab_flat_block_dense_pinch_no_pickup.mp4):
  the flat faces reach up to `442/500` and `385/500` active taxels, but the
  object slips during upward motion and never leaves the table.

The cup mesh is imported from the Newton asset cache as geometry only; all
body creation, SDF collision, gravity, motion and tactile computation are in
IsaacLab. The flat block is an IsaacLab mesh. Each video is a fully decoded
400-frame H.264 with one trace-driven world view above the left/right native
`20 x 25` signed normal-force and signed two-axis shear maps. Those force-only
collections explicitly disabled optical capture, so official R15 RGB/depth is
marked unavailable rather than fabricated. Exact commands and claim boundaries are in the
[rigid-demo reproduction guide](experiments/isaaclab_rigid_tactile_demos/REPRODUCE.md).

## IsaacLab-only deformable demo expansion

Official IsaacLab v2.3.2 TacSL binds its contact object through a rigid SDF,
while native IsaacLab deformables expose `SoftBodyView`. The local
deformable-surface extension keeps the official R15 `20 x 25` taxel geometry,
frames, output contract and TacSL normal/friction equations, but performs the
surface query against the current PhysX collision-tetrahedron boundary. It
therefore follows the actual deforming surface and its causal surface velocity;
there is no hidden rigid core, rigid-contact wrench or generated taxel field.
This is explicitly a project extension, not an upstream claim that official
v2.3.2 TacSL supports deformables out of the box.

The [stable soft-block lift](experiments/isaaclab_soft_tactile_demos/soft_block_bottom_support_success/isaaclab_soft_block_stable_lift_world_tactile.mp4)
raises the native `DeformableObject` by `0.1656 m`. Both R15s report contact on
all `400/400` frames, and the lifted interval has median active areas of
`386/500` and `417/500` taxels. Maximum surface-extent change is `1.9 mm` and
maximum R15 penetration is `0.47 mm`.

The matched [post-lift soft-block drop](experiments/isaaclab_soft_tactile_demos/soft_block_postlift_drop/isaaclab_soft_block_postlift_drop_world_tactile.mp4)
first reaches `0.1656 m`, then physically pulls both supports downward and
clear. Bilateral contact disappears on the next recorded frame and the object
falls `0.2734 m`; the peak collision stress occurs at ground impact, while
maximum R15 penetration remains below `0.46 mm`. Both H.264 videos show the
recorded deforming collision surface above the two synchronized signed
normal/shear maps. Reproduction and claim boundaries are in the
[soft-demo guide](experiments/isaaclab_soft_tactile_demos/REPRODUCE.md).

The stable interval's taxel forces reconstruct to about `0.139 N` upward on
the object, versus its `0.196 N` weight. This is not a hidden force-balance
success claim: official TacSL is a separately parameterized penalty sensor,
while PhysX compliant collision supplies the actual supporting wrench. The
maps are valid TacSL force/shear signals and spatial contact evidence, but
their absolute Newton scale remains uncalibrated to the PhysX solver or real
GelSight hardware.

## Representation foundation

The Plan-14 IsaacLab common adapter has completed a fresh full 660-frame
official SUGAR G1 CarryBox cycle using the current detector and patch geometry.
Contact begins near frame 250, remains bilateral through the lift/carry
interval, and returns to zero after release. The box reaches `0.836 m` above
its initial height; 266 frames contain bilateral contact and 192 frames are
both lifted and bilateral. The run keeps the raw `[660,2,27,20,25]` normal
field, signed two-axis shear, real per-patch metric sizes, per-taxel poses,
center-palm R15 RGB/depth and tactile-only slip evidence. Its ignored runtime
package explicitly stores independent force and optical sequence/timestamp/dt
fields and is
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
The 240-frame record contains 240 contact frames, a zero-speed `STICK` phase,
a slow `0.006 m/s` sweep dominated by `INCIPIENT`, and a fast `0.030 m/s`
sweep containing 19 held-out `GROSS` frames; the detector reports `GROSS` on
18 of the 20 commanded fast-phase frames. Against held-out velocity its binary
precision is `96.88%` and recall is `96.12%`, with four false positives, five
false negatives, a two-frame (`40 ms`) incipient onset delay and one-frame
(`20 ms`) gross onset delay. The video and raw trace
are under
`experiments/newton_universal_tactile/isaaclab_r15_capsule_slip/`; the signal
is the official R15 per-taxel force/shear/RGB/depth stream, not rigid-contact
force or a binary label. The trace also preserves every taxel's world position,
scalar-last `xyzw` orientation, RGB/depth frame and the independent tactile and
optical clocks from sequence `0` through `239` (`0.02` through `4.80 s`).

The current Newton SUGAR result uses the exact G1 URDF, the 54 physical
IsaacLab anatomical collision meshes and the repository CarryBox mesh. The
non-optical patch centers align to the official IsaacLab taxel centers with
`0.0000735 mm` median and `0.000544 mm` maximum error. The G1 is kinematically
driven by the recorded official motion while the real `0.3023376 kg` box is a
free Newton body under gravity. All non-patch robot collision shapes are
disabled, so an uninstrumented arm or torso cannot secretly support the box.
Four VBD substeps and per-substep interpolation remove the old 20-ms replay
jump. Across source frames `260...515`, the box is physically lifted `0.922 m`,
then returned to the ground; left/right tactile is nonzero on `255/256` and
`256/256` frames, with `255` bilateral frames. The maximum independently
reconstructed solved-force-to-grid difference is `2.74e-4 N`.

The synchronized Newton G1 H.264 shows the actual VTK world state above both
complete anatomical hand maps. It is a dynamic load-bearing CarryBox result,
but not a closed-loop policy result: the robot motion remains kinematic. The
late set-down produces a stiff penalty-contact spike (up to `456` raw contacts),
so the field is native solved Newton tactile but is not force-calibrated to the
IsaacLab TacSL magnitude or to hardware. Earlier failed releases from frames
200, 214 and 380 remain useful negative diagnostics, but no longer describe the
final per-substep-interpolated configuration.

The actual 660-frame IsaacLab trace and 256-frame Newton trace pass the same
runtime contract: 54 patches, `20 x 25` grids, signed local-Z normal and
signed local-XY shear, penetration, taxel world pose in `xyzw`, and explicit
source clocks. IsaacLab retains official R15 RGB/depth; Newton explicitly marks
optical unavailable and retains every solved raw contact instead.

The same public sensor also runs on Newton's actual rigid Panda hydroelastic
cube scene. Its 240-frame dynamic record has left/right pad contact on
`184/185` frames, lifts the cube `0.224 m`, and conserves the native
SolverMuJoCo contact force to `5.72e-6 N`. The video is an actual VTK/EGL render
of the simulated Panda and cube, not a schematic projection.

The unchanged `SensorTactile` also runs on Newton's official Franka plus
deformable rubber-duck example. Its native signal comes from solved
particle-rigid VBD force. The retained 360-frame pinch/lift/hold interval has
nonzero force on one left finger surface for `360/360` frames and the opposing
right surface for `272/360`; the other six real collision surfaces correctly
remain zero. Its maximum raw-force-to-grid conservation residual is
`0.000610352 N`. Newton has no GelSight optical stream, so every Newton record
reports optical unavailable rather than displaying a fabricated image.

The controlled Newton plate/cube sequence physically separates stationary,
slow-stick, incipient and fast gross-slip intervals at friction coefficient
`0.005`. Against held-out relative tangential velocity, the tactile-only
detector has `79.82%` precision, `95.79%` recall and `89.67%` exact ordinal
accuracy. Incipient and gross onset are each detected two 60-Hz frames
(`33.3 ms`) after the first matching held-out frame. The instantaneous binary
comparison contains `23` false positives and `4` false negatives; most false
positives are expected hysteresis around sinusoidal turning points. This is a
simulator check, not hardware slip validation.

The current human-review videos are:

- `experiments/newton_universal_tactile/isaaclab_carrybox_universal_current/carrybox_native_tactile_slip.mp4`
- `experiments/newton_universal_tactile/isaaclab_r15_capsule_slip/isaaclab_r15_capsule_tactile_slip.mp4`
- `experiments/newton_universal_tactile/newton_sugar_g1_anatomical54_vbd1200mu20_interp260_516_chunked_final_v3/newton_sugar_g1_carrybox_native_tactile.mp4`
- `experiments/newton_universal_tactile/newton_panda_hydro_cube_full60_300_mj38_v1/newton_panda_hydro_native_tactile.mp4`
- `experiments/newton_universal_tactile/newton_softbody_franka_240_600_v1/newton_softbody_franka_native_tactile.mp4`
- `experiments/newton_universal_tactile/newton_slip_control/native_tactile_slip.mp4`

The first four cover IsaacLab box/non-box and Newton dynamic G1/rigid-object
cases. The fifth is Newton's deformable-object case and the sixth isolates
Newton slip classification. All videos are local ignored artifacts and are
intentionally absent from Git.

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

- [complete force, kinematics, and friction video](experiments/newton_universal_tactile/isaaclab_carrybox_universal_current/carrybox_force_kinematics_friction.mp4)
- [machine-readable values](experiments/newton_universal_tactile/isaaclab_carrybox_universal_current/carrybox_force_kinematics_friction.audit.json)

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
- `PLAN/legacy/12_isaaclab_native_tactile_representation/` and its TODO: completed
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

Newton uses the nested feature branch plus an environment containing Warp,
USD, VTK/EGL, Pillow and `imageio-ffmpeg`. First export the exact physical
IsaacLab patch meshes, make one kinematic Newton reference trace, and solve the
single rigid root-frame bridge from the 52 non-optical patch centers:

```bash
git clone --branch yanhongru/universal-tactile --single-branch \
  https://github.com/yanhr21/Curiosity.git Newton

export CURIOSITY_NEWTON_PYTHON=/absolute/path/to/newton-python
export PYTHONPATH="$PWD:$PWD/Newton"

PATCH_ASSET=experiments/newton_universal_tactile/assets/isaaclab_anatomical54_collision_meshes.npz
ISAAC_TRACE=experiments/newton_universal_tactile/isaaclab_carrybox/whole_hand_trace.npz
NEWTON_REF=experiments/newton_universal_tactile/newton_g1_reference_244_516
ALIGNED_STATE=experiments/newton_universal_tactile/bridge_diagnostics/isaaclab_newton_aligned_root_244_516.npz

PYTHONPATH="$PWD:$PWD/IsaacLab/source/isaaclab:$PWD/IsaacLab/source/isaaclab_assets:$PWD/IsaacLab/source/isaaclab_contrib:$PWD/SUGAR/source/sugar_rl" \
  "$CURIOSITY_ISAAC_PYTHON" \
  scripts/sugar/native_tactile/export_isaaclab_anatomical_patch_collision_asset.py \
  --output "$PATCH_ASSET" --headless --device cuda:0

"$CURIOSITY_NEWTON_PYTHON" \
  scripts/sugar/native_tactile/run_newton_sugar_g1_carrybox_tactile.py \
  --output-root "$NEWTON_REF" --frame-start 244 --frame-stop 516 \
  --no-render --device cuda:0 --robot-state-trace "$ISAAC_TRACE" \
  --anatomical-patch-asset "$PATCH_ASSET" --robot-collisions sensor-only

"$CURIOSITY_NEWTON_PYTHON" \
  scripts/sugar/native_tactile/derive_newton_root_bridge_from_taxels.py \
  --base-bridge "$ISAAC_TRACE" --newton-trace "$NEWTON_REF/trace.npz" \
  --isaaclab-trace "$ISAAC_TRACE" --anatomical-patch-asset "$PATCH_ASSET" \
  --output "$ALIGNED_STATE"
```

Run the authoritative continuous free-box physics trace, then create the
playable world-plus-bilateral-hands video in short EGL workers:

```bash
"$CURIOSITY_NEWTON_PYTHON" \
  scripts/sugar/native_tactile/run_newton_sugar_g1_carrybox_tactile.py \
  --output-root experiments/newton_universal_tactile/newton_g1_dynamic_continuous \
  --frame-start 260 --frame-stop 516 --no-render --dynamic-box \
  --solver vbd --physics-substeps 4 --solver-iterations 8 \
  --vbd-contact-ke 1200 --vbd-contact-kd 0 --contact-friction 2 \
  --box-collision outer-sdf --robot-collisions sensor-only \
  --robot-state-trace "$ALIGNED_STATE" --anatomical-patch-asset "$PATCH_ASSET" \
  --device cuda:0

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/newton_sugar_g1.process \
  --status experiments/newton_universal_tactile/runtime/newton_sugar_g1.status \
  --log experiments/newton_universal_tactile/runtime/newton_sugar_g1.log \
  --tag newton-sugar-g1-dynamic-video \
  -- env CURIOSITY_ROOT="$PWD" PYTHONPATH="$PWD:$PWD/Newton" \
    "$CURIOSITY_NEWTON_PYTHON" \
    scripts/sugar/native_tactile/run_newton_sugar_g1_chunked_render.py \
    --output-root experiments/newton_universal_tactile/newton_g1_dynamic_video \
    --frame-start 260 --frame-stop 516 --chunk-size 50 --render-stride 2 \
    --force-scale-n 40 \
    --solver vbd --physics-substeps 4 --solver-iterations 8 \
    --vbd-contact-ke 1200 --vbd-contact-kd 0 --contact-friction 2 \
    --box-collision outer-sdf --robot-collisions sensor-only \
    --robot-state-trace "$ALIGNED_STATE" --anatomical-patch-asset "$PATCH_ASSET" \
    --device cuda:0

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/newton_panda_hydro.process \
  --status experiments/newton_universal_tactile/runtime/newton_panda_hydro.status \
  --log experiments/newton_universal_tactile/runtime/newton_panda_hydro.log \
  --tag newton-panda-hydro-cube \
  -- env CURIOSITY_ROOT="$PWD" PYTHONPATH="$PWD:$PWD/Newton" \
    "$CURIOSITY_NEWTON_PYTHON" \
    scripts/sugar/native_tactile/run_newton_panda_hydro_chunked_render.py \
    --output-root experiments/newton_universal_tactile/newton_panda_hydro_cube \
    --scene cube --frame-start 60 --frame-stop 300 --chunk-size 50 --device cuda:0

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/newton_universal_tactile/runtime/newton_soft_duck.process \
  --status experiments/newton_universal_tactile/runtime/newton_soft_duck.status \
  --log experiments/newton_universal_tactile/runtime/newton_soft_duck.log \
  --tag newton-softbody-franka \
  -- env CURIOSITY_ROOT="$PWD" PYTHONPATH="$PWD:$PWD/Newton" \
    "$CURIOSITY_NEWTON_PYTHON" \
    scripts/sugar/native_tactile/run_newton_softbody_franka_chunked_render.py \
    --output-root experiments/newton_universal_tactile/newton_softbody_franka \
    --frame-start 240 --frame-stop 600 --chunk-size 50 --device cuda:0
```

Every displayed Newton tactile cell comes from solved `Contacts.force` after
`solver.update_contacts()`. The upper panel is the actual Newton VTK/EGL state.
The G1 runner kinematically advances the robot at four interpolated substeps
while Newton freely simulates the box; Panda and Franka advance their native
rigid and deformable-object dynamics. Because the server VTK process aborts
after roughly 100 renders, the video orchestrators use at most 50 source frames
per worker. The no-render G1 command above is the continuous authoritative
physics trace; every video frame remains paired with the force trace from its
own actual worker simulation. Generated traces and videos remain ignored and
must not enter Git. The retained allocation shell stays alive after each child.

Check the two actual runtime contracts after both traces exist:

```bash
"$CURIOSITY_NEWTON_PYTHON" \
  scripts/sugar/native_tactile/compare_universal_tactile_runtime_contract.py \
  --isaaclab-trace "$ISAAC_TRACE" \
  --newton-trace experiments/newton_universal_tactile/newton_g1_dynamic_continuous/trace.npz \
  --output experiments/newton_universal_tactile/universal_runtime_contract.json
```

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
[`DOCS/legacy/curiosity_workspace_cleanup_20260810.md`](DOCS/legacy/curiosity_workspace_cleanup_20260810.md)
and the experiment-local
[`README`](experiments/native_tactile_representation/README.md).
