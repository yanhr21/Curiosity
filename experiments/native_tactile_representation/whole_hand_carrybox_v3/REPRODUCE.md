# Reproduce the complete CarryBox tactile visualization

## Prerequisites

- Linux compute node with an NVIDIA GPU and a retained Slurm allocation;
- Python 3.11, Isaac Sim 5.1, and the repository's matching `IsaacLab/` tree;
- OpenCV and `imageio-ffmpeg` in that Python environment;
- official SUGAR CarryBox motion `SUGAR/data/CarryBox/data_045/`; and
- released Refiner checkpoint at
  `experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt`.

The data and checkpoint are local-only because of their size. Obtain them by
following the official instructions in `SUGAR/README.md`. Set
`CURIOSITY_ISAAC_PYTHON` when the cluster Python is not the default recorded in
the shell entry point. Do not run the simulator on a login node.

## Canonical result

`successful_grasp/` is the single canonical successful CarryBox bundle. It
contains one complete 660-control-frame trace and no second raw-data copy:

```text
successful_grasp/
  summary.json
  whole_hand_trace.npz
  world_carrybox.mp4
  successful_carrybox_whole_hand_tactile.mp4
  force_kinematics_friction_complete.mp4
  force_kinematics_friction_complete.audit.json
  left_detail.mp4
  right_detail.mp4
  palm_optical.mp4
```

The raw trace records:

- `660` control frames at `50 Hz`;
- `2640` physical substeps at `200 Hz`;
- two hands × 27 physical patches × `20 x 25` taxels;
- signed local-Z force, signed local-XY shear, penetration, taxel pose,
  SDF normal, and relative tangential velocity;
- bilateral center-palm R15 RGB/depth;
- separate per-patch and all-robot PhysX normal/friction references; and
- object state/velocity only in the audit fields.

The main and force videos display source frames `230:660`, covering contact,
pickup, carry, placement, release, and post-release zeros. The free-body force
comparison is shown only while the box is off the ground; after placement the
ground supplies the missing support force. Every renderer decodes its completed
H.264 output from first frame to last before it writes a passing result record.
The canonical main, left-detail, right-detail, bilateral R15, and force videos
each pass `430/430` frames.

## Verified record on 2026-08-11

The single entry point was rerun from the official checkpoint/task on a fresh
output directory in retained job `231928` on `server13`. It completed normally
from `23:43:58Z` to `23:58:21Z`: 660 control frames and 2640 physical substeps
were newly collected, all five presentation videos were newly rendered, and
the entry point's own full-decode check passed. The lightweight result record is
`../runtime/reproduce_complete_carrybox_20260811.result.json`; the exact process,
status, and console records use the same stem.

The fresh run reproduced the canonical motion and sensor result: maximum lift
`0.835659 m`, 266 bilateral-contact frames, 192 lifted-bilateral frames, maxima
of 275/271 active left/right taxels, identical contact confusion counts, and
the same kinematics and force-calibration verdict. The median PhysX balance
residual was `8.88e-7` rather than `8.78e-7` box weights, a floating-point-scale
difference with no change in interpretation. After this check, the duplicate
707 MB trace and videos were moved to the single archive tree so that the
active workspace continues to have only one canonical successful bundle.

The retained canonical files were decoded again from beginning to end after
workspace curation. The final all-outcome check is recorded in
`../runtime/carrybox_workspace_final_validate_20260811.{process,status,log}` and
returned exit code zero:

- `world_carrybox.mp4`: `660/660`, `1280 x 720`, `50 fps`;
- `successful_carrybox_whole_hand_tactile.mp4`: `430/430`,
  `2560 x 1440`, `50 fps`;
- `left_detail.mp4`: `430/430`, `2560 x 1440`, `50 fps`;
- `right_detail.mp4`: `430/430`, `2560 x 1440`, `50 fps`;
- `palm_optical.mp4`: `430/430`, `2560 x 1440`, `50 fps`; and
- `force_kinematics_friction_complete.mp4`: `430/430`,
  `1920 x 1080`, `50 fps`.

The physical release control also fully decodes: its world video is `372/372`
and its main, left, right, and optical videos are each `142/142`. The failed
closure control fully decodes `320/320` world frames and `50/50` bilateral
presentation frames; that deliberately shorter review clip is `25 fps`, while
the other retained videos are `50 fps`. Thus every retained H.264 file across all three
CarryBox outcomes is playable from first frame to last.

`summary.json` records the exact tensor shapes, box mass, episode outcome and
contact counts. `force_kinematics_friction_complete.audit.json` records the
native-clock force, friction, contact-correspondence and kinematic checks. The
six successful collection/render process records are retained together in
`../runtime/carrybox_complete_20260811/`. These are the complete record for
this canonical run; there is no parallel success-version ladder in the active
workspace.

## One reproduction entry point

Run inside an already retained Slurm allocation. The script does not request,
cancel, or release an allocation. Use the retained-child wrapper so the exact
child process group remains independently controllable:

```bash
cd /public/home/yanhongru/Curiosity

bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record experiments/native_tactile_representation/runtime/reproduce_carrybox.process \
  --status experiments/native_tactile_representation/runtime/reproduce_carrybox.status \
  --log experiments/native_tactile_representation/runtime/reproduce_carrybox.log \
  --tag reproduce_complete_carrybox \
  -- bash scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh \
    experiments/native_tactile_representation/reproduced_complete_carrybox \
    successful_grasp
```

The output directory must not already exist. The entry point runs, in order:

1. the official frozen SUGAR Refiner on motion 45 and the dynamic CarryBox;
2. native 54-patch IsaacLab/TacSL collection for all 660 source frames;
3. the complete bilateral anatomical video;
4. full left, right, and R15 optical detail videos; and
5. the clock-correct force/kinematics/friction video and its JSON record; and
6. a complete H.264 decode and source-frame-count check for every generated
   video. The reproduction command returns success only after this passes.

Expected successful output:

```text
reproduced_complete_carrybox/
  summary.json
  whole_hand_trace.npz
  world_carrybox.mp4
  successful_carrybox_whole_hand_tactile.mp4
  left_detail.mp4
  right_detail.mp4
  palm_optical.mp4
  force_kinematics_friction_complete.mp4
  force_kinematics_friction_complete.audit.json
  complete_bundle_validation.json
```

`summary.json` must report 660 control frames and 2640 physics substeps. The
world video must decode 660 frames; each synchronized review video must decode
430 frames for source interval `230:660`. A nonzero command exit, a missing
file, or a shorter decode is a failed reproduction.

The same entry point accepts `failed_grasp` and `failed_closure` as the final
argument. Those modes use the identical scene, sensor tensor, layout, and fixed
display scales; only the physical action intervention and available episode
length differ. The canonical successful bundle is the complete reference route.

The force renderer extracts the three large taxel arrays into a temporary
directory itself and removes that directory on exit. No manual extraction,
hash file, or version ladder is required.

To recheck every retained successful and failed video without recollecting the
simulation, run inside a compute allocation:

```bash
bash scripts/sugar/native_tactile/validate_complete_carrybox_bundle.sh \
  experiments/native_tactile_representation/whole_hand_carrybox_v3 \
  all
```

## Exact active implementation

- Collector: `scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py`
- Main renderer: `scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py`
- Detail renderer: `scripts/sugar/native_tactile/render_sugar_whole_hand_supplement.py`
- Force renderer: `scripts/sugar/native_tactile/render_sugar_force_kinematics_friction.py`
- One-command entry point: `scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh`
- Full-decode check: `scripts/sugar/native_tactile/validate_complete_carrybox_bundle.sh`
- Sensor scene:
  `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py`
- Audit scene:
  `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py`

## Runtime and generality boundary

The sensor values are online and causal in simulation: the force-field branch
updates at every `5 ms` physics step and does not use future frames or offline
video reconstruction. The saved presentation is sampled at the `20 ms`
control clock. This 54-patch plus bilateral optical scene is not wall-clock
real-time on the current GPU; it simulates more slowly than real time.

The hand topology and TacSL SDF calculation are reusable, but the current task
is not object-agnostic at runtime. Every sensor is initialized against
`{ENV_REGEX_NS}/Obj`, and that object must already contain a PhysX SDF mesh.
CarryBox uses the admitted `SMALLBOX_SDF_CFG`. Official KickBox currently uses
plain `BIGBOX_CFG`, not an SDF tactile asset, and its official behavior contacts
the box mainly with the foot rather than the instrumented hands. Therefore the
current CarryBox task cannot simply be renamed to KickBox and called a valid
KickBox tactile result.

A valid KickBox extension needs, before any positive claim:

1. an official-big-box SDF adapter that preserves the released visual/physical
   geometry;
2. a KickBox task configuration that binds the sensors to that SDF object;
3. tactile patches on the actual contact body, normally the foot/leg, unless a
   hand-contact KickBox behavior is deliberately used; and
4. the same contact correspondence, force-clock, visualization, and human
   review checks used here.

External camera occlusion does not block the SDF force field. It still reports
contact on an occluded installed patch. It does not report contact on an
uninstrumented body or an object that was not configured with an SDF.

The sensor also does not infer task identity. It reports local simulated
contact fields for configured sensor/object pairs; recognizing that a scene is
CarryBox or KickBox, or choosing a task strategy, is a separate policy problem.
