# Isaac Carry-Scene Execution

This directory is for the real Isaac/PhysX path. It must not be replaced by a
toy browser, hand-drawn, or kinematic imitation.

Current project priority is direct Isaac scene construction for unknown-load
carrying. Official Arena/Galileo/GR00T assets are useful references and possible
baselines, but they are not a prerequisite for building the core scene.

## Current Main Path

Adaptive direct Isaac sweep:

```bash
STAMP=YYYYMMDD_adaptive_direct_sweep STEPS=180 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_sweep.sh
```

This is the current main direct-scene path. It runs the adaptive active-probing
scaffold across mass, size, COM offset, robot height, arm length, and payload
limits, then writes an aggregate JSON. It is useful for checking task
parameterization and morphology/load-dependent posture-selection plumbing. It
is still diagnostic-only: kinematic carrier, box pose following, no dynamic
robot balance, no contact grasping, no learned policy.

Validated sweep:

```text
experiments/outputs/adaptive_probe_carry_scene_sweeps/adaptive_probe_sweep_20260704_adaptive_direct_sweep1/adaptive_probe_sweep_summary.json
```

Result: 5/5 cases completed, drop cases 0, target threshold hits 5/5 at
0.08 m, minimum support-margin proxy 0.0769 m, strategy counts
`front_carry: 1`, `low_front_carry: 1`, `chest_supported_slow: 3`.

Experimental ANYmal articulation control diagnostic:

```bash
STAMP=YYYYMMDD_anymal_exp_art_smoke \
STEPS=180 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_anymal_experimental_articulation_smoke.sh
```

Current 2026-07-04 result is negative. The local ANYmal-C USD loads and exposes
12 DOFs, but the physics tensor entity remains invalid after warmup, so joint
state reads and position-target validation fail. Do not treat this as walking,
balancing, or carrying evidence, and do not wait on it before advancing the
direct Isaac task scaffold.

Standalone Isaac Sim core-World articulation diagnostic:

```bash
STAMP=YYYYMMDD_core_world_quad_smoke \
STEPS=240 PAYLOAD_MASS=4.0 TARGET_X=0.8 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh
```

This was added as a control-path repair attempt after the USD drive-target
route failed to produce travel and the IsaacLab-backed `SingleArticulation`
path hit `PhysxManager` compatibility errors. It avoids IsaacLab
`SimulationContext` and tensor APIs, builds the same kind of fixed-payload
articulated carrier directly in Isaac Sim core `World`, and tries
`SingleArticulation.apply_action()`. Current 2026-07-04 result is negative:
the direct `SimulationApp` route needed the local IsaacLab headless experience
to resolve extensions, then the custom articulation route stopped around
`SingleArticulation` wrapper/registration and produced no summary. The
AppLauncher variant reached `Creating SingleArticulation wrapper` and also
produced no summary. Do not rerun unchanged.

Dynamic USD/PhysX quadruped fixed-payload route:

```bash
STAMP=YYYYMMDD_usd_dynamic_quad_smoke \
STEPS=900 PAYLOAD_MASS=4.0 TARGET_X=1.0 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

This is the current direct dynamic Isaac attempt. It avoids the failing
IsaacLab tensor APIs and builds the robot from USD rigid bodies, revolute
joints, fixed joints, and USD Physics drive target attributes. The first target
is a dynamically simulated quadruped walking while carrying a physical box
payload fixed to its torso. It is not unknown-object grasping, active
free-contact carry, or learned control. Do not report it as success until a
compute-node smoke proves walking, balance, carried-box travel, no drops, and
no falls.

Current 2026-07-04 status: implemented but not yet walking. GPU articulation
root smoke hit PhysX direct-GPU `setDriveTarget()` errors and travel stayed
0. Disabling articulation root on GPU and CPU removed that error but still
produced 0 travel. The `CONTROL_MODE=core_articulation` path failed before
rollout because deprecated `SingleArticulation` is incompatible with the
current IsaacLab `PhysxManager` context. A later CPU + articulation-root +
USD-drive smoke completed 300/300 steps but still had torso/box travel 0.0.
The next step is to change runtime articulation-control API, not to tune gait
amplitudes.

Adaptive active-probing carry scaffold:

```bash
STAMP=20260704_adaptive_probe_carry_scene_smoke2_clean \
STEPS=300 BOX_MASS=8.0 BOX_SIZE_X=0.58 BOX_SIZE_Y=0.38 BOX_SIZE_Z=0.36 \
BOX_COM_X=0.04 ROBOT_HEIGHT=1.45 ROBOT_MASS=52.0 ARM_LENGTH=0.58 \
MAX_PAYLOAD=16.0 TARGET_X=2.15 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_scene.sh
```

This is now the fastest Isaac path for the actual research question. It builds
approach, probing, posture adjustment, lift, and carry phases; estimates load
from micro-lift/nudge proxy signals; chooses a carry posture from morphology
and load; and logs belief, support margin proxy, effort proxy, drops, target
distance, and strategy. It is diagnostic-only: the carrier is kinematic and the
box pose is followed after the decision.

Validated adaptive smokes:

```text
experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke2_clean/adaptive_probe_carry_scene_summary.json
experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke3_chest/adaptive_probe_carry_scene_summary.json
```

Results: the 8 kg default case selected `low_front_carry` and completed
300/300 steps with drop 0; the short-arm 11 kg larger-box case selected
`chest_supported_slow` and completed 260/260 steps with drop 0. These results
validate scene/plumbing and morphology-dependent posture selection only, not
dynamic humanoid balance, grasping, true contact carrying, or video-conditioned
RL.

Direct carrying-task scene smoke:

```bash
DEVICE=cuda:0 RENDER=0 STEPS=240 BOX_MASS=6.0 TARGET_X=2.2 \
STAMP=20260704_direct_carry_task_scene_smoke4 \
bash scripts/isaac/run_direct_carry_task_scene.sh
```

This is the current fastest Isaac path for task construction. It creates a
kinematic humanoid proxy with approach, probe, lift, and carry phases, a massed
box, a target marker, and CSV/summary metrics for box travel, drop events, and
support-margin proxy. It is diagnostic-only: it is not learned balance,
grasping, contact-rich carrying, or autonomous posture selection.

Current validated smoke:

```text
experiments/outputs/direct_carry_task_scene/20260704_direct_carry_task_scene_smoke4/direct_carry_task_scene_summary.json
```

Result: 240/240 steps, final carry phase, final box-to-target distance
`0.0348 m`, box drop events `0`, minimum support-margin proxy `0.111 m`.

Minimal direct-scene smoke:

```bash
DEVICE=cpu SKIP_ROBOT=1 RENDER=0 STEPS=120 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_YYYYMMDD \
bash scripts/isaac/run_minimal_carry_scene.sh
```

This must run inside a Curiosity-owned `tmux` session with a persistent Slurm
compute allocation. It refuses to run on `mgmtserver*`.

The minimal scene creates:

- Isaac/PhysX primitive floor;
- visual target marker;
- dynamic rigid carry box with configurable mass and size;
- optional G1 articulation after the box-only scene is verified;
- CSV state logging at `minimal_carry_scene_state.csv`.

`SKIP_ROBOT=1` is intentionally a scaffold and smoke test. It validates Isaac
scene construction and box physics only. It is not a carrying policy or success
claim.

Current validated smoke:

```text
/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv
```

In that run the box falls from about `z=0.446` to `z=0.175`, which is the
expected settle height for a `0.35 m` tall box on the floor.

Known limitation:

- `DEVICE=cpu SKIP_ROBOT=1` validates the direct scene and box rigid-body
  dynamics only.
- G1 WBC local assets have been downloaded under the local Isaac asset mirror
  and verified on a compute node with
  `scripts/isaac/check_g1_wbc_local_assets.py`.
- CPU G1 stand smokes reached scene setup and loaded both HomieV2 ONNX files
  but failed before stepping because PhysX tensor state access returned
  `Failed to get DOF positions from backend`, including after the
  Fabric/tensor logging patch.
- The direct scene now defaults to Fabric-enabled tensor state logging. Do not
  spend more time on CPU-only G1 articulation smoke in this cluster
  environment; the next required smoke is
  `DEVICE=cuda:0 SKIP_ROBOT=0 WBC_MODE=stand`.
- The launcher explicitly sets `LC_ALL` and `LANG` defaults to `C.UTF-8`.
  Without an explicit locale, the uv CPython used by the Isaac venv has shown
  intermittent startup failures while importing the standard `encodings`
  package.

## Official Arena Reference Path

Target:

```text
Isaac Lab-Arena Galileo G1 loco-manipulation task
+ Unitree G1 WBC embodiment
+ rigid brown box
+ PhysX simulation
+ official GR00T closed-loop policy checkpoint
+ recorded MP4 camera/viewport evidence
```

The official Arena task is:

```text
galileo_g1_locomanip_pick_and_place
```

The official tutorial describes it as G1 navigating through the Galileo lab
environment, picking up a brown box from a shelf, and placing it into a blue
bin. It uses PhysX at 200 Hz with 50 Hz control.

## Scripts

- `download_arena_g1_official_assets.sh`
  Downloads the official tuned checkpoint, and optionally the generated HDF5
  simulation dataset. This is a download-only script and may run on the login
  node when large downloads are acceptable.

- `run_arena_g1_locomanip_eval.sh`
  Runs the official GR00T server and Arena closed-loop policy evaluation inside
  the prepared Isaac/Arena and GR00T environments. This is the real
  physics-simulation command path and must run only inside a compute
  allocation.

- `build_minimal_carry_scene.py`
  Builds the direct Isaac carry scene and writes pose-state CSV evidence. Use
  `--skip-robot` to avoid G1/Arena imports while validating the basic box scene.
  Use `--wbc-mode stand|walk` for the official Arena HomieV2 WBC smoke.

- `build_direct_carry_task_scene.py`
  Current direct task-scene diagnostic. It avoids the blocked G1 articulation
  tensor path and builds a kinematic humanoid proxy carrying sequence in Isaac:
  approach, probe, lift, carry, target, CSV state log, and summary JSON. Its
  results must not be reported as humanoid locomotion, balance, grasping, or
  learned carrying success.

- `run_direct_carry_task_scene.sh`
  Compute-node launcher for the direct carrying-task scene. Set `STEPS`,
  `BOX_MASS`, `BOX_SIZE_X/Y/Z`, `TARGET_X`, and `RENDER`. It refuses to run on
  login nodes.

- `build_usd_dynamic_quadruped_carry_scene.py`
  Non-tensor dynamic Isaac route. It creates a four-legged robot directly from
  USD/PhysX rigid bodies and joints, attaches a physical box by fixed joint,
  drives the legs through USD Physics drive target attributes, and logs torso
  pose, box pose, travel, tilt, fall, and drop metrics.

- `run_usd_dynamic_quadruped_carry_scene.sh`
  Compute-node launcher for the USD dynamic quadruped route. Key variables:
  `STEPS`, `PAYLOAD_MASS`, `TARGET_X`, `GAIT_FREQUENCY`,
  `HIP_AMPLITUDE_DEG`, `KNEE_AMPLITUDE_DEG`, `DEVICE`, and `RENDER`.

- `run_anymal_experimental_articulation_smoke.py`
  Official-asset articulation control diagnostic. It loads the local ANYmal-C
  USD asset, wraps it with `isaacsim.core.experimental.prims.Articulation`,
  applies position targets to a few DOFs, and logs measured joint motion. This
  is a control-path smoke only.

- `run_anymal_experimental_articulation_smoke.sh`
  Compute-node launcher for the experimental ANYmal articulation smoke.

- `build_core_world_dynamic_quadruped_carry_scene.py`
  Standalone Isaac Sim core-World control-path diagnostic. It creates a custom
  USD articulated quadruped with a fixed payload, registers it through
  `SingleArticulation`, applies joint-position actions, and logs joint motion,
  torso travel, box travel, falls, and drops. This is not a learned carrying
  method.

- `run_core_world_dynamic_quadruped_carry_scene.sh`
  Compute-node launcher for the standalone core-World diagnostic. Key
  variables: `STEPS`, `PAYLOAD_MASS`, `TARGET_X`, `GAIT_FREQUENCY`,
  `HIP_AMPLITUDE_DEG`, `KNEE_AMPLITUDE_DEG`, `DEVICE`, and `RENDER`.

- `build_adaptive_probe_carry_scene.py`
  Current adaptive task-scene scaffold. It avoids external models and directly
  represents active probing, load belief, morphology-aware posture selection,
  and carrying metrics in Isaac. It must not be reported as dynamic robot
  locomotion, learned control, or true contact-box carrying.

- `run_adaptive_probe_carry_scene.sh`
  Compute-node launcher for the adaptive scaffold. Key variables:
  `BOX_MASS`, `BOX_SIZE_X/Y/Z`, `BOX_COM_X/Y/Z`, `ROBOT_HEIGHT`,
  `ROBOT_MASS`, `ARM_LENGTH`, `MAX_PAYLOAD`, `TARGET_X`, `DEVICE`, and
  `RENDER`.

- `run_adaptive_probe_carry_sweep.sh`
  Compute-node launcher for a 5-case adaptive scaffold sweep over box and robot
  parameters. It writes per-case outputs and an aggregate summary under
  `experiments/outputs/adaptive_probe_carry_scene_sweeps/`.

- `aggregate_adaptive_probe_sweep.py`
  Lightweight summary aggregator for adaptive sweep case outputs. It records
  completion, drop cases, target-threshold hits, support-margin proxy, and
  strategy counts.

- `run_anymal_payload_carry.py`
  Official ANYmal-C RSL-RL locomotion payload diagnostic. Current manager-based
  PhysX smokes fail before rollout with `Failed to get DOF velocities from
  backend`, matching the broader IsaacLab articulation tensor issue. Treat all
  outputs as diagnostics, not carrying success.

- `build_contact_carry_scene.py`
  Low-level contact diagnostic with kinematic palms and a dynamic box. Smoke1
  was negative: the dynamic box did not move when palms were moved by USD xform
  edits.

- `build_contact_carry_rigid_scene.py`
  Follow-up low-level contact diagnostic that writes palm poses and velocities
  through the RigidObject simulation API. GPU and CPU smokes both failed with
  `Failed to set rigid body transforms in backend`; do not rerun unchanged.

- `run_minimal_carry_scene.sh`
  Compute-node launcher for the direct Isaac carry scene. Set `SKIP_ROBOT=1`
  for box-only smoke, `WBC_MODE=stand` for G1 standing, and
  `ATTACH_BOX=fixed_torso` only for a labeled payload-balance diagnostic.
  `SKIP_EXPLICIT_STATE_RESET=1` is a diagnostic-only switch for isolating
  Articulation/RigidObject state-write failures; it must not be used as
  carrying success evidence. Box pose can be adjusted with
  `BOX_POS_X/Y/Z`; fixed payload joint placement can be adjusted with
  `ATTACH_LOCAL_POS0_X/Y/Z`.

- `run_g1_wbc_smoke_sequence.sh`
  Compute-node-only sequence runner. It runs stand first, then walk, then a
  fixed-torso payload balance diagnostic. It stops at the first failure. Set
  `RUN_PAYLOAD=0` to run only stand and walk while debugging locomotion. By
  default it runs `check_carry_smoke_summary.py` after each completed smoke;
  set `CHECK_SUMMARY=0` only when debugging summary generation itself.

- `check_g1_wbc_local_assets.py`
  Compute-node-only check that loads the local G1 URDF/mesh assets and official
  HomieV2 ONNX policies without launching Isaac Sim.

- `check_carry_smoke_summary.py`
  Lightweight post-run checker for `minimal_carry_scene_summary.json`. It only
  checks diagnostic gates such as completed steps, fall events, box-drop events,
  and minimum robot travel distance; it is not a full success verifier.

- `build_proxy_carry_scene.py`
  Diagnostic-only Isaac scaffold that avoids the currently failing G1
  articulation tensor path. It creates a kinematic carrier, pose-follow payload
  box, target marker, CSV state log, and summary JSON. It is useful for keeping
  the carry-scene output skeleton moving, but it is not humanoid walking,
  balancing, grasping, or carrying evidence.

## Cluster Rule

Do not run simulation, rendering, model loading, or evaluation on
`mgmtserver02` or any login node. Run `run_arena_g1_locomanip_eval.sh` inside a
Curiosity-owned `tmux` session with a persistent `srun` or `salloc` compute
allocation.

## Required Prepared Environment

The scripts assume the official Arena and GR00T environments are already
installed. They do not install dependencies, build Docker images, create venvs,
or solve packages.

Pinned official code prepared under:

- `external/IsaacLab-Arena`
- `external/IsaacLab-Arena/submodules/IsaacLab` at `55df2c3`
- `external/IsaacLab-Arena/submodules/Isaac-GR00T` at `e29d8fc`

Prepared environments:

- Isaac/Arena: `/public/home/yanhongru/envs/isaac_arena_py312`
- GR00T server: `/public/home/yanhongru/envs/gr00t_n16_py310`

Prepared official inference checkpoint:

```text
/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000
```

Prepared local USD mirror for compute nodes that cannot open the public S3
asset URLs directly:

```text
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/galileo_locomanip.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/brown_box.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/blue_sorting_bin.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy/
```

Current diagnostic status:

- Official GR00T checkpoint loads.
- Arena policy runner reaches scene creation with local USD paths.
- `omni.physics.tensors.impl.api` import mismatch is patched locally in the
  Omniverse extension cache as a compatibility shim.
- No MP4 success evidence exists yet.
- Latest blocker: official Galileo scene PhysX mesh cooking stalls on shelf
  collision meshes after repeated CPU fallback warnings.
- Direct G1 CPU and GPU smokes currently fail before stepping with
  `Failed to get DOF positions from backend`, including after
  `InteractiveScene`, `SKIP_EXPLICIT_STATE_RESET=1`, and
  `DISABLE_USD_PHYSICS_UPDATES=1` diagnostics. This is a G1/IsaacLab tensor
  backend blocker, not a WBC success or failure result.
- Current pivot: use the adaptive active-probing Isaac scaffold as the main
  task-construction path while the G1 articulation tensor path is repaired or
  replaced with a better official Arena entry path.
