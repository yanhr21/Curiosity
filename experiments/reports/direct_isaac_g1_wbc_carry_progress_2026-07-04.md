# Direct Isaac G1 WBC Carry Progress

Date: 2026-07-04.

## Objective

Build a real Isaac physics simulation in which a robot can walk, maintain
balance, and carry a box. Box-only or static-scene diagnostics are not success.

## Current Strategy

Use the direct Isaac scene as the active path:

- primitive floor and target marker;
- physical carry box with mass, size, friction, gravity, and collision;
- G1 robot asset from the local Isaac asset mirror;
- Arena's official G1 HomieV2 WBC stand/walk ONNX policy for locomotion;
- later: contact/carry objective and box interaction.

The official Arena Galileo/GR00T workflow remains a reference/baseline, not the
active blocker.

## Implemented This Turn

- Added the 2026-07-04 execution objective and completion gate to `AGENTS.md`.
- Moved login-node refusal in `scripts/isaac/build_minimal_carry_scene.py`
  before Isaac `AppLauncher` construction.
- Added `--wbc-mode none|stand|walk`, `--walk-command`, and
  `--base-height-command`.
- Added `G1WBCDriver`, a direct adapter around Arena's official G1 HomieV2 WBC
  policy. This is not a toy gait controller.
- Exposed launcher environment variables:
  `WBC_MODE`, `WALK_COMMAND_X`, `WALK_COMMAND_Y`, `WALK_COMMAND_YAW`,
  `BASE_HEIGHT_COMMAND`, `WBC_ASSET_ROOT`, `ATTACH_BOX`, and
  `ATTACH_BODY_PATH`.
- Added `--attach-box fixed_torso` as a physical payload diagnostic. This is
  only for balance-under-load testing and is not a grasp success claim.
- Downloaded the official Arena G1 HomieV2 WBC local assets needed to avoid
  compute-node network fetches:
  `stand.onnx`, `walk.onnx`, `g1_29dof_with_hand.urdf`, and 50 referenced STL
  mesh files under the local Isaac asset mirror.
- Added `scripts/isaac/check_g1_wbc_local_assets.py`, a compute-node-only
  WBC asset check.
- Switched the direct scene back to Fabric-enabled tensor state logging by
  default. USD pose fallback remains diagnostic-only.
- Added `SKIP_EXPLICIT_STATE_RESET=1` as a diagnostic-only launcher switch to
  isolate failures caused by explicit Articulation/RigidObject reset writes.
  This cannot be used as carrying success evidence.
- Added CSV and JSON diagnostic metrics for fall flag, box drop flag, robot
  travel distance, box travel distance, and box-to-target distance.
- Forced `LC_ALL=C.UTF-8` and `LANG=C.UTF-8` in the launcher after uv CPython
  intermittently failed to import the standard `encodings` package without an
  explicit locale.

## Prior Verified Evidence

CPU PhysX box-only smoke:

```text
experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv
```

The carry box fell from approximately `z=0.446` to `z=0.175`, the expected
settle height for a `0.35 m` tall box on the floor. This validates box rigid
body gravity/collision on CPU PhysX only.

Compute-node WBC local asset check:

```text
tmux: curiosity_g1_wbc_cpu_0704
job: 164819 on server53
command:
PYTHONPATH=/public/home/yanhongru/Curiosity/external/IsaacLab-Arena:${PYTHONPATH:-} \
/public/home/yanhongru/envs/isaac_arena_py312/bin/python \
scripts/isaac/check_g1_wbc_local_assets.py
```

Result:

```text
[OK] Local G1 WBC assets loaded.
[OK] Robot DOFs: 43
[OK] Policy: G1DecoupledWholeBodyPolicy
```

This validates local URDF/mesh/ONNX loading only. It is not simulation evidence.

Failed CPU G1 stand smoke:

```text
tmux: curiosity_g1_wbc_cpu_0704
job: 164819 on server53
log: logs/minimal_carry_scene/minimal_carry_scene_20260704_150723.log
output: experiments/outputs/minimal_carry_scene/g1_wbc_stand_20260704_240steps
```

Result: the scene reached setup and loaded both HomieV2 ONNX policies, but
failed before stepping because the PhysX tensor simulation view was invalidated
when reading DOF positions:

```text
Exception: Failed to get DOF positions from backend
```

This is a failed diagnostic, not a WBC policy failure and not a carrying result.
The scene code was subsequently changed to use Fabric-enabled tensor state
logging by default.

Failed CPU G1 stand smoke after Fabric/tensor logging patch:

```text
tmux: curiosity_g1_wbc_cpu_fabric_0704
job: 164824 on server53
log: logs/minimal_carry_scene/minimal_carry_scene_20260704_151309.log
output: experiments/outputs/minimal_carry_scene/g1_wbc_stand_cpu_fabric_20260704_120steps
```

Result: same PhysX tensor backend failure:

```text
Exception: Failed to get DOF positions from backend
```

Conclusion: in this cluster environment, CPU-only G1 articulation tensor
simulation is not a valid execution path for the direct robot smoke. The next
validation must run under a GPU allocation.

## Pending Compute Smoke

Slurm allocation:

```text
tmux: curiosity_g1_wbc_direct_0704_short
job: 164814
status: queued, reason `(Priority)`
scheduled node: server46 as of 16:02
scheduled start: 2026-07-04 17:48:21 as of 16:02
```

Planned command:

```bash
DEVICE=cuda:0 SKIP_ROBOT=0 WBC_MODE=stand RENDER=0 STEPS=240 \
OUTPUT_DIR=/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene/g1_wbc_stand_20260704_240steps \
bash scripts/isaac/run_minimal_carry_scene.sh
```

Automation watcher:

```text
tmux: curiosity_gpu_runner_watch_0704
action: polls job 164814 every 30 seconds and sends the stand/walk sequence
        to curiosity_g1_wbc_direct_0704_short after the job enters RUNNING.
command sent on RUNNING:
RUN_PAYLOAD=0 DEVICE=cuda:0 STAMP=20260704_gpu164814 \
bash scripts/isaac/run_g1_wbc_smoke_sequence.sh
```

Additional Slurm checks:

```text
2026-07-04 16:02:
- Existing Curiosity GPU job 164814 remains the best available path.
- A smaller test-only gpu request estimated start at 2026-07-06 04:28:25.
- test/gaosh/engram partitions rejected the account/partition combination.
- Pending Reflex job 164820 is in /public/home/yanhongru/ICLR2027/Reflex and
  was inspected read-only only; it remains untouched under the resource
  exclusion rule.
```

## Current Non-Completion Status

Incomplete. No robot walking, balancing, or box-carrying evidence exists yet in
the direct Isaac scene.

## Official Isaac Policy Route

After the custom G1/ANYmal tensor routes and hand-authored USD drive routes
failed to produce usable dynamic locomotion, the current Isaac-native route is
to start from NVIDIA's installed policy examples:

```text
script: scripts/isaac/run_official_policy_locomotion_smoke.py
launcher: scripts/isaac/run_official_policy_locomotion_smoke.sh
source: installed isaacsim.robot.policy.examples Go2/H1 flat-terrain wrappers
assets: /public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0
```

Implemented changes:

- Manually exposed the installed extension namespace paths for
  `isaacsim.robot.policy.examples`, `isaacsim.core.experimental.prims`, and
  `isaacsim.core.experimental.utils`.
- Downloaded local official Go2/H1 USD, policy checkpoint, and environment
  config assets so compute nodes do not need network access.
- Replaced blocking standalone `new_stage()` and `next_update_async()` calls
  with reuse of the AppLauncher-created stage plus synchronous
  `simulation_app.update()`.
- Added `PAYLOAD_MODE=fixed_base`, which creates a physical rigid Go2 carry
  box and a fixed joint to `/World/Go2/Geometry/base`.

No-GPU-allocation diagnostic status:

```text
stamp: 20260704_official_go2_policy_sync_smoke7
job: 165202
node: server05
command:
STAMP=20260704_official_go2_policy_sync_smoke7 ROBOT=go2 STEPS=160 \
COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_official_policy_locomotion_smoke.sh
```

The run passed the previous `new_stage()` blocker and reached:

```text
[PROGRESS] Reusing current USD stage
```

It then stalled before local ground creation. Additional progress points were
added around `SimulationManager.set_backend`, `set_physics_sim_device`,
`set_physics_dt`, `PhysicsScene` authoring, and ground creation. This is still
a running diagnostic path, not walking or carrying evidence. If the no-payload
Go2 run passes, the immediate next run is the same launcher with:

Follow-up diagnostics in the server05 no-GPU allocation:

```text
diag13:
  error: AttributeError: type object 'PhysxManager' has no attribute
         'get_active_physics_engine'
  fix: add a local compatibility shim returning "physx"

diag14:
  error: ValueError: Invalid device identifier: cuda:0
  cause: AppLauncher/Articulation default still tried to use cuda:0 even
         though the run requested DEVICE=cpu and the allocation had no GPU
  fix: force IsaacLab PhysicsManager device to cpu in CPU diagnostics

diag15:
  progress: Go2 policy object was created
  blocker: articulation physics simulation view was invalid before
           `robot.initialize()`
  log line:
    Invalid physics simulation view. Articulation
    (['/World/Go2/Geometry/base']) will not be initialized
  fix: add a pre-initialize validity check so the script writes a failure
       summary instead of hanging
```

Conclusion: CPU/no-GPU-allocation can validate local assets and policy wrapper
construction, but it is not enough to initialize the Go2 articulation physics
view. The next active run requests a real GPU:

```text
stamp: 20260704_official_go2_policy_real_gpu_diag16
job: 165252
request: srun -p gpu --gres=gpu:1 --time=00:25:00
command:
STAMP=20260704_official_go2_policy_real_gpu_diag16 ROBOT=go2 STEPS=160 \
COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_official_policy_locomotion_smoke.sh
status when recorded: pending, reason `(Priority)`
```

If the real-GPU no-payload Go2 run passes, the immediate next run is the same
launcher with:

```bash
PAYLOAD_MODE=fixed_base PAYLOAD_MASS=2.0 ROBOT=go2 COMMAND_X=0.6
```

That next run can only be claimed as fixed-payload balance-under-load evidence
if robot and payload travel are nonzero and fall/drop events remain zero.

Actual real-GPU result:

```text
stamp: 20260704_official_go2_policy_real_gpu_diag16
node: server10
gpu: NVIDIA H200
result: negative
```

The Go2 policy object was created on CUDA, but before policy initialization the
articulation physics view was invalid:

```text
[PROGRESS] Policy robot object created: go2
[PROGRESS] Pre-initialize physics view: valid=False initialized=False
[ERROR] Articulation physics tensor entity invalid before initialize
```

Enabling explicit `SimulationManager.set_backend("torch")` in
`20260704_official_go2_policy_real_gpu_configsm_diag17` exited before rollout,
the same as the CPU/no-GPU diagnostics. This is not walking evidence.

Pure `SimulationApp` follow-up:

```text
script: scripts/isaac/run_official_policy_locomotion_simapp_smoke.py
launcher: scripts/isaac/run_official_policy_locomotion_simapp_smoke.sh
```

The default pure Isaac Sim base experience failed because the local registry
mirror does not include `isaacsim.anim.robot.schema`. Using the local
IsaacLab headless experience starts successfully and creates the Go2 policy
object on H200, but `simapp_diag5` through `simapp_diag8` still report
`physics_tensor_entity_invalid` before initialization. Explicit warmup/view
creation and explicit stage-context binding do not make the tensor entity
valid. `simapp_diag9` with explicit `SimulationManager.set_backend` again
exits before rollout. This route is therefore still a negative diagnostic, not
a locomotion or carrying result.

## GPU Allocation Results

GPU allocation:

```text
tmux: curiosity_g1_wbc_direct_0704_short
job: 164814
node: server27
gpu: NVIDIA H200
```

Failed GPU G1 WBC stand smoke:

```text
command:
RUN_PAYLOAD=0 DEVICE=cuda:0 STAMP=20260704_gpu164814 \
bash scripts/isaac/run_g1_wbc_smoke_sequence.sh

log:
logs/minimal_carry_scene/minimal_carry_scene_20260704_160736.log

output:
experiments/outputs/minimal_carry_scene/g1_wbc_stand_20260704_gpu164814_240steps
```

Result: the scene reached setup and loaded both official HomieV2 ONNX policies,
but failed before the first step:

```text
Exception: Failed to get DOF positions from backend
```

Failed GPU diagnostics after that:

```text
WBC_MODE=none SKIP_EXPLICIT_STATE_RESET=1
log: logs/minimal_carry_scene/minimal_carry_scene_20260704_161531.log
output: experiments/outputs/minimal_carry_scene/g1_none_skipreset_20260704_gpu164814_80steps

InteractiveScene + WBC_MODE=none + SKIP_EXPLICIT_STATE_RESET=1
log: logs/minimal_carry_scene/minimal_carry_scene_20260704_161723.log
output: experiments/outputs/minimal_carry_scene/g1_interactive_none_skipreset_20260704_gpu164814_80steps

InteractiveScene + WBC_MODE=none + SKIP_EXPLICIT_STATE_RESET=1
+ DISABLE_USD_PHYSICS_UPDATES=1
log: logs/minimal_carry_scene/minimal_carry_scene_20260704_161834.log
output: experiments/outputs/minimal_carry_scene/g1_interactive_no_usd_sync_20260704_gpu164814_60steps
```

All three failed at the same core point: the G1 actuator path tries to read
`joint_pos`, which calls `get_dof_positions()` on an invalidated PhysX tensor
simulation view. This makes the current direct G1 articulation path unusable
until the tensor backend issue is fixed or a different official Arena entry path
is used. This is not a WBC-policy failure and not carrying evidence.

## Proxy Isaac Scene

To keep the Isaac scene construction moving without making G1 a hard blocker, a
diagnostic-only proxy scene was added:

```text
script: scripts/isaac/build_proxy_carry_scene.py
scene: kinematic carrier + pose-follow payload box + target + CSV/JSON metrics
claim level: diagnostic only; not humanoid walking, not balance, not grasping
```

First proxy fixed-joint attempt:

```text
log: logs/proxy_carry_scene/proxy_carry_scene_20260704_gpu164814_240steps.log
output: experiments/outputs/proxy_carry_scene/proxy_fixed_payload_20260704_gpu164814_240steps
```

Result: carrier moved 0.418 m, but the box did not move through the fixed joint.
This exposed that directly editing the USD transform was not acting as a useful
PhysX kinematic target for the payload joint in this scaffold.

Second proxy pose-follow payload attempt:

```text
log: logs/proxy_carry_scene/proxy_carry_scene_20260704_pose_follow_gpu164814_240steps.log
output: experiments/outputs/proxy_carry_scene/proxy_pose_follow_payload_20260704_gpu164814_240steps
summary: experiments/outputs/proxy_carry_scene/proxy_pose_follow_payload_20260704_gpu164814_240steps/proxy_carry_scene_summary.json
```

Summary:

```json
{
  "completed_steps": 240,
  "max_carrier_travel_xy_m": 0.41846072725738165,
  "max_box_travel_xy_m": 0.4184607272573816,
  "box_drop_events": 0,
  "min_box_target_distance_xy_m": 0.9618416606992946,
  "scene_type": "kinematic_proxy_carrier_pose-follow_payload",
  "success_claim": "diagnostic_only_not_humanoid_control_or_grasp"
}
```

This validates only the Isaac output skeleton: scene creation, target, payload
pose path, metric logging, and summary writing. It must not be used as evidence
that G1 walks, balances, grasps, or carries a box.

## Direct Carrying-Task Scene

To avoid waiting on external models or blocked G1 articulation tensors, a
direct Isaac task-scene diagnostic was added:

```text
script: scripts/isaac/build_direct_carry_task_scene.py
launcher: scripts/isaac/run_direct_carry_task_scene.sh
scene: kinematic humanoid proxy + massed carry box + target + CSV/JSON metrics
phases: approach, probe, lift, carry
claim level: diagnostic only; not learned balance, not contact-rich grasping,
             not autonomous posture selection, and not true robot carrying
```

First structural bug:

```text
log: logs/direct_carry_task_scene/direct_carry_task_scene_20260704_direct_carry_task_scene_smoke2.log
result: failed before scene construction
error: AttributeError: 'NoneType' object has no attribute 'GetPrimAtPath'
cause: cuboids were spawned before SimulationContext had an active USD stage
fix: move design_scene() inside build_simulation_context(...)
```

Validated smoke:

```bash
tmux: curiosity_direct_carry_0704
job: 164957
node: server46
command:
STAMP=20260704_direct_carry_task_scene_smoke4 \
STEPS=240 BOX_MASS=6.0 WALK_SPEED=0.32 TARGET_X=2.2 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_direct_carry_task_scene.sh
```

Output:

```text
log: logs/direct_carry_task_scene/direct_carry_task_scene_20260704_direct_carry_task_scene_smoke4.log
summary: experiments/outputs/direct_carry_task_scene/20260704_direct_carry_task_scene_smoke4/direct_carry_task_scene_summary.json
state_csv: experiments/outputs/direct_carry_task_scene/20260704_direct_carry_task_scene_smoke4/direct_carry_task_scene_state.csv
```

Summary:

```json
{
  "completed_steps": 240,
  "final_phase": "carry",
  "final_box_target_distance_xy_m": 0.034785427018238835,
  "box_drop_events": 0,
  "max_box_travel_xy_m": 1.474785427018239,
  "max_torso_travel_xy_m": 1.764785427018239,
  "min_support_margin_m": 0.11122338693368139,
  "scene_type": "direct_isaac_kinematic_humanoid_proxy_carry_task",
  "success_claim": "diagnostic_only_not_learned_balance_or_grasp_success",
  "kinematic_box_pose_following": true
}
```

Current status: this is now the runnable Isaac task-construction baseline. It
keeps progress moving on scene phases, metrics, and target completion, while
the real robot articulation/control path remains unresolved.

## ANYmal Locomotion Payload Diagnostics

To avoid waiting on humanoid-specific G1 fixes, an official ANYmal locomotion
policy path was added:

```text
script: scripts/isaac/run_anymal_payload_carry.py
launcher: scripts/isaac/run_anymal_payload_carry.sh
policy: official IsaacLab RSL-RL ANYmal-C flat velocity checkpoint
claim level: locomotion payload diagnostic only; not grasp/contact box carrying
```

The local official checkpoint and assets are present:

```text
.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Anymal-C-v0/checkpoint.pt
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd
/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/Props/instanceable_meshes.usd
```

Clean manager-based PhysX no-payload baseline:

```text
tmux: curiosity_anymal_clean_0704
job: 164969
log: logs/anymal_payload_carry/anymal_payload_carry_20260704_anymal_clean_physx_nopayload_smoke14.log
command:
STAMP=20260704_anymal_clean_physx_nopayload_smoke14 \
STEPS=40 NUM_ENVS=1 PAYLOAD_MASS=0.0 COMMAND_X=0.25 DEVICE=cuda:0 \
PHYSICS_BACKEND=physx DISABLE_FABRIC=1 USE_PRETRAINED_CHECKPOINT=0 \
CHECKPOINT=/public/home/yanhongru/Curiosity/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Anymal-C-v0/checkpoint.pt \
bash scripts/isaac/run_anymal_payload_carry.sh
```

Result: the scene reached one environment and simulation start, then failed
inside IsaacLab articulation tensor state update:

```text
Exception: Failed to get DOF velocities from backend
```

This reproduces the same tensor-backend class of failure seen with G1. It is
not a payload or policy failure.

Direct ANYmal task check:

```text
log: logs/anymal_payload_carry/anymal_payload_carry_20260704_anymal_direct_physx_nopayload_smoke15.log
task: Isaac-Velocity-Flat-Anymal-C-Direct-v0
```

Result: reached direct task `gym.make` but did not reach rollout before
interruption. No walking evidence was produced.

## Low-Level Contact Carry Diagnostics

To keep contact-carry progress moving without relying on the broken
articulation tensor path, a low-level Isaac contact scene was added:

```text
script: scripts/isaac/build_contact_carry_scene.py
launcher: scripts/isaac/run_contact_carry_scene.sh
scene: kinematic palms + dynamic box + target
box pose followed: false
claim level: diagnostic only; not dynamic robot balance or learned carrying
```

Smoke1:

```text
log: logs/contact_carry_scene/contact_carry_scene_20260704_contact_carry_smoke1.log
summary: experiments/outputs/contact_carry_scene/20260704_contact_carry_smoke1/contact_carry_scene_summary.json
```

Summary:

```json
{
  "completed_steps": 420,
  "max_box_travel_xy_m": 0.0,
  "max_box_lift_m": 0.0,
  "final_box_target_distance_xy_m": 0.9299999952316285,
  "box_pose_followed": false
}
```

Negative result: directly editing USD transforms for kinematic palms did not
produce effective PhysX contact carrying; the dynamic box remained stationary.

Follow-up implementation added:

```text
script: scripts/isaac/build_contact_carry_rigid_scene.py
launcher: scripts/isaac/run_contact_carry_rigid_scene.sh
change: write kinematic palm pose/velocity through RigidObject simulation API
status: tested; failed in current PhysX tensor backend
```

RigidObject GPU smoke:

```text
tmux: curiosity_contact_rigid_0704
job: 164972
log: logs/contact_carry_rigid_scene/contact_carry_rigid_scene_20260704_contact_carry_rigid_smoke1.log
command:
STAMP=20260704_contact_carry_rigid_smoke1 STEPS=420 BOX_MASS=3.0 TARGET_X=1.35 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_contact_carry_rigid_scene.sh
```

Result:

```text
Exception: Failed to set rigid body transforms in backend
```

RigidObject CPU smoke on the same compute allocation:

```text
log: logs/contact_carry_rigid_scene/contact_carry_rigid_scene_20260704_contact_carry_rigid_cpu_smoke2.log
command:
STAMP=20260704_contact_carry_rigid_cpu_smoke2 STEPS=240 BOX_MASS=3.0 TARGET_X=1.35 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_contact_carry_rigid_scene.sh
```

Result: same backend failure:

```text
Exception: Failed to set rigid body transforms in backend
```

Current interpretation: IsaacLab articulation tensors and IsaacLab rigid-object
write tensors are both unreliable in this environment. USD-transform-only
scenes can step and log, but they do not provide effective contact carrying.
The next low-level path should use a pure Omni/PhysX non-tensor kinematic-target
API or repair the tensor backend before official articulated locomotion can be
used as evidence.

## Adaptive Active-Probing Isaac Scaffold

After the contact and articulation routes failed in the current IsaacLab tensor
backend, the active path was changed to direct Isaac scene construction instead
of waiting for external models.

Added:

```text
script: scripts/isaac/build_adaptive_probe_carry_scene.py
launcher: scripts/isaac/run_adaptive_probe_carry_scene.sh
scene: kinematic humanoid proxy + massed box + target + probing/posture/carry metrics
claim level: diagnostic scaffold only
```

The scaffold includes:

- approach, probe, posture-adjust, lift, and carry phases;
- micro-lift/nudge proxy signals for load and COM belief;
- posture selection from robot morphology and estimated load;
- front/low/chest-supported carry strategy options;
- support-margin proxy, effort proxy, energy proxy, drop flag, and target
  distance logging.

Default 8 kg smoke:

```text
tmux: curiosity_mujoco_payload_0704
job: 164974
node: server46
log: logs/adaptive_probe_carry_scene/adaptive_probe_carry_scene_20260704_adaptive_probe_carry_scene_smoke2_clean.log
summary: experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke2_clean/adaptive_probe_carry_scene_summary.json
command:
STAMP=20260704_adaptive_probe_carry_scene_smoke2_clean STEPS=300 BOX_MASS=8.0 \
BOX_SIZE_X=0.58 BOX_SIZE_Y=0.38 BOX_SIZE_Z=0.36 BOX_COM_X=0.04 \
ROBOT_HEIGHT=1.45 ROBOT_MASS=52.0 ARM_LENGTH=0.58 MAX_PAYLOAD=16.0 \
TARGET_X=2.15 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_scene.sh
```

Result:

```json
{
  "completed_steps": 300,
  "selected_strategy": "low_front_carry",
  "box_drop_events": 0,
  "final_box_target_distance_xy_m": 0.029750006998684242,
  "max_box_travel_xy_m": 1.3997500069986841,
  "min_support_margin_m": 0.06608032522473051
}
```

Short-arm heavy-box smoke:

```text
tmux: curiosity_mujoco_payload_0704
job: 164974
node: server46
log: logs/adaptive_probe_carry_scene/adaptive_probe_carry_scene_20260704_adaptive_probe_carry_scene_smoke3_chest.log
summary: experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke3_chest/adaptive_probe_carry_scene_summary.json
command:
STAMP=20260704_adaptive_probe_carry_scene_smoke3_chest STEPS=260 BOX_MASS=11.0 \
BOX_SIZE_X=0.68 BOX_SIZE_Y=0.42 BOX_SIZE_Z=0.40 BOX_COM_X=0.02 \
ROBOT_HEIGHT=1.25 ROBOT_MASS=44.0 ARM_LENGTH=0.48 MAX_PAYLOAD=15.0 \
TARGET_X=1.85 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_scene.sh
```

Result:

```json
{
  "completed_steps": 260,
  "selected_strategy": "chest_supported_slow",
  "box_drop_events": 0,
  "final_box_target_distance_xy_m": 0.009719068369646866,
  "max_box_travel_xy_m": 1.079719068369647,
  "min_support_margin_m": 0.08541365855806393
}
```

Interpretation: this is useful forward progress because it proves the Isaac
task scaffold now contains active probing structure and morphology-dependent
posture selection. It is not evidence of learned dynamic humanoid walking,
balance, grasping, true contact carrying, or video-conditioned RL, because the
carrier is kinematic and the box pose is followed after the decision.

## USD Dynamic Quadruped Carry Attempt

To move toward actual dynamic walking in Isaac without waiting for external
models, a non-tensor USD/PhysX quadruped route was added:

```text
script: scripts/isaac/build_usd_dynamic_quadruped_carry_scene.py
launcher: scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
scene: dynamic torso, four driven legs, fixed physical box payload
claim level: dynamic fixed-payload diagnostic only if walking is verified
```

Smoke1, articulation root on GPU:

```text
tmux: curiosity_usd_quad_0704
job: 164978
node: server02
summary: experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke1/usd_dynamic_quadruped_carry_summary.json
command:
STAMP=20260704_usd_dynamic_quad_payload_smoke1 STEPS=900 PAYLOAD_MASS=4.0 \
TARGET_X=1.0 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 \
KNEE_AMPLITUDE_DEG=16.0 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

Result: completed 900/900, falls 0, drops 0, but torso and box travel stayed
0.0. PhysX emitted repeated:

```text
PxArticulationJointReducedCoordinate::setDriveTarget(): it is illegal to call
this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled
```

Smoke2, articulation root disabled on GPU:

```text
summary: experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke2_noartroot/usd_dynamic_quadruped_carry_summary.json
command:
STAMP=20260704_usd_dynamic_quad_payload_smoke2_noartroot STEPS=600 \
PAYLOAD_MASS=4.0 TARGET_X=0.8 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 \
KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=0 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

Result: completed 600/600, falls 0, drops 0, but travel stayed 0.0.

Smoke3, articulation root disabled on CPU PhysX inside the same compute
allocation:

```text
summary: experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke3_cpu/usd_dynamic_quadruped_carry_summary.json
command:
STAMP=20260704_usd_dynamic_quad_payload_smoke3_cpu STEPS=400 \
PAYLOAD_MASS=4.0 TARGET_X=0.6 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 \
KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=0 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

Result: completed 400/400, falls 0, drops 0, but travel stayed 0.0.

Smoke4, Isaac Sim `SingleArticulation.apply_action` control path:

```text
log: logs/usd_dynamic_quadruped_carry_scene/usd_dynamic_quadruped_carry_scene_20260704_usd_dynamic_quad_payload_smoke4_core_cpu.log
command:
STAMP=20260704_usd_dynamic_quad_payload_smoke4_core_cpu STEPS=120 \
PAYLOAD_MASS=4.0 TARGET_X=0.3 GAIT_FREQUENCY=1.0 HIP_AMPLITUDE_DEG=14.0 \
KNEE_AMPLITUDE_DEG=12.0 ARTICULATION_ROOT=1 CONTROL_MODE=core_articulation \
DEVICE=cpu RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

Result: failed before rollout:

```text
AttributeError: type object 'PhysxManager' has no attribute '_get_backend_utils'
```

Current interpretation: the USD dynamic robot scene is now present, but no
dynamic walking evidence exists yet. Further progress requires changing the
runtime control mechanism, such as repairing the Isaac Sim
`SingleArticulation` compatibility in the IsaacLab context, using
`isaacsim.core.experimental.prims.Articulation`, or locating a functioning
dynamic-control interface. Do not keep tuning gait amplitudes while travel is
exactly zero.

## Standalone Isaac Core-World Control Path

To stop waiting on external model/code paths, a direct Isaac Sim core-World
diagnostic was added:

```text
script: scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py
launcher: scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh
scene: custom USD articulated quadruped, fixed physical box payload
control: SingleArticulation.apply_action() in Isaac Sim core World
claim level: control-path diagnostic only
```

The intended smoke command is:

```text
STAMP=20260704_core_world_quad_smoke1 STEPS=240 PAYLOAD_MASS=4.0 \
TARGET_X=0.8 GAIT_FREQUENCY=1.1 HIP_AMPLITUDE_DEG=18.0 \
KNEE_AMPLITUDE_DEG=16.0 DEVICE=cpu RENDER=0 \
bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh
```

Run status:

```text
tmux: curiosity_core_world_0704
job: 165036
node: server10
```

Results are negative so far:

- `20260704_core_world_quad_smoke1`: failed at Kit startup because the default
  `isaacsim.exp.base.python.kit` experience could not resolve
  `isaacsim.anim.robot.schema`.
- `20260704_core_world_quad_smoke2_registryfix`: registry mirror was passed and
  resolved package indexes, but the default experience still lacked
  `isaacsim.anim.robot.schema`.
- `20260704_core_world_quad_smoke3_experiencefix` and
  `20260704_core_world_quad_smoke4_progress20`: local IsaacLab headless
  experience started and reached core scene construction, but no summary was
  produced. Progress logging localized the stop around custom
  `SingleArticulation` registration.
- `20260704_core_world_quad_smoke6_applauncher20`: AppLauncher started and
  reached `Creating SingleArticulation wrapper`, then exited without summary.

Additional missed USD/PhysX combination:

```text
summary: experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke5_cpu_artroot/usd_dynamic_quadruped_carry_summary.json
command:
STAMP=20260704_usd_dynamic_quad_payload_smoke5_cpu_artroot STEPS=300 \
PAYLOAD_MASS=4.0 TARGET_X=0.6 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 \
KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=1 CONTROL_MODE=usd_drive_attr \
DEVICE=cpu RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh
```

Result: completed 300/300, falls 0, drops 0, but torso and box travel both
remained 0.0. This rules out the simple hypothesis that CPU PhysX plus
articulation root makes direct USD drive-target actuation work for this custom
quadruped.

Next decision: stop repeating custom USD-drive or custom core
`SingleArticulation` smokes unchanged. The next dynamic-control route should
use an official local articulated asset such as ANYmal-C through a known-good
control interface, `isaacsim.core.experimental` articulation APIs, or a lower
level PhysX dynamic-control API that can show nonzero measured joint motion
before attempting gait tuning.

## Direct Adaptive Sweep

The active path was corrected away from waiting on external models. A direct
Isaac sweep runner was added:

```text
script: scripts/isaac/run_adaptive_probe_carry_sweep.sh
aggregator: scripts/isaac/aggregate_adaptive_probe_sweep.py
claim level: diagnostic scaffold only
```

Compute run:

```text
tmux: curiosity_anymal_exp2_0704
job: 165112
node: server10
command:
STAMP=20260704_adaptive_direct_sweep1 STEPS=180 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_adaptive_probe_carry_sweep.sh
```

Output:

```text
experiments/outputs/adaptive_probe_carry_scene_sweeps/adaptive_probe_sweep_20260704_adaptive_direct_sweep1/adaptive_probe_sweep_summary.json
```

Aggregate result:

```json
{
  "case_count": 5,
  "completed_case_count": 5,
  "failed_summary_count": 0,
  "drop_case_count": 0,
  "target_reached_case_count_threshold_0p08m": 5,
  "min_support_margin_m_over_cases": 0.07688841185014239,
  "strategy_counts": {
    "front_carry": 1,
    "low_front_carry": 1,
    "chest_supported_slow": 3
  }
}
```

Interpretation: direct Isaac task execution, parameterization, active-probing
proxy, belief logging, and morphology/load-dependent posture-selection plumbing
are now running across multiple cases. This still is not dynamic robot
walking, balance, contact grasping, or learned carrying because the carrier is
kinematic and the carried box follows the selected pose after the probing
decision.

## ANYmal Experimental Articulation Probe

The official-asset control probe was tested but should not remain the active
blocker:

```text
script: scripts/isaac/run_anymal_experimental_articulation_smoke.py
launcher: scripts/isaac/run_anymal_experimental_articulation_smoke.sh
latest output:
experiments/outputs/anymal_experimental_articulation_smoke/20260704_anymal_exp_art_smoke9_stagectx/anymal_experimental_articulation_failure_summary.json
```

Result: the local ANYmal-C USD loaded and exposed 12 DOFs, but after warmup
`physics_valid=False` and `physics_initialized=False`; `get_dof_positions()`
failed with:

```text
Instance's physics tensor entity is not valid. Play the simulation/timeline to
re-initialize it
```

This is a negative joint-control diagnostic only. It produced no walking,
balance, joint-motion, or carrying evidence.

## Velocity/Force Dynamic Rigid-Body Probe

To avoid the broken Articulation/RigidObject tensor paths and the ineffective
USD joint-drive path, a dynamic rigid-body control probe was added:

```text
script: scripts/isaac/build_velocity_controlled_dynamic_carry_scene.py
launcher: scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh
scene: dynamic torso rigid body, dynamic fixed-joint payload, visual gait legs
claim level: control-path diagnostic only
```

Velocity-attribute GPU smoke:

```text
tmux: curiosity_velocity_carry_0704
job: 165133
node: server10
command:
STAMP=20260704_velocity_dynamic_carry_smoke1 STEPS=360 PAYLOAD_MASS=5.0 \
TARGET_X=1.2 TARGET_SPEED=0.34 TARGET_HEIGHT=0.58 DEVICE=cuda:0 RENDER=0 \
bash scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh
```

Result:

```json
{
  "completed_steps": 360,
  "fall_events": 0,
  "box_drop_events": 0,
  "max_torso_travel_xy_m": 0.0,
  "max_box_travel_xy_m": 0.0,
  "final_box_target_distance_xy_m": 0.899999988079071
}
```

PhysX also reported:

```text
PxRigidDynamic::setLinearVelocity(): it is illegal to call this method if
PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled
```

Velocity-attribute CPU smoke:

```text
tmux: curiosity_velocity_carry_cpu_0704
job: 165134
node: server44
command:
STAMP=20260704_velocity_dynamic_carry_cpu_smoke1 STEPS=240 \
PAYLOAD_MASS=5.0 TARGET_X=0.9 TARGET_SPEED=0.30 TARGET_HEIGHT=0.58 \
DEVICE=cpu RENDER=0 bash scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh
```

Result: completed 240/240 with falls 0 and drops 0, but torso and box travel
were still 0.0. Interpretation: runtime USD `RigidBodyAPI.velocity` writes are
not an effective control path in this scene. The script now includes
`CONTROL_MODE=physx_force`, which uses
`omni.physx.get_physx_simulation_interface().apply_force_at_pos`; this is the
next smoke to validate before returning to dynamic carrying claims.

PhysX-force dynamic rigid-body smokes:

```text
outputs:
experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_force_dynamic_carry_cpu_smoke1/velocity_controlled_dynamic_carry_summary.json
experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_force_dynamic_carry_gpu_smoke1/velocity_controlled_dynamic_carry_summary.json
experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_force_direct_step_cpu_smoke1/velocity_controlled_dynamic_carry_summary.json
experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_force_direct_step_gpualloc_cpu_smoke1/velocity_controlled_dynamic_carry_summary.json
```

Result: all completed diagnostics reported 0.0 torso travel and 0.0 box
travel. GPU force mode additionally emitted PhysX direct-GPU errors for
runtime `addForce()`/`addTorque()`. Interpretation: this is not a viable
dynamic carrying control path in its current form.

Bare cube force/fall isolation:

```text
scripts:
scripts/isaac/build_physx_force_cube_smoke.py
scripts/isaac/run_physx_force_cube_smoke.sh
outputs:
experiments/outputs/physx_force_cube_smoke/20260704_physx_force_cube_cpu_smoke1/physx_force_cube_summary.json
experiments/outputs/physx_force_cube_smoke/20260704_physx_force_cube_simstep_cpu_smoke2/physx_force_cube_summary.json
```

Result: both direct-step and normal `sim.step()` variants kept the cube at
`[0, 0, 0.75]` with 0 x travel and no observed gravity drop. This rules out
the current direct `CuboidCfg.func` route as dynamic evidence.

RigidObject cube isolation:

```text
scripts:
scripts/isaac/build_physx_force_rigidobject_cube_smoke.py
scripts/isaac/run_physx_force_rigidobject_cube_smoke.sh
outputs:
experiments/outputs/physx_force_rigidobject_cube_smoke/20260704_physx_force_rigidobject_cube_cpu_smoke1/physx_force_rigidobject_cube_summary.json
experiments/outputs/physx_force_rigidobject_cube_smoke/20260704_physx_force_rigidobject_cube_newstage_cpu_smoke3/physx_force_rigidobject_cube_summary.json
```

Result: USD-only readout stayed fixed. Root-state readout with a new stage
failed at step 0 with:

```text
Failed to get rigid body transforms from backend
```

Interpretation: the current IsaacLab RigidObject tensor path is unusable for
active dynamic carry work until the backend invalidation is fixed.

Isaac Sim core `DynamicCuboid` isolation:

```text
scripts:
scripts/isaac/build_core_world_dynamic_cube_smoke.py
scripts/isaac/run_core_world_dynamic_cube_smoke.sh
logs:
logs/core_world_dynamic_cube_smoke/core_world_dynamic_cube_smoke_20260704_core_world_dynamic_cube_velocity_cpu_smoke1.log
logs/core_world_dynamic_cube_smoke/core_world_dynamic_cube_smoke_20260704_core_world_dynamic_cube_velocity_localground_cpu_smoke2.log
```

Result: the first run stalled in `add_default_ground_plane()` while checking
Nucleus asset root. The local-ground rerun passed core World creation and
local ground creation, then stalled before `world.reset()`, likely while
adding the core dynamic object wrapper. This route remains a blocked
diagnostic, not a dynamic-object success.

Current execution conclusion: do not wait on downloaded models, but also do
not keep tuning failed Isaac control paths. The next useful Isaac step is to
port a known-good installed Isaac dynamic-body example exactly, or enter an
official Arena task path whose object/joint state changes are verified before
adding the unknown-box carrying logic.
