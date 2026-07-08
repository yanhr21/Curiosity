# TODO 02: Base Simulated Box Loco-Manipulation Environment

- [x] Pivot base environment construction from blocked official Arena rollout
  to direct Isaac scene construction.
- [x] Add direct Isaac minimal carry scene:
  `scripts/isaac/build_minimal_carry_scene.py`.
- [x] Add compute-node launcher:
  `scripts/isaac/run_minimal_carry_scene.sh`.
- [x] Validate box-only CPU PhysX gravity/collision smoke:
  `experiments/outputs/minimal_carry_scene/smoke_20260703_skip_robot_usd_update_120steps/minimal_carry_scene_state.csv`.
- [ ] Fix and validate the robot tensor state path. CPU and GPU G1 articulation
  smokes both fail before stepping with `Failed to get DOF positions from
  backend`, including after `InteractiveScene`,
  `SKIP_EXPLICIT_STATE_RESET=1`, and `DISABLE_USD_PHYSICS_UPDATES=1`
  diagnostics. This is now a G1/IsaacLab tensor-backend blocker, not a WBC
  policy result.
- [x] Add robot embodiment entry to the direct scene after the box-only scene
  remains stable.
- [x] Add direct-scene adapter for Arena official G1 WBC HomieV2 stand/walk
  policy. This is a diagnostic integration path, not a completed carry policy.
- [x] Download official local Arena G1 HomieV2 WBC assets:
  `stand.onnx`, `walk.onnx`, `g1_29dof_with_hand.urdf`, and the 50 referenced
  STL mesh files.
- [x] Verify local WBC asset loading on a compute node:
  `scripts/isaac/check_g1_wbc_local_assets.py` reported 43 robot DOFs and
  `G1DecoupledWholeBodyPolicy`.
- [x] Re-run CPU `WBC_MODE=stand` after the Fabric/tensor logging patch. It
  still fails in the CPU-only PhysX tensor backend, so CPU robot smoke is not a
  viable path on this cluster.
- [x] Add `SKIP_EXPLICIT_STATE_RESET=1` diagnostic switch to isolate failures
  in explicit Articulation/RigidObject reset writes. This is not a success
  path.
- [x] Add direct-scene diagnostic metrics: fall flag, box drop flag, robot
  travel distance, box travel distance, box-to-target distance, and summary
  JSON.
- [x] Force `LC_ALL=C.UTF-8` and `LANG=C.UTF-8` in the launcher to avoid uv
  CPython startup failures on `encodings`.
- [x] Add `scripts/isaac/run_g1_wbc_smoke_sequence.sh` to run stand, walk, and
  fixed payload diagnostics sequentially in one compute allocation.
- [x] Add `RUN_PAYLOAD=0` option to the smoke sequence for stand/walk-only
  debugging.
- [x] Expose `BOX_POS_X/Y/Z` and `ATTACH_LOCAL_POS0_X/Y/Z` for fast payload
  placement tuning in GPU diagnostics.
- [x] Add `scripts/isaac/check_carry_smoke_summary.py` for lightweight
  post-run checks on completed steps, falls, dropped boxes, and travel distance.
- [x] Run `WBC_MODE=stand` direct-scene smoke in compute allocation. It failed
  before stepping with the same PhysX DOF tensor backend error; no standing
  evidence was produced.
- [x] Add `scripts/isaac/build_proxy_carry_scene.py`, a non-G1 Isaac scaffold
  for carrier + payload + box/target metrics. It is diagnostic-only and must
  not be reported as humanoid walking, balancing, grasping, or carrying
  success.
- [x] Run proxy pose-follow payload smoke on GPU:
  `experiments/outputs/proxy_carry_scene/proxy_pose_follow_payload_20260704_gpu164814_240steps/proxy_carry_scene_summary.json`.
  Result: 240/240 steps, box travel 0.418 m, drop events 0. This validates the
  scene/output skeleton only.
- [x] Add direct Isaac carrying-task scene:
  `scripts/isaac/build_direct_carry_task_scene.py` and
  `scripts/isaac/run_direct_carry_task_scene.sh`. It uses a kinematic humanoid
  proxy with approach/probe/lift/carry phases, a massed box, target marker,
  CSV state log, and summary JSON. It is diagnostic-only and is not a robot
  balance/grasp/carry success claim.
- [x] Run direct carrying-task smoke on GPU:
  `experiments/outputs/direct_carry_task_scene/20260704_direct_carry_task_scene_smoke4/direct_carry_task_scene_summary.json`.
  Command:
  `STAMP=20260704_direct_carry_task_scene_smoke4 STEPS=240 BOX_MASS=6.0 WALK_SPEED=0.32 TARGET_X=2.2 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_direct_carry_task_scene.sh`.
  Result: 240/240 steps, final phase `carry`, final box-to-target distance
  0.0348 m, drop events 0, support-margin proxy minimum 0.111 m.
- [x] Add adaptive active-probing Isaac scaffold:
  `scripts/isaac/build_adaptive_probe_carry_scene.py` and
  `scripts/isaac/run_adaptive_probe_carry_scene.sh`. It directly builds the
  unknown-box task in Isaac without waiting for external models: approach,
  micro-lift/nudge probing proxy, morphology/load posture selection, lift,
  carry, target, CSV metrics, and summary JSON. It is diagnostic-only:
  kinematic carrier and pose-follow box, not dynamic robot balance, learned
  grasping, true contact carrying, or video-conditioned RL.
- [x] Run adaptive Isaac default-load smoke:
  `experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke2_clean/adaptive_probe_carry_scene_summary.json`.
  Command:
  `STAMP=20260704_adaptive_probe_carry_scene_smoke2_clean STEPS=300 BOX_MASS=8.0 BOX_SIZE_X=0.58 BOX_SIZE_Y=0.38 BOX_SIZE_Z=0.36 BOX_COM_X=0.04 ROBOT_HEIGHT=1.45 ROBOT_MASS=52.0 ARM_LENGTH=0.58 MAX_PAYLOAD=16.0 TARGET_X=2.15 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_adaptive_probe_carry_scene.sh`.
  Result: selected `low_front_carry`, completed 300/300 steps, drop events 0,
  final box-to-target distance 0.0298 m, min support-margin proxy 0.0661 m.
- [x] Run adaptive Isaac morphology-change smoke:
  `experiments/outputs/adaptive_probe_carry_scene/20260704_adaptive_probe_carry_scene_smoke3_chest/adaptive_probe_carry_scene_summary.json`.
  Command:
  `STAMP=20260704_adaptive_probe_carry_scene_smoke3_chest STEPS=260 BOX_MASS=11.0 BOX_SIZE_X=0.68 BOX_SIZE_Y=0.42 BOX_SIZE_Z=0.40 BOX_COM_X=0.02 ROBOT_HEIGHT=1.25 ROBOT_MASS=44.0 ARM_LENGTH=0.48 MAX_PAYLOAD=15.0 TARGET_X=1.85 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_adaptive_probe_carry_scene.sh`.
  Result: selected `chest_supported_slow`, completed 260/260 steps, drop
  events 0, final box-to-target distance 0.0097 m, min support-margin proxy
  0.0854 m. This validates posture-selection plumbing, not dynamic success.
- [x] Add `scripts/isaac/run_anymal_payload_carry.py` and launcher for official
  ANYmal locomotion-policy payload diagnostics. This is not grasp/contact
  carry success; payload is currently added base mass plus visual box unless
  extended.
- [x] Re-run manager-based ANYmal clean PhysX no-payload smoke:
  `logs/anymal_payload_carry/anymal_payload_carry_20260704_anymal_clean_physx_nopayload_smoke14.log`.
  Result: scene creation reached one environment, then failed before rollout
  with `Failed to get DOF velocities from backend`. This matches the G1
  articulation tensor failure and is not a payload-policy result.
- [x] Try direct ANYmal task:
  `logs/anymal_payload_carry/anymal_payload_carry_20260704_anymal_direct_physx_nopayload_smoke15.log`.
  Result: entered direct task `gym.make` but did not reach rollout before
  interruption; no walking evidence.
- [x] Add low-level contact scene:
  `scripts/isaac/build_contact_carry_scene.py` and
  `scripts/isaac/run_contact_carry_scene.sh`. It uses kinematic palms and a
  dynamic box with no box pose following.
- [x] Run low-level contact smoke1:
  `experiments/outputs/contact_carry_scene/20260704_contact_carry_smoke1/contact_carry_scene_summary.json`.
  Result: 420/420 steps, but box travel and lift were both 0.0. Negative
  result: editing USD transforms did not act as an effective PhysX kinematic
  target for contact carrying.
- [x] Add RigidObject-driven contact scene:
  `scripts/isaac/build_contact_carry_rigid_scene.py` and
  `scripts/isaac/run_contact_carry_rigid_scene.sh`. It writes kinematic palm
  poses through the rigid-object simulation API instead of direct USD xform
  edits.
- [x] Run RigidObject contact-carry smoke in a clean compute allocation:
  `logs/contact_carry_rigid_scene/contact_carry_rigid_scene_20260704_contact_carry_rigid_smoke1.log`.
  Result: failed at first palm pose write with
  `Failed to set rigid body transforms in backend`.
- [x] Re-run RigidObject contact-carry on compute with `DEVICE=cpu`:
  `logs/contact_carry_rigid_scene/contact_carry_rigid_scene_20260704_contact_carry_rigid_cpu_smoke2.log`.
  Result: same `Failed to set rigid body transforms in backend`.
- [ ] Try a pure Omni/PhysX non-tensor kinematic-target API for contact carry,
  or fix the current IsaacLab/PhysX tensor invalidation before returning to
  official articulated robot policies.
- [x] Add non-tensor USD/PhysX dynamic quadruped fixed-payload scene:
  `scripts/isaac/build_usd_dynamic_quadruped_carry_scene.py` and
  `scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`. This constructs a
  dynamic robot from USD rigid bodies, revolute joints, fixed foot joints,
  drive targets, and a physical box fixed to the torso. It avoids IsaacLab
  articulation/rigid-object tensor APIs.
- [x] Run USD dynamic quadruped smoke1 on compute:
  `experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke1/usd_dynamic_quadruped_carry_summary.json`.
  Command:
  `STAMP=20260704_usd_dynamic_quad_payload_smoke1 STEPS=900 PAYLOAD_MASS=4.0 TARGET_X=1.0 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 KNEE_AMPLITUDE_DEG=16.0 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
  Result: completed 900/900, falls 0, drops 0, but torso and box travel were
  both 0.0. Negative cause: with articulation root on GPU, PhysX reported
  repeated `PxArticulationJointReducedCoordinate::setDriveTarget()` direct-GPU
  errors.
- [x] Run USD dynamic quadruped smoke2 with articulation root disabled:
  `experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke2_noartroot/usd_dynamic_quadruped_carry_summary.json`.
  Command:
  `STAMP=20260704_usd_dynamic_quad_payload_smoke2_noartroot STEPS=600 PAYLOAD_MASS=4.0 TARGET_X=0.8 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=0 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
  Result: completed 600/600, falls 0, drops 0, but travel remained 0.0.
- [x] Run USD dynamic quadruped smoke3 on CPU PhysX inside the compute
  allocation:
  `experiments/outputs/usd_dynamic_quadruped_carry_scene/20260704_usd_dynamic_quad_payload_smoke3_cpu/usd_dynamic_quadruped_carry_summary.json`.
  Command:
  `STAMP=20260704_usd_dynamic_quad_payload_smoke3_cpu STEPS=400 PAYLOAD_MASS=4.0 TARGET_X=0.6 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=0 DEVICE=cpu RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
  Result: completed 400/400, falls 0, drops 0, but travel remained 0.0.
- [x] Add and test `CONTROL_MODE=core_articulation` using Isaac Sim
  `SingleArticulation.apply_action`. Smoke4 failed before rollout:
  `logs/usd_dynamic_quadruped_carry_scene/usd_dynamic_quadruped_carry_scene_20260704_usd_dynamic_quad_payload_smoke4_core_cpu.log`.
  Command:
  `STAMP=20260704_usd_dynamic_quad_payload_smoke4_core_cpu STEPS=120 PAYLOAD_MASS=4.0 TARGET_X=0.3 GAIT_FREQUENCY=1.0 HIP_AMPLITUDE_DEG=14.0 KNEE_AMPLITUDE_DEG=12.0 ARTICULATION_ROOT=1 CONTROL_MODE=core_articulation DEVICE=cpu RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
  Result: `SingleArticulation` failed under the current IsaacLab
  `PhysxManager` context with `AttributeError: type object 'PhysxManager' has
  no attribute '_get_backend_utils'`.
- [ ] Fix the dynamic Isaac control mechanism before further gait tuning.
  Do not just change hip/knee amplitudes while torso/box travel is exactly
  zero. Next valid options: repair `SingleArticulation` compatibility, use the
  non-deprecated `isaacsim.core.experimental.prims.Articulation` API, or locate
  a functioning dynamic-control interface for articulation targets.
- [x] Run standalone Isaac Sim core-World articulation diagnostic:
  `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py` via
  `scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`. Result is
  negative so far. `20260704_core_world_quad_smoke1` failed because direct
  `SimulationApp` used the default base experience and could not resolve
  `isaacsim.anim.robot.schema`. `smoke2` fixed the registry mirror but the
  default experience still lacked that extension. `smoke3/smoke4` used the
  local IsaacLab headless experience and reached core scene creation, but
  produced no summary. Progress logging showed the custom USD articulation
  path stopped around `SingleArticulation` registration. `smoke6_applauncher20`
  used IsaacLab `AppLauncher`, reached `Creating SingleArticulation wrapper`,
  then exited without summary. Passing criterion remains: expected DOFs plus
  nonzero measured joint motion.
- [x] Run missed USD/PhysX combination: CPU PhysX plus articulation root plus
  direct USD drive-target attributes:
  `20260704_usd_dynamic_quad_payload_smoke5_cpu_artroot`.
  Command:
  `STAMP=20260704_usd_dynamic_quad_payload_smoke5_cpu_artroot STEPS=300 PAYLOAD_MASS=4.0 TARGET_X=0.6 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 KNEE_AMPLITUDE_DEG=16.0 ARTICULATION_ROOT=1 CONTROL_MODE=usd_drive_attr DEVICE=cpu RENDER=0 bash scripts/isaac/run_usd_dynamic_quadruped_carry_scene.sh`.
  Result: completed 300/300, falls 0, drops 0, but torso and box travel both
  remained 0.0. This rules out the simplest "CPU articulation root fixes USD
  drive targets" hypothesis.
- [ ] Next dynamic-control route must avoid both custom USD drive-target-only
  actuation and core `SingleArticulation` wrapping of the custom articulation.
  Valid next options: load an official local articulated asset such as ANYmal-C
  through a known-good control interface, use `isaacsim.core.experimental`
  articulation APIs, or locate a lower-level PhysX dynamic-control API with
  verifiable joint-state changes.
- [x] Run official-asset experimental articulation smoke:
  `scripts/isaac/run_anymal_experimental_articulation_smoke.py` via
  `scripts/isaac/run_anymal_experimental_articulation_smoke.sh`. This uses the
  local ANYmal-C USD plus `isaacsim.core.experimental.prims.Articulation`.
  Result is negative: smoke9 exposed 12 DOFs from the official local ANYmal-C
  USD, but the physics tensor entity stayed invalid after warmup and
  `get_dof_positions()` failed with `Instance's physics tensor entity is not
  valid`. This is not walking or carrying evidence.
- [x] Add parameter-sweep runner for the adaptive Isaac scaffold across mass,
  size, COM offset, robot height, arm length, and max payload, with posture
  diversity checks:
  `scripts/isaac/run_adaptive_probe_carry_sweep.sh` and
  `scripts/isaac/aggregate_adaptive_probe_sweep.py`.
- [x] Run adaptive direct Isaac sweep on compute:
  `experiments/outputs/adaptive_probe_carry_scene_sweeps/adaptive_probe_sweep_20260704_adaptive_direct_sweep1/adaptive_probe_sweep_summary.json`.
  Command:
  `STAMP=20260704_adaptive_direct_sweep1 STEPS=180 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_adaptive_probe_carry_sweep.sh`.
  Result: 5/5 cases completed, failed summaries 0, drop cases 0,
  target-threshold hits 5/5 at 0.08 m, min support-margin proxy 0.0769 m,
  strategies `front_carry: 1`, `low_front_carry: 1`,
  `chest_supported_slow: 3`. This validates parameterized direct Isaac
  scaffold execution only; it is still kinematic and pose-following.
- [ ] Replace the adaptive scaffold's pose-follow carry with real dynamic
  contact or a controller-backed articulated carrier before making any robot
  carrying success claim.
- [x] Add official Isaac robot policy standalone smoke:
  `scripts/isaac/run_official_policy_locomotion_smoke.py` and
  `scripts/isaac/run_official_policy_locomotion_smoke.sh`. It ports the
  installed NVIDIA Go2/H1 flat-terrain policy examples into a compute-node
  diagnostic using local mirrored USD/policy/config assets. This is not a box
  carrying claim.
- [x] Download the local official Go2/H1 assets needed for that smoke under
  `/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/`, including
  Go2 USD payload files and Go2/H1 PhysX policy checkpoints/configs.
- [x] Add `PAYLOAD_MODE=fixed_base` to the official-policy smoke. For Go2 it
  authors a physical rigid carry box and a fixed joint to
  `/World/Go2/Geometry/base`, then logs payload pose, payload travel, payload
  drops, robot travel, and falls. This is fixed-payload balance-under-load
  evidence only if a compute-node run verifies nonzero locomotion and no
  safety failure.
- [x] Fix two standalone Isaac integration blockers in the official-policy
  smoke: expose the extension namespace paths for
  `isaacsim.robot.policy.examples`, and avoid blocking `new_stage()`/
  `next_update_async()` calls by reusing the AppLauncher stage and stepping
  through synchronous `simulation_app.update()`.
- [x] Run no-GPU-allocation CPU official Go2 diagnostics in the held
  server05 allocation. Results: `diag13` failed before construction because
  the current IsaacLab `PhysxManager` lacked
  `get_active_physics_engine`; a compatibility shim fixed that. `diag14`
  reached policy construction but failed with `Invalid device identifier:
  cuda:0`; forcing `PhysicsManager._device = "cpu"` fixed construction.
  `diag15` created the Go2 policy object but reported
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized` and hung at `robot.initialize()`. The script now
  checks `is_physics_tensor_entity_valid()` before initialization and writes a
  failure summary instead of hanging.
- [ ] Run and verify real-GPU official Go2 diagnostic
  `20260704_official_go2_policy_real_gpu_diag16`. Current Slurm state when
  submitted: job `165252`, partition `gpu`, request `--gres=gpu:1`, reason
  `(Priority)`. Passing criterion: completed steps, Go2 travel greater than
  0.5 m under command `[1, 0, 0]`, fall events 0, and no policy/asset error.
- [x] Run real-GPU official Go2 AppLauncher diagnostic
  `20260704_official_go2_policy_real_gpu_diag16` on server10. Result:
  official Go2 policy object was created on H200/CUDA, but the articulation
  physics tensor entity was invalid before initialization:
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized`. Enabling explicit `SimulationManager.set_backend`
  in `diag17` still exited before rollout. No walking evidence.
- [x] Add pure Isaac Sim `SimulationApp` official Go2 diagnostic:
  `scripts/isaac/run_official_policy_locomotion_simapp_smoke.py` and
  `scripts/isaac/run_official_policy_locomotion_simapp_smoke.sh`. The default
  base experience failed because the local registry mirror lacks
  `isaacsim.anim.robot.schema`; using the local IsaacLab headless experience
  started successfully.
- [x] Run pure SimulationApp Go2 diagnostics `simapp_diag3` through
  `simapp_diag9` on real GPU. Result: local Go2 assets and policy wrapper load,
  but the articulation physics tensor entity remains invalid before
  initialization. Explicit warmup/view creation does not fix it; explicit
  `SimulationManager.set_backend` still exits before rollout. No walking
  evidence and no carrying evidence.
- [x] Run pure SimulationApp Go2 diagnostic
  `20260704_go2_simapp_fresh_stage_diag10` with a freshly created USD stage,
  local IsaacLab headless experience, H200 GPU, explicit IsaacLab
  `PhysicsManager._device` sync, and a best-effort
  `SimulationManager.set_physics_sim_device` call without `set_backend`.
  Result: still negative. The Go2 policy object was created, but
  `is_physics_tensor_entity_valid()` was false before initialization:
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized`. Fresh stage and device sync did not fix the
  official-policy tensor-articulation path.
- [ ] If the no-payload Go2 smoke passes, immediately run:
  `PAYLOAD_MODE=fixed_base PAYLOAD_MASS=2.0 ROBOT=go2 COMMAND_X=0.6` through
  the same launcher. Passing criterion: nonzero robot and payload travel,
  fall events 0, payload drop events 0. Do not label it unknown-box carrying.
- [x] Add velocity/force-controlled dynamic rigid-body Isaac carry probe:
  `scripts/isaac/build_velocity_controlled_dynamic_carry_scene.py` and
  `scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh`. This is a
  diagnostic only: dynamic torso rigid body, fixed-joint dynamic payload, and
  visual gait legs. It is not legged articulation control or learned carrying.
- [x] Run `CONTROL_MODE=velocity_attr` dynamic rigid-body probe on GPU:
  `experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_velocity_dynamic_carry_smoke1/velocity_controlled_dynamic_carry_summary.json`.
  Command:
  `STAMP=20260704_velocity_dynamic_carry_smoke1 STEPS=360 PAYLOAD_MASS=5.0 TARGET_X=1.2 TARGET_SPEED=0.34 TARGET_HEIGHT=0.58 DEVICE=cuda:0 RENDER=0 bash scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh`.
  Result: completed 360/360, falls 0, drops 0, but torso and box travel were
  0.0. PhysX direct-GPU rejected runtime linear velocity writes with
  `PxRigidDynamic::setLinearVelocity()` direct-GPU API errors.
- [x] Run `CONTROL_MODE=velocity_attr` dynamic rigid-body probe on CPU:
  `experiments/outputs/velocity_controlled_dynamic_carry_scene/20260704_velocity_dynamic_carry_cpu_smoke1/velocity_controlled_dynamic_carry_summary.json`.
  Command:
  `STAMP=20260704_velocity_dynamic_carry_cpu_smoke1 STEPS=240 PAYLOAD_MASS=5.0 TARGET_X=0.9 TARGET_SPEED=0.30 TARGET_HEIGHT=0.58 DEVICE=cpu RENDER=0 bash scripts/isaac/run_velocity_controlled_dynamic_carry_scene.sh`.
  Result: completed 240/240, falls 0, drops 0, but torso and box travel were
  still 0.0. Runtime USD `RigidBodyAPI.velocity` writes are not an effective
  control path in this scene.
- [x] Run `CONTROL_MODE=physx_force` dynamic rigid-body probe through
  `omni.physx.get_physx_simulation_interface().apply_force_at_pos` on compute.
  CPU, GPU, and direct-step CPU smokes all completed without torso/box travel:
  `20260704_force_dynamic_carry_cpu_smoke1`,
  `20260704_force_dynamic_carry_gpu_smoke1`,
  `20260704_force_direct_step_cpu_smoke1`, and
  `20260704_force_direct_step_gpualloc_cpu_smoke1`. GPU also rejected runtime
  `addForce()`/`addTorque()` under direct GPU API. Do not tune gains on this
  route while travel remains exactly zero.
- [x] Add and run a bare non-tensor `CuboidCfg.func` force/fall isolation:
  `scripts/isaac/build_physx_force_cube_smoke.py` and
  `scripts/isaac/run_physx_force_cube_smoke.sh`. Results
  `20260704_physx_force_cube_cpu_smoke1` and
  `20260704_physx_force_cube_simstep_cpu_smoke2` both kept the cube at
  `[0, 0, 0.75]` with 0 x travel and no gravity drop. Treat direct
  `CuboidCfg.func` rigid-body creation as non-evidence until a separate
  observable dynamics path is found.
- [x] Add and run `RigidObjectCfg` force/fall isolation:
  `scripts/isaac/build_physx_force_rigidobject_cube_smoke.py` and
  `scripts/isaac/run_physx_force_rigidobject_cube_smoke.sh`. USD-only pose
  smoke stayed at 0 travel; root-state smoke with new stage failed at step 0
  with `Failed to get rigid body transforms from backend`. This confirms the
  current IsaacLab RigidObject tensor path is not usable as the active dynamic
  carry route.
- [x] Add Isaac Sim core `DynamicCuboid` smoke:
  `scripts/isaac/build_core_world_dynamic_cube_smoke.py` and
  `scripts/isaac/run_core_world_simapp_dynamic_cube_smoke.sh`. The working
  route is pure `SimulationApp` plus Isaac Sim core `World`, local ground, and
  CPU PhysX. Run
  `20260704_core_world_simapp_cube_velocity_cpu_diag3` completed 180/180 steps
  with cube x travel 0.315 m and no script error. This is the first verified
  non-tensor dynamic-body motion path in the current environment.
- [x] Add pure core-World fixed-payload dynamic carrier:
  `scripts/isaac/build_core_world_simapp_fixed_payload_carry.py` and
  `scripts/isaac/run_core_world_simapp_fixed_payload_carry.sh`. Run
  `20260704_core_world_simapp_fixed_payload_centerweld_diag2` completed
  240/240 steps with carrier and payload both traveling 0.3596 m, relative
  payload error 0.0 m, fall events 0, and payload drop events 0. This is real
  Isaac dynamic rigid-body fixed-load motion, not legged walking, unknown-box
  grasping, or learned carrying.
- [x] Record negative fixed-payload offset result:
  `20260704_core_world_simapp_fixed_payload_carry_diag1` moved the carrier but
  a front-offset fixed joint snapped/oscillated, ending with payload relative
  error about 0.495 m. Use center-weld only as the current stable physical-load
  diagnostic until the offset joint is repaired.
- [x] Re-run standalone core-World custom articulated quadruped after adding
  the `SimulationManager` compatibility shim:
  `20260704_core_world_quad_payload_shim_diag7`. Result: `SingleArticulation`
  initialized and exposed 8 DOFs, and joint positions responded to commands,
  but torso and payload travel stayed 0.0 m. Around step 200, PhysX produced
  non-finite broadphase bounds and joint states became NaN. This proves partial
  articulation-control plumbing only; it is not a walking or carrying path.
- [ ] Next dynamic Isaac attempt should build on the verified pure
  `SimulationApp` core-World dynamic-body path, not the broken tensor-policy
  routes. Immediate valid work: add morphology/probing/task metrics around the
  stable dynamic fixed-payload carrier, then replace the velocity-commanded
  carrier with a controller-backed articulated base only after the articulation
  instability is fixed.
- [x] Run the new velocity-assisted articulated fixed-payload diagnostic:
  `BASE_VELOCITY_ASSIST=1 TARGET_SPEED=0.24 STEPS=180 PAYLOAD_MASS=4.0
  DEVICE=cpu bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Run `20260704_core_world_quad_assisted_diag1` initialized 8 DOFs, produced
  max joint motion 0.382 rad, fall events 0, drop events 0, nonfinite joint
  events 0, and control errors 0. However torso and payload travel stayed
  0.0 m, so `SingleArticulation.set_linear_velocity()` did not provide a
  usable carrier-translation path for the custom articulation. This is a
  negative result, not walking or carrying evidence.
- [x] Fix articulated fixed-payload pose metrics to use Isaac runtime prim
  state instead of USD local-to-world transforms. Added a `SingleRigidPrim`
  wrapper for the carried box and changed torso/box metrics to
  `get_world_pose()` from `SingleArticulation`/`SingleRigidPrim`.
- [x] Run runtime-pose velocity-assisted articulated fixed-payload diagnostic:
  `20260704_core_world_dynamic_quadruped_fixed_payload_diag2_runtime_pose`,
  Slurm job `165357` on `server46`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_dynamic_quadruped_fixed_payload_diag2_runtime_pose DEVICE=cpu STEPS=260 PAYLOAD_MASS=4.0 TARGET_X=0.8 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 GAIT_FREQUENCY=1.0 HIP_AMPLITUDE_DEG=12.0 KNEE_AMPLITUDE_DEG=10.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: completed 260/260, max joint motion 0.649 rad, torso travel
  0.2107 m, box travel 0.2063 m, box drops 0, but fall gate triggered
  19 times and max tilt reached 3.269 rad. This proves runtime pose/travel
  logging and root velocity effect, but it is unstable and not carrying
  success.
- [x] Run slower velocity-assisted articulated fixed-payload diagnostic:
  `20260704_core_world_dynamic_quadruped_fixed_payload_diag3_slow_stable`,
  Slurm job `165357` on `server46`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_dynamic_quadruped_fixed_payload_diag3_slow_stable DEVICE=cpu STEPS=260 PAYLOAD_MASS=4.0 TARGET_X=0.8 TARGET_SPEED=0.08 BASE_VELOCITY_ASSIST=1 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: completed 260/260, torso travel 0.0977 m, box travel 0.2133 m,
  box drops 0, but fall gate still triggered 21 times. Slowing and reducing
  joint amplitude did not create a stable physical gait for this simplified
  quadruped.
- [x] Add `BASE_ASSIST_MODE=pose` to the custom articulated fixed-payload
  route. This explicitly pose-commands the articulation root while keeping the
  USD articulation, joint targets, fixed payload, runtime pose logging, and
  fall/drop metrics. It is a scaffold mode, not physical balance evidence.
- [x] Record invalid syntax-sync attempt:
  `20260704_core_world_dynamic_quadruped_fixed_payload_diag4_pose_assist`
  failed before Isaac startup with a transient stale/partial file syntax error
  on the compute node. It produced no simulation evidence and is superseded by
  `diag5`.
- [x] Run pose-assisted articulated fixed-payload scaffold diagnostic:
  `20260704_core_world_dynamic_quadruped_fixed_payload_diag5_pose_assist_retry`,
  Slurm job `165357` on `server46`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_dynamic_quadruped_fixed_payload_diag5_pose_assist_retry DEVICE=cpu STEPS=260 PAYLOAD_MASS=4.0 TARGET_X=0.8 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: completed 260/260, `base_assist_mode=pose`, max joint motion
  0.3726 rad, torso travel 0.2340 m, box travel 0.2340 m, final box-target
  distance 0.4515 m, min torso z 0.6193 m, max tilt 0, fall events 0, box drop
  events 0, and control errors 0. This is the current best articulated
  fixed-payload task scaffold, but it is not unassisted locomotion.
- [x] Add articulated staged-free-box support to
  `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py`: new
  `PAYLOAD_MODE=staged_free_box`, `STAGED_ATTACH_MODE=pose-lock|fixed-joint`,
  `BOX_X`, `ATTACH_AFTER_STEP`, and `PROBE_SPEED` controls. The box begins as
  a free dynamic body, then the script runs probe, staged lift, attach, carry,
  and target-hold phases while logging attach step, probe displacement,
  relative error, target hold, fall events, and box-drop events.
- [x] Run articulated staged-free-box short pose-lock scaffold diagnostic:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag1_pose_lock`.
  This extends the `diag5` articulated scaffold from fixed payload to a box
  that begins as a free dynamic body, is probed, staged-lifted, attached, and
  carried with logged target/fall/drop/relative-error metrics. It still uses
  `BASE_ASSIST_MODE=pose` and `STAGED_ATTACH_MODE=pose-lock`, so it must not
  be reported as unassisted walking, physical grasping, or final carrying
  success.
  Slurm allocation: `165365` in `isaac-setup`, initially pending with
  `(Priority)` at creation time.
  Command:
  `timeout 420s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag1_pose_lock DEVICE=cpu STEPS=360 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=pose-lock BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: completed 360/360, attached at step 90, probe displacement
  0.20965 m, torso travel 0.2430 m, box travel 0.20965 m, final target
  distance 0.11705 m, relative error after attach about `3e-08`, fall/drop
  events 0, and no disjoint warning. It did not reach target hold because the
  run was too short.
- [x] Run articulated staged-free-box longer pose-lock target-hold diagnostic:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold`,
  Slurm job `165365` on `server63`.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=pose-lock BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Checker:
  `python3 scripts/isaac/check_dynamic_quadruped_carry_summary.py experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold/core_world_dynamic_quadruped_carry_summary.json --log logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260704_core_world_dynamic_quadruped_staged_free_box_diag2_pose_lock_target_hold.log --expect-payload-mode staged_free_box --expect-staged-attach-mode pose-lock --expect-base-assist-mode pose --require-attach --forbid-disjoint-warning --max-fall-events 0 --max-box-drop-events 0 --min-target-hold-steps 5 --max-target-distance 0.05 --max-relative-error 0.001 --max-peak-relative-error 0.001 --min-torso-travel 0.35 --min-box-travel 0.35 --max-tilt 0.01`.
  Result: pass as an articulated staged-free-box scaffold only. Completed
  540/540, attached at step 90, probe displacement 0.37165 m,
  `target_hold_steps=5`, final target distance 0.04495 m, torso travel
  0.4050 m, box travel 0.37165 m, fall/drop events 0, disjoint warning false,
  and final/peak relative errors about `3e-08`/`3.8e-08`. It still uses
  root pose assist and pose-lock attach, so it is not unassisted locomotion or
  physical grasping.
- [x] Run staged-free-box fixed-joint physical-constraint experiment:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag3_fixed_joint`,
  Slurm job `165365` on `server63`.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag3_fixed_joint DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=fixed-joint BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: fail. PhysX reported disjointed body transforms for
  `/World/Robot/StagedFreeBoxJoint`; box drop events 2, target hold 0, final
  target distance 1695.86 m, final relative error 2831.10 m, peak relative
  error 2892.10 m, and max joint motion exploded. This is negative evidence
  for the pre-created/standard fixed-joint staged attach path.
- [x] Change staged fixed-joint experiment to create the joint at attach time
  instead of pre-creating a disabled joint, then run:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag4_fixed_joint_runtime_create`,
  Slurm job `165365` on `server63`.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag4_fixed_joint_runtime_create DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=fixed-joint BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Checker result: fail. Runtime joint creation still produced a disjoint
  fixed-joint warning, box drop events 2, target hold 0, final target distance
  1695.86 m, final relative error 2831.10 m, and peak relative error
  2892.10 m. This shows the fixed-joint failure is not just joint creation
  timing; the next physical attach path should use a better grasp/contact
  formulation instead of tuning this fixed-joint route unchanged.
- [x] Add lightweight checker
  `scripts/isaac/check_dynamic_quadruped_carry_summary.py` for custom
  articulated carry summaries. It validates completed steps, payload mode,
  attach mode, base assist mode, attach presence, target distance, target hold,
  fall/drop, relative error, travel, tilt, and disjoint fixed-joint warnings.
- [x] Run articulated staged-free-box `velocity-servo` attach diagnostic:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag5_velocity_servo`.
  This keeps the box dynamic after attach and applies a capped velocity servo
  toward the carry pose. It is a transition scaffold between pose-lock and a
  real contact grasp; it is still not physical grasping.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag5_velocity_servo DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=velocity-servo BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Result: partial/strict fail. Completed 540/540, attached at step 90,
  `target_hold_steps=5`, fall/drop events 0, disjoint warning false, torso
  travel 0.4050 m, box travel 0.4800 m, but root pose assist did not stop at
  the target; final target distance was 0.1533 m and final relative error was
  0.10835 m.
- [x] Add staged-free-box target-stop clamp for pose-assisted root motion:
  after attach, `BASE_ASSIST_MODE=pose` clamps root x at `target_x - 0.26`
  so the carry pose stops at the target instead of walking past it.
- [x] Run velocity-servo with target-stop clamp:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag6_velocity_servo_target_stop`,
  Slurm job `165369` on `server10`.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag6_velocity_servo_target_stop DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=velocity-servo BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Checker result: strict fail. Completed 540/540, attached at step 90,
  `target_hold_steps=5`, fall/drop events 0, disjoint warning false, root
  stopped near 0.360 m, torso travel 0.36005 m, box travel 0.43619 m, but
  velocity-servo lagged behind the carry pose: final target distance
  0.10941 m and final relative error 0.10943 m. This is better than
  fixed-joint because it avoids explosions, but it is not yet a valid dynamic
  attach replacement for pose-lock. Next work should add a proper contact or
  multi-proxy grasp/hold model, not just increase the fixed-joint route.
- [x] Add staged-free-box `contact-proxy` attach mode to
  `scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py`. The mode
  creates dynamic left/right palm, chest, and forearm-shelf proxy bodies and
  drives those proxies around the carry pose after staged attach. The box is
  not directly pose-locked or velocity-servoed in this mode.
- [x] Run staged-free-box `contact-proxy` attach diagnostic:
  `20260704_core_world_dynamic_quadruped_staged_free_box_diag7_contact_proxy`.
  This creates dynamic palm/chest/shelf proxy bodies and, after staged attach,
  drives the proxies around the carry pose instead of directly pose-locking or
  velocity-servoing the box. It is a contact/hold scaffold and may fail if the
  proxy contacts are insufficient.
  Command:
  `timeout 520s env STAMP=20260704_core_world_dynamic_quadruped_staged_free_box_diag7_contact_proxy DEVICE=cpu STEPS=540 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=contact-proxy BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.18 BASE_VELOCITY_ASSIST=1 BASE_ASSIST_MODE=pose ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 GAIT_FREQUENCY=0.7 HIP_AMPLITUDE_DEG=2.0 KNEE_AMPLITUDE_DEG=2.0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Checker:
  `python3 scripts/isaac/check_dynamic_quadruped_carry_summary.py experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260704_core_world_dynamic_quadruped_staged_free_box_diag7_contact_proxy/core_world_dynamic_quadruped_carry_summary.json --log logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260704_core_world_dynamic_quadruped_staged_free_box_diag7_contact_proxy.log --expect-payload-mode staged_free_box --expect-staged-attach-mode contact-proxy --expect-base-assist-mode pose --require-attach --require-contact-proxy --forbid-disjoint-warning --max-fall-events 0 --max-box-drop-events 0 --min-target-hold-steps 5 --max-target-distance 0.05 --max-relative-error 0.05 --max-peak-relative-error 0.12 --min-torso-travel 0.35 --min-box-travel 0.25 --max-tilt 0.01 --max-contact-proxy-gap 0.20`.
  Result: fail. Completed 540/540, attached at step 90, contact proxy enabled,
  fall events 0, and no disjoint warning. However box drop events were 34,
  target hold 0, final target distance 0.28981 m, max box travel only
  0.19499 m, final/peak relative error 0.60057 m, and max contact-proxy gap
  1.37539 m. Interpretation: current dynamic proxy bodies do not establish a
  stable grip/hold; the next contact work needs pre-closed geometry, stronger
  shelf support, normal clamping, or a controlled grip-force/constraint
  hybrid, not just moving proxy bodies toward nominal targets.
- [x] Add a dynamic adaptive fixed-payload carry task on top of the verified
  pure `SimulationApp` core-World dynamic carrier. It should keep the success
  claim diagnostic-only, but add task target distance, mass/morphology
  strategy selection, speed adaptation, balance-margin proxy, effort proxy,
  fall/drop metrics, and per-step CSV evidence around the physical welded
  payload.
- [x] Run adaptive dynamic fixed-payload long-target diagnostic:
  `20260704_core_world_adaptive_payload_diag1`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_diag1 DEVICE=cpu STEPS=360 TARGET_X=1.05 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 ROBOT_HEIGHT=1.35 ROBOT_MASS=48.0 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.34 bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: selected `low_front_carry`, carrier and payload traveled 0.3197 m,
  relative payload error 0.0 m, fall events 0, drop events 0, minimum
  balance-margin proxy 0.0872 m, but final target distance was 0.7303 m. This
  validates dynamic payload transport metrics, not target completion.
- [x] Run adaptive dynamic fixed-payload short-target diagnostic:
  `20260704_core_world_adaptive_payload_target_diag2`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_target_diag2 DEVICE=cpu STEPS=360 TARGET_X=0.30 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 ROBOT_HEIGHT=1.35 ROBOT_MASS=48.0 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.34 bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: selected `low_front_carry`, completed 360/360 steps, carrier and
  payload traveled 0.3197 m, final payload-target distance 0.0197 m, payload
  relative error 0.0 m, fall events 0, drop events 0, minimum balance-margin
  proxy 0.0872 m. This is a dynamic fixed-payload task diagnostic, not legged
  walking or unknown free-object carrying.
- [x] Attempt adaptive dynamic fixed-payload chest-strategy diagnostic:
  `20260704_core_world_adaptive_payload_chest_diag3`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_chest_diag3 DEVICE=cpu STEPS=360 TARGET_X=0.20 PAYLOAD_MASS=12.0 PAYLOAD_COM_X=0.02 ROBOT_HEIGHT=1.25 ROBOT_MASS=44.0 ARM_LENGTH=0.45 MAX_PAYLOAD=15.0 BASE_SPEED=0.34 bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: interrupted after remaining at early `SimulationApp` startup
  protobuf warnings with no stage progress. No summary was produced; this is
  not a strategy-diversity result.
- [ ] Next direct Isaac step: either run the chest-strategy diagnostic in a
  fresh allocation, or add a small in-process sweep wrapper that runs one
  `SimulationApp` startup and multiple adaptive cases to avoid repeated Kit
  startup stalls.
- [x] Try the chest-strategy diagnostic again in a fresh allocation:
  `20260704_core_world_adaptive_payload_chest_diag4_fresh`, Slurm job
  `165264` on `server02`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_chest_diag4_fresh DEVICE=cpu STEPS=360 TARGET_X=0.20 PAYLOAD_MASS=12.0 PAYLOAD_COM_X=0.02 ROBOT_HEIGHT=1.25 ROBOT_MASS=44.0 ARM_LENGTH=0.45 MAX_PAYLOAD=15.0 BASE_SPEED=0.34 bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: interrupted after more than one minute at early `SimulationApp`
  startup with only protobuf registration warnings and no stage progress. No
  summary was produced. Treat this as a startup/runtime instability, not as a
  policy or strategy failure.
- [x] Implement a one-startup adaptive payload sweep runner or add a
  deterministic `CASE_PRESET` path to the current script so multiple strategy
  cases can be exercised without repeatedly starting Kit.
- [x] Run one-startup adaptive payload strategy sweep:
  `20260704_core_world_adaptive_payload_strategy_sweep1`, Slurm job `165266`
  on `server02`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_strategy_sweep1 DEVICE=cpu PRESET_SWEEP=strategy_smoke bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: 2/2 cases completed in one `SimulationApp` startup. Strategy counts:
  `low_front_carry: 1`, `chest_supported_slow: 1`. Low-front case moved
  carrier/payload 0.3197 m and ended 0.0197 m from target. Chest-supported
  case moved carrier/payload 0.2485 m and ended 0.0485 m from target. Both had
  fall events 0, drop events 0, and payload relative error 0.0 m. This
  validates dynamic fixed-payload strategy plumbing only; still not legged
  walking or unknown free-object carrying.
- [x] Upgrade the adaptive dynamic fixed-payload scene with walking-support
  proxy instrumentation: visible left/right support-foot markers, gait
  frequency, stance width, step length, support state, and
  `min_support_margin_x_proxy_m` in summaries/CSV. This keeps the Isaac task
  interface moving without waiting for external model downloads, while still
  labeling the result as diagnostic-only.
- [x] Run walking-support-proxy adaptive strategy sweep:
  `20260704_core_world_adaptive_payload_walkproxy_diag1`, Slurm job `165292`
  on `server02`.
  Command:
  `STAMP=20260704_core_world_adaptive_payload_walkproxy_diag1 DEVICE=cpu PRESET_SWEEP=strategy_smoke bash scripts/isaac/run_core_world_simapp_adaptive_payload_carry.sh`.
  Result: 2/2 cases completed. Strategy counts:
  `low_front_carry: 1`, `chest_supported_slow: 1`. Low-front case moved
  carrier/payload 0.3287 m, final payload-target distance 0.0287 m,
  `min_support_margin_x_proxy_m` 0.1185 m, fall events 0, drop events 0.
  Chest-supported case moved carrier/payload 0.2568 m, final payload-target
  distance 0.0568 m, `min_support_margin_x_proxy_m` 0.1109 m, fall events 0,
  drop events 0. This verifies the upgraded scene instrumentation and dynamic
  fixed-payload transport only; it is not real legged walking, contact
  grasping, unknown-object carrying, or learned balance.
- [x] Add official Go2 callback locomotion smoke:
  `scripts/isaac/run_official_go2_callback_locomotion_smoke.py` and
  `scripts/isaac/run_official_go2_callback_locomotion_smoke.sh`.
  This follows NVIDIA's installed Go2 tests more closely than the older manual
  loop: fresh stage, `SimulationManager.set_physics_sim_device`,
  `SimulationManager.set_physics_dt`, `timeline.play()`, and
  `SimulationManager.register_callback(..., IsaacEvents.POST_PHYSICS_STEP)`.
  It also supports `PAYLOAD_MODE=fixed_base` for a later fixed payload check.
- [x] Run official Go2 callback no-payload walking diagnostic:
  `20260704_go2_callback_nopayload_diag1`, Slurm job `165274`.
  Command:
  `STAMP=20260704_go2_callback_nopayload_diag1 DEVICE=cuda STEPS=220 WARMUP_STEPS=20 COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 PAYLOAD_MODE=none bash scripts/isaac/run_official_go2_callback_locomotion_smoke.sh`.
  Result: negative/incomplete. The script reached fresh stage and local ground
  setup, then exited before printing `SimulationManager device/dt set` and
  produced no summary. This suggests a hard exit around
  `SimulationManager.set_physics_sim_device(...)` or
  `SimulationManager.set_physics_dt(...)`, not a walking result.
- [ ] Re-run official Go2 callback smoke with finer SimulationManager setter
  tracing and `SIMULATION_MANAGER_MODE=skip_device_dt DEVICE=cuda:0` to isolate
  whether the setter path is the hard-exit cause.
- [x] Attempt same-allocation Go2 callback skip-device-dt diagnostic:
  `20260704_go2_callback_skipdevdt_diag2`.
  Command:
  `STAMP=20260704_go2_callback_skipdevdt_diag2 DEVICE=cuda:0 SIMULATION_MANAGER_MODE=skip_device_dt STEPS=120 WARMUP_STEPS=8 COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 PAYLOAD_MODE=none bash scripts/isaac/run_official_go2_callback_locomotion_smoke.sh`.
  Result: negative/incomplete. Reusing the same shell/allocation after
  `diag1`, the process emitted only early protobuf registration warnings and
  no `SimulationApp started` progress before returning. No summary was
  produced. Treat this as repeated Kit startup instability, not walking
  evidence.
- [x] Run fresh-allocation Go2 callback skip-device-dt diagnostic:
  `20260704_go2_callback_skipdevdt_diag2_fresh`, Slurm job `165278`.
  Result: completed 120/120 script steps and produced a summary, but callback
  init attempts and forward calls were both 0, Go2 travel remained 0.0 m, and
  later log lines still reported
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized`. Skipping device/dt avoids the earlier hard exit,
  but it does not create a usable physics callback/view.
- [x] Attempt same-allocation Go2 callback `dt_only` diagnostic:
  `20260704_go2_callback_dtonly_diag3`.
  Result: interrupted after remaining at early `SimulationApp` protobuf
  warnings with no `SimulationApp started` progress. No summary was produced.
  Treat as repeated Kit startup instability from reusing the same shell after
  a previous Kit run.
- [x] Run fresh-allocation Go2 callback `dt_only` diagnostic:
  `20260704_go2_callback_dtonly_diag3_fresh`.
  Result: negative/incomplete. The script reached
  `Setting SimulationManager physics dt=0.005`, then exited before printing
  `SimulationManager physics dt set` and produced no summary. This localizes a
  hard-exit path to `SimulationManager.set_physics_dt(...)` in the current
  headless experience.
- [x] Run fresh-allocation Go2 callback `device_only` diagnostic to check
  whether `SimulationManager.set_physics_sim_device(...)` is also a hard-exit
  path, or whether only `set_physics_dt(...)` is unsafe.
  Run `20260704_go2_callback_deviceonly_diag4_fresh`, Slurm job `165287`, on
  `server02`. Result: negative/incomplete. The script reached
  `Setting SimulationManager physics device=cuda:0`, then exited before
  printing `SimulationManager physics device set` and produced no summary.
  This localizes a second hard-exit path to
  `SimulationManager.set_physics_sim_device(...)` in the current headless
  experience.
- [ ] Do not retry official Go2 callback/official-policy routes unchanged.
  Current evidence: with SimulationManager setters the process hard-exits;
  without setters no physics callback/view is created and Go2 travel remains
  0.0 m. A valid next robot-locomotion route must use a different app/test
  harness, a known-good official test runner, or a non-SimulationManager
  control path.
- [x] Fix the official Go2 callback launcher's `OFFICIAL_TEST_KIT_ARGS=1`
  plumbing so NVIDIA policy-test Kit settings are passed through
  `SimulationApp` `extra_args`, not accidentally left in Kit's unknown-arg
  list. Also clear parsed script args before `SimulationApp` startup to avoid
  polluting Kit arguments.
- [x] Run corrected official-test-Kit-args Go2 diagnostic:
  `20260704_go2_callback_officialkit_extraargs_diag7`, Slurm job `165296` on
  `server02`.
  Command:
  `STAMP=20260704_go2_callback_officialkit_extraargs_diag7 DEVICE=cuda:0 OFFICIAL_TEST_KIT_ARGS=1 SIMULATION_MANAGER_MODE=official_device_dt STEPS=160 WARMUP_STEPS=12 COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 PAYLOAD_MODE=none bash scripts/isaac/run_official_go2_callback_locomotion_smoke.sh`.
  Result: negative/incomplete. Startup args now correctly include the official
  physx-test Kit settings, but the process still exited at
  `SimulationManager.set_physics_sim_device(cuda:0)` before printing
  `SimulationManager physics device set`. No summary was produced.
- [x] Run corrected official-test-Kit-args Go2 skip-device-dt diagnostic:
  `20260704_go2_callback_officialkit_extraargs_skip_diag8`, same Slurm job
  `165296`.
  Command:
  `STAMP=20260704_go2_callback_officialkit_extraargs_skip_diag8 DEVICE=cuda:0 OFFICIAL_TEST_KIT_ARGS=1 SIMULATION_MANAGER_MODE=skip_device_dt STEPS=120 WARMUP_STEPS=8 COMMAND_X=1.0 COMMAND_Y=0.0 COMMAND_YAW=0.0 PAYLOAD_MODE=none bash scripts/isaac/run_official_go2_callback_locomotion_smoke.sh`.
  Result: negative. Completed 120/120 script steps and wrote a summary, but
  `callback_forward_calls=0`, `callback_init_attempts=0`, and
  `travel_xy_m=0.0`; log still reported
  `Invalid physics simulation view. Articulation (['/World/Go2/Geometry/base'])
  will not be initialized`. Correct official Kit settings alone do not repair
  the handwritten `SimulationApp` Go2 callback route.
- [x] Probe official `kit/dev/repo.sh test --help` inside the same allocation
  only to identify the test-runner path. Result: interrupted because it tried
  to fetch `python@3.12.13-nv3-manylinux_2_35-x86_64.tar.gz` from NVIDIA
  packman. Do not use this runner on compute until all packman dependencies are
  prepared locally on the login node.
- [ ] Next Isaac robot-motion route should not be another unchanged Go2
  callback run. Immediate valid work: build a non-tensor, core-World
  quasi-static walking carrier where dynamic torso/payload motion is tied to
  moving support feet or constraints, then later replace that controller with a
  verified official policy if a working test harness is prepared.
- [x] Add non-tensor core-World quasi-static walker carry scene:
  `scripts/isaac/build_core_world_simapp_quasistatic_walker_carry.py` and
  `scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh`.
  The scene uses a dynamic walker body, dynamic physical payload fixed by USD
  joint, four visible/colliding support feet, leg struts, gait phases,
  support-state logging, support-margin gating, target hold, fall/drop metrics,
  and CSV/summary outputs. It remains diagnostic-only: torso translation is
  still commanded through rigid-body velocity control, not a verified
  articulated legged policy.
- [x] Run quasi-static walker long-target diagnostic:
  `20260704_core_world_quasistatic_walker_diag1`, Slurm job `165299` on
  `server02`.
  Result: completed 420/420, body/payload traveled 0.1818 m, fall/drop 0,
  payload relative error 0.0, but final target distance was 0.1982 m and
  `min_support_margin_m=-0.0248`. Negative control: the initial diagonal
  trot-like support pattern was not quasi-static enough.
- [x] Replace diagonal support with one-foot-swing, three-foot-stance creep
  gait in the quasi-static walker scene.
- [x] Run quasi-static walker creep diagnostic:
  `20260704_core_world_quasistatic_walker_creep_diag2`, same Slurm job
  `165299`.
  Result: support margin stayed positive and the payload passed near the
  0.18 m target, but without target hold it overshot; final target distance
  was 0.114 m.
- [x] Add target-hold and `min_payload_target_distance_xy_m` metric to the
  quasi-static walker scene.
- [x] Run quasi-static walker target-hold diagnostic:
  `20260704_core_world_quasistatic_walker_hold_diag3`, same Slurm job
  `165299`.
  Command:
  `STAMP=20260704_core_world_quasistatic_walker_hold_diag3 DEVICE=cpu STEPS=420 TARGET_X=0.18 PAYLOAD_MASS=8.0 PAYLOAD_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 bash scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh`.
  Result: completed 420/420, body/payload traveled 0.1659 m, final and
  minimum payload-target distance 0.0141 m, payload relative error 0.0,
  `min_support_margin_m=0.1325`, balance-gate slowdowns 0, fall events 0, and
  payload drop events 0. This is the current strongest direct Isaac
  carrying-task diagnostic, but it is not final success because it is not a
  verified articulated walking robot policy and does not handle an unknown
  free object through contact grasping.
- [x] Stop treating official model/policy recovery as a prerequisite for scene
  construction. Current direct path is to build the Isaac task scaffold first,
  then replace placeholders with controller-backed locomotion, contact
  grasping, and learned active probing.
- [x] Add staged free-box carry diagnostic:
  `scripts/isaac/build_core_world_simapp_staged_free_box_carry.py` and
  `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`. The box
  starts as a free dynamic rigid body; the carrier approaches, runs a probing
  phase, then creates a logged runtime fixed joint after an explicit staged
  lift/hold event before carrying to target. This is an Isaac task-structure
  diagnostic only, not contact grasping, not articulated locomotion, and not a
  learned policy.
- [ ] Run staged free-box carry smoke on a compute node. Passing diagnostic
  criterion: script starts in Isaac, writes summary JSON, records attach step,
  completes requested steps, reaches nonzero box travel, fall events 0, and
  box drop events 0. Even if it passes, report it only as staged free-box to
  fixed-joint carry scaffolding. Attempts
  `20260704_core_world_staged_free_box_diag1` on `server56` and
  `20260704_core_world_staged_free_box_diag2` on `server02` were interrupted
  because `SimulationApp` did not finish startup; no summary was produced.
  A 20-step health check of the previously verified quasi-static walker in the
  same allocation also stalled at the same `SimulationApp` initialization
  point, so this is recorded as an Isaac startup/allocation issue rather than
  evidence against the staged free-box scene.
- [x] Verify Isaac startup on a fresh allocation after earlier startup stalls:
  `20260704_core_world_quasistatic_health_server10`, Slurm job `165306` on
  `server10`.
  Command:
  `timeout 240s env STAMP=20260704_core_world_quasistatic_health_server10 DEVICE=cpu STEPS=20 TARGET_X=0.02 bash scripts/isaac/run_core_world_simapp_quasistatic_walker_carry.sh`.
  Result: completed 20/20 steps, wrote summary JSON, fall 0, drop 0. This
  confirms Isaac startup can work in a fresh allocation; repeated
  `SimulationApp` startup inside the same allocation remains unreliable.
- [x] Run staged free-box smoke:
  `20260704_core_world_staged_free_box_diag3_server10`, same Slurm job
  `165306` on `server10`.
  Command:
  `timeout 360s env STAMP=20260704_core_world_staged_free_box_diag3_server10 DEVICE=cpu STEPS=520 TARGET_X=0.42 BOX_X=0.26 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: completed 520/520 steps, attach step 260, 77 probe attempts, body
  travel 0.3492 m, box travel 0.0563 m, fall events 0, box drop events 0.
  Negative attach-quality result: PhysX warned the runtime fixed joint had
  disjoint body transforms and the box snapped backward at attach; final
  box-target distance was 0.1648 m and
  `box_relative_error_m_after_attach=0.1868`. This validates staged free-box
  task execution only, not stable carrying.
- [x] Revise staged free-box attach logic after `diag3`: split attach into
  `staged_lift_settle` and `staged_attach_constraint`, then create the runtime
  fixed joint from the measured body-box relative pose and record
  `attach_local_pos0_m` in the summary.
- [x] Run the revised staged free-box attach smoke in a fresh allocation as
  the first Isaac command:
  `20260704_core_world_staged_free_box_diag4_settle_attach`, Slurm job
  `165310` on `server10`.
  Command:
  `timeout 360s env STAMP=20260704_core_world_staged_free_box_diag4_settle_attach DEVICE=cpu STEPS=560 TARGET_X=0.42 BOX_X=0.26 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: completed 560/560 steps, attach prep step 260, attach step 261,
  final box-target distance 0.0111 m, body travel 0.3591 m, box travel
  0.1515 m, min support margin 0.1299 m, fall events 0, box drop events 0.
  Improvement over `diag3`: target completion and larger box travel. Remaining
  negative: PhysX still warned that the runtime fixed joint had disjoint body
  transforms, and `box_relative_error_m_after_attach=0.0824`. This is still a
  staged scaffold diagnostic, not stable contact attach or final carrying.
- [x] Improve staged/free-box attach quality diagnostics. The fixed-joint route
  was tested and remains negative; a kinematic pose-lock task scaffold was
  added as an explicitly labeled non-physical placeholder so scene construction
  can continue. Next physical options remain: create the
  constraint while simulation is paused or before the next physics step,
  pre-author a disabled/empty joint and only populate targets at attach time,
  or replace the fixed joint with a D6/constraint formulation that preserves
  the measured relative pose without snapping. Passing diagnostic criterion:
  no disjoint-transform warning and lower post-attach relative error.
- [x] Implement the pre-authored disabled joint option for staged free-box
  attach. The scene now creates `/World/StagedCarryRuntimeFixedJoint` with
  `physics:jointEnabled=false` before `world.reset()`, then writes the measured
  local pose and enables it at attach time. Syntax check passed; simulation
  validation is still pending.
- [x] Run fresh-allocation/compute smokes for the pre-authored disabled joint
  attach version and follow-up fixes.
  `20260704_core_world_staged_free_box_diag6_preauth_short`, Slurm job
  `165319` on `server57`, command:
  `timeout 300s env STAMP=20260704_core_world_staged_free_box_diag6_preauth_short DEVICE=cpu STEPS=260 TARGET_X=0.30 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: completed 260/260 and attach step 91, but checker failed:
  final target distance 0.1373 m,
  `box_relative_error_m_after_attach=0.1428`, and two disjoint-transform
  warnings. This showed pre-authoring alone did not fix attach quality.
  `diag7_nonpenetrating` moved the staged hold to a non-penetrating carry
  offset; it avoided the old overlap but exposed a phase-order bug and did not
  attach. `diag8_phase_anchor_fix` fixed phase order and did attach, but the
  fixed joint still produced disjoint warnings, a large snap, and one
  box-drop event. Conclusion: current fixed-joint route is a negative result,
  not stable carrying.
- [x] Add explicit `ATTACHMENT_MODE=kinematic-pose-lock` scaffold to the staged
  free-box scene. In this mode the box still starts as a free dynamic rigid
  body and goes through probe/lift/attach phases, but after attach it is locked
  by pose update, not by a physical grasp or joint. This is a task-interface
  scaffold only.
- [x] Run kinematic staged free-box scaffold diagnostic:
  `20260704_core_world_staged_free_box_diag9_kinematic_pose_lock`, Slurm job
  `165319` on `server57`.
  Command:
  `timeout 300s env STAMP=20260704_core_world_staged_free_box_diag9_kinematic_pose_lock DEVICE=cpu STEPS=360 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=kinematic-pose-lock bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker command:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag9_kinematic_pose_lock/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag9_kinematic_pose_lock.log --require-attach --forbid-disjoint-warning --max-target-distance 0.03 --max-relative-error 0.05`.
  Result: pass as scaffold only: completed 360/360, attach step 91, final
  box-target distance 0.01455 m, `box_relative_error_m_after_attach=0.000245`,
  fall events 0, box drop events 0, no disjoint warning, body travel
  0.20545 m, box travel 0.47545 m, strategy `low_front_creep`. This must not
  be reported as physical grasping, dynamic humanoid locomotion, or final
  carrying success.
- [x] Run dynamic-rigidbody carrier variant of the staged free-box scaffold:
  `20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock`,
  Slurm job `165325` on `server10`.
  Command:
  `timeout 360s env STAMP=20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock DEVICE=cpu STEPS=360 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=kinematic-pose-lock CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker command:
  `python3 scripts/isaac/check_staged_free_box_summary.py <summary> --log <log> --require-attach --forbid-disjoint-warning --max-target-distance 0.03 --max-relative-error 0.05 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10`.
  Actual checker path:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag10_dynamic_velocity_pose_lock.log --require-attach --forbid-disjoint-warning --max-target-distance 0.03 --max-relative-error 0.05 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10`.
  Result: pass as a higher-fidelity scaffold only: completed 360/360, dynamic
  rigid-body velocity-commanded carrier mode, attach step 91, body travel
  0.20545 m, box travel 0.47545 m, final box-target distance 0.01455 m,
  post-attach relative error `2.78e-08`, fall events 0, box drop events 0,
  no disjoint warning, and `min_support_margin_m=0.13252`. This is a real
  Isaac dynamic rigid-body carrier scaffold with walking-support proxy, but it
  still is not physical grasping or verified articulated locomotion.
- [x] Run heavier/short-arm chest-supported dynamic-rigidbody staged free-box
  scaffold:
  `20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock`, Slurm
  job `165329` on `server46`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock DEVICE=cpu STEPS=560 TARGET_X=0.70 BOX_X=0.16 BOX_MASS=12.0 BOX_SIZE_X=0.42 BOX_SIZE_Y=0.28 BOX_SIZE_Z=0.28 BOX_COM_X=0.05 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.38 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.05 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=kinematic-pose-lock CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker command:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag11_chest_dynamic_pose_lock.log --require-attach --forbid-disjoint-warning --max-target-distance 0.04 --max-relative-error 0.05 --min-body-travel 0.18 --min-box-travel 0.45 --min-support-margin 0.10 --expect-strategy chest_supported_creep --expect-attachment-mode kinematic-pose-lock --expect-carrier-mode dynamic-velocity`.
  Result: pass as a multi-posture scaffold only. The run selected
  `chest_supported_creep`, completed 560/560 steps, attached at step 91, body
  travel was 0.22527 m, box travel was 0.52527 m, final box-target distance
  was 0.01473 m, post-attach relative error was `2.78e-08`, fall events 0,
  box drop events 0, no disjoint warning, and `min_support_margin_m=0.12430`.
  This verifies strategy switching under a heavier box and shorter arm
  morphology inside the dynamic rigid-body carrier scaffold, but it is still
  not physical grasping or verified articulated locomotion.
- [x] Run dynamic velocity-servo grasp proxy diagnostic:
  `20260704_core_world_staged_free_box_diag12_velocity_servo_grasp`, Slurm job
  `165332` on `server46`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag12_velocity_servo_grasp DEVICE=cpu STEPS=420 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=velocity-servo-grasp CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker command:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag12_velocity_servo_grasp/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag12_velocity_servo_grasp.log --require-attach --forbid-disjoint-warning --max-target-distance 0.05 --max-relative-error 0.08 --max-peak-relative-error 0.12 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10 --expect-strategy low_front_creep --expect-attachment-mode velocity-servo-grasp --expect-carrier-mode dynamic-velocity`.
  Result: pass as a non-contact grasp-proxy diagnostic. The box remains a
  dynamic rigid body after attach and is controlled by velocity servo rather
  than direct pose-lock. The run completed 420/420, selected
  `low_front_creep`, attached at step 91, body travel was 0.20545 m, box
  travel was 0.47544 m, final target distance was 0.01456 m, final relative
  error was `3.67e-06`, peak relative error was `3.91e-06`, fall/drop events
  were 0, no disjoint warning appeared, and `min_support_margin_m=0.13252`.
  This is closer to a physical grasp replacement than `kinematic-pose-lock`,
  but it is still not contact grasping or verified articulated locomotion.
- [x] Run explicit hand/chest contact-proxy servo diagnostic:
  `20260704_core_world_staged_free_box_diag13_contact_proxy_servo`, Slurm job
  `165340` on `server10`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag13_contact_proxy_servo DEVICE=cpu STEPS=420 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=contact-proxy-servo CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag13_contact_proxy_servo/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag13_contact_proxy_servo.log --require-attach --require-contact-proxy --forbid-disjoint-warning --max-target-distance 0.05 --max-relative-error 0.08 --max-peak-relative-error 0.12 --max-contact-proxy-gap 0.02 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10 --expect-strategy low_front_creep --expect-attachment-mode contact-proxy-servo --expect-carrier-mode dynamic-velocity`.
  Purpose: expose left/right palm and chest support proxy geometry in the
  Isaac scene and record grip-gap metrics while still keeping the box dynamic.
  This is the intended bridge from invisible velocity servo toward later real
  palm/chest contact or constraint grasping.
  Result: pass as a contact-proxy servo diagnostic. The run completed
  420/420, selected `low_front_creep`, attached at step 91, body travel was
  0.20545 m, box travel was 0.47544 m, final target distance was 0.01456 m,
  final relative error was `3.67e-06`, peak relative error was `3.91e-06`,
  contact-proxy grip gap was `3.67e-06` m, max grip gap was `3.91e-06` m,
  fall/drop events were 0, no disjoint warning appeared, and
  `min_support_margin_m=0.13252`. This exposes left/right palm and chest
  support proxy geometry while keeping the box dynamic, but it is still not
  physical contact grasping or verified articulated locomotion.
- [x] Add `ATTACHMENT_MODE=dynamic-contact-proxy` to the staged free-box
  scene. This mode uses dynamic left/right palm, chest, and forearm-shelf proxy
  rigid bodies. After the staged lift/attach event, the box is not directly
  velocity-servoed; the proxy bodies are velocity-servoed and the box is moved
  through PhysX contact. This remains a diagnostic bridge because the lift is
  still staged and the carrier is not an articulated walking robot.
- [x] Run dynamic contact-proxy negative control:
  `20260704_core_world_staged_free_box_diag14_dynamic_contact_proxy`, Slurm job
  `165343` on `server10`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag14_dynamic_contact_proxy DEVICE=cpu STEPS=420 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: fail. Dynamic proxies were active before attach and contaminated the
  probing phase, then pushed the box past the target. Checker failures:
  27 box drop events, final target distance 0.4703 m, final relative error
  0.2790 m, peak relative error 0.4172 m, max contact-proxy grip gap
  0.3974 m. This is a useful negative result and motivated standby gating.
- [x] Add pre-attach standby gating for dynamic contact proxies and run:
  `20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby`,
  Slurm job `165343` on `server10`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby DEVICE=cpu STEPS=420 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag15_dynamic_contact_proxy_standby.log --require-attach --require-contact-proxy --forbid-disjoint-warning --max-target-distance 0.05 --max-relative-error 0.08 --max-peak-relative-error 0.12 --max-contact-proxy-gap 0.12 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10 --expect-strategy low_front_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity`.
  Result: pass as the strongest current staged free-box Isaac diagnostic.
  Completed 420/420, selected `low_front_creep`, attached at step 91, body
  travel was 0.25052 m, box travel was 0.48505 m, final target distance was
  0.01490 m, final relative error was 0.06311 m, peak relative error was
  0.06348 m, contact-proxy grip gap was 0.06643 m, max grip gap was
  0.06818 m, fall/drop events were 0, no disjoint warning appeared, and
  `min_support_margin_m=0.13252`. This is not final robot-carrying success:
  the lift is staged, the proxy bodies are velocity commanded, and locomotion
  is still a dynamic-body support proxy rather than an articulated robot gait.
- [x] Run stricter dynamic-contact balance/hold gate:
  `20260704_core_world_staged_free_box_diag16_dynamic_contact_balance_hold`.
  Code changes added summary/checker fields for `target_hold_steps`,
  `carry_phase_steps`, `min_stance_count`,
  `min_support_margin_after_attach_m`, and `max_command_speed_mps`.
  Planned checker gate:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag16_dynamic_contact_balance_hold/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag16_dynamic_contact_balance_hold.log --require-attach --require-contact-proxy --require-dynamic-contact-proxy --forbid-disjoint-warning --max-target-distance 0.05 --max-relative-error 0.08 --max-peak-relative-error 0.12 --max-contact-proxy-gap 0.12 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10 --min-support-margin-after-attach 0.10 --min-stance-count 3 --min-target-hold-steps 5 --expect-strategy low_front_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity`.
  Result: pass. Completed 430/430, selected `low_front_creep`, attached at
  step 91, body travel was 0.25052 m, box travel was 0.48435 m, final target
  distance was 0.01474 m, final and peak relative error were 0.06428 m,
  contact-proxy grip gap was 0.06825 m, max grip gap was 0.06825 m,
  `target_hold_steps=24`, `carry_phase_steps=338`, `min_stance_count=3.0`,
  `min_support_margin_after_attach_m=0.13252`, max command speed was
  0.174 m/s, fall/drop events were 0, and no disjoint warning appeared. This
  is the strongest current balance/hold gate for the staged free-box scaffold,
  but it is still not final success because the carrier is not yet a verified
  articulated walking robot.
- [x] Expose artificial vertical stabilization in the staged free-box carrier:
  added `BODY_VERTICAL_MODE=preserve`, `PHYSICAL_SUPPORT_MODE=deck`,
  `SUPPORT_DECK_GAP`, `body_vertical_velocity_preserve_available`,
  `min_body_z_m`, and `max_body_z_deviation_m`. The goal is to distinguish
  previous horizontal velocity commands that implicitly zeroed vertical
  velocity from a stricter gravity/contact-support diagnostic.
- [x] Run preserve-z physical support-deck diagnostic:
  `20260704_core_world_staged_free_box_diag17_dynamic_contact_preserve_z_deck`,
  Slurm job `165348` on `server28`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag17_dynamic_contact_preserve_z_deck DEVICE=cpu STEPS=430 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 BODY_VERTICAL_MODE=preserve PHYSICAL_SUPPORT_MODE=deck ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker result: fail. `body_vertical_velocity_preserve_available=True`,
  completed 430/430, attached at step 105, body travel 0.26733 m, box travel
  0.80080 m, but box drop events were 18, final target distance was 0.31088 m,
  final relative error was 1.17468 m, peak relative error was 1.22473 m, max
  contact-proxy grip gap was 0.46564 m, and `max_body_z_deviation_m=1.39366`.
  Interpretation: preserving vertical velocity exposed that the support deck
  was overconstrained/too tightly preloaded and launched the body upward before
  attach; previous zero-z-velocity diagnostics must not be reported as true
  gravity-balanced locomotion.
- [x] Re-run the preserve-z support-deck diagnostic in a fresh allocation after
  the `SUPPORT_DECK_GAP` patch. First attempted stamp
  `20260704_core_world_staged_free_box_diag18_dynamic_contact_preserve_z_deck_gap`
  was interrupted during a second consecutive Kit startup in the same
  allocation and produced no summary. Planned command should use
  `SUPPORT_DECK_GAP=0.02`:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag18_dynamic_contact_preserve_z_deck_gap DEVICE=cpu STEPS=430 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 BODY_VERTICAL_MODE=preserve PHYSICAL_SUPPORT_MODE=deck SUPPORT_DECK_GAP=0.02 ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker must include
  `--expect-body-vertical-mode preserve --expect-physical-support-mode deck --expect-support-deck-gap 0.02 --max-body-z-deviation 0.08`.
  A fresh Slurm allocation attempt `165353` stayed pending with reason
  `(Priority)` and was canceled before allocation, so no diag18 result exists
  yet. This item is closed as an invalid/no-result attempt rather than carried
  forward as the active path.
- [x] Add `PHYSICAL_SUPPORT_MODE=runway`, a fixed long support surface that
  does not follow the body every step. This tests whether preserve-z gravity
  support can avoid the moving-deck energy injection observed in `diag17`.
- [x] Run preserve-z fixed-runway diagnostic:
  `20260704_core_world_staged_free_box_diag19_dynamic_contact_preserve_z_runway`,
  Slurm job `165355` on `server02`.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag19_dynamic_contact_preserve_z_runway DEVICE=cpu STEPS=430 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 BODY_VERTICAL_MODE=preserve PHYSICAL_SUPPORT_MODE=runway SUPPORT_DECK_GAP=0.02 ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker result: fail. It completed 430/430, `body_vertical_velocity_preserve_available=True`,
  attached at step 98, body travel was 0.21877 m, box travel was 0.46461 m,
  final target distance was 0.04524 m, fall/drop events were 0, and no
  disjoint warning appeared. However `target_hold_steps=0`,
  `max_body_z_deviation_m=1.92729`, final relative error was 1.69750 m, peak
  relative error was 1.70690 m, and max contact-proxy grip gap was 0.21576 m.
  Interpretation: a fixed runway removed box drops but still injected enough
  vertical contact impulse to lift the body far above its intended height.
  The preserve-z support-surface route is not valid balance evidence yet.
  Next valid work should either create a geometry-consistent body/support
  setup with no launch, or move back toward a real articulated foot-contact
  controller rather than tuning the staged carrier as if it were a robot gait.
- [x] Run geometry-corrected preserve-z fixed-runway diagnostic:
  `20260704_core_world_staged_free_box_diag20_dynamic_contact_preserve_z_runway_heightfix`.
  Code now records `support_surface_top_z_m`,
  `initial_body_bottom_z_m`, and `initial_body_support_clearance_m`, and places
  the support surface using the observed Isaac Core cuboid half-extent
  convention. Purpose: test only whether the previous preserve-z body launch
  was caused by support/body overlap. This remains a diagnostic scaffold, not
  articulated robot carrying evidence.
  Planned Slurm allocation: `165357` in `isaac-setup`, pending with
  `(Priority)` at creation time.
  Command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag20_dynamic_contact_preserve_z_runway_heightfix DEVICE=cpu STEPS=430 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 BODY_VERTICAL_MODE=preserve PHYSICAL_SUPPORT_MODE=runway SUPPORT_DECK_GAP=0.02 ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker included
  `--expect-body-vertical-mode preserve --expect-physical-support-mode runway --expect-support-deck-gap 0.02 --max-body-z-deviation 0.08`.
  Result: fail under strict gate. Completed 430/430 and attached at step 91;
  final target distance was 0.01735 m, body travel 0.28876 m, box travel
  0.47797 m, fall/drop events 0, no disjoint warning, and the corrected
  initial body/support clearance was 0.0200 m. However target hold steps were
  0, `max_body_z_deviation_m=0.47210`, final relative error was 0.28226 m,
  peak relative error was 0.32530 m, and max contact-proxy grip gap was
  0.19360 m. Interpretation: height correction improved the extreme launch
  from `diag19` but preserve-z staged support still is not valid balance or
  contact-carry evidence. Do not keep tuning this as if it were robot gait.
- [x] Add lightweight staged free-box result checker:
  `scripts/isaac/check_staged_free_box_summary.py`. It validates completed
  steps, attach step, target distance, post-attach relative error, fall/drop,
  and optional absence of the disjoint fixed-joint warning. It is safe on the
  login node because it only reads JSON/log text.
- [x] Run checker on `20260704_core_world_staged_free_box_diag4_settle_attach`
  with `--require-attach --forbid-disjoint-warning --max-target-distance 0.03
  --max-relative-error 0.05`. Result: fail, because
  `box_relative_error_m_after_attach=0.0824` and the log contains the disjoint
  warning. This confirms `diag4` must not be reported as stable attach
  success.
- [x] Run direct Isaac staged free-box diagnostic with body-level
  `height-servo` carrier control instead of waiting on external models or
  official policy baselines. This is a labeled scaffold, not articulated robot
  balance. Planned command:
  `timeout 420s env STAMP=20260704_core_world_staged_free_box_diag21_dynamic_contact_height_servo DEVICE=cpu STEPS=430 TARGET_X=0.65 BOX_X=0.16 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=90 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=18.0 BODY_HEIGHT_MAX_Z_SPEED=0.80 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Planned checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260704_core_world_staged_free_box_diag21_dynamic_contact_height_servo/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260704_core_world_staged_free_box_diag21_dynamic_contact_height_servo.log --require-attach --require-contact-proxy --require-dynamic-contact-proxy --forbid-disjoint-warning --max-target-distance 0.05 --max-relative-error 0.08 --max-peak-relative-error 0.12 --max-contact-proxy-gap 0.12 --min-body-travel 0.18 --min-box-travel 0.40 --min-support-margin 0.10 --min-support-margin-after-attach 0.10 --min-stance-count 3 --min-target-hold-steps 5 --max-body-z-deviation 0.08 --expect-body-vertical-mode height-servo --expect-physical-support-mode none --expect-strategy low_front_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity`.
  Result: fail under strict gate, but it is useful negative evidence. Completed
  430/430, attached at step 91, fall/drop events 0, no disjoint warning,
  `max_body_z_deviation_m=0.03973`, body travel 0.23770 m, and box travel
  0.70160 m. It failed because the target-hold threshold was too tight for the
  dynamic contact proxy: `target_hold_steps=0`, final target distance
  0.22551 m after overshoot, final relative error 0.19346 m, peak relative
  error 0.19583 m, and max contact-proxy gap 0.17568 m.
- [x] Run `diag22` with the same height-servo/dynamic-contact-proxy scene but
  parameterized target arrival control: `TARGET_HOLD_RADIUS=0.04` and
  `TARGET_SLOW_RADIUS=0.12`. This tests whether `diag21` failed because the
  target hold radius was below the observed closest approach, not because
  height-servo itself is unstable.
  Result: fail under strict final-distance/contact gate. Completed 430/430,
  attached at step 91, fall/drop events 0, no disjoint warning,
  `target_hold_steps=33`, `max_body_z_deviation_m=0.03973`, body travel
  0.16084 m, and box travel 0.62874 m. It still ended outside target with
  final distance 0.16028 m because the phase logic exited target hold after
  contact rebound. Code now records and uses `target_hold_latched` so once the
  target is reached it stays in hold.
- [x] Run `diag23` with target-hold latch enabled, same
  `height-servo`/`dynamic-contact-proxy` setup, `TARGET_HOLD_RADIUS=0.04`, and
  `TARGET_SLOW_RADIUS=0.12`.
  Result: fail under strict gate and correctly exposes a false-positive risk.
  Completed 430/430, attached at step 91, `target_hold_latched=True`,
  `target_hold_steps=207`, fall/drop events 0, no disjoint warning, and
  `max_body_z_deviation_m=0.03973`. However body travel was only 0.04232 m
  while box travel was 0.51028 m; final target distance was 0.08264 m, final
  relative error 0.19821 m, and max contact-proxy gap 0.17965 m. This means
  box-only arrival can let the proxy push the box toward the target without
  the carrier actually moving enough. Code now adds `TARGET_BODY_MARGIN` and
  requires body x to approach `target_x - attach_local_x` before latching
  target hold.
- [x] Run `diag24` with body-aware target hold enabled:
  `TARGET_HOLD_RADIUS=0.04`, `TARGET_SLOW_RADIUS=0.12`,
  `TARGET_BODY_MARGIN=0.02`, `BODY_VERTICAL_MODE=height-servo`,
  `ATTACHMENT_MODE=dynamic-contact-proxy`.
  Result: fail under strict gate, but it fixed the false target-hold latch.
  Completed 430/430, attached at step 91, fall/drop events 0, no disjoint
  warning, `max_body_z_deviation_m=0.03973`, body travel 0.17504 m, and box
  travel 0.64341 m. It did not latch target hold because the body had not
  reached the body-aware target position (`target_body_x_m=0.20009`); final
  target distance was 0.17318 m, final relative error 0.19877 m, and max
  contact-proxy gap 0.17985 m. Interpretation: side/chest/shelf contact proxy
  can push the box forward but cannot restrain it from leading the body. Code
  now adds `FrontStopProxy` to the dynamic contact proxy group.
- [x] Run `diag25` with the added `FrontStopProxy`, same body-aware
  height-servo dynamic-contact-proxy setup.
  Result: fail under strict gate but materially improved contact holding.
  Completed 430/430, attached at step 154, fall/drop events 0, no disjoint
  warning, final relative error 0.07332 m, final contact-proxy gap 0.07133 m,
  max contact-proxy gap 0.11776 m, final target distance 0.06006 m, and box
  travel 0.43047 m. Compared with `diag24`, the front stop reduced the final
  relative error from 0.19877 m to 0.07332 m. It still failed because body
  travel was only 0.12554 m, target hold never latched, peak relative error was
  0.13089 m, and `max_body_z_deviation_m=0.15208`. Next diagnostic should give
  the body enough carry time and strengthen height-servo recovery.
- [x] Run `diag26` with `FrontStopProxy`, longer horizon, and stronger
  height-servo: `STEPS=650`, `BODY_HEIGHT_GAIN=30.0`,
  `BODY_HEIGHT_MAX_Z_SPEED=1.20`.
  Result: fail under strict gate but close on the horizontal carry task.
  Completed 650/650, attached at step 148, body travel 0.20005 m, box travel
  0.51237 m, final target distance 0.04528 m, final contact-proxy gap
  0.07851 m, peak gap 0.09538 m, fall/drop events 0, and no disjoint warning.
  It failed because target hold never latched, final relative error was
  0.09012 m, and `max_body_z_deviation_m=0.14254`. Interpretation:
  front-stop contact plus longer horizon makes the staged carry reach the
  target region, but height-servo/contact coupling still injects too much
  vertical deviation for the strict physical-support gate.
- [x] Run `diag27` as a stable scaffold comparison: same front-stop,
  body-aware, long-horizon dynamic-contact-proxy setup but
  `BODY_VERTICAL_MODE=zero`. This is not physical balance evidence; it tests
  whether the remaining failure is height-servo/contact coupling rather than
  target/proxy geometry.
  Result: fail under strict z/relative-error gate but it produced the best
  horizontal scaffold behavior so far. Completed 650/650, attached at step 91,
  `target_hold_latched=True`, `target_hold_steps=305`, final target distance
  0.04748 m, body travel 0.19681 m, box travel 0.46254 m, final/peak
  contact-proxy gap 0.09784/0.11914 m, fall/drop events 0, and no disjoint
  warning. It failed because `max_body_z_deviation_m=0.19743` and final
  relative error was 0.10559 m. Code now adds explicit `BODY_VERTICAL_MODE=height-lock`
  as a labeled scaffold that locks carrier height after each physics step
  while leaving the box dynamic/contact-carried.
- [x] Run `diag28` with `BODY_VERTICAL_MODE=height-lock`, front-stop,
  body-aware target hold, and long horizon. This can only be claimed as a
  stable Isaac task scaffold, not physical balance.
  Result: fail and do not pursue unchanged. Completed 650/650 but never
  attached: `attach_step=None`, body travel -0.02931 m, box travel -0.00348 m,
  and final target distance 0.49360 m. Height lock kept
  `max_body_z_deviation_m=0.0`, but it disrupted approach/proxy interaction so
  badly that the staged carry never started. Interpretation: do not use
  `height-lock` as the active path; it is too artificial and harms the task
  sequence. Current best scaffold remains `diag27` for stable horizontal
  carry/hold behavior and `diag26` for the most physical-ish height-servo
  comparison, both clearly labeled as not real balance.
- [ ] Next direct Isaac work should stay on scene/control construction, not
  model waiting: either reduce `height-servo`/front-stop vertical coupling
  while preserving the `diag26` horizontal behavior, or replace the
  velocity-commanded carrier with a real articulated foot-contact controller.
  Do not report `diag27` as physical balance or real grasping success.
- [x] Run `diag29` with `CARRY_GEOMETRY_MODE=nonpenetrating` to move away from
  the legacy potentially overlapping carry pose. Planned command:
  `timeout 720s env STAMP=20260704_core_world_staged_free_box_diag29_nonpenetrating_front_stop_height_servo DEVICE=cpu STEPS=900 TARGET_X=1.65 BOX_X=0.95 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.30 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=180 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_CLEARANCE=0.03 TARGET_HOLD_RADIUS=0.06 TARGET_SLOW_RADIUS=0.18 TARGET_BODY_MARGIN=0.03 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=30.0 BODY_HEIGHT_MAX_Z_SPEED=1.20 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  This is a harder and more physical geometry diagnostic, not a success claim.
  Result: fail before attach. Completed 900/900 with fall/drop events 0 and no
  disjoint warning, but `attach_step=None`, body travel 0.44107 m, box travel
  ~0, final target distance 0.70000 m, and `max_body_z_deviation_m=0.16989`.
  Cause: the phase logic still used the legacy `box_x - 0.16` approach stop,
  which is wrong for `actual_staged_carry_x_m=0.85`. Code now records
  `approach_body_x_m` and uses `box_x - carry_x` in nonpenetrating mode.
- [x] Run `diag30` after the nonpenetrating approach-trigger fix with the same
  geometry and target, requiring attach and nonzero carry.
  Result: fail after attach. Completed 900/900, attached at step 219,
  `approach_body_x_m=0.10`, body travel 0.69132 m, fall events 0, and no
  disjoint warning. The approach fix worked, but the nonpenetrating carry
  dropped the box: `box_drop_events=58`, box travel only 0.05254 m, final
  target distance 0.64801 m, final relative error 0.56029 m, and max
  contact-proxy gap 0.53409 m. Code now adds `CARRY_Z_OFFSET` so long-front
  nonpenetrating carries can lift the box higher instead of dragging it low.
- [x] Run `diag31` with nonpenetrating geometry plus `CARRY_Z_OFFSET=0.18`.
  Result: fail. Completed 900/900, attached at step 219, body travel
  0.69153 m, fall events 0, and no disjoint warning, but
  `box_drop_events=59`, box travel -0.01895 m, final target distance
  0.71897 m, final relative error 0.71582 m, and max contact-proxy gap
  0.60932 m. Lifting the carry pose alone did not solve long-front
  nonpenetrating contact closure. Code now exposes `CONTACT_PROXY_GAIN` and
  `CONTACT_PROXY_MAX_SPEED` for the dynamic proxy controller.
- [x] Run `diag32` with nonpenetrating geometry, lifted carry,
  `CONTACT_PROXY_GAIN=30`, `CONTACT_PROXY_MAX_SPEED=2.0`, and slower
  `BASE_SPEED=0.18`.
  Result: fail only on target reach/hold, with major contact improvement.
  Completed 1200/1200, attached at step 340, body travel 0.54573 m, box travel
  0.39087 m, fall/drop events 0, no disjoint warning, final relative error
  0.05155 m, peak relative error 0.05155 m, final/peak contact-proxy gap
  0.04924/0.04931 m, and `max_body_z_deviation_m=0.15341`. It failed because
  target was too far (`final_box_target_distance_xy_m=0.30921`,
  `target_hold_steps=0`). This shows the nonpenetrating/lifted/strong-proxy
  contact route can hold the box; next diagnostic should keep the same
  controller and place the target within the reached trajectory to verify
  body-aware target hold.
- [x] Run `diag33` with the same nonpenetrating/lifted/strong-proxy controller
  as `diag32`, but `TARGET_X=1.35`.
  Result: fail only because target hold did not latch. Completed 1200/1200,
  attached at step 340, body travel 0.48528 m, box travel 0.33042 m,
  fall/drop events 0, no disjoint warning, final target distance 0.06991 m,
  final/peak relative error 0.05155/0.05155 m, final/peak proxy gap
  0.04923/0.04931 m, and `max_body_z_deviation_m=0.15341`. This satisfies the
  contact, carry, and safety scaffold checks, but misses
  `TARGET_HOLD_RADIUS=0.06`. Next run extends the horizon without changing
  controller behavior.
- [x] Run `diag34` same as `diag33`, but `STEPS=1350`.
  Result: pass under the declared nonpenetrating staged-free-box scaffold gate.
  Completed 1350/1350, attached at step 340, `target_hold_latched=True`,
  `target_hold_steps=97`, body travel 0.49526 m, box travel 0.34040 m,
  final target distance 0.05999 m, final/peak relative error
  0.05155/0.05155 m, final/peak contact-proxy gap 0.04926/0.04931 m,
  `min_support_margin_after_attach_m=0.13407`, `min_stance_count=3.0`,
  `max_body_z_deviation_m=0.15341`, fall/drop events 0, and no disjoint
  warning. This is a strong direct Isaac task scaffold with nonpenetrating
  carry geometry, lifted carry pose, dynamic contact proxies, and body-aware
  target hold. It is still not final robot success because the carrier is a
  velocity-commanded dynamic body with support proxies, not an articulated
  walking controller.
- [x] Run `diag35` for a heavier `chest_supported_creep` strategy using the
  same nonpenetrating/lifted/strong-proxy controller, to test posture diversity
  instead of only low-front carry.
  Result: pass under the declared nonpenetrating staged-free-box scaffold gate.
  Completed 1600/1600, selected `chest_supported_creep`, attached at step 339,
  `target_hold_latched=True`, `target_hold_steps=295`, body travel 0.38340 m,
  box travel 0.24055 m, final target distance 0.05992 m, final/peak relative
  error 0.04152/0.04167 m, final/peak contact-proxy gap 0.04456/0.04667 m,
  `min_support_margin_after_attach_m=0.13828`, `min_stance_count=3.0`,
  `max_body_z_deviation_m=0.15145`, fall/drop events 0, and no disjoint
  warning. This gives two passing nonpenetrating staged-free-box scaffolds:
  `diag34` for `low_front_creep` and `diag35` for `chest_supported_creep`.
  They are still not final robot success because the carrier remains a
  velocity-commanded dynamic body with support proxies, not an articulated
  walking robot.
- [ ] Next required milestone: replace the velocity-commanded body carrier in
  the staged free-box scene with an articulated foot-contact carrier while
  preserving the proven nonpenetrating/lifted/strong-proxy task interface and
  checker gates from `diag34`/`diag35`.
- [ ] Add articulated-carrier evidence gates before claiming robot walking:
  staged-free-box summaries now expose `carrier_evidence_mode`,
  `articulated_carrier_enabled`, `articulated_joint_count`, and
  `foot_contact_drive_enabled`, and the staged checker has
  `--require-articulated-carrier`. Existing `diag34`/`diag35` should not pass
  this gate because they are support-proxy carrier scaffolds.
- [ ] Run `diag36` on the custom articulated dynamic quadruped path with
  staged free box, contact proxy, front stop, stronger proxy gain/speed, and
  joint-motion checker. This is an articulated scaffold migration diagnostic,
  not final unassisted locomotion because base pose assist may still be used.
  Result: fail, but it proves the custom articulated path is active. Completed
  700/700, attached at step 90, `max_joint_motion_rad=0.65770`, torso travel
  0.36005 m, box travel 0.51631 m, fall events 0, tilt 0, no disjoint warning,
  and control errors 0. It failed because `box_drop_events=55`,
  `target_hold_steps=0`, final target distance 0.21247 m, final relative error
  0.52425 m, peak relative error 0.65225 m, and max proxy gap 1.37791 m. Code
  now adds `TARGET_HOLD_RADIUS` and target-hold latch/feedforward-stop logic
  to the dynamic quadruped staged path.
- [x] Run `diag37` with the dynamic quadruped target-hold latch enabled.
  Result: fail. Completed 700/700, attached at step 90,
  `max_joint_motion_rad=0.66152`, box travel 0.51631 m, fall events 0, tilt 0,
  no disjoint warning, and control errors 0. The target-hold latch fired too
  early: torso travel was only 0.03840 m, `box_drop_events=55`,
  final target distance 0.21247 m, final/peak relative error
  0.68832/0.69670 m, and max contact-proxy gap 1.37791 m.
- [x] Add body-aware target-hold gating to the dynamic quadruped staged path.
  `target_hold` now requires the box to be near the target and the carrier
  body to have reached the corresponding support position; pose assist also
  uses configurable `carry_local_x` instead of the previous hardcoded `0.26`.
- [x] Run `diag38` with body-aware target-hold gating.
  Result: fail, but it fixed the early-hold bug. Completed 760/760, attached
  at step 90, `target_hold_body_ready=True`, torso travel 0.36005 m,
  box travel 0.51631 m, `max_joint_motion_rad=0.65770`, fall events 0, tilt 0,
  no disjoint warning, and control errors 0. It still failed because the box
  dropped before hold: `box_drop_events=61`, `target_hold_steps=0`, final
  target distance 0.21247 m, final/peak relative error 0.52425/0.65225 m.
- [x] Add pre-placement of contact proxies during staged lift/attach in the
  dynamic quadruped path, matching the successful staged free-box scaffold's
  lift/settle behavior. Max contact-proxy grip gap is now tracked after attach
  instead of being polluted by pre-attach standby distance.
- [x] Run `diag39b` on the custom articulated dynamic quadruped path with
  staged free box, preplaced contact proxies, body-aware hold, and joint-motion
  checker.
  Result: pass under the declared articulated-scaffold diagnostic gate.
  Completed 760/760, attached at step 90, `target_hold_latched=True`,
  `target_hold_steps=26`, `target_hold_body_ready=True`, torso travel
  0.33040 m, box travel 0.40108 m, final target distance 0.07327 m,
  final/peak relative error 0.10293/0.10440 m, final/peak contact-proxy gap
  0.08576/0.08787 m, `max_joint_motion_rad=0.87277`, fall/drop events 0,
  tilt 0, no disjoint warning, and control errors 0. This is real progress on
  the direct Isaac path, but it is still not final robot carrying success:
  the quadruped body uses pose assist, the grasp is staged, and the
  palm/chest/shelf/front-stop contact proxies are engineering scaffolds rather
  than a learned physical hand/controller.
- [x] Run `diag40` replacing pose assist with root velocity assist.
  Result: fail, but it proves `diag39b` depends on root pose writes. Completed
  760/760 with `root_pose_write_count=0`, `root_velocity_write_count=760`,
  attached at step 90, and drop events 0. It failed with 72 fall events,
  final target distance 3.66150 m, and max tilt 3.09467 rad. This means simple
  velocity assist cannot replace pose teleporting.
- [x] Add `BASE_ASSIST_MODE=upright_velocity`, which writes root linear and
  angular velocities for height/lateral/upright stabilization without writing
  root pose. Add evidence fields for root pose, linear velocity, and angular
  velocity write counts.
- [x] Run `diag41` with upright velocity assist.
  Result: fail but informative. It reduced roll/pitch compared with `diag40`
  and kept drop events 0, but torso height fell below the fall threshold:
  60 fall events, final target distance 2.76723 m, max tilt 0.12054 rad,
  `root_pose_write_count=0`, `root_velocity_write_count=760`,
  `root_angular_velocity_write_count=760`.
- [x] Add x-position velocity servo, height-gain tuning, post-step velocity
  assist option, and `BASE_X_COMMAND_SCALE` diagnostic knob for the
  `upright_velocity` route. These are still scaffold controls, but they avoid
  direct `set_world_pose` root teleportation.
- [x] Run `diag42`/`diag43`/`diag44b` to test upright velocity plus x servo.
  Result: all fail target reach but improve safety relative to `diag40`.
  `diag42` kept fall/drop events 0 but final target distance was 2.25080 m.
  Stronger x servo in `diag43` reduced final target distance to 0.75432 m,
  with fall/drop events 0, `root_pose_write_count=0`, final/peak relative
  error 0.16750/0.16941 m, and max contact-proxy gap 0.18166 m. `diag44b`
  added post-step velocity writes but reproduced `diag43`, so the timing
  change did not help.
- [x] Diagnose `BASE_X_COMMAND_SCALE` propagation. Launcher/env runs did not
  pass the new value into Python summary, while direct Python invocation did.
  Until the launcher path is repaired, use direct Python commands for new
  `base_x_command_scale` diagnostics and record the exact command.
- [x] Run direct Python `diag45e` and `diag46b` with
  `base_x_command_scale=-1.0`.
  Result: negative scale is not the solution. Direct `diag45e` confirmed the
  scale reaches argparse and causes positive world-x motion, but it drives the
  carrier far past the target before/after attach. `diag46b` disabled
  pre-attach x velocity and still failed badly: completed 760/760, fall/drop
  events 0, `root_pose_write_count=0`, `root_velocity_write_count=760`,
  final target distance 4.22659 m, final/peak relative error
  0.13542/0.14207 m, and max contact-proxy gap 0.15202 m.
- [x] Next required control milestone: stop trying to make root velocity writes
  act as locomotion. Build a foot/support driven carrier controller or import
  a functioning controller-backed robot policy so forward progress comes from
  foot contacts/support dynamics rather than root pose or root velocity
  shortcuts. Keep the `diag39b` task gate as a scaffold baseline, but do not
  claim walking/balance success from root-assisted runs.
- [x] Add strict no-root checker gates for dynamic quadruped carry summaries.
  `check_dynamic_quadruped_carry_summary.py` now supports
  `--max-root-pose-writes`, `--max-root-velocity-writes`, and
  `--max-root-angular-velocity-writes`.
- [x] Add a direct Isaac `SUPPORT_DRIVE` diagnostic path for contact-support
  pads in `build_core_world_dynamic_quadruped_carry_scene.py` and launcher
  support in `run_core_world_dynamic_quadruped_carry_scene.sh`. This is a
  labeled contact-support scaffold, not a final locomotion controller.
- [x] Run `diag47_support_drive_no_root` in Slurm job `165551` on `server46`.
  Result: stopped early. Kinematic support pads produced repeated PhysX errors
  because `setLinearVelocity` is illegal on kinematic bodies. Code was changed
  so support pads are dynamic high-mass rigid bodies instead.
- [x] Run
  `20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root`
  in the same compute allocation. Result: fail, but root-write free. Completed
  760/760, staged free box, contact proxies enabled, attached at step 90,
  `root_pose_write_count=0`, `root_velocity_write_count=0`,
  `root_angular_velocity_write_count=0`, and support pad writes 3040 pose /
  3040 velocity. It failed with `fall_events=70`, `box_drop_events=53`,
  `target_hold_steps=0`, final target distance `4.11404 m`,
  max relative error `37.74582 m`, max contact-proxy gap `37.78694 m`, and max
  tilt `3.19234 rad`. Late non-finite PhysX state appeared after the carrier
  had already fallen badly. Do not treat this as carrying success.
  Command:
  `timeout 620s env STAMP=20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root DEVICE=cpu STEPS=760 PAYLOAD_MODE=staged_free_box STAGED_ATTACH_MODE=contact-proxy BOX_X=0.30 PAYLOAD_MASS=4.0 TARGET_X=0.62 TARGET_SPEED=0.16 TARGET_HOLD_RADIUS=0.08 TARGET_BODY_MARGIN=0.03 MIN_HOLD_TORSO_TRAVEL=0.25 CARRY_LOCAL_X=0.26 CARRY_LOCAL_Z=0.03 CONTACT_PROXY_GAIN=30.0 CONTACT_PROXY_MAX_SPEED=2.0 ATTACH_AFTER_STEP=90 PROBE_SPEED=0.035 SUPPORT_DRIVE=1 SUPPORT_DRIVE_GAIN=3.0 SUPPORT_DRIVE_MAX_SPEED=0.45 SUPPORT_PAD_Z=0.018 GAIT_FREQUENCY=1.2 HIP_AMPLITUDE_DEG=18.0 KNEE_AMPLITUDE_DEG=16.0 BASE_VELOCITY_ASSIST=0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag47b_support_drive_dynamic_pads_no_root.log`.
- [x] Run
  `20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root` to
  isolate whether the custom articulated carrier can stand without root writes.
  Result: fail. Fixed payload, target speed 0, gait amplitudes 0, support
  drive off, completed 240/240 with all root write counts 0, but
  `fall_events=20`, max tilt `2.81150 rad`, and final target distance
  `1.43736 m`. This shows the immediate blocker is stand/balance of the custom
  articulated carrier, before long-distance carrying or video conditioning.
  Command:
  `timeout 300s env STAMP=20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root DEVICE=cpu STEPS=240 PAYLOAD_MODE=fixed_joint_to_torso PAYLOAD_MASS=4.0 TARGET_X=0.0 TARGET_SPEED=0.0 SUPPORT_DRIVE=0 GAIT_FREQUENCY=1.0 HIP_AMPLITUDE_DEG=0.0 KNEE_AMPLITUDE_DEG=0.0 BASE_VELOCITY_ASSIST=0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag48_stand_fixed_payload_no_root.log`.
- [x] Add explicit no-root stand diagnostic controls to the dynamic quadruped
  path: hip/knee neutral angles, stance half-length/width, foot size, contact
  friction, and hip/knee PD stiffness/damping/max-force. The launcher now
  passes these as environment variables.
- [x] Run
  `20260705_core_world_dynamic_quad_diag49_neutral_stand_fixed_payload_no_root`
  in Slurm job `165568` on `server53`. Result: fail. Fixed 4 kg payload,
  zero root writes, neutral hip/knee targets, zero gait amplitudes, completed
  240/240 with `fall_events=21`, `box_drop_events=0`, max tilt
  `3.20087 rad`, and max joint motion `0.33950 rad`. This proves `diag48`
  was not failing only because of the old `-5/-18 deg` neutral offset.
  Command:
  `timeout 300s env STAMP=20260705_core_world_dynamic_quad_diag49_neutral_stand_fixed_payload_no_root DEVICE=cpu STEPS=240 PAYLOAD_MODE=fixed_joint_to_torso PAYLOAD_MASS=4.0 TARGET_X=0.0 TARGET_SPEED=0.0 SUPPORT_DRIVE=0 GAIT_FREQUENCY=1.0 HIP_NEUTRAL_DEG=0.0 KNEE_NEUTRAL_DEG=0.0 HIP_AMPLITUDE_DEG=0.0 KNEE_AMPLITUDE_DEG=0.0 BASE_VELOCITY_ASSIST=0 bash scripts/isaac/run_core_world_dynamic_quadruped_carry_scene.sh`.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag49_neutral_stand_fixed_payload_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag49_neutral_stand_fixed_payload_no_root.log`.
- [x] Run
  `20260705_core_world_dynamic_quad_diag50_widefeet_stand_fixed_payload_no_root`.
  Result: fail but informative. Same neutral no-root stand test with 4 kg
  payload, wider stance (`0.30 x 0.26 m`) and larger feet
  (`0.34 x 0.18 x 0.055 m`) delayed the first fall from about step 20 to
  step 40, but still ended with `fall_events=21`, `box_drop_events=11`, max
  tilt `2.99903 rad`.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag50_widefeet_stand_fixed_payload_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag50_widefeet_stand_fixed_payload_no_root.log`.
- [x] Run
  `20260705_core_world_dynamic_quad_diag51_widefeet_light_payload_stand_no_root`.
  Result: fail but improved. Same wide body with 0.5 kg payload completed
  240/240 with zero root writes, `fall_events=14`, `box_drop_events=8`, max
  tilt `1.93540 rad`, and no non-finite joints. Lower payload helps but does
  not solve no-root stand.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag51_widefeet_light_payload_stand_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag51_widefeet_light_payload_stand_no_root.log`.
- [x] Run
  `20260705_core_world_dynamic_quad_diag52_highfriction_highpd_light_payload_stand_no_root`.
  Result: fail, not valid stand evidence. High friction (`4.0/3.5`) and high
  PD (`hip 5000/450/4000`, `knee 4500/420/3500`) reduced fall events to 3 and
  drop events to 0, but produced 19 non-finite joint events and repeated PhysX
  non-finite broadphase errors.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag52_highfriction_highpd_light_payload_stand_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag52_highfriction_highpd_light_payload_stand_no_root.log`.
- [x] Run
  `20260705_core_world_dynamic_quad_diag53_midfriction_midpd_light_payload_stand_no_root`.
  Result: fail. Moderate friction/PD avoided the non-finite state but regressed
  to `fall_events=24`, `box_drop_events=16`, max tilt `2.32909 rad`. Do not
  keep blindly increasing contact friction or PD.
  Output:
  `experiments/outputs/core_world_dynamic_quadruped_carry_scene/20260705_core_world_dynamic_quad_diag53_midfriction_midpd_light_payload_stand_no_root/`.
  Log:
  `logs/core_world_dynamic_quadruped_carry_scene/core_world_dynamic_quadruped_carry_scene_20260705_core_world_dynamic_quad_diag53_midfriction_midpd_light_payload_stand_no_root.log`.
- [ ] Build a no-root stand/balance controller diagnostic before any more
  long-distance staged-carry tuning. Required first gate: fixed payload,
  target speed 0, 240+ steps, root pose/velocity/angular-velocity writes all
  0, fall/drop events 0, max tilt below 0.85 rad, and no non-finite state.
- [ ] Replace or redesign the current custom two-DOF vertical-leg carrier
  before more carry attempts. The evidence from `diag49`-`diag53` shows that
  neutral targets, wider feet, lighter payload, friction, and PD tuning are
  insufficient for no-root standing. Next viable options are a true
  controller-backed quadruped/humanoid or a redesigned statically stable
  articulated leg/foot model with explicit balance control.
- [x] Prepare a stricter MuJoCo dynamic quadruped fixed-payload baseline for
  controller-backed direction finding. `run_quadruped_payload_carry.py` now
  records `assist_mode`, external force/torque write counts, and root
  pose/velocity write counts; `run_quadruped_payload_carry.sh` exposes the
  assist-force parameters; `check_quadruped_payload_summary.py` adds a
  lightweight JSON gate for travel, fall events, tilt, root writes, and
  external-force writes. This baseline is still a fallback diagnostic with a
  welded payload and optional external body-force stabilizer, not final
  unknown-box carrying.
- [ ] Run the MuJoCo dynamic quadruped baseline in a compute allocation with
  `ASSIST_MODE=body_force`, require nonzero travel, fall events 0, root writes
  0, and record the external-force write count. Then run `ASSIST_MODE=none` as
  a negative control if the assisted run passes.
  2026-07-05 resource note: attempted allocations `165590`, `165591`,
  `165595`, and `165597` did not produce a usable persistent shell for running
  the diagnostic. No MuJoCo rollout result exists yet.
- [x] Reset the active work back to direct Isaac scene construction after user
  correction instead of waiting on models or MuJoCo fallback.
- [x] Run direct Isaac staged free-box nonpenetrating low-front carry rerun:
  `20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating`.
  Command:
  `timeout 900s env STAMP=20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating DEVICE=cpu STEPS=1350 TARGET_X=1.35 BOX_X=0.95 BOX_MASS=8.0 BOX_COM_X=0.04 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.18 GAIT_FREQUENCY=1.15 ATTACH_AFTER_STEP=180 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_CLEARANCE=0.03 CARRY_Z_OFFSET=0.18 TARGET_HOLD_RADIUS=0.06 TARGET_SLOW_RADIUS=0.18 TARGET_BODY_MARGIN=0.03 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=30.0 BODY_HEIGHT_MAX_Z_SPEED=1.20 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=30.0 CONTACT_PROXY_MAX_SPEED=2.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating.log --require-attach --require-contact-proxy --require-dynamic-contact-proxy --forbid-disjoint-warning --max-target-distance 0.061 --max-relative-error 0.08 --max-peak-relative-error 0.08 --max-contact-proxy-gap 0.08 --min-body-travel 0.45 --min-box-travel 0.30 --min-support-margin 0.10 --min-support-margin-after-attach 0.10 --min-stance-count 3 --min-target-hold-steps 80 --max-body-z-deviation 0.18 --expect-body-vertical-mode height-servo --expect-physical-support-mode none --expect-strategy low_front_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity --expect-carry-geometry-mode nonpenetrating`.
  Result: pass. Completed 1350/1350; attach step 340; target hold 97; body
  travel `0.49526 m`; box travel `0.34040 m`; final target distance
  `0.05999 m`; final/peak relative error `0.05155/0.05155 m`; final/peak
  contact-proxy gap `0.04926/0.04931 m`; min post-attach support margin
  `0.13407 m`; max body-z deviation `0.15341 m`; fall/drop events 0; no
  disjoint warning. Output:
  `experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating/`.
  Log:
  `logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag54_direct_isaac_nonpenetrating.log`.
- [x] Run chest-supported direct Isaac posture-diversity rerun:
  `20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete`.
  Command:
  `timeout 1000s env STAMP=20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete DEVICE=cpu STEPS=1600 TARGET_X=1.25 BOX_X=0.95 BOX_MASS=12.0 BOX_COM_X=0.04 ROBOT_MASS=44.0 ROBOT_HEIGHT=1.25 ARM_LENGTH=0.45 MAX_PAYLOAD=15.0 BASE_SPEED=0.18 GAIT_FREQUENCY=1.05 ATTACH_AFTER_STEP=180 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_CLEARANCE=0.03 CARRY_Z_OFFSET=0.12 TARGET_HOLD_RADIUS=0.06 TARGET_SLOW_RADIUS=0.18 TARGET_BODY_MARGIN=0.03 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=30.0 BODY_HEIGHT_MAX_Z_SPEED=1.20 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=30.0 CONTACT_PROXY_MAX_SPEED=2.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete.log --require-attach --require-contact-proxy --require-dynamic-contact-proxy --forbid-disjoint-warning --max-target-distance 0.061 --max-relative-error 0.08 --max-peak-relative-error 0.08 --max-contact-proxy-gap 0.08 --min-body-travel 0.35 --min-box-travel 0.20 --min-support-margin 0.10 --min-support-margin-after-attach 0.10 --min-stance-count 3 --min-target-hold-steps 80 --max-body-z-deviation 0.18 --expect-body-vertical-mode height-servo --expect-physical-support-mode none --expect-strategy chest_supported_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity --expect-carry-geometry-mode nonpenetrating`.
  Result: fail but useful. Completed 1600/1600; selected
  `chest_supported_creep`; attach step 350; fall/drop events 0; body travel
  `0.46327 m`; box travel `0.24708 m`; min post-attach support margin
  `0.13779 m`; max body-z deviation `0.15531 m`; no disjoint warning. Failed
  because target hold was 0, final target distance was `0.06386 m`,
  final/peak relative error `0.11752/0.11780 m`, and peak contact-proxy gap
  `0.11627 m`. Output:
  `experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete/`.
  Log:
  `logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag56_direct_isaac_chest_supported_complete.log`.
- [ ] Next direct Isaac scene step: keep the `diag54` low-front scaffold as the
  passing baseline, and tune the chest-supported contact closure/target phase
  without weakening safety gates. Candidate changes: increase chest/front-stop
  proxy authority for heavy loads, reduce target distance for chest-supported
  validation, or extend horizon before tightening target hold.
- [x] Expose dynamic contact-proxy mass/thickness knobs for staged-free-box
  runs without changing defaults. Added `PALM_PROXY_MASS`,
  `CHEST_PROXY_MASS`, `SHELF_PROXY_MASS`, `FRONT_STOP_PROXY_MASS`,
  `PALM_PROXY_THICKNESS`, `CHEST_PROXY_THICKNESS`, and
  `FRONT_STOP_PROXY_THICKNESS` to
  `scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`; the Python
  script now records these fields in the summary JSON.
- [x] Attempt stronger chest-supported proxy run:
  `20260705_core_world_staged_free_box_diag57_chest_supported_stronger_proxy`.
  Command:
  `timeout 1100s env STAMP=20260705_core_world_staged_free_box_diag57_chest_supported_stronger_proxy DEVICE=cpu STEPS=1800 TARGET_X=1.25 BOX_X=0.95 BOX_MASS=12.0 BOX_COM_X=0.04 ROBOT_MASS=44.0 ROBOT_HEIGHT=1.25 ARM_LENGTH=0.45 MAX_PAYLOAD=15.0 BASE_SPEED=0.18 GAIT_FREQUENCY=1.05 ATTACH_AFTER_STEP=180 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_CLEARANCE=0.03 CARRY_Z_OFFSET=0.12 TARGET_HOLD_RADIUS=0.06 TARGET_SLOW_RADIUS=0.18 TARGET_BODY_MARGIN=0.03 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=30.0 BODY_HEIGHT_MAX_Z_SPEED=1.20 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=45.0 CONTACT_PROXY_MAX_SPEED=3.0 PALM_PROXY_MASS=90.0 CHEST_PROXY_MASS=130.0 SHELF_PROXY_MASS=140.0 FRONT_STOP_PROXY_MASS=120.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: no rollout. In Slurm job `165621` on `server10`, the run stalled
  during SimulationApp startup after protobuf warnings and created only an
  empty output directory plus a 394-byte log.
  Log:
  `logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag57_chest_supported_stronger_proxy.log`.
- [x] Run short startup health smoke in the same allocation:
  `20260705_core_world_staged_free_box_diag57a_startup_health`.
  Result: no rollout; it stalled at the same SimulationApp startup point.
  Treat `diag57`/`diag57a` as server10/Kit startup negative results only.
- [ ] Re-run the stronger chest-supported proxy diagnostic in a fresh
  allocation that passes a short Isaac startup health smoke, then apply the
  same strict checker gates used for `diag56`.
- [x] Add root/shortcut evidence counters to staged-free-box summaries:
  `body_root_velocity_command_count`, `body_root_pose_write_count`,
  `box_pose_write_count`, and `box_velocity_command_count`. Added checker
  gates `--max-body-root-velocity-commands`,
  `--max-body-root-pose-writes`, and `--max-box-pose-writes`. These gates are
  required before any articulated walking claim: existing velocity-commanded
  body scaffolds must fail a no-root walking gate.
- [x] Add staged-free-box checker flag `--require-no-root-shortcut`. It
  requires `articulated_carrier_enabled=true`, body root velocity commands 0,
  body root pose writes 0, and box pose writes 0. Audit result: applying this
  gate to old passing scaffold `diag54` correctly fails because it is
  `support-proxy`, not an articulated carrier.
- [x] Attempt fresh startup health smoke on `server10`:
  `20260705_core_world_staged_free_box_diag58_startup_health` in Slurm job
  `165631`. Result: no rollout; same SimulationApp startup stall after
  protobuf warnings. The allocation was stopped and released.
- [x] Request startup health smoke allocation `165638` with `--exclude=server10`.
  Result: no rollout. It did not produce a usable shell within the wait window
  and was cancelled/released. No non-server10 Isaac result exists yet for the
  stronger proxy settings.
- [ ] Re-request a non-server10 Isaac allocation when GPU priority allows.
  First run the 40-step startup health smoke; only then run stronger
  chest-supported proxy diagnostics with the updated root-shortcut checker
  fields.
- [x] Request non-`server10` startup health allocation `165646` on `server36`.
  Run:
  `timeout 420s env STAMP=20260705_core_world_staged_free_box_diag60_startup_health_server36 DEVICE=cpu STEPS=40 TARGET_X=1.0 BOX_X=0.60 BOX_MASS=4.0 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.12 ATTACH_AFTER_STEP=20 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_Z_OFFSET=0.12 BODY_VERTICAL_MODE=height-servo ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=30.0 CONTACT_PROXY_MAX_SPEED=2.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: inconclusive. It showed only SimulationApp boot plus protobuf
  warnings during a short observation window and was interrupted before a
  full timeout-backed startup conclusion. Do not treat this as a confirmed
  Isaac stall or contact-control result.
- [ ] Re-run startup health with a real elapsed-time wait before interpreting
  startup status.
- [x] Run real elapsed-time startup health in non-`server10` allocation:
  `20260705_core_world_staged_free_box_diag61_startup_health_server46`, Slurm
  job `165651` on `server46`.
  Command:
  `timeout 420s env STAMP=20260705_core_world_staged_free_box_diag61_startup_health_server46 DEVICE=cpu STEPS=40 TARGET_X=1.0 BOX_X=0.60 BOX_MASS=4.0 ROBOT_MASS=48.0 ROBOT_HEIGHT=1.20 ARM_LENGTH=0.52 MAX_PAYLOAD=16.0 BASE_SPEED=0.12 ATTACH_AFTER_STEP=20 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_Z_OFFSET=0.12 BODY_VERTICAL_MODE=height-servo ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=30.0 CONTACT_PROXY_MAX_SPEED=2.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Result: pass as a startup/scene health smoke. It completed 40/40, entered
  Isaac, attached at step 21, wrote summary and CSV, and had fall/drop events
  0. Output:
  `experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag61_startup_health_server46/`.
  Log:
  `logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag61_startup_health_server46.log`.
- [x] Run stronger chest-supported direct Isaac scaffold:
  `20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46`.
  Command:
  `timeout 1200s env STAMP=20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46 DEVICE=cpu STEPS=1800 TARGET_X=1.25 BOX_X=0.95 BOX_MASS=12.0 BOX_COM_X=0.04 ROBOT_MASS=44.0 ROBOT_HEIGHT=1.25 ARM_LENGTH=0.45 MAX_PAYLOAD=15.0 BASE_SPEED=0.18 GAIT_FREQUENCY=1.05 ATTACH_AFTER_STEP=180 CARRY_GEOMETRY_MODE=nonpenetrating CARRY_CLEARANCE=0.03 CARRY_Z_OFFSET=0.12 TARGET_HOLD_RADIUS=0.06 TARGET_SLOW_RADIUS=0.18 TARGET_BODY_MARGIN=0.03 BODY_VERTICAL_MODE=height-servo BODY_HEIGHT_GAIN=30.0 BODY_HEIGHT_MAX_Z_SPEED=1.20 PHYSICAL_SUPPORT_MODE=none ATTACHMENT_MODE=dynamic-contact-proxy CARRIER_MODE=dynamic-velocity CONTACT_PROXY_GAIN=45.0 CONTACT_PROXY_MAX_SPEED=3.0 PALM_PROXY_MASS=90.0 CHEST_PROXY_MASS=130.0 SHELF_PROXY_MASS=140.0 FRONT_STOP_PROXY_MASS=120.0 bash scripts/isaac/run_core_world_simapp_staged_free_box_carry.sh`.
  Scaffold checker:
  `python3 scripts/isaac/check_staged_free_box_summary.py experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46/core_world_simapp_staged_free_box_carry_summary.json --log logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46.log --require-attach --require-contact-proxy --require-dynamic-contact-proxy --forbid-disjoint-warning --max-target-distance 0.061 --max-relative-error 0.08 --max-peak-relative-error 0.08 --max-contact-proxy-gap 0.08 --min-body-travel 0.35 --min-box-travel 0.20 --min-support-margin 0.10 --min-support-margin-after-attach 0.10 --min-stance-count 3 --min-target-hold-steps 80 --max-body-z-deviation 0.18 --expect-body-vertical-mode height-servo --expect-physical-support-mode none --expect-strategy chest_supported_creep --expect-attachment-mode dynamic-contact-proxy --expect-carrier-mode dynamic-velocity --expect-carry-geometry-mode nonpenetrating`.
  Result: pass under scaffold gate. Completed 1800/1800; selected
  `chest_supported_creep`; attached at step 421; target hold 335; body travel
  `0.37242 m`; box travel `0.25688 m`; final target distance `0.04367 m`;
  final/peak relative error `0.01475/0.05307 m`; final/peak proxy gap
  `0.01287/0.05447 m`; min post-attach support margin `0.13825 m`;
  max body-z deviation `0.14629 m`; fall/drop events 0; no disjoint warning.
  Output:
  `experiments/outputs/core_world_simapp_staged_free_box_carry/20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46/`.
  Log:
  `logs/core_world_simapp_staged_free_box_carry/core_world_simapp_staged_free_box_carry_20260705_core_world_staged_free_box_diag62_chest_supported_stronger_proxy_server46.log`.
- [x] Apply no-root shortcut checker to `diag62`.
  Result: expected fail, so do not report `diag62` as final robot walking
  evidence. Failures: no articulated carrier, `body_root_velocity_command_count=1800`,
  and `box_pose_write_count=1`.
- [ ] Replace the velocity-commanded support-proxy carrier with an articulated
  foot-contact carrier that can pass `--require-no-root-shortcut`.
- [x] Correct staged-free-box evidence semantics before continuing Isaac scene
  work: `articulated-foot-contact` is now recorded as requested but not
  enabled unless an actual articulated implementation exists, and the checker
  reports both fields. Also added root-shortcut counters to the quasi-static
  walker scaffold. This is not a rollout result.
- [ ] Build the next direct Isaac diagnostic around no-root articulated
  standing first: articulated carrier enabled, positive joint count,
  foot-contact drive enabled, root velocity writes 0, root pose writes 0,
  box/payload pose writes 0, fall/drop 0 over a short fixed-payload stand.
- [ ] After no-root stand passes, add slow support-foot locomotion with a fixed
  payload and require nonzero commanded travel without root writes. Only then
  re-enable staged free-box contact proxies.
- [ ] If `20260704_go2_callback_nopayload_diag1` passes, run the same callback
  path with `PAYLOAD_MODE=fixed_base PAYLOAD_MASS=2.0` and require nonzero
  robot and payload travel with fall/drop events 0. This would still be fixed
  payload carrying, not unknown free-object grasping.
- [ ] Run `WBC_MODE=walk` direct-scene smoke in compute allocation.
- [ ] Add basic contact/drop/fall metrics to the direct scene.
- [ ] Add box parameter sweep for mass, size, initial pose, and friction.
- [ ] Locate the official Arena entry command for `g1_locomanip_pnp` as a
  reference/baseline path, not as the active blocker.
- [x] Create a browser diagnostic visualization for approach, probing,
  posture adjustment, lift, and carry under
  `src/carrying_visualization/`.
- [x] Add a static five-panel storyboard SVG for the same diagnostic sequence.
- [x] Run lightweight static checks for the visualization code on the login
  node.
- [ ] Generate a compute-node screenshot or MP4 for the diagnostic
  visualization.
- [x] Prepare strict Isaac Lab-Arena G1 loco-manipulation execution scripts
  under `scripts/isaac/`.
- [x] Fetch official Arena pinned submodule code for IsaacLab and Isaac-GR00T.
- [ ] Confirm official GR00T checkpoint completeness.
- [ ] Run official Arena `galileo_g1_locomanip_pick_and_place` with GR00T
  closed-loop policy in compute allocation as a reference/baseline.
- [ ] Save real Isaac/Arena MP4 evidence for G1 carrying the brown box.
- [ ] Run the smallest official rollout/evaluation path in compute allocation.
- [ ] Save logs under `logs/` and report under `experiments/reports/`.
- [ ] Export or record MP4 evidence only; do not record AVI as active
  evidence.
- [ ] Identify where object mass, pose, friction, and dimensions are defined.
- [ ] Identify where robot morphology, torque limits, and contact geometry are
  defined.
- [ ] Identify existing success/failure metrics and missing carrying metrics.
- [ ] Write a reproduction report before modifying the task.
