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
