# TODO 14: Newton/IsaacLab Universal Native Tactile and Slip

## A. Reset and sources

- [x] Pause tactile-policy training without releasing retained GPU
  allocations.
- [x] Clone branch `mike_2026_7_21_newton` at `3d26173b` into `Newton/` and
  create feature branch `yanhongru/universal-tactile`.
- [x] Read the Newton sensor/contact/hydroelastic paths and the official
  IsaacLab TacSL data/force paths.
- [x] Freeze the backend-specific truth boundary in Plan 14 and AGENTS.md.

## B. Common contract

- [x] Implement the backend-neutral frame, patch geometry, availability and
  clock contract without object state or contact proxies.
- [x] Standardize world quaternions as `xyzw` by explicitly converting the
  official IsaacLab `wxyz` order, and maintain an independent optical clock.
- [x] Apply the same `xyzw` convention in every active force renderer and
  force-balance calculation.
- [x] Persist force and optical sequence/timestamp/dt fields directly in the
  scene traces rather than reconstructing them from frame indices.
- [x] Implement conservative raw-sample-to-grid serialization and verify
  signed vector-force conservation.
- [x] Add backend-neutral feature extraction used by visualization and slip
  detection.

## C. Newton native sensor

- [x] Add public `newton.sensors.SensorTactile` using solved native contact
  wrenches and native effective contact points/separation.
- [x] Support multiple patches, counterparts and worlds with fixed metric
  surface frames.
- [x] Support dynamic, kinematic and world-fixed Newton sensing shapes.
- [x] Expose raw contacts and dense signed normal/XY-shear/penetration fields
  with sequence and timestamp.
- [x] Add Newton `unittest` coverage for signs, shape-order symmetry,
  transforms, filtering, conservation and resets.
- [x] Replace the panda tactile demo's monkeypatch/aggregate-force path with
  the public sensor.

## D. IsaacLab official adapter

- [x] Adapt official `VisuoTactileSensorData` to the common frame without
  changing the TacSL equation or taxel order.
- [x] Preserve per-counterpart streams for scenes with multiple declared SDF
  contact objects and expose an explicit derived aggregate.
- [x] Preserve force and optical clocks and mark unavailable optical data
  instead of fabricating it.
- [x] Add focused adapter tests using real official sensor tensors.

## E. Causal slip detector

- [x] Implement continuous tactile-only slip evidence in metric units.
- [x] Implement hysteretic `NO_CONTACT/STICK/INCIPIENT/GROSS` state for both
  Torch/IsaacLab and Warp/Newton execution.
- [x] Build an IsaacLab official-R15 controlled stick-to-slide capsule case;
  use relative tangential velocity only as a held-out diagnostic label.
- [x] Build the corresponding controlled Newton stick-to-slide case.
- [x] Report IsaacLab precision/recall, false positives and measured onset
  delay against real incipient and above-threshold gross intervals.
- [x] Report the corresponding Newton delay, false positives and failures.

## F. Scene evidence

- [x] Complete a dynamically load-bearing Newton SUGAR G1 CarryBox grasp with
  synchronized world and both 27-patch tactile maps. The G1 motion is
  kinematic, the `0.3023376 kg` box is free under Newton gravity, and the
  source-`260...515` run lifts it `0.922 m` before set-down.
- [x] Run a Newton non-box scene with the unchanged sensor.
- [x] Run IsaacLab SUGAR G1 CarryBox with the common adapter.
- [x] Run an IsaacLab non-box swept-capsule scene with the unchanged official
  R15 adapter.
- [x] Produce separately playable H.264 scene files and corresponding concise
  numerical records for IsaacLab CarryBox/capsule and Newton dynamic G1,
  rigid Panda, soft Franka and controlled slip.
- [ ] Obtain explicit human review of all four videos.

## G. Reproduction and handoff

- [x] Document exact environment, sensor construction, update order, commands,
  outputs and claim boundaries for both engines.
- [x] Confirm no tactile policy training, object-state input, proxy force,
  fabricated optical data, experiment artifact, or large binary enters Git.
- [x] Commit and push the root documentation/adapters and the Newton feature
  branch separately.

## H. IsaacLab-only diverse tactile demos (2026-08-12)

- [x] Freeze the new-demo backend to IsaacLab; allow Newton repository assets
  only as imported assets, never as the simulator for these runs.
- [x] Complete a dual-official-R15, large-contact rigid cup/container pickup
  in IsaacLab with synchronized world and force/shear video; preserve optical
  availability explicitly. That force-only run disabled optical capture and
  therefore marks RGB/depth unavailable.
- [x] Reuse that exact IsaacLab path for a physical post-lift slip/drop caused
  by grip relaxation; do not zero or fabricate tactile data.
- [x] Complete at least one additional rigid-object shape with the same native
  tactile collector.
- [x] Start the official IsaacLab deformable-object runtime on the retained
  GPU as the soft-body baseline. This validates `SoftBodyView` execution only
  and is not counted as a tactile demo.
- [x] Resolve the official-TacSL/deformable compatibility boundary with a
  labeled project extension: preserve official R15 taxels, frames, data and
  TacSL force law while querying the current native `SoftBodyView`
  collision-tetrahedron surface. No rigid hidden core or rigid-contact proxy.
- [x] Complete a native IsaacLab deformable pickup and a matched physical
  post-lift drop. The success has `400/400` bilateral frames and a `0.1656 m`
  lift; the failure loses both tactile fields after physical support removal
  and falls `0.2734 m`.
- [x] Record direct reproduction commands and concise physical/tactile results
  for every retained success and failure video. No training is authorized.

These checked items are retained detached-fixture diagnostics and no longer
count as the active object-demo deliverable after the 2026-08-13 correction.

## I. Complete-G1 IsaacLab object demos (active, 2026-08-13)

- [x] Freeze the runtime to IsaacLab/PhysX and the robot to complete SUGAR G1;
  Newton is permitted only as an object-asset source.
- [x] Reuse the existing G1 CarryBox collector with exactly 27 physical TacSL
  patches per hand; reject detached R15 fixtures as completion evidence.
- [x] Add the official PickBottle `data_017`/released 510-D Tracker route, a
  bottle outer-shell SDF asset, object-swappable trace collection, and a
  renderer that consumes recorded full-G1/object/taxel states. No CarryBox
  action trace is accepted for this route.
- [ ] Restore a working retained H200 Kit/Vulkan runtime, then run the formal
  PickBottle collector with empty `DISPLAY`. H200 viability is already proven
  by complete CarryBox job `231928`; current starts fail before scene creation
  across `server13` and `server53`, so do not call H200 unsupported or
  misattribute the failure to PickBottle/TacSL. A separate clean Isaac Sim 5.1
  runtime is prepared; retained job `237668` is queued to run its final H200
  canary and then the formal collector.
- [ ] Complete one varied rigid-object pickup in which the full G1 physically
  moves the object and record the full-G1 plus bilateral 27-patch H.264.
- [ ] Complete a physical rigid-object failure with the identical G1 scene,
  collector and tactile contract.
- [ ] Complete another rigid shape using that unchanged complete-G1 path.
- [ ] Obtain human-visible review of the rigid success/failure videos.
- [ ] Only after the rigid cases pass, begin the complete-G1 soft-body case.
- [ ] Use the released official PickBottle Tracker and one exact official
  PickBottle motion for the non-box rigid case; never replay CarryBox actions
  on the bottle. Preserve the official 510-D Tracker input and 29-D output.
