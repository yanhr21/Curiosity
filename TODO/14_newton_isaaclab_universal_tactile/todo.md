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
