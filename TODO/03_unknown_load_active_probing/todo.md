# TODO 03: Unknown Load And Active Probing

- [ ] Add or locate object randomization hooks for mass, CoM, dimensions, and
  friction.
- [ ] Add carry-specific termination: fall, drop, severe slip, contact loss,
  excessive torque, and unsafe object acceleration.
- [ ] Add metrics: distance, duration, torque/energy cost, peak torque,
  balance margin, slip, drop, contact loss, falls, recovery.
- [ ] Add probing phase actions without hardcoding one fixed sequence.
- [ ] Add belief/state logging for inferred load and contact stability.
- [ ] Run no-probing diagnostic baseline.
- [ ] Run scripted-probing diagnostic baseline.
- [ ] Run active-probing diagnostic policy.
- [x] Run two-stage probe-selected posture scaffold diagnostic on 2026-07-06:
  vertical micro-lift selected `close_mid`; horizontal push-pull selected
  `front_reach`; both selected carries passed strict 64 cm / 8 kg gates in
  Slurm job `167441`.
- [x] Rerun same-episode online probe-adaptive support scaffold diagnostic
  after fixing Slurm job `167449` launcher parse failure. This diagnostic
  adapts support step height, double-support fraction, stance x, and swing x
  from probe telemetry within the same episode; it does not yet switch hold
  geometry or claim learned control. Retry2 Slurm job `167452` applied the
  vertical-probe medium support profile successfully but stopped on an overly
  strict support-foot x-motion gate; rerun after lowering that suite gate to
  `0.25 m`. Retry3 Slurm job `167455` passed both vertical and horizontal
  probe cases.
- [ ] Add same-episode hold/contact geometry adaptation from probe telemetry:
  close/support box against torso for medium/high risk, allow farther reach for
  low risk, and record changed hold parameters in the same normalized summary.
  The first x-cradle attempt in Slurm job `167459` is a negative result because
  the rear pusher launched the box along x; continue with side-clamp lateral
  closure instead. Side-clamp retry1 Slurm job `167460` exposed a clamp pad
  geometry bug; rerun after fixing pad y-size to use `clamp_pad_thickness`.
  Side-clamp retry4/retry5 showed a repeated blocker: clamp closure targets
  are commanded (`0.054 m`) but measured clamp joint motion remains near zero,
  causing box drops. Do not keep repeating the same side-clamp route without a
  materially different actuation/contact design.
- [ ] Promote diagnostics to real training only after reporting command,
  allocation, GPU utilization evidence, and held-out protocol.
