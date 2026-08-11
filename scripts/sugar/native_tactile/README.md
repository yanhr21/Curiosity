# Native tactile: active code map

The active Plan-14 path is a no-training native tactile and slip interface for
IsaacLab and Newton. CarryBox visualization remains the IsaacLab foundation;
the nested `Newton/` clone contains the solved-contact backend. Existing policy
scripts reproduce historical diagnostics only. No numbered experiment ladder
is an active entry point.

## Active universal tactile and slip

- `universal.py`: common native frame plus direct official TacSL and Newton
  solved-contact adapters. Dense scalar layout is
  `[batch,patch,row,column]`; signed shear appends two channels. Released TacSL
  row/column order and signs are unchanged. The adapter explicitly converts
  IsaacLab's `wxyz` taxel orientations to the common Newton-compatible `xyzw`
  order and advances force and optical clocks independently.
- `slip.py`: causal tactile-history-only load, friction utilization,
  center-of-pressure motion, footprint transport, load loss and hysteretic
  `NO_CONTACT/STICK/INCIPIENT/GROSS` state.
- `evaluate_tactile_only_slip.py`: post-run comparison against simulator
  relative tangential velocity. That velocity is evaluation-only and never
  enters `slip.py`.
- `collect_sugar_whole_hand_carrybox.py`: official SUGAR G1 CarryBox collector
  using all 54 physical TacSL patches and the common adapter.
- `render_sugar_whole_hand_carrybox.py`: synchronized world plus both complete
  anatomical hands, raw signed force/shear and per-patch slip state.
- `run_isaaclab_r15_capsule_slip.py`: unchanged official R15 adapter on a
  controlled swept capsule with fixed, `0.006 m/s` incipient, `0.030 m/s`
  gross and return phases; simulator relative velocity remains a held-out
  label.
- `Newton/tactile_video.py`: public `newton.sensors.SensorTactile` box/pen
  evidence entry point; no monkeypatch, `kh * depth`, aggregate wrench or
  fabricated optical output. Its world panel is synchronized directly from
  Newton body state as top/side projections, while the lower panels retain the
  two spatial force/shear fields and tactile-only slip state.
- `Newton/tactile_slip_demo.py`: controlled Newton plate/cube sequence with
  stationary, slow-stick, incipient and gross-slip intervals. The detector
  reads only `SensorTactile`; actual relative tangential velocity is displayed
  and scored only as a held-out label.

## Reproduce Plan 14

Run these commands serially inside an existing retained GPU allocation. They
write only ignored files below `experiments/newton_universal_tactile/`.

Newton cube, pen, controlled slip, and tests:

```bash
export PYTHONPATH="$PWD:$PWD/Newton"
NEWTON_PY=/public/home/yanhongru/envs/tactile_genesis_snapshot_py312/bin/python

"$NEWTON_PY" Newton/tactile_video.py --scene cube --frames 600 \
  --normal-scale-n 5 --device cuda:0 \
  --output experiments/newton_universal_tactile/newton_cube/native_tactile.mp4
"$NEWTON_PY" Newton/tactile_video.py --scene pen --frames 600 \
  --normal-scale-n 5 --device cuda:0 \
  --output experiments/newton_universal_tactile/newton_pen/native_tactile.mp4
"$NEWTON_PY" Newton/tactile_slip_demo.py --frames 300 --device cuda:0 \
  --output experiments/newton_universal_tactile/newton_slip_control/native_tactile_slip.mp4
"$NEWTON_PY" -m unittest newton.tests.test_sensor_tactile \
  newton.tests.test_mujoco_solver.TestUpdateContactsPointPositions.test_contact_points_populated -v
"$NEWTON_PY" tests/native_tactile/test_newton_adapter.py -v
```

IsaacLab CarryBox collection and post-processing:

```bash
export PYTHONPATH="$PWD:$PWD/IsaacLab/source/isaaclab:$PWD/IsaacLab/source/isaaclab_assets:$PWD/IsaacLab/source/isaaclab_contrib:$PWD/SUGAR/source/sugar_rl"
ISAAC_PY=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
CARRY=experiments/newton_universal_tactile/isaaclab_carrybox_universal_current

"$ISAAC_PY" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$CARRY" --scenario successful_grasp --max-steps 660 \
  --headless --enable_cameras --device cuda:0
"$ISAAC_PY" scripts/sugar/native_tactile/evaluate_tactile_only_slip.py \
  --run-root "$CARRY" --output "$CARRY/slip_evaluation.json"
"$ISAAC_PY" scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  --run-root "$CARRY" --output "$CARRY/carrybox_native_tactile_slip.mp4" \
  --title "IsaacLab native TacSL | SUGAR G1 CarryBox | tactile-only slip" --fps 50
"$ISAAC_PY" scripts/sugar/native_tactile/render_sugar_force_kinematics_friction.py \
  --run-root "$CARRY" --output "$CARRY/carrybox_force_kinematics_friction.mp4" \
  --start-frame 230 --end-frame 520 --fps 50
```

IsaacLab official-R15 non-box controlled slip:

```bash
"$ISAAC_PY" scripts/sugar/native_tactile/run_isaaclab_r15_capsule_slip.py \
  --output-root experiments/newton_universal_tactile/isaaclab_r15_capsule_slip \
  --frames 240 --fps 50 --device cuda:0 --headless --enable_cameras
```

The Isaac Sim process can take time to shut down after writing a complete
trace and video. When interruption is required, terminate only the recorded
child process group; do not send a generic terminal `Ctrl+C` or exit the
retained allocation shell.

## Paused no-RGB Tracker-command training

- `run_native_tactile_bcppo_training.sh tracker_preflight_tactile ...`: live
  288-step one-update preflight from a shared base checkpoint.
- `run_native_tactile_bcppo_training.sh tracker_preflight_zero ...`: matched
  exact-zero/no-read preflight, run only after the live arm.
- `run_native_tactile_bcppo_training.sh tracker_tactile ...` and
  `tracker_zero`: official 24-step/update matched training arms.
- `summarize_tracker_command_preflights.py`: requires the `504-D` non-tactile
  actor, `324000-D` raw tactile tensor, `890-D` critic/teacher, real signal and
  encoder optimization in the live arm, and exact-zero/no-update behavior in
  the control.
- `run_native_tactile_bcppo_evaluation.sh tracker_tactile|tracker_zero ...`:
  deterministic frozen physical evaluation.

The `504-D` non-tactile actor input preserves the official Tracker's `35-D`
command and five-frame robot/action/gravity histories, then adds current base
linear velocity and phase. It excludes contact label and measured box pose.
Set `CURIOSITY_TRACKER_WARM_START_CHECKPOINT` to the released CarryBox
`tracker.pt` only for the common base initialization; use
`CURIOSITY_TRACKER_BASE_CHECKPOINT` to start both matched arms from that same
saved base. The released Tracker and Refiner use their original unnormalized
observation scale, so both actor and critic empirical normalization remain
disabled. The root [`README`](../../../README.md#paused-historical-504-d-tracker-command-experiment)
contains the exact common-base, frozen admission, serial preflight, training,
and evaluation commands.

## Complete CarryBox visualization

Run only this shell entry point inside an existing retained GPU allocation:

```bash
bash scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh \
  experiments/native_tactile_representation/reproduced_complete_carrybox \
  successful_grasp
```

It executes the following fixed sequence:

1. `collect_sugar_whole_hand_carrybox.py` records the official frozen SUGAR
   motion-45 CarryBox rollout and all native tactile/physical fields.
2. `render_sugar_whole_hand_carrybox.py` renders the synchronized world plus
   bilateral anatomical overview.
3. `render_sugar_whole_hand_supplement.py` renders left detail, right detail,
   and bilateral palm R15 RGB/depth.
4. `render_sugar_force_kinematics_friction.py` renders the native-clock force,
   friction, kinematics, and calibration audit.
5. `validate_complete_carrybox_bundle.sh` fully decodes every H.264 file and
   checks its frame count against the recorded source interval.

Shared tensor/layout code is in `representation.py`. Physical cross-checks are
implemented by `audit_sugar_whole_hand_carrybox.py` and
`audit_sugar_whole_hand_pair.py`. The exact output contract and current result
are documented in
`experiments/native_tactile_representation/whole_hand_carrybox_v3/REPRODUCE.md`.

## Historical `890-D` matched policy fusion

- `run_native_tactile_bcppo_preflight.sh`: serious official-width warm-start
  and adapter structural check.
- `run_native_tactile_bcppo_training.sh`: matched native-tactile or exact-zero
  training arm.
- `run_native_tactile_bcppo_evaluation.sh`: deterministic frozen evaluation.
- `compare_native_tactile_training_endpoints.py`: endpoint comparison; it does
  not by itself establish tactile usefulness.
- `summarize_native_tactile_frozen_pair.py`: validates that one frozen
  tactile/zero pair uses the same seed, reference, disabled events, and
  physical condition, then writes the direct metric differences.
- `summarize_native_tactile_dependence.py`: validates one checkpoint under
  live, evaluation-time zeroed, and fixed anatomical-patch-permuted tactile;
  it reports same-state action dependence and closed-loop trajectory changes.
- `run_frozen_tactile_policy_visualizations.sh`: one serial reproduction entry
  point for those three frozen rollouts, their synchronized world plus
  bilateral 27-patch videos, full-decode records, and matched summary.
- `summarize_native_tactile_authority_curve.py`: validates the predeclared
  `0/0.25/0.5/0.75/1.0` evaluation-only tactile-column response curve and
  reports common-horizon reward, tracking, lift, and action authority.
- `run_native_tactile_authority_curve.sh`: one serial retained-allocation
  entry point that evaluates all five authority scales and writes that checked
  summary.
- `audit_bounded_native_tactile_fusion.py`: CPU audit for exact official
  zero-tactile recovery, the `0.15` first-layer correction cap, frozen base
  gradients, live tactile-path gradients, and supported-sample distillation.
- `audit_action_residual_native_tactile_fusion.py`: CPU audit for the next
  direct `0.1` normalized-action tactile residual bound while retaining exact
  official behavior at zero tactile.
- `summarize_native_tactile_contact_teacher_alignment.py`: compares live,
  zeroed, and patch-permuted tactile against the same privileged teacher on
  the same physically supported states in a frozen rollout.
- `summarize_native_tactile_common_horizon.py`: compares reward, tracking, and
  lift over the shorter rollout's exact transition count so unequal
  termination lengths cannot reverse the interpretation.
- `compose_native_tactile_policy_pair.py`: scales two already completed policy
  videos to equal panels, ends at the shorter rollout, writes browser-compatible
  H.264, and fully decodes the result before reporting success.
- `run_native_tactile_teacher_residual_gate.sh`: collects the fixed
  train/selection/two-test contact split with the exact-zero actor and trains
  only the existing serious spatial adapter against the official teacher; it
  then runs the independent prediction/checkpoint audit.
- `audit_native_tactile_teacher_residual_gate.py`: independently reconstructs
  the selected checkpoint changes and every saved held-out prediction metric.
- `run_frozen_teacher_residual_policy_gate.sh`: evaluates that selected
  checkpoint with live versus exact-zero/no-read tactile on the two untouched
  physical conditions before any further PPO.
- `summarize_native_tactile_teacher_residual_policy_gate.py`: applies the
  predeclared common-horizon reward, tracking, and termination decision.
- `run_teacher_residual_policy_visualizations.sh`: serially records and renders
  live-versus-exact-zero behavior on both untouched physical conditions, with
  the world view and all 54 anatomical patch maps synchronized in H.264.
- `audit_canonical_trace_fusion.py` and `render_canonical_trace_fusion.py`:
  causal adapter inspection on the canonical CarryBox trace.
- `launch_retained_child.sh`: records and isolates a child process group while
  leaving the retained Slurm shell alive.

## Capability boundary

The tactile fields are read online and causally at the simulator physics clock.
They are not reconstructed from the saved world video. The current 54-patch
plus bilateral optical scene runs slower than wall-clock real time. IsaacLab
has separately demonstrated the same official R15 adapter on CarryBox and a
swept capsule; Newton uses its own solved-contact sensor on the Panda box/pen
scenes. Reusing either sensor requires compatible contact geometry and sensors
on the bodies that actually make contact. The sensor reports local contact; it
does not identify a task or infer a scene label.
