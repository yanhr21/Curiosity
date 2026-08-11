# Native tactile: active code map

This directory has two active routes. CarryBox visualization is the completed
representation route; tactile-versus-zero training is the current policy route.
No numbered experiment ladder is an entry point.

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

## Matched policy fusion

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
- `audit_canonical_trace_fusion.py` and `render_canonical_trace_fusion.py`:
  causal adapter inspection on the canonical CarryBox trace.
- `launch_retained_child.sh`: records and isolates a child process group while
  leaving the retained Slurm shell alive.

## Capability boundary

The tactile fields are read online and causally at the simulator physics clock.
They are not reconstructed from the saved world video. The current 54-patch
plus bilateral optical scene runs slower than wall-clock real time, and the
validated object/task is CarryBox only. Reusing the sensor topology for another
object requires a compatible SDF and sensors on the bodies that actually make
contact. The sensor reports local contact; it does not identify the task or
infer that a scene is CarryBox or KickBox.
