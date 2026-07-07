# Current Operator Status

Timestamp: 2026-07-07 09:20 CST.

This is a status snapshot only. It is not a carrying-success claim.

## Latest G1 Boxtilt Status

- `169465` / `g1_bxrollgated` completed on `server44` with Slurm state
  `FAILED`, exit `1:0`, elapsed `00:03:01`. Aggregate:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target_refine/20260707_g1_boxtilt_heavy_lateral_roll_target_refine_gated/boxtilt_heavy_lateral_roll_target_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. Gated lateral roll-target
  remained negative: stronger box-source roll targets collapsed early, and
  the best `avg_pos_g020_l030` case only reached a transient 137-step target
  window before late fall/drop.
- `169472` / `g1_bxavgshort` completed on `server36` with Slurm state
  `FAILED`, exit `1:0`, elapsed `00:00:54`. Aggregate:
  `experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/20260707_g1_boxtilt_avgpos_short_window_760/boxtilt_avgpos_short_window_summary.json`.
  Result: strict `fail`, but a useful current-progress clip candidate:
  `760/760` steps, fall/drop `0/0`, final robot/box target-directed travel
  `2.25514/2.25542 m`, target-window end streak `133`, and final-hold end
  streak `100`; failures are excessive tilt (`0.624/0.649 rad`) and large
  lateral error (`1.086/1.284 m`). Do not call it robust or completed
  carrying.
- `169476` / `g1_bxlathold` completed on `server36` with Slurm state
  `FAILED`, exit `1:0`, elapsed `00:02:41`. Aggregate:
  `experiments/outputs/core_world_g1_boxtilt_lateral_hold_refine/20260707_g1_boxtilt_lateral_hold_refine_760/boxtilt_lateral_hold_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. It runs
  `scripts/isaac/run_core_world_g1_boxtilt_lateral_hold_refine_suite.sh` to
  test whether small final-hold lateral correction or box-lateral correction
  can reduce the `169472` side drift without reintroducing falls/drops. It
  did reduce lateral error in some cases, but every case broke stability or
  target-window dwell: `190/175`, `323/310`, `220/65`, and `295/0`
  fall/drop counts respectively. This route should be stopped unless paired
  with a materially different balance controller.
- `169488` / `g1_bxshortviz` is queued/running through tmux
  `codex_g1_boxtilt_shortviz_0707`. It completed on `server43` with Slurm
  state `COMPLETED`, exit `0:0`, elapsed `00:01:49`. It runs
  `scripts/isaac/run_core_world_g1_boxtilt_short_window_record_and_fallback.sh`
  to rerun the `169472` 760-step short-window boxtilt case with replay CSV
  recording, then render GIF/MP4/poster fallback visuals to
  `experiments/visuals/g1_replay_showcase/20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback/`.
  This is visualization-only progress material, not a strict success.
- `169501` / `g1_bxvizfix2` completed on `server39` with Slurm state
  `COMPLETED`, exit `0:0`, elapsed `00:00:15`. It regenerated the same visual
  directory with correct boxtilt labels and aggregate checker status. The
  checked poster now says `G1 boxtilt short-window progress` and
  `strict checker: fail`. Primary files:
  `g1_boxtilt_short_window_progress_annotated.mp4`,
  `g1_boxtilt_short_window_progress.mp4`,
  `g1_boxtilt_short_window_progress.gif`, and
  `g1_boxtilt_short_window_progress_poster.png`.
- `169514` / `g1_bxfinstandc2` completed on `server36` with Slurm state
  `FAILED`, exit `1:0`, elapsed `00:02:53`. It was the CPU compute backup for
  `scripts/isaac/run_core_world_g1_boxtilt_final_stand_refine_suite.sh`.
  Aggregate:
  `experiments/outputs/core_world_g1_boxtilt_final_stand_refine/20260707_g1_boxtilt_final_stand_refine_760_cpu_backup2/boxtilt_final_stand_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. Final-stand blending preserved
  box drops at `0` but introduced late falls (`8-10` fall events) and worse
  tilt (`0.934-0.993 rad` robot, `0.889-0.948 rad` box). It should not be
  treated as the missing stabilizer for the heavy boxtilt branch.
- `169508` / `g1_bxfinstand` stayed pending on `gpu` and was cancelled after
  `169514` produced the diagnostic result.
- `169519` / `g1_bxgeomc` completed on `server26` through tmux
  `curiosity_g1_boxtilt_geom_cpu_0707`. It ran
  `scripts/isaac/run_core_world_g1_boxtilt_geometry_refine_suite.sh` after
  exposing `FREE_CRADLE_LOCAL_Y` in the low-cradle launcher. Aggregate:
  `experiments/outputs/core_world_g1_boxtilt_geometry_refine/20260707_g1_boxtilt_geometry_refine_760_cpu_backup/boxtilt_geometry_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. Negative/positive box-cradle Y
  offsets, wider lid/rails, and final-hold chest pad all failed. The least
  bad lateral-error case (`box_cradle_y_neg003`) still had `291` falls /
  `163` drops, while the other three cases had target-window streak `0`.
  These small passive contact-geometry edits should not be repeated as the
  active rescue path for the 0.75 kg boxtilt branch.

## Running Or Queued

- No Curiosity simulation/render job is currently running or queued, and no
  Curiosity simulation/render job is running on the login node.
- `169316` / `any_payload` completed on `server36` with no rollout and no
  summary. The policy-backed ANYmal payload wrapper failed during IsaacLab
  `gym.make` initialization with `Failed to get DOF velocities from backend`.
- `169317` / `any_nofab` completed on `server02` with the same failure after
  setting `DISABLE_FABRIC=1`: `Simulation view object is invalidated and
  cannot be used again to call getDofVelocities`, then `Failed to get DOF
  velocities from backend`. Stop the unchanged ANYmal IsaacLab/RSL-RL wrapper
  route; it is not walking or carrying evidence in this environment.
- `169319` / `g1_render` failed on `server39` in `00:00:23` with no frames
  and no MP4. The render summary reports missing capture extensions:
  `ModuleNotFoundError: No module named 'omni.replicator'` and
  `ModuleNotFoundError: No module named 'isaacsim.core.rendering_manager'`.
  This is a render-environment failure, not control evidence.
- `169324` / `g1_fallback` completed on `server36` in `00:00:11` with exit
  `0:0`. It generated a 1600x900 schematic GIF/poster with 83 frames from the
  verified G1 low-carry replay CSV to
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_policy_replay_fallback/`.
  This is visualization-only replay, not Isaac camera render or new control
  evidence.
- `169326` / `g1_vizmp4` failed immediately with exit `127:0` because system
  `ffmpeg` was unavailable. `169327` / `g1_iomp4` then completed on
  `server36` in `00:00:06` with exit `0:0` using `imageio_ffmpeg`, producing
  `g1_lowcarry_replay_fallback.mp4` and
  `g1_lowcarry_replay_fallback_annotated.mp4`.
- `169332` / `g1_probe_load` completed on `server39` in `00:02:20` with
  Slurm state `FAILED`, exit `1:0`, because the aggregate summary is strict
  `fail`. It used the new
  `scripts/isaac/run_core_world_g1_probe_selected_load_validation_suite.sh`
  to run front-bumper probe -> selector -> selected target-window validation
  at `0.25`, `0.50`, and `0.75 kg`. Aggregate:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Result: `1/3` passed. The selector chose `lowcarry` for all masses while
  ignoring hidden mass. `0.50 kg` passed; `0.25 kg` failed with `384` falls /
  `225` drops and severe over-travel; `0.75 kg` failed with `346` falls /
  `284` drops. Current probe/selector is not a useful unknown-load adaptation.
- `169334` / `g1_pr_audit` completed on `server36` in `00:00:01` with exit
  `0:0`. Report:
  `experiments/reports/2026-07-07_g1_probe_selected_load_feature_audit.json`.
  It found that the current probe stage is already unsafe for all three
  masses: every probe summary failed with `240` fall events and about
  `210-211` box-drop events, while the selector still chose `lowcarry`.
  Probe target-directed box travel was not a monotonic mass signal.
- Safe-probe implementation is now in place: `--probe-collision-window` and
  `--probe-end-step` keep the front probe pad collision-disabled except during
  a short configured window, and the selector can reject unsafe probes by
  fall/drop/tilt gates. Lightweight syntax and diff checks passed.
- `169335` / `g1_safe_probe` completed on `server39` in `00:02:17` with
  Slurm state `FAILED`, exit `1:0`, because the aggregate summary is strict
  `fail`. Aggregate:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_safe_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Result: `0/3` cases passed. The safe collision-window probe avoided the old
  unsafe-probe collapse but became non-informative: all three masses had
  `probe_active_steps=1`, `probe_box_target_directed_travel_m=0`, and selected
  `chestpad` without using hidden mass. `0.25 kg` and `0.50 kg` validation had
  fall/drop `0/0` but target-window end streak `0`; `0.75 kg` failed with
  `482` falls / `439` drops and negative final target-directed box travel.
  Next active-probe work must tune for a safe but nonzero interaction signal
  before rerunning full probe-selected validation.
- `169337` / `g1_pr_signal` completed on `server39` with Slurm exit `1:0`.
  It ran the short 0.50 kg safe-probe signal bracket, not a full carrying
  validation. Aggregate:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_safe_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
  Result: strict `fail`, `safe_signal_cases=[]`. Four cases (`base_end80`,
  `end120`, `x046_end120`, `x048_wide_end120`) all completed only `41/180`
  steps, hit `Exception: Failed to get root link transforms from backend`, had
  `probe_active_steps=1`, and had max probe target-directed box motion `0`.
  Fall/drop stayed `0/0`, but this is invalid as probe evidence because the
  rollout did not complete. Do not rerun `PROBE_COLLISION_WINDOW=1` unchanged.
- Probe selector safety has been hardened after `169337`: probe summaries with
  non-null `error` now fail selection, and the probe-selected load-validation
  suite requires `completed_steps >= PROBE_FREE_STEPS` by default before
  posture selection. This prevents step-41 failed probes from being treated as
  valid unknown-load probe inputs.
- `169338` / `g1_preprobe` completed on `server39` with exit `0:0`. It is a
  probe-only follow-up, not a full carrying validation. Summary:
  `experiments/outputs/core_world_g1_safe_probe_signal_bracket/20260707_g1_precontact_probe_signal_bracket_fresh/safe_probe_signal_bracket_summary.json`.
  Result: diagnostic `pass`. With `PROBE_COLLISION_WINDOW_MODE=0`, all four
  pre-authored always-colliding 0.50 kg probe cases completed `180/180` steps
  with `error=null`, fall/drop `0/0`, `probe_active_steps=80`, root/box
  rollout writes `0`, and nonzero max probe target-directed box motion:
  about `0.11805`, `0.15116`, `0.12182`, and `0.12966 m`. This validates a
  backend-stable probe signal route at 0.50 kg only; it is not unknown-load
  discrimination or carrying success.
- `169339` / `g1_pr_loadsig` completed on `server39` with exit `0:0`. It was
  a probe-only multiload check using the conservative `small_x042` precontact
  probe at 0.25, 0.50, and 0.75 kg. Summary:
  `experiments/outputs/core_world_g1_precontact_probe_multiload_signal/20260707_g1_precontact_probe_multiload_signal_fresh/precontact_probe_multiload_signal_summary.json`.
  Result: diagnostic `pass`, `3/3` cases passed. All completed `180/180`
  steps with `error=null`, fall/drop `0/0`, `probe_active_steps=80`, and
  root/box rollout writes `0`. Max probe target-directed box motion was
  `0.14978 m` at 0.25 kg, `0.11805 m` at 0.50 kg, and `0.16064 m` at 0.75 kg.
  This gives a stable observed-interaction probe feature, but scalar motion is
  not monotonic in mass, so it is not a reliable mass estimator by itself.
- The G1 probe selector now has optional diagnostic risk gates for high probe
  target motion, robot tilt, box tilt, and relative-offset risk. These are
  heuristic gates, not learned system identification.
- `169340` / `g1_prepsel` completed on `server39`. It ran precontact
  probe-selected load validation at 0.25, 0.50, and 0.75 kg using the stable
  `small_x042` probe and heuristic thresholds
  `HIGH_PROBE_TRAVEL_THRESHOLD=0.14` and
  `PROBE_BOX_TILT_RISK_THRESHOLD=0.30`. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_precontact_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Result: strict `fail`, `1/3` cases passed. The selector ignored hidden
  mass and selected `chestpad` for `0.25 kg` and `0.75 kg`, and `lowcarry`
  for `0.50 kg`. The `0.50 kg` selected lowcarry validation passed with
  fall/drop `0/0`, final robot/box target-directed travel
  `2.29876/2.34645 m`, and target-window end streak `164`. The `0.25 kg`
  selected chestpad case was stable but under-traveled and had target-window
  streak `0`. The `0.75 kg` selected chestpad case failed badly with `482`
  falls / `439` drops, negative final target-directed travel, and max
  robot/box tilt about `1.314/1.312 rad`. This proves the probe/selector
  pipeline runs end to end without hidden mass, but not that unknown-load
  carrying is solved.
- `169346` / `g1_showviz` completed on `server39` with Slurm state
  `COMPLETED`, exit `0:0`, elapsed `00:00:45`. It was submitted through tmux
  `codex_g1_showcase_record_visual2_0707`. It uses
  `scripts/isaac/run_core_world_g1_current_showcase_record_and_fallback.sh`
  to record a fresh replay CSV from the current narrow 0.50 kg passing
  lowcarry configuration, then render a GIF/poster/MP4 presentation fallback
  on the compute node. Source rollout:
  `experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_lowcarry_current_pass_replay_record/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json`.
  It passed the narrow diagnostic again: `819/819` steps, fall/drop `0/0`,
  replay CSV recorded, final robot/box target-directed travel
  `2.29876/2.34645 m`, and target-window end streak `164`. Output directory:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_presentation_fallback/`.
  It contains `83` frames, `g1_lowcarry_replay_fallback.gif`,
  `g1_lowcarry_replay_fallback.mp4`,
  `g1_lowcarry_replay_fallback_annotated.mp4`, and
  `g1_lowcarry_replay_fallback_poster.png`. This is visualization-only
  replay evidence, not Isaac camera render and not new control evidence.
- `169350` / `g1_isarend` completed on `server39` with Slurm state `FAILED`,
  exit `1:0`, elapsed `00:00:19`. It attempted a true Isaac replay render
  from the fresh `169346` replay CSV to
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_current_pass_isaac_replay_render/`.
  Render summary status is `fail`, captured frames `0`. The environment is
  still missing both capture routes:
  `ModuleNotFoundError: No module named 'omni.replicator'` and
  `ModuleNotFoundError: No module named 'isaacsim.core.rendering_manager'`.
  Do not rerun this true Isaac replay-render path unchanged; the current
  usable presentation artifact is the fallback replay MP4/GIF/poster from
  `169346`.
- `169354` / `g1_boxtilt` completed on `server39` with Slurm state `FAILED`,
  exit `1:0`, elapsed `00:01:17`. It was submitted through tmux
  `codex_g1_boxtilt_load_probe_0707`. It runs the new
  `scripts/isaac/run_core_world_g1_boxtilt_load_probe_suite.sh` across
  `0.25`, `0.50`, and `0.75 kg` with `LARGERBOX_STRICT_MODE=boxtilt`.
  Summary:
  `experiments/outputs/core_world_g1_boxtilt_load_probe/20260707_g1_boxtilt_load_probe_fresh/boxtilt_load_probe_summary.json`.
  Result: strict `fail`, `0/3` cases passed. `0.25 kg` failed with
  `47` falls / `33` drops. `0.50 kg` failed with `329` falls / `0` drops
  and large lateral drift. `0.75 kg` was safer than chestpad: fall/drop
  `0/0`, max robot/box tilt `0.27226/0.29542 rad`, final relative offset
  `0.11864 m`, but it still under-traveled and drifted laterally, with
  target-window streak `0`. Interpretation: boxtilt is not a strict success,
  but it is a safer heavy-load fallback than the earlier chestpad selection.
- The probe selector was changed to a diagnostic three-branch heuristic:
  low-risk probes select `lowcarry`; high-motion probes without tilt/offset/
  size risk select `boxtilt`; resistant or tilt/offset/size-risk probes
  select `chestpad`.
- `169355` / `g1_3branch` completed on `server39` with Slurm state `FAILED`,
  exit `1:0`, elapsed `00:02:10`. It was submitted through tmux
  `codex_g1_threebranch_probe_selected_0707`. It reruns precontact
  probe-selected load validation at `0.25`, `0.50`, and `0.75 kg` with the
  three-branch selector. Summary:
  `experiments/outputs/core_world_g1_probe_selected_load_validation/20260707_g1_threebranch_probe_selected_load_validation_fresh/probe_selected_load_validation_summary.json`.
  Result: strict `fail`, `1/3` cases passed. `0.25 kg` selected `chestpad`,
  fall/drop `0/0`, final robot/box travel `0.46484/0.51690 m`,
  target-window streak `0`. `0.50 kg` selected `lowcarry` and passed with
  fall/drop `0/0`, final robot/box travel `2.29876/2.34645 m`, and
  target-window end streak `164`. `0.75 kg` selected `boxtilt`, fall/drop
  `0/0`, final robot/box travel `1.12217/1.07559 m`, target-window streak
  `0`, failing on lateral errors (`robot 0.81011 m`, box `0.70503 m`) and
  under-travel. This is safety progress versus the prior two-branch selector
  because the heavy case no longer catastrophically falls/drops, but it is not
  task success.
- `169366` / `g1_bxlat` completed on `server36` with Slurm state `FAILED`,
  exit `1:0`, elapsed `00:02:47`. It was submitted through tmux
  `codex_g1_boxtilt_heavy_lateral_0707`. It runs
  `scripts/isaac/run_core_world_g1_boxtilt_heavy_lateral_target_suite.sh` for
  the `0.75 kg` boxtilt branch. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_target/20260707_g1_boxtilt_heavy_lateral_target_fresh/boxtilt_heavy_lateral_target_summary.json`.
  Result: strict `fail`, `0/6` cases passed. Baseline kept fall/drop `0/0`
  but had target-window streak `0`. `hold_lat_off` and `box_lat_sign_neg`
  caused large falls/drops; `box_progress_lat` over-drove and failed with
  `297` falls / `76` drops. The useful variant is `hold_lat_reverse`, which
  kept fall/drop `0/0` and reached target-window stable/longest streak
  `152/152`, but failed to stop there and ended over-traveled with large
  lateral error. Next action is a stop/hold refinement on `hold_lat_reverse`.
- `169371` / `g1_bxstop` completed on `server39` with Slurm exit `1:0`.
  Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_stop_refine/20260707_g1_boxtilt_heavy_stop_refine_fresh/boxtilt_heavy_stop_refine_summary.json`.
  Result: strict `fail`, `0/4` cases passed. The best partial signal was
  `stop_165_180_finalzero`: fall/drop `0/0`, target-window stable/longest
  streak `184/184`, but end streak `0`, over-travel, high lateral error, and
  excessive tilt. This improves dwell only, not final stop/hold.
- `169303` / `g1_lowload` completed on `server36` in `00:01:55` with Slurm
  exit `1:0`. This is a strict negative load-robustness result for the current
  G1 AGILE low-carry front-tray setup. The `0.50 kg` case passed and
  reproduced the fresh 819-step low-carry result with fall/drop `0/0`, final
  robot/box target-directed travel `2.29876/2.34645 m`, and target-window
  end streak `164`. The `0.25 kg` case failed after entering the window with
  `384` falls / `225` drops; the `0.75 kg` case failed before final hold with
  `346` falls / `284` drops.
- `169304` / `g1_lowrepair` was submitted through tmux
  `codex_g1_lowrepair_0707` to run
  `scripts/isaac/run_core_world_g1_lowcarry_load_repair_suite.sh` with suite
  prefix `20260707_g1_lowcarry_load_repair_fresh`: `0.25 kg` final-window
  freeze, `0.25 kg` policy-then-stand hold, and `0.75 kg`
  chestpad/retention/slow carry.
- `169304` / `g1_lowrepair` completed on `server36` in `00:01:58` with Slurm
  exit `1:0`. Aggregate summary is strict `fail`, `0/3` cases passing.
  `0.25 kg` final-window freeze failed with `418` falls / `102` drops;
  `0.25 kg` policy-then-stand failed with `550` falls / `536` drops;
  `0.75 kg` chestpad/retention/slow failed with `930` falls / `856` drops and
  negative final box target-directed travel. Current scalar final-hold,
  freeze, stand-blend, chestpad, retention, and slow-speed tweaks are not
  sufficient.
- `169309` / `g1_massband` was submitted through tmux
  `codex_g1_massband_0707` to run
  `scripts/isaac/run_core_world_g1_lowcarry_mass_band_suite.sh` with
  suite prefix `20260707_g1_lowcarry_mass_band_fresh`, testing `0.35`,
  `0.40`, `0.45`, `0.55`, `0.60`, and `0.65 kg` around the verified
  `0.50 kg` pass.
- `169309` / `g1_massband` completed on `server36` in `00:03:54` with Slurm
  exit `1:0`. Aggregate summary is strict `fail`, `1/6` cases passing. The
  `0.35 kg` case passed with fall/drop `0/0`, target-window end streak `108`,
  max robot/box tilt `0.24340/0.41646 rad`, and rollout root/box writes
  `0/0/0`. `0.40 kg` failed by early lateral/roll fall with `398` falls;
  `0.45 kg` briefly reached the target window but late-failed with
  `87` falls / `60` drops; `0.55 kg` failed with `383` falls / `170` drops;
  `0.60 kg` was a near-miss with fall/drop `0/0` and target-window end streak
  `108`, but failed strict box-tilt gate at `0.63855 rad > 0.45`; `0.65 kg`
  failed with `414` falls / `154` drops.
- `169311` / `g1_edgerep` was submitted through tmux
  `codex_g1_edgerepair_0707` to run
  `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_suite.sh` with suite
  prefix `20260707_g1_lowcarry_edge_repair_fresh`, targeting the `0.60 kg`
  box-tilt near-miss using tighter lid/chestpad variants and the `0.45 kg`
  late fall/drop case using tighter lid plus final zero corrections.
- `169311` / `g1_edgerep` completed on `server36` in `00:02:32` with Slurm
  exit `1:0`. Aggregate summary is strict `fail`, `0/4` cases passing.
  The `0.60 kg` tight-lid, tight-lid-slow, and chestpad-hold variants all
  worsened the original near-miss and produced falls/drops. The `0.45 kg`
  tight-lid/final-zero case improved fall/drop to `0/0` with no rollout
  root/box writes, but under-traveled to about `1.52 m`, target-window streak
  stayed `0`, and max robot/box tilt `0.47062/0.48852 rad` still exceeded
  strict gates.
- `169312` / `g1_edgerep2` was submitted through tmux
  `codex_g1_edgerepair_v2_0707` to run
  `scripts/isaac/run_core_world_g1_lowcarry_edge_repair_v2_suite.sh` with
  suite prefix `20260707_g1_lowcarry_edge_repair_v2_fresh`, testing delayed
  final-hold thresholds for the `0.45 kg` partial improvement and non-pinching
  rail/no-lid geometry for the `0.60 kg` box-tilt near-miss.
- `169312` / `g1_edgerep2` completed on `server36` in `00:02:38` with Slurm
  exit `1:0`. Aggregate summary is strict `fail`, `0/4` cases passing.
  Delaying `0.45 kg` final-hold moved farther but crossed the stability
  boundary: final080 reached final robot/box target-directed travel
  `2.61657/2.21137 m` but failed with `306` falls / `118` drops; final100
  failed with `331` falls / `310` drops. For `0.60 kg`, side-rail-only failed
  with `273` falls / `246` drops and no-lid/tall-rails failed with
  `135` falls / `124` drops.
- Current stop decision: stop sweeping scalar final thresholds, tight/lower
  lids, chestpad, side rails, or no-lid geometry as the main G1 low-carry
  load-robustness route. The next step needs a materially different
  controller/backend or policy adaptation.
- `168997` / `g1_tw_arrest` completed on `server59` with Slurm exit `0:0`,
  but the comparison is strict `fail`.
- `169004` / `g1_box_pd` completed on `server43` with Slurm exit `0:0`. It
  ran the new box-progress closed-loop command controller for two 0.5 kg cases
  and wrote
  `experiments/reports/2026-07-07_g1_box_progress_controller_comparison_after_run.json`.
  The comparison is strict `fail`: `load05_box_progress_pd` had `158` falls
  and `23` drops, while `load05_box_progress_conservative` had `166` falls,
  `143` drops, and overran to about `5.166 m` final box target-directed
  travel. Box-progress command feedback alone is negative evidence.
- `169006` / `g1_box_ret` completed on `server43` with Slurm exit `0:0`. It
  ran one 0.5 kg case combining box-progress command control with
  box-retention posture feedback and wrote
  `experiments/reports/2026-07-07_g1_box_progress_retention_comparison_after_run.json`.
- The comparison is strict `fail`: `load05_box_progress_retention` had
  `0` box drops but `470` falls, final box target-directed travel about
  `0.247 m`, max robot/box tilt about `2.423/2.251 rad`, and no final hold.
  Retention posture feedback is negative for the current G1 wrapper route.
- `169008` / `prism_ref` failed on `server43` with exit `1:0`, but this is an
  invalid configuration failure, not physical evidence. The log showed
  `ENABLE_HORIZONTAL_LEGS=0`, then stopped at 260 steps with
  `RuntimeError: guarded_prelift_quasistatic_step_cycle requires --enable-horizontal-legs`.
  Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid/`.
- `curiosity_prismatic_reference_visual_watch_0707` is waiting in tmux. After
  `169008` leaves the queue and its summary/CSV exist, it will submit a
  compute-node `prism_viz` job to generate a schematic GIF/poster at
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid/`.
- `169015` / `prism_hist_viz` completed on `server36` with exit `0:0`. It
  generated
  an enhanced schematic GIF/poster from the already completed historical
  prismatic reference run
  `20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a`.
  This is intended as the fastest current presentation visual while the fresh
  GPU validation remains queued.
- `169019` / `prism_ref_cpu` completed on `server36` with Slurm state
  `FAILED`, exit `2:0`. It was a CPU compute-node fresh rerun of the
  prismatic reference. The rollout completed 760 steps with fall/drop `0/0`,
  but strict checker gates failed because post-settle payload travel reached
  only about `-0.0760 m` and final post-settle payload target distance was
  about `0.0940 m`. Treat it as a negative fresh-validation result.
- `169026` / `prism_ref_mcpu` completed the matched prismatic rollout on
  `server36`, but Slurm state is `FAILED`, exit `2:0`, because the old checker
  used an over-strict whole-trajectory relative-offset gate. The same summary
  passes the corrected checker at
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/reference_check_corrected.json`.
  Key corrected-pass metrics: 2880/2880 steps, fall/drop `0/0`, root/body/box
  writes all zero in rollout, active probe 80 steps without hidden ground
  truth, final post-settle payload travel `-0.17994 m`, target distance
  `0.00994 m`, max tilt `0.10644 rad`, and max post-settle relative offset
  `0.01160 m`.
- `prism_mviz` has been submitted through tmux
  `curiosity_prismatic_matched_visual_0707`; Slurm job `169027` completed on
  `server36` with exit `0:0` and generated a fresh matched GIF/poster from
  `169026` under
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/`.
- `169029` / `prism_suite` ran on `server36` and completed with Slurm exit
  `1:0` because the suite is strict `fail`. It ran four corrected-checker
  scaffold cases: nominal 10 kg mid carry, near-chest high 12 kg, long-reach
  low 8 kg, and bulky 10 kg. Suite summary:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_summary.json`.
  Three cases passed. The only failure is `near_chest_12kg_high`: fall/drop
  `0/0`, max tilt about `0.0995 rad`, max post-settle relative offset about
  `0.00894 m`, but post-settle payload travel was `0.147676 m`, below the
  `0.15 m` gate. This is an under-travel/early-stop failure, not a stability
  or drop failure.
- `169031` / `prism_nc_tight` completed on `server36`. It reran the
  failed near-chest high 12 kg posture with
  `GUARDED_STEP_TARGET_TOLERANCE=0.015`. Output:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_near_chest_12kg_high_tighttol/`.
  Slurm exit is `2:0` due to the launcher's older global relative-error gate,
  but rechecking with the corrected suite gate passed: fall/drop `0/0`,
  post-settle payload travel `0.18562 m`, target distance `0.01562 m`, and max
  tilt `0.09948 rad`.
- After-retry aggregate:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_after_retry_summary.json`
  is `status=pass` with 4/4 corrected-checker cases passing. This is still
  prismatic scaffold evidence only, not humanoid walking or learned carrying.
- MuJoCo assisted quadruped payload diagnostic has been submitted through tmux
  `curiosity_mujoco_quad_payload_assisted_0707`, job-name `mj_quad_payload`.
  It tests a more robot-like multi-joint quadruped with a 4 kg welded payload
  and explicit body-force stabilizer. Expected output:
  `experiments/outputs/mujoco_quadruped_payload/20260707_mujoco_quad_assisted_payload4kg/`.
  Result: strict `fail`; it traveled about `1.738 m` but had `94` fall events
  and max tilt about `3.159 rad`.
- Conservative MuJoCo assisted quadruped retry has been submitted through
  tmux `curiosity_mujoco_quad_payload_conservative_0707`, job-name
  `mj_quad_cons`, using target speed `0.20 m/s`, stronger z force, and
  stronger stabilizing torque. Expected output:
  `experiments/outputs/mujoco_quadruped_payload/20260707_mujoco_quad_assisted_payload4kg_conservative/`.
  Result: strict `fail`, but stable-undertravel rather than falling. It had
  fall events `0`, max tilt about `0.129 rad`, no root pose/velocity writes,
  and travel about `0.118 m`, below the `0.20 m` diagnostic gate.
- MuJoCo middle-speed run `169042` / `mj_quad_mid` completed on `server01`
  with Slurm state `FAILED` because the checker failed: it traveled about
  `1.161 m`, but had `31` fall events and max tilt about `3.270 rad`.
- MuJoCo bracket run `169044` / `mj_quad_bracket` completed on `server01`
  with Slurm exit `0:0`. It ran three welded-payload body-force-assisted
  quadruped diagnostics: `v022_fx130` passed with travel `0.323 m`, fall
  events `0`, max tilt `0.315 rad`; `v024_fx115` passed with travel
  `0.539 m`, fall events `0`, max tilt `0.470 rad`; `v026_fx105` failed with
  travel `0.637 m`, `18` fall events, and max tilt `0.954 rad`. This brackets
  the stability boundary around `0.24-0.26 m/s`. It is robot-like diagnostic
  evidence only: payload is welded and explicit stabilizer force/torque is
  used.
- MuJoCo robot-like visual `169047` / `mj_quad_viz2` completed on `server01`
  and generated:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback.gif`
  and
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback_poster.png`.
  This visual shows a dynamic quadruped body, legs, welded box, path trace,
  and metrics; it is still a schematic replay, not a simulator camera render.
- MuJoCo free-box contact route was added with
  `scripts/mujoco/run_quadruped_freebox_carry.py`,
  `scripts/mujoco/run_quadruped_freebox_carry.sh`, and
  `scripts/mujoco/check_quadruped_freebox_summary.py`. It uses a separate
  freejoint box in a torso-mounted tray, still with explicit body-force
  stabilization on the robot torso.
- MuJoCo free-box contact results are partial/negative. Permission-only
  failure `169077` is invalid physics evidence. Valid free-box rollouts:
  `169079` retained robot/box with fall/drop `0/0` but under-traveled
  (`max_box_travel_x_m=0.0716`, final `-0.0228`). `169081` bracket showed
  `2kg_v024` moved the free box about `0.1986 m` with fall `0`, but failed
  retention with `22` drop events and relative error `0.286 m`; faster cases
  lost the box and fell. Stop/hold follow-ups `169083`, `169087`, `169091`,
  `169092`, and `169096` latched near `0.15 m` and kept fall `0`, but all
  failed final retention/drop gates. Best final box travel was about
  `0.140 m`, but final relative error remained about `0.240 m` and drop
  events remained nonzero. This is progress beyond welded payload, but not a
  pass.
- MuJoCo free-box retention-force case `169100`
  (`20260707_mujoco_quad_freebox_2kg_v024_stop015_hold012_retention_spring`)
  passed the strict diagnostic checker. It used a 2 kg separate freejoint box,
  torso tray geometry, explicit body-force torso stabilization, and
  `RETENTION_FORCE_MODE=relative_spring` with equal and opposite forces
  applied between torso and box. Metrics: 3000/3000 steps, fall/drop `0/0`,
  root pose/velocity writes `0/0`, box pose/velocity writes `0/0`, retention
  force writes `3000`, max/final box travel `0.18242/0.18239 m`, max/final
  box-torso relative error `0.08091/0.07865 m`, max tilt `0.28755 rad`, min
  box z `0.69617 m`, and target-stop hold `1441` steps. This is the strongest
  current robot-like free-box diagnostic, but not final success because both
  torso stabilization and retention force are hand-authored controllers.
- Retention-force load validation completed as Slurm job `169116`
  (`mj_free_rnogpu`) on `server26` in 7 seconds. It used the no-GPU compute
  path in tmux `curiosity_mujoco_quad_freebox_retention_loads_nogpu_0707`.
  The redundant GPU-allocating queue job `169114` was canceled. Results:
  1 kg, 2 kg, and 3 kg all passed the strict diagnostic checker. All completed
  3000/3000 steps with fall/drop `0/0`, root pose/velocity writes `0/0`, box
  pose/velocity writes `0/0`, external force/torque writes `3000/3000`,
  retention force writes `3000`, target-stop latched, and target-stop hold
  above 600 steps. Final box travel was `0.19116 m`, `0.18239 m`, and
  `0.18993 m`; final relative error was `0.05206 m`, `0.07865 m`, and
  `0.08790 m`; max tilt was `0.27486 rad`, `0.28755 rad`, and `0.31475 rad`.
- Assist-reduction bracket completed as Slurm job `169120` (`mj_free_ared`)
  on `server01` in 8 seconds. The 2 kg retention-force case passed at 75%,
  50%, and about 33% body-force caps with essentially the same metrics:
  fall/drop `0/0`, final box travel about `0.181-0.182 m`, final relative
  error about `0.078 m`, max tilt about `0.288 rad`, and no root/box pose or
  velocity writes. This does not prove that the stabilizer is removable; it
  shows the earlier cap levels were likely not binding.
- Assist-floor probe completed as Slurm job `169121` (`mj_free_afloor`) on
  `server01` in 6 seconds. All three boundary probes failed strict gates.
  The 10% cap case had `77` falls, `9` drops, max box travel only
  `0.00666 m`, final box travel `-0.22714 m`, max tilt `0.87946 rad`, min box
  z `0.51059 m`, and no target latch. Zero caps and `ASSIST_MODE=none` had
  the same qualitative failure: `129` falls, `124` drops, max box travel about
  `0.00574 m`, final box travel `-0.55868 m`, max tilt `1.60894 rad`, min box
  z `0.31277 m`, and no target latch.
- First foot-IK support replacement probe completed as Slurm job `169126`
  (`mj_free_footik`) on `server01` in 7 seconds. It added
  `LEG_DRIVE_MODE=foot_ik` with planar two-link IK for stance/swing foot
  targets and ran four 2 kg no-body-assist free-box probes. All failed strict
  carry gates. `slow_short` reached max box travel `0.16824 m` and latched
  target stop, but fell/dropped (`117` falls, `111` drops). `nominal` also
  fell/dropped and moved backward. `faster_long` and `high_clearance` were
  useful negatives: fall/drop `0/0`, max tilt about `0.24 rad`, low relative
  error, but final box travel was backward (`-1.26814 m` and `-1.06543 m`).
- Negative-stride foot-IK probe is queued as Slurm job `169127`
  (`mj_free_fikneg`) in tmux
  `curiosity_mujoco_quad_freebox_foot_ik_negstride_0707`. It tests whether
  the stable backward foot-IK gait can be converted into stable forward travel
  by reversing stride sign.
- Negative-stride status update: inline job `169127` completed but is invalid
  as named evidence because shell expansion produced timestamp-only default
  outputs. Corrected script
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_suite.sh` ran as
  Slurm job `169130` on `server01`; all cases failed. The useful case is
  `neg_high`: no body assist, target latch, max/final box travel
  `0.45486/0.37860 m`, but `111` falls, `107` drops, max tilt `1.71246 rad`,
  min box z `0.38892 m`, and max relative error `0.23669 m`.
- Target-latched stride scaling was added to foot-IK and tested by
  `scripts/mujoco/run_quadruped_freebox_foot_ik_negstride_stop_suite.sh` as
  Slurm job `169135`; all stop/hold cases still failed. They kept positive
  final travel (`0.268-0.356 m`) but still had about `109-111` falls and
  `106-107` drops. First fall appears shortly after target latch around step
  `800`, so stopping at 0.15 m is still too late or too abrupt.
- Early-stop foot-IK suite is queued as Slurm job `169136`
  (`mj_free_fikearly`) in tmux
  `curiosity_mujoco_quad_freebox_foot_ik_early_stop_0707`, testing stop
  thresholds 0.08/0.10/0.12 m.
- Early-stop result: Slurm job `169136` completed on `server01`; all cases
  failed. Main cases preserved positive final travel around `0.300-0.311 m`
  and latched early, but still had about `110-111` falls and `107` drops.
- Lateral-retention result: Slurm job `169138` completed on `server01`; all
  y-axis equal-opposite retention cases failed. Final travel remained positive
  around `0.311-0.356 m`, but falls/drops remained `111/107`.
- Coarse roll-foot feedback result: Slurm job `169145` completed on
  `server01`; all cases failed strict carry gates. Positive gains kept some
  forward travel but fell/dropped. Negative gains stabilized better but drove
  backward; `roll_neg006` had fall/drop `0/0`, max tilt `0.25466 rad`, and
  min box z `0.69162 m`, but final box travel was `-1.01914 m`.
- Fine roll-feedback sweep is queued as Slurm job `169150`
  (`mj_free_fikroll2`) in tmux
  `curiosity_mujoco_quad_freebox_foot_ik_roll_feedback_fine_0707`.
- Fine roll-feedback result: Slurm job `169150` completed on `server01`; all
  cases failed. `roll_neg004` is the best forward/stability compromise so far:
  final box travel `0.52200 m`, target latch, max tilt `0.87115 rad`, but
  still `71` falls and `70` drops. Gains from `-0.045` through `-0.055` had
  fall/drop `0/0` and max tilt around `0.255-0.258 rad`, but walked backward
  with final travel around `-0.893` to `-1.005 m`.
- Lateral hip DOF was added to the MuJoCo quadruped via `*_hip_roll` joints
  and actuators, with `HIP_ROLL_BASE` / `HIP_ROLL_FEEDBACK_GAIN` controls.
  Slurm job `169159` (`mj_free_fikhip`) tested lateral hip support. All cases
  failed. Small base roll `0.05` retained positive final travel `0.19946 m`
  but still had `101` falls and `100` drops; base roll `0.10` and feedback
  variants had fall/drop `0/0` but walked backward around `-1.09` to
  `-1.12 m`.
- Slurm job `169161` (`mj_free_fikhips`) tested lateral hip stride variants.
  All failed. Best positive travel case `base006_neg12` had final travel
  `0.34346 m` but still `95` falls and `91` drops. Stable cases walked
  backward.
- Target-latched hold-brace controls were added:
  `HOLD_STANCE_FOOT_Z_DOWN` and `HOLD_HIP_ROLL_BASE`. Slurm job `169162`
  (`mj_free_fikbrace`) tested braced hold postures. All failed; bracing kept
  positive travel in several cases but still had roughly `95-114` falls and
  `89-109` drops.
- Closed-loop foot-placement controls were added:
  `CLOSED_LOOP_FOOT_PLACEMENT`, `STRIDE_VELOCITY_GAIN`,
  `STRIDE_POSITION_GAIN`, and `STRIDE_CLIP`. Slurm job `169164`
  (`mj_free_fikcl`) completed on `server36`; all five cases failed. No-hip
  cases latched the target and held for about `2183-2206` steps with positive
  final travel around `0.135-0.148 m`, but collapsed in the hold phase
  (`108-109` falls, `105-106` drops, max tilt about `1.60 rad`, min box z
  about `0.32-0.33 m`). Hip-base cases had fall/drop `0/0` and low tilt but
  walked backward with final box travel around `-1.04` to `-1.30 m`.
- Target-latched static support placement was added after `169164`:
  `HOLD_FRONT_FOOT_X`, `HOLD_REAR_FOOT_X`, and
  `HOLD_PITCH_FOOT_X_GAIN`. Slurm job `169173` (`mj_free_fikhold`) completed
  on `server30`; all cases failed. Wider fore-aft hold support kept no-hip
  final travel positive around `0.150-0.156 m` but still had `109` falls and
  `105` drops. Small hip-base `0.03` reached final travel `0.18645 m` but
  still had `99` falls and `96` drops; hip-base `0.05` reduced falls/drops to
  `48/43` but walked backward and never latched. CSV inspection shows roll
  dominates the collapse and box/torso y drift approaches about `-0.95 m`.
- Lateral-centering hold suite has been submitted through tmux
  `curiosity_mujoco_quad_freebox_centered_hold_0707` as Slurm job `169181`
  (`mj_free_fikcent`) and completed on `server30`. All cases failed. y-axis
  equal-and-opposite retention did not remove the roll/drop failure. No-hip
  cases still had about `110` falls and `105` drops. The best forward-travel
  hip-base case was `centered_hip004_y180`: final box travel `0.19379 m`,
  but `96` falls and `93` drops. Hip-base `0.05` reduced falls/drops to
  `85/80` but under-traveled at final box travel `0.08569 m`.
- Hip-roll feedback hold suite has been submitted through tmux
  `curiosity_mujoco_quad_freebox_hip_feedback_hold_0707` as Slurm job
  `169183` (`mj_free_fikfb`) and completed on `server30`. All cases failed.
  Negative hip-roll feedback kept forward travel but still fell/dropped:
  best final travel was `0.20149 m` with `105` falls and `102` drops.
  Positive feedback could stabilize but walked backward; `hip005_fbpos025`
  had fall/drop `0/0` but final box travel `-1.08736 m` and no target latch.
- Strong joint-servo support test was added with `ACTUATOR_KP` and
  `ACTUATOR_KV` controls in the MuJoCo free-box runner. It has been submitted
  through tmux `curiosity_mujoco_quad_freebox_strong_servo_0707` as Slurm job
  `169189` (`mj_free_fikservo`) and completed on `server26`. All cases
  failed. Hip-base `0.03` with stronger servos was stable fall/drop `0/0`,
  but walked backward with final box travel around `-1.40 m` and never
  latched the target. No-hip and hip-feedback cases either under-traveled or
  still fell/dropped. Current conclusion: stop small-parameter tuning of the
  hand-authored foot-IK family; replace the support controller itself while
  preserving the same free-box/no-root-write/no-box-write/no-body-force gates.
- First support-controller replacement route has been added:
  `SUPPORT_CONTROLLER_MODE=stance_force` in
  `scripts/mujoco/run_quadruped_freebox_carry.py`. It maps desired stance
  support/propulsion forces through foot Jacobians into actuated joint
  generalized torques, and records `support_joint_torque_write_count`.
  It does not write root/box pose or velocity and does not enable torso
  body-force assist. Slurm job `169200` (`mj_free_sf`) completed on
  `server01`; all cases failed. Positive force scale was ineffective or
  moved backward. Negative force scale produced real forward motion and early
  target latch: `sf_neg_nominal` final box travel `0.63974 m`, target hold
  `2761` steps; `sf_neg_hip003` final box travel `0.53802 m`, target hold
  `2761` steps. Both failed due to falls/drops and high box-torso relative
  error. This is negative evidence, but a better support-controller direction
  than the previous foot-IK tuning.
- Stance-force brake/early-stop follow-up has been submitted through tmux
  `curiosity_mujoco_quad_freebox_stance_force_brake_0707` as Slurm job
  `169208` (`mj_free_sfbrake`) and completed on `server01`. All cases failed.
  `brake_s004_hn006` was stable with fall/drop `0/0`, max tilt `0.31098 rad`,
  min box z `0.70184 m`, and final relative error `0.03184 m`, but walked
  backward with final box travel `-0.49488 m`. `brake_scale05_s006` retained
  positive final travel `0.28517 m` and final relative error `0.13416 m`, but
  still had `103` falls and `99` drops. The active stance-force problem is
  now finding the middle between stable-backward and positive-falling.
- Stance-force refine suite has been submitted through tmux
  `curiosity_mujoco_quad_freebox_stance_force_refine_0707` as Slurm job
  `169210` (`mj_free_sfref`). It sweeps smaller negative force scale, small
  hip-base, stronger vertical/roll/pitch support, and neutral or weak braking.

## Current Visual Status

- Best immediately usable presentation artifact:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/g1_lowcarry_replay_fallback.gif`.
- Poster:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/g1_lowcarry_replay_fallback_poster.png`.
- Summary:
  `experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/g1_replay_presentation_fallback_summary.json`.
- This is a schematic replay visualization from the passed `168632` CSV, not
  an Isaac camera render and not new control evidence.
- Real Isaac replay rendering remains blocked in this local install:
  `168900` failed with zero frames because `omni.replicator.core` could not
  resolve `omni.kit.pip_archive`, and the app-screenshot fallback could not
  resolve `omni.kit.viewport.window`.
- New prismatic-specific schematic renderer is ready:
  `scripts/isaac/render_prismatic_reference_presentation_fallback.py`. It will
  generate a clearer robot+box+target visual after `169008`, but it is still
  a schematic fallback, not an Isaac camera render.
- Enhanced historical prismatic visual completed as `169015`. Output:
  `experiments/visuals/prismatic_reference_showcase/20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a/`.
  It shows a side-view carrier body, four prismatic legs/feet, physical
  cradle, free box, target line, path trace, phase label, support-foot count,
  and strict non-success disclaimer.
- Immediate prismatic visual is now available. Manifest:
  `experiments/reports/2026-07-07_prismatic_showcase_visual_manifest.md`.
  GIF:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback.gif`.
  Poster:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback_poster.png`.
  One-page showcase:
  `slides/2026-07-07_isaac_carry_showcase.html`.
- Best current robot-like visual:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback.gif`.
- Poster:
  `experiments/visuals/mujoco_quadruped_payload/20260707_mujoco_quad_payload4kg_v024_fx115_visual/mujoco_quadruped_payload_fallback_poster.png`.
- Boundary: this MuJoCo visual is clearer than the prismatic block scaffold
  because it shows a multi-joint robot-like body and a payload, but it is not
  Isaac camera output and not physical free-box carrying because the payload
  is welded and body-force stabilization is explicit.

## Current Control Evidence

- Valid narrow source rollout: `168632` replay-record retry. It passed with
  fall/drop `0/0`, replay CSV present, and no rollout root pose/root velocity/
  box pose writes.
- Broad posture/load gauntlet `168850` failed strictly. All five cases failed.
  Four had large fall/drop counts; `boxtilt_diagnostic` had fall/drop `0/0`
  but failed target-window/final-hold progress.
- Terminal chest-pad contact and rescue are negative:
  `168802` and `168896` failed strict gates. Stop treating chest-pad
  geometry/timing tweaks as the main path.
- Non-pad balance rescue `168972` is also negative. Both cases reached the
  target vicinity but failed late with falls/drops and target-window end
  streak `0`.
- Late recovery `168995` is negative. `nopad_late_gentle_rescue` briefly
  entered the target window but overran and fell/dropped; `nopad_late_stand_blend`
  under-traveled and also fell/dropped.
- Target-window arrest `168997` is negative. Both 0.5 kg cases under-traveled
  and fell/dropped. `load05_window_zero_arrest` ended with box travel about
  `1.178 m`, `229` falls, and `18` drops. `load05_window_reverse_brake` ended
  with box travel about `1.269 m`, `255` falls, and `241` drops.
- New implementation after `168997`: optional box-progress / box-lateral
  closed-loop command controller. Result `169004` is negative: both new cases
  failed with falls/drops and no final hold.
- Additional implementation after `169004` submission: optional
  box-retention posture feedback. Result `169006` is negative: it removed box
  drops in the tested case but caused severe falls and failed travel/hold.
- Materially different baseline queued as `169008`: no-root prismatic
  articulated carrier, free 10 kg box in cradle, active probe, probe-adaptive
  gait/posture, strict no-root/no-box-write checker. This is not humanoid
  walking or final success, but it is a stronger physical carrying substrate
  than the failed G1 wrapper micro-tuning branches.
- Existing prismatic reference evidence available before the fresh rerun:
  `20260706_prismatic_cradle_probe_adaptive_posture_standard10_mid_retry24a`
  used `payload_mode=cradle_free_box`,
  `motion_mode=guarded_prelift_quasistatic_step_cycle`, a 10 kg free box,
  active probe for 80 steps with no hidden ground-truth load readout, fall/drop
  `0/0`, body root pose/velocity writes `0/0`, box pose writes `0`, final
  post-settle payload travel `-0.1758 m`, target distance `0.0058 m`, and
  max tilt `0.0973 rad`. This is a prismatic scaffold reference only, not
  humanoid walking, learned policy, or final box-carrying success.
- Fresh CPU rerun `169019` did not reproduce the historical pass under the
  shorter 760-step strict validation settings, and GPU rerun `169008` was
  invalid due to missing horizontal legs. Matched rerun `169026` produced a
  fresh physical scaffold rollout that passes the corrected checker, while
  retaining the non-success boundary: prismatic scaffold only, not humanoid
  walking or learned carrying.
- The near-chest high 12 kg retry fixed the early-stop under-travel under the
  corrected suite gate. The next implementation step should move beyond this
  scaffold toward a more robot-like articulated gait backend, because the full
  completion audit still fails on humanoid walking/learned carrying.
- The active MuJoCo bracket produced two pass diagnostics and one failure
  boundary; the best current robot-like point is `v024_fx115`, but it is only
  a backend candidate because the payload is welded and the stabilizer is
  explicit.
- The active free-box contact route has strict diagnostic passes after adding
  audited equal-opposite retention force: separate 1/2/3 kg freejoint boxes
  move about `0.18-0.19 m` and hold without fall/drop or root/box pose
  shortcuts. This is better evidence than welded payload and passive tray, but
  still not final robot carrying because the robot has explicit body-force
  stabilization and the grip retention is hand-authored.
- Reducing body-force caps down to about one third did not change the 2 kg
  outcome, but the 10%/zero/no-assist floor probe failed. The current support
  boundary is therefore clear: the free-box retention scaffold needs a
  nontrivial torso stabilization mechanism; it is not unassisted locomotion.
- The first no-body-assist foot-IK route is not a carry success yet, but it is
  a real support-controller step beyond body-force: two settings balanced and
  retained the box with no external stabilizer, only in the wrong travel
  direction. Reversing stride produced forward travel but not stable hold;
  falls/drops remain the blocker. Roll-to-foot-height feedback affects balance
  strongly but currently trades off against forward travel.
- The new lateral hip DOF confirms this is no longer just a missing joint
  problem. The controller is still open-loop enough that it finds either
  stable backward walking or forward motion followed by roll/fall/drop.
- First stance-force support replacement results are not a pass, but they are
  the active path after foot-IK. Slurm job `169200` showed negative
  force-scale stance support can create real forward motion and early target
  latch without root/box writes or torso body-force assist, but it overdrives
  into falls/drops and box relative error. `169208` and `169210` narrowed the
  boundary: scale about `-0.50` with hip-base `0.04-0.05` is stable fall/drop
  `0/0` but walks backward, while scale about `-0.35` can retain positive
  final travel but still falls/drops. The next run should search between
  `-0.38` and `-0.45`, not return to the exhausted foot-IK family.
- Stance-force boundary job `169216` completed on `server43` in 36 seconds
  and is a valid negative result. All eight cases used no root/box pose or
  velocity writes and no torso body-force assist, and all had fall/drop `0/0`
  with good box retention, but every case walked backward with final box
  travel around `-0.69` to `-0.75 m`, max positive travel only about
  `0.031-0.033 m`, and no target latch. The added damping/slowdown was too
  conservative; the next run should move back toward the `169210`
  positive-but-falling endpoint near scale `-0.35`.
- Stance-force edge job `169230` completed on `server59` in 19 seconds and is
  a valid negative result. It restored positive travel and target latch, but
  exposed the next blocker: post-latch hold collapse. Positive-travel cases
  reached final box travel `0.17284-0.27607 m` and target-hold `1396-1907`
  steps, but still had `62-81` falls and `57-76` drops. The stable case had
  fall/drop `0/0` but stopped too early and ended at final box travel
  `-0.19660 m`. Next work should add a post-latch hold stabilizer inside the
  stance-force controller.
- Hold-stabilizer job `169235` completed on `server59` in 16 seconds and is a
  valid negative result. New hold-only support parameters were added, but
  stop-0.05 positive-travel cases still collapsed after latch. The stable
  stop-0.04 case kept fall/drop `0/0`, max tilt `0.26166 rad`, and target
  hold `2748`, but drifted backward to final box travel `-0.20337 m`. The
  next narrow probe should use this stable early-stop setup with a small
  positive post-latch creep speed to preserve final travel.
- Hold-creep job `169236` completed on `server59` in 14 seconds and is a
  valid negative result. Positive post-latch speed with positive hold
  horizontal scale moved the stable stop-0.04 setup farther backward
  (`-1.07` to `-1.23 m` final box travel), while stop-0.045 restored positive
  travel but fell/dropped. The next narrow probe should use small positive
  hold speed with negative hold horizontal scale.
- Hold-creep negative-fx job `169242` completed on `server59` in 15 seconds
  and is a valid negative result. All cases were stable fall/drop `0/0`, but
  max positive travel stayed around `0.05 m` and final travel remained
  strongly negative (`-0.80` to `-0.93 m`). The stop-0.04 family is a stable
  backward basin, so the next useful probe should return to stop-0.05 and
  strengthen post-latch static support geometry.
- Wide-hold-support job `169247` completed on `server59` in 14 seconds and is
  a valid negative result. Wider static support did not fix the post-latch
  collapse: all cases kept positive final box travel `0.20196-0.27827 m`, but
  still had about `68-69` falls and `63` drops. This ends the current small
  sign/geometry sweep; the next controller needs stronger COM/centroidal
  balance feedback or a controller-backed legged policy.
- First COM-support job `169254` completed on `server59` in 17 seconds and is
  a valid negative result. COM-based foot-force redistribution was added and
  audited, but applying it during approach broke target latch for positive
  COM-x gains and walked backward. Negative COM-x restored one positive travel
  case but worsened fall/drop. Next probe should enable COM feedback only
  after target latch.
- Hold-only COM-support job `169263` completed on `server59` in 18 seconds
  and is a valid negative result. Pre-latch COM feedback was disabled, so
  approach and target latch were preserved, but all cases still collapsed
  during hold with `77-78` falls and `72-73` drops. Vertical COM force
  redistribution alone is not enough; next controller should add horizontal
  foot-force pitch/roll damping or a fuller centroidal wrench controller.
- Hold-only lateral foot-force job `169267` completed on `server59` in
  17 seconds and is a valid negative result. Lateral foot-force roll damping
  preserved positive travel and latch but did not reduce the post-latch
  collapse; falls/drops remained about `77-79` / `72-73`. The next useful
  probe should alter hold-only leg posture feedback, not just force
  redistribution.
- Hold-only posture-feedback job `169272` completed on `server59` in
  17 seconds and is a valid negative result, but it produced the clearest
  improvement in the current stance-force family. Negative hold hip-roll
  feedback reduced max tilt from the repeated `~3.24 rad` collapse to about
  `1.69-1.77 rad`, kept positive final travel, and reduced final relative
  error near `0.11 m`. It still failed with `77` falls and `73` drops, so it
  is not a pass. Next work should refine this negative hold hip-feedback
  branch.
- Hold-posture refine v1 job `169285` completed on `server30` in 21 seconds
  and is a valid negative result. All eight cases failed identically in a
  stable-backward/no-latch basin: final/max box travel was
  `-0.81105/0.03260 m`, target-stop latched was `false`, target-hold was `0`,
  fall/drop were `0/0`, max tilt was `0.24458 rad`, min box height was
  `0.71517 m`, and final/max relative error was `0.03911/0.07584 m`. Because
  this suite changed global actuator/support/retention settings relative to
  `169272`, it does not cleanly test the negative hold hip-feedback idea.
- Hold-posture refine v2 has been added as
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_posture_refine_v2_suite.sh`
  and submitted as Slurm job `169288` through tmux
  `codex_mj_refine_v2_0707`. It restores the `169272` global baseline and
  varies only post-latch posture parameters.
- Hold-posture refine v2 job `169288` completed on `server36` in 26 seconds
  and is a valid negative result. It preserved target latch and forward
  progress, but all ten cases still failed with `77` falls and `73-74` drops.
  Best tilt was `refine2_hip_neg030`: final/max box travel
  `0.26439/0.29245 m`, max tilt `1.68055 rad`, min box height `0.36177 m`,
  final/max relative error `0.11087/0.18279 m`, target-hold `1797`.
  Its CSV shows first fall at step `1480` and first drop at step `1560`;
  lateral drift/roll is the dominant failure, not target latch or x progress.
- Hold-lateral-posture combination suite has been added as
  `scripts/mujoco/run_quadruped_freebox_stance_force_hold_lateral_posture_suite.sh`
  and submitted as Slurm job `169291` through tmux `codex_mj_latpost_0707`.
  It tests whether hold-only lateral stance-force terms can suppress the
  box-y/roll collapse when combined with the best negative hold hip feedback.
- Hold-lateral-posture job `169291` completed on `server36` in 16 seconds and
  is a valid negative result. All eight cases failed. Lateral stance-force
  saturated (`64.46-120 N`) and kept latch/travel, but fall/drop stayed
  `77-78` / `73`. Best max-tilt case was `latpost_hip040_combo_pos` with
  final/max travel `0.28423/0.32303 m`, max tilt `1.65259 rad`, min box
  height `0.33075 m`, final/max relative error `0.10548/0.20650 m`, and
  target-hold `1797`.
- World-y hold correction has been added to the MuJoCo route and submitted as
  Slurm job `169292` through tmux `codex_mj_worldy_0707`. It is a direct test
  of whether suppressing global lateral drift through stance-foot Jacobian
  torques can prevent the roll/drop failure.
- World-y hold correction job `169292` completed on `server36` in 16 seconds
  and is a valid negative result. All eight cases failed. Lateral world-y
  force saturated (`120-160 N`), but `max_abs_box_y_m` remained
  `0.972-1.177 m`, fall/drop stayed `77-78` / `73-74`, and max tilt stayed
  `1.71-1.78 rad`. This is not the missing stabilizer.
- Hold-only support authority scales have been added and submitted as Slurm
  job `169293` through tmux `codex_mj_authority_0707`. This tests whether the
  post-latch collapse is caused by foot-force or joint-torque saturation.
- Hold-support authority job `169293` completed on `server36` in 16 seconds
  and is a valid negative result. Torque-only post-latch authority reproduced
  the `refine2_hip_neg030` baseline exactly; adding more foot force, height,
  and damping mostly worsened tilt or relative error. This is not just a
  force/torque cap issue.
- A centroidal stance-foot wrench distribution mode has been added and
  submitted as Slurm job `169294` through tmux `codex_mj_centroidal_0707`.
  It is the first materially different balance formulation in the current
  MuJoCo route.
- Centroidal support job `169294` completed on `server36` in 13 seconds and
  is a valid negative result. All six cases failed and were worse than the
  best heuristic stance-force baseline. Most did not latch; the only latched
  case, `centroid_negscale_authority`, still had `110` falls and `103` drops.
  The simplified MuJoCo hand-controller route should be treated as exhausted
  for credible fall/drop-free carrying unless a genuinely new controller or
  policy backend is introduced.
- Active route switched back to the G1 AGILE policy backend. The strongest
  historical low-carry run is
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`:
  819 steps, free box, G1 USD, AGILE ONNX policy, no rollout root/box pose
  writes, fall/drop `0/0`, final robot/box target-directed travel
  `2.29876/2.34645 m`, max robot/box tilt `0.20860/0.41361 rad`, and
  end-of-run target-window streak `164` steps. It remains diagnostic because
  it covers only one low-carry posture with engineered cradle/lid support.
- Fresh low-carry reproduction is pending as Slurm job `169302`
  (`g1_lowrepro`) through tmux `codex_g1_lowcarry_repro_0707`, suite prefix
  `20260707_g1_targetwindow_lowcarry_repro`.
- Fresh low-carry reproduction job `169302` completed on `server36` in
  `00:00:45` and passed. It reproduced the 819-step low-carry result with
  fall/drop `0/0`, final robot/box target-directed travel
  `2.29876/2.34645 m`, max robot/box tilt `0.20860/0.41361 rad`, min
  robot/box z `0.75211/0.80838 m`, target-window end streak `164`, and no
  rollout root pose/velocity or box pose writes.
- Low-carry load validation `169303` (`g1_lowload`) already completed with
  Slurm exit `1:0` and strict aggregate failure. Only the `0.50 kg` case
  passed; `0.25 kg` failed with `384` falls / `225` drops and `0.75 kg`
  failed with `346` falls / `284` drops. Do not describe the current
  low-carry front-tray setup as load robust.
- Boxtilt heavy stop-refine job `169371` (`g1_bxstop`) completed on
  `server39` with strict failure `0/4`. The useful case was
  `stop_165_180_finalzero`: it kept fall/drop `0/0` and increased
  target-window stable/longest streak to `184/184`, but it still ended
  outside the window with end streak `0`, over-travel
  `2.37121/2.41655 m`, lateral error `1.72462/1.81844 m`, and excessive
  tilt. This is a dwell improvement, not a stop/hold solution.
- Boxtilt heavy window-freeze job `169411` (`g1_bxfreeze`) completed on
  `server02` with Slurm exit `1:0`. Summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_window_freeze/20260707_g1_boxtilt_heavy_window_freeze/boxtilt_heavy_window_freeze_summary.json`.
  Result: strict `fail`, `0/4` cases passed. All freeze/brake variants
  reintroduced falls/drops (`105/92`, `110/26`, `77/49`, `109/95`). Best
  window dwell was `122` stable steps with end streak `0`, worse than
  `169371` `finalzero` (`184` stable steps, fall/drop `0/0`). Do not keep
  adding freeze/brake variants on this branch.
- Boxtilt heavy terminal-lateral suite has been submitted as Slurm job
  `169419` (`g1_bxtermlat`) through tmux `codex_g1_boxtilt_termlat_0707`.
  It tests terminal-only, thresholded lateral correction on the safer
  `169371` terminal/final hold setup, without freeze/brake. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_terminal_lateral/20260707_g1_boxtilt_heavy_terminal_lateral/boxtilt_heavy_terminal_lateral_summary.json`.
- Boxtilt heavy terminal-lateral job `169419` (`g1_bxtermlat`) completed on
  `server02` with Slurm exit `1:0`. Result: strict `fail`, `0/4` cases
  passed. All variants failed before terminal latch with `448` falls /
  `293` drops, final robot/box target-directed travel `0.59292/0.54745 m`,
  and target-window stable steps `0`. Conclusion: pre-terminal lateral
  correction is required for this branch; terminal-only correction is not a
  viable repair.
- Added dynamic lateral-roll balance target support to the G1 Core scene.
  It maps target-line lateral error from the robot, box, or their average
  into a bounded roll target for the existing ankle/hip balance-feedback
  controller. This changes joint targets only; it does not write root/box
  rollout state. The first diagnostic suite is Slurm job `169432`
  (`g1_bxrolltarget`) through tmux `codex_g1_boxtilt_rolltarget_0707`.
  Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target/20260707_g1_boxtilt_heavy_lateral_roll_target/boxtilt_heavy_lateral_roll_target_summary.json`.
- Boxtilt heavy lateral-roll-target job `169432` (`g1_bxrolltarget`)
  completed on `server02` with strict failure `0/4`. The useful information:
  `avg_sign_neg` preserved fall/drop `0/0` but drifted laterally
  `1.87642/2.11266 m` with target-window dwell `0`; `box_sign_neg` reduced
  final lateral error to `0.86138/0.94767 m` and reached 53 target-window
  steps, but failed with `162` falls / `127` drops. A low-gain refinement
  suite was first submitted as Slurm job `169446`, but that pending job was
  cancelled before rollout after adding hold-delay/ramp/tilt gates to the
  controller.
- Gated boxtilt heavy lateral-roll-target refine is now pending as Slurm job
  `169465` (`g1_bxrollgated`) through tmux
  `codex_g1_boxtilt_rollgated_0707`. Expected summary:
  `experiments/outputs/core_world_g1_boxtilt_heavy_lateral_roll_target_refine/20260707_g1_boxtilt_heavy_lateral_roll_target_refine_gated/boxtilt_heavy_lateral_roll_target_refine_summary.json`.
- Selected-branch horizon/hold repair job `169529` (`g1_branchhor`) completed
  on `server26` with strict failure `0/4`. Summary:
  `experiments/outputs/core_world_g1_selected_branch_horizon_repair/20260707_g1_selected_branch_horizon_repair_cpu/selected_branch_horizon_repair_summary.json`.
  Both `0.25 kg` chest-pad 1600-step variants collapsed late with `526/373`
  fall/drop events and no target-window dwell. `0.75 kg` boxtilt default
  failed with `257/168` fall/drop events. `0.75 kg` boxtilt stop/final-zero
  briefly reached the target window (`184` stable steps, `166` final-hold
  stable steps) but then failed final hold with `304/290` fall/drop events
  and end streak `0`. This proves the selected-branch failures are not just
  short-horizon artifacts.
- Completion audit remains `fail`. The full task is not achieved.

## Next Decision

- Next controller work should stay on stance-force / support-force control:
  narrow the stable-backward versus positive-falling boundary, then add better
  stop/hold damping. Do not keep tuning only open-loop foot-IK hip base,
  roll-height feedback, or body-force assist.
- For visuals, use the schematic GIF/poster only with the explicit label that
  it is not an Isaac camera render.
