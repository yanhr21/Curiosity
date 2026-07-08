# G1 Carry Active Pipeline Status

Generated UTC: `2026-07-06T18:34:14.863884+00:00`

This report is a read-only status summary, not a success claim.

## Overall

- Pipeline status: `incomplete`
- Completion audit status: `fail`
- Failure classification status: `incomplete_or_failed`
- Next-action report status: `pass`

## Slurm Jobs

- `168972` `g1_balance_rescue` `PENDING` elapsed `0:00` start `2026-07-07T03:00:56` reason/node `(Priority)`
- `168801` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168802` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168849` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168850` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168851` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168882` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168883` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168895` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168896` `` `not_in_squeue` elapsed `` start `` reason/node ``
- `168900` `` `not_in_squeue` elapsed `` start `` reason/node ``

## Missing Artifacts

- `render_summary`
- `fallback_render_summary`
- `fallback_render_check`
- `fallback_abs_render_summary`
- `fallback_abs_render_check`
- `fallback_direct_render_summary`
- `contact_rescue_comparison`
- `contact_rescue_abs_comparison`
- `balance_rescue_comparison`

## Failing Artifacts

- `render_check`
- `fallback_direct_render_check`
- `fallback_ext_render_summary`
- `fallback_ext_render_check`
- `contact_summary`
- `contact_check`
- `contact_comparison_pending`
- `contact_comparison_after_168802`
- `contact_rescue_direct_comparison`
- `posture_gauntlet_summary`
- `completion_audit_current`
- `completion_audit_after_gauntlet`

## Completion Failures

- `terminal_contact: fail`
- `terminal_contact: terminal-contact case terminal_contact_pending status fail != pass`
- `terminal_contact: fall_events 104 > 0`
- `terminal_contact: min_robot_z_m 0.2364286184310913 < 0.45`
- `terminal_contact: max_tilt_rad 1.8426766960933354 > 0.35`
- `terminal_contact: max_box_tilt_rad 1.923876730713594 > 0.45`
- `terminal_contact: agile_command_hold_final_min_robot_z_m 0.2364286184310913 < 0.45`
- `terminal_contact: agile_command_hold_final_min_box_z_m 0.2299124002456665 < 0.45`
- `terminal_contact: final_box_robot_relative_offset_error_m 0.34432064098164455 > 0.25`
- `terminal_contact: max_abs_robot_target_lateral_error_m 0.9213434202745814 > 0.8`
- `terminal_contact: max_abs_box_target_lateral_error_m 1.1230842760065292 > 0.8`
- `terminal_contact: abs(final_robot_target_lateral_error_m) 0.8976693051886185 > 0.6`
- `terminal_contact: abs(final_box_target_lateral_error_m) 1.060917585886226 > 0.6`
- `terminal_contact: target_window_both_stable_steps 0 < 80`
- `terminal_contact: target_window_both_longest_streak_steps 0 < 50`
- `terminal_contact: target_window_both_streak_at_end_steps 0 < 40`
- `terminal_contact: target_window_both_final_hold_stable_steps 0 < 80`
- `terminal_contact: target_window_both_final_hold_longest_streak_steps 0 < 50`
- `terminal_contact: target_window_both_final_hold_streak_at_end_steps 0 < 40`
- `terminal_contact: agile_command_hold_final_active_steps 247 < 399`
- `terminal_contact: check_status fail != pass`
- `terminal_contact: fall_events 104 > 0`
- `terminal_contact: final_box_target_directed_travel_m 1.8879464016621244 < 2.0`
- `terminal_contact: final_relative_error_m 0.34432064098164455 > 0.25`
- `terminal_contact: final_hold_active_steps 247 < 399`
- `terminal_contact: target_window_end_streak 0 < 40`
- `terminal_contact: target_window_final_hold_end_streak 0 < 40`
- `posture_load_gauntlet: fail`
- `posture_load_gauntlet: gauntlet status fail != pass`
- `posture_load_gauntlet: gauntlet case failed: /public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_posture_gauntlet_after_contact_lowcarry_base/agile_low_cradle_freebox_walk`
- `posture_load_gauntlet: fall_events 402 > 0`
- `posture_load_gauntlet: box_drop_events 235 > 0`
- `posture_load_gauntlet: min_robot_z_m 0.18365252017974854 < 0.45`
- `posture_load_gauntlet: min_box_z_m 0.09988047182559967 < 0.2`
- `posture_load_gauntlet: max_tilt_rad 3.1359342685942813 > 0.35`
- `posture_load_gauntlet: max_box_tilt_rad 2.5042509283047365 > 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_robot_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_box_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_max_tilt_rad 999.0 > 0.35`
- `posture_load_gauntlet: agile_command_hold_final_max_box_tilt_rad 999.0 > 0.45`
- `posture_load_gauntlet: final_box_robot_relative_offset_error_m 0.47605088304516924 > 0.25`
- `posture_load_gauntlet: target_window_both_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: target_window_both_final_hold_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_final_hold_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_final_hold_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: agile_command_hold_final_active_steps 0 < 399`
- `posture_load_gauntlet: fall_events 402 > 0`
- `posture_load_gauntlet: box_drop_events 235 > 0`
- `posture_load_gauntlet: gauntlet case failed: /public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_posture_gauntlet_after_contact_chestpad_terminal/agile_low_cradle_freebox_walk`
- `posture_load_gauntlet: fall_events 343 > 0`
- `posture_load_gauntlet: box_drop_events 319 > 0`
- `posture_load_gauntlet: min_robot_z_m 0.11958088725805283 < 0.45`
- `posture_load_gauntlet: min_box_z_m 0.09807153791189194 < 0.2`
- `posture_load_gauntlet: max_tilt_rad 1.9233685539075036 > 0.35`
- `posture_load_gauntlet: max_box_tilt_rad 1.8063629451074918 > 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_robot_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_box_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_max_tilt_rad 999.0 > 0.35`
- `posture_load_gauntlet: agile_command_hold_final_max_box_tilt_rad 999.0 > 0.45`
- `posture_load_gauntlet: final_box_robot_relative_offset_error_m 0.3550549729700313 > 0.25`
- `posture_load_gauntlet: final_robot_target_directed_travel_m -0.09559770210183034 < 0.02`
- `posture_load_gauntlet: final_box_target_directed_travel_m -0.28225579857826233 < 0.02`
- `posture_load_gauntlet: abs(final_robot_target_lateral_error_m) 0.6383956687526111 > 0.6`
- `posture_load_gauntlet: target_window_both_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: target_window_both_final_hold_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_final_hold_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_final_hold_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: agile_command_hold_final_active_steps 0 < 399`
- `posture_load_gauntlet: fall_events 343 > 0`
- `posture_load_gauntlet: box_drop_events 319 > 0`
- `posture_load_gauntlet: gauntlet case failed: /public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_posture_gauntlet_after_contact_boxtilt_diagnostic/agile_low_cradle_freebox_walk`
- `posture_load_gauntlet: target_window_both_stable_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_longest_streak_steps 0 < 30`
- `posture_load_gauntlet: target_window_both_streak_at_end_steps 0 < 20`
- `posture_load_gauntlet: gauntlet case failed: /public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_posture_gauntlet_after_contact_lowcarry_lightbox/agile_low_cradle_freebox_walk`
- `posture_load_gauntlet: fall_events 520 > 0`
- `posture_load_gauntlet: box_drop_events 210 > 0`
- `posture_load_gauntlet: min_robot_z_m 0.12798045575618744 < 0.45`
- `posture_load_gauntlet: min_box_z_m 0.06924931704998016 < 0.2`
- `posture_load_gauntlet: max_tilt_rad 1.976862137683362 > 0.35`
- `posture_load_gauntlet: max_box_tilt_rad 2.8517940640758637 > 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_robot_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_box_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_max_tilt_rad 999.0 > 0.35`
- `posture_load_gauntlet: agile_command_hold_final_max_box_tilt_rad 999.0 > 0.45`
- `posture_load_gauntlet: final_box_robot_relative_offset_error_m 0.3679364306224896 > 0.25`
- `posture_load_gauntlet: final_box_target_directed_travel_m -0.11452754586935043 < 0.02`
- `posture_load_gauntlet: max_abs_robot_target_lateral_error_m 2.217674721443062 > 0.8`
- `posture_load_gauntlet: max_abs_box_target_lateral_error_m 2.2180097103118896 > 0.8`
- `posture_load_gauntlet: abs(final_robot_target_lateral_error_m) 1.9930101915804737 > 0.6`
- `posture_load_gauntlet: abs(final_box_target_lateral_error_m) 2.2180097103118896 > 0.6`
- `posture_load_gauntlet: target_window_both_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: target_window_both_final_hold_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_final_hold_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_final_hold_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: agile_command_hold_final_active_steps 0 < 399`
- `posture_load_gauntlet: fall_events 520 > 0`
- `posture_load_gauntlet: box_drop_events 210 > 0`
- `posture_load_gauntlet: gauntlet case failed: /public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_g1_posture_gauntlet_after_contact_lowcarry_heavybox/agile_low_cradle_freebox_walk`
- `posture_load_gauntlet: fall_events 506 > 0`
- `posture_load_gauntlet: box_drop_events 402 > 0`
- `posture_load_gauntlet: min_robot_z_m 0.17133642733097076 < 0.45`
- `posture_load_gauntlet: min_box_z_m 0.0830710232257843 < 0.2`
- `posture_load_gauntlet: max_tilt_rad 3.139767641922347 > 0.35`
- `posture_load_gauntlet: max_box_tilt_rad 3.139014503135704 > 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_robot_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_min_box_z_m None < 0.45`
- `posture_load_gauntlet: agile_command_hold_final_max_tilt_rad 999.0 > 0.35`
- `posture_load_gauntlet: agile_command_hold_final_max_box_tilt_rad 999.0 > 0.45`
- `posture_load_gauntlet: final_box_robot_relative_offset_error_m 0.26585385382433796 > 0.25`
- `posture_load_gauntlet: final_robot_target_directed_travel_m -0.4573382070602402 < 0.02`
- `posture_load_gauntlet: final_box_target_directed_travel_m -0.40845637023448944 < 0.02`
- `posture_load_gauntlet: max_abs_robot_target_lateral_error_m 0.8741873715656551 > 0.8`
- `posture_load_gauntlet: max_abs_box_target_lateral_error_m 0.8875020146369934 > 0.8`
- `posture_load_gauntlet: abs(final_robot_target_lateral_error_m) 0.6279710417970943 > 0.6`
- `posture_load_gauntlet: abs(final_box_target_lateral_error_m) 0.7425490617752075 > 0.6`
- `posture_load_gauntlet: target_window_both_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: target_window_both_final_hold_stable_steps 0 < 80`
- `posture_load_gauntlet: target_window_both_final_hold_longest_streak_steps 0 < 50`
- `posture_load_gauntlet: target_window_both_final_hold_streak_at_end_steps 0 < 40`
- `posture_load_gauntlet: agile_command_hold_final_active_steps 0 < 399`
- `posture_load_gauntlet: fall_events 506 > 0`
- `posture_load_gauntlet: box_drop_events 402 > 0`

## Failure Categories

- `queued`
- `log_present_no_known_error`
- `missing_dependency`
- `missing_artifact`
- `control_fall_or_drop`
- `control_fall_or_drop`
- `json_status_failure`

## Recommended Actions

- Priority `15`: `run_nopad_balance_rescue_followup`
  Reason: Terminal chest-pad rescue variants all failed; switch the next targeted diagnostic to non-pad final-window balance/freeze stabilization.
- Priority `35`: `continue_terminal_chestpad_stabilization`
  Reason: Chest-pad posture/contact case failed strict gates.
- Priority `40`: `stabilize_distinct_boxtilt_or_replace_with_valid_second_posture`
  Reason: A distinct non-lowcarry posture failed strict gates.
- Priority `50`: `add_load_adaptive_stop_hold_or_probe_conditioning`
  Reason: Held-out load case failed strict gates; low-carry is not load-robust.
- Priority `50`: `add_load_adaptive_stop_hold_or_probe_conditioning`
  Reason: Held-out load case failed strict gates; low-carry is not load-robust.
- Priority `60`: `inspect_failed_gauntlet_case`
  Reason: A gauntlet case failed and needs case-specific analysis.

## Source Reports

- `pipeline_status`: `experiments/reports/2026-07-07_g1_active_pipeline_status_after_watchers.json`
- `completion_audit`: `experiments/reports/2026-07-07_g1_carry_completion_audit_after_gauntlet.json`
- `failure_classification`: `experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_after_watchers.json`
- `next_actions`: `experiments/reports/2026-07-07_g1_next_carry_actions_after_audit.json`
