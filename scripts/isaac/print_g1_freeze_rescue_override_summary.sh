#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUMMARY_PATH="${1:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_close_front_freeze_rescue_override/20260707_g1_lowcarry_close_front_freeze_rescue_override/close_front_freeze_rescue_override_summary.json}"

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "Missing summary: ${SUMMARY_PATH}" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse ${SUMMARY_PATH}" >&2
  exit 2
fi

echo "summary=${SUMMARY_PATH}"
echo -e "case\tstatus\tfall/drop\tfirst_fall/drop\ttarget_stable\tlongest/end\ttravel_robot/box\tlateral_robot/box\ttilt_robot/box\toverride\tovr_steps\tovr_first"
jq -r '
  .cases[]
  | select(type == "object" and has("fall_events"))
  | [
      (.case_dir | split("/")[-2] | sub("^20260707_g1_lowcarry_close_front_freeze_rescue_override_"; "")),
      (.check_status // "missing"),
      ((.fall_events|tostring) + "/" + (.box_drop_events|tostring)),
      (((.first_fall_step // "-")|tostring) + "/" + ((.first_box_drop_step // "-")|tostring)),
      (.target_window_both_stable_steps // 0),
      (((.target_window_both_longest_streak_steps // 0)|tostring) + "/" + ((.target_window_both_streak_at_end_steps // 0)|tostring)),
      (((.final_robot_target_directed_travel_m // 0)|tostring) + "/" + ((.final_box_target_directed_travel_m // 0)|tostring)),
      (((.final_robot_target_lateral_error_m // 0)|tostring) + "/" + ((.final_box_target_lateral_error_m // 0)|tostring)),
      (((.max_tilt_rad // 0)|tostring) + "/" + ((.max_box_tilt_rad // 0)|tostring)),
      (.agile_command_hold_rescue_overrides_final_freeze // false),
      (.agile_command_hold_rescue_override_freeze_active_steps // 0),
      (.agile_command_hold_rescue_override_freeze_first_active_step // "-")
    ]
  | @tsv
' "${SUMMARY_PATH}"
