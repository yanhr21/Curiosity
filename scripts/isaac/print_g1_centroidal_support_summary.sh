#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUMMARY_PATH="${1:-${ROOT_DIR}/experiments/outputs/core_world_g1_closefront_heldout_geometry_centroidal_support/20260707_g1_closefront_heldout_geometry_centroidal_support/closefront_heldout_geometry_centroidal_support_summary.json}"

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "Missing summary: ${SUMMARY_PATH}" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse ${SUMMARY_PATH}" >&2
  exit 2
fi

echo "summary=${SUMMARY_PATH}"
echo -e "case\tstatus\tfall/drop\tfirst_fall/drop\ttarget_stable\tlongest/end\ttravel_robot/box\tlateral_robot/box\ttilt_robot/box\tcentroidal_steps\tcentroidal_first\tlat_err\tpitch_adj\troll_adj\tmax_adj\twrites"
jq -r '
  .cases[]
  | select(type == "object" and has("fall_events"))
  | [
      (.case_dir | split("/")[-2] | sub("^20260707_g1_closefront_heldout_geometry_centroidal_support_"; "")),
      (.check_status // "missing"),
      ((.fall_events|tostring) + "/" + (.box_drop_events|tostring)),
      (((.first_fall_step // "-")|tostring) + "/" + ((.first_box_drop_step // "-")|tostring)),
      (.target_window_both_stable_steps // 0),
      (((.target_window_both_longest_streak_steps // 0)|tostring) + "/" + ((.target_window_both_streak_at_end_steps // 0)|tostring)),
      (((.final_robot_target_directed_travel_m // 0)|tostring) + "/" + ((.final_box_target_directed_travel_m // 0)|tostring)),
      (((.final_robot_target_lateral_error_m // 0)|tostring) + "/" + ((.final_box_target_lateral_error_m // 0)|tostring)),
      (((.max_tilt_rad // 0)|tostring) + "/" + ((.max_box_tilt_rad // 0)|tostring)),
      (.terminal_centroidal_support_active_steps // 0),
      (.terminal_centroidal_support_first_active_step // "-"),
      (.terminal_centroidal_support_last_lateral_error_m // 0),
      (.terminal_centroidal_support_last_pitch_adjust_rad // 0),
      (.terminal_centroidal_support_last_roll_adjust_rad // 0),
      (((.terminal_centroidal_support_max_abs_pitch_adjust_rad // 0)|tostring) + "/" + ((.terminal_centroidal_support_max_abs_roll_adjust_rad // 0)|tostring)),
      (((.root_pose_write_count_rollout // 0)|tostring) + "/" + ((.root_velocity_write_count_rollout // 0)|tostring) + "/" + ((.box_pose_write_count_rollout // 0)|tostring))
    ]
  | @tsv
' "${SUMMARY_PATH}"
