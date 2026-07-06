#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 showcase capture on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

export SUITE_STAMP="${SUITE_STAMP:-20260706_showcase_g1_lowcarry_168398_rgb}"
export RUN_NOBOX=0 RUN_FIXED=0 RUN_FREE=1 STRICT=0 DEVICE=cpu
export COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-0}"
export TARGET_X=-1.2 TARGET_Y=0.0
export TARGET_WINDOW_CENTER=2.0 TARGET_WINDOW_HALFWIDTH=0.35

export FREE_STEPS="${SHOWCASE_FREE_STEPS:-819}"
export FREE_BOX_MASS=0.5
export FREE_BOX_SIZE_X=0.14 FREE_BOX_SIZE_Y=0.10 FREE_BOX_SIZE_Z=0.08
export FREE_BOX_POS_X=-0.18
export FREE_CRADLE_LOCAL_X=-0.18 FREE_CRADLE_LOCAL_Z=0.00
export LARGERBOX_STRICT_MODE=lowcarry

export AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=0.18
export AGILE_COMMAND_HOLD_SCALE=0.35
export AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65
export AGILE_COMMAND_HOLD_TERMINAL_LATCH=1
export AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015
export AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6
export AGILE_COMMAND_HOLD_FINAL_LATCH=1
export AGILE_COMMAND_HOLD_FINAL_SCALE=0.0

export AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=1
export AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE=0.10
export AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE=0.35
export AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START=0.14
export AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP=0.35
export AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START=2.0
export AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP=7.0
export AGILE_COMMAND_HOLD_ADAPTIVE_REL_START=0.10
export AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP=0.26
export AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING=0.25
export AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT=1
export AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START=0.12
export AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP=0.35
export AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_START=2.0
export AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_STOP=7.0

export AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1
export AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45
export AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006
export AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015
export AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0
export AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1
export AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1
export AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30
export AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35
export AGILE_COMMAND_HOLD_YAW_CORRECTION=1
export AGILE_COMMAND_HOLD_YAW_GAIN=0.0
export AGILE_COMMAND_HOLD_YAW_LIMIT=0.0
export AGILE_COMMAND_HOLD_YAW_SIGN=-1.0

export BALANCE_FEEDBACK_CONTROLLER=1
export BALANCE_FEEDBACK_BASE=command
export BALANCE_ADJUSTMENT_LIMIT=0.08
export BALANCE_PITCH_GAIN=0.10
export BALANCE_PITCH_RATE_GAIN=0.006
export BALANCE_PITCH_SIGN=1.0
export BALANCE_PITCH_ACTIVATION_THRESHOLD=0.05
export BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD=0.20
export BALANCE_ROLL_GAIN=0.06
export BALANCE_ROLL_RATE_GAIN=0.003
export BALANCE_ROLL_LEFT_ANKLE_SCALE=1.0
export BALANCE_ROLL_RIGHT_ANKLE_SCALE=-1.0
export BALANCE_ROLL_LEFT_HIP_SCALE=-0.5
export BALANCE_ROLL_RIGHT_HIP_SCALE=0.5
export BALANCE_ROLL_ACTIVATION_THRESHOLD=0.05
export BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD=0.20
export BALANCE_START_ON_AGILE_HOLD=1

export CRADLE_TOP_LID_ENABLED=1
export CRADLE_TOP_LID_ENABLE_ON_HOLD=1
export CRADLE_TOP_LID_LOCAL_Z=0.13
export CRADLE_TOP_LID_THICKNESS=0.014
export CRADLE_TOP_LID_X_SCALE=1.15
export CRADLE_TOP_LID_Y_SCALE=1.10
export CRADLE_SIDE_RAIL_HEIGHT=0.10
export CRADLE_END_STOP_HEIGHT=0.11
export CRADLE_CHEST_PAD_ENABLED=0
export CRADLE_CHEST_PAD_ENABLE_ON_HOLD=0

export FREE_MIN_ROBOT_TRAVEL=0.02 FREE_MIN_BOX_TRAVEL=0.02
export FREE_MAX_TILT=0.35 FREE_MAX_BOX_TILT=0.45
export FREE_MAX_ROBOT_LATERAL_ERROR=0.80
export FREE_MAX_BOX_LATERAL_ERROR=0.80
export FREE_MAX_FINAL_ROBOT_LATERAL_ERROR=0.60
export FREE_MAX_FINAL_BOX_LATERAL_ERROR=0.60
export FREE_MAX_FINAL_REL=0.25
export FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35
export FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35
export MIN_FINAL_HOLD_ROBOT_Z=0.45
export MIN_FINAL_HOLD_BOX_Z=0.45
export MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=399
export MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80
export MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50
export MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40
export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80
export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50
export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40

export CAPTURE_RGB="${SHOWCASE_CAPTURE_RGB:-1}"
export CAPTURE_RGB_EVERY_N_STEPS="${CAPTURE_RGB_EVERY_N_STEPS:-8}"
export CAPTURE_RGB_WIDTH="${CAPTURE_RGB_WIDTH:-1280}"
export CAPTURE_RGB_HEIGHT="${CAPTURE_RGB_HEIGHT:-720}"
export CAPTURE_RGB_RT_SUBFRAMES="${CAPTURE_RGB_RT_SUBFRAMES:-4}"
export CAPTURE_CAMERA_X="${CAPTURE_CAMERA_X:-1.8}"
export CAPTURE_CAMERA_Y="${CAPTURE_CAMERA_Y:--2.4}"
export CAPTURE_CAMERA_Z="${CAPTURE_CAMERA_Z:-1.25}"
export CAPTURE_LOOK_AT_X="${CAPTURE_LOOK_AT_X:--0.45}"
export CAPTURE_LOOK_AT_Y="${CAPTURE_LOOK_AT_Y:-0.0}"
export CAPTURE_LOOK_AT_Z="${CAPTURE_LOOK_AT_Z:-0.82}"
if [[ "${SHOWCASE_RECORD_REPLAY:-0}" == "1" ]]; then
  export RECORD_REPLAY_CSV=1
  export RECORD_REPLAY_EVERY_N_STEPS="${RECORD_REPLAY_EVERY_N_STEPS:-10}"
fi

scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh

case_dir="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}/agile_low_cradle_freebox_walk"
frame_dir="${case_dir}/rgb_frames"
movie_path="${case_dir}/showcase_g1_lowcarry.mp4"
annotated_movie_path="${case_dir}/showcase_g1_lowcarry_annotated.mp4"
if [[ "${CAPTURE_RGB}" != "1" ]]; then
  echo "[INFO] Capture disabled; replay trajectory, if requested, is under: ${case_dir}/core_world_g1_box_scene_replay.csv"
elif command -v ffmpeg >/dev/null 2>&1 && find "${frame_dir}" -type f -name '*.png' | grep -q .; then
  list_file="${case_dir}/showcase_frames.txt"
  find "${frame_dir}" -type f -name '*.png' | sort | awk '{print "file " q $0 q; print "duration 0.066"}' q="'" > "${list_file}"
  ffmpeg -y -hide_banner -loglevel warning -f concat -safe 0 -i "${list_file}" -pix_fmt yuv420p "${movie_path}"
  echo "[INFO] Showcase video written to: ${movie_path}"
  if ffmpeg -y -hide_banner -loglevel warning -i "${movie_path}" \
    -vf "drawbox=x=18:y=18:w=720:h=90:color=black@0.45:t=fill,drawtext=x=36:y=34:fontcolor=white:fontsize=28:text='G1 low-carry diagnostic pass',drawtext=x=36:y=68:fontcolor=white:fontsize=20:text='narrow setting, not generalized unknown-load success'" \
    -pix_fmt yuv420p "${annotated_movie_path}"; then
    echo "[INFO] Annotated showcase video written to: ${annotated_movie_path}"
  else
    echo "[WARN] Annotated video generation failed; raw video remains: ${movie_path}" >&2
  fi
else
  echo "[WARN] ffmpeg unavailable or no PNG frames found; RGB frames are under: ${frame_dir}"
fi
