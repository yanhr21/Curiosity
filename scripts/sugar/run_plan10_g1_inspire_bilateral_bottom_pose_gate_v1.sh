#!/usr/bin/env bash
set -euo pipefail

# No-learning bilateral load-bearing posture diagnostic. Both official Inspire
# palm local -X surfaces address hash-bound cooked points on the same real
# CarryBox bottom face, one on each side of the central setup pedestal. The
# inherited producer performs zero lift; contact and policy claims remain off.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_bilateral_bottom_pose_gate_v1_20260807}
export PLAN10_LEFT_CONTACT_PCA_M="-0.115420159657893 -0.0383276216757876 -0.164207544914922"
export PLAN10_LEFT_OUTWARD_PCA="0.00950507372068899 0.356552877370988 -0.934226792172026"
export PLAN10_LEFT_TILT_TANGENT_RAD=0.0
export PLAN10_LEFT_TILT_HEIGHT_RAD=0.0
export PLAN10_LEFT_NORMAL_ROLL_RAD=0.0
export PLAN10_LEFT_PALM_INSET_M=0.003
export PLAN10_RIGHT_PALM_INSET_M=0.003
export PLAN10_CONTACT_GEOMETRY_SOURCE="${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_bilateral_bottom_support_geometry_contract_v1_20260807/derivation.json"

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_true_bottom_side_pose_gate_v1.sh"
