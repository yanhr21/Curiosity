#!/usr/bin/env bash
set -euo pipefail

# One bounded revision of the exact asymmetric support geometry. The left
# cooked side point is moved toward the robot to clear the wrist; the right
# cooked bottom target and all other zero-lift diagnostic settings are fixed.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_rear_lower_side_right_bottom_pose_gate_v2_20260807}
export PLAN10_LEFT_CONTACT_PCA_M="-0.18748614192008972 -0.11950920437785337 -0.07906173477774237"
export PLAN10_CONTACT_GEOMETRY_SOURCE="${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_rear_lower_side_right_bottom_geometry_contract_v2_20260807/derivation.json"

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_lower_side_right_bottom_pose_gate_v1.sh"
