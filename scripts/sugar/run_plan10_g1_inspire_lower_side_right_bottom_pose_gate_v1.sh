#!/usr/bin/env bash
set -euo pipefail

# No-learning asymmetric load-bearing pose gate. Both target positions and
# outward normals are taken from hash-bound cooked CarryBox collision geometry:
# left palm on a true lower side face, right palm on the bottom face. The run
# requests zero lift and cannot establish grasp, tactile, policy, or success.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_lower_side_right_bottom_pose_gate_v1_20260807}
export PLAN10_LEFT_CONTACT_PCA_M=${PLAN10_LEFT_CONTACT_PCA_M:-"-0.19103866815567017 -0.07450920437785333 -0.07906173477774237"}
export PLAN10_LEFT_OUTWARD_PCA=${PLAN10_LEFT_OUTWARD_PCA:-"-0.9962605834007263 -0.07864987105131149 -0.03576560318470001"}
export PLAN10_RIGHT_CONTACT_PCA_M=${PLAN10_RIGHT_CONTACT_PCA_M:-"0.124540961176844 -0.0357127777286392 -0.160768136581671"}
export PLAN10_RIGHT_OUTWARD_PCA=${PLAN10_RIGHT_OUTWARD_PCA:-"0.00950507372068899 0.356552877370988 -0.934226792172026"}
export PLAN10_CONTACT_GEOMETRY_SOURCE=${PLAN10_CONTACT_GEOMETRY_SOURCE:-"${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_lower_side_right_bottom_geometry_contract_v1_20260807/derivation.json"}
export PLAN10_LEFT_TILT_TANGENT_RAD=0.0
export PLAN10_LEFT_TILT_HEIGHT_RAD=0.0
export PLAN10_LEFT_NORMAL_ROLL_RAD=0.0
export PLAN10_RIGHT_TILT_TANGENT_RAD=0.0
export PLAN10_RIGHT_TILT_HEIGHT_RAD=0.0
export PLAN10_RIGHT_NORMAL_ROLL_RAD=0.0
export PLAN10_LEFT_PALM_INSET_M=${PLAN10_LEFT_PALM_INSET_M:-0.003}
export PLAN10_RIGHT_PALM_INSET_M=${PLAN10_RIGHT_PALM_INSET_M:-0.003}
export PLAN10_LIFT_HEIGHT_M=0.0

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_true_bottom_side_pose_gate_v1.sh"
