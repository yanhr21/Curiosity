#!/usr/bin/env bash
set -euo pipefail

# No-learning mirror of the previous asymmetric posture gate. The left palm
# addresses an exact cooked bottom target and the right palm addresses an exact
# cooked lower-side target. Both points, normals, and the later tilt pivot are
# frozen in one hash-bound geometry contract before observing behavior.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_left_bottom_right_lower_side_pose_gate_v1_20260807}
export PLAN10_LEFT_CONTACT_PCA_M=${PLAN10_LEFT_CONTACT_PCA_M:-"-0.115420159657893 -0.0383276216757876 -0.164207544914922"}
export PLAN10_LEFT_OUTWARD_PCA=${PLAN10_LEFT_OUTWARD_PCA:-"0.00950507372068899 0.356552877370988 -0.934226792172026"}
export PLAN10_RIGHT_CONTACT_PCA_M=${PLAN10_RIGHT_CONTACT_PCA_M:-"0.21628780663013458 -0.07450920437785333 -0.07906173477774237"}
export PLAN10_RIGHT_OUTWARD_PCA=${PLAN10_RIGHT_OUTWARD_PCA:-"0.995533287525177 0.0850074291229248 0.0410749614238739"}
export PLAN10_CONTACT_GEOMETRY_SOURCE=${PLAN10_CONTACT_GEOMETRY_SOURCE:-"${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_left_bottom_right_lower_side_geometry_contract_v1_20260807/derivation.json"}
export PLAN10_LEFT_PALM_INSET_M=${PLAN10_LEFT_PALM_INSET_M:-0.003}
export PLAN10_RIGHT_PALM_INSET_M=${PLAN10_RIGHT_PALM_INSET_M:-0.003}

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_lower_side_right_bottom_pose_gate_v1.sh"
