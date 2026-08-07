#!/usr/bin/env bash
set -euo pipefail

# No-learning lower-edge hook-grasp pose gate. Both palms address exact cooked
# side-face points immediately above the bottom edge; subsequent finger closure
# may only be attempted if this zero-lift geometry is mechanically admissible.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_bilateral_lower_edge_side_hook_pose_gate_v1_20260807}
export PLAN10_LEFT_CONTACT_PCA_M=${PLAN10_LEFT_CONTACT_PCA_M:-"-0.1888846904039383 -0.07450920437785333 -0.13906173477774242"}
export PLAN10_LEFT_OUTWARD_PCA=${PLAN10_LEFT_OUTWARD_PCA:-"-0.9962605834007263 -0.07864987105131149 -0.03576560318470001"}
export PLAN10_RIGHT_CONTACT_PCA_M=${PLAN10_RIGHT_CONTACT_PCA_M:-"0.21876338124275208 -0.07450920437785333 -0.13906173477774242"}
export PLAN10_RIGHT_OUTWARD_PCA=${PLAN10_RIGHT_OUTWARD_PCA:-"0.995533287525177 0.0850074291229248 0.0410749614238739"}
export PLAN10_CONTACT_GEOMETRY_SOURCE=${PLAN10_CONTACT_GEOMETRY_SOURCE:-"${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_bilateral_lower_edge_side_hook_geometry_contract_v1_20260807/derivation.json"}
export PLAN10_LEFT_PALM_INSET_M=${PLAN10_LEFT_PALM_INSET_M:-0.003}
export PLAN10_RIGHT_PALM_INSET_M=${PLAN10_RIGHT_PALM_INSET_M:-0.003}

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_lower_side_right_bottom_pose_gate_v1.sh"
