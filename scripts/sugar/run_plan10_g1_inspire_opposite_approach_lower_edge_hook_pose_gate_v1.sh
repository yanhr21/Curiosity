#!/usr/bin/env bash
set -euo pipefail

# Geometry source for an opposite-side robot approach. The source run itself
# remains zero-lift and uses the default root only to serialize exact targets;
# a separate full-body scan owns the 180-degree root relocation and admission.

ROOT=/public/home/yanhongru/Curiosity
export PLAN10_OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_opposite_approach_lower_edge_hook_pose_gate_v1_20260807}
export PLAN10_LEFT_CONTACT_PCA_M="0.210224449634552 0.02549079562214676 -0.13906173477774242"
export PLAN10_LEFT_OUTWARD_PCA="0.995533287525177 0.0850074291229248 0.0410749614238739"
export PLAN10_RIGHT_CONTACT_PCA_M="-0.19677920639514923 0.02549079562214676 -0.13906173477774242"
export PLAN10_RIGHT_OUTWARD_PCA="-0.9962605834007263 -0.07864987105131149 -0.03576560318470001"
export PLAN10_CONTACT_GEOMETRY_SOURCE="${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/carrybox_opposite_approach_lower_edge_hook_geometry_contract_v1_20260807/derivation.json"

exec bash "${ROOT}/scripts/sugar/run_plan10_g1_inspire_bilateral_lower_edge_side_hook_pose_gate_v1.sh"
