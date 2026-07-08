#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run planted online probe-adaptive hold posture audit on login/management node: $(hostname)" >&2
  exit 2
fi

export SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_online_probe_adaptive_hold_posture5_planted_slip_audit_64cm_8kg}"
export PLANTED_STANCE_RAIL_PROPULSION=1
export FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1
export REQUIRE_PLANTED_STANCE_RAIL_PROPULSION=1
export REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS=1
export MAX_NEAR_GROUND_FOOT_SPEED="${MAX_NEAR_GROUND_FOOT_SPEED:-0.80}"
export MAX_NEAR_GROUND_FOOT_SLIP="${MAX_NEAR_GROUND_FOOT_SLIP:-0.20}"

echo "[PLANTED_SLIP_AUDIT] SUITE_STAMP=${SUITE_STAMP}"
echo "[PLANTED_SLIP_AUDIT] CASE_FILTER=${CASE_FILTER:-<all>}"
echo "[PLANTED_SLIP_AUDIT] PLANTED_STANCE_RAIL_PROPULSION=${PLANTED_STANCE_RAIL_PROPULSION}"
echo "[PLANTED_SLIP_AUDIT] FREEZE_COMMANDED_STANCE_FOOT_TARGETS=${FREEZE_COMMANDED_STANCE_FOOT_TARGETS}"
echo "[PLANTED_SLIP_AUDIT] MAX_NEAR_GROUND_FOOT_SPEED=${MAX_NEAR_GROUND_FOOT_SPEED}"
echo "[PLANTED_SLIP_AUDIT] MAX_NEAR_GROUND_FOOT_SLIP=${MAX_NEAR_GROUND_FOOT_SLIP}"

bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh
