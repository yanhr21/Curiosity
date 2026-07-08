#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run online probe-adaptive hold posture slip audit on login/management node: $(hostname)" >&2
  exit 2
fi

export SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_online_probe_adaptive_hold_posture5_slip_audit_64cm_8kg}"
export MAX_NEAR_GROUND_FOOT_SPEED="${MAX_NEAR_GROUND_FOOT_SPEED:-0.80}"
export MAX_NEAR_GROUND_FOOT_SLIP="${MAX_NEAR_GROUND_FOOT_SLIP:-0.20}"

echo "[SLIP_AUDIT] SUITE_STAMP=${SUITE_STAMP}"
echo "[SLIP_AUDIT] MAX_NEAR_GROUND_FOOT_SPEED=${MAX_NEAR_GROUND_FOOT_SPEED}"
echo "[SLIP_AUDIT] MAX_NEAR_GROUND_FOOT_SLIP=${MAX_NEAR_GROUND_FOOT_SLIP}"

bash scripts/isaac/run_direct_carry_online_probe_adaptive_hold_posture_suite.sh
