#!/usr/bin/env bash
set -euo pipefail

# Refresh Phase07 evidence gates inside an already-held Slurm allocation.
# This is audit/data-index refresh only: no training, no rendering, no
# inference, and no curiosity-success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
RUN_TAG="${RUN_TAG:-phase07_evidence_refresh_v1_20260628}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports

{
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'SLURM_JOB_ID=%q\n' "$SLURM_JOB_ID"
  printf 'HOSTNAME=%q\n' "$(hostname)"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'CLASSIFICATION=%q\n' "phase07_evidence_refresh_only_not_training_not_success_claim"
} >"$ROOT/logs/newton/${RUN_TAG}_env.sh"

echo "PHASE07_EVIDENCE_REFRESH_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NOTE=audit_refresh_only_not_training_not_rendering_not_inference_not_success_claim"

echo "=== PHASE07_ACTION_BRIDGE_BACKFILL_REFRESH_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_candidate_action_bridge_backfill_in_alloc.sh"
echo "=== PHASE07_ACTION_BRIDGE_BACKFILL_REFRESH_END ==="

echo "=== PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_REFRESH_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_mainstream_adapter_conversion_preflight_in_alloc.sh"
echo "=== PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_REFRESH_END ==="

echo "=== PHASE07_MAINSTREAM_STAGE1_DATASET_INDEX_REFRESH_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_mainstream_stage1_dataset_index_in_alloc.sh"
echo "=== PHASE07_MAINSTREAM_STAGE1_DATASET_INDEX_REFRESH_END ==="

echo "=== PHASE07_STAGE1_NO_HELDOUT_LEAKAGE_REFRESH_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_stage1_no_heldout_leakage_in_alloc.sh"
echo "=== PHASE07_STAGE1_NO_HELDOUT_LEAKAGE_REFRESH_END ==="

echo "=== PHASE07_OFFICIAL_METHOD_READINESS_REFRESH_START ==="
RUN_TAG="phase07_official_method_readiness_v1_20260627" \
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_official_method_readiness_in_alloc.sh"
echo "=== PHASE07_OFFICIAL_METHOD_READINESS_REFRESH_END ==="

echo "=== PHASE07_HELDOUT_COMPARISON_REFRESH_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase07_heldout_comparison_report_v1.py" \
  --root "$ROOT" \
  --output "$ROOT/experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_heldout_comparison_report_v1.md"
echo "=== PHASE07_HELDOUT_COMPARISON_REFRESH_END ==="

echo "=== PHASE07_HARD_TRAINING_EVIDENCE_GATE_REFRESH_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/audit_phase07_hard_training_evidence_gate_v1.py" \
  --root "$ROOT" \
  --output "$ROOT/experiments/outputs/phase07_hard_training_evidence_gate_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_hard_training_evidence_gate_v1.md"
echo "=== PHASE07_HARD_TRAINING_EVIDENCE_GATE_REFRESH_END ==="

"$NEWTON_VENV/bin/python" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
hard_gate = json.loads((root / "experiments/outputs/phase07_hard_training_evidence_gate_v1_20260627.json").read_text(encoding="utf-8"))
heldout = json.loads((root / "experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json").read_text(encoding="utf-8"))
readiness = json.loads((root / "experiments/outputs/phase07_official_method_readiness_v1_20260627.json").read_text(encoding="utf-8"))
summary = {
    "classification": "phase07_evidence_refresh_summary_v1",
    "status": "pass",
    "hard_gate_status": hard_gate.get("status"),
    "final_curiosity_success_allowed": hard_gate.get("final_curiosity_success_allowed"),
    "hard_gate_open_items": hard_gate.get("open_items", []),
    "heldout_comparison_status": heldout.get("status"),
    "heldout_missing_or_failed_entry_count": heldout.get("missing_or_failed_entry_count"),
    "heldout_curiosity_beats_all_strongest_baselines_without_safety_regression": heldout.get(
        "curiosity_beats_all_strongest_baselines_without_safety_regression"
    ),
    "official_method_readiness_status": readiness.get("status"),
    "official_method_comparison_ready": readiness.get("official_method_comparison_ready"),
    "not_training": True,
    "not_rendering": True,
    "not_inference": True,
    "not_success_claim": True,
}
path = root / "experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "PHASE07_EVIDENCE_REFRESH_END"
