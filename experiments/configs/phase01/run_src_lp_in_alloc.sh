#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_src_lp_$(date +%Y%m%d_%H%M%S)}"
SRC_LP_CONFIG="${SRC_LP_CONFIG:-$ROOT/experiments/configs/phase01/src_lp_manifest.json}"
LP_SCORE_CONFIG="${LP_SCORE_CONFIG:-$ROOT/experiments/configs/phase01/src_lp_scores.json}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/src_lp}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-phase01/core/src_lp}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" "$ROOT/experiments/outputs/$OUTPUT_SUBDIR"

if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: required envs missing under envs/." >&2
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required to verify H200 GPU evidence." >&2
  exit 4
fi
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 01 source-matched LP scoring requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "PHASE01_SRC_LP_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "SRC_LP_CONFIG=$SRC_LP_CONFIG"
echo "LP_SCORE_CONFIG=$LP_SCORE_CONFIG"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "DEVICE=$DEVICE"
echo "NOTE=source_matched_learning_progress_scoring_not_policy_training_not_curiosity_success"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

sanity_log="$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}_official_sensor_contact_sanity.log"
sanity_json="$ROOT/experiments/outputs/$OUTPUT_SUBDIR/${RUN_TAG}_fresh_newton_sensor_contact_sanity.json"

echo "=== OFFICIAL_NEWTON_SANITY_START ==="
set +e
(
  cd "$ROOT/external/newton"
  timeout 900 "$NEWTON_VENV/bin/python" -m newton.examples.sensors.example_sensor_contact --device "$DEVICE" --viewer null --num-frames 160 --test --quiet
) >"$sanity_log" 2>&1
sanity_exit=$?
set -e
"$NEWTON_VENV/bin/python" - "$sanity_log" "$sanity_json" "$sanity_exit" "$DEVICE" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
exit_code = int(sys.argv[3])
device = sys.argv[4]
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
payload = {
    "status": "pass" if exit_code == 0 and "Traceback" not in text else "fail",
    "classification": "fresh_official_newton_sensor_contact_sanity",
    "command": ["timeout", "900", "python", "-m", "newton.examples.sensors.example_sensor_contact", "--device", device, "--viewer", "null", "--num-frames", "160", "--test", "--quiet"],
    "cwd": "external/newton",
    "exit_code": exit_code,
    "traceback_absent": "Traceback" not in text,
    "log": str(log_path),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(1)
PY
echo "=== OFFICIAL_NEWTON_SANITY_END ==="

echo "=== SRC_LP_MANIFEST_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/phase01/build_src_lp_manifest.py" \
  --root "$ROOT" \
  --config "$SRC_LP_CONFIG" \
  --fresh-sanity-json "$sanity_json"
echo "=== SRC_LP_MANIFEST_END ==="

echo "=== SRC_LP_SCORES_START ==="
"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/compute_curiosity_learning_progress_v1.py" \
  --config "$LP_SCORE_CONFIG" \
  --root "$ROOT" \
  --fresh-sanity-json "$sanity_json"
echo "=== SRC_LP_SCORES_END ==="
echo "PHASE01_SRC_LP_EXIT=0"
