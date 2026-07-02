#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/learning_progress.json}"
RUN_TAG="${RUN_TAG:-p01_lp_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/lp}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-phase01/core/lp}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" "$ROOT/experiments/outputs/$OUTPUT_SUBDIR"

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv; configure envs/ locally before compute use." >&2
  exit 4
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv; configure envs/residual_adapter locally before compute use." >&2
  exit 6
fi

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "PHASE01_LEARNING_PROGRESS_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "CONFIG=$CONFIG"
echo "DEVICE=$DEVICE"
echo "POLICY_UPDATED=false"
echo "NOTE=learning_progress_scoring_not_policy_training_not_curiosity_success"

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
    "command": [
        "timeout",
        "900",
        "python",
        "-m",
        "newton.examples.sensors.example_sensor_contact",
        "--device",
        device,
        "--viewer",
        "null",
        "--num-frames",
        "160",
        "--test",
        "--quiet",
    ],
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

"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/compute_curiosity_learning_progress_v1.py" \
  --config "$CONFIG" \
  --root "$ROOT" \
  --fresh-sanity-json "$sanity_json"
echo "PHASE01_LEARNING_PROGRESS_EXIT=0"
