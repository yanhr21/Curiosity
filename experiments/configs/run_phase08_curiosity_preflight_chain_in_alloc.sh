#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase08_curiosity_preflight_chain_v1_20260628}"
SOURCE_COMPAT_CONFIG="${SOURCE_COMPAT_CONFIG:-$ROOT/experiments/configs/phase08_advantage_source_compat_v1.json}"
FORWARD_PREFLIGHT_CONFIG="${FORWARD_PREFLIGHT_CONFIG:-$ROOT/experiments/configs/phase08_curiosity_forward_model_preflight_v1.json}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv under envs/." >&2
  exit 3
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs data/processed

echo "PHASE08_CURIOSITY_PREFLIGHT_CHAIN_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "SOURCE_COMPAT_CONFIG=$SOURCE_COMPAT_CONFIG"
echo "FORWARD_PREFLIGHT_CONFIG=$FORWARD_PREFLIGHT_CONFIG"
echo "NOTE=preflight_only_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,210p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

sanity_log="$ROOT/logs/newton/${RUN_TAG}_source_compat_official_sensor_contact_sanity.log"
sanity_json="$ROOT/experiments/outputs/${RUN_TAG}_source_compat_fresh_newton_sensor_contact_sanity.json"

echo "=== SOURCE_COMPAT_OFFICIAL_NEWTON_SANITY_START ==="
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
echo "=== SOURCE_COMPAT_OFFICIAL_NEWTON_SANITY_END ==="

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase08_advantage_source_compat_v1.py" \
  --config "$SOURCE_COMPAT_CONFIG" \
  --root "$ROOT" \
  --fresh-sanity-json "$sanity_json"

CONFIG="$FORWARD_PREFLIGHT_CONFIG" \
RUN_TAG="${RUN_TAG}_forward_model_preflight" \
DEVICE="$DEVICE" \
  bash "$ROOT/experiments/configs/run_curiosity_forward_model_preflight_in_alloc.sh"

echo "PHASE08_CURIOSITY_PREFLIGHT_CHAIN_END"
