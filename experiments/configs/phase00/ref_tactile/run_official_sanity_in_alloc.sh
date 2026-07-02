#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_ref_sanity_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_v1.3}"
TACCEL_ROOT="${TACCEL_ROOT:-$ROOT/external/Taccel}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TACCEL_VENV="${TACCEL_VENV:-$ROOT/envs/taccel/.venv}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/sanity/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/sanity/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/sanity/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$TACCEL_ROOT" "$NEWTON_VENV/bin/python" "$TACCEL_VENV/bin/python"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for GPU evidence." >&2
  exit 4
fi

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 00 reference tactile sanity requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
taccel_commit="$(git -C "$TACCEL_ROOT" rev-parse HEAD)"

echo "PHASE00_REF_TACTILE_OFFICIAL_SANITY_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "TACCEL_ROOT=$TACCEL_ROOT"
echo "TACCEL_COMMIT=$taccel_commit"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "NOTE=official_sanity_only_not_training_not_curiosity_success"

newton_log="$LOG_DIR/newton_sensor_contact.log"
taccel_log="$LOG_DIR/taccel_peg.log"
summary_json="$OUTPUT_DIR/official_sanity_summary.json"
summary_md="$REPORT_DIR/official_sanity.md"

set +e
(
  cd "$NEWTON_ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1200 "$NEWTON_VENV/bin/python" -m newton.examples.sensors.example_sensor_contact \
    --device "$DEVICE" --viewer null --num-frames 120 --test --quiet
) >"$newton_log" 2>&1
newton_exit=$?

(
  cd "$TACCEL_ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$TACCEL_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1200 "$TACCEL_VENV/bin/python" -m examples.peg --num_envs 1 --export_mesh
) >"$taccel_log" 2>&1
taccel_exit=$?
set -e

"$NEWTON_VENV/bin/python" - "$summary_json" "$summary_md" "$newton_log" "$taccel_log" "$newton_exit" "$taccel_exit" "$RUN_TAG" "$newton_commit" "$taccel_commit" "$gpu_names" <<'PY'
import json
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
newton_log = Path(sys.argv[3])
taccel_log = Path(sys.argv[4])
newton_exit = int(sys.argv[5])
taccel_exit = int(sys.argv[6])
run_tag = sys.argv[7]
newton_commit = sys.argv[8]
taccel_commit = sys.argv[9]
gpu_names = sys.argv[10]

def log_status(path: Path, exit_code: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "exit_code": exit_code,
        "log": str(path),
        "traceback_absent": "Traceback" not in text,
        "status": "pass" if exit_code == 0 and "Traceback" not in text else "fail",
    }

payload = {
    "classification": "phase00_reference_tactile_official_sanity_v1",
    "run_tag": run_tag,
    "status": "pass",
    "not_training_result": True,
    "not_curiosity_success": True,
    "gpu_names": gpu_names,
    "newton": {
        "commit": newton_commit,
        "command": "python -m newton.examples.sensors.example_sensor_contact --device cuda:0 --viewer null --num-frames 120 --test --quiet",
        **log_status(newton_log, newton_exit),
    },
    "taccel": {
        "commit": taccel_commit,
        "command": "python -m examples.peg --num_envs 1 --export_mesh",
        **log_status(taccel_log, taccel_exit),
    },
}
if payload["newton"]["status"] != "pass" or payload["taccel"]["status"] != "pass":
    payload["status"] = "fail"
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
summary_md.write_text(
    "# Phase 00 Reference Tactile Official Sanity\n\n"
    f"- run_tag: `{run_tag}`\n"
    f"- status: `{payload['status']}`\n"
    f"- Newton: `{payload['newton']['status']}` `{newton_commit}`\n"
    f"- Taccel: `{payload['taccel']['status']}` `{taccel_commit}`\n"
    f"- Newton log: `{newton_log}`\n"
    f"- Taccel log: `{taccel_log}`\n"
    "\nThis is official sanity evidence only, not training and not curiosity success.\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(1)
PY

echo "PHASE00_REF_TACTILE_OFFICIAL_SANITY_END"
