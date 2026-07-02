#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_resid_base_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/resid_base_train.json}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
RUN_MODE="${RUN_MODE:-train}"
DEVICE="${DEVICE:-cuda:0}"
ALLOW_REAL_TRAINING="${ALLOW_REAL_TRAINING:-0}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/resid/base}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ "$RUN_MODE" == "train" && "$ALLOW_REAL_TRAINING" != "1" ]]; then
  echo "ERROR: RUN_MODE=train requires ALLOW_REAL_TRAINING=1." >&2
  exit 3
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR"
if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: required envs missing under envs/." >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required to verify H200 GPU evidence." >&2
  exit 5
fi
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 01 residual training requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 6
fi

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

output_dir="$("$TRAINER_VENV/bin/python" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["output_dir"])
PY
)"
mkdir -p "$ROOT/$output_dir"

echo "PHASE01_RESID_BASE_TRAIN_START"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_MODE=$RUN_MODE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "CONFIG=$CONFIG"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "DEVICE=$DEVICE"
echo "OUTPUT_DIR=$output_dir"
echo "NOTE=learned_no_curiosity_baseline_component_not_curiosity_success"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

sanity_log="$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}_official_sensor_contact_sanity.log"
sanity_json="$ROOT/$output_dir/${RUN_TAG}_fresh_newton_sensor_contact_sanity.json"

echo "=== OFFICIAL_NEWTON_SANITY_START ==="
set +e
(
  cd "$ROOT/external/newton"
  timeout 900 "$NEWTON_VENV/bin/python" -m newton.examples.sensors.example_sensor_contact --device "$DEVICE" --viewer null --num-frames 160 --test --quiet
) >"$sanity_log" 2>&1
sanity_exit=$?
set -e
"$TRAINER_VENV/bin/python" - "$sanity_log" "$sanity_json" "$sanity_exit" "$DEVICE" <<'PY'
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

util_csv="$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}_gpu_utilization.csv"
util_json="$ROOT/$output_dir/${RUN_TAG}_gpu_utilization.json"
monitor_pid=""
if [[ "$RUN_MODE" == "train" ]]; then
  echo "timestamp,utilization.gpu [%],memory.used [MiB]" >"$util_csv"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits >>"$util_csv" || true
      sleep 30
    done
  ) &
  monitor_pid="$!"
fi

set +e
"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/train_residual_adapter_v1.py" \
  --config "$CONFIG" \
  --root "$ROOT" \
  --fresh-sanity-json "$sanity_json" \
  --run-tag "$RUN_TAG" \
  --run-mode "$RUN_MODE" \
  --device "$DEVICE"
trainer_exit=$?
set -e

if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  "$TRAINER_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$util_csv" "$util_json" "$CONFIG" "$trainer_exit" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
util_csv = Path(sys.argv[3])
util_json = Path(sys.argv[4])
config_path = Path(sys.argv[5])
trainer_exit = int(sys.argv[6])
config = json.loads(config_path.read_text(encoding="utf-8"))
summary_path = root / config["output_dir"] / f"{run_tag}_summary.json"
threshold = float(config["real_training"]["min_gpu_utilization_percent"])
values = []
memory_values = []
if util_csv.exists():
    with util_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                values.append(float(row["utilization.gpu [%]"]))
                memory_values.append(float(row["memory.used [MiB]"]))
            except (KeyError, TypeError, ValueError):
                continue
payload = {
    "classification": "phase01_resid_base_gpu_utilization_v1",
    "run_tag": run_tag,
    "csv": str(util_csv),
    "sample_count": len(values),
    "min_gpu_utilization_percent": min(values) if values else None,
    "max_gpu_utilization_percent": max(values) if values else None,
    "mean_gpu_utilization_percent": sum(values) / len(values) if values else None,
    "samples_below_threshold": sum(1 for value in values if value < threshold),
    "max_memory_used_mib": max(memory_values) if memory_values else None,
    "threshold_percent": threshold,
    "status": "pass" if values and (sum(values) / len(values)) >= threshold else "fail",
}
util_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gpu_names"] = sys.argv[7] if len(sys.argv) > 7 else None
    summary["gpu_utilization"] = payload
    summary["baseline_component_not_curiosity_success"] = True
    if summary.get("run_mode") == "train" and payload["status"] != "pass":
        summary["real_training_result"] = False
        failures = list(summary.get("failures", []))
        failures.append("gpu_utilization_below_threshold")
        summary["failures"] = failures
        summary["status"] = "fail"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if trainer_exit != 0:
    raise SystemExit(trainer_exit)
if payload["status"] != "pass":
    raise SystemExit(7)
PY
fi

if [[ "$trainer_exit" -ne 0 ]]; then
  exit "$trainer_exit"
fi
echo "PHASE01_RESID_BASE_TRAIN_END"
