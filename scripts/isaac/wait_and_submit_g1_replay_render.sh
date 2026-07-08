#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != mgmtserver* ]]; then
  echo "This watcher is intended for the login node because it only waits and submits srun jobs: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
RECORD_DIR="${RECORD_DIR:?Set RECORD_DIR to the replay-record case directory}"
RENDER_DIR="${RENDER_DIR:?Set RENDER_DIR to the replay-render output directory}"
POLL_SECONDS="${POLL_SECONDS:-30}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
PARTITION="${PARTITION:-cpu}"
JOB_NAME="${JOB_NAME:-g1_replay_viz}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
GRES="${GRES:-gpu:1}"
CAPTURE_EVERY_N_ROWS="${CAPTURE_EVERY_N_ROWS:-1}"
MAX_FRAMES="${MAX_FRAMES:--1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"

cd "${ROOT_DIR}"

summary_path="${RECORD_DIR}/core_world_g1_box_scene_summary.json"
replay_csv="${RECORD_DIR}/core_world_g1_box_scene_replay.csv"

echo "[REPLAY-WAITER] waiting for record summary: ${summary_path}"
while [[ ! -f "${summary_path}" ]]; do
  echo "[REPLAY-WAITER] $(date '+%F %T') still waiting for summary"
  sleep "${POLL_SECONDS}"
done

check_json="$(python3 - "${summary_path}" "${replay_csv}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
replay_csv = Path(sys.argv[2])
summary = json.loads(summary_path.read_text())
failures = []
if summary.get("status") != "pass":
    failures.append(f"record status is {summary.get('status')!r}")
if not bool(summary.get("record_replay_csv")):
    failures.append("record_replay_csv is not true")
fall_events = summary.get("fall_events")
box_drop_events = summary.get("box_drop_events")
if fall_events is None or int(fall_events) != 0:
    failures.append(f"fall_events is {fall_events!r}")
if box_drop_events is None or int(box_drop_events) != 0:
    failures.append(f"box_drop_events is {box_drop_events!r}")
if not replay_csv.is_file():
    failures.append(f"missing replay csv: {replay_csv}")
else:
    rows = max(0, sum(1 for _ in replay_csv.open("r", encoding="utf-8")) - 1)
    if rows < 20:
        failures.append(f"replay rows {rows} < 20")

print(json.dumps({
    "status": "pass" if not failures else "fail",
    "failures": failures,
    "record_status": summary.get("status"),
    "record_replay_csv": summary.get("record_replay_csv"),
    "fall_events": summary.get("fall_events"),
    "box_drop_events": summary.get("box_drop_events"),
}, sort_keys=True))
PY
)"
echo "[REPLAY-WAITER] record check: ${check_json}"

if [[ "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "${check_json}")" != "pass" ]]; then
  echo "[REPLAY-WAITER] record failed; not submitting render" >&2
  exit 1
fi

echo "[REPLAY-WAITER] submitting render for ${replay_csv}"
srun \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gres="${GRES}" \
  --time="${TIME_LIMIT}" \
  --job-name="${JOB_NAME}" \
  --export=ALL,REPLAY_CSV="${replay_csv}",OUTPUT_DIR="${RENDER_DIR}",CAPTURE_EVERY_N_ROWS="${CAPTURE_EVERY_N_ROWS}",MAX_FRAMES="${MAX_FRAMES}",WIDTH="${WIDTH}",HEIGHT="${HEIGHT}" \
  bash scripts/isaac/run_core_world_g1_replay_showcase_render.sh
