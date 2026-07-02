#!/usr/bin/env bash
set -euo pipefail

# Encode Phase 00 long-horizon PNG frame sequences into real MP4 videos.
# Must run inside a Curiosity-owned H200 Slurm allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-phase00_video_mp4_export_h200_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
VIDEO_EXPORT_VENV="${VIDEO_EXPORT_VENV:-$ROOT/envs/trex_dataset/.venv}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_ROW_FILES="${VIDEO_ROW_FILES:-$ROOT/experiments/outputs/phase00_core_asset_generation_h200_long_20260629_182052_phase00_cell_rows.jsonl:$ROOT/experiments/outputs/phase00_core_asset_generation_h200_long_repair2_20260629_183216_phase00_cell_rows.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/${RUN_TAG}_phase00_video_mp4_summary.json}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/${RUN_TAG}_phase00_video_mp4.md}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports

if [[ ! -x "$VIDEO_EXPORT_VENV/bin/python" ]]; then
  echo "ERROR: missing local video export venv python: $VIDEO_EXPORT_VENV/bin/python" >&2
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required to verify H200 GPU evidence." >&2
  exit 4
fi

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 00 MP4 export requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

echo "PHASE00_VIDEO_MP4_EXPORT_H200_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "VIDEO_EXPORT_VENV=$VIDEO_EXPORT_VENV"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "VIDEO_ROW_FILES=$VIDEO_ROW_FILES"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "REPORT_PATH=$REPORT_PATH"

"$VIDEO_EXPORT_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$VIDEO_ROW_FILES" "$VIDEO_FPS" "$OUTPUT_JSON" "$REPORT_PATH" "$gpu_names" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
row_files = [Path(item) for item in sys.argv[3].split(":") if item]
fps = float(sys.argv[4])
output_json = Path(sys.argv[5])
report_path = Path(sys.argv[6])
gpu_names = sys.argv[7]

if fps <= 0:
    raise SystemExit(f"VIDEO_FPS must be positive, got {fps}")

try:
    import cv2
except Exception as exc:
    raise SystemExit(f"OpenCV/cv2 is required in the prebuilt venv for MP4 export: {exc}") from exc

rows_by_key = {}
source_rows = []
for row_file in row_files:
    if not row_file.exists():
        raise SystemExit(f"missing row file: {row_file}")
    for line in row_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "generated_pending_manual_review":
            continue
        key = f"{row['split']}/{row['cell']}"
        # Later row files override earlier partial evidence for the same key.
        rows_by_key[key] = row
        source_rows.append({"key": key, "row_file": str(row_file)})

rows = [rows_by_key[key] for key in sorted(rows_by_key)]
if not rows:
    raise SystemExit("no generated Phase 00 rows found for MP4 export")

outputs = []
for row in rows:
    key = f"{row['split']}/{row['cell']}"
    visual_dir = root / row["visual_dir"]
    frames_dir = visual_dir / "video_frames"
    frame_paths = sorted(frames_dir.glob("video_frame_*.png"))
    if not frame_paths:
        outputs.append({
            "cell": key,
            "status": "failed_missing_video_frames",
            "visual_dir": str(visual_dir),
            "frame_count": 0,
        })
        continue

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        outputs.append({
            "cell": key,
            "status": "failed_first_frame_unreadable",
            "visual_dir": str(visual_dir),
            "frame_count": len(frame_paths),
        })
        continue
    height, width = first.shape[:2]
    mp4_path = visual_dir / "rollout_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open VideoWriter for {mp4_path}")

    unreadable = []
    size_mismatch = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable.append(frame_path.name)
            continue
        if frame.shape[:2] != (height, width):
            size_mismatch.append(frame_path.name)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()

    probe = cv2.VideoCapture(str(mp4_path))
    opened = bool(probe.isOpened())
    encoded_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    encoded_fps = float(probe.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    probe.release()
    file_size = mp4_path.stat().st_size if mp4_path.exists() else 0

    status = "pass"
    failures = []
    if unreadable:
        status = "failed_unreadable_frames"
        failures.append(f"unreadable={len(unreadable)}")
    if file_size <= 0 or not opened or encoded_frames <= 0:
        status = "failed_invalid_mp4"
        failures.append("invalid_or_empty_mp4")

    summary_path = root / row["summary"]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["phase00_mp4_video_export"] = {
            "status": status,
            "path": str(mp4_path),
            "format": "mp4",
            "codec": "mp4v",
            "source_video_frames_dir": str(frames_dir),
            "source_frame_count": len(frame_paths),
            "encoded_frame_count": encoded_frames,
            "fps_requested": fps,
            "fps_observed": encoded_fps,
            "width": width,
            "height": height,
            "file_size_bytes": file_size,
            "unreadable_frames": unreadable[:10],
            "size_mismatch_frames": size_mismatch[:10],
            "generated_inside_h200_slurm_allocation": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        visual_summary = visual_dir / "summary.json"
        if visual_summary.exists():
            visual_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs.append({
        "cell": key,
        "status": status,
        "failures": failures,
        "mp4": str(mp4_path),
        "source_frame_count": len(frame_paths),
        "encoded_frame_count": encoded_frames,
        "fps_requested": fps,
        "fps_observed": encoded_fps,
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "visual_dir": str(visual_dir),
    })

failed = [item for item in outputs if item["status"] != "pass"]
payload = {
    "classification": "phase00_video_mp4_export_h200_summary",
    "run_tag": run_tag,
    "status": "pass" if not failed else "failed_or_partial",
    "not_training_result": True,
    "not_curiosity_success_claim": True,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": os.uname().nodename,
    "gpu_names": gpu_names,
    "h200_verified": "H200" in gpu_names.upper(),
    "row_files": [str(path) for path in row_files],
    "unique_generated_cells": len(rows),
    "passed_count": len(outputs) - len(failed),
    "failed_count": len(failed),
    "outputs": outputs,
}
output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Phase 00 MP4 Video Export H200 Report",
    "",
    f"- run tag: `{run_tag}`",
    f"- status: `{payload['status']}`",
    f"- slurm job: `{payload['slurm_job_id']}`",
    f"- hostname: `{payload['hostname']}`",
    f"- gpu names: `{gpu_names}`",
    f"- unique generated cells: `{len(rows)}`",
    f"- passed MP4 exports: `{len(outputs) - len(failed)}`",
    f"- failed MP4 exports: `{len(failed)}`",
    "",
    "This is video visualization evidence only. It is not training evidence and not a curiosity success claim.",
    "",
    "## Videos",
    "",
]
for item in outputs:
    lines.append(
        f"- `{item['cell']}` status `{item['status']}` frames `{item['encoded_frame_count']}` "
        f"fps `{item['fps_observed']}` path `{item['mp4']}`"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(payload, indent=2, sort_keys=True))
if failed:
    raise SystemExit(1)
PY

echo "PHASE00_VIDEO_MP4_EXPORT_H200_END"
