#!/usr/bin/env bash
set -euo pipefail

# Encode Phase 01 held-out baseline PNG frame sequences into MP4 videos.
# This is visualization evidence only; it is not training and not a curiosity
# success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_base_mp4_$(date +%Y%m%d_%H%M%S)}"
BASELINE_RUN_TAG="${BASELINE_RUN_TAG:-p01_base_heldout_r1_20260630_0120}"
VIDEO_EXPORT_VENV="${VIDEO_EXPORT_VENV:-$ROOT/envs/trex_dataset/.venv}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VISUAL_ROOT="${VISUAL_ROOT:-$ROOT/experiments/visuals/phase01/core/baselines}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/experiments/outputs/phase01/core/baselines}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/phase01/core/baselines/${RUN_TAG}_mp4_summary.json}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/phase01/core/baselines/${RUN_TAG}_mp4.md}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/phase01/core/baselines" "$OUTPUT_ROOT" "$(dirname "$REPORT_PATH")"

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
  echo "ERROR: Phase 01 MP4 export requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

echo "PHASE01_BASELINE_MP4_EXPORT_START"
echo "RUN_TAG=$RUN_TAG"
echo "BASELINE_RUN_TAG=$BASELINE_RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "VIDEO_EXPORT_VENV=$VIDEO_EXPORT_VENV"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "VISUAL_ROOT=$VISUAL_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "REPORT_PATH=$REPORT_PATH"
echo "NOTE=mp4_visualization_evidence_only_not_training_not_curiosity_success"

"$VIDEO_EXPORT_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$BASELINE_RUN_TAG" "$VIDEO_FPS" "$VISUAL_ROOT" "$OUTPUT_ROOT" "$OUTPUT_JSON" "$REPORT_PATH" "$gpu_names" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
baseline_run_tag = sys.argv[3]
fps = float(sys.argv[4])
visual_root = Path(sys.argv[5])
output_root = Path(sys.argv[6])
output_json = Path(sys.argv[7])
report_path = Path(sys.argv[8])
gpu_names = sys.argv[9]

try:
    import cv2
except Exception as exc:
    raise SystemExit(f"OpenCV/cv2 is required in the prebuilt venv for MP4 export: {exc}") from exc

visual_dirs = sorted(path for path in visual_root.glob(f"{baseline_run_tag}_*") if path.is_dir())
if not visual_dirs:
    raise SystemExit(f"no baseline visual dirs matched {baseline_run_tag}_* under {visual_root}")

outputs = []
for visual_dir in visual_dirs:
    eval_tag = visual_dir.name
    frames_dir = visual_dir / "video_frames"
    frame_paths = sorted(frames_dir.glob("video_frame_*.png"))
    summary_path = output_root / f"{eval_tag}_summary.json"
    visual_summary_path = visual_dir / "summary.json"
    if not frame_paths:
        outputs.append({"eval_tag": eval_tag, "status": "failed_missing_video_frames", "frame_count": 0})
        continue
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        outputs.append({"eval_tag": eval_tag, "status": "failed_first_frame_unreadable", "frame_count": len(frame_paths)})
        continue
    height, width = first.shape[:2]
    mp4_path = visual_dir / "rollout_video.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open VideoWriter for {mp4_path}")
    unreadable = []
    resized = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable.append(frame_path.name)
            continue
        if frame.shape[:2] != (height, width):
            resized.append(frame_path.name)
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
    mp4_payload = {
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
        "resized_frames": resized[:10],
        "generated_inside_h200_slurm_allocation": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    for path in (summary_path, visual_summary_path):
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["phase01_mp4_video_export"] = mp4_payload
            path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append({
        "eval_tag": eval_tag,
        "status": status,
        "failures": failures,
        "mp4": str(mp4_path),
        "source_frame_count": len(frame_paths),
        "encoded_frame_count": encoded_frames,
        "fps_observed": encoded_fps,
        "file_size_bytes": file_size,
        "visual_dir": str(visual_dir),
    })

failed = [item for item in outputs if item["status"] != "pass"]
payload = {
    "classification": "phase01_baseline_mp4_export_summary",
    "run_tag": run_tag,
    "baseline_run_tag": baseline_run_tag,
    "status": "pass" if not failed else "failed_or_partial",
    "not_training_result": True,
    "not_curiosity_success_claim": True,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": os.uname().nodename,
    "gpu_names": gpu_names,
    "h200_verified": "H200" in gpu_names.upper(),
    "visual_root": str(visual_root),
    "evaluations": len(outputs),
    "passed_count": len(outputs) - len(failed),
    "failed_count": len(failed),
    "outputs": outputs,
}
output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Phase 01 Baseline MP4 Export Report",
    "",
    f"- run tag: `{run_tag}`",
    f"- baseline run tag: `{baseline_run_tag}`",
    f"- status: `{payload['status']}`",
    f"- slurm job: `{payload['slurm_job_id']}`",
    f"- hostname: `{payload['hostname']}`",
    f"- gpu names: `{gpu_names}`",
    f"- evaluations: `{len(outputs)}`",
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
        f"- `{item['eval_tag']}` status `{item['status']}` frames `{item['encoded_frame_count']}` "
        f"fps `{item['fps_observed']}` path `{item['mp4']}`"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(payload, indent=2, sort_keys=True))
if failed:
    raise SystemExit(1)
PY

echo "PHASE01_BASELINE_MP4_EXPORT_END"
