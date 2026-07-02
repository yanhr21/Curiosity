#!/usr/bin/env bash
set -euo pipefail

# Run Phase 00 core asset generation/validation inside an H200 Slurm allocation.
# This script must never be launched directly on a login node.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-phase00_core_asset_generation_h200_$(date +%Y%m%d_%H%M%S)}"
CATALOG="${CATALOG:-$ROOT/experiments/configs/phase00_core_tabletop_asset_catalog_v1.json}"
DEVICE="${DEVICE:-cuda:0}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-$ROOT/external/newton-assets-cache}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/${RUN_TAG}_phase00_core_asset_generation_h200.md}"
AGGREGATE_JSON="${AGGREGATE_JSON:-$ROOT/experiments/outputs/${RUN_TAG}_phase00_core_asset_generation_h200_summary.json}"
PHASE00_MIN_NUM_STEPS="${PHASE00_MIN_NUM_STEPS:-1800}"
PHASE00_NUM_STEPS="${PHASE00_NUM_STEPS:-1800}"
PHASE00_PRE_RECORD_WARMUP_STEPS="${PHASE00_PRE_RECORD_WARMUP_STEPS:-60}"
PHASE00_FINAL_HOLD_DURATION="${PHASE00_FINAL_HOLD_DURATION:-12.0}"
PHASE00_HOLD_DURATION_MIN="${PHASE00_HOLD_DURATION_MIN:-8.0}"
PHASE00_VIDEO_FRAME_STRIDE="${PHASE00_VIDEO_FRAME_STRIDE:-3}"
PHASE00_VIDEO_FPS="${PHASE00_VIDEO_FPS:-20}"
PHASE00_CELL_FILTER="${PHASE00_CELL_FILTER:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi
if ! [[ "$PHASE00_MIN_NUM_STEPS" =~ ^[0-9]+$ && "$PHASE00_NUM_STEPS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PHASE00_MIN_NUM_STEPS and PHASE00_NUM_STEPS must be integer frame counts." >&2
  exit 8
fi
if (( PHASE00_NUM_STEPS < PHASE00_MIN_NUM_STEPS )); then
  echo "ERROR: Phase 00 full asset generation refuses short rollouts: PHASE00_NUM_STEPS=$PHASE00_NUM_STEPS < PHASE00_MIN_NUM_STEPS=$PHASE00_MIN_NUM_STEPS." >&2
  echo "Use a separately labeled diagnostic script only if the user explicitly asks for a smoke test." >&2
  exit 9
fi
if [[ -z "${PHASE00_SAMPLE_STEPS:-}" ]]; then
  last_step=$((PHASE00_NUM_STEPS - 1))
  PHASE00_SAMPLE_STEPS="0,$((PHASE00_NUM_STEPS / 6)),$((PHASE00_NUM_STEPS / 3)),$((PHASE00_NUM_STEPS / 2)),$((PHASE00_NUM_STEPS * 2 / 3)),$((PHASE00_NUM_STEPS * 5 / 6)),$last_step"
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

if [[ ! -f "$CATALOG" ]]; then
  echo "ERROR: missing catalog: $CATALOG" >&2
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python: $NEWTON_VENV/bin/python" >&2
  exit 4
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required for catalog iteration." >&2
  exit 5
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required to verify H200 GPU evidence." >&2
  exit 6
fi

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 00 requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 7
fi

echo "PHASE00_CORE_ASSET_GENERATION_H200_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CATALOG=$CATALOG"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "NEWTON_CACHE_PATH=$NEWTON_CACHE_PATH"
echo "DEVICE=$DEVICE"
echo "PHASE00_MIN_NUM_STEPS=$PHASE00_MIN_NUM_STEPS"
echo "PHASE00_NUM_STEPS=$PHASE00_NUM_STEPS"
echo "PHASE00_SAMPLE_STEPS=$PHASE00_SAMPLE_STEPS"
echo "PHASE00_PRE_RECORD_WARMUP_STEPS=$PHASE00_PRE_RECORD_WARMUP_STEPS"
echo "PHASE00_FINAL_HOLD_DURATION=$PHASE00_FINAL_HOLD_DURATION"
echo "PHASE00_HOLD_DURATION_MIN=$PHASE00_HOLD_DURATION_MIN"
echo "PHASE00_VIDEO_FRAME_STRIDE=$PHASE00_VIDEO_FRAME_STRIDE"
echo "PHASE00_VIDEO_FPS=$PHASE00_VIDEO_FPS"
echo "PHASE00_CELL_FILTER=$PHASE00_CELL_FILTER"

echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

cell_rows="$ROOT/experiments/outputs/${RUN_TAG}_phase00_cell_rows.jsonl"
: >"$cell_rows"

map_visual_cue() {
  case "${1:-not_specified}" in
    truthful*|truthful_fill|truthful_fill_cue) echo "truthful_fill_cue" ;;
    hidden*|hidden_fill|hidden_fill_cue) echo "hidden_fill_cue" ;;
    misleading*|misleading_empty|misleading_full|misleading_fill|misleading_fill_cue) echo "misleading_fill_cue" ;;
    *) echo "not_specified" ;;
  esac
}

map_family_scene() {
  case "$1" in
    cup_like_official) echo "cube existing_cup_asset official_cup_asset" ;;
    box_procedural) echo "cube official_object official_cube_box_proxy" ;;
    cylinder_procedural) echo "pen official_object official_pen_cylinder_like_proxy" ;;
    *) echo "unsupported unsupported unsupported" ;;
  esac
}

apply_modality_mask() {
  local eval_tag="$1"
  local mask_mode="$2"
  "$NEWTON_VENV/bin/python" - "$ROOT" "$eval_tag" "$mask_mode" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
run_tag = sys.argv[2]
mask_mode = sys.argv[3]
npz_path = root / "experiments" / "outputs" / f"{run_tag}.npz"
summary_path = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
visual_validation_path = root / "experiments" / "outputs" / f"{run_tag}_visual_validation.json"

if not npz_path.exists():
    raise SystemExit(f"missing npz for modality mask: {npz_path}")

with np.load(npz_path, allow_pickle=False) as data:
    arrays = {key: data[key].copy() for key in data.files}

mask_id = {
    "vision_contact": 0,
    "contact_only_masked_vision": 1,
    "vision_only_masked_contact": 2,
    "alternating_mask": 3,
}.get(mask_mode, -1)

vision_mask = np.ones_like(arrays["newton.camera.object_z"], dtype=np.int32)
contact_mask = np.ones_like(arrays["newton.panda.rigid_contact_count"], dtype=np.int32)

if mask_mode == "contact_only_masked_vision":
    if "newton.camera.color_rgba" in arrays:
        arrays["newton.camera.color_rgba"] = np.zeros_like(arrays["newton.camera.color_rgba"])
    if "newton.camera.depth" in arrays:
        arrays["newton.camera.depth"] = np.zeros_like(arrays["newton.camera.depth"])
    vision_mask[...] = 0
elif mask_mode == "vision_only_masked_contact":
    arrays["newton.panda.rigid_contact_count"] = np.zeros_like(arrays["newton.panda.rigid_contact_count"])
    contact_mask[...] = 0
elif mask_mode == "alternating_mask":
    if "newton.camera.color_rgba" in arrays and arrays["newton.camera.color_rgba"].shape[0] > 0:
        arrays["newton.camera.color_rgba"][1::2] = 0
    if "newton.camera.depth" in arrays and arrays["newton.camera.depth"].shape[0] > 0:
        arrays["newton.camera.depth"][1::2] = 0
    arrays["newton.panda.rigid_contact_count"][::2] = 0
    vision_mask[1::2] = 0
    contact_mask[::2] = 0

arrays["candidate.modality.mask_mode_id"] = np.asarray([mask_id], dtype=np.int32)
arrays["candidate.modality.vision_available_mask"] = vision_mask
arrays["candidate.modality.contact_available_mask"] = contact_mask
np.savez_compressed(npz_path, **arrays)

summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
summary["phase00_modality_mask"] = {
    "status": "applied" if mask_mode != "vision_contact" else "identity",
    "mask_mode": mask_mode,
    "source_namespace": "candidate.modality.*",
    "applied_after_newton_export_inside_h200_allocation": True,
    "note": "Masking is a dataset-modality intervention over exported arrays; raw Newton rollout visuals remain under the visual directory for manual inspection.",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "experiments" / "visuals" / "phase00" / run_tag / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

visual_validation = json.loads(visual_validation_path.read_text(encoding="utf-8")) if visual_validation_path.exists() else {}
visual_validation["phase00_modality_mask"] = summary["phase00_modality_mask"]
visual_validation_path.write_text(json.dumps(visual_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary["phase00_modality_mask"], indent=2, sort_keys=True))
PY
}

run_cell() {
  local split="$1"
  local cell_json="$2"
  local cell family mass friction cue mask com_offset object_com_offset_xyz scene tracked proxy_label visual_cue eval_tag status blocker
  cell="$(jq -r '.cell' <<<"$cell_json")"
  if [[ -n "$PHASE00_CELL_FILTER" ]]; then
    case ",$PHASE00_CELL_FILTER," in
      *",$cell,"*) ;;
      *) return 0 ;;
    esac
  fi
  family="$(jq -r '.family' <<<"$cell_json")"
  mass="$(jq -r '.mass_kg // empty' <<<"$cell_json")"
  friction="$(jq -r '.friction_mu // empty' <<<"$cell_json")"
  cue="$(jq -r '.visual_cue // "not_specified"' <<<"$cell_json")"
  mask="$(jq -r '.mask_mode // "vision_contact"' <<<"$cell_json")"
  com_offset="$(jq -r '.center_of_mass_offset_m // 0.0' <<<"$cell_json")"
  object_com_offset_xyz="$com_offset,0,0"
  read -r scene tracked proxy_label < <(map_family_scene "$family")
  visual_cue="$(map_visual_cue "$cue")"
  eval_tag="${RUN_TAG}_${split}_${cell}"
  status="pass_candidate"
  blocker=""

  if [[ "$scene" == "unsupported" ]]; then
    status="blocked_unsupported_family"
    blocker="No faithful exporter mapping exists for family=$family."
  fi
  if [[ "$status" == pass_candidate ]]; then
    echo "=== PHASE00_CELL_START split=$split cell=$cell family=$family proxy=$proxy_label ==="
    RUN_TAG="$eval_tag" \
    NEWTON_VENV="$NEWTON_VENV" \
    TRAINER_VENV="$TRAINER_VENV" \
    SCENE="$scene" \
    TRACKED_OBJECT="$tracked" \
    CONTROLLER_MODE="lift_hold_feedback" \
    FINAL_HOLD_DURATION="$PHASE00_FINAL_HOLD_DURATION" \
    LIFT_HEIGHT_MIN="0.12" \
    HOLD_DURATION_MIN="$PHASE00_HOLD_DURATION_MIN" \
    DROP_HEIGHT_LOSS="0.05" \
    PHYSICS_VARIANT_LABEL="phase00_${split}_${cell}_${proxy_label}" \
    OBJECT_MASS_KG="$mass" \
    OBJECT_FRICTION_MU="$friction" \
    OBJECT_COM_OFFSET_XYZ="$object_com_offset_xyz" \
    FILL_LABEL="$cell" \
    NOMINAL_VISUAL_FILL="0.5" \
    VISUAL_FILL_CUE="$visual_cue" \
    VISUAL_FILL_CUE_RENDERED="0" \
    FEEDBACK_MIN_CONTACT_COUNT="20" \
    FEEDBACK_ACCEL_THRESHOLD="6.5" \
    FEEDBACK_HEIGHT_DROP_THRESHOLD="0.015" \
    PRE_RECORD_WARMUP_STEPS="$PHASE00_PRE_RECORD_WARMUP_STEPS" \
    NUM_STEPS="$PHASE00_NUM_STEPS" \
    SAMPLE_STEPS="$PHASE00_SAMPLE_STEPS" \
    VIDEO_FRAME_STRIDE="$PHASE00_VIDEO_FRAME_STRIDE" \
    VIDEO_FPS="$PHASE00_VIDEO_FPS" \
    DEVICE="$DEVICE" \
    NEWTON_CACHE_PATH="$NEWTON_CACHE_PATH" \
      bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
        2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

    apply_modality_mask "$eval_tag" "$mask"

    RUN_TAG="$eval_tag" \
    NEWTON_VENV="$NEWTON_VENV" \
    BASELINE_NAME="phase00_asset_generation_${proxy_label}" \
    MASS_LABEL="$mass" \
    FRICTION_LABEL="$friction" \
    POSE_SEED="$cell" \
    MANUAL_VISUAL_INSPECTION="pending_manual_review" \
      bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"

    status="generated_pending_manual_review"
    blocker=""
    echo "=== PHASE00_CELL_END split=$split cell=$cell status=$status ==="
  else
    echo "PHASE00_CELL_BLOCKED split=$split cell=$cell family=$family status=$status blocker=$blocker"
  fi

  jq -cn \
    --arg split "$split" \
    --arg cell "$cell" \
    --arg family "$family" \
    --arg proxy_label "$proxy_label" \
    --arg mask_mode "$mask" \
    --arg visual_cue "$visual_cue" \
    --arg object_com_offset_xyz "$object_com_offset_xyz" \
    --arg status "$status" \
    --arg blocker "$blocker" \
    --arg run_tag "$eval_tag" \
    --arg summary "experiments/outputs/${eval_tag}_summary.json" \
    --arg metrics "experiments/outputs/${eval_tag}_metrics.json" \
    --arg visual_dir "experiments/visuals/phase00/${eval_tag}" \
    '{split:$split, cell:$cell, family:$family, proxy_label:$proxy_label, mask_mode:$mask_mode, visual_cue:$visual_cue, object_com_offset_xyz:$object_com_offset_xyz, status:$status, blocker:$blocker, run_tag:$run_tag, summary:$summary, metrics:$metrics, visual_dir:$visual_dir}' \
    >>"$cell_rows"
}

for split in train validation held_out; do
  while IFS= read -r cell_json; do
    run_cell "$split" "$cell_json"
  done < <(jq -c ".split_cells.${split}[]" "$CATALOG")
done

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$cell_rows" "$AGGREGATE_JSON" "$REPORT_PATH" "$gpu_names" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
rows_path = Path(sys.argv[3])
aggregate_path = Path(sys.argv[4])
report_path = Path(sys.argv[5])
gpu_names = sys.argv[6]

rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    row["summary_exists"] = (root / row["summary"]).exists()
    row["metrics_exists"] = (root / row["metrics"]).exists()
    row["contact_sheet_exists"] = (root / row["visual_dir"] / "contact_sheet.png").exists()
    row["frame_browser_exists"] = (root / row["visual_dir"] / "frame_browser.html").exists()
    row["rollout_video_exists"] = (root / row["visual_dir"] / "rollout_video.gif").exists()
    if row["status"] == "generated_pending_manual_review":
        required = [
            row["summary_exists"],
            row["metrics_exists"],
            row["contact_sheet_exists"],
            row["frame_browser_exists"],
            row["rollout_video_exists"],
        ]
        if not all(required):
            row["status"] = "failed_missing_generated_artifact"
            row["blocker"] = "One or more required generated artifacts are missing."

passed_like = [r for r in rows if r["status"] == "generated_pending_manual_review"]
blocked = [r for r in rows if r["status"].startswith("blocked") or r["status"].startswith("failed")]
payload = {
    "classification": "phase00_core_asset_generation_h200_summary",
    "run_tag": run_tag,
    "status": "incomplete_manual_review_or_blockers" if blocked or passed_like else "failed_no_cells_generated",
    "not_training_result": True,
    "not_curiosity_success_claim": True,
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "hostname": __import__("socket").gethostname(),
    "gpu_names": gpu_names,
    "h200_verified": "H200" in gpu_names.upper(),
    "generation_profile": {
        "minimum_num_steps": int(__import__("os").environ.get("PHASE00_MIN_NUM_STEPS", "1800")),
        "num_steps": int(__import__("os").environ.get("PHASE00_NUM_STEPS", "1800")),
        "sample_steps": __import__("os").environ.get("PHASE00_SAMPLE_STEPS"),
        "pre_record_warmup_steps": __import__("os").environ.get("PHASE00_PRE_RECORD_WARMUP_STEPS"),
        "final_hold_duration_s": __import__("os").environ.get("PHASE00_FINAL_HOLD_DURATION"),
        "hold_duration_min_s": __import__("os").environ.get("PHASE00_HOLD_DURATION_MIN"),
        "video_frame_stride": __import__("os").environ.get("PHASE00_VIDEO_FRAME_STRIDE"),
        "video_fps": __import__("os").environ.get("PHASE00_VIDEO_FPS"),
        "short_rollout_refusal_enabled": True,
    },
    "rows": rows,
    "generated_pending_manual_review_count": len(passed_like),
    "blocked_or_failed_count": len(blocked),
    "manual_visual_inspection_required": True,
}
aggregate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Phase 00 Core Asset Generation H200 Report",
    "",
    f"- run tag: `{run_tag}`",
    f"- status: `{payload['status']}`",
    f"- slurm job: `{payload['slurm_job_id']}`",
    f"- hostname: `{payload['hostname']}`",
    f"- gpu names: `{gpu_names}`",
    f"- generation profile: `{payload['generation_profile']}`",
    f"- generated pending manual review: `{len(passed_like)}`",
    f"- blocked or failed: `{len(blocked)}`",
    "",
    "This report is asset-generation evidence only. It is not training and not a curiosity success claim.",
    "",
    "## Cells",
    "",
]
for row in rows:
    lines.append(
        f"- `{row['split']}/{row['cell']}` family `{row['family']}` proxy `{row['proxy_label']}` "
        f"mask `{row['mask_mode']}` status `{row['status']}` blocker `{row['blocker']}` "
        f"video `{row['rollout_video_exists']}` contact_sheet `{row['contact_sheet_exists']}`"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE00_CORE_ASSET_GENERATION_H200_END"
