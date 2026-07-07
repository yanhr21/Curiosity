#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to generate G1 fallback visualization on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
RECORD_STAMP="${RECORD_STAMP:-20260707_g1_lowcarry_060_runtime_chestpad_showcase_record_min700}"
VIS_STAMP="${VIS_STAMP:-20260707_g1_lowcarry_060_runtime_chestpad_showcase_fallback_min700}"
RECORD_DIR="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${RECORD_STAMP}/agile_low_cradle_freebox_walk"
OUTPUT_DIR="${ROOT_DIR}/experiments/visuals/g1_replay_showcase/${VIS_STAMP}"
REPLAY_CSV="${RECORD_DIR}/core_world_g1_box_scene_replay.csv"
RECORD_SUMMARY="${RECORD_DIR}/core_world_g1_box_scene_summary.json"
CHECKER_SUMMARY="${RECORD_DIR}/check.json"

cd "${ROOT_DIR}"

"${ISAAC_VENV}/bin/python" scripts/isaac/render_g1_replay_presentation_fallback.py \
  --replay-csv "${REPLAY_CSV}" \
  --record-summary "${RECORD_SUMMARY}" \
  --checker-summary "${CHECKER_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  --title "G1 low-carry 0.60 kg runtime chest-pad replay" \
  --subtitle "schematic from recorded Isaac rollout; true Isaac RGB render unavailable in current Kit" \
  --max-frames "${MAX_FRAMES:-96}" \
  --width "${WIDTH:-1600}" \
  --height "${HEIGHT:-900}" \
  --gif-duration-ms "${GIF_DURATION_MS:-70}"

"${ISAAC_VENV}/bin/python" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys

import imageio.v2 as imageio

output_dir = Path(sys.argv[1])
frames = sorted((output_dir / "frames").glob("*.png"))
if not frames:
    raise SystemExit(f"no frames found in {output_dir / 'frames'}")

for name in ("g1_lowcarry_runtime_chestpad_fallback.mp4", "g1_lowcarry_runtime_chestpad_fallback_annotated.mp4"):
    with imageio.get_writer(str(output_dir / name), fps=15, macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"[INFO] wrote {output_dir / name}")
PY

echo "record_dir=${RECORD_DIR}"
echo "visual_dir=${OUTPUT_DIR}"
echo "mp4=${OUTPUT_DIR}/g1_lowcarry_runtime_chestpad_fallback_annotated.mp4"
