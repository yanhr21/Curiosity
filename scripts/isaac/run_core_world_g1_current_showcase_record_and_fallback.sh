#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 showcase record/render on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
SUITE_STAMP="${SUITE_STAMP:-20260707_g1_lowcarry_current_pass_replay_record}"
VIS_STAMP="${VIS_STAMP:-20260707_g1_lowcarry_current_pass_presentation_fallback}"
MAX_FRAMES="${MAX_FRAMES:-96}"
WIDTH="${WIDTH:-1600}"
HEIGHT="${HEIGHT:-900}"
GIF_DURATION_MS="${GIF_DURATION_MS:-70}"

cd "${ROOT_DIR}"

SHOWCASE_CAPTURE_RGB=0 \
SHOWCASE_RECORD_REPLAY=1 \
SUITE_STAMP="${SUITE_STAMP}" \
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
  bash scripts/isaac/run_core_world_g1_showcase_lowcarry_capture.sh

record_dir="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}/agile_low_cradle_freebox_walk"
replay_csv="${record_dir}/core_world_g1_box_scene_replay.csv"
record_summary="${record_dir}/core_world_g1_box_scene_summary.json"
output_dir="${ROOT_DIR}/experiments/visuals/g1_replay_showcase/${VIS_STAMP}"

"${ISAAC_VENV}/bin/python" scripts/isaac/render_g1_replay_presentation_fallback.py \
  --replay-csv "${replay_csv}" \
  --record-summary "${record_summary}" \
  --output-dir "${output_dir}" \
  --max-frames "${MAX_FRAMES}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --gif-duration-ms "${GIF_DURATION_MS}"

"${ISAAC_VENV}/bin/python" - "${output_dir}" <<'PY'
from pathlib import Path
import sys

try:
    import imageio.v2 as imageio
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] imageio unavailable, leaving GIF/poster only: {exc}")
    raise SystemExit(0)

output_dir = Path(sys.argv[1])
frames = sorted((output_dir / "frames").glob("*.png"))
if not frames:
    raise SystemExit(f"no frames for MP4 in {output_dir / 'frames'}")

for name in ("g1_lowcarry_replay_fallback.mp4", "g1_lowcarry_replay_fallback_annotated.mp4"):
    with imageio.get_writer(str(output_dir / name), fps=15, macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"[INFO] wrote {output_dir / name}")
PY

echo "record_summary=${record_summary}"
echo "replay_csv=${replay_csv}"
echo "visual_dir=${output_dir}"
