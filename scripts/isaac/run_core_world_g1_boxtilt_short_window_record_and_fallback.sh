#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 boxtilt record/render on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
SUITE_STAMP="${SUITE_STAMP:-20260707_g1_boxtilt_avgpos_short_window_760_replay_record}"
VIS_STAMP="${VIS_STAMP:-20260707_g1_boxtilt_avgpos_short_window_760_progress_fallback}"
MAX_FRAMES="${MAX_FRAMES:-96}"
WIDTH="${WIDTH:-1600}"
HEIGHT="${HEIGHT:-900}"
GIF_DURATION_MS="${GIF_DURATION_MS:-70}"

cd "${ROOT_DIR}"

set +e
RECORD_REPLAY_CSV=1 \
RECORD_REPLAY_EVERY_N_STEPS="${RECORD_REPLAY_EVERY_N_STEPS:-10}" \
SUITE_STAMP="${SUITE_STAMP}" \
OUTPUT_ROOT="${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/${SUITE_STAMP}" \
  bash scripts/isaac/run_core_world_g1_boxtilt_avgpos_short_window_suite.sh
record_status=$?
set -e

record_dir="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}/agile_low_cradle_freebox_walk"
replay_csv="${record_dir}/core_world_g1_box_scene_replay.csv"
record_summary="${record_dir}/core_world_g1_box_scene_summary.json"
checker_summary="${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_avgpos_short_window/${SUITE_STAMP}/boxtilt_avgpos_short_window_summary.json"
output_dir="${ROOT_DIR}/experiments/visuals/g1_replay_showcase/${VIS_STAMP}"

"${ISAAC_VENV}/bin/python" scripts/isaac/render_g1_replay_presentation_fallback.py \
  --replay-csv "${replay_csv}" \
  --record-summary "${record_summary}" \
  --checker-summary "${checker_summary}" \
  --output-dir "${output_dir}" \
  --title "G1 boxtilt short-window progress" \
  --subtitle "0.75 kg free box, strict checker fails on lateral drift/tilt; not Isaac camera render" \
  --gif-name "g1_boxtilt_short_window_progress.gif" \
  --poster-name "g1_boxtilt_short_window_progress_poster.png" \
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

for name in ("g1_boxtilt_short_window_progress.mp4", "g1_boxtilt_short_window_progress_annotated.mp4"):
    with imageio.get_writer(str(output_dir / name), fps=15, macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"[INFO] wrote {output_dir / name}")
PY

cat > "${output_dir}/README.md" <<EOF
# G1 Boxtilt Short-Window Progress Fallback

This is a visualization-only fallback from a recorded replay CSV. It is not an
Isaac camera render and not a carrying-success claim.

- Record status from strict checker: ${record_status}
- Replay CSV: ${replay_csv}
- Summary: ${record_summary}
- Strict checker summary: ${checker_summary}
- Key expected interpretation: short-window 0.75 kg boxtilt progress, with no
  fall/drop in the 760-step window but strict failures on lateral drift and tilt.
EOF

echo "record_status=${record_status}"
echo "record_summary=${record_summary}"
echo "checker_summary=${checker_summary}"
echo "replay_csv=${replay_csv}"
echo "visual_dir=${output_dir}"
