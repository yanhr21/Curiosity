#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 replay render on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
EXPERIENCE="${EXPERIENCE:-}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/name=local_default --/exts/omni.kit.registry.nucleus/registries/0/url=file://${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/name=local_sdk --/exts/omni.kit.registry.nucleus/registries/1/url=file://${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${ISAAC_VENV}/lib/python3.12/site-packages/isaacsim}"
KIT_EXT_FOLDERS="${KIT_EXT_FOLDERS:-${ISAAC_SIM_ROOT}/exts:${ISAAC_SIM_ROOT}/extscache:${ISAAC_SIM_ROOT}/extsPhysics:${ISAAC_SIM_ROOT}/extsDeprecated:${ISAAC_SIM_ROOT}/kit/exts:${ISAAC_SIM_ROOT}/kit/extscore:${ISAAC_SIM_ROOT}/isaacsim/exts:${ISAAC_SIM_ROOT}/isaacsim/extscache:${ISAAC_SIM_ROOT}/isaacsim/extsPhysics:${ISAAC_SIM_ROOT}/isaacsim/extsDeprecated}"

IFS=':' read -r -a kit_ext_folder_array <<< "${KIT_EXT_FOLDERS}"
for ext_folder in "${kit_ext_folder_array[@]}"; do
  if [[ -d "${ext_folder}" ]]; then
    KIT_ARGS="${KIT_ARGS} --ext-folder=${ext_folder}"
  fi
done

REPLAY_CSV="${REPLAY_CSV:?Set REPLAY_CSV to core_world_g1_box_scene_replay.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/visuals/g1_replay_showcase/$(date +%Y%m%d_%H%M%S)}"
CAPTURE_EVERY_N_ROWS="${CAPTURE_EVERY_N_ROWS:-1}"
MAX_FRAMES="${MAX_FRAMES:--1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
DEVICE="${DEVICE:-cpu}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python3 -m py_compile scripts/isaac/render_core_world_g1_replay_showcase.py
python3 -m py_compile scripts/isaac/check_core_world_g1_replay_showcase.py

render_args=(
  --viz none \
  --enable_cameras \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --replay-csv "${REPLAY_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --resolution "${WIDTH}" "${HEIGHT}" \
  --capture-every-n-rows "${CAPTURE_EVERY_N_ROWS}" \
  --max-frames "${MAX_FRAMES}" \
  --follow-frame \
  --frame-zoom "${FRAME_ZOOM:-0.42}"
)
if [[ -n "${EXPERIENCE}" ]]; then
  render_args+=(--experience "${EXPERIENCE}")
fi

"${ISAAC_VENV}/bin/python" scripts/isaac/render_core_world_g1_replay_showcase.py "${render_args[@]}"

frame_dir="${OUTPUT_DIR}/rgb_frames"
movie_path="${OUTPUT_DIR}/g1_replay_showcase.mp4"
annotated_movie_path="${OUTPUT_DIR}/g1_replay_showcase_annotated.mp4"
if command -v ffmpeg >/dev/null 2>&1 && find "${frame_dir}" -type f -name '*.png' | grep -q .; then
  list_file="${OUTPUT_DIR}/g1_replay_frames.txt"
  find "${frame_dir}" -type f -name '*.png' | sort | awk '{print "file " q $0 q; print "duration 0.066"}' q="'" > "${list_file}"
  ffmpeg -y -hide_banner -loglevel warning -f concat -safe 0 -i "${list_file}" -pix_fmt yuv420p "${movie_path}"
  if ffmpeg -y -hide_banner -loglevel warning -i "${movie_path}" \
    -vf "drawbox=x=18:y=18:w=800:h=90:color=black@0.45:t=fill,drawtext=x=36:y=34:fontcolor=white:fontsize=28:text='G1 low-carry replay visualization',drawtext=x=36:y=68:fontcolor=white:fontsize=20:text='rendered replay of recorded pass, not new control evidence'" \
    -pix_fmt yuv420p "${annotated_movie_path}"; then
    echo "[INFO] Annotated replay video written to: ${annotated_movie_path}"
  fi
  echo "[INFO] Replay video written to: ${movie_path}"
else
  echo "[WARN] ffmpeg unavailable or no frames found; trying imageio MP4 fallback from: ${frame_dir}" >&2
  "${ISAAC_VENV}/bin/python" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys

try:
    import imageio.v2 as imageio
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] imageio unavailable, leaving PNG frames only: {exc}")
    raise SystemExit(0)

output_dir = Path(sys.argv[1])
frame_dir = output_dir / "rgb_frames"
frames = sorted(frame_dir.glob("*.png"))
if not frames:
    print(f"[WARN] no PNG frames found for imageio MP4 fallback in {frame_dir}")
    raise SystemExit(0)

movie_path = output_dir / "g1_replay_showcase.mp4"
annotated_movie_path = output_dir / "g1_replay_showcase_annotated.mp4"
for path in (movie_path, annotated_movie_path):
    with imageio.get_writer(str(path), fps=15, macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    print(f"[INFO] imageio wrote {path}")
PY
fi

record_dir="$(dirname "${REPLAY_CSV}")"
python3 scripts/isaac/check_core_world_g1_replay_showcase.py \
  --record-dir "${record_dir}" \
  --render-dir "${OUTPUT_DIR}" \
  --min-replay-rows "${MIN_REPLAY_ROWS:-20}" \
  --min-frames "${MIN_RENDER_FRAMES:-10}" \
  --expected-width "${WIDTH}" \
  --expected-height "${HEIGHT}" \
  --output "${OUTPUT_DIR}/g1_replay_showcase_check.json"
