#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_usd_probe_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_main}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
USD_PATH="${USD_PATH:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_main_usd_v1_20260701_041900/panda_hydro.usd}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python" "$USD_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

summary_json="$OUTPUT_DIR/usd_scene_capability_probe.json"
summary_md="$REPORT_DIR/usd_scene_capability_probe.md"

export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"

"$NEWTON_VENV/bin/python" - "$summary_json" "$summary_md" "$RUN_TAG" "$NEWTON_ROOT" "$USD_PATH" <<'PY'
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
run_tag = sys.argv[3]
newton_root = Path(sys.argv[4])
usd_path = Path(sys.argv[5])

def import_status(name: str) -> dict:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "module": getattr(mod, "__file__", None)}

def cmd_status(name: str) -> dict:
    path = shutil.which(name)
    return {"available": path is not None, "path": path}

imports = {
    name: import_status(name)
    for name in [
        "pxr.Usd",
        "pxr.UsdGeom",
        "pxr.UsdAppUtils",
        "pxr.UsdImagingGL",
        "OpenGL.GL",
        "glfw",
        "PIL.Image",
        "imageio",
        "cv2",
        "av",
        "newton.viewer",
    ]
}

viewer_classes = {}
try:
    import newton.viewer as viewer

    for name in ["ViewerUSD", "ViewerGL", "ViewerRTX", "ViewerNull"]:
        viewer_classes[name] = hasattr(viewer, name)
except Exception as exc:  # noqa: BLE001
    viewer_classes["import_error"] = f"{type(exc).__name__}: {exc}"

stage_info = {
    "usd_path": str(usd_path),
    "exists": usd_path.exists(),
    "size_bytes": usd_path.stat().st_size if usd_path.exists() else 0,
}
if imports["pxr.Usd"]["ok"]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        stage_info["open_ok"] = False
    else:
        prims = [p for p in stage.Traverse()]
        stage_info.update(
            {
                "open_ok": True,
                "prim_count": len(prims),
                "start_time_code": stage.GetStartTimeCode(),
                "end_time_code": stage.GetEndTimeCode(),
                "time_codes_per_second": stage.GetTimeCodesPerSecond(),
            }
        )

newton_commit = subprocess.check_output(
    ["git", "-C", str(newton_root), "rev-parse", "HEAD"], text=True
).strip()

payload = {
    "classification": "phase00_usd_scene_capability_probe",
    "run_tag": run_tag,
    "not_training_result": True,
    "not_curiosity_success": True,
    "newton_root": str(newton_root),
    "newton_commit": newton_commit,
    "commands": {name: cmd_status(name) for name in ["usdrecord", "usdview", "usdcat", "ffmpeg"]},
    "imports": imports,
    "viewer_classes": viewer_classes,
    "stage": stage_info,
    "raster_path_status": "candidate" if imports["pxr.UsdImagingGL"]["ok"] else "blocked_missing_pxr_usdimaginggl",
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary_md.write_text(
    "# Phase 00 USD Scene Capability Probe\n\n"
    f"- run_tag: `{run_tag}`\n"
    f"- Newton commit: `{newton_commit}`\n"
    f"- USD: `{usd_path}`\n"
    f"- stage open ok: `{stage_info.get('open_ok')}`\n"
    f"- stage prim count: `{stage_info.get('prim_count')}`\n"
    f"- raster path status: `{payload['raster_path_status']}`\n"
    f"- usdrecord: `{payload['commands']['usdrecord']['path']}`\n"
    f"- usdview: `{payload['commands']['usdview']['path']}`\n"
    f"- ffmpeg: `{payload['commands']['ffmpeg']['path']}`\n\n"
    "Classification: capability probe only. This is not rendering success, "
    "training, or curiosity success.\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
