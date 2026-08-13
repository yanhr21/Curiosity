#!/usr/bin/env bash
# Start Isaac Sim entirely from a node-local Python/runtime copy.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside a retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NODE_LOCAL_ROOT" >&2
  exit 2
fi

RUNTIME_ROOT="$1"
PYTHON_BIN="$RUNTIME_ROOT/env/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing node-local Python: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/kit-portable"
env -u VK_ICD_FILENAMES \
  DISPLAY= \
  PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache" \
  "$PYTHON_BIN" -u -c \
  'from isaacsim import SimulationApp; app=SimulationApp({"headless": True, "multi_gpu": False}); print("NODE_LOCAL_ISAAC_STARTED", flush=True); [app.update() for _ in range(5)]; app.close(); print("NODE_LOCAL_ISAAC_CLOSED", flush=True)' \
  --portable-root "$RUNTIME_ROOT/kit-portable" \
  --/renderer/multiGpu/enabled=false \
  --/renderer/multiGpu/autoEnable=false \
  --/renderer/multiGpu/maxGpuCount=1
