#!/bin/bash
# Plan 16 GPU validation + render, inside the OCI-ord CUDA container.
#
# Hydroelastic SDF is CUDA-only (wp.Volume.allocate_by_tiles, wp.Texture3D), so
# the friction SCALE path and the contact-surface channels can only be exercised
# here -- everything else also runs on the login-node CPU device.
#
# Usage, from a persistent allocation held in screen (see alloc.sh):
#   bash launch_gpu.sh
#
# Staged MARK_* so a truncated log still says how far it got.
set -u

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SN_OUT=$REPO/sugar_newton/_gpu_out
mkdir -p "$SN_OUT"

echo "===== MARK_NODE ====="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

echo "===== MARK_SETUP ====="
# The container root is ephemeral: every srun starts from the pristine image, so the GL/Xvfb
# libs must be reinstalled. They cache on Lustre. The conda env persists there already.
bash "$REPO/slurm/setup_container.sh" 2>&1 | tail -3
# shellcheck source=/dev/null
. "$REPO/env/activate.sh" || exit 2
cd "$REPO" || exit 2

echo "===== MARK_FRICTION_MATERIAL ====="
python -m sugar_newton.validation.friction 2>&1 | grep -v "load on device" | tail -14
echo "material_rc=${PIPESTATUS[0]}"

echo "===== MARK_FRICTION_HYDRO ====="
# The one that cannot run on CPU at all: hydroelastic is what allocates and
# fills rigid_contact_friction, the per-contact scale.
python -m sugar_newton.validation.friction --hydroelastic 2>&1 | grep -v "load on device" | tail -16
echo "hydro_rc=${PIPESTATUS[0]}"

echo "===== MARK_INCLINE ====="
python -m sugar_newton.validation.incline 2>&1 | grep -v "load on device" | tail -16
echo "incline_rc=${PIPESTATUS[0]}"

echo "===== MARK_PRESSURE ====="
# Channels 9-10. Needs the contact surface, so it is GPU-only like the scale half.
python -m sugar_newton.validation.pressure 2>&1 | grep -v "load on device" | tail -14
echo "pressure_rc=${PIPESTATUS[0]}"

echo "===== MARK_RENDER ====="
# Software mesa + Xvfb: the GPU runs the physics, mesa rasterizes the viewer.
# slurm/render_env.sh exports G1_XVFB=1, which selects a windowed GLX context --
# pyglet EGL headless fails here (no /dev/dri render node permission).
source "$REPO/slurm/render_env.sh"
cd "$REPO" || exit 2
echo "DISPLAY=$DISPLAY G1_XVFB=$G1_XVFB"
python -m sugar_newton.validation.render_friction \
    --out "$SN_OUT/friction_sweep" --frames 600 2>&1 | grep -v "load on device" | tail -20
echo "render_rc=${PIPESTATUS[0]}"
kill "${XVFB_PID:-0}" 2>/dev/null

echo "===== MARK_DONE ====="
echo "trace=$([ -f "$SN_OUT/friction_sweep/trace.npz" ] && echo yes || echo NO)"
echo "frames=$(ls "$SN_OUT"/friction_sweep/frames/f*.png 2>/dev/null | wc -l)"
echo "Composite + mp4 run on the LOGIN node (matplotlib + ffmpeg are not here):"
echo "  python -m sugar_newton.validation.compose_friction_video --run $SN_OUT/friction_sweep"
date
