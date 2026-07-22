#!/bin/bash
# Shared Xvfb + mesa-software-GL env prelude (source this before uv run ... example_g1_in_sage.py).
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
OUT=$REPO/renders
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json
export HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg PATH="$REPO/renders/_toolcache/bin:$HOME/.local/bin:$PATH"
pkill -f "Xvfb :99" 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
export XVFB_PID=$!; sleep 3
export DISPLAY=:99 G1_XVFB=1
# Software GL: the stock CUDA image ships libGL + swrast_dri.so, so DON'T force
# GALLIUM_DRIVER=llvmpipe (that driver isn't always present) — let mesa auto-pick swrast.
export LIBGL_ALWAYS_SOFTWARE=true __GLX_VENDOR_LIBRARY_NAME=mesa
export PYOPENGL_PLATFORM=glx MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450
unset __EGL_VENDOR_LIBRARY_FILENAMES PYGLET_HEADLESS EGL_PLATFORM
cd "$REPO"
