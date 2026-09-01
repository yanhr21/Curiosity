#!/bin/bash
# Xvfb + software-GL prelude for the viewer paths that cannot use headless EGL. SOURCE this.
#
#   source env/activate.sh && source slurm/render_env.sh
#
# Prefer plain EGL (just env/activate.sh) where it works -- it renders on the GPU. This file
# is the fallback for viewers that need a real GLX context: pyglet's EGL backend fails on
# these nodes for want of /dev/dri render-node permission, so the frames are rasterised on
# the CPU by mesa while the GPU still runs the physics.
#
# Leaves XVFB_PID set so the caller can kill the server afterwards.
_RB_SLURM_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RB_REPO=${RB_REPO:-$(dirname -- "$_RB_SLURM_DIR")}
# Only used by the older newton-side examples; overridable and not required by sugar_newton.
export RB_SCENE=${RB_SCENE:-/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json}

export HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
pkill -f "Xvfb :99" 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
export XVFB_PID=$!; sleep 3
export DISPLAY=:99 G1_XVFB=1
# Software GL. Do NOT force GALLIUM_DRIVER=llvmpipe: the stock CUDA image ships libGL and
# swrast_dri.so but not always llvmpipe, so let mesa auto-pick swrast.
export LIBGL_ALWAYS_SOFTWARE=true __GLX_VENDOR_LIBRARY_NAME=mesa
export PYOPENGL_PLATFORM=glx MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450
unset __EGL_VENDOR_LIBRARY_FILENAMES PYGLET_HEADLESS EGL_PLATFORM
