#!/bin/bash
# Hardware-GL prelude: headless EGL on the NVIDIA driver, no Xvfb, no software raster.
#
# The container ships libEGL_nvidia.so.0 but only a MESA entry in the glvnd ICD directory
# (/usr/share/glvnd/egl_vendor.d/50_mesa.json), so EGL enumerates zero NVIDIA devices and
# every attempt to use it lands on swrast -- which is why the old prelude gave up and ran
# the whole viewer on the CPU through Xvfb. Adding the missing ICD file is the fix.
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
ICD=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ ! -f "$ICD" ]; then
  printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > "$ICD" 2>/dev/null \
    && echo "wrote $ICD" || echo "WARNING: could not write $ICD (not root?)"
fi
export HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg PATH="$REPO/renders/_toolcache/bin:$HOME/.local/bin:$PATH"
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$ICD"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MESA_GL_VERSION_OVERRIDE MESA_GLSL_VERSION_OVERRIDE
unset DISPLAY G1_XVFB __GLX_VENDOR_LIBRARY_NAME
cd "$REPO"
