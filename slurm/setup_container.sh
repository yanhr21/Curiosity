#!/bin/bash
# Per-container OS setup: the GL stack and the Xvfb X server.
#
# Why this exists: the container's root filesystem is EPHEMERAL. Every fresh `srun` starts
# from the pristine image, so system GL libs and Xvfb are gone and must be reinstalled. The
# conda env does not help here -- it lives on Lustre and persists, but conda-forge does not
# ship the Xvfb *server*, and libEGL_nvidia has to come from the image.
#
# To make that fast AND network-independent (archive.ubuntu.com is flaky from these nodes),
# the .deb files are cached on Lustre the first time apt succeeds and installed from that
# cache thereafter.
#
# Persistent state on Lustre, surviving every container:
#   slurm/_toolcache/apt-debs/*.deb   cached GL/Xvfb packages
#
# Moved here from third_party/newton/renders/setup_container.sh: it is project
# infrastructure, not part of Newton, and keeping it in the submodule made the branch
# depend on a sibling checkout.
set -u
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEBCACHE=$REPO/slurm/_toolcache/apt-debs
mkdir -p "$DEBCACHE"

PKGS="libegl1 libgles2 libglvnd0 libgl1-mesa-dri libglx-mesa0 xvfb \
      libxrender1 libxext6 libxi6 libxinerama1 libxcursor1 libxrandr2 libxxf86vm1 \
      x11-common xserver-common xfonts-base xfonts-encodings"

# 1. GL/Xvfb system libs -- clear any stale dpkg lock from a prior apt first
pkill -9 apt apt-get dpkg 2>/dev/null
rm -f /var/lib/dpkg/lock* /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null
dpkg --configure -a >/dev/null 2>&1

if ls "$DEBCACHE"/*.deb >/dev/null 2>&1; then
    echo "[setup] installing GL/Xvfb from Lustre deb cache ($(ls "$DEBCACHE"/*.deb | wc -l) debs, offline)"
    dpkg -i "$DEBCACHE"/*.deb >/dev/null 2>&1
    dpkg --configure -a >/dev/null 2>&1
else
    echo "[setup] no deb cache yet -- downloading via apt (one-time; retries flaky network)"
    for i in 1 2 3 4 5; do apt-get update -y && break; echo "  apt update retry $i"; sleep 8; done
    for i in 1 2 3 4 5; do
        apt-get install -y --no-install-recommends $PKGS && break
        echo "  apt install retry $i"; sleep 8
    done
    cp -n /var/cache/apt/archives/*.deb "$DEBCACHE"/ 2>/dev/null
    echo "[setup] cached $(ls "$DEBCACHE"/*.deb 2>/dev/null | wc -l) debs to $DEBCACHE"
fi

# 2. verify -- actually START Xvfb. Presence alone is not enough: Xvfb dies at runtime if
# the xkb keymaps under /usr/share/X11/xkb are missing, which `command -v` will not catch.
command -v Xvfb >/dev/null || { echo "SETUP_FAIL_XVFB_MISSING"; exit 1; }
[ -d /usr/share/X11/xkb ] || echo "[setup] WARN /usr/share/X11/xkb missing -- Xvfb will fail (need xkb-data deb)"
pkill -f "Xvfb :97" 2>/dev/null; rm -f /tmp/.X97-lock
Xvfb :97 -screen 0 640x480x24 -nolisten tcp >/tmp/xvfb97.log 2>&1 &
_xp=$!; sleep 3
if kill -0 $_xp 2>/dev/null; then echo "[setup] Xvfb starts OK"; kill $_xp 2>/dev/null; else
    echo "[setup] Xvfb FAILED to start:"; cat /tmp/xvfb97.log; echo "SETUP_FAIL_XVFB_START"; exit 1
fi
ldconfig -p 2>/dev/null | grep -q libGL.so && echo "[setup] libGL present" || echo "[setup] WARN libGL not in ldconfig"

# 3. Headless hardware GL. The image ships libEGL_nvidia.so.0 but has no NVIDIA entry in
# the glvnd ICD directory, so EGL enumerates zero NVIDIA devices and every viewer silently
# falls back to software rasterisation -- the A100 sits idle while the CPU draws frames.
# One file fixes it.
ICD=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ ! -f "$ICD" ]; then
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > "$ICD" 2>/dev/null && echo "wrote $ICD"
fi

echo "SETUP_CONTAINER_DONE"
