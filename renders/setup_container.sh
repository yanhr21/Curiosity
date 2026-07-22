#!/bin/bash
# Reproducible setup for the stock CUDA container (oci-ord `cuda_docker_ver3.sqsh`).
#
# Why this exists: the container's root filesystem is EPHEMERAL — every fresh `srun`
# starts from the pristine image, so the system GL libs + the Xvfb X server (needed for
# software-GL headless rendering) are gone and must be reinstalled. conda/uv envs live on
# Lustre and persist; these OS packages do not, and conda-forge doesn't ship the Xvfb server.
#
# To make that fast AND network-independent (archive.ubuntu.com is flaky here), we cache the
# .deb packages on Lustre the first time apt succeeds, then install from that cache thereafter.
#
# Persistent state on Lustre (survives every container):
#   $TOOLS/bin/uv         - the uv launcher
#   $TOOLS/apt-debs/*.deb - cached GL/Xvfb packages
#   <repo>/.venv          - the Newton python env (built by `uv sync`, elsewhere)
set -u
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
TOOLS=$REPO/renders/_toolcache
DEBCACHE=$TOOLS/apt-debs
export HF_HUB_DISABLE_XET=1
export PATH="$TOOLS/bin:$HOME/.local/bin:$PATH"
mkdir -p "$TOOLS/bin" "$DEBCACHE"

PKGS="libegl1 libgles2 libglvnd0 libgl1-mesa-dri libglx-mesa0 xvfb \
      libxrender1 libxext6 libxi6 libxinerama1 libxcursor1 libxrandr2 libxxf86vm1 \
      x11-common xserver-common xfonts-base xfonts-encodings"

# 1. uv launcher (persist on Lustre; .venv itself already persists)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$TOOLS/bin:$PATH"
uv --version || { echo "SETUP_FAIL_UV"; exit 1; }

# 2. GL/Xvfb system libs — clear any stale dpkg lock from a prior apt first
pkill -9 apt apt-get dpkg 2>/dev/null
rm -f /var/lib/dpkg/lock* /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null
dpkg --configure -a >/dev/null 2>&1

if ls "$DEBCACHE"/*.deb >/dev/null 2>&1; then
  echo "[setup] installing GL/Xvfb from Lustre deb cache ($(ls "$DEBCACHE"/*.deb | wc -l) debs, offline)"
  dpkg -i "$DEBCACHE"/*.deb >/dev/null 2>&1
  dpkg --configure -a >/dev/null 2>&1
else
  echo "[setup] no deb cache yet — downloading via apt (one-time; retries flaky network)"
  for i in 1 2 3 4 5; do apt-get update -y && break; echo "  apt update retry $i"; sleep 8; done
  for i in 1 2 3 4 5; do
    apt-get install -y --no-install-recommends $PKGS && break
    echo "  apt install retry $i"; sleep 8
  done
  # seed the Lustre cache for every future container
  cp -n /var/cache/apt/archives/*.deb "$DEBCACHE"/ 2>/dev/null
  echo "[setup] cached $(ls "$DEBCACHE"/*.deb 2>/dev/null | wc -l) debs to $DEBCACHE"
fi

# 3. verify — actually START Xvfb (presence alone isn't enough: Xvfb dies at runtime if the
# xkb-data keymaps under /usr/share/X11/xkb are missing, which `command -v` won't catch).
command -v Xvfb >/dev/null || { echo "SETUP_FAIL_XVFB_MISSING"; exit 1; }
[ -d /usr/share/X11/xkb ] || echo "[setup] WARN /usr/share/X11/xkb missing — Xvfb will fail (need xkb-data deb)"
pkill -f "Xvfb :97" 2>/dev/null; rm -f /tmp/.X97-lock
Xvfb :97 -screen 0 640x480x24 -nolisten tcp >/tmp/xvfb97.log 2>&1 &
_xp=$!; sleep 3
if kill -0 $_xp 2>/dev/null; then echo "[setup] Xvfb starts OK"; kill $_xp 2>/dev/null; else
  echo "[setup] Xvfb FAILED to start:"; cat /tmp/xvfb97.log; echo "SETUP_FAIL_XVFB_START"; exit 1
fi
ldconfig -p 2>/dev/null | grep -q libGL.so && echo "[setup] libGL present" || echo "[setup] WARN libGL not in ldconfig"
echo "SETUP_CONTAINER_DONE"
