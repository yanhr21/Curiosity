#!/bin/bash
# Create (or update) the conda env described by env/environment.yml.
#
#   bash env/setup_env.sh              # create, or update if it already exists
#   RB_ENV_NAME=rb_test bash env/setup_env.sh
#
# Run this on the LOGIN node. It needs outbound network for pip, the container root is
# ephemeral so anything installed there is lost, and a conda env on Lustre is visible from
# inside the container because slurm/devnode.sbatch mounts /lustre:/lustre.
#
# Expect ~10 GB and a long install: torch's cu128 wheels plus the nvidia-* CUDA runtime
# libraries dominate.
set -u
ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname -- "$ENV_DIR")
NAME=${RB_ENV_NAME:-robotbaby}

if ! command -v conda >/dev/null 2>&1; then
    for c in "${RB_CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" \
             /lustre/fsw/portfolios/nvr/users/"$USER"/miniconda3; do
        [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ] && { . "$c/etc/profile.d/conda.sh"; break; }
    done
fi
command -v conda >/dev/null 2>&1 || { echo "no conda found; set RB_CONDA_BASE"; exit 2; }

# The submodule carries newton itself, which activate.sh puts on PYTHONPATH rather than
# installing -- but if it is missing the env is useless, so fail here rather than at import.
if [ ! -d "$REPO/third_party/newton/newton" ]; then
    echo "third_party/newton is empty -- initialising the submodule first"
    git -C "$REPO" submodule update --init --recursive third_party/newton || exit 2
fi

if conda env list | awk '{print $1}' | grep -qx "$NAME"; then
    echo "===== updating existing env '$NAME'"
    conda env update -n "$NAME" -f "$ENV_DIR/environment.yml" --prune || exit 2
else
    echo "===== creating env '$NAME'"
    # environment.yml carries the name; -n overrides it so RB_ENV_NAME works.
    conda env create -n "$NAME" -f "$ENV_DIR/environment.yml" || exit 2
fi

echo "===== verifying"
# Import through activate.sh so this checks the same PYTHONPATH the runners will use, and
# would catch a stale env that resolves newton from somewhere outside the repo.
# shellcheck source=/dev/null
RB_ENV_NAME=$NAME . "$ENV_DIR/activate.sh" || exit 2
python - <<'PY'
import importlib, sys
print("python  ", sys.version.split()[0], sys.executable)
for m in ("warp", "newton", "mujoco_warp", "torch", "open3d", "pyglet", "scipy",
          "trimesh", "matplotlib", "imageio", "rsl_rl"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?'):20s} {getattr(mod, '__file__', '')}")
    except Exception as e:
        print(f"  {m:12s} FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
PY
echo "===== env '$NAME' ready; use: source env/activate.sh"
