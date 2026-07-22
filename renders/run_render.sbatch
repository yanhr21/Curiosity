#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p backfill_singlenode
#SBATCH --gres=gpu:1
#SBATCH -t 120
#SBATCH --job-name=nw_render
#SBATCH -o /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/render_%j.log

# Build the Newton GPU env (uv) and render G1 walking in a SAGE-10k scene.
# Staged with MARK_* so the log shows exactly where it stops. Not `set -e`.
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
OUT=$REPO/renders
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json
mkdir -p "$OUT"
export HF_HUB_DISABLE_XET=1
export MPLBACKEND=Agg

echo "===== MARK_NODE ====="; hostname; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "===== MARK_NET ====="
curl -s -m 12 -o /dev/null -w "pypi=%{http_code}\n" https://pypi.org/simple/ || echo "NET_FAIL_pypi"
curl -s -m 12 -o /dev/null -w "hf=%{http_code}\n" https://huggingface.co || echo "NET_FAIL_hf"

echo "===== MARK_UV ====="
source /lustre/fsw/portfolios/nvr/users/shengzew/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate base 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3 || pip install -q uv
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv && uv --version

echo "===== MARK_UVSYNC ====="
cd "$REPO" || exit 2
uv sync --extra examples --extra torch-cu12 2>&1 | tail -40
echo "uv sync rc=${PIPESTATUS[0]}"

echo "===== MARK_SMOKE ====="
uv run python -c "import warp as wp; wp.init(); print('warp', wp.__version__, 'device', wp.get_device()); import newton; print('newton', newton.__version__); import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -20

echo "===== MARK_INGEST ====="
uv run python -m scene_ingest.adapters.sage "$SCENE" 2>&1 | tail -6

echo "===== MARK_RENDER ====="
uv run python example_g1_in_sage.py --scene "$SCENE" --record "$OUT/g1_in_sage.mp4" \
    --frames 250 --command 1.0,0.0,0.1 --spawn 2.5,3.5 2>&1 | tail -80
echo "render rc=${PIPESTATUS[0]}"

echo "===== MARK_DONE ====="; ls -la "$OUT"/*.mp4 2>/dev/null || echo "NO MP4"
