#!/bin/bash
# Tactile field extraction on the dev node. Usage: bash renders/run_tactile.sh <tag> [extra args...]
set -u
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
SCENE=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/datasets/mike_data/robot_baby_data/_inspect/layout_84b703fb.json
TAG=${1:-run}; shift || true
export PATH="$REPO/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1
cd "$REPO"
uv run python tactile_field.py --scene "$SCENE" --out "renders/tactile_$TAG.npz" "$@"
