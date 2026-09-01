#!/bin/bash
# Run g1_carrybox_policy configurations one at a time on a single GPU.
#
# The parallel runner is the right tool for fidelity sweeps, but not for timing: the
# observation is built in NumPy on the CPU every frame, so eight concurrent rollouts
# contend for cores and the fps column picks up a ~20 % spread that has nothing to do with
# the configuration. Anything quoted as a speed number should come from here.
#
#   bash sugar_newton/ser_policy.sh sugar_newton/sweeps/speed_top.txt 0
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
OUT=$CUR/sugar_newton/_ser
mkdir -p "$OUT"
export PATH="$NT/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
export PYTHONPATH="$CUR:$NT"
export CUDA_VISIBLE_DEVICES="${2:-0}"
cd "$NT" || exit 2

while IFS=$'\t' read -r label cfg; do
    case "$label" in ''|\#*) continue;; esac
    echo "=== $label :: $cfg"
    uv run python -m sugar_newton.validation.g1_carrybox_policy $cfg \
        > "$OUT/${label}.log" 2>&1
    grep -E "frames in|box lift|joint tracking|actuator saturation|graph captured" \
        "$OUT/${label}.log" | sed 's/^/    /'
done < "$1"
echo "=== serial sweep done ==="
