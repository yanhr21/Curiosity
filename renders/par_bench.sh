#!/bin/bash
# Run one bench configuration per GPU, in parallel, on the 8-GPU dev node.
#
# The scene is a single environment, so one config barely occupies an A100 -- eight of
# them fit side by side and a sweep that took an hour serially takes as long as its
# slowest row. Each line of the config file is a full argument string for
# sugar_newton.validation.allegro_bench.
#
#   bash renders/par_bench.sh renders/configs.txt
set -u
CUR=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
bash "$NT/renders/setup_container.sh" 2>&1 | tail -1
export PATH="$NT/renders/_toolcache/bin:$PATH" HF_HUB_DISABLE_XET=1 MPLBACKEND=Agg
export PYTHONPATH="$CUR:$NT"
cd "$NT"
nvidia-smi -L | sed 's/^/  /'
i=0
while IFS= read -r cfg; do
  case "$cfg" in ''|\#*) continue;; esac
  echo "GPU $i <- $cfg"
  CUDA_VISIBLE_DEVICES=$i nohup uv run python -m sugar_newton.validation.allegro_bench $cfg \
      > "renders/par_${i}.log" 2>&1 &
  i=$((i+1))
done < "$1"
wait
echo "=== all $i configurations done ==="
for j in $(seq 0 $((i-1))); do
  echo "--- GPU $j ---"
  grep -E "^ *[0-9]e[+]|cheapest|nothing reached|no configuration" "renders/par_${j}.log" || tail -3 "renders/par_${j}.log"
done
