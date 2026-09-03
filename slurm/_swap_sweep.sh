#!/bin/bash
# Throughput vs env count for the sugar_swap refiner, on one dedicated GPU.
set -u
source env/activate.sh || exit 2
export PYTHONUNBUFFERED=1
for N in 512 1024 2048 4096; do
    echo "===== SWEEP_N=$N"
    timeout -k 20 900 python -u -m sugar_swap.train \
        --num-envs "$N" --max-iterations 3 --save-interval 1000000 \
        --eval-minutes 100000 --logger tensorboard \
        --run-name "sweep_$N" --log-root logs/swap_sweep 2>&1 \
      | grep -E "Computation:|Iteration time|Total timesteps|out of memory|CUDA error|RuntimeError|torch.OutOfMemory"
    echo "===== SWEEP_N=$N rc=$?"
done
echo "===== SWEEP_DONE"
