#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
OUT=results_panda_iters.csv
COMMON="--world-count 4 --warmup 40 --sample-frames 220 --device cuda:0 --scene pen --ls-iters 100"
echo "solver,iterations,ls_iters,worlds,bodies,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen" > $OUT
for IT in 1 2 5 10 15 30 60; do
  CUDA_VISIBLE_DEVICES=0 python bench_panda_hydro.py --iters $IT $COMMON 2>/dev/null | tail -1 >> $OUT
  echo "done iters=$IT"
done
echo "=== ITERS RESULTS ==="; cat $OUT
