#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
OUT=results_panda_ls.csv
COMMON="--world-count 4 --warmup 40 --sample-frames 220 --device cuda:0 --scene pen --iters 15"
echo "solver,iterations,ls_iters,worlds,bodies,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen" > $OUT
for LS in 5 10 25 50 100 200; do
  CUDA_VISIBLE_DEVICES=1 python bench_panda_hydro.py --ls-iters $LS $COMMON 2>/dev/null | tail -1 >> $OUT
  echo "done ls=$LS"
done
echo "=== LS RESULTS ==="; cat $OUT
