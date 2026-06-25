#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
H="solver,iterations,ls_iters,substeps,impratio,kh,worlds,bodies,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen"
COMMON="--world-count 4 --warmup 40 --sample-frames 220 --device cuda:0 --scene pen"
OUT=results_panda_kh.csv; echo "$H" > $OUT
for KH in 1e9 1e10 1e11 1e12; do
  CUDA_VISIBLE_DEVICES=1 python bench_panda_hydro.py --kh $KH $COMMON 2>/dev/null | tail -1 >> $OUT
  echo "done kh=$KH"
done
echo "=== KH ==="; cat results_panda_kh.csv
