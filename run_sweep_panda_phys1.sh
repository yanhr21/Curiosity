#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
H="solver,iterations,ls_iters,substeps,impratio,kh,worlds,bodies,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen"
COMMON="--world-count 4 --warmup 40 --sample-frames 220 --device cuda:0 --scene pen"

OUT=results_panda_substeps.csv; echo "$H" > $OUT
for SS in 2 5 10 20 40; do
  CUDA_VISIBLE_DEVICES=0 python bench_panda_hydro.py --substeps $SS $COMMON 2>/dev/null | tail -1 >> $OUT
  echo "done substeps=$SS"
done

OUT=results_panda_impratio.csv; echo "$H" > $OUT
for IR in 1 10 100 1000 10000; do
  CUDA_VISIBLE_DEVICES=0 python bench_panda_hydro.py --impratio $IR $COMMON 2>/dev/null | tail -1 >> $OUT
  echo "done impratio=$IR"
done
echo "=== SUBSTEPS ==="; cat results_panda_substeps.csv; echo "=== IMPRATIO ==="; cat results_panda_impratio.csv
