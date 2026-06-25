#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
H="solver,iterations,ls_iters,substeps,impratio,kh,worlds,bodies,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen,lift_best_mm,lift_worst_mm,blowup"
B="--world-count 4 --warmup 40 --sample-frames 220 --device cuda:0 --scene pen"
OUT=results_kh_stress.csv; echo "$H" > $OUT
run() { CUDA_VISIBLE_DEVICES=1 python bench_panda_hydro.py "$@" $B 2>/dev/null | tail -1 >> $OUT; echo "done: $*"; }
# conditioning: does high kh still work with few solver iterations?
run --kh 1e11 --iters 2
run --kh 1e13 --iters 2
run --kh 1e11 --iters 5
run --kh 1e13 --iters 5
# stiffness x large timestep (stability coupling)
run --kh 1e11 --substeps 2
run --kh 1e14 --substeps 2
run --kh 1e15 --substeps 2
echo "=== KH STRESS ==="; cat $OUT
