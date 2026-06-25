#!/bin/bash
source /home/shengzew/miniconda3/etc/profile.d/conda.sh && conda activate newton
cd /home/shengzew/Desktop/research/newton
OUT=results_fps_collision.csv
SCENE="--num-pyramids 2 --pyramid-size 8 --world-count 64 --substeps 10 --settle-frames 150 --timing-frames 150 --device cuda:0"
CUDA_VISIBLE_DEVICES=0 python bench_fps_vs_collision.py --header > $OUT
for IT in 1 2 4 8 16 32 64; do
  CUDA_VISIBLE_DEVICES=0 python bench_fps_vs_collision.py --solver xpbd --iterations $IT $SCENE 2>/dev/null | tail -1 >> $OUT
  echo "done xpbd iters=$IT"
done
echo "=== RESULTS ==="
cat $OUT
