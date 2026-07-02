# Newton 8c501 Compute Sanity Handoff

Date: 2026-07-01

Classification: handoff plan only. No benchmark, tactile export, render,
training, or Gate review is executed by this file.

## Source

- Newton source path: `external/newton_8c501`
- Commit: `8c501b47847569fecdda97a9f7f01205c6f7964f`
- Required runtime environment: prebuilt `envs/newton/.venv`
- Required execution: Curiosity-owned tmux-held H200 Slurm allocation

## Step 1: Runtime Benchmark

```bash
JOB_ID=<RUNNING_H200_JOB> \
WINDOW_NAME=p00_bench_8c501 \
RUN_TAG=p00_bench_8c501_hot_v1_20260701 \
NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_8c501 \
NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv \
SCENE=cube WORLD_COUNT=1 NUM_FRAMES=2500 BENCHMARK_SECONDS=30 \
bash experiments/configs/phase00/ref_tactile/launch_newton_hydro_benchmark_tmux.sh
```

Success condition: generated summary status is `pass` and runtime is around
the accepted `80 FPS` continuation threshold. `82 FPS` is historical reference
only and must not block downstream tactile export.

## Step 2: Dense Tactile Export

Run after Step 1 executes successfully around 80 FPS.

```bash
JOB_ID=<RUNNING_H200_JOB> \
WINDOW_NAME=p00_mjw_8c501 \
RUN_TAG=p00_mjw_8c501_marker_v1_20260701 \
NEWTON_ROOT=/public/home/yanhongru/Curiosity/external/newton_8c501 \
NEWTON_VENV=/public/home/yanhongru/Curiosity/envs/newton/.venv \
NUM_FRAMES=240 MAP_SIZE=32 FPS=30 \
MATERIAL_LABEL=steel_candidate OVERRIDE_MU=0.3 OVERRIDE_KH=1000000000000 \
SCENE_CAMERA=1 \
bash experiments/configs/phase00/ref_tactile/launch_mjw_direct_tactile_export_tmux.sh
```

Expected outputs:

- `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile_summary.json`
- `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile.avi`

## Step 3: Reference Compare

```bash
JOB_ID=<RUNNING_H200_JOB> \
WINDOW_NAME=p00_refcmp_8c501 \
RUN_TAG=p00_refcmp_8c501_marker_v1_20260701 \
CANDIDATE_VIDEO=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile.avi \
CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile_summary.json \
bash experiments/configs/phase00/ref_tactile/launch_reference_video_compare_tmux.sh
```

## Step 4: Channel Audit

```bash
JOB_ID=<RUNNING_H200_JOB> \
WINDOW_NAME=p00_chan_8c501 \
RUN_TAG=p00_chan_8c501_marker_v1_20260701 \
CANDIDATE_VIDEO=/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile.avi \
CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile_summary.json \
bash experiments/configs/phase00/ref_tactile/launch_channel_semantic_audit_tmux.sh
```

## Step 5: Gate Review

Run only after Steps 1-4 have output summaries.

```bash
JOB_ID=<RUNNING_H200_JOB> \
WINDOW_NAME=p00_gate_8c501 \
RUN_TAG=p00_gate_8c501_marker_v1_20260701 \
BENCHMARK_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_8c501_hot_v1_20260701/newton_hydro_benchmark_summary.json \
CANDIDATE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_marker_v1_20260701/candidate_mjw_direct_tactile_summary.json \
REFERENCE_COMPARE_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_marker_v1_20260701/reference_video_compare_summary.json \
CHANNEL_AUDIT_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_8c501_marker_v1_20260701/channel_semantic_audit_summary.json \
REFERENCE_ENV_AVAILABILITY_SUMMARY=/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json \
REFERENCE_ASSET_AVAILABILITY_SUMMARY=/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json \
REFERENCE_ASSET_REUSE_PLAN=/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json \
bash experiments/configs/phase00/ref_tactile/launch_phase00_gate_review_tmux.sh
```

## Boundary

These commands are not a curiosity-training path. They only determine whether
latest Newton `8c501...` can replace the previous d58 evidence chain. Gate 00F
still requires official UniVTAC/TaCauchy sanity or an accepted faithful
blocker.
