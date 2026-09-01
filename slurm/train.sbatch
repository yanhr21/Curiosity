#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --job-name=rb_train
#SBATCH -o slurm/train_%j.log
#
# One 4-hour leg of a BCPPO training run on Newton.
#
#   cd <repo> && sbatch slurm/train.sbatch
#   RB_RUN=my_run RB_ENVS=256 sbatch slurm/train.sbatch
#
# Every GPU partition here caps at 4 h, and the BCPPO curriculum needs ~3000 iterations, so
# a full run is many legs. This script therefore **auto-resumes**: it picks the
# highest-numbered checkpoint in the run's log directory and continues from it, so
# resubmitting the same command advances the same run instead of restarting it. Submit the
# next leg whenever the previous one ends -- or chain with `--dependency=afterany:<jobid>`.
#
# Submit from the repo root: -o above is relative to it, and the repo is found via
# SLURM_SUBMIT_DIR (Slurm copies the batch script to its spool dir, so $0 is useless).
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

RUN=${RB_RUN:-carrybox_bcppo}
ENVS=${RB_ENVS:-512}
ITERS=${RB_ITERS:-3000}
EVAL_MIN=${RB_EVAL_MIN:-10}
LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
LOGGER=${RB_LOGGER:-wandb}
EXTRA=${RB_EXTRA:-}

echo "===== TRAIN_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "run=$RUN envs=$ENVS iters=$ITERS eval_min=$EVAL_MIN logger=$LOGGER"

# WANDB_API_KEY is not in the container's environment by default and ~ is /root inside it,
# so the key is passed through explicitly. train_bcppo.py fails fast if it is missing.
srun --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL,WANDB_API_KEY="${WANDB_API_KEY:-}" \
     bash -lc "set -u; cd $REPO && \
       echo '===== TRAIN_SETUP =====' && bash slurm/setup_container.sh 2>&1 | tail -4 && \
       source env/activate.sh && source slurm/render_env_egl.sh && \
       LOGDIR=$LOGROOT/$RUN && \
       RESUME=\$(ls -1 \$LOGDIR/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' \
                 | sort -n | tail -1) && \
       if [ -n \"\$RESUME\" ]; then \
           echo \"===== RESUMING from iteration \$RESUME\" ; \
           RESUME_ARG=\"--resume \$LOGDIR/model_\$RESUME.pt\" ; \
       else \
           echo '===== FRESH START' ; RESUME_ARG='' ; \
       fi && \
       echo '===== TRAIN_START =====' && \
       python -u -m sugar_newton.rl.train_bcppo \
         --num-envs $ENVS --max-iterations $ITERS \
         --eval-minutes $EVAL_MIN --video-frames 400 \
         --run-name $RUN --log-root $LOGROOT --logger $LOGGER \
         \$RESUME_ARG $EXTRA"
echo "===== TRAIN_EXIT ====="; date
