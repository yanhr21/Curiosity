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
# On partition choice, because the obvious reasoning is wrong here. interactive_singlenode's
# QOS allows only ONE running job per user, so this chain queues behind any dev node you are
# also holding (reason: QOSMaxJobsPerUserLimit). batch_singlenode's QOS has no such cap --
# but measured 2026-09-01 it had 6 of 9 nodes DRAINED and 64 jobs pending, while
# interactive_singlenode had 1227 nodes and a 15-job queue. Pool size beats the per-user cap:
# the uncapped partition never schedules. So this stays on interactive_singlenode and you
# release the dev node to let training start. Override with `sbatch -p ...`.
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
# One leg reaches ~240 iterations at 512 worlds, so checkpoint far more often than that:
# the 4 h wall clock kills the job outright and only saved iterations survive.
SAVE=${RB_SAVE:-25}
LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
LOGGER=${RB_LOGGER:-wandb}
EXTRA=${RB_EXTRA:-}

# Credentials live OUTSIDE the repo on purpose. RB_WANDB_ENV points at a shell snippet that
# exports WANDB_API_KEY; it is sourced here, on the submitting node, so the key lands in the
# job environment and then in --export below. Inside the container $HOME is /root, so
# ~/.netrc is not the one on Lustre and cannot be relied on.
if [ "$LOGGER" = "wandb" ]; then
    for _we in "${RB_WANDB_ENV:-}" "$HOME/files/wandb.env" \
               /lustre/fsw/portfolios/nvr/users/"$USER"/files/wandb.env; do
        if [ -n "$_we" ] && [ -r "$_we" ]; then
            # shellcheck source=/dev/null
            set +H; . "$_we"; echo "sourced wandb credentials from $_we"; break
        fi
    done
    [ -n "${WANDB_API_KEY:-}" ] || echo "WARNING: no WANDB_API_KEY found; the run will abort"
fi
# rsl_rl reads the wandb ENTITY from WANDB_USERNAME, not WANDB_ENTITY
# (rsl_rl/utils/wandb_utils.py:33). Unset it lands the run in the personal entity.
export WANDB_USERNAME=${RB_WANDB_ENTITY:-nvr-amri}

echo "===== TRAIN_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "run=$RUN envs=$ENVS iters=$ITERS eval_min=$EVAL_MIN logger=$LOGGER"
echo "wandb entity=$WANDB_USERNAME key_len=${#WANDB_API_KEY}"

# WANDB_API_KEY is not in the container's environment by default, so it is passed explicitly.
# train_bcppo.py fails fast if it is missing rather than running for minutes unlogged.
srun --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL,WANDB_API_KEY="${WANDB_API_KEY:-}",WANDB_USERNAME="$WANDB_USERNAME" \
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
         --num-envs $ENVS --max-iterations $ITERS --save-interval $SAVE \
         --eval-minutes $EVAL_MIN --video-frames 400 \
         --run-name $RUN --log-root $LOGROOT --logger $LOGGER \
         \$RESUME_ARG $EXTRA"
echo "===== TRAIN_EXIT ====="; date
