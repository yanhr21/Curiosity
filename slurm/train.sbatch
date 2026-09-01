#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p backfill_singlenode
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH --job-name=rb_train
#SBATCH -o slurm/train_%j.log
#
# A BCPPO training run on Newton, sized to finish in one job.
#
#   cd <repo> && sbatch slurm/train.sbatch
#   RB_RUN=my_run RB_ENVS=256 sbatch slurm/train.sbatch
#
# **Auto-resumes** regardless of why it stopped: it picks the highest-numbered checkpoint in
# the run's log directory and continues from it. That makes preemption, a wall-clock kill and
# a manual resubmission all the same event, and it is what lets this run on a preemptible
# partition safely.
#
# On partition choice, because two plausible answers are both wrong here. The 3000-iteration
# curriculum is ~49 h at 512 worlds, and every *interactive* and *batch* GPU partition caps
# at 4 h, which would mean ~13 chained legs. Measured 2026-09-01:
#
#   interactive_singlenode  1227 nodes, 470 GPUs free -- but its QOS allows ONE running job
#                           per user, so the chain queues behind any dev node you hold
#                           (QOSMaxJobsPerUserLimit) no matter how idle the cluster is.
#   batch_singlenode        no per-user cap -- but 6 of 9 nodes DRAINED and 64 jobs pending,
#                           so it never schedules. Pool size beats the per-user cap.
#   polar3 / polar4         500-700 nodes, but >3000 jobs pending and still a 4 h cap.
#   backfill_singlenode     1227 nodes, 31 jobs pending, NO QOS cap, and 7 days of wall
#                           clock. The whole run fits in one job.
#
# So: backfill. The cost is preemption, and PreemptMode=REQUEUE means a preempted job is put
# back in the queue and restarted rather than lost -- which, with --save-interval 25 (~25
# min) and the auto-resume above, costs at most one checkpoint interval. -t is 3 days, not 7:
# long enough for ~49 h of training with margin, and a shorter request backfills sooner.
#
# Safe to queue on SEVERAL partitions at once. Which one frees first is unpredictable here, so
# the useful thing is to sit in all of their queues -- but two legs training the same run
# directory would interleave checkpoints and corrupt the run. A 60 s heartbeat file in the run
# directory prevents it: a starting leg that finds a heartbeat under 3 minutes old exits
# immediately. The check is a plain file mtime rather than a squeue lookup because squeue does
# not exist inside the container, and the heartbeat dies with the job whether it exits, is
# killed by the wall clock, or is preempted.
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
       LOGDIR=$LOGROOT/$RUN && mkdir -p \$LOGDIR && \
       HB=\$LOGDIR/.heartbeat && \
       if [ -f \$HB ] && [ \$(( \$(date +%s) - \$(cat \$HB 2>/dev/null || echo 0) )) -lt 180 ]; \
       then echo \"===== ANOTHER LEG IS LIVE (heartbeat \$(( \$(date +%s) - \$(cat \$HB) ))s \
old); exiting so the two do not share \$LOGDIR\"; exit 0; fi && \
       ( while true; do date +%s > \$HB; sleep 60; done ) & \
       HBPID=\$! && trap \"kill \$HBPID 2>/dev/null; rm -f \$HB\" EXIT && \
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
