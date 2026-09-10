#!/usr/bin/env bash
# CPU controller on login host; evaluation is sent to the existing compute shell.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
PZW_SESSION=curiosity_pzw_h200_1d_20260908
PZW_PANE=%712
cd "$ROOT"
tmux has-session -t "$PZW_SESSION"
test "$(tmux list-panes -t "$PZW_SESSION" -F '#{pane_id}')" = "$PZW_PANE"
ssh -o BatchMode=yes -o ConnectTimeout=10 server52 \
    /usr/bin/python3.10 "$ROOT/scripts/sugar/demo_following/paper_zero_wam/stop_at_checkpoint.py" \
    --formal-root "$ROOT/experiments/demo_following/paper_zero_wam_v1/formal" \
    --trainer-pid 150590 --torchrun-pid 138146 --job-id 285508 --host server52
tmux has-session -t "$PZW_SESSION"
test "$(tmux list-panes -t "$PZW_SESSION" -F '#{pane_id}')" = "$PZW_PANE"
test "$(squeue -j 285508 -h -o '%T')" = RUNNING
tmux send-keys -t "$PZW_PANE" -l \
    'bash scripts/sugar/native_tactile/launch_retained_child.sh --foreground --record experiments/demo_following/paper_zero_wam_v1/logs/held_285508_eval_step700.record --status experiments/demo_following/paper_zero_wam_v1/logs/held_285508_eval_step700.status --log experiments/demo_following/paper_zero_wam_v1/logs/held_285508_eval_step700.log --tag paper_zero_wam_step700_interim_evaluation -- bash scripts/sugar/demo_following/run_paper_zero_wam_step700_eval_held.sh'
tmux send-keys -t "$PZW_PANE" Enter
printf 'Step-700 evaluation command delivered to retained compute pane %s.\n' "$PZW_PANE"
