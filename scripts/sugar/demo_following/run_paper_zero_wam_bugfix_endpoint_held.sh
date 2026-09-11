#!/usr/bin/env bash
# Follow the completed 32-update endpoint with frozen controls and, when
# sampled actions fail their baselines, a privileged inverse-dynamics probe.
# This never trains, regenerates the original predictions, or launches physics.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
ENDPOINT="$EXPERIMENT/overfit_resampled_noise_20260911"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
cd "$ROOT"
export PYTHONPATH="$EXPERIMENT/repair_python_deps:/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages:$ROOT"
export OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PZW_WAN_CHECKPOINT PZW_VAE_PATH PZW_ALLOW_NON_SLURM
exec 9>"$EXPERIMENT/single_gpu_pipeline.lock"
flock -n 9
/usr/bin/python3.10 - "$ENDPOINT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
endpoint = json.loads((root / 'OVERFIT_RESULT.json').read_text())
workflow = json.loads((root / 'DIAGNOSTIC_RESULT.json').read_text())
rendered = json.loads((root / 'training_videos/RENDER_RESULT.json').read_text())
assert endpoint['execution_completed'] and endpoint['optimizer_steps'] == 32
assert workflow['requested_workflow_completed']
assert rendered['execution_completed'] and len(rendered['cases']) == 8
assert sorted(case['slot'] for case in rendered['cases']) == list(range(8))
assert all(case['target_split'] == 'train' and case['render_frame_count'] == 8
           for case in rendered['cases'])
PY
/usr/bin/python3.10 \
    scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
    scripts.sugar.demo_following.paper_zero_wam.probe_overfit_inverse_dynamics \
    --endpoint "$ENDPOINT" --output "$ENDPOINT/ACTION_RECONSTRUCTION.json" \
    --action-readback-only
action_probe_needed=$(/usr/bin/python3.10 - "$ENDPOINT/ACTION_RECONSTRUCTION.json" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
print('yes' if result['inverse_dynamics_probe_needed'] else 'no')
PY
)
/usr/bin/python3.10 \
    scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
    scripts.sugar.demo_following.paper_zero_wam.probe_overfit_inverse_dynamics \
    --endpoint "$ENDPOINT" --output "$ENDPOINT/FROZEN_VAE_ALL_FRAMES.json" \
    --vae-controls-only
if [[ "$action_probe_needed" == yes ]]; then
    /usr/bin/python3.10 \
        scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
        scripts.sugar.demo_following.paper_zero_wam.probe_overfit_inverse_dynamics \
        --endpoint "$ENDPOINT" --output "$ENDPOINT/INVERSE_DYNAMICS_PROBE.json"
else
    printf '%s\n' '{"inverse_dynamics_probe_skipped":"all_eight_sampled_actions_pass_baseline_and_temporal_readbacks","overall_goal_complete":false}'
fi
