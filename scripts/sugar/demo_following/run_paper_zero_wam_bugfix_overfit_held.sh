#!/usr/bin/env bash
# First 32-update generative-overfit endpoint on the original PhysX corpus.
# Later continuation is evidence-driven; no formal/physics stage is launched.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
cd "$ROOT"
export PYTHONPATH="$EXPERIMENT/repair_python_deps:/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages:$ROOT"
export OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Do not let portability overrides silently select another checkpoint or VAE.
unset PZW_WAN_CHECKPOINT PZW_VAE_PATH PZW_ALLOW_NON_SLURM
export PZW_FFMPEG=/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
exec 9>"$EXPERIMENT/single_gpu_pipeline.lock"
flock -n 9
/usr/bin/python3.10 - <<'PY'
import json
from collections import Counter
from pathlib import Path
from scripts.sugar.demo_following.paper_zero_wam.config import repaired_overfit_config
config = repaired_overfit_config()
rows = [json.loads(line) for line in config.resolved(config.manifest).read_text().splitlines() if line]
expected = {('train', 'CarryBox'): 80, ('train', 'KickBox'): 80,
            ('validation', 'CarryBox'): 10, ('validation', 'KickBox'): 10,
            ('test', 'CarryBox'): 10, ('test', 'KickBox'): 9}
assert Counter((r['split'], r['task']) for r in rows) == expected
cache = config.resolved(config.latent_cache)
audit = json.loads((cache / 'LATENT_RESULT.json').read_text())
assert audit['passed'] and audit['trajectory_count'] == 199
assert audit['robot_latents_bitwise_unchanged_count'] == 199
assert audit['prompt_coverage_exact_count'] == 199 and audit['action_training_rows'] == 112000
assert audit['action_trace_file_count'] == 8
assert audit['action_trace_shape_environment_counts'] == {'700x24x29': 24, '700x25x29': 175}
fresh = json.loads(config.resolved(config.output_root).joinpath(
    'bugfix_audit_20260911/OFFICIAL_FORWARD_AUDIT.json').read_text())
assert fresh['passed'] and fresh['official_layers'] == 30
assert fresh['mask_forward_backward_passed'] and fresh['official_parameters'] == 4999787712
print(json.dumps({'original_PhysX_corpus_retained': True, 'trajectories': 199,
                  'fresh_official_forward_audit_passed': True, 'hash_checks': False}))
PY
exec /usr/bin/python3.10 \
    scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
    torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
    scripts.sugar.demo_following.paper_zero_wam.train_single_gpu \
    --mode overfit --fixed-noise-overfit-diagnostic --repaired-overfit \
    --resampled-noise-overfit --save-overfit-optimizer \
    --output-dir "$EXPERIMENT/overfit_resampled_noise_20260911"
