#!/usr/bin/env bash
# Recover only a pre-result infrastructure failure of the one fixed overfit run.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
ADMISSION="$ROOT/scripts/sugar/demo_following/advance_paper_zero_wam_after_heldout.sh"
RESULT="$EXPERIMENT/overfit/OVERFIT_RESULT.json"
RECOVERY="$EXPERIMENT/OVERFIT_RUNTIME_RECOVERY.json"
JOBS="$EXPERIMENT/JOBS.json"
mkdir -p "$EXPERIMENT/logs"

if [[ -f "$RESULT" ]]; then
    PYTHONPATH="$PZW_SITE_PACKAGES:$ROOT" "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
from scripts.sugar.demo_following.paper_zero_wam.config import PaperZeroWAMConfig
from scripts.sugar.demo_following.paper_zero_wam.train import overfit_execution_decision
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = PaperZeroWAMConfig()
execution = overfit_execution_decision(
    Path(sys.argv[3]), Path(sys.argv[4]), 32, config
)
if (result.get("protocol") != "paper_zero_wam_sugar_overfit_v1"
        or result.get("execution_completed") is not True
        or int(result.get("optimizer_steps", -1)) != 32
        or int(result.get("architecture_parameter_count", -1))
        != config.expected_parameter_count
        or result.get("batch") != {
            "world_size": 8,
            "packed_samples_per_rank": 1,
            "global_packed_samples": 8,
            "gradient_accumulation_steps": 1,
        }
        or result.get("hash_checks") is not False
        or execution.get("passed") is not True
        or result.get("execution_decision") != execution):
    raise SystemExit("invalid overfit terminal; refusing both suppression and replay")
path = Path(sys.argv[2])
previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
path.write_text(json.dumps({
    "protocol": "paper_zero_wam_overfit_runtime_recovery_v1",
    "recovery_required": False,
    "reason": "machine_readable_overfit_result_present",
    "overfit_passed": result.get("passed") is True,
    "infrastructure_retry_count": int(previous.get("infrastructure_retry_count", 0)),
    "additional_scientific_overfit_experiments": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RESULT" "$RECOVERY" "$EXPERIMENT/overfit/TRAIN_TRACE.jsonl" \
    "$EXPERIMENT/schedule/TRAIN_SCHEDULE.jsonl"
    exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/paper_zero_wam/runtime_recovery.py" \
    --jobs "$JOBS" --job-key "overfit_job" --log-prefix "overfit" \
    --output "${RECOVERY%.json}_INTERRUPTION_DECISION.json"

attempt=$(
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
print(int(value.get("infrastructure_retry_count", 0)) + 1)
' "$RECOVERY"
)
if (( attempt > 3 )); then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "protocol": "paper_zero_wam_overfit_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": True,
    "reason": "three_pre_result_infrastructure_failures",
    "infrastructure_retry_count": 3,
    "additional_scientific_overfit_experiments": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY"
    exit 1
fi

# The original dependents can never run after a failed dependency.  Cancel
# only their still-pending job IDs before creating the identical replacement
# chain; no completed or running computation is targeted.
mapfile -t stale_jobs < <(
    "$PYTHON_BIN" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "formal_recovery_job", "heldout_recovery_job", "render_recovery_job",
    "physical_admission_job", "render_job", "heldout_job", "formal_job",
):
    if value.get(key):
        print(value[key])
' "$JOBS"
)
for job in "${stale_jobs[@]}"; do
    state=$(squeue -h -j "$job" -o '%T' || true)
    if [[ "$state" == "PENDING" ]]; then
        scancel "$job"
    fi
done

overfit_job=$(sbatch --parsable \
    --no-requeue \
    --job-name=pzw_overfit \
    --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=16:00:00 \
    --output="$EXPERIMENT/logs/overfit_%j.out" \
    --error="$EXPERIMENT/logs/overfit_%j.err" \
    "$RUNNER" overfit)

formal_job=$(sbatch --parsable --dependency="afterok:$overfit_job" \
    --no-requeue \
    --job-name=pzw_formal \
    --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=4-00:00:00 \
    --output="$EXPERIMENT/logs/formal_%j.out" \
    --error="$EXPERIMENT/logs/formal_%j.err" \
    "$RUNNER" formal)

formal_recovery_job=$(sbatch --parsable --dependency="afterany:$formal_job" \
    --no-requeue \
    --job-name=pzw_formal_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/formal_recover_%j.out" \
    --error="$EXPERIMENT/logs/formal_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_formal.sh")

heldout_job=$(sbatch --parsable --dependency="afterok:$formal_job" \
    --no-requeue \
    --job-name=pzw_heldout \
    --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=1-00:00:00 \
    --output="$EXPERIMENT/logs/heldout_%j.out" \
    --error="$EXPERIMENT/logs/heldout_%j.err" \
    "$RUNNER" evaluate_heldout)

heldout_recovery_job=$(sbatch --parsable --dependency="afterany:$heldout_job" \
    --no-requeue \
    --job-name=pzw_heldout_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/heldout_recover_%j.out" \
    --error="$EXPERIMENT/logs/heldout_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_heldout.sh")

render_job=$(sbatch --parsable --dependency="afterok:$heldout_job" \
    --no-requeue \
    --job-name=pzw_render \
    --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=12:00:00 \
    --output="$EXPERIMENT/logs/render_%j.out" \
    --error="$EXPERIMENT/logs/render_%j.err" \
    "$RUNNER" render_openloop)

render_recovery_job=$(sbatch --parsable --dependency="afterany:$render_job" \
    --no-requeue \
    --job-name=pzw_render_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/render_recover_%j.out" \
    --error="$EXPERIMENT/logs/render_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_render.sh")

physical_admission_job=$(sbatch --parsable --dependency="afterok:$heldout_job" \
    --no-requeue \
    --job-name=pzw_physical_admit \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=01:00:00 \
    --output="$EXPERIMENT/logs/physical_admit_%j.out" \
    --error="$EXPERIMENT/logs/physical_admit_%j.err" \
    "$ADMISSION")

next_recovery_job=$(sbatch --parsable --dependency="afterany:$overfit_job" \
    --no-requeue \
    --job-name=pzw_overfit_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/overfit_recover_%j.out" \
    --error="$EXPERIMENT/logs/overfit_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_overfit.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
recovery_path, jobs_path = map(Path, sys.argv[1:3])
attempt = int(sys.argv[3])
(
    overfit, formal, formal_recovery, heldout, heldout_recovery,
    render, render_recovery, admission, next_recovery,
) = sys.argv[4:13]
recovery_path.write_text(json.dumps({
    "protocol": "paper_zero_wam_overfit_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": False,
    "reason": "previous_overfit_exited_without_machine_readable_result",
    "infrastructure_retry_count": attempt,
    "additional_scientific_overfit_experiments": 0,
    "replacement_overfit_job": overfit,
    "next_recovery_job": next_recovery,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
jobs.update({
    "overfit_job": overfit,
    "formal_job": formal,
    "formal_recovery_job": formal_recovery,
    "heldout_job": heldout,
    "heldout_recovery_job": heldout_recovery,
    "render_job": render,
    "render_recovery_job": render_recovery,
    "physical_admission_job": admission,
    "overfit_recovery_job": next_recovery,
})
jobs_path.write_text(json.dumps(jobs, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY" "$JOBS" "$attempt" "$overfit_job" "$formal_job" \
    "$formal_recovery_job" "$heldout_job" "$heldout_recovery_job" \
    "$render_job" "$render_recovery_job" "$physical_admission_job" \
    "$next_recovery_job"

printf 'replacement_overfit=%s formal=%s formal_recovery=%s heldout=%s heldout_recovery=%s render=%s render_recovery=%s admission=%s recovery=%s\n' \
    "$overfit_job" "$formal_job" "$formal_recovery_job" "$heldout_job" \
    "$heldout_recovery_job" "$render_job" "$render_recovery_job" \
    "$physical_admission_job" "$next_recovery_job"
