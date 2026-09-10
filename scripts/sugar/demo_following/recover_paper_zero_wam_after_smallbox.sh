#!/usr/bin/env bash
# Resume only SMALLBOX array elements that exited before publishing a terminal result.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
RECOVERY="$EXPERIMENT/formal/SMALLBOX_RUNTIME_RECOVERY.json"
JOBS="$EXPERIMENT/JOBS.json"
PHYSICAL_ROOT="$EXPERIMENT/formal/smallbox_physical"
mkdir -p "$EXPERIMENT/logs" "$EXPERIMENT/formal"

missing=$(
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected_prompts = [
    {
        "split": "train",
        "task": task,
        "source_motion_id": source_motion_id,
        "latent_key": "prompt_latents",
    }
    for task, source_motion_id in (("CarryBox", 45), ("KickBox", 21))
]
missing = []
for index in range(5):
    path = root / f"profile_batch{index:02d}" / "BATCH_RESULT.json"
    if not path.is_file():
        missing.append(index)
        continue
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("protocol") != "paper_zero_wam_smallbox_physical_batch_v1"
            or int(value.get("profile_batch", -1)) != index
            or value.get("passed_execution_contract") is not True
            or int(value.get("checkpoint_step", -1)) != 4200
            or int(value.get("architecture_parameter_count", -1)) != 10658724829
            or value.get("hash_checks") is not False
            or int(value.get("adapted_rollout_count", -1)) != 8
            or int(value.get("released_endpoint_rollout_count", -1)) != 8
            or int(value.get("rollout_count", -1)) != 16
            or value.get("prompt_specs") != expected_prompts
            or value.get("video_frame_counts") != [131] * 4):
        raise SystemExit(f"invalid terminal SMALLBOX result: {path}")
print(",".join(map(str, missing)))
' "$PHYSICAL_ROOT"
)

if [[ -z "$missing" ]]; then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
path.write_text(json.dumps({
    "protocol": "paper_zero_wam_smallbox_runtime_recovery_v1",
    "recovery_required": False,
    "reason": "all_five_terminal_batch_results_present",
    "infrastructure_retry_count": int(previous.get("infrastructure_retry_count", 0)),
    "additional_scientific_rollouts": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY"
    exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/paper_zero_wam/runtime_recovery.py" \
    --jobs "$JOBS" --job-key "physical_array_job" --log-prefix "smallbox" \
    --indices "$missing" \
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
    "protocol": "paper_zero_wam_smallbox_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": True,
    "reason": "three_pre_result_infrastructure_failures",
    "missing_profile_batches": [int(v) for v in sys.argv[2].split(",")],
    "infrastructure_retry_count": 3,
    "additional_scientific_rollouts": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY" "$missing"
    exit 1
fi

mapfile -t stale_jobs < <(
    "$PYTHON_BIN" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("motion_admission_job", "physical_aggregate_job"):
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

physical_job=$(sbatch --parsable --array="$missing%1" --no-requeue \
    --job-name=pzw_smallbox --partition=gpu --nodes=1 --ntasks=1 \
    --cpus-per-task=64 --gres=gpu:NVIDIAH200:8 --time=1-00:00:00 \
    --output="$EXPERIMENT/logs/smallbox_%A_%a.out" \
    --error="$EXPERIMENT/logs/smallbox_%A_%a.err" \
    "$RUNNER" physical_smallbox)

aggregate_job=$(sbatch --parsable --dependency="afterok:$physical_job" \
    --no-requeue \
    --job-name=pzw_smallbox_result --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --mem=8G --time=01:00:00 \
    --output="$EXPERIMENT/logs/smallbox_result_%j.out" \
    --error="$EXPERIMENT/logs/smallbox_result_%j.err" \
    "$RUNNER" aggregate_smallbox)

motion_admission_job=$(sbatch --parsable --dependency="afterok:$aggregate_job" \
    --no-requeue \
    --job-name=pzw_motion_admit --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --mem=8G --time=01:00:00 \
    --output="$EXPERIMENT/logs/motion_admit_%j.out" \
    --error="$EXPERIMENT/logs/motion_admit_%j.err" \
    "$ROOT/scripts/sugar/demo_following/advance_paper_zero_wam_after_smallbox.sh")

next_recovery=$(sbatch --parsable --dependency="afterany:$physical_job" \
    --no-requeue \
    --job-name=pzw_smallbox_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/smallbox_recover_%j.out" \
    --error="$EXPERIMENT/logs/smallbox_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_smallbox.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
recovery, jobs = map(Path, sys.argv[1:3])
attempt = int(sys.argv[3])
missing = [int(value) for value in sys.argv[4].split(",")]
physical, aggregate, admission, guardian = sys.argv[5:9]
recovery.write_text(json.dumps({
    "protocol": "paper_zero_wam_smallbox_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": False,
    "reason": "array_exited_before_all_terminal_batch_results",
    "missing_profile_batches": missing,
    "infrastructure_retry_count": attempt,
    "additional_scientific_rollouts": 0,
    "replacement_array_job": physical,
    "next_recovery_job": guardian,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
payload = json.loads(jobs.read_text(encoding="utf-8"))
payload.update({
    "physical_array_job": physical,
    "physical_aggregate_job": aggregate,
    "motion_admission_job": admission,
    "smallbox_recovery_job": guardian,
})
jobs.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY" "$JOBS" "$attempt" "$missing" "$physical_job" "$aggregate_job" \
    "$motion_admission_job" "$next_recovery"

printf 'replacement_smallbox=%s missing=%s aggregate=%s motion_admission=%s recovery=%s\n' \
    "$physical_job" "$missing" "$aggregate_job" "$motion_admission_job" "$next_recovery"
