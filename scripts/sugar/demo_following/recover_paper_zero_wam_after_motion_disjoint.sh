#!/usr/bin/env bash
# Resume only held-out source jobs that exited before publishing a terminal result.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
PHYSICAL_ROOT="$EXPERIMENT/formal/motion_disjoint_physical"
RECOVERY="$EXPERIMENT/formal/MOTION_DISJOINT_RUNTIME_RECOVERY.json"
JOBS="$EXPERIMENT/JOBS.json"
mkdir -p "$EXPERIMENT/logs" "$EXPERIMENT/formal"

missing=$(
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
with manifest.open("r", encoding="utf-8") as stream:
    manifest_rows = [json.loads(line) for line in stream if line.strip()]
rows = sorted(
    (row for row in manifest_rows if row["split"] == "test"),
    key=lambda row: (row["task"], int(row["source_motion_id"])),
)
root = Path(sys.argv[2])
case_manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if (case_manifest.get("protocol") != "paper_zero_wam_motion_disjoint_cases_v1"
        or case_manifest.get("frozen_before_model_outcomes") is not True
        or case_manifest.get("conditions")
        != ["matched", "reversed", "same_task_alternate", "wrong_task"]
        or case_manifest.get("hash_checks") is not False
        or len(case_manifest.get("cases", [])) != 19):
    raise SystemExit("invalid frozen motion-disjoint case manifest")
missing = []
for index, row in enumerate(rows):
    directory = root / "source{:02d}_{}_{:03d}".format(
        index, row["task"], int(row["source_motion_id"])
    )
    path = directory / "SOURCE_RESULT.json"
    if not path.is_file():
        missing.append(index)
        continue
    value = json.loads(path.read_text(encoding="utf-8"))
    case = case_manifest["cases"][index]
    expected_prompts = case.get("prompts")
    if (
        int(case.get("source_index", -1)) != index
        or case.get("split") != "test"
        or case.get("target_task") != row["task"]
        or int(case.get("source_motion_id", -1)) != int(row["source_motion_id"])
        or not isinstance(expected_prompts, list)
        or len(expected_prompts) != 4
    ):
        raise SystemExit(f"frozen case/source mismatch at index {index}")
    if (
        value.get("protocol") != "paper_zero_wam_motion_disjoint_physical_source_v1"
        or int(value.get("source_index", -1)) != index
        or value.get("target_task") != row["task"]
        or int(value.get("source_motion_id", -1)) != int(row["source_motion_id"])
        or int(value.get("adapted_rollout_count", -1)) != 40
        or int(value.get("released_endpoint_rollout_count", -1)) != 20
        or int(value.get("rollout_count", -1)) != 60
        or value.get("prompt_specs") != expected_prompts
        or value.get("visualization_frame_counts") != [131] * 10
        or value.get("hash_checks") is not False
    ):
        raise SystemExit(f"invalid terminal motion source result: {path}")
    for profile_batch in range(5):
        batch_path = directory / f"profile_batch{profile_batch:02d}" / "BATCH_RESULT.json"
        if not batch_path.is_file():
            raise SystemExit(f"terminal source lacks motion batch result: {batch_path}")
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if (
            batch.get("protocol") != "paper_zero_wam_motion_disjoint_physical_batch_v1"
            or batch.get("passed_execution_contract") is not True
            or int(batch.get("source_index", -1)) != index
            or batch.get("target_task") != row["task"]
            or int(batch.get("source_motion_id", -1)) != int(row["source_motion_id"])
            or int(batch.get("profile_batch", -1)) != profile_batch
            or int(batch.get("checkpoint_step", -1)) != 4200
            or int(batch.get("architecture_parameter_count", -1)) != 10658724829
            or batch.get("hash_checks") is not False
            or int(batch.get("adapted_rollout_count", -1)) != 8
            or int(batch.get("released_endpoint_rollout_count", -1)) != 4
            or int(batch.get("rollout_count", -1)) != 12
            or batch.get("prompt_specs") != expected_prompts
            or batch.get("video_frame_counts") != [131] * 2
        ):
            raise SystemExit(f"invalid terminal motion batch result: {batch_path}")
print(",".join(map(str, missing)))
' "$ROOT/experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl" \
    "$PHYSICAL_ROOT" \
    "$EXPERIMENT/formal/motion_disjoint_cases/MOTION_DISJOINT_CASES.json"
)

if [[ -z "$missing" ]]; then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
path.write_text(json.dumps({
    "protocol": "paper_zero_wam_motion_disjoint_runtime_recovery_v1",
    "recovery_required": False,
    "reason": "all_nineteen_terminal_source_results_present",
    "infrastructure_retry_count": int(previous.get("infrastructure_retry_count", 0)),
    "additional_scientific_rollouts": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY"
    exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/paper_zero_wam/runtime_recovery.py" \
    --jobs "$JOBS" --job-key "motion_array_job" --log-prefix "motion" \
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
    "protocol": "paper_zero_wam_motion_disjoint_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": True,
    "reason": "three_pre_result_infrastructure_failures",
    "missing_source_indices": [int(v) for v in sys.argv[2].split(",")],
    "infrastructure_retry_count": 3,
    "additional_scientific_rollouts": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY" "$missing"
    exit 1
fi

aggregate_job=$(
    "$PYTHON_BIN" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value.get("motion_aggregate_job", ""))
' "$JOBS"
)
if [[ -n "$aggregate_job" ]]; then
    state=$(squeue -h -j "$aggregate_job" -o '%T' || true)
    if [[ "$state" == "PENDING" ]]; then
        scancel "$aggregate_job"
    fi
fi

motion_job=$(sbatch --parsable --array="$missing%1" --no-requeue \
    --job-name=pzw_motion_grid --partition=gpu --nodes=1 --ntasks=1 \
    --cpus-per-task=64 --gres=gpu:NVIDIAH200:8 --time=2-00:00:00 \
    --output="$EXPERIMENT/logs/motion_%A_%a.out" \
    --error="$EXPERIMENT/logs/motion_%A_%a.err" \
    "$RUNNER" motion_disjoint)

aggregate_job=$(sbatch --parsable --dependency="afterok:$motion_job" \
    --no-requeue \
    --job-name=pzw_motion_result --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=32G --time=04:00:00 \
    --output="$EXPERIMENT/logs/motion_result_%j.out" \
    --error="$EXPERIMENT/logs/motion_result_%j.err" \
    "$RUNNER" aggregate_motion_disjoint)

next_recovery=$(sbatch --parsable --dependency="afterany:$motion_job" \
    --no-requeue \
    --job-name=pzw_motion_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/motion_recover_%j.out" \
    --error="$EXPERIMENT/logs/motion_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_motion_disjoint.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
recovery, jobs = map(Path, sys.argv[1:3])
attempt = int(sys.argv[3])
missing = [int(value) for value in sys.argv[4].split(",")]
motion, aggregate, guardian = sys.argv[5:8]
recovery.write_text(json.dumps({
    "protocol": "paper_zero_wam_motion_disjoint_runtime_recovery_v1",
    "recovery_required": True,
    "recovery_exhausted": False,
    "reason": "array_exited_before_all_terminal_source_results",
    "missing_source_indices": missing,
    "infrastructure_retry_count": attempt,
    "additional_scientific_rollouts": 0,
    "replacement_array_job": motion,
    "next_recovery_job": guardian,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
payload = json.loads(jobs.read_text(encoding="utf-8"))
payload.update({
    "motion_array_job": motion,
    "motion_aggregate_job": aggregate,
    "motion_recovery_job": guardian,
})
jobs.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
' "$RECOVERY" "$JOBS" "$attempt" "$missing" "$motion_job" "$aggregate_job" \
    "$next_recovery"

printf 'replacement_motion=%s missing=%s aggregate=%s recovery=%s\n' \
    "$motion_job" "$missing" "$aggregate_job" "$next_recovery"
