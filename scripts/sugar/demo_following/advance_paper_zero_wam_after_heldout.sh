#!/usr/bin/env bash
# Automatically admit or close the paired SMALLBOX physical stage.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
HELDOUT="$EXPERIMENT/formal/heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json"
HELDOUT_SCORES="$EXPERIMENT/formal/heldout_prompt_gate/HELDOUT_PROMPT_SCORES.jsonl"
FORMAL="$EXPERIMENT/formal/FORMAL_TRAINING_RESULT.json"
FORMAL_TRACE="$EXPERIMENT/formal/TRAIN_TRACE.jsonl"
SCHEDULE="$EXPERIMENT/schedule/TRAIN_SCHEDULE.jsonl"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
DECISION="$EXPERIMENT/formal/PHYSICAL_ADMISSION_RESULT.json"
mkdir -p "$EXPERIMENT/logs" "$EXPERIMENT/formal"

refresh_results() {
    if ! PYTHONPATH="$ROOT" "$PYTHON_BIN" -c \
        'from scripts.sugar.demo_following.paper_zero_wam.results import refresh_results_document; refresh_results_document()'; then
        echo 'warning: results Markdown refresh failed; physical admission state is unchanged' >&2
    fi
}

decision=$(
    PYTHONPATH="$PZW_SITE_PACKAGES:$ROOT" "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
from scripts.sugar.demo_following.paper_zero_wam.config import PaperZeroWAMConfig
from scripts.sugar.demo_following.paper_zero_wam.evaluate_heldout import aggregate
from scripts.sugar.demo_following.paper_zero_wam.train import formal_training_decision
heldout = json.load(open(sys.argv[1], encoding="utf-8"))
formal = json.load(open(sys.argv[2], encoding="utf-8"))
config = PaperZeroWAMConfig()
recomputed_formal = formal_training_decision(
    Path(sys.argv[3]), Path(sys.argv[4]), config
)
score_records = [
    json.loads(line)
    for line in Path(sys.argv[5]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
recomputed_heldout = aggregate(score_records)
if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or formal.get("execution_completed") is not True
        or int(formal.get("optimizer_steps", -1)) != 4200
        or int(formal.get("architecture_parameter_count", -1))
        != config.expected_parameter_count
        or formal.get("hash_checks") is not False
        or formal.get("formal_decision") != recomputed_formal
        or formal.get("passed") is not recomputed_formal["passed"]):
    raise SystemExit("invalid formal terminal at physical admission")
if (heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or heldout.get("execution_completed") is not True
        or int(heldout.get("checkpoint_step", -1)) != 4200
        or int(heldout.get("architecture_parameter_count", -1))
        != config.expected_parameter_count
        or int(heldout.get("group_count", -1)) != 390
        or int(heldout.get("score_instance_count", -1)) != 1950
        or heldout.get("hash_checks") is not False
        or heldout.get("decision") != recomputed_heldout
        or heldout.get("passed") is not recomputed_heldout["passed"]):
    raise SystemExit("invalid held-out terminal at physical admission")
if formal.get("passed") is not True:
    print("formal_training_decision_failed")
elif heldout.get("passed") is not True:
    print("motion_disjoint_heldout_prompt_gate_failed")
else:
    print("passed")
' "$HELDOUT" "$FORMAL" "$FORMAL_TRACE" "$SCHEDULE" "$HELDOUT_SCORES"
)

if [[ "$decision" != "passed" ]]; then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "protocol": "paper_zero_wam_physical_admission_v1",
    "admitted": False,
    "reason": sys.argv[2],
    "optimizer_updates_added": 0,
    "threshold_or_training_sweep_started": False,
    "hash_checks": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$DECISION" "$decision"
    refresh_results
    exit 0
fi

physical_job=$(sbatch --parsable \
    --array=0-4%1 \
    --no-requeue \
    --job-name=pzw_smallbox \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=1-00:00:00 \
    --output="$EXPERIMENT/logs/smallbox_%A_%a.out" \
    --error="$EXPERIMENT/logs/smallbox_%A_%a.err" \
    "$RUNNER" physical_smallbox)

aggregate_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$physical_job" \
    --job-name=pzw_smallbox_result \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --output="$EXPERIMENT/logs/smallbox_result_%j.out" \
    --error="$EXPERIMENT/logs/smallbox_result_%j.err" \
    "$RUNNER" aggregate_smallbox)

motion_admission_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$aggregate_job" \
    --job-name=pzw_motion_admit \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --output="$EXPERIMENT/logs/motion_admit_%j.out" \
    --error="$EXPERIMENT/logs/motion_admit_%j.err" \
    "$ROOT/scripts/sugar/demo_following/advance_paper_zero_wam_after_smallbox.sh")

recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$physical_job" \
    --job-name=pzw_smallbox_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/smallbox_recover_%j.out" \
    --error="$EXPERIMENT/logs/smallbox_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_smallbox.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
decision = Path(sys.argv[1])
jobs = Path(sys.argv[2])
physical_job, aggregate_job, motion_admission_job, recovery_job = sys.argv[3:7]
decision.write_text(json.dumps({
    "protocol": "paper_zero_wam_physical_admission_v1",
    "admitted": True,
    "reason": "motion_disjoint_heldout_prompt_gate_passed",
    "physical_array_job": physical_job,
    "aggregate_job": aggregate_job,
    "motion_admission_job": motion_admission_job,
    "recovery_job": recovery_job,
    "profile_batches": 5,
    "paired_profiles": 20,
    "adapted_rollouts": 40,
    "released_endpoint_rollouts": 40,
    "rollouts": 80,
    "hash_checks": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
payload = json.loads(jobs.read_text(encoding="utf-8")) if jobs.exists() else {}
payload.update({
    "physical_array_job": physical_job,
    "physical_aggregate_job": aggregate_job,
    "motion_admission_job": motion_admission_job,
    "smallbox_recovery_job": recovery_job,
})
jobs.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
' "$DECISION" "$EXPERIMENT/JOBS.json" "$physical_job" "$aggregate_job" "$motion_admission_job" "$recovery_job"

refresh_results

printf 'physical_array_job=%s aggregate_job=%s motion_admission_job=%s recovery_job=%s\n' \
    "$physical_job" "$aggregate_job" "$motion_admission_job" "$recovery_job"
