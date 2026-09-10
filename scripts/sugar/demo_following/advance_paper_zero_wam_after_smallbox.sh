#!/usr/bin/env bash
# Admit the full motion-disjoint physical grid without a human transition.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
DECISION="$EXPERIMENT/formal/MOTION_DISJOINT_ADMISSION_RESULT.json"
mkdir -p "$EXPERIMENT/logs" "$EXPERIMENT/formal"

refresh_results() {
    if ! PYTHONPATH="$ROOT" "$PYTHON_BIN" -c \
        'from scripts.sugar.demo_following.paper_zero_wam.results import refresh_results_document; refresh_results_document()'; then
        echo 'warning: results Markdown refresh failed; motion admission state is unchanged' >&2
    fi
}

decision=$(
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
formal = json.loads((root / "FORMAL_TRAINING_RESULT.json").read_text())
heldout = json.loads((root / "heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json").read_text())
smallbox = json.loads((root / "smallbox_physical/SMALLBOX_RESULT.json").read_text())
expected_prompts = [
    {
        "split": "train",
        "task": task,
        "source_motion_id": source_motion_id,
        "latent_key": "prompt_latents",
    }
    for task, source_motion_id in (("CarryBox", 45), ("KickBox", 21))
]
if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or formal.get("execution_completed") is not True
        or int(formal.get("optimizer_steps", -1)) != 4200
        or not isinstance(formal.get("formal_decision"), dict)):
    raise SystemExit("invalid formal terminal at motion admission")
if (heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or heldout.get("execution_completed") is not True
        or int(heldout.get("checkpoint_step", -1)) != 4200
        or int(heldout.get("group_count", -1)) != 390
        or int(heldout.get("score_instance_count", -1)) != 1950
        or not isinstance(heldout.get("decision"), dict)):
    raise SystemExit("invalid held-out terminal at motion admission")
if (smallbox.get("protocol") != "paper_zero_wam_smallbox_physical_result_v1"
        or int(smallbox.get("checkpoint_step", -1)) != 4200
        or int(smallbox.get("architecture_parameter_count", -1)) != 10658724829
        or int(smallbox.get("profile_count", -1)) != 20
        or int(smallbox.get("adapted_rollout_count", -1)) != 40
        or int(smallbox.get("released_endpoint_rollout_count", -1)) != 40
        or int(smallbox.get("rollout_count", -1)) != 80
        or smallbox.get("prompt_specs") != expected_prompts
        or smallbox.get("visualization_frame_counts") != [131] * 20
        or len(smallbox.get("videos", [])) != 20
        or smallbox.get("hash_checks") is not False):
    raise SystemExit("invalid SMALLBOX terminal at motion admission")
if formal.get("passed") is not True:
    print("formal_training_decision_failed")
elif heldout.get("passed") is not True:
    print("motion_disjoint_heldout_prompt_gate_failed")
elif smallbox.get("passed") is not True:
    print("paired_smallbox_task_switch_failed")
else:
    print("passed")
' "$EXPERIMENT/formal"
)

if [[ "$decision" != "passed" ]]; then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "protocol": "paper_zero_wam_motion_disjoint_admission_v1",
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

motion_job=$(sbatch --parsable \
    --array=0-18%1 \
    --no-requeue \
    --job-name=pzw_motion_grid \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=2-00:00:00 \
    --output="$EXPERIMENT/logs/motion_%A_%a.out" \
    --error="$EXPERIMENT/logs/motion_%A_%a.err" \
    "$RUNNER" motion_disjoint)

aggregate_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$motion_job" \
    --job-name=pzw_motion_result \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=04:00:00 \
    --output="$EXPERIMENT/logs/motion_result_%j.out" \
    --error="$EXPERIMENT/logs/motion_result_%j.err" \
    "$RUNNER" aggregate_motion_disjoint)

recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$motion_job" \
    --job-name=pzw_motion_recover \
    --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/motion_recover_%j.out" \
    --error="$EXPERIMENT/logs/motion_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_motion_disjoint.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
decision_path, jobs_path = map(Path, sys.argv[1:3])
motion_job, aggregate_job, recovery_job = sys.argv[3:6]
decision_path.write_text(json.dumps({
    "protocol": "paper_zero_wam_motion_disjoint_admission_v1",
    "admitted": True,
    "reason": "formal_heldout_smallbox_gates_passed",
    "motion_array_job": motion_job,
    "aggregate_job": aggregate_job,
    "recovery_job": recovery_job,
    "test_sources": 19,
    "profiles_per_source": 10,
    "conditions_per_profile": 4,
    "endpoint_routes_per_profile": 2,
    "adapted_rollouts": 760,
    "released_endpoint_rollouts": 380,
    "rollouts": 1140,
    "optimizer_updates_added": 0,
    "hash_checks": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
payload = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {}
payload.update({
    "motion_array_job": motion_job,
    "motion_aggregate_job": aggregate_job,
    "motion_recovery_job": recovery_job,
})
jobs_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
' "$DECISION" "$EXPERIMENT/JOBS.json" "$motion_job" "$aggregate_job" "$recovery_job"

refresh_results

printf 'motion_array_job=%s aggregate_job=%s recovery_job=%s\n' \
    "$motion_job" "$aggregate_job" "$recovery_job"
