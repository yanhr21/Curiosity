#!/usr/bin/env bash
# Retry the frozen held-out score only after a pre-result runtime failure.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
ADMISSION="$ROOT/scripts/sugar/demo_following/advance_paper_zero_wam_after_heldout.sh"
RESULT="$EXPERIMENT/formal/heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json"
SCORES="$EXPERIMENT/formal/heldout_prompt_gate/HELDOUT_PROMPT_SCORES.jsonl"
RECOVERY="$EXPERIMENT/HELDOUT_RUNTIME_RECOVERY.json"
JOBS="$EXPERIMENT/JOBS.json"

if [[ -f "$RESULT" ]]; then
    PYTHONPATH="$PZW_SITE_PACKAGES:$ROOT" "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
from scripts.sugar.demo_following.paper_zero_wam.config import PaperZeroWAMConfig
from scripts.sugar.demo_following.paper_zero_wam.evaluate_heldout import aggregate
result = json.loads(Path(sys.argv[1]).read_text())
scores = Path(sys.argv[2])
records = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines() if line.strip()] if scores.is_file() else []
decision = aggregate(records)
config = PaperZeroWAMConfig()
if (result.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or result.get("execution_completed") is not True
        or int(result.get("checkpoint_step", -1)) != 4200
        or int(result.get("architecture_parameter_count", -1))
        != config.expected_parameter_count
        or int(result.get("group_count", -1)) != 390
        or int(result.get("score_instance_count", -1)) != 1950
        or result.get("hash_checks") is not False
        or len(records) != 390
        or [int(row.get("group_index", -1)) for row in records] != list(range(390))
        or result.get("decision") != decision
        or result.get("passed") is not decision["passed"]):
    raise SystemExit("invalid held-out terminal evidence; refusing both suppression and replay")
path = Path(sys.argv[3])
previous = json.loads(path.read_text()) if path.exists() else {}
path.write_text(json.dumps({
    "protocol": "paper_zero_wam_heldout_runtime_recovery_v1",
    "recovery_required": False,
    "reason": "machine_readable_heldout_result_present",
    "heldout_passed": result.get("passed") is True,
    "score_instance_count": int(result.get("score_instance_count", -1)),
    "infrastructure_retry_count": int(previous.get("infrastructure_retry_count", 0)),
}, indent=2, sort_keys=True) + "\n")
' "$RESULT" "$SCORES" "$RECOVERY"
    exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/paper_zero_wam/runtime_recovery.py" \
    --jobs "$JOBS" --job-key "heldout_job" --log-prefix "heldout" \
    --output "${RECOVERY%.json}_INTERRUPTION_DECISION.json"

attempt=$("$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()) if p.exists() else {}
print(int(x.get("infrastructure_retry_count", 0)) + 1)
' "$RECOVERY")
if (( attempt > 3 )); then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "protocol":"paper_zero_wam_heldout_runtime_recovery_v1",
    "recovery_required":True,
    "recovery_exhausted":True,
    "reason":"three_pre_result_heldout_runtime_failures",
    "infrastructure_retry_count":3,
}, indent=2, sort_keys=True)+"\n")
' "$RECOVERY"
    exit 1
fi

mapfile -t stale_jobs < <("$PYTHON_BIN" -c '
import json, sys
x=json.load(open(sys.argv[1]))
for key in ("render_recovery_job", "physical_admission_job", "render_job"):
    if x.get(key): print(x[key])
' "$JOBS")
for job in "${stale_jobs[@]}"; do
    [[ "$(squeue -h -j "$job" -o '%T' || true)" == PENDING ]] && scancel "$job"
done

heldout_job=$(sbatch --parsable \
    --no-requeue \
    --job-name=pzw_heldout --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=1-00:00:00 \
    --output="$EXPERIMENT/logs/heldout_%j.out" --error="$EXPERIMENT/logs/heldout_%j.err" \
    "$RUNNER" evaluate_heldout)
render_job=$(sbatch --parsable --dependency="afterok:$heldout_job" \
    --no-requeue \
    --job-name=pzw_render --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=12:00:00 \
    --output="$EXPERIMENT/logs/render_%j.out" --error="$EXPERIMENT/logs/render_%j.err" \
    "$RUNNER" render_openloop)
render_recovery_job=$(sbatch --parsable --dependency="afterany:$render_job" \
    --no-requeue \
    --job-name=pzw_render_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/render_recover_%j.out" \
    --error="$EXPERIMENT/logs/render_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_render.sh")
physical_admission_job=$(sbatch --parsable --dependency="afterok:$heldout_job" \
    --no-requeue \
    --job-name=pzw_physical_admit --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --mem=8G --time=01:00:00 \
    --output="$EXPERIMENT/logs/physical_admit_%j.out" \
    --error="$EXPERIMENT/logs/physical_admit_%j.err" "$ADMISSION")
next_recovery_job=$(sbatch --parsable --dependency="afterany:$heldout_job" \
    --no-requeue \
    --job-name=pzw_heldout_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/heldout_recover_%j.out" \
    --error="$EXPERIMENT/logs/heldout_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_heldout.sh")

"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
r,j=map(Path,sys.argv[1:3]); attempt=int(sys.argv[3])
heldout,render,render_recovery,admission,next_recovery=sys.argv[4:9]
r.write_text(json.dumps({
 "protocol":"paper_zero_wam_heldout_runtime_recovery_v1",
 "recovery_required":True,"recovery_exhausted":False,
 "reason":"previous_heldout_exited_without_machine_readable_result",
 "infrastructure_retry_count":attempt,"replacement_heldout_job":heldout,
 "next_recovery_job":next_recovery,
},indent=2,sort_keys=True)+"\n")
x=json.loads(j.read_text()); x.update({
 "heldout_job":heldout,"heldout_recovery_job":next_recovery,
 "render_job":render,"render_recovery_job":render_recovery,
 "physical_admission_job":admission,
}); j.write_text(json.dumps(x,sort_keys=True)+"\n")
' "$RECOVERY" "$JOBS" "$attempt" "$heldout_job" "$render_job" \
  "$render_recovery_job" "$physical_admission_job" "$next_recovery_job"

printf 'heldout_retry=%s render=%s render_recovery=%s admission=%s recovery=%s\n' \
  "$heldout_job" "$render_job" "$render_recovery_job" \
  "$physical_admission_job" "$next_recovery_job"
