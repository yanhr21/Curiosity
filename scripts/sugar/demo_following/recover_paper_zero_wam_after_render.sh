#!/usr/bin/env bash
# Retry only a fixed predictive render that exited before its result file.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
RESULT="$EXPERIMENT/formal/heldout_openloop_videos/RENDER_RESULT.json"
RECOVERY="$EXPERIMENT/RENDER_RUNTIME_RECOVERY.json"
JOBS="$EXPERIMENT/JOBS.json"

if [[ -f "$RESULT" ]]; then
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
result=json.loads(Path(sys.argv[1]).read_text()); p=Path(sys.argv[2])
cases=result.get("cases", [])
if (result.get("protocol") != "paper_zero_wam_heldout_openloop_render_v1"
        or result.get("execution_completed") is not True
        or int(result.get("checkpoint_step", -1)) != 4200
        or len(cases) != 8
        or set(result.get("execution_checks", {}).values()) != {True}
        or any(not Path(str(row.get("video", ""))).is_file()
               or Path(str(row.get("video", ""))).stat().st_size <= 0 for row in cases)):
    raise SystemExit("invalid predictive-render terminal evidence; refusing both suppression and replay")
old=json.loads(p.read_text()) if p.exists() else {}
p.write_text(json.dumps({
 "protocol":"paper_zero_wam_render_runtime_recovery_v1",
 "recovery_required":False,"reason":"machine_readable_render_result_present",
 "checkpoint_step":int(result.get("checkpoint_step",-1)),
 "case_count":len(result.get("cases",[])),
 "infrastructure_retry_count":int(old.get("infrastructure_retry_count",0)),
},indent=2,sort_keys=True)+"\n")
' "$RESULT" "$RECOVERY"
    exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/paper_zero_wam/runtime_recovery.py" \
    --jobs "$JOBS" --job-key "render_job" --log-prefix "render" \
    --output "${RECOVERY%.json}_INTERRUPTION_DECISION.json"

attempt=$("$PYTHON_BIN" -c '
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text()) if p.exists() else {}
print(int(x.get("infrastructure_retry_count",0))+1)
' "$RECOVERY")
if (( attempt > 3 )); then
    "$PYTHON_BIN" -c '
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 "protocol":"paper_zero_wam_render_runtime_recovery_v1",
 "recovery_required":True,"recovery_exhausted":True,
 "reason":"three_pre_result_render_runtime_failures",
 "infrastructure_retry_count":3,
},indent=2,sort_keys=True)+"\n")
' "$RECOVERY"
    exit 1
fi

render_job=$(sbatch --parsable \
    --no-requeue \
    --job-name=pzw_render --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 --time=12:00:00 \
    --output="$EXPERIMENT/logs/render_%j.out" --error="$EXPERIMENT/logs/render_%j.err" \
    "$RUNNER" render_openloop)
next_recovery_job=$(sbatch --parsable --dependency="afterany:$render_job" \
    --no-requeue \
    --job-name=pzw_render_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/render_recover_%j.out" \
    --error="$EXPERIMENT/logs/render_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_render.sh")
"$PYTHON_BIN" -c '
import json,sys
from pathlib import Path
r,j=map(Path,sys.argv[1:3]);attempt=int(sys.argv[3]);render,recovery=sys.argv[4:6]
r.write_text(json.dumps({
 "protocol":"paper_zero_wam_render_runtime_recovery_v1",
 "recovery_required":True,"recovery_exhausted":False,
 "reason":"previous_render_exited_without_machine_readable_result",
 "infrastructure_retry_count":attempt,"replacement_render_job":render,
 "next_recovery_job":recovery,
},indent=2,sort_keys=True)+"\n")
x=json.loads(j.read_text());x.update({"render_job":render,"render_recovery_job":recovery})
j.write_text(json.dumps(x,sort_keys=True)+"\n")
' "$RECOVERY" "$JOBS" "$attempt" "$render_job" "$next_recovery_job"
printf 'render_retry=%s recovery=%s\n' "$render_job" "$next_recovery_job"
