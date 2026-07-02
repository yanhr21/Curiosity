#!/usr/bin/env bash
set -euo pipefail

# Common guard for official-method Phase07 comparison runners.
# It refuses to run unless the official environment/checkpoint/stage-1/leakage
# gates are satisfied. It does not perform toy inference or train anything.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="${METHOD:?METHOD is required}"
ENV_PATH="${ENV_PATH:?ENV_PATH is required}"
STAGE1_FILES="${STAGE1_FILES:?STAGE1_FILES is required}"
CHECKPOINT_GLOBS="${CHECKPOINT_GLOBS:?CHECKPOINT_GLOBS is required}"
BLOCKER_PATH="${BLOCKER_PATH:-}"
ALLOW_OFFICIAL_COMPARISON="${ALLOW_OFFICIAL_COMPARISON:-0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: official comparison runner must run inside a held Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports

RUN_TAG="${RUN_TAG:-phase07_${METHOD}_official_comparison_gate_20260627}"
{
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'METHOD=%q\n' "$METHOD"
  printf 'SLURM_JOB_ID=%q\n' "$SLURM_JOB_ID"
  printf 'HOSTNAME=%q\n' "$(hostname)"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'ENV_PATH=%q\n' "$ENV_PATH"
  printf 'ALLOW_OFFICIAL_COMPARISON=%q\n' "$ALLOW_OFFICIAL_COMPARISON"
  printf 'CLASSIFICATION=%q\n' "official_method_comparison_gate_no_toy_no_success_claim"
} >"$ROOT/logs/newton/${RUN_TAG}_env.sh"

failures=()

if [[ "$ALLOW_OFFICIAL_COMPARISON" != "1" ]]; then
  failures+=("ALLOW_OFFICIAL_COMPARISON_not_set")
fi

if [[ ! -e "$ROOT/$ENV_PATH" ]]; then
  failures+=("missing_env=$ENV_PATH")
fi

while IFS= read -r rel_path; do
  [[ -n "$rel_path" ]] || continue
  if [[ ! -f "$ROOT/$rel_path" ]]; then
    failures+=("missing_stage1_file=$rel_path")
  fi
done <<<"$STAGE1_FILES"

leakage_json="$ROOT/experiments/outputs/phase07_stage1_no_heldout_leakage_v1_20260627.json"
if [[ ! -f "$leakage_json" ]]; then
  failures+=("missing_stage1_no_heldout_leakage_json")
elif ! "$ROOT/envs/newton/.venv/bin/python" - "$leakage_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("no_held_out_leakage_proven") is True else 1)
PY
then
  failures+=("stage1_no_heldout_leakage_not_proven")
fi

checkpoint_count=0
while IFS= read -r pattern; do
  [[ -n "$pattern" ]] || continue
  if compgen -G "$ROOT/checkpoints/$pattern" >/dev/null; then
    checkpoint_count=$((checkpoint_count + 1))
  fi
  if compgen -G "$ROOT/external/checkpoints/$pattern" >/dev/null; then
    checkpoint_count=$((checkpoint_count + 1))
  fi
done <<<"$CHECKPOINT_GLOBS"

blocker_status=""
if [[ -n "$BLOCKER_PATH" && -f "$ROOT/$BLOCKER_PATH" ]]; then
  blocker_status="$("$ROOT/envs/newton/.venv/bin/python" - "$ROOT/$BLOCKER_PATH" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", ""))
PY
)"
fi
if [[ "$checkpoint_count" -eq 0 && ( -z "$blocker_status" || "$blocker_status" == "template_unfilled_not_a_blocker" ) ]]; then
  failures+=("missing_official_checkpoint_or_filled_blocker")
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
  printf 'OFFICIAL_COMPARISON_GATE_STATUS=blocked\n'
  printf 'METHOD=%s\n' "$METHOD"
  printf 'FAILURES=%s\n' "${failures[*]}"
  exit 40
fi

cat <<EOF
OFFICIAL_COMPARISON_GATE_STATUS=ready_to_implement_method_specific_official_call
METHOD=$METHOD
NOTE=All guard files exist, but method-specific official inference/fine-tune code must be implemented next. This script does not run toy inference and is not a success claim.
EOF
