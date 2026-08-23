#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 Z|P|PS CHECKPOINT TRAIN_SEED EVAL_SEED CORRECTED_SCALE_FILE OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi

branch=$1
checkpoint=$2
train_seed=$3
evaluation_seed=$4
scale_file=$5
output_root=$6
device=${7:-cuda:0}

case "$branch" in
    Z|P|PS) ;;
    *) echo "branch must be Z, P, or PS" >&2; exit 2 ;;
esac
case "$train_seed:$evaluation_seed" in
    151014:152014|151015:152015|151016:152016) ;;
    *) echo "unexpected Plan-15 checkpoint/evaluation seed pairing" >&2; exit 2 ;;
esac

# Formal PS endpoints are frozen and inspected before their evaluation child is
# admitted.  This also prevents a long serial launcher from silently starting
# the next formal seed immediately after an endpoint.
if [[ "$branch" == PS && "${PLAN15_ALLOW_PS_ENDPOINT_EVALUATION:-0}" != 1 ]]; then
    echo "PS seed $train_seed endpoint is frozen for review; rerun this evaluation explicitly with PLAN15_ALLOW_PS_ENDPOINT_EVALUATION=1" >&2
    exit 75
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

# Training and frozen evaluation are one serial Plan-15 pipeline.  Use the
# same stable lock as the corrected formal trainer so a second allocation
# cannot launch evaluation while a training child is still active.
mkdir -p "$repo_root/experiments"
exec 9>"$repo_root/experiments/.plan15_training.lock"
if ! flock -n 9; then
    echo "another Plan-15 training/evaluation process still owns the pipeline lock" >&2
    exit 75
fi

python_bin=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
checkpoint=$(realpath "$checkpoint")
scale_file=$(realpath "$scale_file")
if [[ ! -s "$scale_file" ]]; then
    echo "corrected scale file is missing: $scale_file" >&2
    exit 2
fi
mass_factors=(1.0 1.5 3.0 6.0 10.0)

validate_sensor_read_contract() {
    local summary=$1
    "$python_bin" - "$branch" "$summary" <<'PY'
import json
from pathlib import Path
import sys

branch = sys.argv[1]
path = Path(sys.argv[2])
summary = json.loads(path.read_text(encoding="utf-8"))
if summary.get("branch") != branch:
    raise SystemExit(f"branch mismatch in {path}")
required = {
    "profiles",
    "num_envs",
    "max_steps",
    "evaluator_reads_tacsl",
    "trace_patch_semantics",
    "evaluator_tacsl_feature_read_calls",
    "evaluator_slip_detector_update_calls",
}
missing = sorted(required - summary.keys())
if missing:
    raise SystemExit(
        f"{branch} evaluator sensor-read evidence missing in {path}: {missing}"
    )
profiles = int(summary["profiles"])
num_envs = int(summary["num_envs"])
max_steps = int(summary["max_steps"])
if profiles % num_envs:
    raise SystemExit(f"invalid evaluation batching in {path}")
expected_live_calls = profiles // num_envs * max_steps
reads = int(summary["evaluator_tacsl_feature_read_calls"])
slip_updates = int(summary["evaluator_slip_detector_update_calls"])
if branch == "Z":
    valid = (
        summary.get("evaluator_reads_tacsl") is False
        and summary.get("trace_patch_semantics") == "exact_zero_control"
        and reads == 0
        and slip_updates == 0
    )
else:
    valid = (
        summary.get("evaluator_reads_tacsl") is True
        and summary.get("trace_patch_semantics")
        == "online_tacsl_evaluator_label"
        and reads == expected_live_calls
        and slip_updates == expected_live_calls
    )
if not valid:
    raise SystemExit(
        f"{branch} evaluator sensor-read contract failed in {path}: "
        f"reads={reads}, slip_updates={slip_updates}, "
        f"expected_live_calls={expected_live_calls}"
    )
PY
}

mkdir -p "$output_root"
for mass_factor in "${mass_factors[@]}"; do
    factor_tag=${mass_factor/./p}x
    output="$output_root/train_${train_seed}_eval_${evaluation_seed}_${factor_tag}"
    if [[ -s "$output/summary.json" && -s "$output/frozen_evaluation_trace.npz" ]]; then
        if grep -q '"evaluation_view": "strict_sugar_reference"' "$output/summary.json"; then
            validate_sensor_read_contract "$output/summary.json"
            echo "[PLAN15 SEED SWEEP] already complete: $output"
            continue
        fi
        echo "existing output is not strict SUGAR evaluation; choose a new OUTPUT_ROOT: $output" >&2
        exit 1
    fi
    echo "[PLAN15 SEED SWEEP] branch=$branch train_seed=$train_seed eval_seed=$evaluation_seed factor=$mass_factor"
    PLAN15_PIPELINE_LOCK_HELD=1 \
    "$python_bin" -u SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py \
        --branch "$branch" \
        --checkpoint "$checkpoint" \
        --patch-scale-file "$scale_file" \
        --output-root "$output" \
        --training-seed "$train_seed" \
        --seed "$evaluation_seed" \
        --mass-factor "$mass_factor" \
        --motion-folder "$repo_root/SUGAR/data/CarryBox/data_045" \
        --motion-id 0 \
        --profiles 20 \
        --num-envs 4 \
        --max-steps 450 \
        --post-jump-window 80 \
        --headless \
        --device "$device"
    if [[ ! -s "$output/summary.json" || ! -s "$output/frozen_evaluation_trace.npz" ]]; then
        echo "frozen evaluation did not write its required outputs: $output" >&2
        exit 1
    fi
    validate_sensor_read_contract "$output/summary.json"
done
