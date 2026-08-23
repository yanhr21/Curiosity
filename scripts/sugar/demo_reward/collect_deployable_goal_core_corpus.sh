#!/usr/bin/env bash
# Collect the motion-disjoint actual-contact corpus with the deployable 121-D goal-policy core.

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/contact_event_reward_redesign_v1/deployable_goal_core_corpus_v1}"
SHARDS="$ROOT/experiments/demo_following/runtime_assets/contact_event/corpus_shards"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained GPU Slurm allocation" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

collect_shard() {
    local task=$1
    local shard=$2
    local envs=$3
    local seed=$4
    local lower
    local checkpoint_dir
    lower=$(printf '%s' "$task" | tr '[:upper:]' '[:lower:]')
    if [[ "$task" == "CarryBox" ]]; then
        checkpoint_dir="CarryBox"
    else
        checkpoint_dir="KickBox"
    fi
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_reward/collect_official_tracker_contact_events.py" \
        --task-family "$task" \
        --motion-folder "$SHARDS/${lower}_shard${shard}" \
        --generator-checkpoint "$ROOT/SUGAR/demo_ckpts/$checkpoint_dir/generator.ckpt" \
        --checkpoint "$ROOT/SUGAR/demo_ckpts/$checkpoint_dir/tracker.pt" \
        --output-dir "$OUTPUT_ROOT/${lower}_shard${shard}_seed${seed}" \
        --num-envs "$envs" --steps 700 --seed "$seed" --headless --device cuda:0
    jq -e '.passed == true' "$OUTPUT_ROOT/${lower}_shard${shard}_seed${seed}/RESULT.json" >/dev/null
}

for shard in 00 01 02 03; do
    collect_shard CarryBox "$shard" 25 "$((271100 + 10#$shard))"
done
for shard in 00 01 02; do
    collect_shard KickBox "$shard" 25 "$((271200 + 10#$shard))"
done
collect_shard KickBox 03 24 271203

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_reward/audit_actual_contact_event_corpus.py" \
    --corpus-root "$OUTPUT_ROOT" \
    --output-dir "${OUTPUT_ROOT}_audit"
jq -e '.passed == true' "${OUTPUT_ROOT}_audit/RESULT.json" >/dev/null
