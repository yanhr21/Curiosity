#!/usr/bin/env bash
# Collect the frozen ten-candidate reservoir used to admit five BPP demos per motion.

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
CONTRACT="${CONTRACT:-$ROOT/scripts/sugar/demo_following/config/bpp_five_variant_corpus_v1.json}"
BASE="${1:-$ROOT/experiments/demo_following/bpp_official_v1/sugar_five_variant_corpus_v1}"
ACTION_ROOT="$BASE/action_candidates"
STAGING_ROOT="$BASE/action_candidate_staging"
ACTION_AUDIT_ROOT="$BASE/action_candidate_audits"
ACTION_ADMISSION="$BASE/action_admission_v1"
PROMPT_ROOT="${PROMPT_ROOT:-$ROOT/experiments/demo_following/zero_wam_official_v1/prompt_rgb_isolated_v1}"
SHARDS="$ROOT/experiments/demo_following/runtime_assets/contact_event/corpus_shards"

# The formal collector writes and closes RESULT.json/TRACE.npz before exiting.
# Avoid the Isaac Sim teardown hang observed on Slurm-remapped non-zero H200s;
# this does not alter renderer inputs, PhysX state, policy inference, or traces.
export SUGAR_FAST_EXIT_AFTER_RESULT=1

# H200 nodes can retain a full local /tmp from unrelated or expired Isaac jobs.
# Keep every job-local Isaac/Kit temporary path on the node's large tmpfs so a
# full /tmp cannot prevent portable-root creation or corrupt startup evidence.
# The formal RESULT/TRACE artifacts still go to the shared immutable corpus.
BPP_JOB_TMP_ROOT="${BPP_JOB_TMP_ROOT:-/dev/shm/Curiosity_bpp_${SLURM_JOB_ID}}"
export TMPDIR="$BPP_JOB_TMP_ROOT/tmp"
export ISAACLAB_TMP_ROOT="$BPP_JOB_TMP_ROOT/isaaclab"
export SUGAR_UNITREE_TMP_ROOT="$BPP_JOB_TMP_ROOT/unitree"
export SUGAR_KIT_PORTABLE_ROOT="$BPP_JOB_TMP_ROOT/kit"
mkdir -p "$TMPDIR" "$ISAACLAB_TMP_ROOT" "$SUGAR_UNITREE_TMP_ROOT" "$SUGAR_KIT_PORTABLE_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "BPP action collection requires a retained Slurm compute allocation" >&2
    exit 2
fi
if [[ "$(hostname)" == login* || "$(hostname)" == mgmtserver* ]]; then
    echo "BPP action collection is forbidden on a login node" >&2
    exit 2
fi
jq -e '
    .protocol == "sugar_bpp_five_variant_corpus_contract_v1"
    and .candidate_variant_count_per_source_motion == 10
    and .admitted_variant_count_per_source_motion == 5
    and (.variants | length) == 10
    and .render_contract.camera_keys == ["agentview_rgb", "eye_in_hand_rgb"]
    and .render_contract.camera_candidate_contract.sha256 == "9276b901a33b788fb03b93a1b5fe2abaf5093bb44cf51376617e5ea787165daf"
    and .exact_admitted_totals.train_target_samples_per_epoch == 560000
    and .exact_admitted_totals.train_target_sample_exposures_at_goal_chain_7_epochs == 3920000
' "$CONTRACT" >/dev/null
if [[ ! -d "$PROMPT_ROOT" ]]; then
    echo "missing immutable prompt corpus: $PROMPT_ROOT" >&2
    exit 2
fi

collect_shard() {
    local variant_id=$1
    local task=$2
    local shard=$3
    local envs=$4
    local seed=$5
    local lower checkpoint_dir variant_root output result staging staging_result
    lower=$(printf '%s' "$task" | tr '[:upper:]' '[:lower:]')
    checkpoint_dir="$task"
    variant_root="$ACTION_ROOT/candidate_variant_${variant_id}"
    output="$variant_root/${lower}_shard${shard}_seed${seed}"
    result="$output/RESULT.json"

    if [[ -f "$result" && -f "$output/TRACE.npz" ]]; then
        jq -e --argjson variant "$variant_id" --argjson seed "$seed" '
            .passed == true
            and .bpp_variant_id == $variant
            and .seed == $seed
            and .steps == 700
        ' "$result" >/dev/null
        return
    fi
    if [[ -e "$output" ]]; then
        echo "refusing incomplete/nonempty candidate shard: $output" >&2
        exit 2
    fi
    staging="$STAGING_ROOT/job_${SLURM_JOB_ID}/candidate_variant_${variant_id}/${lower}_shard${shard}_seed${seed}"
    staging_result="$staging/RESULT.json"
    if [[ -e "$staging" ]]; then
        echo "refusing pre-existing staging shard: $staging" >&2
        exit 2
    fi

    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_reward/collect_official_tracker_contact_events.py" \
        --task-family "$task" \
        --motion-folder "$SHARDS/${lower}_shard${shard}" \
        --generator-checkpoint "$ROOT/SUGAR/demo_ckpts/$checkpoint_dir/generator.ckpt" \
        --checkpoint "$ROOT/SUGAR/demo_ckpts/$checkpoint_dir/tracker.pt" \
        --output-dir "$staging" \
        --num-envs "$envs" --steps 700 --seed "$seed" \
        --bpp-variant-id "$variant_id" \
        --headless --device cuda:0
    jq -e --argjson variant "$variant_id" --argjson seed "$seed" '
        .passed == true
        and .bpp_variant_id == $variant
        and .seed == $seed
        and (.startup_profile | length) == 5
    ' "$staging_result" >/dev/null
    if [[ ! -f "$staging/TRACE.npz" ]]; then
        echo "completed staging shard has no TRACE.npz: $staging" >&2
        exit 2
    fi
    mkdir -p "$variant_root"
    mv "$staging" "$output"
    jq -e --argjson variant "$variant_id" --argjson seed "$seed" '
        .passed == true
        and .bpp_variant_id == $variant
        and .seed == $seed
        and .steps == 700
    ' "$result" >/dev/null
}

for variant_id in $(seq 0 9); do
    for shard_number in $(seq 0 3); do
        shard=$(printf '%02d' "$shard_number")
        carry_seed=$(jq -r ".variants[$variant_id].carry_shard_seeds[$shard_number]" "$CONTRACT")
        kick_seed=$(jq -r ".variants[$variant_id].kick_shard_seeds[$shard_number]" "$CONTRACT")
        collect_shard "$variant_id" CarryBox "$shard" 25 "$carry_seed"
        if [[ "$shard_number" -eq 3 ]]; then
            kick_envs=24
        else
            kick_envs=25
        fi
        collect_shard "$variant_id" KickBox "$shard" "$kick_envs" "$kick_seed"
    done

    audit_dir="$ACTION_AUDIT_ROOT/candidate_variant_${variant_id}"
    if [[ ! -f "$audit_dir/RESULT.json" ]]; then
        "$PYTHON_BIN" \
            "$ROOT/scripts/sugar/demo_following/audit_zero_wam_sugar_action_corpus.py" \
            --action-corpus "$ACTION_ROOT/candidate_variant_${variant_id}" \
            --prompt-corpus "$PROMPT_ROOT" \
            --output-dir "$audit_dir"
    fi
    jq -e '.passed == true and .trajectory_count == 199 and .transition_count == 139300' \
        "$audit_dir/RESULT.json" >/dev/null
done

if [[ ! -f "$ACTION_ADMISSION/RESULT.json" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/audit_bpp_five_variant_action_corpus.py" \
        --contract "$CONTRACT" \
        --action-corpus-root "$ACTION_ROOT" \
        --action-audit-root "$ACTION_AUDIT_ROOT" \
        --output-dir "$ACTION_ADMISSION"
fi
jq -e '
    .passed == true
    and .candidate_trajectory_count == 1990
    and .admitted_trajectory_count == 995
    and .train_five_action_interval_count == 112000
' "$ACTION_ADMISSION/RESULT.json" >/dev/null

echo "BPP_CANDIDATE_ACTION_COLLECTION_COMPLETE root=$ACTION_ROOT admission=$ACTION_ADMISSION"
