#!/usr/bin/env bash
# Query the canonical Zero-WAM repository and run the fail-closed release audit.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_DIR="${1:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/official_release_status}"
CHECKPOINT_DIR="${2:-}"
STRICT_LOAD_RESULT="${3:-}"
TAGS_JSON="$OUTPUT_DIR/GITHUB_TAGS.json"
RELEASES_JSON="$OUTPUT_DIR/GITHUB_RELEASES.json"
CANDIDATE_REFS_JSON="$OUTPUT_DIR/OFFICIAL_CANDIDATE_REFS.json"
CANDIDATE_INDEX="$OUTPUT_DIR/OFFICIAL_CANDIDATE_INDEX.json"
DISCOVERY_DIR="$OUTPUT_DIR/discovery"
COMMIT_JSON="$DISCOVERY_DIR/SELECTED_COMMIT.json"
TREE_JSON="$DISCOVERY_DIR/SELECTED_TREE.json"
AUDIT_DIR="$OUTPUT_DIR/audit"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm allocation" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl --fail --silent --show-error --max-time 30 \
    'https://api.github.com/repos/robbyant-research/Zero-WAM/tags?per_page=100' \
    -o "$TAGS_JSON"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl --fail --silent --show-error --max-time 30 \
    'https://api.github.com/repos/robbyant-research/Zero-WAM/releases?per_page=100' \
    -o "$RELEASES_JSON"

jq -n \
    --slurpfile tags "$TAGS_JSON" \
    --slurpfile releases "$RELEASES_JSON" \
    '[{source_kind:"main",source_ref:"main"}]
     + ($tags[0] | map({source_kind:"tag",source_ref:.name}))
     + ($releases[0] | map({source_kind:"release",source_ref:.tag_name}))
     | group_by(.source_ref)
     | map({
         source_ref: .[0].source_ref,
         source_kinds: (map(.source_kind) | unique)
       })' >"$CANDIDATE_REFS_JSON"

discovered_source_ref_count=$(jq '[.[].source_kinds[]] | length' "$CANDIDATE_REFS_JSON")
jq -n \
    --arg protocol "zero_wam_official_multiref_candidate_index_v1" \
    --arg repository "robbyant-research/Zero-WAM" \
    --arg api_origin "https://api.github.com" \
    --argjson discovered_source_ref_count "$discovered_source_ref_count" \
    '{
      protocol: $protocol,
      repository: $repository,
      api_origin: $api_origin,
      discovered_source_ref_count: $discovered_source_ref_count,
      resolution_error_count: 0,
      candidates: []
    }' >"$CANDIDATE_INDEX"

candidate_index=0
while IFS= read -r ref_row; do
    source_ref=$(jq -r '.source_ref' <<<"$ref_row")
    source_kinds=$(jq -c '.source_kinds' <<<"$ref_row")
    source_refs=$(jq -nc --arg ref "$source_ref" --argjson kinds "$source_kinds" \
        '$kinds | map($ref)')
    encoded_ref=$(jq -rn --arg ref "$source_ref" '$ref | @uri')
    candidate_dir=$(printf "%s/candidate_%03d" "$OUTPUT_DIR" "$candidate_index")
    candidate_commit="$candidate_dir/COMMIT.json"
    candidate_tree="$candidate_dir/TREE.json"
    mkdir -p "$candidate_dir"

    set +e
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        curl --fail --silent --show-error --max-time 30 \
        "https://api.github.com/repos/robbyant-research/Zero-WAM/commits/$encoded_ref" \
        -o "$candidate_commit"
    commit_rc=$?
    set -e
    if [[ "$commit_rc" -ne 0 ]]; then
        jq '.resolution_error_count += 1' "$CANDIDATE_INDEX" \
            >"$CANDIDATE_INDEX.tmp"
        mv "$CANDIDATE_INDEX.tmp" "$CANDIDATE_INDEX"
        candidate_index=$((candidate_index + 1))
        continue
    fi
    commit=$(jq -r '.sha' "$candidate_commit")
    if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
        jq '.resolution_error_count += 1' "$CANDIDATE_INDEX" \
            >"$CANDIDATE_INDEX.tmp"
        mv "$CANDIDATE_INDEX.tmp" "$CANDIDATE_INDEX"
        candidate_index=$((candidate_index + 1))
        continue
    fi
    set +e
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        curl --fail --silent --show-error --max-time 30 \
        "https://api.github.com/repos/robbyant-research/Zero-WAM/git/trees/$commit?recursive=1" \
        -o "$candidate_tree"
    tree_rc=$?
    set -e
    if [[ "$tree_rc" -ne 0 ]]; then
        jq '.resolution_error_count += 1' "$CANDIDATE_INDEX" \
            >"$CANDIDATE_INDEX.tmp"
        mv "$CANDIDATE_INDEX.tmp" "$CANDIDATE_INDEX"
        candidate_index=$((candidate_index + 1))
        continue
    fi

    release_assets=$(jq -c --arg ref "$source_ref" \
        '[.[] | select(.tag_name == $ref) | .assets[] |
          {name:.name,bytes:.size,browser_download_url:.browser_download_url}]' \
        "$RELEASES_JSON")
    relative_commit=$(realpath --relative-to="$OUTPUT_DIR" "$candidate_commit")
    relative_tree=$(realpath --relative-to="$OUTPUT_DIR" "$candidate_tree")
    jq \
        --argjson source_kinds "$source_kinds" \
        --argjson source_refs "$source_refs" \
        --arg commit_json "$relative_commit" \
        --arg tree_json "$relative_tree" \
        --argjson release_assets "$release_assets" \
        '.candidates += [{
          source_kinds:$source_kinds,
          source_refs:$source_refs,
          commit_json:$commit_json,
          tree_json:$tree_json,
          release_assets:$release_assets
        }]' "$CANDIDATE_INDEX" >"$CANDIDATE_INDEX.tmp"
    mv "$CANDIDATE_INDEX.tmp" "$CANDIDATE_INDEX"
    candidate_index=$((candidate_index + 1))
done < <(jq -c '.[]' "$CANDIDATE_REFS_JSON")

"$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/sugar/demo_following/discover_zero_wam_official_release.py" \
    --candidate-index "$CANDIDATE_INDEX" \
    --output-dir "$DISCOVERY_DIR" >/dev/null

COMMIT=$(jq -r '.sha' "$COMMIT_JSON")

REPO_DIR=""
if [[ "$COMMIT" != "5a8a2da069392c1974ee98941ada13a5208b0ca5" ]]; then
    REPO_DIR="$OUTPUT_DIR/repo_$COMMIT"
    ARCHIVE="$OUTPUT_DIR/repo_$COMMIT.tar.gz"
    if [[ ! -d "$REPO_DIR" ]]; then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
            curl --fail --location --retry 3 \
            "https://codeload.github.com/robbyant-research/Zero-WAM/tar.gz/$COMMIT" \
            -o "$ARCHIVE"
        mkdir -p "$REPO_DIR"
        tar --extract --gzip --file "$ARCHIVE" --strip-components=1 --directory "$REPO_DIR"
    fi
fi

ARGS=(
    --commit-json "$COMMIT_JSON"
    --tree-json "$TREE_JSON"
    --discovery-json "$DISCOVERY_DIR/OFFICIAL_RELEASE_DISCOVERY.json"
    --output-dir "$AUDIT_DIR"
)
if [[ -n "$REPO_DIR" ]]; then
    ARGS+=(--repo-dir "$REPO_DIR")
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
    ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [[ -n "$STRICT_LOAD_RESULT" ]]; then
    ARGS+=(--strict-load-result "$STRICT_LOAD_RESULT")
fi
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sugar/demo_following/audit_zero_wam_official_release.py" \
    "${ARGS[@]}"

if ! jq -e '.release_available == true' "$AUDIT_DIR/OFFICIAL_RELEASE_AUDIT.json" >/dev/null; then
    exit 3
fi
