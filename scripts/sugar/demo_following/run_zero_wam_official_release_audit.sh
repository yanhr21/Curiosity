#!/usr/bin/env bash
# Query the canonical Zero-WAM repository and run the fail-closed release audit.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_DIR="${1:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/official_release_status}"
CHECKPOINT_DIR="${2:-}"
STRICT_LOAD_RESULT="${3:-}"
COMMIT_JSON="$OUTPUT_DIR/GITHUB_MAIN_COMMIT.json"
TREE_JSON="$OUTPUT_DIR/GITHUB_MAIN_TREE.json"
AUDIT_DIR="$OUTPUT_DIR/audit"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm allocation" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl --fail --silent --show-error --max-time 30 \
    https://api.github.com/repos/robbyant-research/Zero-WAM/commits/main \
    -o "$COMMIT_JSON"
COMMIT=$(jq -r '.sha' "$COMMIT_JSON")
if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "canonical GitHub API did not return a full commit" >&2
    exit 3
fi
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl --fail --silent --show-error --max-time 30 \
    "https://api.github.com/repos/robbyant-research/Zero-WAM/git/trees/$COMMIT?recursive=1" \
    -o "$TREE_JSON"

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
