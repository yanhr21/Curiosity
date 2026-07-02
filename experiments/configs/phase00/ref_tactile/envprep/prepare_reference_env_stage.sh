#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TARGET="${TARGET:-}"
STAGE="${STAGE:-preflight}"
EXECUTE="${EXECUTE:-0}"
ACK_HEAVY_LOCAL_ENV_BUILD="${ACK_HEAVY_LOCAL_ENV_BUILD:-}"
ALLOW_LOGIN_ENV_BUILD="${ALLOW_LOGIN_ENV_BUILD:-}"
CONDA_EXE="${CONDA_EXE:-$ROOT/envs/taccel/miniforge/bin/conda}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/newton/phase00/ref_tactile/envprep}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/experiments/reports/phase00/ref_tactile/envprep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/experiments/outputs/phase00/ref_tactile/envprep}"

usage() {
  cat <<'EOF'
Usage:
  TARGET=univtac|tacauchy STAGE=<stage> EXECUTE=0|1 \
    bash experiments/configs/phase00/ref_tactile/envprep/prepare_reference_env_stage.sh

Stages:
  preflight
  create_env
  install_isaac
  install_isaaclab
  install_curobo_or_assets
  install_tacex_core
  build_uipc
  setup_assets
  official_sanity

Default EXECUTE=0 prints the staged commands and writes evidence files.
EXECUTE=1 is guarded and must never be used on compute nodes.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ "$TARGET" != "univtac" && "$TARGET" != "tacauchy" ]]; then
  usage >&2
  fail "TARGET must be univtac or tacauchy"
fi

case "$STAGE" in
  preflight|create_env|install_isaac|install_isaaclab|install_curobo_or_assets|install_tacex_core|build_uipc|setup_assets|official_sanity) ;;
  *) usage >&2; fail "unknown STAGE: $STAGE" ;;
esac

host="$(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  fail "refusing environment installation/preparation inside Slurm allocation $SLURM_JOB_ID"
fi
if [[ "$EXECUTE" == "1" ]]; then
  if [[ "$ACK_HEAVY_LOCAL_ENV_BUILD" != "yes" ]]; then
    fail "EXECUTE=1 requires ACK_HEAVY_LOCAL_ENV_BUILD=yes"
  fi
  if [[ "$host" == mgmtserver* && "$ALLOW_LOGIN_ENV_BUILD" != "yes" ]]; then
    fail "refusing heavy environment build on login node $host without ALLOW_LOGIN_ENV_BUILD=yes"
  fi
fi

repo=""
expected_commit=""
env_prefix=""
python_version=""
isaac_sim=""
isaac_lab=""
case "$TARGET" in
  univtac)
    repo="$ROOT/external/UniVTAC"
    expected_commit="05bcd3edb92237107efa40105292a24f1a9fd761"
    env_prefix="$ROOT/envs/univtac/conda"
    python_version="3.10"
    isaac_sim="4.5.0"
    isaac_lab="v2.1.1"
    ;;
  tacauchy)
    repo="$ROOT/external/TaCauchy"
    expected_commit="c228cfe9050904cd5d71d64f6eb5104768d4cbda"
    env_prefix="$ROOT/envs/tacauchy/conda"
    python_version="3.11"
    isaac_sim="5.0.0"
    isaac_lab="v2.2.1"
    ;;
esac

mkdir -p "$LOG_ROOT/$TARGET" "$REPORT_ROOT/$TARGET" "$OUTPUT_ROOT/$TARGET"
log_path="$LOG_ROOT/$TARGET/${STAGE}.log"
report_path="$REPORT_ROOT/$TARGET/${STAGE}.md"
status_path="$OUTPUT_ROOT/$TARGET/${STAGE}_status.json"

commands_file="$OUTPUT_ROOT/$TARGET/${STAGE}_commands.sh"
mkdir -p "$(dirname "$commands_file")"

observed_commit=""
if [[ -d "$repo/.git" ]]; then
  observed_commit="$(git -C "$repo" rev-parse HEAD)"
fi

write_commands() {
  : >"$commands_file"
  case "$STAGE" in
    preflight)
      cat >>"$commands_file" <<EOF
test -d "$repo/.git"
test "\$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -x "$CONDA_EXE"
EOF
      ;;
    create_env)
      cat >>"$commands_file" <<EOF
mkdir -p "$(dirname "$env_prefix")"
"$CONDA_EXE" create -y -p "$env_prefix" python="$python_version"
EOF
      ;;
    install_isaac)
      if [[ "$TARGET" == "univtac" ]]; then
        cat >>"$commands_file" <<EOF
"$env_prefix/bin/python" -m pip install --upgrade pip
"$env_prefix/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$env_prefix/bin/python" -m pip install 'isaacsim[all,extscache]==$isaac_sim' --extra-index-url https://pypi.nvidia.com
EOF
      else
        cat >>"$commands_file" <<EOF
"$env_prefix/bin/python" -m pip install --upgrade pip
"$env_prefix/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"$env_prefix/bin/python" -m pip install 'isaacsim[all,extscache]==$isaac_sim' --extra-index-url https://pypi.nvidia.com
EOF
      fi
      ;;
    install_isaaclab)
      cat >>"$commands_file" <<EOF
mkdir -p "$ROOT/external/IsaacLab_refs"
cd "$ROOT/external/IsaacLab_refs"
test -d "IsaacLab_$isaac_lab" || git clone https://github.com/isaac-sim/IsaacLab "IsaacLab_$isaac_lab"
cd "IsaacLab_$isaac_lab"
git checkout "$isaac_lab"
"$env_prefix/bin/python" -m pip install flatdict==4.0.1 --no-build-isolation || true
./isaaclab.sh --install
EOF
      ;;
    install_curobo_or_assets)
      if [[ "$TARGET" == "univtac" ]]; then
        cat >>"$commands_file" <<EOF
cd "$repo/third_party"
test -d curobo || git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout 0a50de1ba72db304195d59d9d0b1ed269696047f
"$env_prefix/bin/python" -m pip install warp-lang==1.0.0 --no-build-isolation
"$env_prefix/bin/python" -m pip install -e . --no-build-isolation
EOF
      else
        cat >>"$commands_file" <<EOF
cd "$repo"
./scripts/setup_assets.sh
EOF
      fi
      ;;
    install_tacex_core)
      if [[ "$TARGET" == "univtac" ]]; then
        tacex_root="$repo/third_party/TacEx"
      else
        tacex_root="$repo"
      fi
      cat >>"$commands_file" <<EOF
cd "$tacex_root"
./tacex.sh -i
EOF
      ;;
    build_uipc)
      if [[ "$TARGET" == "univtac" ]]; then
        tacex_root="$repo/third_party/TacEx"
      else
        tacex_root="$repo"
      fi
      cat >>"$commands_file" <<EOF
cd "$tacex_root"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CMAKE_CUDA_ARCHITECTURES=89
"$env_prefix/bin/python" -m pip install -e source/tacex_uipc -v --no-build-isolation
EOF
      ;;
    setup_assets)
      if [[ "$TARGET" == "univtac" ]]; then
        cat >>"$commands_file" <<EOF
cd "$repo/third_party/TacEx"
# Verify modified bundled TacEx assets before official UniVTAC collection.
find source/tacex_assets/tacex_assets/data -maxdepth 3 -type f | sed -n '1,20p'
EOF
      else
        cat >>"$commands_file" <<EOF
cd "$repo"
./scripts/setup_assets.sh
EOF
      fi
      ;;
    official_sanity)
      if [[ "$TARGET" == "univtac" ]]; then
        cat >>"$commands_file" <<EOF
cd "$repo"
bash collect_data.sh grasp_classify demo 0
EOF
      else
        cat >>"$commands_file" <<EOF
cd "$repo"
"$env_prefix/bin/python" scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight
EOF
      fi
      ;;
  esac
}

write_commands

status="dry_run"
blocker=""
target_python="$env_prefix/bin/python"
{
  echo "PHASE00_REFERENCE_ENV_STAGE_START"
  echo "TARGET=$TARGET"
  echo "STAGE=$STAGE"
  echo "EXECUTE=$EXECUTE"
  echo "HOST=$host"
  echo "REPO=$repo"
  echo "EXPECTED_COMMIT=$expected_commit"
  echo "OBSERVED_COMMIT=${observed_commit:-missing}"
  echo "ENV_PREFIX=$env_prefix"
  echo "COMMANDS_FILE=$commands_file"
  echo "NOTE=envprep_stage_not_training_not_compute_sanity"
  echo "COMMANDS_BEGIN"
  sed 's/^/  /' "$commands_file"
  echo "COMMANDS_END"

  if [[ ! -d "$repo/.git" ]]; then
    status="blocked_missing_repo"
    blocker="missing official repository"
  elif [[ "$observed_commit" != "$expected_commit" ]]; then
    status="blocked_commit_mismatch"
    blocker="expected $expected_commit observed $observed_commit"
  elif [[ ! -x "$CONDA_EXE" ]]; then
    status="blocked_missing_conda"
    blocker="missing conda executable at $CONDA_EXE"
  elif [[ "$EXECUTE" == "1" ]]; then
    status="blocked_execute_not_enabled_in_stage_runner"
    blocker="stage runner intentionally records commands only; execute commands manually only after review"
  elif [[ "$STAGE" == "preflight" ]]; then
    status="dry_run_preflight_ready"
  elif [[ "$STAGE" == "create_env" ]]; then
    status="dry_run_create_env_ready_not_executed"
  elif [[ ! -x "$target_python" ]]; then
    status="blocked_missing_target_env"
    blocker="missing target Python at $target_python; run controlled create_env before this stage"
  fi

  cat >"$status_path" <<EOF
{
  "target": "$TARGET",
  "stage": "$STAGE",
  "status": "$status",
  "blocker": "$blocker",
  "host": "$host",
  "repo": "$repo",
  "expected_commit": "$expected_commit",
  "observed_commit": "${observed_commit:-}",
  "env_prefix": "$env_prefix",
  "target_python": "$target_python",
  "commands_file": "$commands_file",
  "classification": "phase00_reference_envprep_stage_not_training_not_compute_sanity"
}
EOF

  cat >"$report_path" <<EOF
# Reference Env Stage: $TARGET / $STAGE

- Target: \`$TARGET\`
- Stage: \`$STAGE\`
- Status: \`$status\`
- Blocker: \`${blocker:-none}\`
- Host: \`$host\`
- Repository: \`$repo\`
- Expected commit: \`$expected_commit\`
- Observed commit: \`${observed_commit:-missing}\`
- Env prefix: \`$env_prefix\`
- Target Python: \`$target_python\`
- Commands file: \`$commands_file\`

This is environment-preparation planning only. It is not training, not an
official sanity result, and not curiosity progress.
EOF

  echo "STATUS_PATH=$status_path"
  echo "REPORT_PATH=$report_path"
  echo "PHASE00_REFERENCE_ENV_STAGE_END"
} 2>&1 | tee "$log_path"
