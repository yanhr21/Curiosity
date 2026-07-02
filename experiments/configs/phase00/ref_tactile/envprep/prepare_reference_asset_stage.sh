#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TARGET="${TARGET:-tacauchy}"
SOURCE="${SOURCE:-univtac_bundled_tacex}"
STAGE="${STAGE:-audit}"
EXECUTE="${EXECUTE:-0}"
ACK_ASSET_MUTATION="${ACK_ASSET_MUTATION:-}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/newton/phase00/ref_tactile/envprep/assets}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT/experiments/reports/phase00/ref_tactile/envprep/assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/experiments/outputs/phase00/ref_tactile/envprep/assets}"

usage() {
  cat <<'EOF'
Usage:
  TARGET=tacauchy SOURCE=univtac_bundled_tacex STAGE=audit|reuse_copy|verify EXECUTE=0 \
    bash experiments/configs/phase00/ref_tactile/envprep/prepare_reference_asset_stage.sh

Default EXECUTE=0 only writes commands, status, report, and log evidence.
EXECUTE=1 is intentionally blocked in this runner; asset mutation must be
approved and performed through a reviewed command path.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ "$TARGET" != "tacauchy" ]]; then
  usage >&2
  fail "TARGET currently must be tacauchy"
fi
if [[ "$SOURCE" != "univtac_bundled_tacex" ]]; then
  usage >&2
  fail "SOURCE currently must be univtac_bundled_tacex"
fi
case "$STAGE" in
  audit|reuse_copy|verify) ;;
  *) usage >&2; fail "unknown STAGE: $STAGE" ;;
esac
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  fail "refusing asset preparation inside Slurm allocation $SLURM_JOB_ID"
fi

host="$(hostname)"
source_root="$ROOT/external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data"
target_root="$ROOT/external/TaCauchy/source/tacex_assets/tacex_assets/data"
mkdir -p "$LOG_ROOT/$TARGET" "$REPORT_ROOT/$TARGET" "$OUTPUT_ROOT/$TARGET"
log_path="$LOG_ROOT/$TARGET/${STAGE}.log"
report_path="$REPORT_ROOT/$TARGET/${STAGE}.md"
status_path="$OUTPUT_ROOT/$TARGET/${STAGE}_status.json"
commands_file="$OUTPUT_ROOT/$TARGET/${STAGE}_commands.sh"

write_commands() {
  : >"$commands_file"
  case "$STAGE" in
    audit)
      cat >>"$commands_file" <<EOF
du -sh "$source_root" "$target_root"
find "$source_root" -maxdepth 3 -type f | wc -l
find "$target_root" -maxdepth 3 -type f | wc -l
test -f "$source_root/Sensors/GelSight_Mini/Sensor.usd"
test -d "$source_root/Props/tactile_test_shapes"
EOF
      ;;
    reuse_copy)
      cat >>"$commands_file" <<EOF
rsync -a --ignore-existing "$source_root/" "$target_root/"
EOF
      ;;
    verify)
      cat >>"$commands_file" <<EOF
test -f "$target_root/Sensors/GelSight_Mini/Sensor.usd"
test -d "$target_root/Props/tactile_test_shapes"
find "$target_root/Props/tactile_test_shapes" -maxdepth 1 -name '*.usd' | wc -l
find "$target_root/Sensors/GelSight_Mini" -maxdepth 3 -type f | sed -n '1,40p'
EOF
      ;;
  esac
}

write_commands

status="dry_run"
blocker=""
if [[ ! -d "$source_root" ]]; then
  status="blocked_missing_source_assets"
  blocker="missing source asset root $source_root"
elif [[ ! -d "$target_root" ]]; then
  status="blocked_missing_target_asset_root"
  blocker="missing target asset root $target_root"
elif [[ "$EXECUTE" == "1" ]]; then
  status="blocked_execute_not_enabled_in_asset_stage_runner"
  blocker="asset mutation requires explicit reviewed command execution; this runner records commands only"
elif [[ "$STAGE" == "audit" ]]; then
  status="dry_run_asset_audit_ready"
elif [[ "$STAGE" == "reuse_copy" ]]; then
  status="dry_run_asset_reuse_copy_not_executed"
elif [[ "$STAGE" == "verify" && ! -f "$target_root/Sensors/GelSight_Mini/Sensor.usd" ]]; then
  status="blocked_missing_reused_assets"
  blocker="target does not yet contain $target_root/Sensors/GelSight_Mini/Sensor.usd"
elif [[ "$STAGE" == "verify" ]]; then
  status="dry_run_asset_verify_ready"
fi

{
  echo "PHASE00_REFERENCE_ASSET_STAGE_START"
  echo "TARGET=$TARGET"
  echo "SOURCE=$SOURCE"
  echo "STAGE=$STAGE"
  echo "EXECUTE=$EXECUTE"
  echo "HOST=$host"
  echo "SOURCE_ROOT=$source_root"
  echo "TARGET_ROOT=$target_root"
  echo "COMMANDS_FILE=$commands_file"
  echo "STATUS=$status"
  echo "BLOCKER=${blocker:-none}"
  echo "COMMANDS_BEGIN"
  sed 's/^/  /' "$commands_file"
  echo "COMMANDS_END"

  cat >"$status_path" <<EOF
{
  "target": "$TARGET",
  "source": "$SOURCE",
  "stage": "$STAGE",
  "status": "$status",
  "blocker": "$blocker",
  "host": "$host",
  "source_root": "$source_root",
  "target_root": "$target_root",
  "commands_file": "$commands_file",
  "classification": "phase00_reference_asset_stage_not_executed"
}
EOF

  cat >"$report_path" <<EOF
# Reference Asset Stage: $TARGET / $STAGE

- Target: \`$TARGET\`
- Source: \`$SOURCE\`
- Stage: \`$STAGE\`
- Status: \`$status\`
- Blocker: \`${blocker:-none}\`
- Host: \`$host\`
- Source root: \`$source_root\`
- Target root: \`$target_root\`
- Commands file: \`$commands_file\`

This is asset-preparation planning only. It is not asset setup, not a download,
not a file copy, not official sanity, and not curiosity progress.
EOF

  echo "STATUS_PATH=$status_path"
  echo "REPORT_PATH=$report_path"
  echo "PHASE00_REFERENCE_ASSET_STAGE_END"
} 2>&1 | tee "$log_path"
