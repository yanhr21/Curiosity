#!/usr/bin/env bash
# Run the released Google Research XIRL/TCC trainer on clean SUGAR videos.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus}")"
RUN_ROOT="$(realpath -m "${2:-$ROOT/experiments/demo_following/official_xirl_tcc_v1/pretrain_runs}")"
EXPERIMENT_NAME="${3:-sugar_carry_kick_tcc_seed271402}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OFFICIAL_ROOT="$ROOT/experiments/runtime_assets/official_google_research_xirl/xirl"
TORCHKIT_ROOT="$ROOT/experiments/runtime_assets/official_torchkit_v0p0p2"
XMAGICAL_ROOT="$ROOT/experiments/runtime_assets/official_xmagical_v0p0p2"
COMPAT_DEPS="$ROOT/experiments/runtime_assets/official_xirl_py311_compat_deps"
COMPAT_CODE="$ROOT/scripts/sugar/demo_following/xirl_compat"
CONFIG="$ROOT/scripts/sugar/demo_following/xirl_configs/sugar_carry_kick_tcc.py"
DEVICE_PATCH="$COMPAT_CODE/official_xirl_pytorch_device.patch"
LOSS_FILE="$OFFICIAL_ROOT/xirl/losses.py"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside retained compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Official XIRL/TCC training requires a Slurm GPU job" >&2
    exit 2
fi
for path in "$DATA_ROOT/train/CarryBox" "$DATA_ROOT/train/KickBox" \
            "$DATA_ROOT/valid/CarryBox" "$DATA_ROOT/valid/KickBox" \
            "$DATA_ROOT/test/CarryBox" "$DATA_ROOT/test/KickBox"; do
    [[ -d "$path" ]] || { echo "missing XIRL corpus path: $path" >&2; exit 2; }
done
for result in "$DATA_ROOT/RENDER_RESULT_CarryBox_000_099.json" \
              "$DATA_ROOT/RENDER_RESULT_KickBox_000_098.json"; do
    jq -e '.passed == true and .frames_per_motion == 64 and .resolution == [320, 320]' \
        "$result" >/dev/null || { echo "invalid XIRL render result: $result" >&2; exit 2; }
done
for contract in \
    "train CarryBox 80" "valid CarryBox 10" "test CarryBox 10" \
    "train KickBox 80" "valid KickBox 10" "test KickBox 9"; do
    read -r split task videos <<<"$contract"
    path="$DATA_ROOT/$split/$task"
    actual_videos=$(find "$path" -mindepth 1 -maxdepth 1 -type d | wc -l)
    actual_frames=$(find "$path" -mindepth 2 -maxdepth 2 -type f -name '*.png' | wc -l)
    if [[ "$actual_videos" -ne "$videos" || "$actual_frames" -ne $((videos * 64)) ]]; then
        echo "incomplete XIRL corpus: $split/$task videos=$actual_videos frames=$actual_frames" >&2
        exit 2
    fi
done

# The released PyTorch 1.7 code creates the one-hot identity matrix on CPU and
# then indexes it with CUDA labels.  Modern PyTorch rejects that cross-device
# indexing.  Apply the tracked one-line device-preserving compatibility patch;
# its numerical loss definition is unchanged.
if grep -Fq 'torch.eye(K)[y]' "$LOSS_FILE"; then
    patch --batch --forward -d "$OFFICIAL_ROOT" -p1 < "$DEVICE_PATCH"
fi
grep -Fq 'torch.eye(K, device=y.device)[y]' "$LOSS_FILE" || {
    echo "official XIRL device compatibility patch is absent" >&2
    exit 2
}

mkdir -p "$RUN_ROOT"
XIRL_TMPDIR="/public/home/yanhongru/.xirl_tmp_${SLURM_JOB_ID}"
mkdir -p "$XIRL_TMPDIR"
export SUGAR_XIRL_DATA_ROOT="$DATA_ROOT"
export SUGAR_XIRL_RUN_ROOT="$RUN_ROOT"
export TMPDIR="$XIRL_TMPDIR"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONPATH="$COMPAT_CODE:$COMPAT_DEPS:$XMAGICAL_ROOT:$TORCHKIT_ROOT:$OFFICIAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$OFFICIAL_ROOT"
RUN_DIR="$RUN_ROOT/$EXPERIMENT_NAME"
RESULT="$RUN_DIR/temporal_retrieval_result.json"
latest_step=-1
if [[ -d "$RUN_DIR/checkpoints" ]]; then
    while IFS= read -r checkpoint; do
        candidate="$(basename "$checkpoint" .ckpt)"
        if [[ "$candidate" =~ ^[0-9]+$ ]]; then
            candidate_value=$((10#$candidate))
            if ((candidate_value > latest_step)); then
                latest_step="$candidate_value"
            fi
        fi
    done < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name '*.ckpt' | sort)
fi
if ((latest_step < 4001)); then
    resume_args=()
    if [[ -d "$RUN_DIR" ]]; then
        resume_args+=(--resume)
    fi
    "$PYTHON_BIN" pretrain.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --config "$CONFIG" \
        --device cuda:0 \
        "${resume_args[@]}"
fi

cd "$ROOT"
if ! jq -e '.protocol == "official_xirl_tcc_sugar_motion_disjoint_v1" and has("passed")' \
    "$RESULT" >/dev/null 2>&1; then
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_official_xirl_tcc_temporal_retrieval.py" \
        --corpus "$DATA_ROOT" \
        --run-dir "$RUN_DIR" \
        --output "$RESULT" \
        --device cuda:0
fi

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_XIRL_TCC_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
