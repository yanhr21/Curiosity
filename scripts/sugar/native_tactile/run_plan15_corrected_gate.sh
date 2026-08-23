#!/usr/bin/env bash
# Recalibrate corrected force channels, then execute all three live runtime preflights.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "this gate must run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
output_root=$1
device=${2:-cuda:0}
python_bin=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
if [[ "$output_root" != /* ]]; then
    output_root="$repo_root/$output_root"
fi
case "$output_root" in
    "$repo_root"/experiments/*) ;;
    *) echo "OUTPUT_ROOT must remain below $repo_root/experiments" >&2; exit 2 ;;
esac
if [[ -e "$output_root" ]]; then
    echo "refusing to overwrite corrected gate output: $output_root" >&2
    exit 2
fi
mkdir -p "$output_root"

export PYTHONPATH="$repo_root/IsaacLab/source/isaaclab:$repo_root/IsaacLab/source/isaaclab_assets:$repo_root/IsaacLab/source/isaaclab_contrib:$repo_root/IsaacLab/source/isaaclab_rl:$repo_root/SUGAR/source/sugar_rl:$repo_root/SUGAR/source/sugar_il:$repo_root/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"
export CURIOSITY_ANATOMICAL_TACSL_CONTACT_OFFSET_M=0.0003
export CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS=7294.8755
export CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS=9

calibration="$output_root/corrected_motion45_scale_sweep"
env -u VK_ICD_FILENAMES "$python_bin" -u \
    "$repo_root/scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py" \
    --output-root "$calibration" \
    --motion-folder "$repo_root/SUGAR/data/CarryBox/data_045" \
    --motion-id 0 \
    --max-steps 420 \
    --jump-delay-frames 30 \
    --device "$device"

scale_file="$calibration/patch_channel_scales.json"
"$python_bin" - "$scale_file" "$calibration/leakage_audit.json" "$calibration/slip_evaluation.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = "plan15_live_patch_channel_scales_v3_extent_offset_calibrated"
if payload.get("schema") != expected:
    raise SystemExit(f"unexpected corrected scale schema: {payload.get('schema')}")
if len(payload.get("patch_channel_scales", [])) != 9:
    raise SystemExit("corrected scale file does not contain nine channels")

leakage = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if leakage.get("fixed_action_max_abs_error") != 0.0:
    raise SystemExit("paired calibration did not replay identical actions")
if leakage.get("paired_jump_frame_max_error") != 0:
    raise SystemExit("paired calibration mass events are not frame matched")
if leakage.get("actor_observation_width") != 504:
    raise SystemExit("calibration actor observation is not the deployable 504-D contract")
if leakage.get("actor_contains_measured_object_state") is not False:
    raise SystemExit("calibration actor leaked measured object state")

slip = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
binary = slip["binary_slip_contact_supported"]
if binary["precision"] < 0.8 or binary["recall"] < 0.8:
    raise SystemExit(
        "corrected causal slip failed the predeclared 0.8 precision/recall gate"
    )
onset = slip["onset_detection"]
onset_total = onset["detected"] + onset["missed"]
if onset_total == 0 or onset["missed"] / onset_total > 0.2:
    raise SystemExit("corrected causal slip missed more than 20% of slip onsets")
if slip["loaded_contact_loss_alert"]["no_contact_nonloss_slip_alerts"] != 0:
    raise SystemExit("corrected causal slip alerts without contact-loss evidence")
PY

for branch in Z P PS; do
    task="Sugar-G129dof-CarryBox-OnlineMass-Patch-${branch}-Preflight-BCPPO"
    branch_root="$output_root/preflight_${branch,,}"
    env -u VK_ICD_FILENAMES "$python_bin" -u \
        "$repo_root/SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py" \
        --task "$task" \
        --patch-scale-file "$scale_file" \
        --seed 151014 \
        --num_envs 1 \
        --log_dir "$branch_root" \
        --headless \
        --device "$device"
    "$python_bin" - "$branch_root/plan15_live_preflight.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("overall_pass"):
    raise SystemExit(f"runtime preflight failed: {path}")
PY
done

"$python_bin" - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = {
    "schema": "plan15_corrected_source_runtime_gate_v1",
    "motion": "CarryBox/data_045 local motion 0",
    "scale_file": str(root / "corrected_motion45_scale_sweep/patch_channel_scales.json"),
    "preflights": {
        branch.upper(): str(root / f"preflight_{branch}/plan15_live_preflight.json")
        for branch in ("z", "p", "ps")
    },
    "overall_pass": True,
    "next_step": "if every preflight passes, run the fixed 3x PS overfit next; any failed preflight stops this branch",
}
(root / "gate_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
