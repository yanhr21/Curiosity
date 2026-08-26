#!/usr/bin/env bash
# Exercise the official CHORD adapter through the real SUGAR/PhysX training stack.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:?smoke output directory is required}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_chord_smoke_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""
export SUGAR_CROSS_SKILL_RECOVERY=1
export SUGAR_CROSS_SKILL_CARRY_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/tracker.pt"
export SUGAR_CROSS_SKILL_KICK_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt"
export SUGAR_CROSS_SKILL_CARRY_GENERATOR_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
export SUGAR_CROSS_SKILL_CARRY_PREFIX_STEPS=33
export SUGAR_CROSS_SKILL_CARRY_PREFIX_SCHEDULE=33
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1
export SUGAR_TRANSITION_SELECTED_SKILL_ID=1
export SUGAR_TRANSITION_RECOVERY_REWARD=1
export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$OUTPUT/prefix_audit.json"
export SUGAR_OFFICIAL_CHORD_REWARD=1
export SUGAR_OFFICIAL_CHORD_ROOT="$ROOT/experiments/runtime_assets/official_chord_5654c50e"
export SUGAR_OFFICIAL_CHORD_REFERENCE_GEOMETRY="$ROOT/experiments/demo_following/sugar_demo_chord_geometry_v2/kick21/contact_geometry.npz"
export SUGAR_OFFICIAL_CHORD_OBJECT_USD="$ROOT/SUGAR/descriptions/objects/big_box/obj_aligned.usd"
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

cd "$ROOT/SUGAR"
"$PYTHON_BIN" -u scripts/sugar_rl/train.py \
    --task Sugar-G129dof-KickBox-CausalTemporalActionComposition \
    --num_envs 4 --max_iterations 1 --seed 171648 \
    --log_dir "$OUTPUT/train" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"

jq -e '
  .official_chord_runtime_reward.enabled == true and
  .official_chord_runtime_reward.reward_calls > 0 and
  .official_chord_runtime_reward.live_physx_contact_points_and_forces == true and
  .official_chord_runtime_reward.actor_observation_augmented == false and
  .official_chord_runtime_reward.binary_contact_label_used == false
' "$OUTPUT/prefix_audit.json" >/dev/null
printf 'smoke_pass=1\n' > "$OUTPUT/SMOKE_STATUS.env"
