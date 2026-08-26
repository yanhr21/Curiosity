#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Run from an existing Slurm H200 shell inside tmux.  This launcher contains no
# human authorization state: every transition is decided by a checked JSON gate.

set -euo pipefail

case "$(hostname)" in
  mgmtserver*|login*)
    echo "refusing to run Newton physics on a login node" >&2
    exit 2
    ;;
esac

repo=/public/home/yanhongru/Curiosity
python_bin=/public/home/yanhongru/envs/isaac_arena_py312/bin/python
rsl_rl_root=/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/rsl_rl
export PYTHONPATH=/public/home/yanhongru/envs/newton_warp_114:${repo}/third_party/newton:${repo}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "${repo}"

teacher_run=experiments/sugar_reproduction/outputs/newton_refiner_ppo_20260827/carrybox_refiner_newton_seed171701_reset_fresh256
teacher_checkpoint=${teacher_run}/model_255.pt
teacher_training_result=${teacher_run}/TRAINING_RESULT.json
teacher_gate=experiments/sugar_reproduction/outputs/newton_refiner_open_loop_20260827/formal20_adapted256fresh/RESULT.json
tracker_data=experiments/sugar_reproduction/outputs/newton_refiner_dataset_20260827/rollout_datasets/refiner/rl_dataset
tracker_warm_start=SUGAR/demo_ckpts/CarryBox/tracker.pt
log_root=experiments/sugar_reproduction/outputs/newton_bcppo_h200_20260827/logs
smoke_name=carrybox_bcppo_acting_handoff_seed171702_smoke12
formal_name=carrybox_bcppo_acting_handoff_seed171702_formal3000
smoke_dir=${log_root}/${smoke_name}
formal_dir=${log_root}/${formal_name}

echo "AUTO_BCPPO_CHAIN_WAITING_FOR_REFINER_GATE"
while [[ ! -f "${teacher_gate}" ]]; do
  sleep 30
done

if ! "${python_bin}" - "${teacher_training_result}" "${teacher_gate}" <<'PY'
import json
import sys

training = json.load(open(sys.argv[1]))
gate = json.load(open(sys.argv[2]))
checks = gate.get("checks", {})
required = (
    "all_profiles_finished",
    "checkpoint_actor_is_exact_890_to_29_official_mlp",
    "no_active_profile_diverged",
    "physical_lift_fraction_passes",
    "strict_completion_fraction_passes",
)
passed = (
    training.get("pass") is True
    and gate.get("passed") is True
    and all(checks.get(key) is True for key in required)
)
print(
    "AUTO_REFINER_GATE",
    {
        "training_pass": training.get("pass"),
        "physical_pass": gate.get("passed"),
        "lifted": gate.get("lifted_profile_count"),
        "strict": gate.get("strict_complete_profile_count"),
        "required": gate.get("required_profile_count"),
    },
    flush=True,
)
raise SystemExit(0 if passed else 1)
PY
then
  echo "AUTO_BCPPO_CHAIN_REJECTED_REFINER_GATE"
  exit 0
fi

if [[ -e "${smoke_dir}" ]]; then
  echo "refusing to overwrite existing fresh smoke directory: ${smoke_dir}" >&2
  exit 3
fi

mkdir -p sugar_newton/_gpu_out
set -o pipefail
"${python_bin}" -m sugar_newton.rl.train_bcppo \
  --num-envs 1 \
  --max-iterations 12 \
  --save-interval 6 \
  --clips data_075_075_t0 \
  --motion-root "${tracker_data}" \
  --teacher-motion-root SUGAR/data/CarryBox \
  --teacher-ckpt "${teacher_checkpoint}" \
  --teacher-gate-result "${teacher_gate}" \
  --student-warm-start "${tracker_warm_start}" \
  --episode-length 300 \
  --seed 171702 \
  --run-name "${smoke_name}" \
  --log-root "${log_root}" \
  --logger tensorboard \
  --video-interval 0 \
  --rsl-rl-root "${rsl_rl_root}" \
  2>&1 | tee sugar_newton/_gpu_out/newton_bcppo_acting_handoff_seed171702_smoke12_h200.log

if ! "${python_bin}" - "${smoke_dir}/TRAINING_RESULT.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
required = {
    "passed": True,
    "acting_teacher_parameters_frozen": True,
    "distillation_teacher_parameters_frozen": True,
    "policy_parameters_finite": True,
    "teacher_prefix_ppo_credit": False,
}
failures = {
    key: result.get(key)
    for key, expected in required.items()
    if result.get(key) is not expected
}
if result.get("cumulative_handoffs", 0) <= 0:
    failures["cumulative_handoffs"] = result.get("cumulative_handoffs")
if result.get("cumulative_policy_control_steps", 0) <= 0:
    failures["cumulative_policy_control_steps"] = result.get(
        "cumulative_policy_control_steps"
    )
if result.get("divergence_rate", 1.0) > 0.005:
    failures["divergence_rate"] = result.get("divergence_rate")
print("AUTO_HANDOFF_SMOKE_GATE", {"failures": failures, **result}, flush=True)
raise SystemExit(1 if failures else 0)
PY
then
  echo "AUTO_BCPPO_CHAIN_REJECTED_HANDOFF_SMOKE"
  exit 0
fi

if [[ -e "${formal_dir}" ]]; then
  echo "refusing to overwrite existing fresh formal directory: ${formal_dir}" >&2
  exit 4
fi

echo "AUTO_BCPPO_FORMAL3000_START"
"${python_bin}" -m sugar_newton.rl.train_bcppo \
  --num-envs 8 \
  --max-iterations 3000 \
  --save-interval 250 \
  --motion-root "${tracker_data}" \
  --teacher-motion-root SUGAR/data/CarryBox \
  --teacher-ckpt "${teacher_checkpoint}" \
  --teacher-gate-result "${teacher_gate}" \
  --student-warm-start "${tracker_warm_start}" \
  --episode-length 300 \
  --seed 171702 \
  --run-name "${formal_name}" \
  --log-root "${log_root}" \
  --logger tensorboard \
  --video-interval 0 \
  --rsl-rl-root "${rsl_rl_root}" \
  2>&1 | tee sugar_newton/_gpu_out/newton_bcppo_acting_handoff_seed171702_formal3000_h200.log
