#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${ROOT_DIR}/IsaacLab}"
ENV_SITE_PACKAGES="${ENV_SITE_PACKAGES:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
INFERENCE_VIDEO="${INFERENCE_VIDEO:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/released_inference/${TASK_NAME}/videos/play/rl-video-step-0.mp4}"
REFINER_LOG="${REFINER_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_carrybox_full_resume_model5000_cpu_partition_any.log}"
ACTIVE_PIPELINE_LOG="${ACTIVE_PIPELINE_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_downstream_refiner10000_cpu_active.log}"
SLURM_JOB_ID="${SLURM_JOB_ID:-178916}"
REFINER5000_EVAL_DIR="${REFINER5000_EVAL_DIR:-${OUTPUT_DIR}/eval/refiner_model5000_rollout_eval_novideo/raw_npz/trajectory_complete}"
REFINER5000_SUMMARY_LOG="${REFINER5000_SUMMARY_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260712_sugar_refiner5000_eval_summary.log}"
REFINER_TRAINING_CURVES="${REFINER_TRAINING_CURVES:-${OUTPUT_DIR}/visualizations/refiner_training_curves.png}"
REFINER5000_ROLLOUT_VIS="${REFINER5000_ROLLOUT_VIS:-${OUTPUT_DIR}/visualizations/refiner_model5000_rollout_summary.png}"
REFINER5000_ROLLOUT_VIS_JSON="${REFINER5000_ROLLOUT_VIS_JSON:-${OUTPUT_DIR}/visualizations/refiner_model5000_rollout_summary.json}"
REFINER10000_MODEL="${REFINER10000_MODEL:-${OUTPUT_DIR}/logs/refiner/model_10000.pt}"
REFINER10000_EXPORT="${REFINER10000_EXPORT:-${OUTPUT_DIR}/ckpts/refiner_model10000.pt}"
REFINER10000_PIPELINE_EXPORT_PROVENANCE="${REFINER10000_PIPELINE_EXPORT_PROVENANCE:-${OUTPUT_DIR}/ckpts/refiner.pt.provenance.txt}"
REFINER10000_STOP_LOG="${REFINER10000_STOP_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_refiner_model_10000_stop_watch.log}"
REFINER10000_CHECKPOINT_AUDIT_LOG="${REFINER10000_CHECKPOINT_AUDIT_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_refiner_model10000_checkpoint_audit.log}"
REFINER10000_EVAL_DIR="${REFINER10000_EVAL_DIR:-${OUTPUT_DIR}/eval/refiner_model10000_rollout_eval_novideo/raw_npz/trajectory_complete}"
REFINER10000_EVAL_LOG="${REFINER10000_EVAL_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_carrybox_refiner10000_rollout_eval.log}"
REFINER10000_VIDEO_LOG="${REFINER10000_VIDEO_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_carrybox_refiner10000_video_eval.log}"
REFINER10000_VIDEO="${REFINER10000_VIDEO:-${OUTPUT_DIR}/visualizations/refiner_model10000_rollout_video.mp4}"
REFINER10000_ROLLOUT_VIS="${REFINER10000_ROLLOUT_VIS:-${OUTPUT_DIR}/visualizations/refiner_model10000_rollout_summary.png}"
REFINER10000_ROLLOUT_VIS_JSON="${REFINER10000_ROLLOUT_VIS_JSON:-${OUTPUT_DIR}/visualizations/refiner_model10000_rollout_summary.json}"
REFINER10000_FULL_ROLLOUT_VIS="${REFINER10000_FULL_ROLLOUT_VIS:-${OUTPUT_DIR}/visualizations/refiner_model10000_full_rollout_summary.png}"
REFINER10000_FULL_ROLLOUT_VIS_JSON="${REFINER10000_FULL_ROLLOUT_VIS_JSON:-${OUTPUT_DIR}/visualizations/refiner_model10000_full_rollout_summary.json}"

missing=0
present=0
warn=0

pass() {
  present=$((present + 1))
  echo "[SUGAR-AUDIT] PASS $*"
}

fail() {
  missing=$((missing + 1))
  echo "[SUGAR-AUDIT] MISSING $*"
}

note() {
  warn=$((warn + 1))
  echo "[SUGAR-AUDIT] NOTE $*"
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ -s "${path}" ]]; then
    pass "${label}: ${path}"
  else
    fail "${label}: ${path}"
  fi
}

check_absent() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    pass "${label}: ${path}"
  else
    fail "${label}: unexpected file exists at ${path}"
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    pass "${label}: ${path}"
  else
    fail "${label}: ${path}"
  fi
}

check_glob_nonempty() {
  local label="$1"
  local pattern="$2"
  local old_nullglob
  old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  local matches=( ${pattern} )
  eval "${old_nullglob}" 2>/dev/null || shopt -u nullglob
  local count="${#matches[@]}"
  if (( count > 0 )); then
    pass "${label}: count=${count} pattern=${pattern}"
  else
    fail "${label}: pattern=${pattern}"
  fi
}

dist_version() {
  local package="$1"
  local old_nullglob
  old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  local metas=(
    "${ENV_SITE_PACKAGES}/${package}-"*.dist-info/METADATA
    "${ENV_SITE_PACKAGES}/${package//-/_}-"*.dist-info/METADATA
  )
  eval "${old_nullglob}" 2>/dev/null || shopt -u nullglob
  if (( ${#metas[@]} > 0 )); then
    local meta="${metas[0]}"
    grep -E '^Version:' "${meta}" | head -1 | awk '{print $2}'
  fi
}

check_dist() {
  local label="$1"
  local package="$2"
  local expected="${3:-}"
  local version
  version="$(dist_version "${package}")"
  if [[ -z "${version}" ]]; then
    fail "${label}: ${package} dist-info missing under ${ENV_SITE_PACKAGES}"
  elif [[ -n "${expected}" && "${version}" != "${expected}" ]]; then
    fail "${label}: ${package}==${version}, expected ${expected}"
  else
    pass "${label}: ${package}==${version}"
  fi
}

echo "[SUGAR-AUDIT] checked=$(date '+%F %T %Z')"
echo "[SUGAR-AUDIT] root=${ROOT_DIR}"
echo "[SUGAR-AUDIT] sugar_dir=${SUGAR_DIR}"
echo "[SUGAR-AUDIT] isaaclab_dir=${ISAACLAB_DIR}"
echo "[SUGAR-AUDIT] output_dir=${OUTPUT_DIR}"

if [[ -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  pass "vendored official SUGAR source exists"
  echo "[SUGAR-AUDIT] sugar_upstream_commit=$(tr -d '[:space:]' < "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT")"
else
  fail "vendored official SUGAR source exists: ${SUGAR_DIR}"
fi

if [[ -f "${ISAACLAB_DIR}/VERSION" ]]; then
  isaaclab_version="v$(tr -d '[:space:]' < "${ISAACLAB_DIR}/VERSION")-curiosity-glue"
  if [[ "${isaaclab_version}" == v2.3.0* ]]; then
    pass "vendored IsaacLab v2.3.0-compatible source: ${isaaclab_version}"
  else
    fail "vendored IsaacLab v2.3.0-compatible source: ${isaaclab_version}"
  fi
else
  fail "vendored IsaacLab source exists: ${ISAACLAB_DIR}"
fi

check_dir "official CarryBox data" "${SUGAR_DIR}/data/${TASK_NAME}"
check_file "official G1 URDF" "${SUGAR_DIR}/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
check_file "official small box USD" "${SUGAR_DIR}/descriptions/objects/small_box/obj_aligned.usd"
check_file "official demo tracker checkpoint" "${SUGAR_DIR}/demo_ckpts/${TASK_NAME}/tracker.pt"
check_file "official demo generator checkpoint" "${SUGAR_DIR}/demo_ckpts/${TASK_NAME}/generator.ckpt"

check_dist "Python 3.11 env package" "isaacsim" "5.1.0.0"
check_dist "Python 3.11 env package" "isaaclab" "0.47.2"
check_dist "Python 3.11 env package" "sugar-rl" "0.1.0"
check_dist "Python 3.11 env package" "sugar-il" "0.1.0"
check_dist "Python 3.11 env package" "rsl-rl-lib" "3.0.1"
check_dist "Python 3.11 env package" "numpy" "1.26.0"
check_dist "downstream package" "zarr" "2.12.0"
check_dist "downstream package" "numcodecs" "0.12.1"
check_dist "downstream package" "hydra-core"
check_dist "downstream package" "omegaconf"
check_dist "downstream package" "diffusers" "0.32.1"
check_dist "downstream package" "accelerate" "1.2.1"
check_dist "downstream package" "timm" "1.0.12"
check_dist "downstream package" "datasets" "2.6.1"
check_dist "downstream package" "numba"
check_dist "downstream package" "pydantic" "2.11.4"

check_file "official inference smoke video" "${INFERENCE_VIDEO}"
check_file "full refiner periodic checkpoint model_0" "${OUTPUT_DIR}/logs/refiner/model_0.pt"
check_file "full refiner periodic checkpoint model_1000" "${OUTPUT_DIR}/logs/refiner/model_1000.pt"
check_file "full refiner periodic checkpoint model_2000" "${OUTPUT_DIR}/logs/refiner/model_2000.pt"
check_file "operator-selected refiner stop checkpoint model_10000" "${REFINER10000_MODEL}"
check_file "operator-selected refiner model_10000 immutable copy" "${REFINER10000_EXPORT}"
check_file "operator-selected downstream refiner provenance" "${REFINER10000_PIPELINE_EXPORT_PROVENANCE}"
check_file "model_10000 exact-stop watcher log" "${REFINER10000_STOP_LOG}"
check_file "model_10000 internal-iteration audit log" "${REFINER10000_CHECKPOINT_AUDIT_LOG}"
check_absent "operator stop boundary has no model_11000" "${OUTPUT_DIR}/logs/refiner/model_11000.pt"
latest_refiner_model="$(
  find "${OUTPUT_DIR}/logs/refiner" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -nE 's/^model_([0-9]+)\.pt$/\1/p' \
    | sort -n \
    | tail -1
)"
if [[ -n "${latest_refiner_model}" ]]; then
  echo "[SUGAR-AUDIT] latest_refiner_periodic_checkpoint=model_${latest_refiner_model}.pt"
  case "${latest_refiner_model}" in
    0|1000|2000)
      ;;
    *)
      check_file "latest full refiner periodic checkpoint model_${latest_refiner_model}" "${OUTPUT_DIR}/logs/refiner/model_${latest_refiner_model}.pt"
      ;;
  esac
fi
check_file "full refiner final checkpoint model_30000" "${OUTPUT_DIR}/logs/refiner/model_30000.pt"
check_file "operator-selected downstream refiner export" "${OUTPUT_DIR}/ckpts/refiner.pt"
if [[ -s "${OUTPUT_DIR}/logs/refiner/model_30000.pt" && -s "${OUTPUT_DIR}/ckpts/refiner.pt" ]]; then
  pass "paper-schedule full refiner checkpoint and export are both present"
else
  fail "paper-schedule full refiner checkpoint and export are both present"
fi
check_glob_nonempty "5000-step refiner eval complete trajectories" "${REFINER5000_EVAL_DIR}/*.npz"
check_file "5000-step refiner eval summary" "${REFINER5000_SUMMARY_LOG}"
check_file "refiner training curve visualization" "${REFINER_TRAINING_CURVES}"
check_file "5000-step refiner rollout visualization" "${REFINER5000_ROLLOUT_VIS}"
check_file "5000-step refiner rollout visualization data" "${REFINER5000_ROLLOUT_VIS_JSON}"
if [[ -s "${REFINER5000_SUMMARY_LOG}" ]]; then
  grep -E '^\[SUGAR-REFINER5000-SUMMARY\] (sampled_refiner_window_completion_rate_percent=|comparable_to_paper=|paper_table1_sugar_carrybox_train_sr=|paper_table1_sugar_carrybox_test_sr=)' "${REFINER5000_SUMMARY_LOG}" \
    | tail -8 || true
fi

check_glob_nonempty "10000-step refiner eval complete trajectories" "${REFINER10000_EVAL_DIR}/*.npz"
check_file "10000-step refiner eval log" "${REFINER10000_EVAL_LOG}"
check_file "10000-step refiner video eval log" "${REFINER10000_VIDEO_LOG}"
check_file "10000-step refiner rollout video" "${REFINER10000_VIDEO}"
check_file "10000-step refiner rollout visualization" "${REFINER10000_ROLLOUT_VIS}"
check_file "10000-step refiner rollout visualization data" "${REFINER10000_ROLLOUT_VIS_JSON}"
check_file "10000-step full refiner rollout visualization" "${REFINER10000_FULL_ROLLOUT_VIS}"
check_file "10000-step full refiner rollout visualization data" "${REFINER10000_FULL_ROLLOUT_VIS_JSON}"

check_glob_nonempty "refiner rollout raw complete trajectories" "${OUTPUT_DIR}/rollout_datasets/refiner/raw_npz/trajectory_complete/*.npz"
check_dir "refiner processed RL dataset" "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset"
check_file "tracker final checkpoint model_30000" "${OUTPUT_DIR}/logs/tracker/model_30000.pt"
check_file "tracker exported checkpoint" "${OUTPUT_DIR}/ckpts/tracker.pt"
check_glob_nonempty "tracker rollout raw complete trajectories" "${OUTPUT_DIR}/rollout_datasets/tracker/raw_npz/trajectory_complete/*.npz"
check_dir "tracker processed IL zarr dataset" "${OUTPUT_DIR}/rollout_datasets/tracker/il_dataset"
check_file "generator final checkpoint epoch=1000" "${OUTPUT_DIR}/logs/generator/epoch_checkpoints/epoch=1000.ckpt"
check_file "generator exported checkpoint" "${OUTPUT_DIR}/ckpts/generator.ckpt"

# Visualization is a completion gate once the corresponding trained artifact
# exists. Before that point the missing model/dataset remains the primary gate.
if [[ -s "${OUTPUT_DIR}/ckpts/tracker.pt" ]]; then
  check_file "tracker training curve visualization" "${OUTPUT_DIR}/visualizations/tracker_training_curves.png"
elif compgen -G "${OUTPUT_DIR}/logs/tracker/events.out.tfevents.*" >/dev/null; then
  check_file "live tracker training curve visualization" "${OUTPUT_DIR}/visualizations/tracker_training_curves.png"
fi
if [[ -s "${OUTPUT_DIR}/ckpts/generator.ckpt" ]]; then
  check_file "generator training curve visualization" "${OUTPUT_DIR}/visualizations/generator_training_curves.png"
fi

if [[ -s "${REFINER_LOG}" ]]; then
  latest_iteration="$(
    tail -500 "${REFINER_LOG}" \
      | sed -E 's/\x1b\[[0-9;]*[[:alpha:]]//g' \
      | sed -nE 's/.*Learning iteration[[:space:]]+([0-9]+)\/([0-9]+).*/\1\/\2/p' \
      | tail -1
  )"
  if [[ -n "${latest_iteration}" ]]; then
    echo "[SUGAR-AUDIT] latest_refiner_progress=${latest_iteration}"
  fi
  if grep -Eq 'Traceback \(most recent call last\)|FileNotFoundError|Boost\.Python\.ArgumentError|RuntimeError|CUDA out of memory|\[Error\]' "${REFINER_LOG}"; then
    fail "full refiner log has no fatal-pattern matches: ${REFINER_LOG}"
  else
    pass "full refiner log has no fatal-pattern matches: ${REFINER_LOG}"
  fi
else
  fail "full refiner log exists: ${REFINER_LOG}"
fi

if [[ -s "${ACTIVE_PIPELINE_LOG}" ]]; then
  active_iteration="$(
    tail -500 "${ACTIVE_PIPELINE_LOG}" \
      | sed -E 's/\x1b\[[0-9;]*[[:alpha:]]//g' \
      | sed -nE 's/.*Learning iteration[[:space:]]+([0-9]+)\/([0-9]+).*/\1\/\2/p' \
      | tail -1
  )"
  if [[ -n "${active_iteration}" ]]; then
    echo "[SUGAR-AUDIT] latest_active_pipeline_rl_progress=${active_iteration}"
  fi
  if grep -Eq 'Traceback \(most recent call last\)|FileNotFoundError|Boost\.Python\.ArgumentError|RuntimeError|CUDA out of memory|\[Error\]' "${ACTIVE_PIPELINE_LOG}"; then
    fail "active downstream pipeline log has no fatal-pattern matches: ${ACTIVE_PIPELINE_LOG}"
  else
    pass "active downstream pipeline log has no fatal-pattern matches: ${ACTIVE_PIPELINE_LOG}"
  fi
else
  fail "active downstream pipeline log exists: ${ACTIVE_PIPELINE_LOG}"
fi

if command -v squeue >/dev/null 2>&1; then
  if squeue -h -j "${SLURM_JOB_ID}" 2>/dev/null | grep -q .; then
    note "tracked refiner Slurm job ${SLURM_JOB_ID} is still active"
    squeue -h -j "${SLURM_JOB_ID}" -o '[SUGAR-AUDIT] slurm %.18i %.12P %.28j %.8T %.12M %.10l %.20R' 2>/dev/null
  else
    note "tracked refiner Slurm job ${SLURM_JOB_ID} is not active in squeue"
  fi
  squeue -h -u "${USER}" -o '%i|%P|%j|%T|%M|%l|%R' 2>/dev/null \
    | awk -F'|' '$3 ~ /^sugar_/ {print "[SUGAR-AUDIT] active_sugar_job=" $0}' || true
fi

echo "[SUGAR-AUDIT] summary_present=${present}"
echo "[SUGAR-AUDIT] summary_missing=${missing}"
echo "[SUGAR-AUDIT] summary_notes=${warn}"

if (( missing > 0 )); then
  echo "[SUGAR-AUDIT] reproduction_status=incomplete"
  exit 1
fi

echo "[SUGAR-AUDIT] reproduction_status=artifact_set_present"
