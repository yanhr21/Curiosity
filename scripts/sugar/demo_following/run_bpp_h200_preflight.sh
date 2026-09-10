#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/public/home/yanhongru/Curiosity"
PYTHON_BIN="/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PROJECT_ROOT}/experiments/demo_following/bpp_official_v1/preflight_${STAMP}"
ZERO_WAM_OUT="${PROJECT_ROOT}/experiments/demo_following/zero_wam_official_v1/official_release_status_${STAMP}"
BPP_ARTIFACT_ROOT="${PROJECT_ROOT}/experiments/demo_following/bpp_official_v1/artifacts"
BPP_SOURCE_DIR="${BPP_ARTIFACT_ROOT}/behavior_prompting_b5b494e"
BPP_CHECKPOINT_DIR="${BPP_ARTIFACT_ROOT}/liberogen_goal_chain_ab2ce394"
BPP_CHECKPOINT_FILE="${BPP_CHECKPOINT_DIR}/liberogen_goal_chain_behavior_prompting.ckpt"
BPP_EXAMPLE_DIR="${BPP_ARTIFACT_ROOT}/libero_gen_goal_chain_example_412c273"
BPP_EXAMPLE_REL="demonstration_data/libero_goal_chain_firststep_view/open_the_top_drawer_demo.hdf5"
BPP_EXAMPLE_FILE="${BPP_EXAMPLE_DIR}/${BPP_EXAMPLE_REL}"
BPP_EXPECTED_COMMIT="b5b494ed05fdd5d79e57b8b27072d3fa109ebfd0"
BPP_HF_REVISION="ab2ce394ad4a30b83356083ebdd32cb528be87f2"
BPP_CHECKPOINT_SHA256="f37e769c9460810be5f59b0ed77e1503c9dbbf2ed9780da85fe2760aac0ef7b4"
BPP_CHECKPOINT_BYTES="6915881550"
BPP_EXAMPLE_REVISION="412c273b2c284183032a75eef67b7ee596d94efa"
BPP_EXAMPLE_SHA256="1bb805496e2069df1174bc22a7db1b88481abf7da861a7a09f4f85d354895988"
BPP_EXAMPLE_BYTES="400515452"
BPP_CORPUS_CONTRACT_SHA256="eeba619a94154cfaa705606b145f3cb7c11ab7343c5960fd7fcde3d0a21ce5ed"
BPP_CAMERA_CONTRACT_SHA256="9276b901a33b788fb03b93a1b5fe2abaf5093bb44cf51376617e5ea787165daf"
BPP_INTERFACE_CONTRACT_SHA256="b7b1780672fcf3462e707fcbf769fc6479834c471ba2b34815db05cdea36644a"
BPP_SCHEDULE_CONTRACT_SHA256="bfe38f38b660d38305a5caba6c778997bd772bf69b1ab383362d9fdad33ec213"

# The official checkpoint is a multi-gigabyte Xet-backed object.  Cluster
# compute nodes reach the Hub through an SSH-forwarded HTTP proxy; the regular
# Hub/LFS downloader is substantially more reliable on that route and still
# resolves the exact pinned revision and SHA256 below.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this preflight must run inside a Slurm allocation" >&2
  exit 2
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if ! grep -q "H200" <<<"${GPU_NAMES}"; then
  echo "ERROR: expected an H200 compute node, found: ${GPU_NAMES}" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/preflight.log") 2>&1

cd "${PROJECT_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=${GPU_NAMES//$'\n'/,}"
echo "STARTED_AT=$(date --iso-8601=seconds)"

PYTHONPYCACHEPREFIX="${OUT_DIR}/pycache" \
  "${PYTHON_BIN}" -m compileall -q scripts/sugar/demo_following

while IFS= read -r shell_file; do
  bash -n "${shell_file}"
done < <(find scripts/sugar/demo_following -type f -name '*.sh' -print | sort)

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("scripts/sugar/demo_following/config").glob("*.json")):
    with path.open("r", encoding="utf-8") as stream:
        json.load(stream)
    print(f"JSON_OK {path}")
PY

sha256sum \
  scripts/sugar/demo_following/config/bpp_five_variant_corpus_v1.json \
  scripts/sugar/demo_following/config/bpp_dual_camera_candidates_v1.json \
  scripts/sugar/demo_following/config/bpp_official_interface_adapter_v1.json \
  scripts/sugar/demo_following/config/bpp_exact_training_schedule_v1.json \
  > "${OUT_DIR}/contract_sha256.txt"
cat > "${OUT_DIR}/contract_sha256_expected.txt" <<EOF
${BPP_CORPUS_CONTRACT_SHA256}  scripts/sugar/demo_following/config/bpp_five_variant_corpus_v1.json
${BPP_CAMERA_CONTRACT_SHA256}  scripts/sugar/demo_following/config/bpp_dual_camera_candidates_v1.json
${BPP_INTERFACE_CONTRACT_SHA256}  scripts/sugar/demo_following/config/bpp_official_interface_adapter_v1.json
${BPP_SCHEDULE_CONTRACT_SHA256}  scripts/sugar/demo_following/config/bpp_exact_training_schedule_v1.json
EOF
sha256sum --check "${OUT_DIR}/contract_sha256_expected.txt"

git diff --check
git status --short > "${OUT_DIR}/git_status_short.txt"

set +e
bash scripts/sugar/demo_following/run_zero_wam_official_release_audit.sh "${ZERO_WAM_OUT}"
ZERO_WAM_STATUS=$?
set -e

printf '%s\n' "${ZERO_WAM_STATUS}" > "${OUT_DIR}/zero_wam_audit_exit_code.txt"
printf '%s\n' "${ZERO_WAM_OUT}" > "${OUT_DIR}/zero_wam_audit_output_dir.txt"

mkdir -p "${BPP_ARTIFACT_ROOT}"
if [[ ! -e "${BPP_SOURCE_DIR}" ]]; then
  git clone --recurse-submodules \
    https://github.com/real-stanford/behavior_prompting.git \
    "${BPP_SOURCE_DIR}"
elif [[ ! -d "${BPP_SOURCE_DIR}/.git" ]]; then
  echo "ERROR: BPP source path exists but is not a Git checkout: ${BPP_SOURCE_DIR}" >&2
  exit 4
fi

git -C "${BPP_SOURCE_DIR}" fetch origin main --tags --force
BPP_ORIGIN="$(git -C "${BPP_SOURCE_DIR}" remote get-url origin)"
if [[ "${BPP_ORIGIN}" != "https://github.com/real-stanford/behavior_prompting.git" ]]; then
  echo "ERROR: unexpected BPP origin: ${BPP_ORIGIN}" >&2
  exit 5
fi
BPP_CURRENT_MAIN="$(git -C "${BPP_SOURCE_DIR}" rev-parse origin/main)"
if ! git -C "${BPP_SOURCE_DIR}" cat-file -e "${BPP_EXPECTED_COMMIT}^{commit}"; then
  git -C "${BPP_SOURCE_DIR}" fetch origin "${BPP_EXPECTED_COMMIT}"
fi
git -C "${BPP_SOURCE_DIR}" cat-file -e "${BPP_EXPECTED_COMMIT}^{commit}"
git -C "${BPP_SOURCE_DIR}" checkout --detach "${BPP_EXPECTED_COMMIT}"
BPP_COMMIT="$(git -C "${BPP_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${BPP_COMMIT}" != "${BPP_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: failed to checkout audited BPP commit ${BPP_EXPECTED_COMMIT}; found ${BPP_COMMIT}" >&2
  exit 6
fi
git -C "${BPP_SOURCE_DIR}" submodule sync --recursive
git -C "${BPP_SOURCE_DIR}" submodule update --init --recursive
if [[ -n "$(git -C "${BPP_SOURCE_DIR}" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: official BPP checkout is dirty after recursive checkout" >&2
  exit 7
fi

printf '%s\n' "${BPP_ORIGIN}" > "${OUT_DIR}/bpp_origin.txt"
printf '%s\n' "${BPP_CURRENT_MAIN}" > "${OUT_DIR}/bpp_current_main_commit.txt"
printf '%s\n' "${BPP_COMMIT}" > "${OUT_DIR}/bpp_commit.txt"
git -C "${BPP_SOURCE_DIR}" ls-tree -r --full-tree "${BPP_COMMIT}" \
  > "${OUT_DIR}/bpp_recursive_tree.txt"
git -C "${BPP_SOURCE_DIR}" submodule status --recursive \
  > "${OUT_DIR}/bpp_submodules.txt"

mkdir -p "${BPP_CHECKPOINT_DIR}" "${BPP_EXAMPLE_DIR}"
checkpoint_ready=false
checkpoint_actual_bytes=""
checkpoint_actual_sha256=""
if [[ -s "${BPP_CHECKPOINT_FILE}" ]]; then
  checkpoint_actual_bytes="$(stat --printf='%s' "${BPP_CHECKPOINT_FILE}")"
  checkpoint_actual_sha256="$(sha256sum "${BPP_CHECKPOINT_FILE}" | cut -d' ' -f1)"
  if [[ "${checkpoint_actual_bytes}" == "${BPP_CHECKPOINT_BYTES}" ]] \
    && [[ "${checkpoint_actual_sha256}" == "${BPP_CHECKPOINT_SHA256}" ]]; then
    checkpoint_ready=true
    echo "BPP_CHECKPOINT_CACHED_VERIFIED=${BPP_CHECKPOINT_FILE}"
  fi
fi

example_ready=false
example_actual_bytes=""
example_actual_sha256=""
if [[ -s "${BPP_EXAMPLE_FILE}" ]]; then
  example_actual_bytes="$(stat --printf='%s' "${BPP_EXAMPLE_FILE}")"
  example_actual_sha256="$(sha256sum "${BPP_EXAMPLE_FILE}" | cut -d' ' -f1)"
  if [[ "${example_actual_bytes}" == "${BPP_EXAMPLE_BYTES}" ]] \
    && [[ "${example_actual_sha256}" == "${BPP_EXAMPLE_SHA256}" ]]; then
    example_ready=true
    echo "BPP_OFFICIAL_EXAMPLE_CACHED_VERIFIED=${BPP_EXAMPLE_FILE}"
  fi
fi

if [[ "${checkpoint_ready}" != true || "${example_ready}" != true ]]; then
  "${PYTHON_BIN}" - <<PY
from huggingface_hub import hf_hub_download

if "${checkpoint_ready}" != "true":
    path = hf_hub_download(
        repo_id="austinpatel/liberogen_goal_chain",
        filename="liberogen_goal_chain_behavior_prompting.ckpt",
        revision="${BPP_HF_REVISION}",
        local_dir="${BPP_CHECKPOINT_DIR}",
    )
    print(f"BPP_CHECKPOINT={path}")

if "${example_ready}" != "true":
    example_path = hf_hub_download(
        repo_id="austinpatel/libero_gen_goal_chain_hdf5",
        repo_type="dataset",
        filename="${BPP_EXAMPLE_REL}",
        revision="${BPP_EXAMPLE_REVISION}",
        local_dir="${BPP_EXAMPLE_DIR}",
    )
    print(f"BPP_OFFICIAL_EXAMPLE={example_path}")
PY
fi

test -s "${BPP_CHECKPOINT_FILE}"
test -s "${BPP_EXAMPLE_FILE}"
if [[ "${checkpoint_ready}" != true ]]; then
  checkpoint_actual_bytes="$(stat --printf='%s' "${BPP_CHECKPOINT_FILE}")"
  checkpoint_actual_sha256="$(sha256sum "${BPP_CHECKPOINT_FILE}" | cut -d' ' -f1)"
fi
if [[ "${example_ready}" != true ]]; then
  example_actual_bytes="$(stat --printf='%s' "${BPP_EXAMPLE_FILE}")"
  example_actual_sha256="$(sha256sum "${BPP_EXAMPLE_FILE}" | cut -d' ' -f1)"
fi
printf '%s  %s\n' "${checkpoint_actual_sha256}" "${BPP_CHECKPOINT_FILE}" \
  > "${OUT_DIR}/bpp_checkpoint_sha256.txt"
printf '%s  %s\n' "${BPP_CHECKPOINT_SHA256}" "${BPP_CHECKPOINT_FILE}" \
  > "${OUT_DIR}/bpp_checkpoint_sha256_expected.txt"
printf 'bytes=%s\n' "${checkpoint_actual_bytes}" \
  > "${OUT_DIR}/bpp_checkpoint_bytes.txt"
printf '%s  %s\n' "${example_actual_sha256}" "${BPP_EXAMPLE_FILE}" \
  > "${OUT_DIR}/bpp_official_example_sha256.txt"
printf '%s  %s\n' "${BPP_EXAMPLE_SHA256}" "${BPP_EXAMPLE_FILE}" \
  > "${OUT_DIR}/bpp_official_example_sha256_expected.txt"
printf 'bytes=%s\n' "${example_actual_bytes}" \
  > "${OUT_DIR}/bpp_official_example_bytes.txt"
if [[ "${checkpoint_actual_bytes}" != "${BPP_CHECKPOINT_BYTES}" ]] \
  || [[ "${checkpoint_actual_sha256}" != "${BPP_CHECKPOINT_SHA256}" ]]; then
  echo "ERROR: checkpoint bytes or SHA256 differ from frozen artifact" >&2
  exit 8
fi
if [[ "${example_actual_bytes}" != "${BPP_EXAMPLE_BYTES}" ]] \
  || [[ "${example_actual_sha256}" != "${BPP_EXAMPLE_SHA256}" ]]; then
  echo "ERROR: official example bytes or SHA256 differ from frozen artifact" >&2
  exit 9
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

import h5py

required_obs = {
    "agentview_rgb",
    "eye_in_hand_rgb",
    "ee_pos",
    "ee_ori",
    "gripper_states",
}
report = {
    "dataset_revision": "${BPP_EXAMPLE_REVISION}",
    "relative_path": "${BPP_EXAMPLE_REL}",
    "demos": [],
}
with h5py.File("${BPP_EXAMPLE_FILE}", "r") as handle:
    if "data" not in handle:
        raise RuntimeError("official example lacks /data")
    demo_names = sorted(handle["data"].keys())
    if len(demo_names) < 2:
        raise RuntimeError("official pair-prompt example needs at least two demos")
    for demo_name in demo_names:
        demo = handle["data"][demo_name]
        if "actions" not in demo or "obs" not in demo:
            raise RuntimeError(f"{demo_name} lacks actions or obs")
        missing_obs = required_obs - set(demo["obs"].keys())
        if missing_obs:
            raise RuntimeError(f"{demo_name} lacks obs keys {sorted(missing_obs)}")
        action_shape = tuple(demo["actions"].shape)
        if len(action_shape) != 2 or action_shape[1] != 7:
            raise RuntimeError(f"unexpected raw action shape for {demo_name}: {action_shape}")
        rows = action_shape[0]
        obs_shapes = {}
        for key in sorted(required_obs):
            shape = tuple(demo["obs"][key].shape)
            if not shape or shape[0] != rows:
                raise RuntimeError(f"row mismatch for {demo_name}/{key}: {shape} vs {rows}")
            obs_shapes[key] = list(shape)
        report["demos"].append({
            "name": demo_name,
            "action_shape": list(action_shape),
            "obs_shapes": obs_shapes,
        })
report["demo_count"] = len(report["demos"])
report["total_action_rows"] = sum(x["action_shape"][0] for x in report["demos"])
report["passed"] = True
Path("${OUT_DIR}/bpp_official_example_structure.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import torch

scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
if scanner is None:
    raise RuntimeError("PyTorch lacks non-executing checkpoint global scanner")
allowed = {
    "_codecs.encode",
    "collections.OrderedDict",
    "collections.defaultdict",
    "dill._dill._load_type",
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.listconfig.ListConfig",
    "omegaconf.nodes.AnyNode",
    "torch.FloatStorage",
    "torch._utils._rebuild_tensor_v2",
    "typing.Any",
}
found = set(scanner("${BPP_CHECKPOINT_FILE}"))
unexpected = found - allowed
report = {
    "checkpoint_sha256": "${BPP_CHECKPOINT_SHA256}",
    "hf_revision": "${BPP_HF_REVISION}",
    "allowed_globals": sorted(allowed),
    "found_globals": sorted(found),
    "unexpected_globals": sorted(unexpected),
    "passed": not unexpected,
}
Path("${OUT_DIR}/bpp_checkpoint_pickle_scan.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
Path("${OUT_DIR}/bpp_checkpoint_unsafe_globals.txt").write_text(
    "\n".join(sorted(found)) + ("\n" if found else ""),
    encoding="utf-8",
)
if unexpected:
    raise RuntimeError(f"unexpected checkpoint pickle globals: {sorted(unexpected)}")
PY

echo "FINISHED_AT=$(date --iso-8601=seconds)"
echo "ZERO_WAM_STATUS=${ZERO_WAM_STATUS}"
echo "BPP_COMMIT=${BPP_COMMIT}"

if [[ "${ZERO_WAM_STATUS}" -ne 0 ]]; then
  echo "Zero-WAM remains fail-closed or the official audit failed; inspect the archived result before any Zero-WAM training."
fi
