#!/usr/bin/env bash
# Execute one autonomous stage of the full-width paper Zero-WAM experiment.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/usr/bin/python3.10
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
SCHEDULE="$EXPERIMENT/schedule/TRAIN_SCHEDULE.jsonl"
IMPORT_BOOTSTRAP="$ROOT/scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py"
STAGE=${1:?stage must be precompute, overfit, formal, evaluate_heldout, render_openloop, physical_smallbox, aggregate_smallbox, motion_disjoint, or aggregate_motion_disjoint}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "paper Zero-WAM stages run only inside Slurm" >&2
    exit 2
fi
case "$(hostname)" in
    login*|mgmtserver*) echo "paper Zero-WAM stages cannot run on a login node" >&2; exit 2 ;;
esac
GPU_COUNT=${SLURM_GPUS_ON_NODE:-0}
if [[ "$STAGE" == "aggregate_smallbox" || "$STAGE" == "aggregate_motion_disjoint" ]]; then
    if [[ "$GPU_COUNT" != "0" ]]; then
        echo "physical aggregation is a CPU-only stage" >&2
        exit 2
    fi
elif [[ "$STAGE" == "precompute" ]]; then
    if [[ "$GPU_COUNT" != "1" && "$GPU_COUNT" != "8" ]]; then
        echo "VAE materialization requires one or eight H200 GPUs" >&2
        exit 2
    fi
elif [[ "$GPU_COUNT" != "8" ]]; then
    echo "paper Zero-WAM model training requires one complete 8xH200 node" >&2
    exit 2
fi

export PYTHONPATH="$PZW_SITE_PACKAGES:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$EXPERIMENT/logs"

case "$STAGE" in
    precompute)
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node="$GPU_COUNT" "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.precompute_latents \
            --cache-root "$EXPERIMENT/latents"
        ;;
    overfit)
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.train \
            --mode overfit \
            --overfit-steps 32 \
            --schedule "$SCHEDULE" \
            --output-dir "$EXPERIMENT/overfit"
        ;;
    formal)
        "$PYTHON_BIN" -c '
import json
from pathlib import Path
p = Path("'"$EXPERIMENT"'/overfit/OVERFIT_RESULT.json")
r = json.loads(p.read_text())
if (r.get("protocol") != "paper_zero_wam_sugar_overfit_v1"
        or r.get("execution_completed") is not True
        or int(r.get("optimizer_steps", -1)) != 32):
    raise SystemExit("the single full-width 32-update overfit did not complete")
'
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.train \
            --mode formal \
            --schedule "$SCHEDULE" \
            --output-dir "$EXPERIMENT/formal"
        ;;
    evaluate_heldout)
        "$PYTHON_BIN" -c '
import json
from pathlib import Path
p = Path("'"$EXPERIMENT"'/formal/FORMAL_TRAINING_RESULT.json")
r = json.loads(p.read_text())
if (r.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or r.get("execution_completed") is not True
        or int(r.get("optimizer_steps", -1)) != 4200
        or not isinstance(r.get("formal_decision"), dict)):
    raise SystemExit("formal 4,200-step training did not complete its decision")
'
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.evaluate_heldout \
            --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
            --output-dir "$EXPERIMENT/formal/heldout_prompt_gate"
        ;;
    render_openloop)
        "$PYTHON_BIN" -c '
import json
from pathlib import Path
root = Path("'"$EXPERIMENT"'/formal")
formal = json.loads((root / "FORMAL_TRAINING_RESULT.json").read_text())
heldout = json.loads((root / "heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json").read_text())
if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or formal.get("execution_completed") is not True
        or int(formal.get("optimizer_steps", -1)) != 4200
        or heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or heldout.get("execution_completed") is not True
        or int(heldout.get("checkpoint_step", -1)) != 4200
        or int(heldout.get("group_count", -1)) != 390
        or int(heldout.get("score_instance_count", -1)) != 1950
        or not isinstance(heldout.get("decision"), dict)):
    raise SystemExit("formal/held-out terminal contract invalid at predictive render")
'
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.render_openloop \
            --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
            --output-dir "$EXPERIMENT/formal/heldout_openloop_videos"
        ;;
    physical_smallbox)
        "$PYTHON_BIN" -c '
import json
from pathlib import Path
heldout = json.loads(Path("'"$EXPERIMENT"'/formal/heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json").read_text())
formal = json.loads(Path("'"$EXPERIMENT"'/formal/FORMAL_TRAINING_RESULT.json").read_text())
if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or formal.get("execution_completed") is not True
        or int(formal.get("optimizer_steps", -1)) != 4200
        or heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or heldout.get("execution_completed") is not True
        or int(heldout.get("checkpoint_step", -1)) != 4200
        or int(heldout.get("score_instance_count", -1)) != 1950):
    raise SystemExit("formal/held-out terminal contract invalid")
if formal.get("passed") is not True or heldout.get("passed") is not True:
    raise SystemExit("formal and motion-disjoint held-out gates did not admit physical execution")
'
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.rollout_smallbox \
            --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
            --output-root "$EXPERIMENT/formal/smallbox_physical" \
            --profile-batch "${SLURM_ARRAY_TASK_ID:?physical job requires array index 0..4}"
        ;;
    aggregate_smallbox)
        exec "$PYTHON_BIN" \
            -m scripts.sugar.demo_following.paper_zero_wam.aggregate_smallbox \
            --physical-root "$EXPERIMENT/formal/smallbox_physical"
        ;;
    motion_disjoint)
        "$PYTHON_BIN" -c '
import json
from pathlib import Path
root = Path("'"$EXPERIMENT"'/formal")
formal = json.loads((root / "FORMAL_TRAINING_RESULT.json").read_text())
heldout = json.loads((root / "heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json").read_text())
smallbox = json.loads((root / "smallbox_physical/SMALLBOX_RESULT.json").read_text())
expected_prompts = [
    {
        "split": "train",
        "task": task,
        "source_motion_id": source_motion_id,
        "latent_key": "prompt_latents",
    }
    for task, source_motion_id in (("CarryBox", 45), ("KickBox", 21))
]
if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
        or formal.get("execution_completed") is not True
        or int(formal.get("optimizer_steps", -1)) != 4200
        or heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
        or heldout.get("execution_completed") is not True
        or int(heldout.get("checkpoint_step", -1)) != 4200
        or int(heldout.get("score_instance_count", -1)) != 1950
        or smallbox.get("protocol") != "paper_zero_wam_smallbox_physical_result_v1"
        or int(smallbox.get("checkpoint_step", -1)) != 4200
        or int(smallbox.get("architecture_parameter_count", -1)) != 10658724829
        or int(smallbox.get("profile_count", -1)) != 20
        or int(smallbox.get("adapted_rollout_count", -1)) != 40
        or int(smallbox.get("released_endpoint_rollout_count", -1)) != 40
        or int(smallbox.get("rollout_count", -1)) != 80
        or smallbox.get("prompt_specs") != expected_prompts
        or smallbox.get("visualization_frame_counts") != [131] * 20
        or len(smallbox.get("videos", [])) != 20
        or smallbox.get("hash_checks") is not False):
    raise SystemExit("formal/held-out/SMALLBOX terminal contract invalid")
if not all(value.get("passed") is True for value in (formal, heldout, smallbox)):
    raise SystemExit("formal, held-out, and SMALLBOX gates did not admit motion-disjoint execution")
'
        exec "$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
            --standalone --nproc_per_node=8 "$IMPORT_BOOTSTRAP" \
            scripts.sugar.demo_following.paper_zero_wam.rollout_motion_disjoint \
            --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
            --output-root "$EXPERIMENT/formal/motion_disjoint_physical" \
            --case-manifest "$EXPERIMENT/formal/motion_disjoint_cases/MOTION_DISJOINT_CASES.json" \
            --source-index "${SLURM_ARRAY_TASK_ID:?motion-disjoint job requires array index 0..18}"
        ;;
    aggregate_motion_disjoint)
        exec "$PYTHON_BIN" \
            -m scripts.sugar.demo_following.paper_zero_wam.aggregate_motion_disjoint \
            --physical-root "$EXPERIMENT/formal/motion_disjoint_physical" \
            --heldout-result "$EXPERIMENT/formal/heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json" \
            --output "$EXPERIMENT/formal/motion_disjoint_physical/MOTION_DISJOINT_RESULT.json"
        ;;
    *)
        echo "unknown stage: $STAGE" >&2
        exit 2
        ;;
esac
