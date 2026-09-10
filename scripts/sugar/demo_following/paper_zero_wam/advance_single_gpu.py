"""Automatic post-render physics continuation inside the existing allocation.

Reuses the frozen scientific reducers and original PhysX evaluators. There are
no scheduler submissions, new models, threshold changes, or file digests.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .data import read_jsonl
from .evaluate_heldout import aggregate, heldout_groups, validate_score_prefix
from .results import refresh_results_document_best_effort
from .train import formal_training_decision
from .train_single_gpu import EXECUTION, single_gpu_checkpoint_step


def run(module, *arguments, distributed=False):
    bootstrap = PROJECT_ROOT / "scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py"
    command = [sys.executable, str(bootstrap)]
    if distributed:
        command += ["torch.distributed.run", "--standalone", "--nproc_per_node=1", str(bootstrap)]
    command += ["scripts.sugar.demo_following.paper_zero_wam." + module, *map(str, arguments)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def admission(path, protocol, admitted, reason):
    write_json_atomic(path, dict(protocol=protocol, admitted=admitted, reason=reason,
                                held_allocation=os.environ["SLURM_JOB_ID"],
                                execution_world_size=1, optimizer_updates_added=0,
                                threshold_or_training_sweep_started=False, hash_checks=False))
    refresh_results_document_best_effort()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_STEP_ID") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("physics continuation requires the existing held compute step")
    root = args.formal_root.resolve()
    config = PaperZeroWAMConfig()
    formal = json.loads((root / "FORMAL_TRAINING_RESULT.json").read_text())
    heldout = json.loads((root / "heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json").read_text())
    decision = formal_training_decision(root / "TRAIN_TRACE.jsonl",
        config.resolved(config.output_root) / "schedule/TRAIN_SCHEDULE.jsonl", config,
        execution_world_size=1, accumulation_steps=8)
    score_records = read_jsonl(root / "heldout_prompt_gate/HELDOUT_PROMPT_SCORES.jsonl")
    _, groups = heldout_groups(config)
    validate_score_prefix(score_records, groups, config)
    heldout_decision = aggregate(score_records)
    if (formal.get("protocol") != "paper_zero_wam_sugar_formal_v1"
            or formal.get("execution_completed") is not True
            or formal.get("optimizer_steps") != 4200
            or formal.get("architecture_parameter_count") != config.expected_parameter_count
            or formal.get("batch") != EXECUTION or formal.get("hash_checks") is not False
            or formal.get("formal_decision") != decision
            or formal.get("passed") is not decision["passed"]):
        raise ValueError("formal terminal differs from the complete single-GPU trace")
    if (heldout.get("protocol") != "paper_zero_wam_motion_disjoint_prompt_gate_v2"
            or heldout.get("execution_completed") is not True
            or heldout.get("checkpoint_step") != 4200
            or heldout.get("architecture_parameter_count") != config.expected_parameter_count
            or heldout.get("group_count") != 390 or heldout.get("score_instance_count") != 1950
            or heldout.get("decision") != heldout_decision
            or heldout.get("passed") is not heldout_decision["passed"]
            or heldout.get("hash_checks") is not False):
        raise ValueError("held-out terminal differs from all 390 complete score groups")
    passed = decision["passed"] and heldout_decision["passed"]
    reason = ("passed" if passed else "formal_training_decision_failed" if not decision["passed"]
              else "motion_disjoint_heldout_prompt_gate_failed")
    checkpoint = root / "latest_checkpoint"
    checkpoint_state = json.loads((checkpoint / "STATE.json").read_text())
    binding = heldout.get("score_journal_binding") or {}
    if (binding.get("checkpoint_directory") != str(checkpoint.resolve())
            or binding.get("checkpoint_state") != checkpoint_state):
        raise ValueError("held-out scores are not bound to the current formal checkpoint")
    if single_gpu_checkpoint_step(checkpoint_state, config) != 4200:
        raise ValueError("partial recovery checkpoint cannot enter physical evaluation")
    admission(root / "PHYSICAL_ADMISSION_RESULT.json", "paper_zero_wam_physical_admission_v1",
              passed, reason)
    if not passed:
        return
    smallbox_root = root / "smallbox_physical"
    for profile_batch in range(5):
        run("rollout_smallbox", "--single-gpu", "--checkpoint", checkpoint,
            "--output-root", smallbox_root, "--profile-batch", profile_batch, distributed=True)
    # Reopen every raw adapted/endpoint trace; never admit from summary counts alone.
    run("aggregate_smallbox", "--physical-root", smallbox_root)
    smallbox = json.loads((smallbox_root / "SMALLBOX_RESULT.json").read_text())
    passed = smallbox.get("passed") is True
    admission(root / "MOTION_DISJOINT_ADMISSION_RESULT.json", "paper_zero_wam_motion_disjoint_admission_v1",
              passed, "passed" if passed else "paired_smallbox_task_switch_failed")
    if not passed:
        return
    motion_root = root / "motion_disjoint_physical"
    for source_index in range(19):
        run("rollout_motion_disjoint", "--single-gpu", "--checkpoint", checkpoint,
            "--output-root", motion_root, "--source-index", source_index,
            "--case-manifest", root / "motion_disjoint_cases/MOTION_DISJOINT_CASES.json", distributed=True)
    run("aggregate_motion_disjoint", "--physical-root", motion_root,
        "--heldout-result", root / "heldout_prompt_gate/HELDOUT_PROMPT_RESULT.json",
        "--output", motion_root / "MOTION_DISJOINT_RESULT.json")
    refresh_results_document_best_effort()


if __name__ == "__main__":
    main()
