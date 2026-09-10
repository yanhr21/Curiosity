"""Stop the identified live trainer only after the user-requested step-700 save.

CPU process control only: no model load, tensor scan, digest, or GPU allocation.
The existing trainer is not hot-patched or restarted before the target boundary.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def read_json(path):
    return json.loads(path.read_text())


def publish(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def last_trace_row(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 131072))
        lines = stream.read().splitlines()
    return json.loads(lines[-1])


def process_identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] == "Z":
        raise ProcessLookupError(pid)
    return dict(pid=pid, ppid=int(stat[1]), pgid=int(stat[2]), start_ticks=int(stat[19]),
                command=(root / "cmdline").read_bytes().replace(b"\0", b" ").decode())


def checkpoint_ready(state, progress, row, target=700):
    if row["optimizer_step"] + 1 > target:
        raise ValueError("trainer passed requested stop boundary")
    return (state.get("protocol") == "paper_zero_wam_single_gpu_checkpoint_v2"
            and state.get("completed_optimizer_steps") == target
            and state.get("optimizer_boundary_complete") is True
            and state.get("completed_epochs") == target // 280
            and state.get("steps_in_current_epoch") == target % 280
            and state.get("hash_checks") is False
            and progress.get("optimizer_steps_completed") == target
            and progress.get("checkpoint_complete") is True
            and progress.get("formal_execution_complete") is False
            and row.get("optimizer_step") == target - 1
            and row.get("optimizer_applied") is True
            and row.get("amp_scaler_skipped") is False
            and row.get("all_trainable_parameters_finite") is True
            and row.get("gpu_parameter_master_readback_exact") is True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--torchrun-pid", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    root = args.formal_root.resolve()
    result_path = root / "STEP700_STOP_RESULT.json"
    armed_path = root / "STEP700_STOP_ARMED.json"
    if result_path.exists() or armed_path.exists():
        raise ValueError("refusing duplicate stop controller or overwritten evidence")
    if socket.gethostname().split(".")[0] != args.host:
        raise ValueError("wrong compute host")
    identity = process_identity(args.trainer_pid)
    if (identity["ppid"] != args.torchrun_pid or identity["pgid"] != args.trainer_pid
            or "paper_zero_wam.train_single_gpu --mode formal" not in identity["command"]
            or str(root) not in identity["command"]):
        raise ValueError("trainer process identity mismatch")
    environment = dict(item.split(b"=", 1) for item in
                       (Path("/proc") / str(args.trainer_pid) / "environ").read_bytes().split(b"\0")
                       if b"=" in item)
    if environment.get(b"SLURM_JOB_ID") != args.job_id.encode() or environment.get(b"SLURM_STEP_ID") != b"0":
        raise ValueError("trainer belongs to another allocation or compute step")
    trace_path = root / "TRAIN_TRACE.jsonl"
    row = last_trace_row(trace_path)
    if row["optimizer_step"] + 1 >= 700:
        raise ValueError("controller must be armed before update 700")
    evidence = dict(protocol="paper_zero_wam_user_step700_stop_v1", target_step=700,
                    trainer=identity, slurm_job_id=args.job_id, host=args.host,
                    armed_at=time.time(), observed_updates_at_arm=row["optimizer_step"] + 1,
                    allocation_release_requested=False, automatic_training_continuation=False,
                    hash_checks=False)
    publish(armed_path, evidence)
    print(json.dumps(dict(state="armed", **evidence)), flush=True)
    while True:
        if process_identity(args.trainer_pid) != identity:
            raise ValueError("trainer identity changed before target checkpoint")
        checkpoint = (root / "latest_checkpoint").resolve()
        state = read_json(checkpoint / "STATE.json")
        progress = read_json(root / "FORMAL_PROGRESS.json")
        try:
            row = last_trace_row(trace_path)
        except json.JSONDecodeError:
            # The writer may be appending its next line; never interpret a partial read as terminal.
            time.sleep(0.2)
            continue
        if checkpoint_ready(state, progress, row):
            if checkpoint.parent != root or checkpoint.name not in ("checkpoint_slot0", "checkpoint_slot1"):
                raise ValueError("unexpected checkpoint target")
            files = {name: (checkpoint / name).stat().st_size
                     for name in ("model_rank00.pt", "optimizer_rank00.pt", "STATE.json")}
            if any(size <= 0 for size in files.values()) or list(checkpoint.glob("*.tmp")):
                raise ValueError("checkpoint file publication incomplete")
            if process_identity(args.trainer_pid) != identity:
                raise ValueError("trainer identity changed before signal")
            # Only the verified trainer child PGID; never the retained srun/allocation shell.
            os.killpg(identity["pgid"], signal.SIGTERM)
            evidence.update(checkpoint_directory=str(checkpoint), checkpoint_state=state,
                            checkpoint_file_bytes=files, signal="SIGTERM", signalled_at=time.time())
            publish(root / "STEP700_STOP_SIGNAL.json", evidence)
            for _ in range(300):
                try:
                    current = process_identity(args.trainer_pid)
                except (FileNotFoundError, ProcessLookupError):
                    break
                if current != identity:
                    raise ValueError("PID reused while waiting for trainer exit")
                time.sleep(0.2)
            else:
                raise RuntimeError("identified trainer did not exit after SIGTERM")
            if last_trace_row(trace_path)["optimizer_step"] != 699:
                raise ValueError("unexpected extra applied update after step 700")
            pipeline_status = root.parent / "logs" / f"held_{args.job_id}_resume_latest.status"
            for _ in range(600):
                try:
                    process_identity(args.torchrun_pid)
                except (FileNotFoundError, ProcessLookupError):
                    if pipeline_status.exists():
                        break
                time.sleep(0.2)
            else:
                raise RuntimeError("training pipeline has not returned to the retained shell")
            evidence.update(training_stopped=True, completed_optimizer_steps=700,
                            formal_execution_complete=False, stopped_at=time.time(),
                            previous_pipeline_status=pipeline_status.read_text())
            publish(result_path, evidence)
            print(json.dumps(dict(state="training_stopped", **evidence)), flush=True)
            return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
