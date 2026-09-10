"""Permit identical recovery only for observed scheduler interruptions.

Missing output alone never establishes an infrastructure failure. This module
uses only the standard library and does not load a model or calculate digests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


INTERRUPTED_STATES = {"TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}
LIVE_STATES = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
NUMERICAL_FAILURE = re.compile(
    r"FloatingPointError:|non-finite loss at step|inactive/non-finite gradient at step"
    r"|inactive/non-finite update at step|non-finite trainable parameter after"
    r"|failed its execution trace contract"
)


def classify_job(job_id: str, rows: list[dict[str, str]], numerical_failure: bool) -> dict:
    exact = [row for row in rows if row.get("job_id") == job_id]
    state = exact[0]["state"].split()[0].rstrip("+") if len(exact) == 1 else None
    if numerical_failure:
        reason = "numerical_or_execution_contract_failure"
    elif len(exact) != 1:
        reason = "missing_or_ambiguous_scheduler_record"
    elif state in LIVE_STATES:
        reason = "original_job_still_live"
    elif state in INTERRUPTED_STATES:
        reason = "scheduler_confirmed_infrastructure_interruption"
    else:
        reason = "termination_does_not_prove_infrastructure_interruption"
    return {
        "job_id": job_id,
        "scheduler_records": exact,
        "numerical_failure_in_log": numerical_failure,
        "recovery_allowed": reason == "scheduler_confirmed_infrastructure_interruption",
        "reason": reason,
    }


def inspect_job(job_id: str, logs: Path, prefix: str) -> dict:
    # -X excludes batch/extern steps; JobID (not JobIDRaw) preserves array IDs.
    completed = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id,
         "--format=JobID%64,State%64,ExitCode"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3:
            rows.append(dict(zip(("job_id", "state", "exit_code"), fields[:3])))
    paths = [logs / f"{prefix}_{job_id}.{extension}" for extension in ("out", "err")]
    matches = []
    # Scan complete logs: an earlier numerical error may precede a timeout.
    for path in paths:
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    if NUMERICAL_FAILURE.search(line):
                        matches.append({"path": str(path), "line": number,
                                        "text": line.strip()[:1000]})
    decision = classify_job(job_id, rows, bool(matches))
    decision["failure_log_evidence"] = matches
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = {
        "protocol": "paper_zero_wam_infrastructure_recovery_decision_v1",
        "job_key": args.job_key, "recovery_allowed": False, "hash_checks": False,
    }
    try:
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
        base = str(jobs[args.job_key])
        if not re.fullmatch(r"[0-9]+", base):
            raise ValueError("expected exact numeric Slurm job ID")
        if args.indices and not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.indices):
            raise ValueError("expected explicit comma-separated array indices")
        indices = args.indices.split(",") if args.indices else []
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate array indices")
        ids = [f"{base}_{int(index)}" for index in indices] or [base]
        records = [inspect_job(job, args.jobs.parent / "logs", args.log_prefix) for job in ids]
        decision.update(jobs=records, recovery_allowed=all(r["recovery_allowed"] for r in records))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        decision.update(reason="scheduler_evidence_unavailable", error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(decision, sort_keys=True))
    if not decision["recovery_allowed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
