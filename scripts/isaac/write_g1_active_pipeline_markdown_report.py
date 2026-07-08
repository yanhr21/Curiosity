#!/usr/bin/env python3
"""Write a compact Markdown report for the active G1 carrying pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
DEFAULT_PIPELINE_STATUS = ROOT / "experiments/reports/2026-07-07_g1_active_pipeline_status_current.json"
DEFAULT_COMPLETION_AUDIT = ROOT / "experiments/reports/2026-07-07_g1_carry_completion_audit_current.json"
DEFAULT_FAILURE_CLASSIFICATION = ROOT / "experiments/reports/2026-07-07_g1_active_pipeline_failure_classification_current.json"
DEFAULT_NEXT_ACTIONS = ROOT / "experiments/reports/2026-07-07_g1_next_carry_actions_current.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-status", type=Path, default=DEFAULT_PIPELINE_STATUS)
    parser.add_argument("--completion-audit", type=Path, default=DEFAULT_COMPLETION_AUDIT)
    parser.add_argument("--failure-classification", type=Path, default=DEFAULT_FAILURE_CLASSIFICATION)
    parser.add_argument("--next-actions", type=Path, default=DEFAULT_NEXT_ACTIONS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text())


def _bullet_list(items: list[Any], fallback: str = "None") -> list[str]:
    if not items:
        return [f"- {fallback}"]
    return [f"- `{item}`" for item in items]


def main() -> int:
    args = parse_args()
    pipeline = _load(args.pipeline_status)
    audit = _load(args.completion_audit)
    failure = _load(args.failure_classification)
    next_actions = _load(args.next_actions)

    lines: list[str] = []
    lines.append("# G1 Carry Active Pipeline Status")
    lines.append("")
    lines.append(f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("This report is a read-only status summary, not a success claim.")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Pipeline status: `{pipeline.get('status')}`")
    lines.append(f"- Completion audit status: `{audit.get('status')}`")
    lines.append(f"- Failure classification status: `{failure.get('status')}`")
    lines.append(f"- Next-action report status: `{next_actions.get('status')}`")
    lines.append("")

    lines.append("## Slurm Jobs")
    lines.append("")
    jobs = list(pipeline.get("tracked_slurm_jobs") or [])
    if not jobs:
        lines.append("- No tracked Slurm job snapshot reported.")
    else:
        for job in jobs:
            lines.append(
                "- `{job_id}` `{name}` `{state}` elapsed `{elapsed}` start `{start}` reason/node `{reason}`".format(
                    job_id=job.get("job_id"),
                    name=job.get("name", ""),
                    state=job.get("state"),
                    elapsed=job.get("elapsed", ""),
                    start=job.get("start_time", ""),
                    reason=job.get("node_or_reason", job.get("error", "")),
                )
            )
    lines.append("")

    lines.append("## Missing Artifacts")
    lines.append("")
    lines.extend(_bullet_list(list(pipeline.get("missing_artifacts") or []), "No missing artifacts reported"))
    lines.append("")

    lines.append("## Failing Artifacts")
    lines.append("")
    lines.extend(_bullet_list(list(pipeline.get("failing_artifacts") or []), "No failing artifacts reported"))
    lines.append("")

    lines.append("## Completion Failures")
    lines.append("")
    lines.extend(_bullet_list(list(audit.get("completion_failures") or []), "No completion failures reported"))
    lines.append("")

    lines.append("## Failure Categories")
    lines.append("")
    lines.extend(_bullet_list(list(failure.get("categories") or []), "No failure categories reported"))
    lines.append("")

    lines.append("## Recommended Actions")
    lines.append("")
    actions = list(next_actions.get("actions") or [])
    if not actions:
        lines.append("- No recommended actions reported.")
    else:
        for action in actions:
            lines.append(f"- Priority `{action.get('priority')}`: `{action.get('action')}`")
            lines.append(f"  Reason: {action.get('reason')}")
    lines.append("")

    lines.append("## Source Reports")
    lines.append("")
    for label, path in (
        ("pipeline_status", args.pipeline_status),
        ("completion_audit", args.completion_audit),
        ("failure_classification", args.failure_classification),
        ("next_actions", args.next_actions),
    ):
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
