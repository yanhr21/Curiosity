#!/usr/bin/env python3
"""Export direct Isaac carry summaries as task-contract JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from direct_carry_task_contract import episode_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export direct Isaac carry task episodes to JSONL.")
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        default=[],
        help="Direct backend summary or all-posture summary.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_rows(path: Path) -> list[dict]:
    data = _load(path)
    rows = []
    if isinstance(data.get("postures"), list):
        for posture in data["postures"]:
            episode_summary = dict(posture)
            child_path_text = posture.get("summary_path")
            if child_path_text:
                child_path = Path(child_path_text)
                if child_path.exists():
                    child_summary = _load(child_path)
                    episode_summary = dict(child_summary)
                    episode_summary.update(posture)
            episode_id = f"{path.stem}:{posture.get('posture', len(rows))}"
            rows.append(
                episode_row(
                    source_summary=str(path),
                    episode_id=episode_id,
                    summary=episode_summary,
                    parent_summary=data,
                )
            )
    else:
        episode_id = f"{path.stem}:{data.get('carry_posture', data.get('controller_mode', 'episode'))}"
        rows.append(episode_row(source_summary=str(path), episode_id=episode_id, summary=data))
    return rows


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.summary:
        rows.extend(_iter_rows(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"output": str(args.output), "episode_count": len(rows), "schema": "direct_isaac_carry_task_episode_v1"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
