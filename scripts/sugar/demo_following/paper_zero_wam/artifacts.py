"""Small artifact-writing helpers shared by the executable experiment stages."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON result only after its complete contents reach a sibling file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    """Publish a UTF-8 text artifact with the same sibling-replace boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def emit_json_best_effort(payload: dict[str, Any], *, error: bool = False) -> None:
    """Emit observational JSON without changing a completed stage's exit status."""

    try:
        print(
            json.dumps(payload, sort_keys=True),
            file=sys.stderr if error else sys.stdout,
            flush=True,
        )
    except Exception:
        # Result/trace files are authoritative.  A broken Slurm log stream may
        # not retroactively fail a stage after those files were committed.
        return
