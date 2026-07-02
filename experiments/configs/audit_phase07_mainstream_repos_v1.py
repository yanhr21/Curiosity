"""Audit official mainstream baseline repository reachability for Phase07.

This script is intentionally lightweight: it does not clone repositories,
download checkpoints, train models, or run inference. It records whether the
official repositories are reachable and whether local official clones or
matching checkpoints already exist in the Curiosity workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


CANDIDATES = [
    {
        "name": "OpenPI_pi0",
        "repo": "https://github.com/Physical-Intelligence/openpi.git",
        "expected_local_dirs": ["external/openpi"],
        "checkpoint_globs": ["*openpi*", "*pi0*", "*pi05*"],
    },
    {
        "name": "Diffusion_Policy",
        "repo": "https://github.com/real-stanford/diffusion_policy.git",
        "expected_local_dirs": ["external/diffusion_policy"],
        "checkpoint_globs": ["*diffusion*"],
    },
    {
        "name": "Open_X_RT_X",
        "repo": "https://github.com/google-deepmind/open_x_embodiment.git",
        "expected_local_dirs": ["external/open_x_embodiment"],
        "checkpoint_globs": ["*rtx*", "*openx*", "*open_x*"],
    },
    {
        "name": "NVIDIA_Isaac_GR00T",
        "repo": "https://github.com/NVIDIA/Isaac-GR00T.git",
        "expected_local_dirs": ["external/Isaac-GR00T", "external/isaac-gr00t"],
        "checkpoint_globs": ["*gr00t*", "*groot*"],
    },
]


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "exit_code": None,
            "stdout": exc.stdout.strip() if isinstance(exc.stdout, str) else "",
            "stderr": "timeout",
        }


def _glob_any(root: Path, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    if not root.exists():
        return hits
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                try:
                    hits.append(str(path.relative_to(root.parent)))
                except ValueError:
                    hits.append(str(path))
    return sorted(set(hits))


def _candidate_status(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    ls_remote = _run(["git", "ls-remote", candidate["repo"], "HEAD"], cwd=root)
    head = None
    if ls_remote["exit_code"] == 0 and ls_remote["stdout"]:
        head = ls_remote["stdout"].split()[0]
    local_dirs = []
    for item in candidate["expected_local_dirs"]:
        path = root / item
        local_dirs.append({"path": item, "exists": path.exists(), "is_git_repo": (path / ".git").exists()})
    checkpoint_hits = _glob_any(root / "checkpoints", candidate["checkpoint_globs"])
    return {
        "name": candidate["name"],
        "official_repo": candidate["repo"],
        "repo_reachable": ls_remote["exit_code"] == 0 and head is not None,
        "head": head,
        "ls_remote": ls_remote,
        "local_official_dirs": local_dirs,
        "matching_checkpoint_files_under_checkpoints": checkpoint_hits,
        "phase07_comparison_status": "not_satisfied",
        "required_next_step": "Clone/use official code and checkpoints for a faithful Phase07 comparison, or document a concrete official incompatibility blocker.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_mainstream_repo_reachability_audit_v1_20260627.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = [_candidate_status(root, candidate) for candidate in CANDIDATES]
    payload = {
        "classification": "phase07_mainstream_repo_reachability_audit_v1",
        "status": "pass_reachability_audit_gate_still_open",
        "not_training": True,
        "not_success_claim": True,
        "root": str(root),
        "candidates": candidates,
        "phase07_mainstream_gate_satisfied": False,
        "reason": "Reachability and local-file audit only; no official baseline has been run or blocked.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
