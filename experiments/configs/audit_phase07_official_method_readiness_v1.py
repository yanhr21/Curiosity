"""Audit official-method readiness for Phase07 mainstream comparison.

This is a lightweight, read-only gate. It does not install environments,
download checkpoints, materialize datasets, train, run inference, or claim that
curiosity beats mainstream methods. Its purpose is to prevent a repository
clone, a stage-1 index, or a diagnostic adapter from being treated as a
completed official-method comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


METHODS = {
    "openpi_pi0": {
        "repo_dir": "external/openpi",
        "env_candidates": ["envs/openpi/.venv", "envs/openpi_pi0/.venv"],
        "blocker_candidates": [
            "experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/openpi_pi0_checkpoint_blocker.json"
        ],
        "checkpoint_globs": ["*openpi*", "*pi0*", "*pi05*"],
        "stage1_files": [
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/openpi_lerobot_stage1/episodes.jsonl",
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/openpi_lerobot_stage1/openpi_phase07_mapping.json",
        ],
        "runner_candidates": [
            "experiments/configs/run_phase07_openpi_official_comparison_in_alloc.sh",
            "experiments/configs/eval_phase07_openpi_official_policy_v1.py",
        ],
        "official_basis": "OpenPI official LeRobot data path plus official checkpoint/policy config.",
    },
    "gr00t": {
        "repo_dir": "external/Isaac-GR00T",
        "env_candidates": ["envs/gr00t/.venv", "envs/isaac_gr00t/.venv"],
        "blocker_candidates": [
            "experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/gr00t_checkpoint_blocker.json"
        ],
        "checkpoint_globs": ["*gr00t*", "*groot*", "*n1*"],
        "stage1_files": [
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/episodes.jsonl",
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/meta/modality.json",
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/meta/info.json",
        ],
        "runner_candidates": [
            "experiments/configs/run_phase07_gr00t_official_comparison_in_alloc.sh",
            "experiments/configs/eval_phase07_gr00t_official_policy_v1.py",
        ],
        "official_basis": "Isaac GR00T official GR00T-flavored LeRobot v2 path plus official checkpoint.",
    },
    "diffusion_policy": {
        "repo_dir": "external/diffusion_policy",
        "env_candidates": ["envs/diffusion_policy/.venv", "envs/diffusion_policy/conda"],
        "blocker_candidates": [
            "experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/diffusion_policy_checkpoint_blocker.json"
        ],
        "checkpoint_globs": ["*diffusion*", "*pusht*", "*robomimic*"],
        "stage1_files": [
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/diffusion_policy_stage1/episodes.jsonl",
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/diffusion_policy_stage1/shape_meta.json",
        ],
        "runner_candidates": [
            "experiments/configs/run_phase07_diffusion_policy_official_comparison_in_alloc.sh",
            "experiments/configs/eval_phase07_diffusion_policy_official_policy_v1.py",
        ],
        "official_basis": "Official Diffusion Policy Dataset/EnvRunner/config path.",
    },
    "rtx": {
        "repo_dir": "external/open_x_embodiment",
        "env_candidates": ["envs/rtx/.venv", "envs/open_x_embodiment/.venv"],
        "blocker_candidates": [
            "experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/rtx_checkpoint_blocker.json"
        ],
        "checkpoint_globs": ["*rtx*", "*rt_1_x*", "*open_x*", "*openx*"],
        "stage1_files": [
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/rtx_stage1/episodes.jsonl",
            "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/rtx_stage1/rtx_phase07_mapping.json",
        ],
        "runner_candidates": [
            "experiments/configs/run_phase07_rtx_official_comparison_in_alloc.sh",
            "experiments/configs/eval_phase07_rtx_official_policy_v1.py",
        ],
        "official_basis": "Open X-Embodiment/RT-X official RGB+task+7D action path.",
    },
}


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _glob_check(root: Path, patterns: list[str]) -> list[str]:
    base_dirs = [root / "checkpoints", root / "external" / "checkpoints"]
    hits: list[str] = []
    for base in base_dirs:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path.is_file():
                    try:
                        hits.append(str(path.relative_to(root)))
                    except ValueError:
                        hits.append(str(path))
    return sorted(set(hits))


def _method_status(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    repo_path = root / spec["repo_dir"]
    envs = [{"path": item, "exists": (root / item).exists()} for item in spec["env_candidates"]]
    checkpoints = _glob_check(root, spec["checkpoint_globs"])
    blockers = []
    for item in spec.get("blocker_candidates", []):
        path = root / item
        blocker_status = None
        if path.is_file():
            try:
                blocker_status = json.loads(path.read_text(encoding="utf-8")).get("status")
            except Exception:  # noqa: BLE001
                blocker_status = "unreadable"
        blockers.append({"path": item, "exists": path.is_file(), "status": blocker_status})
    stage1 = [{"path": item, "exists": (root / item).is_file()} for item in spec["stage1_files"]]
    runners = [{"path": item, "exists": (root / item).is_file()} for item in spec["runner_candidates"]]
    checkpoint_access = None
    checkpoint_access_path = root / "experiments/outputs/phase07_official_checkpoint_access_v1_20260627.json"
    if checkpoint_access_path.is_file():
        try:
            access_payload = json.loads(checkpoint_access_path.read_text(encoding="utf-8"))
            checkpoint_access = access_payload.get("methods", {}).get(name)
        except Exception:  # noqa: BLE001
            checkpoint_access = {"status": "unreadable"}
    failures: list[str] = []
    if not repo_path.exists():
        failures.append("missing_official_repo")
    if repo_path.exists() and not (repo_path / ".git").exists():
        failures.append("repo_not_git_checkout")
    if not any(item["exists"] for item in envs):
        failures.append("missing_prepared_env_under_envs")
    filled_blocker_exists = any(
        item["exists"] and item.get("status") not in {None, "template_unfilled_not_a_blocker"} for item in blockers
    )
    if not checkpoints and not filled_blocker_exists:
        failures.append("missing_official_checkpoint_or_recorded_checkpoint_blocker")
    if not all(item["exists"] for item in stage1):
        failures.append("missing_stage1_dataset_index")
    if not any(item["exists"] for item in runners):
        failures.append("missing_official_closed_loop_comparison_runner")
    return {
        "method": name,
        "status": "pass" if not failures else "open_not_ready",
        "official_basis": spec["official_basis"],
        "repo": {"path": spec["repo_dir"], "exists": repo_path.exists(), "git_head": _git_head(repo_path)},
        "env_candidates": envs,
        "checkpoint_hits": checkpoints,
        "checkpoint_access_probe": checkpoint_access,
        "checkpoint_blocker_candidates": blockers,
        "stage1_files": stage1,
        "closed_loop_runner_candidates": runners,
        "failures": failures,
        "not_success_claim": True,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase07 Official Method Readiness V1",
        "",
        "Date: 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a read-only audit. It does not install environments, download checkpoints, train, infer, or claim mainstream comparison success.",
        "",
        "## Methods",
        "",
    ]
    for name, item in payload["methods"].items():
        lines.append(f"- `{name}`: `{item['status']}`; failures={item['failures']}")
    lines.extend(
        [
            "",
            "## Completion Impact",
            "",
            f"- official method comparison ready: `{payload['official_method_comparison_ready']}`",
            "- A repository clone, stage-1 index, or diagnostic adapter is not enough to satisfy the mainstream gate.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_official_method_readiness_v1_20260627.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/reports/2026-06-27_phase07_official_method_readiness_v1.md"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    methods = {name: _method_status(root, name, spec) for name, spec in METHODS.items()}
    ready = all(item["status"] == "pass" for item in methods.values())
    payload = {
        "classification": "phase07_official_method_readiness_v1",
        "status": "pass" if ready else "open_not_ready",
        "official_method_comparison_ready": ready,
        "methods": methods,
        "not_training": True,
        "not_env_install": True,
        "not_checkpoint_download": True,
        "not_inference": True,
        "not_success_claim": True,
        "required_before_mainstream_success_claim": [
            "prepared official environments under envs/",
            "official checkpoints or documented faithful checkpoint blockers",
            "stage-1 or full dataset materialization without held-out leakage",
            "official closed-loop Phase07 comparison runners with metrics and full videos",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
