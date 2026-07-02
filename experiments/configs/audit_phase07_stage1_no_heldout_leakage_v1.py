"""Audit Phase07 mainstream stage-1 indices for held-out leakage.

This is a lightweight metadata audit. It reads JSON/JSONL indices created by
the stage-1 mainstream dataset-index builder and verifies that held-out
episodes are marked eval-only everywhere. It does not preprocess data,
materialize datasets, train, run inference, or claim success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HELD_OUT_CELLS = {
    "empty_high_misleading",
    "full_low_hidden",
    "three_quarter_low_misleading",
}

METHOD_EPISODES = {
    "openpi_pi0": "openpi_lerobot_stage1/episodes.jsonl",
    "gr00t": "gr00t_lerobot_v2_stage1/episodes.jsonl",
    "diffusion_policy": "diffusion_policy_stage1/episodes.jsonl",
    "rtx": "rtx_stage1/episodes.jsonl",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{lineno} must contain a JSON object")
            rows.append(payload)
    return rows


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _audit_rows(rows: list[dict[str, Any]], source: str, require_all_splits: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    split_counts: dict[str, int] = {}
    held_out_indices: list[int] = []
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for row in rows:
        split = str(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
        episode_index = int(row.get("episode_index", -1))
        cell = str(row.get("cell", ""))
        held_out_flag = bool(row.get("must_not_use_for_training", row.get("held_out_training_forbidden", False)))
        if split == "held_out_eval_only" or cell in HELD_OUT_CELLS:
            held_out_indices.append(episode_index)
            if split != "held_out_eval_only":
                failures.append(f"held_out_cell_wrong_split:episode={episode_index}:cell={cell}:split={split}")
            if not held_out_flag:
                failures.append(f"held_out_episode_not_marked_forbidden:episode={episode_index}:cell={cell}")
        if split == "train":
            train_indices.append(episode_index)
            if held_out_flag:
                failures.append(f"train_episode_marked_forbidden:episode={episode_index}")
            if cell in HELD_OUT_CELLS:
                failures.append(f"held_out_cell_in_train:episode={episode_index}:cell={cell}")
        if split == "validation":
            validation_indices.append(episode_index)
            if cell in HELD_OUT_CELLS:
                failures.append(f"held_out_cell_in_validation:episode={episode_index}:cell={cell}")
    if require_all_splits and not train_indices:
        failures.append("missing_train_episodes")
    if require_all_splits and not validation_indices:
        failures.append("missing_validation_episodes")
    if require_all_splits and not held_out_indices:
        failures.append("missing_held_out_eval_only_episodes")
    return {
        "source": source,
        "status": "pass" if not failures else "fail",
        "episode_count": len(rows),
        "split_counts": split_counts,
        "held_out_episode_indices": sorted(held_out_indices),
        "train_episode_indices": sorted(train_indices),
        "validation_episode_indices": sorted(validation_indices),
        "failures": failures,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase07 Stage-1 No Held-Out Leakage Audit V1",
        "",
        "Date: 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This audit reads stage-1 JSON/JSONL metadata only. It does not preprocess datasets, train, infer, render, or claim success.",
        "",
        "## Sources",
        "",
    ]
    for name, item in payload["sources"].items():
        lines.append(f"- `{name}`: `{item['status']}`; split_counts={item.get('split_counts')}; failures={item.get('failures')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_stage1_no_heldout_leakage_v1_20260627.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/reports/2026-06-27_phase07_stage1_no_heldout_leakage_v1.md"),
    )
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    stage1_dir = args.stage1_dir if args.stage1_dir.is_absolute() else root / args.stage1_dir
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    manifest = _read_json(stage1_dir / "manifest.json")
    sources: dict[str, Any] = {}
    failures: list[str] = []
    if manifest is None:
        failures.append("missing_stage1_manifest")
    else:
        main_rows = _read_jsonl(stage1_dir / "episodes.jsonl")
        sources["main"] = _audit_rows(main_rows, "episodes.jsonl")
        for split in ["train", "validation", "held_out_eval_only"]:
            split_path = stage1_dir / "splits" / f"{split}.jsonl"
            if not split_path.is_file():
                sources[f"split_{split}"] = {
                    "source": str(split_path.relative_to(stage1_dir)),
                    "status": "fail",
                    "failures": ["missing_split_index"],
                }
                continue
            split_rows = _read_jsonl(split_path)
            split_audit = _audit_rows(split_rows, str(split_path.relative_to(stage1_dir)), require_all_splits=False)
            wrong_split = [row.get("episode_index") for row in split_rows if row.get("split") != split]
            if wrong_split:
                split_audit["status"] = "fail"
                split_audit["failures"].append(f"wrong_split_entries={wrong_split}")
            sources[f"split_{split}"] = split_audit
        for method, rel_path in METHOD_EPISODES.items():
            path = stage1_dir / rel_path
            if not path.is_file():
                sources[method] = {
                    "source": rel_path,
                    "status": "fail",
                    "failures": ["missing_method_episodes_jsonl"],
                }
                continue
            rows = _read_jsonl(path)
            sources[method] = _audit_rows(rows, rel_path)
        for name, item in sources.items():
            if item.get("status") != "pass":
                failures.append(f"{name}:{item.get('failures')}")
    payload = {
        "classification": "phase07_stage1_no_heldout_leakage_v1",
        "status": "pass" if not failures else ("open_missing_stage1" if "missing_stage1_manifest" in failures else "fail"),
        "stage1_dir": _relative_or_absolute(root, stage1_dir),
        "sources": sources,
        "failures": failures,
        "no_held_out_leakage_proven": not failures,
        "not_training": True,
        "not_data_preprocessing": True,
        "not_inference": True,
        "not_success_claim": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pass and payload["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
