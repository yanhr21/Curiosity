#!/usr/bin/env python3
"""Fail-closed admission audit for an official Zero-WAM release.

This inventory deliberately cannot pass from a project page, the public Wan
base alone, or a locally written replacement.  A passing result also requires a
separate strict-load result produced from the released Zero-WAM implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "robbyant-research/Zero-WAM"
PAGE_ONLY_BASELINE = "5a8a2da069392c1974ee98941ada13a5208b0ca5"
CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml"}
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}
ENTRYPOINT_RE = re.compile(r"(^|/)(train|infer|inference|eval|evaluate|demo|generate)[^/]*", re.I)
DATA_RE = re.compile(r"(^|/)(data|dataset|datasets|dataloader|manifest|schema)(/|[_\.])", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-json", type=Path, required=True)
    parser.add_argument("--tree-json", type=Path, required=True)
    parser.add_argument("--discovery-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--strict-load-result", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_files(root: Path | None) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def relative_paths(files: list[Path], root: Path | None) -> list[str]:
    if root is None:
        return []
    return [path.relative_to(root).as_posix() for path in files]


def code_semantic_hits(repo_dir: Path | None, code_paths: list[str]) -> dict[str, bool]:
    terms = {
        "world_or_video_model": re.compile(r"world.?model|video.?model|video.?transformer", re.I),
        "action_model": re.compile(r"action.?model|action.?transformer|action.?flow", re.I),
        "ifp": re.compile(r"in.?context future|\bIFP\b|future.?chunk", re.I),
        "mixture_of_transformers": re.compile(r"mixture.?of.?transformers|\bMoT\b", re.I),
    }
    hits = {name: False for name in terms}
    if repo_dir is None:
        return hits
    for relpath in code_paths:
        path = repo_dir / relpath
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in terms.items():
            hits[name] = hits[name] or bool(pattern.search(text))
    return hits


def weight_inventory(root: Path | None) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    for path in local_files(root):
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        size = path.stat().st_size
        total += size
        records.append({"path": path.relative_to(root).as_posix(), "bytes": size})
    return records, total


def main() -> None:
    args = parse_args()
    commit = read_json(args.commit_json)
    tree = read_json(args.tree_json)
    sha = str(commit.get("sha", ""))
    html_url = str(commit.get("html_url", ""))
    tree_paths = sorted(
        str(item.get("path"))
        for item in tree.get("tree", [])
        if isinstance(item, dict) and item.get("type") == "blob" and item.get("path")
    )

    repo_files = local_files(args.repo_dir)
    repo_paths = relative_paths(repo_files, args.repo_dir)
    inventory_paths = sorted(set(tree_paths) | set(repo_paths))
    code_paths = [path for path in inventory_paths if Path(path).suffix.lower() in CODE_SUFFIXES]
    python_paths = [path for path in code_paths if path.endswith(".py")]
    entrypoints = [path for path in code_paths if ENTRYPOINT_RE.search(path)]
    data_paths = [
        path
        for path in inventory_paths
        if DATA_RE.search(path)
        and Path(path).suffix.lower() in (CODE_SUFFIXES | {".json"})
        and not path.startswith("docs/assets/")
    ]
    semantic_hits = code_semantic_hits(args.repo_dir, code_paths)
    weights, weight_bytes = weight_inventory(args.checkpoint_dir)

    strict = read_json(args.strict_load_result) if args.strict_load_result else {}
    discovery = read_json(args.discovery_json) if args.discovery_json else {}
    discovery_checks = (
        discovery.get("checks", {})
        if isinstance(discovery.get("checks", {}), dict)
        else {}
    )
    strict_checks = strict.get("checks", {}) if isinstance(strict.get("checks", {}), dict) else {}
    strict_parameter_count = strict.get("parameter_count", 0)
    strict_hashes = strict.get("checkpoint_sha256", {})

    checks = {
        "canonical_official_repository": OFFICIAL_REPOSITORY.lower() in html_url.lower(),
        "official_main_tag_release_discovery_passed": (
            discovery.get("protocol") == "zero_wam_official_multiref_discovery_v1"
            and discovery.get("repository") == OFFICIAL_REPOSITORY
            and discovery.get("api_origin") == "https://api.github.com"
            and discovery.get("passed") is True
            and all(value is True for value in discovery_checks.values())
        ),
        "selected_ref_commit_matches_audited_commit": (
            discovery.get("selected_commit") == sha
            and isinstance(discovery.get("selected_source_kinds"), list)
            and bool(discovery.get("selected_source_kinds"))
            and set(discovery["selected_source_kinds"]).issubset(
                {"main", "tag", "release"}
            )
        ),
        "selected_ref_artifacts_match_discovery_hashes": (
            discovery.get("selected_commit_json_sha256")
            == file_sha256(args.commit_json)
            and discovery.get("selected_tree_json_sha256")
            == file_sha256(args.tree_json)
        ),
        "selected_ref_is_structural_official_release": (
            discovery.get("release_available") is True
            and discovery.get("selected_structural_release") is True
        ),
        "full_40_character_commit_recorded": bool(re.fullmatch(r"[0-9a-f]{40}", sha)),
        "newer_than_page_only_baseline": sha != PAGE_ONLY_BASELINE,
        "nontrivial_method_code_published": len(python_paths) >= 5,
        "training_or_inference_entrypoint_published": bool(entrypoints),
        "data_schema_or_loader_published": bool(data_paths),
        "released_code_contains_world_action_ifp_mot": all(semantic_hits.values()),
        "official_checkpoint_files_present": len(weights) >= 2 and weight_bytes >= 1_000_000_000,
        "strict_official_model_load_passed": bool(strict.get("passed"))
        and bool(strict_checks.get("strict_load")),
        "official_model_class_and_config_verified": bool(strict_checks.get("official_model_class"))
        and bool(strict_checks.get("official_config_identity")),
        "official_example_reproduced": bool(strict_checks.get("official_example")),
        "checkpoint_hashes_and_parameter_count_recorded": isinstance(strict_hashes, dict)
        and bool(strict_hashes)
        and isinstance(strict_parameter_count, int)
        and strict_parameter_count > 1_000_000_000,
        "no_local_substitute": strict.get("provenance") == "official_zero_wam_release",
    }
    release_available = all(
        checks[name]
        for name in (
            "canonical_official_repository",
            "official_main_tag_release_discovery_passed",
            "selected_ref_commit_matches_audited_commit",
            "selected_ref_artifacts_match_discovery_hashes",
            "selected_ref_is_structural_official_release",
            "full_40_character_commit_recorded",
            "newer_than_page_only_baseline",
            "nontrivial_method_code_published",
            "training_or_inference_entrypoint_published",
            "data_schema_or_loader_published",
            "released_code_contains_world_action_ifp_mot",
        )
    )
    passed = all(checks.values())
    result = {
        "protocol": "official_zero_wam_release_strict_admission_v1",
        "passed": passed,
        "release_available": release_available,
        "official_repository": OFFICIAL_REPOSITORY,
        "commit": sha,
        "commit_date": commit.get("commit", {}).get("committer", {}).get("date"),
        "commit_message": commit.get("commit", {}).get("message"),
        "release_source_kinds": discovery.get("selected_source_kinds", []),
        "release_source_refs": discovery.get("selected_source_refs", []),
        "release_assets": discovery.get("selected_release_assets", []),
        "release_asset_bytes": discovery.get("selected_release_asset_bytes", 0),
        "multiref_candidate_count": discovery.get("candidate_count", 0),
        "discovery_sha256": (
            file_sha256(args.discovery_json) if args.discovery_json else None
        ),
        "page_only_baseline": PAGE_ONLY_BASELINE,
        "tree_file_count": len(tree_paths),
        "python_file_count": len(python_paths),
        "entrypoints": entrypoints,
        "data_paths": data_paths,
        "semantic_code_hits": semantic_hits,
        "checkpoint_file_count": len(weights),
        "checkpoint_bytes": weight_bytes,
        "checkpoint_files": weights,
        "checks": checks,
        "automatic_next_branch": (
            "run_frozen_sugar_prompt_gate" if passed else
            "strict_load_official_release" if release_available else
            discovery.get(
                "automatic_next_branch", "recheck_official_main_tags_and_releases"
            )
        ),
        "claim_boundary": (
            "Passing proves only official release identity, strict loading, and one official example. "
            "It does not prove SUGAR prompt dependence, adaptation, or closed-loop demo following."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "OFFICIAL_RELEASE_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
