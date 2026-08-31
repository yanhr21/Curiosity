#!/usr/bin/env python3
"""Select a structurally real official Zero-WAM release across GitHub refs.

This is a provenance/discovery auditor, not a model implementation.  It accepts
only commit and recursive-tree responses fetched from the canonical official
GitHub repository, inventories main/tags/releases, and emits one selected
commit/tree pair for the stricter source, checkpoint and strict-load audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "zero_wam_official_multiref_discovery_v1"
INDEX_PROTOCOL = "zero_wam_official_multiref_candidate_index_v1"
OFFICIAL_REPOSITORY = "robbyant-research/Zero-WAM"
OFFICIAL_API_ORIGIN = "https://api.github.com"
PAGE_ONLY_BASELINE = "5a8a2da069392c1974ee98941ada13a5208b0ca5"
CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml"}
ENTRYPOINT_RE = re.compile(
    r"(^|/)(train|infer|inference|eval|evaluate|demo|generate)[^/]*", re.I
)
DATA_RE = re.compile(
    r"(^|/)(data|dataset|datasets|dataloader|manifest|schema)(/|[_\.])", re.I
)
SOURCE_PRIORITIES = {"main": 1, "tag": 2, "release": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_index_artifact(index_path: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    root = index_path.parent.resolve()
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def tree_inventory(tree: dict[str, Any]) -> dict[str, Any]:
    paths = sorted(
        str(row.get("path"))
        for row in tree.get("tree", [])
        if isinstance(row, dict)
        and row.get("type") == "blob"
        and isinstance(row.get("path"), str)
    )
    code_paths = [path for path in paths if Path(path).suffix.lower() in CODE_SUFFIXES]
    python_paths = [path for path in code_paths if path.endswith(".py")]
    entrypoints = [path for path in code_paths if ENTRYPOINT_RE.search(path)]
    data_paths = [
        path
        for path in paths
        if DATA_RE.search(path)
        and Path(path).suffix.lower() in (CODE_SUFFIXES | {".json"})
        and not path.startswith("docs/assets/")
    ]
    return {
        "tree_file_count": len(paths),
        "python_file_count": len(python_paths),
        "entrypoints": entrypoints,
        "data_paths": data_paths,
        "nontrivial_method_code_published": len(python_paths) >= 5,
        "training_or_inference_entrypoint_published": bool(entrypoints),
        "data_schema_or_loader_published": bool(data_paths),
    }


def valid_release_assets(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and bool(row["name"])
        and isinstance(row.get("bytes"), int)
        and not isinstance(row.get("bytes"), bool)
        and row["bytes"] >= 0
        and isinstance(row.get("browser_download_url"), str)
        and row["browser_download_url"].startswith(
            "https://github.com/robbyant-research/Zero-WAM/releases/download/"
        )
        for row in value
    )


def evaluate(index_path: Path) -> tuple[dict[str, Any], Path, Path]:
    index = read_json(index_path)
    candidates_raw = index.get("candidates", []) if isinstance(index, dict) else []
    index_identity_ok = (
        isinstance(index, dict)
        and index.get("protocol") == INDEX_PROTOCOL
        and index.get("repository") == OFFICIAL_REPOSITORY
        and index.get("api_origin") == OFFICIAL_API_ORIGIN
        and isinstance(index.get("discovered_source_ref_count"), int)
        and not isinstance(index.get("discovered_source_ref_count"), bool)
        and index["discovered_source_ref_count"] >= 1
        and index.get("resolution_error_count") == 0
        and isinstance(candidates_raw, list)
        and bool(candidates_raw)
    )

    candidates: list[dict[str, Any]] = []
    source_refs: set[tuple[str, str]] = set()
    all_candidate_artifacts_valid = bool(candidates_raw)
    for raw in candidates_raw if isinstance(candidates_raw, list) else []:
        if not isinstance(raw, dict):
            all_candidate_artifacts_valid = False
            continue
        commit_path = resolve_index_artifact(index_path, raw.get("commit_json"))
        tree_path = resolve_index_artifact(index_path, raw.get("tree_json"))
        source_kinds = raw.get("source_kinds")
        refs = raw.get("source_refs")
        sources_valid = (
            isinstance(source_kinds, list)
            and bool(source_kinds)
            and set(source_kinds).issubset(SOURCE_PRIORITIES)
            and isinstance(refs, list)
            and len(refs) == len(source_kinds)
            and all(isinstance(value, str) and bool(value) for value in refs)
        )
        assets = raw.get("release_assets", [])
        row_valid = (
            commit_path is not None
            and tree_path is not None
            and sources_valid
            and valid_release_assets(assets)
        )
        all_candidate_artifacts_valid = all_candidate_artifacts_valid and row_valid
        if not row_valid:
            continue
        commit = read_json(commit_path)
        tree = read_json(tree_path)
        sha = str(commit.get("sha", "")) if isinstance(commit, dict) else ""
        html_url = str(commit.get("html_url", "")) if isinstance(commit, dict) else ""
        canonical_commit = (
            OFFICIAL_REPOSITORY.lower() in html_url.lower()
            and bool(re.fullmatch(r"[0-9a-f]{40}", sha))
        )
        inventory = tree_inventory(tree if isinstance(tree, dict) else {})
        structural_release = (
            index_identity_ok
            and canonical_commit
            and sha != PAGE_ONLY_BASELINE
            and inventory["nontrivial_method_code_published"]
            and inventory["training_or_inference_entrypoint_published"]
            and inventory["data_schema_or_loader_published"]
        )
        for kind, ref in zip(source_kinds, refs, strict=True):
            source_refs.add((kind, ref))
        commit_date = ""
        if isinstance(commit, dict):
            commit_date = str(
                commit.get("commit", {}).get("committer", {}).get("date", "")
            )
        candidates.append(
            {
                "sha": sha,
                "commit_date": commit_date,
                "source_kinds": source_kinds,
                "source_refs": refs,
                "source_priority": max(SOURCE_PRIORITIES[value] for value in source_kinds),
                "release_assets": assets,
                "release_asset_bytes": sum(row["bytes"] for row in assets),
                "canonical_official_commit": canonical_commit,
                "structural_release": structural_release,
                "inventory": inventory,
                "commit_path": commit_path,
                "tree_path": tree_path,
            }
        )

    structurally_released = [row for row in candidates if row["structural_release"]]
    selection_pool = structurally_released or [
        row for row in candidates if "main" in row["source_kinds"]
    ] or candidates
    if not selection_pool:
        raise ValueError("candidate index contains no valid official commit/tree pair")
    selected = max(
        selection_pool,
        key=lambda row: (row["commit_date"], row["source_priority"], row["sha"]),
    )
    candidate_summaries = [
        {key: value for key, value in row.items() if key not in {"commit_path", "tree_path"}}
        for row in candidates
    ]
    checks = {
        "canonical_official_candidate_index": index_identity_ok,
        "all_candidate_artifacts_valid_and_inside_index_root": (
            all_candidate_artifacts_valid
        ),
        "main_ref_inventoried": ("main", "main") in source_refs,
        "every_discovered_source_ref_resolved": (
            isinstance(index, dict)
            and len(source_refs) == index.get("discovered_source_ref_count")
            and index.get("resolution_error_count") == 0
        ),
        "all_source_kinds_are_main_tag_or_release": all(
            kind in SOURCE_PRIORITIES for kind, _ in source_refs
        ),
        "selected_commit_is_canonical_official": selected["canonical_official_commit"],
        "selected_commit_matches_structural_release_decision": (
            selected["structural_release"] == bool(structurally_released)
        ),
    }
    release_available = bool(structurally_released) and all(checks.values())
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "release_available": release_available,
        "repository": OFFICIAL_REPOSITORY,
        "api_origin": OFFICIAL_API_ORIGIN,
        "page_only_baseline": PAGE_ONLY_BASELINE,
        "candidate_count": len(candidates),
        "unique_source_ref_count": len(source_refs),
        "structural_release_candidate_count": len(structurally_released),
        "selected_commit": selected["sha"],
        "candidate_index_sha256": file_sha256(index_path),
        "selected_commit_json_sha256": file_sha256(selected["commit_path"]),
        "selected_tree_json_sha256": file_sha256(selected["tree_path"]),
        "selected_source_kinds": selected["source_kinds"],
        "selected_source_refs": selected["source_refs"],
        "selected_release_assets": selected["release_assets"],
        "selected_release_asset_bytes": selected["release_asset_bytes"],
        "selected_structural_release": selected["structural_release"],
        "candidates": candidate_summaries,
        "checks": checks,
        "automatic_next_branch": (
            "download_and_strict_audit_selected_official_release"
            if release_available
            else "recheck_official_main_tags_and_releases"
        ),
        "claim_boundary": (
            "Passing discovery proves canonical multi-ref provenance and structural source presence "
            "only; it does not prove semantic code identity, checkpoint availability, strict load, "
            "SUGAR adaptation or demo following."
        ),
    }
    return result, selected["commit_path"], selected["tree_path"]


def fixture_commit(sha: str, date: str) -> dict[str, Any]:
    return {
        "sha": sha,
        "html_url": f"https://github.com/{OFFICIAL_REPOSITORY}/commit/{sha}",
        "commit": {"committer": {"date": date}},
    }


def fixture_tree(real: bool) -> dict[str, Any]:
    paths = ["README.md", "LICENSE"]
    if real:
        paths += [
            "zero_wam/video_transformer.py",
            "zero_wam/action_transformer.py",
            "zero_wam/ifp.py",
            "zero_wam/mot.py",
            "zero_wam/model.py",
            "train.py",
            "datasets/schema.json",
        ]
    return {"tree": [{"path": path, "type": "blob"} for path in paths]}


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="zero_wam_release_discovery_") as temp:
        root = Path(temp)
        page_sha = PAGE_ONLY_BASELINE
        tag_sha = "1" * 40
        rows = []
        for index, (sha, real, kinds, refs, date) in enumerate(
            (
                (page_sha, False, ["main"], ["main"], "2026-08-27T00:00:00Z"),
                (tag_sha, True, ["tag", "release"], ["v1.0", "v1.0"], "2026-09-01T00:00:00Z"),
            )
        ):
            candidate = root / f"candidate_{index:03d}"
            candidate.mkdir()
            write_json(candidate / "COMMIT.json", fixture_commit(sha, date))
            write_json(candidate / "TREE.json", fixture_tree(real))
            rows.append(
                {
                    "source_kinds": kinds,
                    "source_refs": refs,
                    "commit_json": f"candidate_{index:03d}/COMMIT.json",
                    "tree_json": f"candidate_{index:03d}/TREE.json",
                    "release_assets": [],
                }
            )
        index_path = root / "CANDIDATES.json"
        write_json(
            index_path,
            {
                "protocol": INDEX_PROTOCOL,
                "repository": OFFICIAL_REPOSITORY,
                "api_origin": OFFICIAL_API_ORIGIN,
                "discovered_source_ref_count": 3,
                "resolution_error_count": 0,
                "candidates": rows,
            },
        )
        positive, _, _ = evaluate(index_path)
        assert positive["passed"] is True, positive
        assert positive["release_available"] is True, positive
        assert positive["selected_commit"] == tag_sha, positive
        assert positive["selected_source_kinds"] == ["tag", "release"], positive

        external = read_json(index_path)
        external["repository"] = "untrusted/Zero-WAM-copy"
        write_json(index_path, external)
        external_result, _, _ = evaluate(index_path)
        assert external_result["release_available"] is False
        assert (
            external_result["checks"]["canonical_official_candidate_index"] is False
        )

        untrusted_asset = {
            "protocol": INDEX_PROTOCOL,
            "repository": OFFICIAL_REPOSITORY,
            "api_origin": OFFICIAL_API_ORIGIN,
            "discovered_source_ref_count": 3,
            "resolution_error_count": 0,
            "candidates": copy.deepcopy(rows),
        }
        untrusted_asset["candidates"][1]["release_assets"] = [
            {
                "name": "zero-wam.safetensors",
                "bytes": 2_000_000_000,
                "browser_download_url": "https://example.invalid/fake.safetensors",
            }
        ]
        write_json(index_path, untrusted_asset)
        untrusted_asset_result, _, _ = evaluate(index_path)
        assert untrusted_asset_result["release_available"] is False
        assert (
            untrusted_asset_result["checks"][
                "all_candidate_artifacts_valid_and_inside_index_root"
            ]
            is False
        )

        unresolved = read_json(index_path)
        unresolved["candidates"] = copy.deepcopy(rows)
        unresolved["resolution_error_count"] = 1
        write_json(index_path, unresolved)
        unresolved_result, _, _ = evaluate(index_path)
        assert unresolved_result["release_available"] is False
        assert (
            unresolved_result["checks"]["every_discovered_source_ref_resolved"]
            is False
        )

        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "selected_tagged_release_while_main_page_only": True,
                    "rejected": [
                        "external_repository_candidate_index",
                        "untrusted_release_asset_url",
                        "unresolved_official_ref",
                    ],
                    "fixture_claim_boundary": "synthetic discovery contract only",
                },
                sort_keys=True,
            )
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.candidate_index is None or args.output_dir is None:
        raise SystemExit("--candidate-index and --output-dir are required")
    result, commit_path, tree_path = evaluate(args.candidate_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "OFFICIAL_RELEASE_DISCOVERY.json", result)
    shutil.copyfile(commit_path, args.output_dir / "SELECTED_COMMIT.json")
    shutil.copyfile(tree_path, args.output_dir / "SELECTED_TREE.json")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
