"""Probe official checkpoint access paths for Phase07 mainstream methods.

This audit does not download checkpoint files. It checks whether the official
remote entry points and local tools are reachable, then records the result for
later checkpoint acquisition or blocker documentation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CHECKPOINT_SOURCES = {
    "openpi_pi0": {
        "kind": "gcs_prefix",
        "tool": "gsutil",
        "targets": [
            "gs://openpi-assets/checkpoints/pi0_base",
            "gs://openpi-assets/checkpoints/pi05_base",
            "gs://openpi-assets/checkpoints/pi05_droid",
        ],
    },
    "gr00t": {
        "kind": "http_collection",
        "tool": "http",
        "targets": [
            "https://huggingface.co/collections/nvidia/gr00t-n17",
        ],
    },
    "diffusion_policy": {
        "kind": "http_directory",
        "tool": "http",
        "targets": [
            "https://diffusion-policy.cs.columbia.edu/data/experiments/",
        ],
    },
    "rtx": {
        "kind": "gcs_prefix",
        "tool": "gsutil",
        "targets": [
            "gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax",
        ],
    },
}


def _run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "stdout_preview": proc.stdout.strip()[:1000],
            "stderr_preview": proc.stderr.strip()[:1000],
            "status": "pass" if proc.returncode == 0 else "fail",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "exit_code": None,
            "stdout_preview": (exc.stdout or "")[:1000] if isinstance(exc.stdout, str) else "",
            "stderr_preview": "timeout",
            "status": "timeout",
        }


def _http_probe(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Phase07CheckpointProbe/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "url": url,
                "status": "pass",
                "http_status": int(response.status),
                "content_type": response.headers.get("content-type"),
                "note": "HEAD request only; no checkpoint download.",
            }
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            get_request = urllib.request.Request(url, method="GET", headers={"User-Agent": "Phase07CheckpointProbe/1.0"})
            try:
                with urllib.request.urlopen(get_request, timeout=timeout) as response:
                    response.read(2048)
                    return {
                        "url": url,
                        "status": "pass",
                        "http_status": int(response.status),
                        "content_type": response.headers.get("content-type"),
                        "note": "Small GET probe only; no checkpoint download.",
                    }
            except Exception as get_exc:  # noqa: BLE001
                return {"url": url, "status": "fail", "error": repr(get_exc)}
        return {"url": url, "status": "fail", "http_status": exc.code, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "status": "fail", "error": repr(exc)}


def _probe_method(name: str, spec: dict[str, Any], timeout: int) -> dict[str, Any]:
    tool = spec["tool"]
    tool_path = shutil.which(tool) if tool != "http" else "urllib"
    results: list[dict[str, Any]] = []
    if tool != "http" and not tool_path:
        return {
            "method": name,
            "status": "fail",
            "tool": tool,
            "tool_path": None,
            "targets": spec["targets"],
            "results": [],
            "failures": [f"missing_tool={tool}"],
        }
    for target in spec["targets"]:
        if spec["kind"] == "gcs_prefix":
            results.append(_run([tool_path or tool, "ls", target], timeout=timeout))
        else:
            results.append(_http_probe(target, timeout=timeout))
    failures = [item for item in results if item.get("status") != "pass"]
    return {
        "method": name,
        "status": "pass" if not failures else "fail",
        "tool": tool,
        "tool_path": tool_path,
        "kind": spec["kind"],
        "targets": spec["targets"],
        "results": results,
        "failures": failures,
        "not_checkpoint_download": True,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase07 Official Checkpoint Access Probe V1",
        "",
        "Date: 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This probe checks remote entry-point access only. It does not download checkpoints and does not satisfy the official-method readiness gate.",
        "",
        "## Methods",
        "",
    ]
    for name, item in payload["methods"].items():
        lines.append(f"- `{name}`: `{item['status']}` via `{item['tool']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_official_checkpoint_access_v1_20260627.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/reports/2026-06-27_phase07_official_checkpoint_access_v1.md"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    methods = {name: _probe_method(name, spec, args.timeout) for name, spec in CHECKPOINT_SOURCES.items()}
    all_pass = all(item["status"] == "pass" for item in methods.values())
    payload = {
        "classification": "phase07_official_checkpoint_access_v1",
        "status": "pass" if all_pass else "partial_or_failed",
        "methods": methods,
        "not_checkpoint_download": True,
        "not_checkpoint": True,
        "not_training": True,
        "not_inference": True,
        "not_success_claim": True,
        "readiness_impact": "Remote access probe only; official checkpoint files or filled blockers are still required.",
    }
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
