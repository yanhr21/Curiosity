#!/usr/bin/env python3
"""Validate Newton visual preview artifacts.

This is a filesystem-level post-render checker. It does not render, simulate,
load Newton, or inspect semantics. It verifies that a preview directory contains
the expected browser, summary, contact sheet, and non-empty PNG frames with
valid PNG dimensions.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class VisualPreviewValidationError(ValueError):
    """Raised when a visual preview artifact is malformed or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualPreviewValidationError(message)


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        _require(signature == PNG_SIGNATURE, f"{path} is not a PNG")
        ihdr_len = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        _require(ihdr_len == 13 and chunk_type == b"IHDR", f"{path} missing IHDR")
        width, height = struct.unpack(">II", handle.read(8))
    _require(width > 0 and height > 0, f"{path} has invalid size {width}x{height}")
    return width, height


def validate(preview_dir: Path, *, min_frames: int) -> dict[str, Any]:
    _require(preview_dir.is_dir(), f"preview dir missing: {preview_dir}")

    browser = preview_dir / "frame_browser.html"
    summary = preview_dir / "summary.json"
    contact_sheet = preview_dir / "contact_sheet.png"
    _require(browser.is_file() and browser.stat().st_size > 0, "frame_browser.html missing or empty")
    _require(summary.is_file() and summary.stat().st_size > 0, "summary.json missing or empty")
    _require(contact_sheet.is_file() and contact_sheet.stat().st_size > 0, "contact_sheet.png missing or empty")

    frame_paths = sorted(preview_dir.glob("frame_*.png"))
    _require(len(frame_paths) >= min_frames, f"only {len(frame_paths)} frames, expected at least {min_frames}")
    _require(contact_sheet not in frame_paths, "contact_sheet unexpectedly matched frame glob")

    with summary.open("r", encoding="utf-8") as handle:
        summary_data = json.load(handle)

    sizes: dict[str, int] = {}
    min_file_size = None
    max_file_size = 0
    for path in frame_paths:
        file_size = path.stat().st_size
        _require(file_size > 0, f"{path} is empty")
        width, height = _png_size(path)
        key = f"{width}x{height}"
        sizes[key] = sizes.get(key, 0) + 1
        min_file_size = file_size if min_file_size is None else min(min_file_size, file_size)
        max_file_size = max(max_file_size, file_size)

    contact_width, contact_height = _png_size(contact_sheet)
    browser_text = browser.read_text(encoding="utf-8", errors="replace")
    _require("frame_" in browser_text, "frame_browser.html does not reference frames")

    return {
        "status": "pass",
        "preview_dir": str(preview_dir),
        "frame_count": len(frame_paths),
        "frame_dimensions": sizes,
        "min_frame_file_size": min_file_size,
        "max_frame_file_size": max_file_size,
        "contact_sheet_size": [contact_width, contact_height],
        "summary_keys": sorted(summary_data.keys()),
        "browser_size_bytes": browser.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview_dir", type=Path)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate(args.preview_dir, min_frames=args.min_frames)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
