#!/usr/bin/env python3
"""Check recorded/replayed G1 showcase artifacts.

This checker is deliberately conservative.  A replay render can be presentable
visual evidence only if the source non-rendered rollout passed, replay CSV was
written, frames exist, and the render summary marks itself as visualization
only rather than new control evidence.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check G1 replay showcase artifacts.")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--min-replay-rows", type=int, default=20)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--min-frame-bytes", type=int, default=4096)
    parser.add_argument("--expected-width", type=int, default=None)
    parser.add_argument("--expected-height", type=int, default=None)
    parser.add_argument("--require-mp4", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    record_summary_path = args.record_dir / "core_world_g1_box_scene_summary.json"
    replay_csv_path = args.record_dir / "core_world_g1_box_scene_replay.csv"
    render_summary_path = args.render_dir / "g1_replay_render_summary.json"
    frame_dir = args.render_dir / "rgb_frames"
    mp4_path = args.render_dir / "g1_replay_showcase.mp4"
    annotated_mp4_path = args.render_dir / "g1_replay_showcase_annotated.mp4"

    record_summary: dict = {}
    render_summary: dict = {}

    if not record_summary_path.is_file():
        failures.append(f"missing record summary: {record_summary_path}")
    else:
        record_summary = _load_json(record_summary_path)
        if record_summary.get("status") != "pass":
            failures.append(f"record status is not pass: {record_summary.get('status')}")
        if not bool(record_summary.get("record_replay_csv")):
            failures.append("record summary does not mark record_replay_csv=true")
        if int(record_summary.get("fall_events", 999999)) != 0:
            failures.append(f"record fall_events != 0: {record_summary.get('fall_events')}")
        if int(record_summary.get("box_drop_events", 999999)) != 0:
            failures.append(f"record box_drop_events != 0: {record_summary.get('box_drop_events')}")

    if not replay_csv_path.is_file():
        failures.append(f"missing replay csv: {replay_csv_path}")
        replay_rows = 0
    else:
        with replay_csv_path.open("r", encoding="utf-8") as f:
            replay_rows = max(0, sum(1 for _ in f) - 1)
        if replay_rows < int(args.min_replay_rows):
            failures.append(f"replay rows {replay_rows} < {args.min_replay_rows}")

    if not render_summary_path.is_file():
        failures.append(f"missing render summary: {render_summary_path}")
    else:
        render_summary = _load_json(render_summary_path)
        if render_summary.get("status") != "pass":
            failures.append(f"render status is not pass: {render_summary.get('status')}")
        if render_summary.get("success_claim") != "visual_replay_only_not_new_control_evidence":
            failures.append("render success_claim is not visualization-only")

    frames = sorted(frame_dir.glob("*.png")) if frame_dir.is_dir() else []
    if len(frames) < int(args.min_frames):
        failures.append(f"frame count {len(frames)} < {args.min_frames}")
    checked_frame_count = 0
    bad_frames: list[str] = []
    sample_frames = frames[:2] + frames[max(0, len(frames) // 2) : max(0, len(frames) // 2) + 1] + frames[-2:]
    seen_samples: set[Path] = set()
    for frame_path in sample_frames:
        if frame_path in seen_samples:
            continue
        seen_samples.add(frame_path)
        checked_frame_count += 1
        frame_size = frame_path.stat().st_size
        if frame_size < int(args.min_frame_bytes):
            bad_frames.append(f"{frame_path.name}: {frame_size} bytes < {args.min_frame_bytes}")
            continue
        dims = _png_dimensions(frame_path)
        if dims is None:
            bad_frames.append(f"{frame_path.name}: invalid PNG header")
            continue
        if args.expected_width is not None and dims[0] != int(args.expected_width):
            bad_frames.append(f"{frame_path.name}: width {dims[0]} != {args.expected_width}")
        if args.expected_height is not None and dims[1] != int(args.expected_height):
            bad_frames.append(f"{frame_path.name}: height {dims[1]} != {args.expected_height}")
    if bad_frames:
        failures.extend(f"bad frame sample: {bad_frame}" for bad_frame in bad_frames)
    if bool(args.require_mp4) and not (mp4_path.is_file() or annotated_mp4_path.is_file()):
        failures.append(f"missing mp4: {mp4_path} or {annotated_mp4_path}")

    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "record_dir": str(args.record_dir),
        "render_dir": str(args.render_dir),
        "record_status": record_summary.get("status"),
        "record_fall_events": record_summary.get("fall_events"),
        "record_box_drop_events": record_summary.get("box_drop_events"),
        "record_replay_csv": record_summary.get("record_replay_csv"),
        "replay_rows": replay_rows,
        "render_status": render_summary.get("status"),
        "render_success_claim": render_summary.get("success_claim"),
        "frame_count": len(frames),
        "checked_frame_count": checked_frame_count,
        "bad_frame_samples": bad_frames,
        "min_frame_bytes": int(args.min_frame_bytes),
        "expected_width": args.expected_width,
        "expected_height": args.expected_height,
        "mp4_exists": mp4_path.is_file(),
        "annotated_mp4_exists": annotated_mp4_path.is_file(),
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
