#!/usr/bin/env python3
"""Write a compact manifest for the current G1 showcase visualization.

This script is intentionally read-only with respect to simulation.  It only
inspects already-produced JSON/check/video/frame files and writes a Markdown
and optional JSON manifest for presentation triage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RECORD_DIR = Path(
    "experiments/outputs/core_world_g1_agile_policy_low_cradle/"
    "20260706_g1_lowcarry_168398_replay_record_retry2/"
    "agile_low_cradle_freebox_walk"
)
DEFAULT_RENDER_DIRS = [
    Path("experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3"),
    Path("experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_960x540"),
    Path("experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_abs_960x540"),
    Path("experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_direct_960x540"),
    Path("experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540"),
]
DEFAULT_PRESENTATION_FALLBACK_DIR = Path(
    "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif"
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def _render_entry(render_dir: Path) -> dict:
    frame_dir = render_dir / "rgb_frames"
    frames = sorted(frame_dir.glob("*.png")) if frame_dir.is_dir() else []
    check = _load_json(render_dir / "g1_replay_showcase_check.json")
    summary = _load_json(render_dir / "g1_replay_render_summary.json")
    return {
        "render_dir": str(render_dir),
        "check_status": check.get("status", "missing"),
        "check_failures": check.get("failures", []),
        "summary_status": summary.get("status", "missing"),
        "summary_success_claim": summary.get("success_claim"),
        "frame_count": len(frames),
        "first_frame": str(frames[0]) if frames else "",
        "last_frame": str(frames[-1]) if frames else "",
        "mp4": _file_info(render_dir / "g1_replay_showcase.mp4"),
        "annotated_mp4": _file_info(render_dir / "g1_replay_showcase_annotated.mp4"),
    }


def _presentation_fallback_entry(path: Path) -> dict:
    summary = _load_json(path / "g1_replay_presentation_fallback_summary.json")
    return {
        "dir": str(path),
        "status": summary.get("status", "missing"),
        "success_claim": summary.get("success_claim"),
        "frame_count": summary.get("frame_count"),
        "gif": _file_info(path / "g1_lowcarry_replay_fallback.gif"),
        "poster": _file_info(path / "g1_lowcarry_replay_fallback_poster.png"),
    }


def _choose_best(entries: list[dict]) -> dict | None:
    passing = [entry for entry in entries if entry["check_status"] == "pass" and entry["frame_count"] > 0]
    if passing:
        return sorted(
            passing,
            key=lambda entry: (
                bool(entry["annotated_mp4"]["exists"]),
                bool(entry["mp4"]["exists"]),
                int(entry["frame_count"]),
            ),
            reverse=True,
        )[0]
    with_frames = [entry for entry in entries if entry["frame_count"] > 0]
    if with_frames:
        return sorted(with_frames, key=lambda entry: int(entry["frame_count"]), reverse=True)[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write G1 showcase visualization manifest.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--render-dir", type=Path, action="append", default=None)
    parser.add_argument("--presentation-fallback-dir", type=Path, default=DEFAULT_PRESENTATION_FALLBACK_DIR)
    parser.add_argument("--output-md", type=Path, default=Path("experiments/reports/2026-07-07_g1_showcase_visual_manifest.md"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/reports/2026-07-07_g1_showcase_visual_manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_dirs = args.render_dir if args.render_dir else DEFAULT_RENDER_DIRS

    record_summary = _load_json(args.record_dir / "core_world_g1_box_scene_summary.json")
    replay_csv = args.record_dir / "core_world_g1_box_scene_replay.csv"
    entries = [_render_entry(render_dir) for render_dir in render_dirs]
    best = _choose_best(entries)
    presentation_fallback = _presentation_fallback_entry(args.presentation_fallback_dir)

    manifest = {
        "status": "ready" if best and best["check_status"] == "pass" else "pending_or_failed",
        "record_dir": str(args.record_dir),
        "record_status": record_summary.get("status", "missing"),
        "record_fall_events": record_summary.get("fall_events"),
        "record_box_drop_events": record_summary.get("box_drop_events"),
        "record_replay_csv": record_summary.get("record_replay_csv"),
        "replay_csv": _file_info(replay_csv),
        "best_render_dir": best["render_dir"] if best else "",
        "best_annotated_mp4": best["annotated_mp4"]["path"] if best and best["annotated_mp4"]["exists"] else "",
        "best_mp4": best["mp4"]["path"] if best and best["mp4"]["exists"] else "",
        "best_first_frame": best["first_frame"] if best else "",
        "presentation_fallback": presentation_fallback,
        "render_entries": entries,
        "claim_limit": (
            "This is a visual replay of a prior recorded low-carry pass. It is not new control "
            "evidence, not proof of arbitrary posture robustness, and not final task completion."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# G1 Showcase Visual Manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Source rollout status: `{manifest['record_status']}`",
        f"- Source fall/drop events: `{manifest['record_fall_events']}` / `{manifest['record_box_drop_events']}`",
        f"- Replay CSV: `{manifest['replay_csv']['path']}`",
        f"- Best render directory: `{manifest['best_render_dir'] or 'missing'}`",
        f"- Best annotated MP4: `{manifest['best_annotated_mp4'] or 'missing'}`",
        f"- Best raw MP4: `{manifest['best_mp4'] or 'missing'}`",
        f"- Best first PNG frame: `{manifest['best_first_frame'] or 'missing'}`",
        f"- Presentation fallback GIF: `{presentation_fallback['gif']['path'] if presentation_fallback['gif']['exists'] else 'missing'}`",
        f"- Presentation fallback poster: `{presentation_fallback['poster']['path'] if presentation_fallback['poster']['exists'] else 'missing'}`",
        "",
        "## Render Candidates",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"### `{entry['render_dir']}`",
                "",
                f"- Check status: `{entry['check_status']}`",
                f"- Summary status: `{entry['summary_status']}`",
                f"- Frames: `{entry['frame_count']}`",
                f"- Annotated MP4 exists: `{entry['annotated_mp4']['exists']}`",
                f"- Raw MP4 exists: `{entry['mp4']['exists']}`",
                f"- First frame: `{entry['first_frame'] or 'missing'}`",
            ]
        )
        if entry["check_failures"]:
            lines.append(f"- Check failures: `{entry['check_failures']}`")
        lines.append("")
    lines.extend(
        [
            "## Claim Limit",
            "",
            manifest["claim_limit"],
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
