#!/usr/bin/env python3
"""Channel-level semantic audit for the Phase 00 reference tactile gate.

This audit is intentionally conservative. It checks whether the current
candidate video exposes the same classes of channels as the reference video
at a visual/layout level. It does not validate photometric marker semantics,
real contact-area semantics, or hardware/Taccel equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from newton_tactile_curiosity.phase00_reference_video_compare import (
    colorfulness,
    edge_density,
    read_video_best,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(round(box[0] * w))))
    y0 = max(0, min(h - 1, int(round(box[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(box[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(box[3] * h))))
    return frame[y0:y1, x0:x1]


def blue_score(img: np.ndarray) -> float:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    blue = arr[..., 2]
    red = arr[..., 0]
    green = arr[..., 1]
    mask = (blue > red + 20.0) & (blue > green + 10.0) & (blue > 70.0)
    return float(mask.mean())


def warm_score(img: np.ndarray) -> float:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    mask = (red > 130.0) & (green > 70.0) & (blue < 120.0)
    return float(mask.mean())


def marker_dot_score(img: np.ndarray) -> float:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    brightness = arr.mean(axis=2)
    local_contrast = float(arr.std())
    bright_on_blue = (brightness > 120.0) & (arr[..., 2] > arr[..., 0] + 10.0)
    return float(bright_on_blue.mean() * min(local_contrast / 30.0, 2.0))


def region_metrics(frames: list[tuple[int, np.ndarray]], box: tuple[float, float, float, float]) -> dict[str, Any]:
    if not frames:
        return {
            "pixel_std_mean": 0.0,
            "edge_density_mean": 0.0,
            "colorfulness_mean": 0.0,
            "blue_score_mean": 0.0,
            "warm_score_mean": 0.0,
            "marker_dot_score_mean": 0.0,
            "nonblank": False,
        }
    stds = []
    edges = []
    colors = []
    blues = []
    warms = []
    markers = []
    for _, frame in frames:
        img = crop(frame, box)
        stds.append(float(img.std()))
        edges.append(edge_density(img))
        colors.append(colorfulness(img))
        blues.append(blue_score(img))
        warms.append(warm_score(img))
        markers.append(marker_dot_score(img))
    return {
        "pixel_std_mean": float(np.mean(stds)),
        "edge_density_mean": float(np.mean(edges)),
        "colorfulness_mean": float(np.mean(colors)),
        "blue_score_mean": float(np.mean(blues)),
        "warm_score_mean": float(np.mean(warms)),
        "marker_dot_score_mean": float(np.mean(markers)),
        "nonblank": bool(max(stds) > 1.0),
    }


def candidate_boxes() -> dict[str, tuple[float, float, float, float]]:
    # Relative boxes for current 1180x980 marker candidate videos.
    return {
        "scene": (16 / 1180, 50 / 980, 576 / 1180, 380 / 980),
        "marker_left": (24 / 1180, 405 / 980, 184 / 1180, 565 / 980),
        "marker_right": (214 / 1180, 405 / 980, 374 / 1180, 565 / 980),
        "fn_left": (620 / 1180, 72 / 980, 780 / 1180, 232 / 980),
        "fn_right": (816 / 1180, 72 / 980, 976 / 1180, 232 / 980),
        "ft_left": (620 / 1180, 280 / 980, 780 / 1180, 440 / 980),
        "ft_right": (816 / 1180, 280 / 980, 976 / 1180, 440 / 980),
        "area_left": (620 / 1180, 488 / 980, 780 / 1180, 648 / 980),
        "area_right": (816 / 1180, 488 / 980, 976 / 1180, 648 / 980),
        "curves": (20 / 1180, 585 / 980, 1120 / 1180, 825 / 980),
    }


def reference_boxes() -> dict[str, tuple[float, float, float, float]]:
    # Coarse bands from the user reference video layout. These are layout
    # checks only, not proof of matching sensor semantics.
    return {
        "scene_band": (0.0, 0.03, 1.0, 0.35),
        "tactile_band": (0.0, 0.30, 1.0, 0.56),
        "mechanics_band": (0.0, 0.52, 1.0, 0.96),
        "left_column": (0.0, 0.0, 0.34, 1.0),
        "middle_column": (0.33, 0.0, 0.67, 1.0),
        "right_column": (0.66, 0.0, 1.0, 1.0),
    }


def pass_item(name: str, passed: bool, evidence: str, limitation: str | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "limitation": limitation}


def make_audit_sheet(
    candidate_frames: list[tuple[int, np.ndarray]],
    reference_frames: list[tuple[int, np.ndarray]],
    out_path: Path,
) -> dict[str, Any]:
    cand = candidate_frames[-1][1] if candidate_frames else np.zeros((980, 1180, 3), dtype=np.uint8)
    ref = reference_frames[-1][1] if reference_frames else np.zeros((1510, 2846, 3), dtype=np.uint8)
    cand_img = Image.fromarray(cand).resize((590, 490), Image.Resampling.LANCZOS)
    ref_img = Image.fromarray(ref).resize((590, 313), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (1180, 850), (236, 236, 230))
    draw = ImageDraw.Draw(sheet)
    sheet.paste(ref_img, (0, 40))
    sheet.paste(cand_img, (590, 40))
    draw.text((10, 12), "reference: coarse scene/tactile/mechanics layout", fill=(20, 20, 20))
    draw.text((600, 12), "candidate: explicit scene/marker/Fn/Ft/area/curves layout", fill=(20, 20, 20))
    for label, box in reference_boxes().items():
        x0 = int(box[0] * 590)
        y0 = 40 + int(box[1] * 313)
        x1 = int(box[2] * 590)
        y1 = 40 + int(box[3] * 313)
        draw.rectangle((x0, y0, x1, y1), outline=(245, 180, 40), width=2)
        draw.text((x0 + 3, y0 + 3), label, fill=(25, 25, 25))
    for label, box in candidate_boxes().items():
        x0 = 590 + int(box[0] * 590)
        y0 = 40 + int(box[1] * 490)
        x1 = 590 + int(box[2] * 590)
        y1 = 40 + int(box[3] * 490)
        draw.rectangle((x0, y0, x1, y1), outline=(40, 190, 220), width=2)
        draw.text((x0 + 3, y0 + 3), label, fill=(25, 25, 25))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return {"path": str(out_path), "width": sheet.width, "height": sheet.height}


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary = load_json(args.candidate_summary)
    gate_review = load_json(args.gate_review_summary)
    ref_meta, ref_frames = read_video_best(args.reference_video, args.samples)
    cand_meta, cand_frames = read_video_best(args.candidate_video, args.samples)

    cand_regions = {name: region_metrics(cand_frames, box) for name, box in candidate_boxes().items()}
    ref_regions = {name: region_metrics(ref_frames, box) for name, box in reference_boxes().items()}

    candidate_has_scene = cand_regions["scene"]["nonblank"] and cand_regions["scene"]["edge_density_mean"] > 0.02
    candidate_has_marker = (
        candidate_summary.get("candidate_gel_marker_render")
        and max(cand_regions["marker_left"]["marker_dot_score_mean"], cand_regions["marker_right"]["marker_dot_score_mean"]) > 0.001
    )
    candidate_has_force_heatmaps = (
        candidate_summary.get("max_pad_object_candidate_fn_sum", 0.0) > 0.0
        and candidate_summary.get("max_pad_object_candidate_ft_sum", 0.0) > 0.0
        and max(cand_regions["fn_left"]["warm_score_mean"], cand_regions["fn_right"]["warm_score_mean"]) > 0.01
    )
    candidate_has_area_overlay = (
        candidate_summary.get("normal_area_overlay")
        and candidate_summary.get("max_left_candidate_contact_area_proxy_cell_ratio", 0.0) > 0.0
        and candidate_summary.get("max_right_candidate_contact_area_proxy_cell_ratio", 0.0) > 0.0
    )
    candidate_has_curves = cand_regions["curves"]["edge_density_mean"] > 0.015

    reference_has_scene = ref_regions["scene_band"]["edge_density_mean"] > 0.02
    reference_has_tactile = ref_regions["tactile_band"]["blue_score_mean"] > 0.02
    reference_has_mechanics = ref_regions["mechanics_band"]["edge_density_mean"] > 0.015
    reference_has_columns = all(ref_regions[key]["nonblank"] for key in ("left_column", "middle_column", "right_column"))

    checks = [
        pass_item("candidate_scene_channel", bool(candidate_has_scene), str(cand_regions["scene"])),
        pass_item("candidate_marker_render_channel", bool(candidate_has_marker), str({k: cand_regions[k] for k in ("marker_left", "marker_right")}), "candidate render is force-derived, not photometrically validated"),
        pass_item("candidate_force_heatmap_channels", bool(candidate_has_force_heatmaps), str({k: cand_regions[k] for k in ("fn_left", "fn_right", "ft_left", "ft_right")})),
        pass_item("candidate_area_proxy_channel", bool(candidate_has_area_overlay), str({k: candidate_summary.get(k) for k in ("max_left_candidate_contact_area_proxy_cell_ratio", "max_right_candidate_contact_area_proxy_cell_ratio")}), "area remains point-contact-density proxy"),
        pass_item("candidate_mechanics_curve_channel", bool(candidate_has_curves), str(cand_regions["curves"])),
        pass_item("reference_scene_tactile_mechanics_layout", bool(reference_has_scene and reference_has_tactile and reference_has_mechanics and reference_has_columns), str(ref_regions), "coarse reference layout detection only"),
    ]

    failed = [item["name"] for item in checks if not item["passed"]]
    hard_blockers = list(gate_review.get("hard_blockers", []))
    semantic_audit_sheet = make_audit_sheet(cand_frames, ref_frames, args.visual_dir / "channel_semantic_audit_sheet.jpg")
    status = "pass_channel_audit_open_validation" if not failed else "open_channel_audit_failed_checks"

    summary = {
        "classification": "phase00_channel_semantic_audit_v1",
        "run_tag": args.run_tag,
        "status": status,
        "not_training_result": True,
        "not_curiosity_success": True,
        "not_photometric_validation": True,
        "curiosity_training_allowed": False,
        "candidate_video": cand_meta,
        "reference_video": ref_meta,
        "candidate_region_metrics": cand_regions,
        "reference_region_metrics": ref_regions,
        "checks": checks,
        "failed_checks": failed,
        "passed_checks": [item["name"] for item in checks if item["passed"]],
        "remaining_validation_blockers": hard_blockers,
        "interpretation": "Channel-level visual layout audit exists, but tactile semantics remain unvalidated; Gate 00D/00E must stay open.",
        "semantic_audit_sheet": semantic_audit_sheet,
        "input_paths": {
            "candidate_summary": str(args.candidate_summary),
            "gate_review_summary": str(args.gate_review_summary),
            "candidate_video": str(args.candidate_video),
            "reference_video": str(args.reference_video),
        },
    }
    summary_path = args.output_dir / "channel_semantic_audit_summary.json"
    report_path = args.report_dir / "channel_semantic_audit.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(
        "# Phase 00 Channel Semantic Audit\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- failed_checks: `{failed}`\n"
        f"- semantic_audit_sheet: `{semantic_audit_sheet['path']}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This is a channel-level visual layout audit, not validated tactile photometry or final Gate 00D/00E completion.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--gate-review-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    summary = run(args)
    return 0 if str(summary["status"]).startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
