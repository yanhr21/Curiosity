#!/usr/bin/env python3
"""Compare the active tactile asset against the user reference video.

This diagnostic extracts synchronized contact sheets and simple visual metrics
from the reference video and the current candidate tactile video. It also emits
a gate checklist describing which reference-video tactile requirements are met
by current evidence and which remain open.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:  # noqa: BLE001
        return None


def sample_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return []
    return [int(v) for v in np.linspace(0, frame_count - 1, min(samples, frame_count), dtype=int)]


def read_video_cv2(path: Path, samples: int) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]]]:
    cv2 = import_cv2()
    if cv2 is None:
        raise RuntimeError("cv2 is unavailable")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[tuple[int, np.ndarray]] = []
    for idx in sample_indices(frame_count, samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append((idx, rgb))
    cap.release()
    meta = {
        "decoder": "cv2",
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": float(frame_count / fps) if fps > 0.0 else None,
        "sampled_frames": len(frames),
        "sample_indices": [int(i) for i, _ in frames],
    }
    return meta, frames


def read_video_avi_dib(path: Path, samples: int) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]]]:
    """Read uncompressed DIB AVI files produced by this project."""
    frames: list[tuple[int, np.ndarray]] = []
    with path.open("rb") as handle:
        data = handle.read()
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise RuntimeError(f"not a RIFF AVI file: {path}")

    avih_pos = data.find(b"avih")
    if avih_pos < 0:
        raise RuntimeError("AVI missing avih chunk")
    avih_size = struct.unpack_from("<I", data, avih_pos + 4)[0]
    avih = data[avih_pos + 8 : avih_pos + 8 + avih_size]
    if len(avih) < 40:
        raise RuntimeError("AVI avih chunk too short")
    microsec_per_frame, _, _, _, total_frames, _, _, _, width, height = struct.unpack_from("<IIIIIIIIII", avih, 0)
    frame_count = int(total_frames)
    fps = float(1_000_000.0 / microsec_per_frame) if microsec_per_frame > 0 else 0.0
    width = int(width)
    height = int(height)
    row_bytes = width * 3
    stride = (row_bytes + 3) & ~3
    expected_size = stride * height
    wanted = set(sample_indices(frame_count, samples))

    movi_pos = data.find(b"movi")
    if movi_pos < 0:
        raise RuntimeError("AVI missing movi list")
    pos = movi_pos + 4
    frame_idx = 0
    while pos + 8 <= len(data) and frame_idx < frame_count:
        fourcc = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        payload_start = pos + 8
        payload_end = payload_start + size
        if fourcc == b"idx1":
            break
        if fourcc in (b"00db", b"00dc"):
            if frame_idx in wanted and payload_end <= len(data) and size >= expected_size:
                payload = data[payload_start : payload_start + expected_size]
                rows = []
                for row in range(height):
                    start = row * stride
                    rows.append(np.frombuffer(payload[start : start + row_bytes], dtype=np.uint8).reshape(width, 3))
                bgr_bottom_up = np.stack(rows, axis=0)
                rgb = bgr_bottom_up[::-1, :, ::-1].copy()
                frames.append((frame_idx, rgb))
            frame_idx += 1
        pos = payload_end + (size % 2)
    meta = {
        "decoder": "avi_dib_builtin",
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": float(frame_count / fps) if fps > 0.0 else None,
        "sampled_frames": len(frames),
        "sample_indices": [int(i) for i, _ in frames],
    }
    return meta, frames


def read_video_imageio(path: Path, samples: int) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]]]:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"imageio unavailable: {type(exc).__name__}: {exc}") from exc

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        meta_in = reader.get_meta_data()
        fps = float(meta_in.get("fps") or 0.0)
        size = meta_in.get("size") or (0, 0)
        width, height = int(size[0]), int(size[1])
        try:
            frame_count = int(reader.count_frames())
        except Exception:  # noqa: BLE001
            duration = float(meta_in.get("duration") or 0.0)
            frame_count = int(round(duration * fps)) if fps > 0.0 and duration > 0.0 else 0
        frames: list[tuple[int, np.ndarray]] = []
        for idx in sample_indices(frame_count, samples):
            frame = np.asarray(reader.get_data(idx))[..., :3].astype(np.uint8)
            frames.append((idx, frame))
        meta = {
            "decoder": "imageio_ffmpeg",
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": float(frame_count / fps) if fps > 0.0 and frame_count > 0 else meta_in.get("duration"),
            "sampled_frames": len(frames),
            "sample_indices": [int(i) for i, _ in frames],
            "raw_metadata": {k: str(v) for k, v in meta_in.items()},
        }
        return meta, frames
    finally:
        reader.close()


def read_video_best(path: Path, samples: int) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]]]:
    errors = []
    try:
        return read_video_cv2(path, samples)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cv2: {type(exc).__name__}: {exc}")
    if path.suffix.lower() != ".avi":
        try:
            return read_video_imageio(path, samples)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"imageio_ffmpeg: {type(exc).__name__}: {exc}")
    if path.suffix.lower() == ".avi":
        try:
            return read_video_avi_dib(path, samples)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"avi_dib_builtin: {type(exc).__name__}: {exc}")
    meta = {
        "decoder": None,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "duration_s": None,
        "sampled_frames": 0,
        "sample_indices": [],
        "decode_errors": errors,
    }
    return meta, []


def edge_density(frame: np.ndarray) -> float:
    gray = np.asarray(frame, dtype=np.float32).mean(axis=2)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    mag = np.zeros_like(gray)
    mag[:, 1:] += gx
    mag[1:, :] += gy
    threshold = max(10.0, float(np.quantile(mag, 0.90)))
    return float((mag > threshold).mean())


def colorfulness(frame: np.ndarray) -> float:
    arr = np.asarray(frame, dtype=np.float32)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    return float(math.sqrt(float(np.var(rg)) + float(np.var(yb))) + 0.3 * math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2)))


def frame_metrics(frames: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
    if not frames:
        return {
            "nonblank": False,
            "pixel_std_mean": 0.0,
            "pixel_std_max": 0.0,
            "edge_density_mean": 0.0,
            "colorfulness_mean": 0.0,
            "motion_delta_mean": 0.0,
        }
    stds = np.asarray([float(np.asarray(frame).std()) for _, frame in frames], dtype=np.float32)
    edges = np.asarray([edge_density(frame) for _, frame in frames], dtype=np.float32)
    colors = np.asarray([colorfulness(frame) for _, frame in frames], dtype=np.float32)
    deltas = []
    for (_, a), (_, b) in zip(frames[:-1], frames[1:], strict=False):
        aa = np.asarray(Image.fromarray(a).resize((160, 90), Image.Resampling.BILINEAR), dtype=np.float32)
        bb = np.asarray(Image.fromarray(b).resize((160, 90), Image.Resampling.BILINEAR), dtype=np.float32)
        deltas.append(float(np.mean(np.abs(aa - bb))))
    return {
        "nonblank": bool(float(stds.max(initial=0.0)) > 1.0),
        "pixel_std_mean": float(stds.mean()),
        "pixel_std_max": float(stds.max(initial=0.0)),
        "edge_density_mean": float(edges.mean()),
        "edge_density_max": float(edges.max(initial=0.0)),
        "colorfulness_mean": float(colors.mean()),
        "colorfulness_max": float(colors.max(initial=0.0)),
        "motion_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "motion_delta_max": float(np.max(deltas)) if deltas else 0.0,
    }


def make_sheet(
    title: str,
    frames: list[tuple[int, np.ndarray]],
    out_path: Path,
    thumb_w: int = 360,
    cols: int = 3,
) -> dict[str, Any]:
    if not frames:
        img = Image.new("RGB", (thumb_w, 120), (240, 240, 236))
        ImageDraw.Draw(img).text((12, 12), f"{title}: no frames decoded", fill=(20, 20, 20))
        img.save(out_path, quality=92)
        return {"path": str(out_path), "width": img.width, "height": img.height, "frames": 0}
    thumbs = []
    for idx, frame in frames:
        im = Image.fromarray(frame)
        scale = thumb_w / max(1, im.width)
        thumb_h = max(1, int(im.height * scale))
        thumb = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (thumb_w, thumb_h + 24), (245, 245, 240))
        panel.paste(thumb, (0, 0))
        ImageDraw.Draw(panel).text((6, thumb_h + 5), f"frame {idx}", fill=(20, 20, 20))
        thumbs.append(panel)
    rows = int(math.ceil(len(thumbs) / cols))
    title_h = 32
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows * cell_h), (236, 236, 230))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill=(20, 20, 20))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = title_h + (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, quality=92)
    return {"path": str(out_path), "width": sheet.width, "height": sheet.height, "frames": len(thumbs)}


def make_comparison_sheet(
    reference_frames: list[tuple[int, np.ndarray]],
    candidate_frames: list[tuple[int, np.ndarray]],
    out_path: Path,
) -> dict[str, Any]:
    pairs = list(zip(reference_frames, candidate_frames, strict=False))
    if not pairs:
        img = Image.new("RGB", (900, 120), (240, 240, 236))
        ImageDraw.Draw(img).text((12, 12), "No comparison frames decoded", fill=(20, 20, 20))
        img.save(out_path, quality=92)
        return {"path": str(out_path), "width": img.width, "height": img.height, "pairs": 0}
    ref_w = 430
    cand_w = 430
    row_h = 260
    title_h = 40
    rows = len(pairs)
    sheet = Image.new("RGB", (ref_w + cand_w, title_h + rows * row_h), (236, 236, 230))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "reference video", fill=(20, 20, 20))
    draw.text((ref_w + 10, 10), "current candidate steel-spec direct-force video", fill=(20, 20, 20))
    for row, ((ref_idx, ref_frame), (cand_idx, cand_frame)) in enumerate(pairs):
        y = title_h + row * row_h
        for x, width, label, idx, frame in [
            (0, ref_w, "ref", ref_idx, ref_frame),
            (ref_w, cand_w, "candidate", cand_idx, cand_frame),
        ]:
            im = Image.fromarray(frame)
            scale = width / max(1, im.width)
            h = min(row_h - 28, max(1, int(im.height * scale)))
            thumb = im.resize((width, h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.text((x + 6, y + h + 5), f"{label} frame {idx}", fill=(20, 20, 20))
    sheet.save(out_path, quality=92)
    return {"path": str(out_path), "width": sheet.width, "height": sheet.height, "pairs": len(pairs)}


def gate_checklist(candidate_summary: dict[str, Any] | None) -> dict[str, Any]:
    current_channels = [
        "real Newton SensorTiledCamera head/right-wrist/left-wrist scene views",
        "left/right candidate direct Fn maps",
        "left/right candidate direct Ft maps",
        "left/right candidate shear arrows in pad-local plane",
        "object_z and candidate force time-series curves",
        "steel-spec mu/kh material arrays and notify_model_changed evidence",
        "compatible-scene SensorContact alignment evidence from p00_mjw_align_v1_20260701_055200",
    ]
    missing = [
        "gel/marker-style tactile camera rendering comparable to the reference video",
        "validated photometric/deformation marker tracking on the pad surface",
        "direct visual overlay of contact normals and contact area in the same direct-force video",
        "reference-video channel-by-channel semantic matching beyond frame-level visual metrics",
        "final Gate 00D/00E review before restarting curiosity training",
    ]
    if candidate_summary is None:
        missing.insert(0, "candidate summary was unavailable")
    elif candidate_summary.get("normal_area_overlay"):
        current_channels.append("candidate contact-normal overlay from MJWarp contact.frame")
        current_channels.append("candidate contact-area proxy overlay from pad-object point-contact density")
        missing = [
            gap
            for gap in missing
            if gap != "direct visual overlay of contact normals and contact area in the same direct-force video"
        ]
        missing.insert(
            2,
            "validated real contact-area semantics beyond the current point-contact-density proxy",
        )
    if candidate_summary is not None and candidate_summary.get("candidate_gel_marker_render"):
        current_channels.append("candidate gel/marker-style rendering derived from direct-force fields")
        missing = [
            gap
            for gap in missing
            if gap != "gel/marker-style tactile camera rendering comparable to the reference video"
        ]
        missing.insert(
            0,
            "validated gel/marker photometric semantics comparable to the reference video",
        )
    return {
        "candidate_current_channels": current_channels,
        "remaining_reference_video_gaps": missing,
        "curiosity_training_allowed": False,
        "reason_curiosity_training_not_allowed": "reference-video-level gel/marker tactile comparison and final Gate 00D/00E review are still open",
    }


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    ref_meta, ref_frames = read_video_best(args.reference_video, args.samples)
    cand_meta, cand_frames = read_video_best(args.candidate_video, args.samples)
    ref_metrics = frame_metrics(ref_frames)
    cand_metrics = frame_metrics(cand_frames)
    ref_sheet = make_sheet("reference video sampled frames", ref_frames, args.visual_dir / "reference_sheet.jpg")
    cand_sheet = make_sheet("candidate steel-spec direct-force sampled frames", cand_frames, args.visual_dir / "candidate_sheet.jpg")
    comparison_sheet = make_comparison_sheet(ref_frames, cand_frames, args.visual_dir / "reference_vs_candidate_sheet.jpg")
    candidate_summary = load_json(args.candidate_summary)

    if ref_metrics["nonblank"] and cand_metrics["nonblank"]:
        status = "pass_reference_comparison_assets"
    elif cand_metrics["nonblank"] and not ref_metrics["nonblank"]:
        status = "blocked_reference_video_decode_missing_dependency"
    else:
        status = "blocked_blank_or_unreadable_candidate_video"

    summary = {
        "classification": "phase00_reference_video_candidate_tactile_comparison_v1",
        "run_tag": args.run_tag,
        "status": status,
        "not_training_result": True,
        "not_curiosity_success": True,
        "reference_video": ref_meta,
        "candidate_video": cand_meta,
        "reference_metrics": ref_metrics,
        "candidate_metrics": cand_metrics,
        "reference_sheet": ref_sheet,
        "candidate_sheet": cand_sheet,
        "comparison_sheet": comparison_sheet,
        "candidate_summary_path": str(args.candidate_summary) if args.candidate_summary else None,
        "candidate_summary_status": candidate_summary.get("status") if candidate_summary else None,
        "candidate_summary_force_sign": candidate_summary.get("force_sign_convention") if candidate_summary else None,
        "gate_checklist": gate_checklist(candidate_summary),
    }
    summary_path = args.output_dir / "reference_video_compare_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    report_path = args.report_dir / "reference_video_compare.md"
    report_path.write_text(
        "# Phase 00 Reference Video Comparison\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- reference: `{args.reference_video}`\n"
        f"- candidate: `{args.candidate_video}`\n"
        f"- reference sampled frames: `{ref_meta['sampled_frames']}`\n"
        f"- candidate sampled frames: `{cand_meta['sampled_frames']}`\n"
        f"- reference nonblank: `{ref_metrics['nonblank']}`\n"
        f"- candidate nonblank: `{cand_metrics['nonblank']}`\n"
        f"- comparison sheet: `{comparison_sheet['path']}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This is a comparison asset and gate checklist, not training and not curiosity success.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    return 0 if summary["status"] in {"pass_reference_comparison_assets", "blocked_reference_video_decode_missing_dependency"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
