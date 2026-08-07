#!/usr/bin/env python3
"""Independently audit a Plan-10 static-feasibility H.264 review movie."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_manifest_path = args.render_root.resolve() / "manifest.json"
    replay_manifest_path = args.replay_root.resolve() / "manifest.json"
    render_manifest = json.loads(render_manifest_path.read_text())
    replay_manifest = json.loads(replay_manifest_path.read_text())
    scan_manifest_path = Path(replay_manifest["scan_manifest"])
    scan_manifest = json.loads(scan_manifest_path.read_text())
    replay_path = Path(replay_manifest["replay"])
    video_path = Path(render_manifest["video"])
    middle_path = Path(render_manifest["middle_frame"])
    if sha256(video_path) != render_manifest["video_sha256"]:
        raise RuntimeError("Video hash mismatch")
    if sha256(middle_path) != render_manifest["middle_frame_sha256"]:
        raise RuntimeError("Middle-frame hash mismatch")
    if sha256(replay_path) != replay_manifest["replay_sha256"]:
        raise RuntimeError("Replay hash mismatch")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_integer = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_integer >> (8 * index)) & 0xFF) for index in range(4))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    expected_frames = int(render_manifest["frame_count"])
    candidate_count = int(render_manifest["candidate_count"])
    hold_frames = int(render_manifest["hold_frames_per_candidate"])
    middle_video_index = (candidate_count // 2) * hold_frames
    middle_png = cv2.imread(str(middle_path), cv2.IMREAD_COLOR)
    if middle_png is None:
        raise RuntimeError("Cannot decode middle PNG")
    difference = middle_png.astype(np.float64) - decoded[middle_video_index].astype(
        np.float64
    )
    mean_squared_error = float(np.mean(difference * difference))
    psnr_db = float(10.0 * np.log10(255.0**2 / mean_squared_error))
    mean_absolute_error = float(np.mean(np.abs(difference)))
    first_last_difference = float(
        np.mean(np.abs(decoded[0].astype(np.float64) - decoded[-1].astype(np.float64)))
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    with np.load(replay_path, allow_pickle=False) as replay:
        recorded_position = np.asarray(
            [record["position_error_m"] for record in scan_manifest["records"]],
            dtype=np.float32,
        )
        recorded_rotation = np.asarray(
            [record["rotation_error_rad"] for record in scan_manifest["records"]],
            dtype=np.float32,
        )
        recorded_nonhand = np.asarray(
            [
                record["nonhand_contact_normal_load_n"]
                for record in scan_manifest["records"]
            ],
            dtype=np.float32,
        )
        recorded_inside = np.asarray(
            [
                record["head_collision_vertices_inside_box_pca_obb"]
                for record in scan_manifest["records"]
            ],
            dtype=np.int32,
        )
        replay_metrics_exact = (
            np.array_equal(replay["position_error_m"], recorded_position)
            and np.array_equal(replay["rotation_error_rad"], recorded_rotation)
            and np.array_equal(replay["nonhand_load_n"], recorded_nonhand)
            and np.array_equal(replay["head_inside_vertices"], recorded_inside)
        )
        no_candidate = not bool(replay["static_candidate"].any())
        replay_candidate_count = int(replay["body_pose_wxyz"].shape[0])

    checks = {
        "static_scan_audit_passed": bool(scan_manifest["audit_passed"]),
        "static_scan_has_zero_candidates": scan_manifest["candidate_count"] == 0,
        "replay_export_passed": bool(replay_manifest["export_passed"]),
        "render_manifest_passed": bool(render_manifest["render_passed"]),
        "hash_chain_exact": (
            render_manifest["replay_manifest_sha256"]
            == sha256(replay_manifest_path)
            and replay_manifest["scan_manifest_sha256"]
            == sha256(scan_manifest_path)
            and replay_manifest["solutions_sha256"]
            == sha256(Path(replay_manifest["solutions"]))
            and replay_manifest["geometry_manifest_sha256"]
            == sha256(Path(replay_manifest["geometry_manifest"]))
        ),
        "replay_metrics_equal_static_manifest": replay_metrics_exact,
        "negative_gate_preserved_in_replay": no_candidate,
        "all_candidates_present": replay_candidate_count == candidate_count,
        "opencv_reports_h264_or_avc1": fourcc.lower() in ("h264", "avc1"),
        "ffmpeg_reports_h264_high_yuv420p": (
            "Video: h264 (High)" in probe.stderr and "yuv420p" in probe.stderr
        ),
        "resolution_1280x720": (width, height) == (1280, 720),
        "fps_matches_manifest": abs(fps - float(render_manifest["fps"])) <= 1.0e-9,
        "reported_frame_count_matches_manifest": reported_frames == expected_frames,
        "decoded_frame_count_matches_manifest": len(decoded) == expected_frames,
        "middle_h264_psnr_ge_30db": psnr_db >= 30.0,
        "middle_h264_mean_absolute_error_le_3": mean_absolute_error <= 3.0,
        "video_has_temporal_change": first_last_difference >= 1.0,
        "middle_frame_nonblank": (
            int(decoded[middle_video_index].max())
            - int(decoded[middle_video_index].min())
            >= 100
        ),
        "official_pose_reconstruction_within_2e_6m": (
            replay_manifest["maximum_reconstructed_hand_position_error_m"]
            <= 2.0e-6
            and replay_manifest["maximum_reconstructed_box_position_error_m"]
            <= 2.0e-6
        ),
    }
    payload = {
        "schema": "plan10_static_feasibility_video_independent_audit_v1",
        "claim_boundary": "A pass proves exact source/hash binding, complete H.264 decoding, and preservation of the negative static result. It does not establish dynamics, grasp, lift, tactile, policy, or success.",
        "checks": checks,
        "independent_video_audit_passed": all(checks.values()),
        "video": str(video_path),
        "video_sha256": sha256(video_path),
        "codec_fourcc": fourcc,
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "decoded_frames": len(decoded),
        "duration_s": len(decoded) / fps,
        "middle_h264_psnr_db": psnr_db,
        "middle_h264_mean_absolute_error": mean_absolute_error,
        "first_last_mean_absolute_difference": first_last_difference,
        "render_manifest": str(render_manifest_path),
        "render_manifest_sha256": sha256(render_manifest_path),
        "replay_manifest": str(replay_manifest_path),
        "replay_manifest_sha256": sha256(replay_manifest_path),
        "scan_manifest": str(scan_manifest_path),
        "scan_manifest_sha256": sha256(scan_manifest_path),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["independent_video_audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
