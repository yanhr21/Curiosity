#!/usr/bin/env python3
"""Render separate H.264 evidence videos for trained XSkill prototype sequences."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch


WIDTH = 960
HEIGHT = 540
FPS = 10
DEFAULT_FFMPEG = Path(
    "/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
MOTIONS = (
    ("CarryBox", 45, "train", "reference"),
    ("KickBox", 21, "train", "reference"),
    ("CarryBox", 99, "test", "heldout"),
    ("KickBox", 89, "test", "heldout"),
)


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("official_xskill_sugar_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prototype_color(index: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[int(index * 179 / 127), 190, 225]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def open_encoder(path: Path, ffmpeg: Path):
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def draw_text(canvas, text, xy, scale=0.55, color=(20, 20, 20), thickness=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_motion(
    directory: Path,
    probabilities: np.ndarray,
    task: str,
    motion_id: int,
    role: str,
    output: Path,
    decision: str,
    ffmpeg: Path,
) -> None:
    frames = []
    for path in sorted(directory.glob("*.png"), key=lambda item: int(item.stem)):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode {path}")
        frames.append(frame)
    if len(frames) != 64 or probabilities.shape != (56, 128):
        raise RuntimeError("expected 64 RGB frames and 56x128 prototype probabilities")
    assignments = np.argmax(probabilities, axis=1)
    encoder = open_encoder(output, ffmpeg)
    if encoder.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")
    for frame_index, world in enumerate(frames):
        canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        canvas[:, :HEIGHT] = cv2.resize(world, (HEIGHT, HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.line(canvas, (HEIGHT, 0), (HEIGHT, HEIGHT), (190, 190, 190), 1)
        clip_index = int(np.clip(frame_index - 8, 0, 55))
        start, end = clip_index, clip_index + 8
        draw_text(canvas, f"Official XSkill | {task} {motion_id}", (565, 34), 0.68, thickness=2)
        draw_text(canvas, f"{role} motion | source split: {directory.parent.parent.name}", (565, 62), 0.48)
        draw_text(canvas, f"RGB frame {frame_index:02d}/63 | clip {start:02d}:{end:02d}", (565, 89), 0.5)
        draw_text(canvas, f"machine gate: {decision}", (565, 116), 0.5)
        draw_text(canvas, "representation only - no policy or action", (565, 143), 0.48, (0, 0, 170), 1)

        timeline_x, timeline_y, timeline_w, timeline_h = 565, 174, 370, 34
        for index, prototype in enumerate(assignments):
            x0 = timeline_x + round(index * timeline_w / len(assignments))
            x1 = timeline_x + round((index + 1) * timeline_w / len(assignments))
            cv2.rectangle(
                canvas,
                (x0, timeline_y),
                (max(x0 + 1, x1), timeline_y + timeline_h),
                prototype_color(int(prototype)),
                -1,
            )
        cursor = timeline_x + round(clip_index * timeline_w / (len(assignments) - 1))
        cv2.line(canvas, (cursor, timeline_y - 5), (cursor, timeline_y + timeline_h + 5), (0, 0, 0), 2)
        draw_text(canvas, "argmax prototype sequence (time ->)", (timeline_x, timeline_y + 58), 0.46)

        current = probabilities[clip_index]
        top = np.argsort(current)[-8:][::-1]
        draw_text(canvas, "current top prototypes", (565, 258), 0.55, thickness=2)
        maximum = max(float(current[top[0]]), 1e-12)
        for rank, prototype in enumerate(top):
            y = 287 + rank * 28
            probability = float(current[prototype])
            bar_width = int(245 * probability / maximum)
            cv2.rectangle(canvas, (645, y - 13), (645 + bar_width, y + 4), prototype_color(int(prototype)), -1)
            draw_text(canvas, f"P{int(prototype):03d}", (565, y + 2), 0.46)
            draw_text(canvas, f"{probability:.4f}", (895, y + 2), 0.42)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0 or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed for {output} with code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    args = parser.parse_args()
    result = json.loads((args.experiment / "REPRESENTATION_RESULT.json").read_text(encoding="utf-8"))
    decision = "PASS" if result["passed"] else "FAIL"
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"H.264 encoder not found: {args.ffmpeg}")
    adapter = load_adapter(args.adapter.resolve())
    official = adapter.import_official_xskill(args.official_repo.resolve())
    device = torch.device(args.device)
    model = adapter.load_model_checkpoint(
        official, args.experiment / "model_epoch79.pt", device
    ).eval()
    pipeline = official.get_transform_pipeline(["center_crop_112_112", "normalize"]).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for page, (task, motion_id, split, role) in enumerate(MOTIONS, start=1):
        directory = args.corpus / split / task / str(motion_id)
        embedded = adapter.embed_motion(model, pipeline, directory, device)
        output = args.output / f"{page:02d}_{task.lower()}_{motion_id}_{role}_xskill_prototypes.mp4"
        render_motion(
            directory,
            embedded["prototype"],
            task,
            motion_id,
            role,
            output,
            decision,
            args.ffmpeg,
        )
        manifest.append({"task": task, "motion_id": motion_id, "role": role, "video": str(output)})
    (args.output / "VIDEO_RESULT.json").write_text(
        json.dumps({"protocol": "official_xskill_prototype_video_v1", "videos": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"XSKILL_PROTOTYPE_VIDEOS_READY count={len(manifest)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
