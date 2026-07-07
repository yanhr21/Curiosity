#!/usr/bin/env python3
"""Render a recorded Core-World G1 replay CSV as RGB frames.

This is a visualization-only replay path.  It does not run the controller or
claim new carrying evidence; it loads the real G1 USD and applies recorded root,
joint, and box states from a prior non-rendered rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac rendering on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render recorded G1 replay CSV frames.")
    parser.add_argument("--replay-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g1-usd", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd"))
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.14, 0.10, 0.08), metavar=("X", "Y", "Z"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--capture-every-n-rows", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--camera-prim", default="/World/G1ReplayCamera")
    parser.add_argument("--frame-prim", default="/World/G1")
    parser.add_argument("--follow-frame", action="store_true")
    parser.add_argument("--frame-zoom", type=float, default=0.42)
    parser.add_argument("--capture-backend", choices=("auto", "replicator", "app-screenshot"), default="auto")
    parser.add_argument(
        "--articulation-wrapper",
        action="store_true",
        help="Use Isaac Core SingleArticulation to set joints. Default is USD Xform-only replay for render robustness.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: E402

WORLD_IMPORT_ERROR = None
try:
    from isaacsim.core.api import World  # noqa: E402
    from isaacsim.core.prims import SingleArticulation  # noqa: E402
except Exception as exc:  # noqa: BLE001
    World = None  # type: ignore[assignment]
    SingleArticulation = None  # type: ignore[assignment]
    WORLD_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[WARN] Core World/SingleArticulation import failed: {WORLD_IMPORT_ERROR}", flush=True)

REPLICATOR_IMPORT_ERROR = None
try:
    print("[PROGRESS] enabling omni.replicator.core", flush=True)
    enable_extension("omni.replicator.core")
    for _ in range(5):
        simulation_app.update()
    import omni.replicator.core as rep  # noqa: E402
except Exception as exc:  # noqa: BLE001
    rep = None  # type: ignore[assignment]
    REPLICATOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[WARN] Replicator import failed: {REPLICATOR_IMPORT_ERROR}", flush=True)

RENDERING_MANAGER_IMPORT_ERROR = None
try:
    print("[PROGRESS] enabling isaacsim.core.rendering_manager", flush=True)
    enable_extension("isaacsim.core.rendering_manager")
    for _ in range(5):
        simulation_app.update()
    from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
except Exception as exc:  # noqa: BLE001
    ViewportManager = None  # type: ignore[assignment]
    RENDERING_MANAGER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[WARN] Rendering manager import failed: {RENDERING_MANAGER_IMPORT_ERROR}", flush=True)


G1_PATH = "/World/G1"
G1_ARTICULATION_PATH = "/World/G1/pelvis"
BOX_PATH = "/World/CarryBoxReplay"
ROBOT_TRAIL_SCOPE = "/World/RobotReplayTrail"
BOX_TRAIL_SCOPE = "/World/BoxReplayTrail"

DEBUG_EVENTS: list[dict[str, object]] = []


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trace(event: str, **payload: object) -> None:
    DEBUG_EVENTS.append({"event": event, **payload})
    try:
        _write_json(args_cli.output_dir / "render_debug_trace.json", {"events": DEBUG_EVENTS})
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not write render debug trace: {type(exc).__name__}: {exc}", flush=True)


def _write_failure_summary(error: BaseException, stage: str) -> None:
    frame_dir = args_cli.output_dir / "rgb_frames"
    summary = {
        "replay_csv": str(args_cli.replay_csv),
        "frame_dir": str(frame_dir),
        "captured_frames": len(list(frame_dir.glob("*.png"))) if frame_dir.is_dir() else 0,
        "status": "fail",
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "failure_stage": stage,
        "capture_backend_requested": str(getattr(args_cli, "capture_backend", "unknown")),
        "capture_backend_used": None,
        "replicator_error": REPLICATOR_IMPORT_ERROR,
        "rendering_manager_error": RENDERING_MANAGER_IMPORT_ERROR,
        "world_import_error": WORLD_IMPORT_ERROR,
        "debug_events": DEBUG_EVENTS,
        "success_claim": "visual_replay_failed_no_control_evidence",
    }
    try:
        _write_json(args_cli.output_dir / "g1_replay_render_summary.json", summary)
        print(f"[INFO] Replay render failure summary written to: {args_cli.output_dir / 'g1_replay_render_summary.json'}", flush=True)
    except Exception as write_exc:  # noqa: BLE001
        print(f"[WARN] could not write render failure summary: {type(write_exc).__name__}: {write_exc}", flush=True)


def _quat_wxyz_to_euler_deg(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _spawn_cube(stage, path: str, size: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _set_camera_look_at(
    stage,
    camera_path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    look_at = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(look_at.GetInverse())


def _camera_target_from_row(row: dict[str, str]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    robot_xyz = _row_xyz(row, "robot") or (0.0, 0.0, 0.7)
    box_xyz = _row_xyz(row, "box") or robot_xyz
    target = (
        0.5 * (robot_xyz[0] + box_xyz[0]),
        0.5 * (robot_xyz[1] + box_xyz[1]),
        max(0.55, 0.5 * (robot_xyz[2] + box_xyz[2])),
    )
    eye = (target[0] + 1.45, target[1] - 2.15, target[2] + 0.95)
    return eye, (target[0], target[1], target[2] + 0.08)


def _write_rgb_png(rgb_data: object, path: Path) -> None:
    array = np.asarray(rgb_data)
    if array.ndim == 1:
        raise RuntimeError(f"Unexpected flat RGB annotator output shape: {array.shape}")
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _capture_app_screenshot_png(path: Path, max_wait_frames: int = 20) -> bool:
    import omni.kit.renderer.capture  # noqa: PLC0415

    renderer = omni.kit.renderer.capture.acquire_renderer_capture_interface()
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer.capture_next_frame_swapchain(str(path))
    for _ in range(2):
        simulation_app.update()
    if hasattr(renderer, "wait_async_capture"):
        renderer.wait_async_capture()
    for _ in range(max_wait_frames):
        if path.is_file():
            return True
        simulation_app.update()
    return path.is_file()


def _set_cube_pose(stage, path: str, row: dict[str, str]) -> None:
    if row.get("box_x", "") == "":
        return
    prim = stage.GetPrimAtPath(path)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(row["box_x"]), float(row["box_y"]), float(row["box_z"])))
    api.SetRotate(Gf.Vec3f(*_quat_wxyz_to_euler_deg(float(row["box_qw"]), float(row["box_qx"]), float(row["box_qy"]), float(row["box_qz"]))))


def _set_robot_xform_pose(stage, path: str, row: dict[str, str]) -> None:
    prim = stage.GetPrimAtPath(path)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(row["robot_x"]), float(row["robot_y"]), float(row["robot_z"])))
    api.SetRotate(
        Gf.Vec3f(
            *_quat_wxyz_to_euler_deg(
                float(row["robot_qw"]),
                float(row["robot_qx"]),
                float(row["robot_qy"]),
                float(row["robot_qz"]),
            )
        )
    )


def _replicator_step(rt_subframes: int = 4) -> None:
    if rep is not None and hasattr(rep, "orchestrator") and hasattr(rep.orchestrator, "step"):
        try:
            rep.orchestrator.step(rt_subframes=max(1, int(rt_subframes)))
            return
        except TypeError:
            rep.orchestrator.step()
            return
    simulation_app.update()


def _camera_path(camera: object) -> str | None:
    get_path = getattr(camera, "GetPath", None)
    if callable(get_path):
        return str(get_path())
    prim_path = getattr(camera, "prim_path", None)
    if prim_path is not None:
        return str(prim_path)
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _row_xyz(row: dict[str, str], prefix: str) -> tuple[float, float, float] | None:
    try:
        return (float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _spawn_replay_trails(stage, rows: list[dict[str, str]], capture_every: int, max_frames: int) -> dict[str, object]:
    """Add simple persistent floor markers so the replay reads as motion."""
    UsdGeom.Scope.Define(stage, ROBOT_TRAIL_SCOPE)
    UsdGeom.Scope.Define(stage, BOX_TRAIL_SCOPE)
    selected_rows = rows[:: max(1, capture_every)]
    if max_frames >= 0:
        selected_rows = selected_rows[:max_frames]
    if not selected_rows:
        return {"trail_markers": 0}

    marker_count = 0
    stride = max(1, len(selected_rows) // 16)
    for marker_idx, row in enumerate(selected_rows[::stride]):
        robot_xyz = _row_xyz(row, "robot")
        box_xyz = _row_xyz(row, "box")
        if robot_xyz is not None:
            path = f"{ROBOT_TRAIL_SCOPE}/robot_{marker_idx:03d}"
            _spawn_cube(stage, path, (0.035, 0.035, 0.018), (0.15, 0.45, 0.82))
            UsdGeom.XformCommonAPI(stage.GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(robot_xyz[0], robot_xyz[1], 0.012))
            marker_count += 1
        if box_xyz is not None:
            path = f"{BOX_TRAIL_SCOPE}/box_{marker_idx:03d}"
            _spawn_cube(stage, path, (0.045, 0.045, 0.022), (0.92, 0.65, 0.18))
            UsdGeom.XformCommonAPI(stage.GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(box_xyz[0], box_xyz[1], 0.016))
            marker_count += 1

    first_robot = _row_xyz(selected_rows[0], "robot")
    last_robot = _row_xyz(selected_rows[-1], "robot")
    first_box = _row_xyz(selected_rows[0], "box")
    last_box = _row_xyz(selected_rows[-1], "box")
    if first_robot is not None:
        _spawn_cube(stage, "/World/ReplayRobotStartMarker", (0.08, 0.08, 0.03), (0.05, 0.22, 0.65))
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/ReplayRobotStartMarker")).SetTranslate(Gf.Vec3d(first_robot[0], first_robot[1], 0.025))
    if last_robot is not None:
        _spawn_cube(stage, "/World/ReplayRobotEndMarker", (0.09, 0.09, 0.035), (0.05, 0.55, 0.18))
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/ReplayRobotEndMarker")).SetTranslate(Gf.Vec3d(last_robot[0], last_robot[1], 0.03))
    if first_box is not None:
        _spawn_cube(stage, "/World/ReplayBoxStartMarker", (0.09, 0.09, 0.035), (0.55, 0.35, 0.08))
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/ReplayBoxStartMarker")).SetTranslate(Gf.Vec3d(first_box[0], first_box[1], 0.035))
    if last_box is not None:
        _spawn_cube(stage, "/World/ReplayBoxTargetMarker", (0.12, 0.12, 0.05), (0.95, 0.2, 0.12))
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/ReplayBoxTargetMarker")).SetTranslate(Gf.Vec3d(last_box[0], last_box[1], 0.045))

    return {
        "trail_markers": marker_count,
        "selected_replay_rows": len(selected_rows),
        "robot_start_xyz": first_robot,
        "robot_end_xyz": last_robot,
        "box_start_xyz": first_box,
        "box_end_xyz": last_box,
    }


def main() -> None:
    _trace("main_entered")
    if not args_cli.replay_csv.is_file():
        raise FileNotFoundError(args_cli.replay_csv)
    if not args_cli.g1_usd.is_file():
        raise FileNotFoundError(args_cli.g1_usd)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args_cli.output_dir / "rgb_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    requested_backend = str(args_cli.capture_backend)
    use_replicator = requested_backend in ("auto", "replicator") and rep is not None
    use_app_screenshot = requested_backend in ("auto", "app-screenshot") and not use_replicator
    _trace(
        "backend_selected",
        requested_backend=requested_backend,
        use_replicator=use_replicator,
        use_app_screenshot=use_app_screenshot,
        replicator_error=REPLICATOR_IMPORT_ERROR,
        rendering_manager_error=RENDERING_MANAGER_IMPORT_ERROR,
        articulation_wrapper=bool(args_cli.articulation_wrapper),
    )
    if requested_backend == "replicator" and rep is None:
        summary = {
            "replay_csv": str(args_cli.replay_csv),
            "frame_dir": str(frame_dir),
            "captured_frames": 0,
            "status": "fail",
            "error": REPLICATOR_IMPORT_ERROR,
            "capture_backend_requested": requested_backend,
            "capture_backend_used": None,
            "success_claim": "visual_replay_failed_no_control_evidence",
        }
        (args_cli.output_dir / "g1_replay_render_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"[INFO] Replay render failure summary written to: {args_cli.output_dir / 'g1_replay_render_summary.json'}", flush=True)
        return
    if use_app_screenshot and ViewportManager is None:
        summary = {
            "replay_csv": str(args_cli.replay_csv),
            "frame_dir": str(frame_dir),
            "captured_frames": 0,
            "status": "fail",
            "error": RENDERING_MANAGER_IMPORT_ERROR,
            "replicator_error": REPLICATOR_IMPORT_ERROR,
            "capture_backend_requested": requested_backend,
            "capture_backend_used": None,
            "success_claim": "visual_replay_failed_no_control_evidence",
        }
        (args_cli.output_dir / "g1_replay_render_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"[INFO] Replay render failure summary written to: {args_cli.output_dir / 'g1_replay_render_summary.json'}", flush=True)
        return

    rows = _read_rows(args_cli.replay_csv)
    _trace("rows_loaded", row_count=len(rows))
    capture_every = max(1, int(args_cli.capture_every_n_rows))
    max_frames = int(args_cli.max_frames)

    _trace("creating_stage")
    create_new_stage()
    stage = get_current_stage()
    UsdGeom.Scope.Define(stage, "/World")
    dome = UsdLux.DomeLight.Define(stage, "/World/ReplayDomeLight")
    dome.CreateIntensityAttr(600.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/ReplayKeyLight")
    distant.CreateIntensityAttr(1400.0)
    UsdGeom.XformCommonAPI(distant).SetRotate(Gf.Vec3f(-45.0, 0.0, 35.0))
    _spawn_cube(stage, "/World/Ground", (5.0, 4.0, 0.05), (0.31, 0.33, 0.33))
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Ground")).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.025))
    _spawn_cube(stage, BOX_PATH, tuple(float(v) for v in args_cli.box_size), (0.58, 0.43, 0.24))
    trail_summary = _spawn_replay_trails(stage, rows, capture_every, max_frames)
    first_eye, first_target = _camera_target_from_row(rows[0] if rows else {})
    _set_camera_look_at(stage, args_cli.camera_prim, first_eye, first_target)

    g1_prim = stage.DefinePrim(G1_PATH, "Xform")
    g1_prim.GetReferences().AddReference(str(args_cli.g1_usd))

    if not bool(args_cli.articulation_wrapper):
        if not use_replicator:
            raise RuntimeError("USD Xform-only replay currently requires Replicator capture.")
        _trace("xform_only_replay_start")
        rep_camera = rep.functional.create.camera(
            position=tuple(float(v) for v in first_eye),
            look_at=tuple(float(v) for v in first_target),
            parent="/World",
            name="G1ReplayRepCamera",
        )
        rep_camera_path = _camera_path(rep_camera)
        _trace("replicator_camera_created", camera_path=rep_camera_path)
        render_product = rep.create.render_product(
            rep_camera,
            resolution=(int(args_cli.resolution[0]), int(args_cli.resolution[1])),
            name="G1ReplayRenderProduct",
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
        capture_backend_used = "replicator_xform_only"
        for _ in range(8):
            _replicator_step()
            simulation_app.update()
        _trace("xform_only_warmup_complete")

        captured = 0
        for row_idx, row in enumerate(rows):
            if row_idx % capture_every != 0:
                continue
            if max_frames >= 0 and captured >= max_frames:
                break
            _set_robot_xform_pose(stage, G1_PATH, row)
            _set_cube_pose(stage, BOX_PATH, row)
            if bool(args_cli.follow_frame):
                eye, target = _camera_target_from_row(row)
                if rep_camera_path is not None:
                    _set_camera_look_at(stage, rep_camera_path, eye, target)
            _replicator_step()
            simulation_app.update()
            frame_path = frame_dir / f"g1_replay_{captured:05d}.png"
            _write_rgb_png(rgb_annotator.get_data(), frame_path)
            captured += 1
            _trace("frame_captured", captured=captured, frame_path=str(frame_path))

        summary = {
            "replay_csv": str(args_cli.replay_csv),
            "frame_dir": str(frame_dir),
            "captured_frames": captured,
            "status": "pass" if captured > 0 else "fail",
            "success_claim": "visual_replay_only_not_new_control_evidence",
            "capture_backend_requested": requested_backend,
            "capture_backend_used": capture_backend_used,
            "replicator_error": REPLICATOR_IMPORT_ERROR,
            "rendering_manager_error": RENDERING_MANAGER_IMPORT_ERROR,
            "world_import_error": WORLD_IMPORT_ERROR,
            "articulation_wrapper": False,
            "visual_markers": trail_summary,
        }
        _write_json(args_cli.output_dir / "g1_replay_render_summary.json", summary)
        print(f"[INFO] Replay render summary written to: {args_cli.output_dir / 'g1_replay_render_summary.json'}", flush=True)
        return

    if World is None or SingleArticulation is None:
        raise RuntimeError(f"Core World/SingleArticulation unavailable: {WORLD_IMPORT_ERROR}")
    _trace("creating_world")
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    robot = SingleArticulation(prim_path=G1_ARTICULATION_PATH, name="g1_replay")
    _trace("world_reset_start")
    world.reset()
    _trace("robot_initialize_start")
    robot.initialize()
    joint_names = list(getattr(robot, "dof_names", []))
    print(f"[PROGRESS] G1 initialized with {len(joint_names)} joints", flush=True)
    _trace("robot_initialized", joint_count=len(joint_names))
    if use_replicator:
        _trace("replicator_render_product_start")
        render_product = rep.create.render_product(
            args_cli.camera_prim,
            resolution=(int(args_cli.resolution[0]), int(args_cli.resolution[1])),
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
        capture_backend_used = "replicator"
        _trace("replicator_render_product_ready")
    else:
        _trace("viewport_create_start")
        ViewportManager.create_viewport_window(
            title="G1 Replay Showcase",
            resolution=(int(args_cli.resolution[0]), int(args_cli.resolution[1])),
            camera=args_cli.camera_prim,
        )
        ViewportManager.set_camera(args_cli.camera_prim)
        ViewportManager.set_resolution((int(args_cli.resolution[0]), int(args_cli.resolution[1])))
        capture_backend_used = "app-screenshot"
        _trace("viewport_ready")
    for _ in range(8):
        world.step(render=True)
        world.render()
        simulation_app.update()
    _trace("warmup_complete")

    captured = 0
    for row_idx, row in enumerate(rows):
        if row_idx % capture_every != 0:
            continue
        if max_frames >= 0 and captured >= max_frames:
            break
        robot.set_world_pose(
            position=np.array([float(row["robot_x"]), float(row["robot_y"]), float(row["robot_z"])], dtype=float),
            orientation=np.array([float(row["robot_qw"]), float(row["robot_qx"]), float(row["robot_qy"]), float(row["robot_qz"])], dtype=float),
        )
        row_joint_names = json.loads(row["joint_names_json"])
        row_joint_positions = json.loads(row["joint_positions_json"])
        if row_joint_names and row_joint_positions:
            ordered = np.array(robot.get_joint_positions(), dtype=float)
            by_name = {name: float(value) for name, value in zip(row_joint_names, row_joint_positions, strict=False)}
            for idx, name in enumerate(joint_names):
                if name in by_name:
                    ordered[idx] = by_name[name]
            robot.set_joint_positions(ordered.tolist())
        _set_cube_pose(stage, BOX_PATH, row)
        if bool(args_cli.follow_frame):
            eye, target = _camera_target_from_row(row)
            _set_camera_look_at(stage, args_cli.camera_prim, eye, target)
        world.step(render=True)
        world.render()
        simulation_app.update()
        frame_path = frame_dir / f"g1_replay_{captured:05d}.png"
        if use_replicator:
            _write_rgb_png(rgb_annotator.get_data(), frame_path)
        else:
            if not _capture_app_screenshot_png(frame_path):
                raise RuntimeError(f"App screenshot capture did not create {frame_path}")
        for _ in range(2):
            world.step(render=True)
            world.render()
            simulation_app.update()
        captured += 1
        _trace("frame_captured", captured=captured, frame_path=str(frame_path))
        if captured % 10 == 0:
            print(f"[PROGRESS] captured {captured} frames", flush=True)

    summary = {
        "replay_csv": str(args_cli.replay_csv),
        "frame_dir": str(frame_dir),
        "captured_frames": captured,
        "status": "pass" if captured > 0 else "fail",
        "success_claim": "visual_replay_only_not_new_control_evidence",
        "capture_backend_requested": requested_backend,
        "capture_backend_used": capture_backend_used,
        "replicator_error": REPLICATOR_IMPORT_ERROR,
        "rendering_manager_error": RENDERING_MANAGER_IMPORT_ERROR,
        "world_import_error": WORLD_IMPORT_ERROR,
        "articulation_wrapper": True,
        "visual_markers": trail_summary,
    }
    _write_json(args_cli.output_dir / "g1_replay_render_summary.json", summary)
    print(f"[INFO] Replay render summary written to: {args_cli.output_dir / 'g1_replay_render_summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write_failure_summary(exc, DEBUG_EVENTS[-1]["event"] if DEBUG_EVENTS else "before_main")
        raise
    finally:
        simulation_app.close()
