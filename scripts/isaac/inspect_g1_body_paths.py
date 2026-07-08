#!/usr/bin/env python3
"""Inspect candidate G1 body prim paths on a compute node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect G1 USD prim paths.")
    parser.add_argument("--g1-usd", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd"))
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


if os.uname().nodename.startswith("mgmtserver"):
    raise RuntimeError("Refusing to inspect Isaac USD through Isaac on a login/management node.")

args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


def main() -> int:
    create_new_stage()
    stage = get_current_stage()
    root = stage.DefinePrim("/World/G1", "Xform")
    root.GetReferences().AddReference(str(args_cli.g1_usd))
    stage.Flatten()
    keywords = (
        "torso",
        "pelvis",
        "shoulder",
        "elbow",
        "wrist",
        "hand",
        "palm",
        "forearm",
        "link",
    )
    paths = []
    body_like = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        lowered = path.lower()
        if any(keyword in lowered for keyword in keywords):
            paths.append({"path": path, "type": prim.GetTypeName(), "name": name})
        has_rigid_body = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if prim.GetTypeName() in {"Xform", "Mesh"} and any(
            keyword in lowered for keyword in ("wrist", "hand", "palm", "elbow", "torso")
        ):
            body_like.append({"path": path, "type": prim.GetTypeName(), "name": name, "has_rigid_body_api": has_rigid_body})
    report = {
        "g1_usd": str(args_cli.g1_usd),
        "candidate_paths": paths,
        "candidate_body_like_paths": body_like,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
