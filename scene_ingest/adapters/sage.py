# SPDX-License-Identifier: Apache-2.0
"""SAGE-10k → :class:`~scene_ingest.spec.SceneSpec` adapter.

SAGE-10k scene layout (``huggingface.co/datasets/nvidia/SAGE-10k``), per extracted scene::

    <scene>/layout_<id>.json      # rooms[] → walls/doors/objects with pose + mass + PBR
    <scene>/objects/<sid>.ply     # one mesh per object (+ <sid>_texture.png)
    <scene>/materials/            # wall/floor/door textures

The layout is Z-up, metres, floor at ``z = 0``. Object placement = ``position`` (m) +
``rotation`` (Euler°, yaw=z) + ``dimensions`` (target AABB); ``place_id`` gives the support
parent (``floor`` / ``wall`` / a parent object id). ``mass`` [kg] and ``pbr_parameters``
(``metallic``/``roughness``) are provided.

.. note::
   ``mesh_up_axis`` and per-mesh pre-pose are calibration knobs — verify against the live
   ``.ply`` meshes in the ``newton`` env (SAGE's GLB export is Y-up; the raw PLY frame must be
   confirmed). See ``claude_context/dataset_ingestion.md`` §4.
"""

from __future__ import annotations

import json
import os

from ..spec import (
    DoorSpec,
    MaterialSpec,
    ObjectSpec,
    RoomSpec,
    SceneSpec,
    WallSpec,
)

# Structural furniture kept fixed to the world so it doesn't drift during settling.
DEFAULT_STATIC_TYPES = frozenset(
    {"table", "bookshelf", "sideboard", "cabinet", "desk", "wardrobe", "shelf", "counter", "sofa", "bed"}
)


def _material_from_pbr(pbr: dict | None) -> MaterialSpec:
    """Heuristic PBR → contact material. All rigid (share ``rigid_kh``); differ by mu/restitution.

    Rougher → higher friction; metallic → lower friction, slight restitution. Tune in-sim.
    """
    pbr = pbr or {}
    metallic = float(pbr.get("metallic", 0.0))
    roughness = float(pbr.get("roughness", 0.7))
    if metallic > 0.5:
        return MaterialSpec(mu=0.4, restitution=0.1)
    return MaterialSpec(mu=0.5 + 0.5 * roughness, restitution=0.0)


def load_sage_scene(
    layout_json_path: str,
    robot=None,
    static_types: frozenset[str] = DEFAULT_STATIC_TYPES,
    mesh_up_axis: str = "z",
    include_missing_meshes: bool = False,
) -> SceneSpec:
    """Parse one extracted SAGE-10k scene into a :class:`SceneSpec`.

    Args:
        layout_json_path: Path to ``layout_<id>.json`` (its dir must hold ``objects/``).
        robot: A :class:`RobotSpec` to attach, or None to leave the scene robot-less.
        static_types: Semantic types fixed to the world (structural furniture).
        mesh_up_axis: Source up-axis of the ``.ply`` meshes ("z" or "y") — calibration.
        include_missing_meshes: Keep objects whose ``.ply`` is absent (default: drop).

    Returns:
        A populated ``SceneSpec`` (multiple rooms are merged; object positions are offset by
        their room ``position``).
    """
    root_dir = os.path.dirname(os.path.abspath(layout_json_path))
    with open(layout_json_path) as f:
        layout = json.load(f)

    scene_id = layout.get("id", os.path.basename(layout_json_path))
    rooms = layout.get("rooms", [])
    # Merge all rooms into one SceneSpec; use the first room's meta for RoomSpec.
    r0 = rooms[0] if rooms else {}
    dims = r0.get("dimensions", {})
    room = RoomSpec(
        room_type=r0.get("room_type", "room"),
        bounds=(float(dims.get("width", 0.0)), float(dims.get("length", 0.0)), float(dims.get("height", 2.7))),
    )

    objects: list[ObjectSpec] = []
    for room_dict in rooms:
        rp = room_dict.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        ox, oy, oz = float(rp.get("x", 0)), float(rp.get("y", 0)), float(rp.get("z", 0))
        # walls / doors (from every room)
        for w in room_dict.get("walls", []):
            a, b = w["start_point"], w["end_point"]
            room.walls.append(
                WallSpec(
                    wall_id=w.get("id", ""),
                    start=(float(a["x"]) + ox, float(a["y"]) + oy),
                    end=(float(b["x"]) + ox, float(b["y"]) + oy),
                    height=float(w.get("height", room.bounds[2])),
                    thickness=float(w.get("thickness", 0.1)),
                )
            )
        for do in room_dict.get("doors", []):
            room.doors.append(
                DoorSpec(
                    door_id=do.get("id", ""),
                    wall_id=do.get("wall_id", ""),
                    position_on_wall=float(do.get("position_on_wall", 0.5)),
                    width=float(do.get("width", 0.9)),
                    height=float(do.get("height", 2.0)),
                    opens_inward=bool(do.get("opens_inward", True)),
                )
            )
        # objects
        for o in room_dict.get("objects", []):
            sid = o.get("source_id")
            mesh_path = os.path.join(root_dir, "objects", f"{sid}.ply") if sid else ""
            if mesh_path and not os.path.exists(mesh_path) and not include_missing_meshes:
                continue
            pos = o.get("position", {})
            rot = o.get("rotation", {})
            d = o.get("dimensions", {})
            otype = o.get("type", "object")
            objects.append(
                ObjectSpec(
                    object_id=o.get("id", sid or ""),
                    mesh_path=mesh_path,
                    position=(float(pos.get("x", 0)) + ox, float(pos.get("y", 0)) + oy, float(pos.get("z", 0)) + oz),
                    rotation_euler_deg=(float(rot.get("x", 0)), float(rot.get("y", 0)), float(rot.get("z", 0))),
                    bbox_dims=(float(d.get("width", 0)), float(d.get("length", 0)), float(d.get("height", 0)))
                    if d
                    else None,
                    mass=float(o["mass"]) if o.get("mass") is not None else None,
                    material=_material_from_pbr(o.get("pbr_parameters")),
                    semantic_type=otype,
                    support_parent=o.get("place_id", "floor"),
                    is_static=otype in static_types,
                    up_axis=mesh_up_axis,
                )
            )

    return SceneSpec(
        scene_id=scene_id,
        source_dataset="sage-10k",
        root_dir=root_dir,
        room=room,
        objects=objects,
        robot=robot,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse a SAGE-10k scene into a SceneSpec and print a summary.")
    ap.add_argument("layout_json", help="Path to an extracted scene's layout_<id>.json")
    args = ap.parse_args()
    spec = load_sage_scene(args.layout_json, robot=None)
    print(spec.summary())
    print(f"static objects: {sum(o.is_static for o in spec.objects)}")
    from collections import Counter

    print("support parents:", dict(Counter(o.support_parent.split('_')[-1] if o.support_parent not in ('floor', 'wall') else o.support_parent for o in spec.objects)))
    print("types:", dict(Counter(o.semantic_type for o in spec.objects)))
