"""Audit official SMP geometry and inherited reference role labels, no updates."""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch

from .data import ROOT, OUTPUT, CORPUS, write_json

sys.path.insert(0, str(ROOT / "scripts/sugar/smp"))
from run_selected_demo_tinymdm import load_clip, build_feature_windows, load_official_feature_functions
from sugar_g1_box_schema import SOURCE_BODY_NAMES


def main():
    device = torch.device("cuda")
    _, util = load_official_feature_functions()
    yaw = torch.tensor([[0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)]], device=device, dtype=torch.float32)
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
    translation = np.asarray([2., -3., 0.], dtype=np.float32)
    geometry = []
    for task, source in (("CarryBox", 45), ("KickBox", 21)):
        robot, obj = load_clip(ROOT / "SUGAR/data" / task / f"data_{source:03d}")
        for start in (0, len(robot["joint_pos"]) // 2, len(robot["joint_pos"]) - 10):
            r = {k: robot[k][start:start + 10].copy() for k in
                 ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
            o = {k: obj[k][start:start + 10].copy() for k in
                 ("obj_trans", "obj_rot", "obj_lin_vel", "obj_ang_vel")}
            expected = build_feature_windows(r, o, device)
            r["body_pos_w"] = r["body_pos_w"] @ rotation.T + translation
            q = torch.as_tensor(r["body_quat_w"][..., [1, 2, 3, 0]], device=device)
            r["body_quat_w"] = util.quat_mul(yaw.expand_as(q), q).cpu().numpy()[..., [3, 0, 1, 2]]
            for key in ("body_lin_vel_w", "body_ang_vel_w"):
                r[key] = r[key] @ rotation.T
            o["obj_trans"] = o["obj_trans"] @ rotation.T + translation
            o["obj_rot"] = rotation @ o["obj_rot"]
            for key in ("obj_lin_vel", "obj_ang_vel"):
                o[key] = o[key] @ rotation.T
            transformed = build_feature_windows(r, o, device)
            geometry.append({"task": task, "source": source, "start": start,
                             "shape": list(expected.shape),
                             "max_absolute_change": float(np.max(np.abs(expected - transformed)))})
    required = {"robot_body_position_w", "robot_body_quaternion_w", "robot_body_linear_velocity_w",
                "robot_body_angular_velocity_w", "robot_joint_position", "robot_joint_velocity",
                "object_root_state_w", "robot_body_names", "robot_joint_names"}
    traces = []
    for path in sorted(CORPUS.glob("*/TRACE.npz")):
        with np.load(path, allow_pickle=False) as z:
            names = [str(v) for v in z["robot_body_names"]]
            traces.append({"path": str(path.relative_to(ROOT)), "missing_official_adapter_fields": sorted(required - set(z.files)),
                           "old_reference_foot_indices_in_recorded_order": [names[6], names[12]],
                           "ankle_roll_indices_in_recorded_order": [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")],
                           "order_differences_from_existing_smp_source_schema": [{"index": i, "actual_name": a, "schema_name": b}
                                                                                   for i, (a, b) in enumerate(zip(names, SOURCE_BODY_NAMES)) if a != b]})
    motions = []
    for directory in sorted((ROOT / "SUGAR/data/KickBox").glob("data_*")):
        if not (directory / "robot_50hz.npz").exists():
            continue
        with np.load(directory / "robot_50hz.npz", allow_pickle=False) as z:
            body = z["body_pos_w"]
            archive_has_body_names = "body_names" in z.files
        with (directory / "obj_motion_global_50hz.pkl").open("rb") as stream:
            position = np.asarray(pickle.load(stream)["obj_trans"])
        binary = np.load(directory / "contact_labels_50hz.npy").astype(bool)
        n = min(len(body), len(position), len(binary))
        active = binary[:n]
        old_distance = np.linalg.norm(body[:n, [6, 12]] - position[:n, None], axis=-1)
        ankle_distance = np.linalg.norm(body[:n, [21, 22]] - position[:n, None], axis=-1)
        old_role = old_distance.argmin(axis=-1)
        ankle_role = ankle_distance.argmin(axis=-1)
        motions.append({"source": int(directory.name.split("_")[-1]), "archive_has_body_names": archive_has_body_names,
                        "frames": n, "active_reference_frames": int(active.sum()),
                        "changed_active_roles": int(((old_role != ankle_role) & active).sum()),
                        "old_indices_mean_height_m": body[:n, [6, 12], 2].mean(axis=0).tolist(),
                        "ankle_indices_mean_height_m": body[:n, [21, 22], 2].mean(axis=0).tolist()})
    total = sum(m["active_reference_frames"] for m in motions)
    changed = sum(m["changed_active_roles"] for m in motions)
    result = {"execution_completed": True, "optimizer_updates": 0,
              "official_smp_coordinate_checks": geometry,
              "official_heading_local_invariance_passed": all(r["max_absolute_change"] < 1e-4 for r in geometry),
              "original_corpus_feature_completeness": traces,
              "reference_role_audit": {"motions": motions, "motion_count": len(motions),
                                       "affected_motions": sum(m["changed_active_roles"] > 0 for m in motions),
                                       "active_reference_frames": total, "changed_active_roles": changed,
                                       "changed_active_fraction": changed / total},
              "scope": "Coordinate check executes official compute_disc_obs through existing G1/box adapter. Reference role check compares inherited 6/12 against ankle21/22 supported by both existing SMP schema and actual runtime names. Source NPZ has no names: validate archive correspondence before regenerating targets. Binary reference annotations remain supervision only, never actual contact forces.",
              "next_action": "Validate the source body order and correct inherited reference foot-role labels in a versioned dataset before further matched training. Retain old checkpoints/labels as historical; collect feature-complete actual states if official SMP scoring of this corpus is needed, without fabricating missing fields."}
    write_json(OUTPUT.parent / "FEATURE_CONTRACT_AUDIT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "reference_role_audit"}), flush=True)
    print(json.dumps({k: v for k, v in result["reference_role_audit"].items() if k != "motions"}), flush=True)


if __name__ == "__main__":
    main()
