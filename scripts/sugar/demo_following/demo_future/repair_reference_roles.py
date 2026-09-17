"""Versioned reference-label repair on original PhysX data; no policy changes.

Validate the archive's body mapping against the official URDF joint origins,
then regenerate reference foot roles and their affected target channels. The
old data and frozen predictor's original normalization contract stay intact.
"""
from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET

import numpy as np

from .data import ROOT, OUTPUT, PARENT, CORPUS, TASKS, actual_entries, actual_future, load_reference, window_features, reverse_reference, mismatch, event_remaining, write_json

REPAIRED = OUTPUT.parent / "dataset_reference_roles_v2"


def source_names():
    source = ROOT / "scripts/sugar/demo_following/analyze_behavior_adherence.py"
    names = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REFERENCE_BODY_NAMES" for t in node.targets))
    return names


def validate_archive_geometry(names):
    urdf = ROOT / "SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
    edges = []
    for joint in ET.parse(urdf).getroot().findall("joint"):
        parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
        if parent not in names or child not in names:
            continue
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        edges.append((names.index(parent), names.index(child), xyz))
    records = []
    for task in TASKS:
        for directory in sorted((ROOT / "SUGAR/data" / task).glob("data_*")):
            path = directory / "robot_50hz.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                position, quaternion = z["body_pos_w"], z["body_quat_w"]
            frame_errors = np.zeros(len(position), dtype=np.float64)
            for parent, child, xyz in edges:
                q = quaternion[:, parent]
                q = q / np.linalg.norm(q, axis=-1, keepdims=True)
                rotated = xyz + 2 * np.cross(q[:, 1:], np.cross(q[:, 1:], xyz) + q[:, :1] * xyz)
                frame_errors = np.maximum(frame_errors, np.linalg.norm(position[:, parent] + rotated - position[:, child], axis=-1))
            records.append({"task": task, "source": int(directory.name.split("_" )[-1]), "frames": len(position),
                            "max_joint_origin_error_m": float(frame_errors.max()),
                            "frame0_joint_origin_error_m": float(frame_errors[0]),
                            "every_tenth_frame_joint_origin_error_m": float(frame_errors[::10].max())})
    # Body identity is checked independently of temporal interpolation rigidity.
    # Preserve the failed all-frame rigid-body hypothesis in the output.
    passed = len(records) == 199 and all(r["frame0_joint_origin_error_m"] < 1e-4 for r in records)
    if not passed:
        raise RuntimeError(f"Source body mapping failed official URDF geometry check: {records}")
    return {"passed": passed, "official_urdf": str(urdf.relative_to(ROOT)), "edges_per_frame": len(edges),
            "identity_check": "All named URDF parent-child joint origins at frame0 of every source motion match within 0.1mm; same mapping checked against every named actual trace",
            "all_frame_rigidity_passed": all(r["max_joint_origin_error_m"] < 1e-4 for r in records),
            "scope": "Every frame audited; nonzero intermediate-frame geometric residuals are retained, not erased or replaced by kinematics. A failed all-frame rigidity assumption must not be confused with body identity.", "records": records}


def corrected_reference(task, source, names):
    reference = load_reference(task, source, ROOT / "SUGAR/data")
    if task == "KickBox":
        body = reference["body_relative_object"]
        feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
        role = np.linalg.norm(body[:, feet], axis=-1).argmin(axis=-1)
        active = reference["contact"].any(axis=-1)
        reference["contact"][:] = False
        reference["contact"][np.flatnonzero(active), 2 + role[active]] = True
        reference["duration"] = event_remaining(reference["contact"])
    return reference


def main():
    names = source_names()
    geometry = validate_archive_geometry(names)
    REPAIRED.mkdir(exist_ok=False)
    entries = {(TASKS.index(e["task"]), e["source_id"]): e for e in actual_entries(CORPUS)}
    summaries = {}
    for split in ("train", "validation", "test"):
        directory = REPAIRED / split
        directory.mkdir()
        with np.load(OUTPUT / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        # The route retains original demo-task/source arrays for the first bank half.
        old_bank = np.load(OUTPUT / split / "demo_bank.npy")
        half = len(old_bank) // 2
        keys = list(zip(route["demo_task"][:half].tolist(), route["demo_source_motion_id"][:half].tolist()))
        references = [corrected_reference(TASKS[int(t)], int(s), names) for t, s in keys]
        bank = np.stack([window_features(r) for r in references] + [window_features(reverse_reference(r)) for r in references])
        unchanged_channels = list(range(122)) + [124, 125] + list(range(128, 132))
        if not np.array_equal(bank[..., unchanged_channels], old_bank[..., unchanged_channels]):
            raise RuntimeError("Repair changed a non-foot reference channel")
        old_target = np.load(OUTPUT / split / "target_mismatch.npy")
        targets = np.empty_like(old_target)
        cache, trace = None, None
        for row, (task, source, anchor) in enumerate(zip(route["base_task"], route["base_source_motion_id"], route["base_anchor_frame"])):
            entry = entries[(int(task), int(source))]
            if entry["trace"] != cache:
                cache = entry["trace"]
                with np.load(cache, allow_pickle=False) as z:
                    if tuple(z["robot_body_names"].tolist()) != names:
                        raise RuntimeError("Actual corpus body order differs from source mapping")
                    trace = {k: z[k] for k in ("object_root_state_w", "robot_body_position_w", "contact", "contact_event_remaining_frames", "motion_regime")}
            for horizon, offset in enumerate((0, 10, 30, 60)):
                actual = actual_future(trace, entry["env"], int(anchor) + offset)
                alignment = int(np.rint(route["horizon_phase"][row, horizon] * np.float32(31)))
                targets[row, :, horizon] = mismatch(actual, bank[route["selected_demo"][row], alignment])
        unaffected = [0, 1, 2, 3, 4, 5, 8, 9, 12]
        if not np.array_equal(targets[..., unaffected], old_target[..., unaffected]):
            raise RuntimeError("Repair changed a non-foot mismatch channel")
        np.save(directory / "demo_bank.npy", bank, allow_pickle=False)
        np.save(directory / "target_mismatch.npy", targets, allow_pickle=False)
        np.savez(directory / "routing.npz", **route)
        delta = np.abs(targets - old_target)
        summaries[split] = {"target_shape": list(targets.shape), "affected_base_histories": int(np.any(delta > 0, axis=(1, 2, 3)).sum()),
                            "per_channel_max_change": delta.max(axis=(0, 1, 2)).tolist(),
                            "non_foot_channels_exactly_preserved": True,
                            "policy_prefix_source": str((PARENT / split / "policy_prefix.npy").relative_to(ROOT)),
                            "policy_prefix_rows": "routing.npz:parent_base_row"}
    result = {"execution_completed": True, "optimizer_updates": 0, "source_geometry": geometry, "reference_body_names": names,
              "repaired_reference_foot_indices": [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")],
              "splits": summaries, "old_dataset": str(OUTPUT.relative_to(ROOT)),
              "semantics": "Reference binary contact is assigned to the closest named ankle; this remains a reference-event heuristic, not actual contact force. Actual PhysX contacts, executed actions, routing, clocks and all non-foot target channels are unchanged.",
              "compatibility": "New reference bank and foot contact/duration labels require new TRAIN-only normalization and matched predictor retraining. Do not attach to an existing frozen reward checkpoint or relabel old calibration as valid for these targets.",
              "next_action": "Complete an observable-coordinate target contract and current-state baseline on corrected labels before the next bounded matched predictor training."}
    write_json(REPAIRED / "MANIFEST.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_geometry", "reference_body_names")}), flush=True)


if __name__ == "__main__":
    main()
