"""Recover current geometry with official MimicKit FK from causal121-D input.

Diagnostic only: no reconstructed motion replaces PhysX records or executed
actions. Actual poses are read solely on the comparison side of this audit.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import numpy as np
import torch

from .data import ROOT, OUTPUT, CORPUS, write_json
from .canonical_data import canonicalize
from .repair_reference_roles import source_names
from build_actual_contact_event_predictor_dataset import quaternion_wxyz_to_rotation6d

sys.path.insert(0, str(ROOT / "scripts/sugar/smp"))
from sugar_g1_box_schema import G1_JOINT_NAMES
from anim.urdf_char_model import URDFCharModel
from util import torch_util


class ObservableGeometryAdapter:
    def __init__(self, device):
        self.device = device
        self.character = URDFCharModel(device)
        self.character.load(str(ROOT / "SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"))
        self.names = source_names()
        body_names = self.character.get_body_names()
        self.body_ids = [body_names.index(name) for name in self.names]
        self.torso = body_names.index("torso_link")
        tree = ast.parse((ROOT / "SUGAR/source/sugar_rl/sugar_rl/assets/robots/unitree.py").read_text())
        cfg = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "UNITREE_G1_29DOF_MIMIC_CFG" for t in node.targets))
        init = next(k.value for k in cfg.keywords if k.arg == "init_state")
        mapping = ast.literal_eval(next(k.value for k in init.keywords if k.arg == "joint_pos"))
        values = []
        for name in G1_JOINT_NAMES:
            matches = [float(v) for expression, v in mapping.items() if re.fullmatch(expression, name)]
            if len(matches) > 1:
                raise RuntimeError("Ambiguous official default joint position")
            values.append(matches[0] if matches else 0.)
        self.default = torch.tensor(values, device=device)
        self.dof_route = []
        for j in range(1, self.character.get_num_joints()):
            joint = self.character.get_joint(j)
            if joint.get_dof_dim():
                if joint.get_dof_dim() != 1:
                    raise RuntimeError("Unexpected G1 non-hinge DOF")
                self.dof_route.append((joint.dof_idx, G1_JOINT_NAMES.index(joint.name)))
        if len(self.dof_route) != 29:
            raise RuntimeError("Incomplete official G1 joint mapping")

    def __call__(self, observation):
        if observation.shape[-1] != 121:
            raise ValueError("Expected unchanged121-D observation")
        gravity = observation[:, :3]
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch = torch.asin(gravity[:, 0].clamp(-1, 1))
        root_q = torch_util.euler_xyz_to_quat(roll, pitch, torch.zeros_like(roll))
        joints = observation[:, 10:39] + self.default
        dof = torch.zeros((len(observation), 29), device=self.device)
        for destination, source in self.dof_route:
            dof[:, destination] = joints[:, source]
        joint_rot = self.character.dof_to_rot(dof)
        position, quaternion = self.character.forward_kinematics(torch.zeros((len(observation), 3), device=self.device), root_q, joint_rot)
        torso_q = quaternion[:, self.torso]
        box_position = position[:, self.torso] + torch_util.quat_rotate(torso_q, observation[:, 97:100])
        body_relative_box = position[:, self.body_ids] - box_position[:, None]
        x, z = observation[:, 100:103], observation[:, 103:106]
        y = torch.cross(z, x, dim=-1)
        rotation6d = torch.cat((torch_util.quat_rotate(torso_q, x), torch_util.quat_rotate(torso_q, y)), dim=-1)
        velocities = torch.cat((torch_util.quat_rotate(torso_q, observation[:, 106:109]), torch_util.quat_rotate(torso_q, observation[:, 109:112])), dim=-1)
        return torch.cat((body_relative_box.flatten(1), torch.zeros((len(observation), 3), device=self.device), rotation6d, velocities), dim=-1)


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    device = torch.device("cuda")
    adapter = ObservableGeometryAdapter(device)
    records = []
    for path in sorted(CORPUS.glob("*/TRACE.npz")):
        with np.load(path, allow_pickle=False) as z:
            observation = z["goal_policy_core_observation"].reshape(-1, 121)
            obj = z["object_root_state_w"].reshape(-1, 13)
            body = z["robot_body_position_w"].reshape(-1, 35, 3)
            root = z["robot_root_state_w"].reshape(-1, 13)
            if tuple(z["robot_body_names"].tolist()) != adapter.names:
                raise RuntimeError("Unexpected actual body order")
        predicted = np.concatenate([adapter(torch.as_tensor(observation[start:start + 2048], device=device)).cpu().numpy()
                                    for start in range(0, len(observation), 2048)])
        raw = np.concatenate(((body - obj[:, None, :3]).reshape(-1, 105), np.zeros((len(obj), 3), dtype=np.float32),
                              quaternion_wxyz_to_rotation6d(obj[:, 3:7]), obj[:, 7:13]), axis=-1)
        expected = canonicalize(raw[:, None], root[:, 3:7])[:, 0]
        errors = np.abs(predicted - expected)
        groups = {name: {"max_absolute_error": float(errors[:, a:b].max()), "mean_absolute_error": float(errors[:, a:b].mean())}
                  for name, a, b in (("body_relative_box", 0, 105), ("box_rotation6d", 108, 114), ("box_velocity", 114, 120))}
        records.append({"trace": str(path.relative_to(ROOT)), "actual_frames": len(obj), "groups": groups})
    passed = all(value["max_absolute_error"] < 1e-3 for record in records for value in record["groups"].values())
    result = {"execution_completed": True, "optimizer_updates": 0, "passed_1e_3_geometry_check": passed,
              "actual_frames": sum(r["actual_frames"] for r in records), "records": records,
              "implementation": "Official MimicKit URDFCharModel.dof_to_rot/forward_kinematics and official G1 URDF/default joint positions; only observation/joint/body-name glue added.",
              "input": "Only current121-D goal-policy observation and fixed official robot constants; yaw fixed to zero, tilt from measured projected gravity. No actual body poses/quaternions/contact forces supplied to the adapter.",
              "scope": "Current geometry recoverability diagnostic, not a learned world model, new physics rollout, generated training corpus or executed-action reconstruction. Original PhysX states/actions remain untouched. Physical contact force/state and reset-relative regime are not recovered by this adapter.",
              "next_action": "Use recoverable geometry as an explicit deterministic input adapter in a matched serious predictor study if the geometry check passes; otherwise diagnose frame/joint observation alignment before training. Keep contact/regime inference separate and never fabricate live tactile signals."}
    write_json(OUTPUT.parent / "OBSERVABLE_GEOMETRY_PROBE.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
