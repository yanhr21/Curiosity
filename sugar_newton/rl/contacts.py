# SPDX-License-Identifier: BSD-3-Clause
"""Newton contact-force history matching SUGAR's CarryBox contact sensors.

The official Tracker reward reads three-frame contact histories.  Newton's MuJoCo-Warp
solver can return the force on every resolved rigid contact; this module reduces those
forces onto robot bodies and separately onto robot/box pairs.  It does not infer contact
from distance, pose, or a reward proxy.
"""

from __future__ import annotations

import torch
import warp as wp


@wp.kernel
def reduce_body_contact_forces(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    contact_force: wp.array[wp.spatial_vector],
    shape_body: wp.array[wp.int32],
    bodies_per_world: wp.int32,
    box_body_local: wp.int32,
    body_force: wp.array[wp.vec3],
    body_box_force: wp.array[wp.vec3],
):
    """Accumulate world-space force on each body and its filtered box pair."""
    i = wp.tid()
    if i >= contact_count[0]:
        return

    s0 = shape0[i]
    s1 = shape1[i]
    b0 = wp.int32(-1)
    b1 = wp.int32(-1)
    if s0 >= 0:
        b0 = shape_body[s0]
    if s1 >= 0:
        b1 = shape_body[s1]

    # Newton convention: the returned force is the force on shape0's body.
    force0 = wp.spatial_top(contact_force[i])
    if b0 >= 0:
        wp.atomic_add(body_force, b0, force0)
        if b1 >= 0 and b1 % bodies_per_world == box_body_local:
            wp.atomic_add(body_box_force, b0, force0)
    if b1 >= 0:
        force1 = -force0
        wp.atomic_add(body_force, b1, force1)
        if b0 >= 0 and b0 % bodies_per_world == box_body_local:
            wp.atomic_add(body_box_force, b1, force1)


class ContactHistory:
    """Three-frame net and robot/box contact history plus foot air-time state."""

    HISTORY = 3
    CONTACT_FORCE_THRESHOLD = 1.0

    def __init__(self, env):
        self.env = env
        total_bodies = env.model.body_count
        self._body_force = wp.zeros(total_bodies, dtype=wp.vec3, device=env.model.device)
        self._body_box_force = wp.zeros(total_bodies, dtype=wp.vec3, device=env.model.device)
        self.net = torch.zeros(
            env.num_envs, self.HISTORY, env.nbody, 3, device=env.device
        )
        self.box = torch.zeros_like(self.net)

        labels = env.body_labels
        self.foot_idx = torch.as_tensor(
            [labels.index("left_ankle_roll_link"), labels.index("right_ankle_roll_link")],
            dtype=torch.long,
            device=env.device,
        )
        self.hand_idx = torch.as_tensor(
            [labels.index("left_rubber_hand"), labels.index("right_rubber_hand")],
            dtype=torch.long,
            device=env.device,
        )
        excluded = {
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
            "box",
        }
        self.undesired_idx = torch.as_tensor(
            [i for i, name in enumerate(labels) if name not in excluded],
            dtype=torch.long,
            device=env.device,
        )
        self.foot_contact = torch.zeros(
            env.num_envs, 2, dtype=torch.bool, device=env.device
        )
        self.first_foot_contact = torch.zeros_like(self.foot_contact)
        self.current_air_time = torch.zeros(env.num_envs, 2, device=env.device)
        self.last_air_time = torch.zeros_like(self.current_air_time)

    def update(self) -> None:
        """Read the final solver contacts for this control step and advance history."""
        env = self.env
        self._body_force.zero_()
        self._body_box_force.zero_()
        wp.launch(
            reduce_body_contact_forces,
            dim=env.contacts.rigid_contact_max,
            inputs=[
                env.contacts.rigid_contact_count,
                env.contacts.rigid_contact_shape0,
                env.contacts.rigid_contact_shape1,
                env.contacts.force,
                env.model.shape_body,
                env.nbody,
                env.box_body,
            ],
            outputs=[self._body_force, self._body_box_force],
            device=env.model.device,
        )
        net = wp.to_torch(self._body_force).view(env.num_envs, env.nbody, 3)
        box = wp.to_torch(self._body_box_force).view(env.num_envs, env.nbody, 3)
        self.net = torch.cat((self.net[:, 1:], net.unsqueeze(1)), dim=1)
        self.box = torch.cat((self.box[:, 1:], box.unsqueeze(1)), dim=1)

        contact = net[:, self.foot_idx].norm(dim=-1) > self.CONTACT_FORCE_THRESHOLD
        self.first_foot_contact = contact & ~self.foot_contact
        landed_after = self.current_air_time + env.dt
        self.last_air_time = torch.where(
            self.first_foot_contact, landed_after, self.last_air_time
        )
        self.current_air_time = torch.where(
            contact, torch.zeros_like(self.current_air_time), landed_after
        )
        self.foot_contact = contact

    def reset(self, env_ids: torch.Tensor) -> None:
        self.net[env_ids] = 0.0
        self.box[env_ids] = 0.0
        self.foot_contact[env_ids] = False
        self.first_foot_contact[env_ids] = False
        self.current_air_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
