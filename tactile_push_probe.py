# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Probe: after frame 420, push the gripper straight down so the pen is driven into
# the cup bottom, and watch whether the two pad shear forces on the pen align.
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.sensors import SensorContact
from newton.examples.robot.example_robot_panda_hydro import Example, broadcast_ik_solution_kernel

_op = newton.CollisionPipeline.contacts
def _pc(self, *a, **k):
    self.model.request_contact_attributes("force")
    return _op(self, *a, **k)
newton.CollisionPipeline.contacts = _pc


def install_push(ex, base, rate):
    base_pos = base[:3].astype(float)
    rq = base[3:7].astype(float)
    st = {"i": 0}

    def push():
        i = st["i"]
        tp = wp.vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2] - rate * i))
        ex.pos_obj.set_target_positions(wp.array([tp], dtype=wp.vec3))
        ex.rot_obj.set_target_rotations(
            wp.array([wp.vec4(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]))], dtype=wp.vec4))
        if ex.graph_ik is not None:
            wp.capture_launch(ex.graph_ik)
        else:
            ex.ik_solver.step(ex.joint_q_ik, ex.joint_q_ik, iterations=ex.ik_iters)
        wp.launch(broadcast_ik_solution_kernel, dim=ex.world_count,
                  inputs=[ex.joint_q_ik, ex.joint_targets_2d, 0.0])  # gripper closed
        wp.copy(ex.control.joint_target_q, ex.joint_targets_2d.flatten())
        st["i"] += 1

    ex.set_joint_targets = push


parser = Example.create_parser()
viewer, args = newton.examples.init(parser)
ex = Example(viewer, args)
m = ex.model
labels = m.body_key if hasattr(m, "body_key") else m.body_label
lb = [i for i, l in enumerate(labels) if "leftfinger" in l][0]
rb = [i for i, l in enumerate(labels) if "rightfinger" in l][0]
obj = [i for i, l in enumerate(labels) if l.endswith("object")][0]
cup = [i for i, l in enumerate(labels) if l.endswith("cup")][0]
sensor = SensorContact(m, sensing_bodies="object", counterpart_bodies=["*leftfinger*", "*rightfinger*", "cup"])
ci = sensor.counterpart_indices[0]
cL, cR, cC = ci.index(lb), ci.index(rb), ci.index(cup)
contacts = ex.contacts
up = np.array([0, 0, 1.0])

PUSH_AFTER, PUSH_N, RATE = 420, 100, 0.001
for f in range(PUSH_AFTER + 1 + PUSH_N):
    if f == PUSH_AFTER + 1:
        install_push(ex, ex.state_0.body_q.numpy()[ex.ee_index].copy(), RATE)
    ex.step()
    if f <= PUSH_AFTER or f % 10:
        continue
    ex.solver.update_contacts(contacts)
    sensor.update(ex.state_0, contacts)
    fmf = sensor.force_matrix_friction.numpy()[0]
    fm = sensor.force_matrix.numpy()[0]
    sL, sR = fmf[cL], fmf[cR]
    nLn, nRn = np.linalg.norm(sL), np.linalg.norm(sR)
    cos = float(np.dot(sL, sR) / (nLn * nRn + 1e-12)) if nLn > 1e-6 and nRn > 1e-6 else float("nan")
    cup_normal_z = float((fm[cC] - fmf[cC])[2])  # cup's vertical force on pen
    penz = float(ex.state_0.body_q.numpy()[obj][2])
    print(f"push {f - PUSH_AFTER:3d} penZ={penz:.3f} | sL·up={float(np.dot(sL, up)):+.2f} "
          f"sR·up={float(np.dot(sR, up)):+.2f} cos(L,R)={cos:+.2f} | cup_Fz_on_pen={cup_normal_z:+.2f}")
