# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Probe: verify hydroelastic contact-surface + SensorContact data flow on panda_hydro.
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact
from newton.examples.robot.example_robot_panda_hydro import Example

# Force the contact surface to be emitted regardless of viewer.
_orig_cfg_init = HydroelasticSDF.Config.__init__
def _cfg_init(self, *a, **k):
    _orig_cfg_init(self, *a, **k)
    self.output_contact_surface = True
HydroelasticSDF.Config.__init__ = _cfg_init

# Ensure the per-contact 'force' attribute is allocated in the example's own
# Contacts buffer (SensorContact reads contacts.force).
_orig_pipe_contacts = newton.CollisionPipeline.contacts
def _pipe_contacts(self, *a, **k):
    self.model.request_contact_attributes("force")
    return _orig_pipe_contacts(self, *a, **k)
newton.CollisionPipeline.contacts = _pipe_contacts

parser = Example.create_parser()
viewer, args = newton.examples.init(parser)
ex = Example(viewer, args)

m = ex.model
labels = m.body_key if hasattr(m, "body_key") else getattr(m, "body_label", None)
print("has body labels:", labels is not None)
finger_bodies = {}
for i, lab in enumerate(labels):
    if any(k in lab for k in ("finger", "hand", "object")):
        print(f"  body {i}: {lab}")
shape_body = m.shape_body.numpy()
# identify left/right finger shape sets by body label
def bodies_matching(sub):
    return [i for i, lab in enumerate(labels) if sub in lab]
left_b = bodies_matching("leftfinger"); right_b = bodies_matching("rightfinger")
print("left finger bodies:", left_b, "right:", right_b, "| object body:", ex.object_body_local)
left_shapes = set(np.where(np.isin(shape_body, left_b))[0].tolist())
right_shapes = set(np.where(np.isin(shape_body, right_b))[0].tolist())
print("left shapes:", left_shapes, "right shapes:", right_shapes)

sensor = SensorContact(m, sensing_bodies=["*leftfinger*", "*rightfinger*"], counterpart_bodies="object")
contacts = ex.contacts

hsdf = ex.collision_pipeline.hydroelastic_sdf
print("hydroelastic_sdf present:", hsdf is not None, "| output_contact_surface:", hsdf.config.output_contact_surface)

for f in range(260):
    ex.step()
    ex.solver.update_contacts(contacts)
    sensor.update(ex.state_0, contacts)
    if f in (60, 120, 180, 240):
        cs = hsdf.get_contact_surface()
        nfaces = int(cs.face_contact_count.numpy()[0])
        tf = sensor.total_force.numpy(); tfr = sensor.total_force_friction.numpy()
        sp = cs.contact_surface_shape_pair.numpy()[:nfaces] if nfaces else np.zeros((0, 2))
        on_left = sum(1 for a, b in sp if a in left_shapes or b in left_shapes)
        on_right = sum(1 for a, b in sp if a in right_shapes or b in right_shapes)
        print(f"frame {f}: faces={nfaces} (L={on_left} R={on_right}) | "
              f"L normal={np.linalg.norm(tf[0]-tfr[0]):.2f}N shear={np.linalg.norm(tfr[0]):.2f}N | "
              f"R normal={np.linalg.norm(tf[1]-tfr[1]):.2f}N shear={np.linalg.norm(tfr[1]):.2f}N")
