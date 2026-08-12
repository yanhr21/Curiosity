#!/usr/bin/env python3

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTactile
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter


class TestNewtonTactileAdapter(unittest.TestCase):
    def test_native_sensor_to_common_frame(self):
        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="pad")
        builder.add_shape_box(body_a, hx=0.02, hy=0.015, hz=0.005, label="pad_shape")
        body_b = builder.add_body(label="object")
        builder.add_shape_box(body_b, hx=0.02, hy=0.015, hz=0.005, label="object_shape")
        model = builder.finalize(device="cpu")
        sensor = SensorTactile(model, sensing_shapes=[0], grid_shape=(4, 5), patch_size=(0.04, 0.03))

        contacts = newton.Contacts(
            2,
            0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )
        contacts.rigid_contact_count.assign([1])
        contacts.rigid_contact_shape0.assign([0, -1])
        contacts.rigid_contact_shape1.assign([1, -1])
        contacts.rigid_contact_point0.assign([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([(0.0, 0.0, -0.001), (0.0, 0.0, 0.0)])
        contacts.rigid_contact_offset0.zero_()
        contacts.rigid_contact_offset1.zero_()
        contacts.rigid_contact_normal.assign([(0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])
        contacts.rigid_contact_margin0.zero_()
        contacts.rigid_contact_margin1.zero_()
        contacts.force.assign([(1.0, -2.0, 3.0, 0.0, 0.0, 0.0), (0.0,) * 6])

        sensor.update(model.state(), contacts, timestamp=0.1)
        frame = NewtonTactileAdapter(sensor, ["pad"]).frame()

        self.assertEqual(frame.backend, "newton_native_contacts")
        self.assertEqual(frame.normal_force_n.shape, (1, 1, 4, 5))
        self.assertEqual(frame.shear_force_xy_n.shape, (1, 1, 4, 5, 2))
        np.testing.assert_allclose(frame.normal_force_n.sum(), 3.0, atol=1.0e-6)
        np.testing.assert_allclose(frame.shear_force_xy_n.sum(axis=(0, 1, 2, 3)), [1.0, -2.0], atol=1.0e-6)
        self.assertEqual(frame.raw_samples.contact_index.tolist(), [0])
        self.assertEqual(frame.raw_samples.contact_kind.tolist(), [0])
        self.assertEqual(frame.raw_samples.counterpart_particle.tolist(), [-1])
        self.assertEqual(frame.optical.available, (False,))
        self.assertEqual(frame.clock.sequence, 0)
        np.testing.assert_allclose(
            frame.taxel_orientation_w_xyzw[..., :3], 0.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            frame.taxel_orientation_w_xyzw[..., 3], 1.0, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
