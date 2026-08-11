# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTactile


def _make_model(*, xform_a=None, xform_b=None, device="cpu"):
    device = wp.get_device(device)
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=xform_a or wp.transform_identity(), label="body_a")
    builder.add_shape_box(body_a, hx=0.02, hy=0.015, hz=0.005, label="patch_a")
    body_b = builder.add_body(xform=xform_b or wp.transform_identity(), label="body_b")
    builder.add_shape_box(body_b, hx=0.02, hy=0.015, hz=0.005, label="object_b")
    return builder.finalize(device=device)


def _make_contacts(
    model,
    *,
    pair=(0, 1),
    point0=(0.0, 0.0, 0.0),
    point1=(0.0, 0.0, -0.002),
    normal=(0.0, 0.0, 1.0),
    force=(1.0, 2.0, 3.0),
):
    capacity = 4
    contacts = newton.Contacts(
        capacity,
        0,
        device=model.device,
        requested_attributes=model.get_requested_contact_attributes(),
    )
    contacts.rigid_contact_count.assign([1])
    contacts.rigid_contact_shape0.assign([pair[0], -1, -1, -1])
    contacts.rigid_contact_shape1.assign([pair[1], -1, -1, -1])
    contacts.rigid_contact_point0.assign([point0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    contacts.rigid_contact_point1.assign([point1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    contacts.rigid_contact_offset0.zero_()
    contacts.rigid_contact_offset1.zero_()
    contacts.rigid_contact_normal.assign([normal, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    contacts.rigid_contact_margin0.zero_()
    contacts.rigid_contact_margin1.zero_()
    contacts.force.assign([(*force, 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 6, (0.0,) * 6])
    return contacts


class TestSensorTactile(unittest.TestCase):
    def test_world_fixed_sensing_shape(self):
        builder = newton.ModelBuilder()
        builder.add_shape_box(body=-1, hx=0.02, hy=0.015, hz=0.005, label="fixed_patch")
        body = builder.add_body(label="object")
        builder.add_shape_box(body, hx=0.02, hy=0.015, hz=0.005, label="object_shape")
        model = builder.finalize(device="cpu")
        sensor = SensorTactile(
            model,
            sensing_shapes=[0],
            grid_shape=(4, 5),
            patch_size=(0.04, 0.03),
        )
        contacts = _make_contacts(
            model,
            point0=(0.005, -0.004, 0.0),
            point1=(0.005, -0.004, -0.002),
            force=(1.0, 2.0, 3.0),
        )
        sensor.update(model.state(), contacts, timestamp=0.0)
        np.testing.assert_allclose(
            sensor.force.numpy()[0].sum(axis=0), [1.0, 2.0, 3.0], atol=1.0e-6
        )
        np.testing.assert_allclose(
            sensor.patch_transform_world.numpy()[0, :3], 0.0, atol=1.0e-6
        )

    def test_cuda_patch_boundary_is_in_bounds(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        model = _make_model(device="cuda:0")
        sensor = SensorTactile(model, sensing_shapes=[0], grid_shape=(20, 25), patch_size=(0.04, 0.03))
        contacts = _make_contacts(model, point0=(0.02, -0.015, 0.0), force=(1.0, 2.0, 3.0))
        sensor.update(model.state(), contacts, timestamp=0.0)
        np.testing.assert_allclose(sensor.force.numpy().sum(axis=1)[0], [1.0, 2.0, 3.0], atol=1.0e-6)

    def test_native_force_sign_and_conservation(self):
        model = _make_model()
        sensor = SensorTactile(
            model,
            sensing_shapes=[0, 1],
            grid_shape=(4, 5),
            patch_size=(0.04, 0.03),
        )
        contacts = _make_contacts(model)
        sensor.update(model.state(), contacts, timestamp=0.25)

        self.assertEqual(int(sensor.raw_count.numpy()[0]), 2)
        raw_force = sensor.raw_force_patch.numpy()[:2]
        np.testing.assert_allclose(raw_force[0], [1.0, 2.0, 3.0], atol=1.0e-6)
        np.testing.assert_allclose(raw_force[1], [-1.0, -2.0, -3.0], atol=1.0e-6)

        dense_force = sensor.force.numpy().sum(axis=1)
        np.testing.assert_allclose(dense_force, raw_force, atol=1.0e-6)
        np.testing.assert_allclose(sensor.unmapped_force_patch.numpy(), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(sensor.max_penetration.numpy().max(axis=1), [0.002, 0.002], atol=1.0e-6)

    def test_grid_axes_match_tacsl_row_x_column_y(self):
        model = _make_model()
        sensor = SensorTactile(
            model,
            sensing_shapes=[0],
            grid_shape=(3, 5),
            patch_size=(0.04, 0.03),
        )
        contacts = _make_contacts(
            model,
            point0=(0.02, -0.015, 0.0),
            force=(1.0, 2.0, 3.0),
        )
        sensor.update(model.state(), contacts, timestamp=0.0)
        active = sensor.active.numpy()[0].reshape(3, 5)
        self.assertEqual(int(active.sum()), 1)
        self.assertEqual(int(active[-1, 0]), 1)

    def test_shape_order_symmetry(self):
        model = _make_model()
        sensor = SensorTactile(model, sensing_shapes=[0], grid_shape=(4, 5), patch_size=(0.04, 0.03))
        state = model.state()

        contacts = _make_contacts(model, pair=(0, 1), force=(2.0, -1.0, 4.0))
        sensor.update(state, contacts, timestamp=0.0)
        force_shape0 = sensor.total_force_patch.numpy()[0].copy()

        contacts = _make_contacts(
            model,
            pair=(1, 0),
            point0=(0.0, 0.0, -0.002),
            point1=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, -1.0),
            force=(-2.0, 1.0, -4.0),
        )
        sensor.update(state, contacts, timestamp=0.1)
        np.testing.assert_allclose(sensor.total_force_patch.numpy()[0], force_shape0, atol=1.0e-6)
        self.assertEqual(int(sensor.raw_sensor_is_shape0.numpy()[0]), 0)

    def test_translation_rotation_invariance(self):
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        transform = wp.transform(wp.vec3(1.0, -2.0, 0.5), rotation)
        model = _make_model(xform_a=transform, xform_b=transform)
        sensor = SensorTactile(model, sensing_shapes=[0], grid_shape=(4, 5), patch_size=(0.04, 0.03))
        force_local = wp.vec3(1.0, 2.0, 3.0)
        force_world = wp.quat_rotate(rotation, force_local)
        contacts = _make_contacts(
            model,
            point0=(0.005, -0.004, 0.0),
            point1=(0.005, -0.004, 0.0),
            force=tuple(force_world),
        )
        sensor.update(model.state(), contacts, timestamp=1.0)

        np.testing.assert_allclose(sensor.raw_force_patch.numpy()[0], force_local, atol=1.0e-6)
        np.testing.assert_allclose(sensor.raw_point_patch.numpy()[0], [0.005, -0.004, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(sensor.force.numpy()[0].sum(axis=0), force_local, atol=1.0e-6)

    def test_counterpart_filter_and_unmapped_force(self):
        model = _make_model()
        sensor = SensorTactile(
            model,
            sensing_shapes=[0],
            counterpart_shapes=[1],
            grid_shape=(4, 5),
            patch_size=(0.01, 0.01),
        )
        contacts = _make_contacts(model, point0=(0.02, 0.0, 0.0), force=(4.0, 5.0, 6.0))
        sensor.update(model.state(), contacts, timestamp=0.0)
        np.testing.assert_allclose(sensor.force.numpy(), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(sensor.unmapped_force_patch.numpy()[0], [4.0, 5.0, 6.0], atol=1.0e-6)
        np.testing.assert_allclose(sensor.total_force_patch.numpy()[0], [4.0, 5.0, 6.0], atol=1.0e-6)

        rejected = SensorTactile(
            model,
            sensing_shapes=[0],
            counterpart_shapes=[0],
            grid_shape=(4, 5),
            patch_size=(0.04, 0.03),
        )
        rejected.update(model.state(), contacts, timestamp=0.0)
        self.assertEqual(int(rejected.raw_count.numpy()[0]), 0)
        np.testing.assert_allclose(rejected.total_force_patch.numpy(), 0.0, atol=1.0e-6)

    def test_clock_and_reset(self):
        model = _make_model()
        sensor = SensorTactile(model, sensing_shapes=[0])
        contacts = _make_contacts(model)
        state = model.state()

        sensor.update(state, contacts, timestamp=2.0)
        self.assertEqual(sensor.sequence, 0)
        self.assertEqual(sensor.dt, 0.0)
        sensor.update(state, contacts, timestamp=2.02)
        self.assertEqual(sensor.sequence, 1)
        self.assertAlmostEqual(sensor.dt, 0.02)
        with self.assertRaises(ValueError):
            sensor.update(state, contacts, timestamp=1.0)

        sensor.reset()
        self.assertEqual(sensor.sequence, -1)
        self.assertEqual(int(sensor.raw_count.numpy()[0]), 0)
        np.testing.assert_allclose(sensor.force.numpy(), 0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
