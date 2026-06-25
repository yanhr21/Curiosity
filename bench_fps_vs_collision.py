# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Throwaway benchmark: FPS vs collision quality (penetration depth).
#
# Builds stacked box pyramids settling under gravity and measures, for a given
# solver setting, both throughput (frames/s, CUDA-graph + warmed + synced) and
# steady-state penetration depth computed solver-agnostically from the contact
# buffers:  d = dot(n, bx_b - bx_a) - (margin0 + margin1);  penetration = max(0, -d).
#
# Prints one CSV line so a sweep can fan out across processes/GPUs.

import argparse
import time

import numpy as np
import warp as wp

import newton

CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
Y_STACK = 5.0


def build_model(num_pyramids: int, pyramid_size: int, world_count: int):
    builder = newton.ModelBuilder()
    builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    box_count = 0
    for pyramid in range(num_pyramids):
        y_offset = pyramid * PYRAMID_SPACING
        for level in range(pyramid_size):
            num_cubes_in_row = pyramid_size - level
            row_width = (num_cubes_in_row - 1) * CUBE_SPACING
            for i in range(num_cubes_in_row):
                x_pos = -row_width / 2 + i * CUBE_SPACING
                z_pos = level * CUBE_SPACING + CUBE_HALF
                y_pos = Y_STACK - y_offset
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x_pos, y_pos, z_pos), q=wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF)
                box_count += 1

    if world_count > 1:
        main = newton.ModelBuilder()
        main.replicate(builder, world_count=world_count)
        model = main.finalize()
    else:
        model = builder.finalize()
    return model, box_count


def make_solver(name: str, model, iterations: int):
    if name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=iterations, rigid_contact_relaxation=0.8)
    if name == "mujoco":
        return newton.solvers.SolverMuJoCo(model, iterations=iterations, ls_iterations=max(2, iterations // 2))
    raise ValueError(name)


def _quat_rotate(q, v):
    # q: (...,4) xyzw ; v: (...,3)
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def penetration_stats(model, contacts, state):
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    normal = contacts.rigid_contact_normal.numpy()[:n]
    m0 = contacts.rigid_contact_margin0.numpy()[:n]
    m1 = contacts.rigid_contact_margin1.numpy()[:n]
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()  # (nbody, 7): px,py,pz, qx,qy,qz,qw

    def to_world(points, shapes):
        bodies = shape_body[shapes]
        out = np.array(points, dtype=np.float64)
        has_body = bodies >= 0
        if np.any(has_body):
            bq = body_q[bodies[has_body]]
            rotated = _quat_rotate(bq[:, 3:7], out[has_body])
            out[has_body] = rotated + bq[:, 0:3]
        return out

    bx_a = to_world(p0, s0)
    bx_b = to_world(p1, s1)
    nrm = np.array(normal, dtype=np.float64)
    thickness = (m0 + m1).astype(np.float64)
    d = np.einsum("ij,ij->i", nrm, bx_b - bx_a) - thickness
    pen = np.clip(-d, 0.0, None)
    return n, float(pen.max()), float(pen[pen > 0].mean()) if np.any(pen > 0) else 0.0, float((pen > 1e-6).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="xpbd", choices=["xpbd", "mujoco"])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--num-pyramids", type=int, default=2)
    ap.add_argument("--pyramid-size", type=int, default=10)
    ap.add_argument("--world-count", type=int, default=64)
    ap.add_argument("--settle-frames", type=int, default=200)
    ap.add_argument("--timing-frames", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()

    if args.header:
        print("solver,iterations,substeps,boxes,worlds,contacts,fps,steps_per_s,max_pen_mm,mean_pen_mm,frac_pen")
        return

    wp.set_device(args.device)
    fps = 100
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / args.substeps

    model, box_count = build_model(args.num_pyramids, args.pyramid_size, args.world_count)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap")
    solver = make_solver(args.solver, model, args.iterations)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = pipeline.contacts()

    def simulate():
        nonlocal state_0, state_1, contacts
        for _ in range(args.substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0, collision_pipeline=pipeline)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    use_graph = wp.get_device().is_cuda
    if use_graph:
        with wp.ScopedCapture() as cap:
            simulate()
        graph = cap.graph

    def run_frame():
        if use_graph:
            wp.capture_launch(graph)
        else:
            simulate()

    # Settle to steady state
    for _ in range(args.settle_frames):
        run_frame()
    wp.synchronize_device()

    # Measure penetration at steady state
    contacts = model.collide(state_0, collision_pipeline=pipeline)
    n_contacts, max_pen, mean_pen, frac_pen = penetration_stats(model, contacts, state_0)

    # Time throughput
    t0 = time.perf_counter()
    for _ in range(args.timing_frames):
        run_frame()
    wp.synchronize_device()
    elapsed = time.perf_counter() - t0

    fps_meas = args.timing_frames / elapsed
    steps_per_s = fps_meas * args.substeps * args.world_count
    print(
        f"{args.solver},{args.iterations},{args.substeps},{box_count},{args.world_count},"
        f"{n_contacts},{fps_meas:.1f},{steps_per_s:.0f},"
        f"{max_pen * 1000:.3f},{mean_pen * 1000:.3f},{frac_pen:.3f}"
    )


if __name__ == "__main__":
    main()
