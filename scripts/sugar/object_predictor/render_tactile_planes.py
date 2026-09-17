"""Actual mesh replay beside centroid-only and tactile-plane diagnostics.

Guide planes depict local orientation, not reconstructed object extent. Fitting
never reads object state. Truth is only rendered or used to report a width error.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import pyrender
import trimesh

from .inspect_contact_surface import fit_surface
from .render_device import configure_retained_egl
from .render_object_state import World, load, true_state, canvas, text, mat, material, PW, PH, TEAL, AMBER


def planes(data, field, frame):
    start, end = field['offset'][frame:frame + 2]
    result = []
    for hand in (0, 1):
        sel = ((field['hand'][start:end] == hand) & (field['pad'][start:end] >= 0)
               & (field['normal_pressure_pa'][start:end] > 0))
        points = field['position_hand_frame_m'][start:end][sel].astype(float)
        area = field['area_m2'][start:end][sel].astype(float)
        load_n = float(np.sum(area * field['normal_pressure_pa'][start:end][sel]))
        if len(points) < 6 or load_n < 2.:
            result.append(None); continue
        center, rms, normal = fit_surface(points, area)
        if rms[1] < .0002 or rms[0] / max(rms[1], 1e-12) > .35:
            result.append(None); continue
        pose = data['hand_pose_w'][frame, hand]
        rot = Rotation.from_quat(pose[3:])
        result.append(dict(center=rot.apply(center) + pose[:3], normal=rot.apply(normal),
                           rms_mm=rms * 1000, count=len(points), area_mm2=area.sum() * 1e6))
    return result


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    configure_retained_egl()
    source = Path(args.source); out = source / 'renders'; out.mkdir(exist_ok=True)
    episode = int(json.loads((source / 'RESULT.json').read_text())['episode'])
    data = load(source / f'episode_{episode:04d}.npz'); field = load(source / 'contact_surface.npz')
    camera_target = true_state(data, 0)[0] + np.array([0., 0., .10])
    base = load('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz')
    world = World(); renderer = pyrender.OffscreenRenderer(PW, PH)
    guides = []
    for color in (TEAL, AMBER):
        # Explicitly a 10 cm orientation guide, not a claimed surface boundary.
        square = trimesh.creation.box([.10, .10, .0003])
        m = material(color); m.baseColorFactor[3] = .28; m.alphaMode = 'BLEND'; m.doubleSided = True
        plane = world.scene.add(pyrender.Mesh.from_trimesh(square, material=m))
        rod = world.scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.cylinder(radius=.0012, height=.06), material=material(color)))
        guides.append((plane, rod))
    if args.preview_frames is not None and not args.preview:
        raise ValueError('--preview-frames requires --preview')
    frames = (args.preview_frames if args.preview_frames is not None else
              [750, 1200, 1500, len(data['timestamp_s']) - 1]) if args.preview else list(range(0, len(data['timestamp_s']), 10))
    if not all(0 <= frame < len(data['timestamp_s']) for frame in frames):
        raise ValueError('Preview frame outside recorded trajectory')
    sequence = out / 'plane_frames'
    if not args.preview:
        sequence.mkdir(exist_ok=True)
    rows = []
    try:
        for sequence_index, frame in enumerate(frames):
            fits = planes(data, field, frame)
            c, q, dims, mass = true_state(data, frame)
            t = float(data['timestamp_s'][frame])
            loads = data['normal_load_n'][frame].reshape(2, 27).sum(1)
            valid = all(f is not None for f in fits)
            width = angle = width_error = None
            if valid:
                n0, n1 = (f['normal'] for f in fits)
                angle = float(np.degrees(np.arccos(np.clip(abs(n0 @ n1), 0, 1))))
                if n0 @ n1 < 0:
                    n1 = -n1
                axis = n0 + n1; axis /= np.linalg.norm(axis)
                if angle <= 10.:
                    width = float(abs((fits[1]['center'] - fits[0]['center']) @ axis))
                    # Validation only: full true mesh width along the inferred axis.
                    v = base['vertices'] * (dims / world.dims)
                    true_width = float(np.ptp(Rotation.from_quat(q).apply(v) @ axis))
                    width_error = abs(width - true_width) * 1000
            im = canvas('触觉几何 · 从接触中心到局部表面方向',
                        '真实采集回放  |  2× 速度  |  几何拟合诊断，非 Utonia 预测或完整形状重建',
                        ['真实箱体与双手', '原压缩表示：接触中心', '触觉表面：局部平面方向'])
            for panel in range(3):
                world.set(data['hand_pose_w'][frame], c, q, dims, data['contact_position_w'][frame], data['normal_load_n'][frame])
                for p, r in guides:
                    world.scene.set_pose(p, mat([0, 0, -20])); world.scene.set_pose(r, mat([0, 0, -20]))
                if panel:
                    world.scene.set_pose(world.object_node, mat([0, 0, -20]))
                if panel == 2:
                    for fit, (p, r) in zip(fits, guides):
                        if fit is None:
                            continue
                        rotation = Rotation.align_vectors(fit['normal'][None], np.array([[0., 0., 1.]]))[0]
                        world.scene.set_pose(p, mat(fit['center'], rotation.as_quat()))
                        world.scene.set_pose(r, mat(fit['center'], rotation.as_quat()))
                world.camera_at(camera_target + [.7, -1.2, .5], camera_target)
                rgb, _ = renderer.render(world.scene, flags=0)
                if rgb.std() < 5:
                    renderer.delete(); renderer = pyrender.OffscreenRenderer(PW, PH)
                    rgb, _ = renderer.render(world.scene, flags=0)
                    if rgb.std() < 5:
                        raise RuntimeError(f'Constant RGB at {frame}/{panel}')
                im.paste(Image.fromarray(rgb), (8 + panel * 532, 144))
            d = ImageDraw.Draw(im)
            text(d, (24, 750), f'时间 {t:.2f} s / 真值质量 {mass:.3f} kg', 25)
            text(d, (24, 794), f'网格离地 {data["validation_full_mesh_min_z_m"][frame] * 100:.1f} cm', 27)
            text(d, (24, 840), f'掌侧载荷 L {loads[0]:.1f} / R {loads[1]:.1f} N', 24)
            counts = (data['normal_load_n'][frame].reshape(2, 27) > .001).sum(1)
            text(d, (556, 750), f'有效区域 L {counts[0]} / R {counts[1]}', 27)
            text(d, (556, 796), '每区域一个中心，隐藏物体真值', 23)
            text(d, (556, 839), '接触点标记放大以便观察', 23)
            text(d, (1088, 750), f'可拟合局部面 {sum(f is not None for f in fits)} / 2', 27)
            if width is not None:
                text(d, (1088, 794), f'估计两侧间距 {width * 100:.2f} cm', 27, AMBER)
                text(d, (1088, 836), f'验证误差 {width_error:.2f} mm / 仅一维', 24)
            elif angle is not None:
                text(d, (1088, 797), f'面方向相差 {angle:.1f}°，不估间距', 23)
            else:
                text(d, (1088, 797), '接触不足，暂不估计间距', 23)
            text(d, (24, 914), '半透明方片仅示意局部方向，不是物体边界；间距假设两侧近似平行。真值不参与几何拟合。', 20)
            if args.preview or frame in (750, 1200, 1500, 2390):
                im.save(out / f'planes_frame_{frame:04d}.png')
            if not args.preview:
                im.save(sequence / f'frame_{sequence_index:04d}.png')
            rows.append(dict(frame=frame, time_s=t, resolved_faces=sum(f is not None for f in fits),
                             plane_angle_deg=angle, inferred_separation_m=width, validation_width_error_mm=width_error))
            if frame % 250 == 0:
                print('RENDER', frame, flush=True)
    finally:
        renderer.delete()
    if not args.preview:
        path = out / 'tactile_plane_carry.mp4'
        # Keep FFmpeg creation after the EGL context has been destroyed. The
        # inline-encoding attempt failed constant RGB before its first frame;
        # its failed log/status are retained. Every rendered PNG remains saved.
        with imageio.get_writer(path, fps=10, codec='libx264', quality=8,
                                pixelformat='yuv420p', macro_block_size=1) as writer:
            for sequence_index in range(len(frames)):
                writer.append_data(imageio.imread(sequence / f'frame_{sequence_index:04d}.png'))
        reader = imageio.get_reader(path); meta = reader.get_meta_data(); count = 0
        for rgb in reader:
            assert rgb.shape == (960, 1600, 3)
            assert all(rgb[144:724, 8 + p * 532:528 + p * 532].std() > 5 for p in range(3))
            count += 1
        reader.close()
        assert count == len(frames) and meta['fps'] == 10
        (out / 'TACTILE_PLANES.json').write_text(json.dumps(dict(frames=count, fps=10, seconds=count / 10,
                 full_decode_nonblack=True, source=str(source), source_controls=len(data['timestamp_s']),
                 rows=rows, model_updates=0, new_physics_steps=0), indent=2))
        print('COMPLETE', count, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--preview-frames', type=int, nargs='+')
    main(parser.parse_args())
