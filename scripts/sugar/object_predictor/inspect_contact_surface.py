"""Inspect saved ideal tactile surface resolution and render it on actual meshes.

Area-weighted PCA consumes tactile positions only. Object-mesh nearest normals
are validation targets, never estimator/controller inputs. No model or physics.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def fit_surface(points, weights):
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    values, vectors = np.linalg.eigh((centered * weights[:, None]).T @ centered / weights.sum())
    return center, np.sqrt(np.maximum(values, 0)), vectors[:, 0]


def inspect(root):
    import trimesh
    from sugar_newton.hand.patches import PATCH_SPECS
    data = dict(np.load(root / 'episode_5007.npz'))
    field = dict(np.load(root / 'contact_surface.npz'))
    base = dict(np.load('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    scale = data['object_dimensions_m'][0] / np.ptp(base['vertices'], axis=0)
    mesh = trimesh.Trimesh(base['vertices'] * scale, base['faces'], process=False)
    rows = []
    load_error = shear_error = area_error = centroid_error = full_load_error = 0.
    for frame, (start, end) in enumerate(zip(field['offset'][:-1], field['offset'][1:])):
        pos = field['position_hand_frame_m'][start:end].astype(float)
        area = field['area_m2'][start:end].astype(float)
        pressure = field['normal_pressure_pa'][start:end].astype(float)
        traction = field['shear_traction_hand_frame_pa'][start:end].astype(float)
        hand = field['hand'][start:end]
        pad = field['pad'][start:end]
        weight = area * pressure
        hand_rot = Rotation.from_quat(data['hand_pose_w'][frame, :, 3:])
        obj_rot = Rotation.from_quat(data['object_pose_w'][frame, 3:])
        normal_load = np.zeros(54)
        contact_area = np.zeros(54)
        shear = np.zeros((54, 3))
        for p in np.unique(pad[pad >= 0]):
            select = pad == p
            assert np.all(hand[select] == p // 27)
            load = weight[select].sum()
            if load <= 1e-6:
                continue
            normal_load[p] = load
            contact_area[p] = area[select].sum()
            shear[p] = hand_rot[p // 27].apply((traction[select] * area[select, None]).sum(0))
            center = hand_rot[p // 27].apply(np.average(pos[select], axis=0, weights=weight[select])) + data['hand_pose_w'][frame, p // 27, :3]
            centroid_error = max(centroid_error, float(np.linalg.norm(center - data['contact_position_w'][frame, p])))
        load_error = max(load_error, float(np.max(abs(normal_load - data['normal_load_n'][frame]))))
        area_error = max(area_error, float(np.max(abs(contact_area - data['contact_area_m2'][frame]))))
        shear_error = max(shear_error, float(np.max(abs(shear - data['shear_force_w'][frame]))))
        for side in (0, 1):
            total = weight[hand == side].sum()
            full_load_error = max(full_load_error, abs(total - float(data['validation_resolved_normal_n'][frame, side])))
            sel = (hand == side) & (pad >= 0) & (weight > 0)
            points = pos[sel]
            if len(points) < 3:
                continue
            center, rms, normal = fit_surface(points, area[sel])
            normal_w = hand_rot[side].apply(normal)
            world = hand_rot[side].apply(points) + data['hand_pose_w'][frame, side, :3]
            local = obj_rot.inv().apply(world - data['object_pose_w'][frame, :3])
            _, distance, faces = mesh.nearest.on_surface(local)
            truth = obj_rot.apply(mesh.face_normals[faces])
            angles = np.degrees(np.arccos(np.clip(abs(truth @ normal_w), 0, 1)))
            labels = [PATCH_SPECS[p % 27].name for p in np.unique(pad[sel])]
            rows.append(dict(frame=frame, time_s=float(data['timestamp_s'][frame]), hand=side,
                             faces=len(points), pads=labels, load_n=float(weight[sel].sum()),
                             unassigned_load_n=float(weight[(hand == side) & (pad < 0)].sum()),
                             area_mm2=float(area[sel].sum() * 1e6), center_hand_m=center.tolist(),
                             rms_mm=(rms * 1000).tolist(), extent_hand_mm=(np.ptp(points, axis=0) * 1000).tolist(),
                             fitted_normal_w=normal_w.tolist(),
                             validation_object_normal_angle_deg=float(np.average(angles, weights=area[sel])),
                             validation_nearest_object_distance_mm=float(np.average(distance, weights=area[sel]) * 1000)))
    primary = [r for r in rows if 14.5 <= r['time_s'] <= 16. and r['load_n'] > 1.]
    summary = {}
    for side in (0, 1):
        part = [r for r in primary if r['hand'] == side]
        summary[str(side)] = dict(frames=len(part), pads=sorted({p for r in part for p in r['pads']}),
                                 median_faces=float(np.median([r['faces'] for r in part])),
                                 median_area_mm2=float(np.median([r['area_mm2'] for r in part])),
                                 median_rms_mm=np.median([r['rms_mm'] for r in part], axis=0).tolist(),
                                 median_validation_object_normal_angle_deg=float(np.median([r['validation_object_normal_angle_deg'] for r in part])))
    report = dict(source=str(root), frames=len(data['timestamp_s']), surface_samples=len(field['pad']),
                  finite=all(np.isfinite(v).all() for v in field.values()),
                  integration_max_errors=dict(pad_load_n=load_error, pad_shear_component_n=shear_error,
                                              pad_area_m2=area_error, active_centroid_m=centroid_error,
                                              full_hand_normal_n=full_load_error),
                  primary_window='14.5 <= t <= 16.0 s; every recorded hand with load > 1 N',
                  primary=summary, rows=rows,
                  limits=['One failed acquisition; no reconstruction or predictor benefit demonstrated.',
                          'Contact interface is an ideal simulated tactile measurement, not a measured hardware depth map.',
                          'PCA is an explicitly labelled geometric diagnostic, not a learned replacement for Utonia.',
                          'Only assigned pad faces enter the fit. Unassigned contact remains separately reported.',
                          'Ground truth mesh normals are used only to measure fit error.'])
    (root / 'CONTACT_SURFACE_REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    return data, field, rows


def render(root, data, field, rows):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from .render_object_state import World, true_state, canvas, text, mat, PW, PH, TEAL
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw
    world = World()
    renderer = pyrender.OffscreenRenderer(PW, PH)
    # Dense points rendered at 0.12 mm radius. Centroid markers in old videos
    # were deliberately larger; they did not represent contact patch size.
    marker = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1, radius=.00012),
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1., .05, .02, 1.], emissiveFactor=[.5, 0, 0]))
    extra = []
    outputs = []
    for frame in (750, 800, 850):
        for node in extra:
            world.scene.remove_node(node)
        extra = []
        start, end = field['offset'][frame:frame + 2]
        hand = field['hand'][start:end]; pad = field['pad'][start:end]
        local = field['position_hand_frame_m'][start:end]
        pressure = field['normal_pressure_pa'][start:end]
        poses = data['hand_pose_w'][frame]
        points = np.empty_like(local)
        for side in (0, 1):
            sel = hand == side
            points[sel] = Rotation.from_quat(poses[side, 3:]).apply(local[sel]) + poses[side, :3]
        for point in points[(pad >= 0) & (pressure > 0)]:
            extra.append(world.scene.add(marker, pose=mat(point)))
        c, q, dims, _ = true_state(data, frame)
        im = canvas('侧夹失败 · 完整接触面与实际手部几何',
                    f'实际记录 {data["timestamp_s"][frame]:.2f} s  |  完整原始网格  |  红点：接触面采样，半径 0.12 mm',
                    ['实际箱体与双手', '左手接触近景 / 隐去物体', '右手接触近景 / 隐去物体'])
        for panel in range(3):
            world.set(poses, c, q, dims, data['contact_position_w'][frame], np.zeros(54))
            if panel == 0:
                world.camera_at(c + [.7, -1.2, .5], c)
            else:
                side = panel - 1
                part = next((r for r in rows if r['frame'] == frame and r['hand'] == side), None)
                center_local = np.array(part['center_hand_m']) if part else np.array([.112, (-1 if side == 0 else 1) * .018, -.0342])
                target = Rotation.from_quat(poses[side, 3:]).apply(center_local) + poses[side, :3]
                # Look from the palmar side; object hidden for this labelled
                # inspection view, while the contact coordinates stay exact.
                eye = target + Rotation.from_quat(poses[side, 3:]).apply([.025, -.055 if side == 0 else .055, .018])
                world.camera_at(eye, target)
                world.scene.set_pose(world.object_node, mat([0, 0, -20]))
                for name, node, _ in world.robot:
                    if name.startswith('right' if side == 0 else 'left'):
                        world.scene.set_pose(node, mat([0, 0, -20]))
            color, _ = renderer.render(world.scene, flags=0)
            if color.std() < 5:
                renderer.delete(); renderer = pyrender.OffscreenRenderer(PW, PH)
                color, _ = renderer.render(world.scene, flags=0)
                if color.std() < 5:
                    raise RuntimeError('Constant RGB in contact surface render')
            im.paste(Image.fromarray(color), (8 + panel * 532, 144))
            draw = ImageDraw.Draw(im)
            if panel == 0:
                text(draw, (24, 760), f'实际网格离地 {data["validation_full_mesh_min_z_m"][frame] * 1000:.2f} mm', 25)
                text(draw, (24, 805), '局部接触不等于稳定抓持', 26)
            else:
                if part:
                    text(draw, (24 + panel * 532, 752), f'{part["faces"]} 个面采样 / {len(part["pads"])} 个区域', 25, TEAL)
                    text(draw, (24 + panel * 532, 792), f'面积 {part["area_mm2"]:.2f} mm² / 载荷 {part["load_n"]:.2f} N', 24)
                    text(draw, (24 + panel * 532, 833), '区域：' + ', '.join(part['pads']), 23)
                else:
                    text(draw, (24 + panel * 532, 760), '该手已无记录的掌侧接触', 25)
        text(ImageDraw.Draw(im), (24, 914), '近景仅隐藏遮挡物体以查看传感区域；接触位置未移动。未运行模型，未重放物理。', 22)
        path = root / f'contact_mesh_frame_{frame:04d}.png'
        im.save(path); outputs.append(str(path)); print('RENDERED', path, flush=True)
    renderer.delete()
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.source)
    data, field, rows = inspect(root)
    render(root, data, field, rows)
