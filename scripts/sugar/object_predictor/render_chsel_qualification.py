"""Render every fixed CHSEL qualification clock with original full hand meshes."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
from PIL import Image, ImageDraw

from .render_object_state import (World, load, true_state, state_from_saved,
    canvas, text, PW, PH, BG, TEAL, AMBER)
from .render_device import configure_retained_egl


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    configure_retained_egl()
    root = Path(args.root); report = json.loads((root / 'RESULT.json').read_text())
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    uses_free = bool(protocol.get('free_points_root'))
    uses_fusion = bool(protocol.get('fusion_scales_file'))
    uses_floor = bool(protocol.get('known_floor'))
    original_start = args.original_start
    if original_start:
        assert protocol.get('local_only')
    out = root / ('renders_original_start' if original_start else 'renders'); out.mkdir(exist_ok=False)
    records = json.loads(Path('experiments/object_predictor_v1/response_surface_dataset_v1/COLLECTION_RESULT.json').read_text())['records']
    records = {r['episode']: r for r in records}
    world = World(ground_height=0. if uses_floor else None); renderer = pyrender.OffscreenRenderer(PW, PH)
    paths = []
    try:
        for row in report['cases']:
            ep, frame = row['episode'], row['frame']
            data = load(Path(records[ep]['source']) / f'episode_{ep}.npz')
            saved = load(root / f'episode_{ep}_frame_{frame}.npz')
            if original_start:
                pose = np.linalg.inv(saved['candidates_hand_to_object'][1])
                saved['prediction'] = saved['original_prediction'].copy()
                saved['prediction'][:3] = pose[:3, 3]
                saved['prediction'][3:9] = pose[:3, :2].T.reshape(6)
            assert np.array_equal(saved['hand_pose_w'], data['hand_pose_w'][frame])
            truth = true_state(data, frame); center, quat, dims, mass = truth
            target = true_state(data, 0)[0] + np.array([0., 0., .10])
            t = float(data['timestamp_s'][frame])
            im = canvas('物体状态估计 · Utonia 先验、触觉与已知地面约束' if uses_floor else
                ('物体状态估计 · Utonia 先验与接触几何融合' if uses_fusion else '触觉几何约束 · 官方 CHSEL 接口验证'),
                f'TRAIN {ep} | 记录时间 {t:.2f} s | 固定八个时刻之一 | 已知物体网格',
                ['实际箱体与双手', '冻结 Utonia 预测', '原预测出发 · 官方局部优化终点' if original_start else
                    ('先验 + 接触几何 + 已知地面' if uses_floor else
                     ('Utonia 持续先验 + 接触几何' if uses_fusion else
                     ('CHSEL：接触 + 手内排除' if uses_free else 'Utonia + 官方 CHSEL')))])
            for panel in range(3):
                for primitive in world.object_node.mesh.primitives:
                    primitive.material.baseColorFactor = [*(np.array(TEAL if panel == 0 else AMBER) / 255), 1.]
                state = truth if panel == 0 else state_from_saved(
                    saved['original_prediction' if panel == 1 else 'prediction'], data['hand_pose_w'][frame])
                world.set(data['hand_pose_w'][frame], *state[:3],
                    data['contact_position_w'][frame], data['normal_load_n'][frame])
                world.camera_at(target + [.8, -1.3, .5], target)
                rgb, _ = renderer.render(world.scene, flags=0)
                if rgb.std() < 5:
                    raise RuntimeError(f'Constant RGB {ep}/{frame}/{panel}')
                image = Image.fromarray(rgb)
                if panel:
                    world.extent_guide(image, center, quat, dims)
                im.paste(image, (8 + panel*532, 144)); draw = ImageDraw.Draw(im)
                x = 24 + panel*532
                if panel == 0:
                    text(draw, (x, 752), f'实际质量 {mass:.3f} kg', 27)
                    text(draw, (x, 797), f'完整网格离地 {data["validation_full_mesh_min_z_m"][frame]*100:.1f} cm', 24)
                    text(draw, (x, 843), '真值仅用于评价和显示', 24)
                else:
                    key = 'utonia' if panel == 1 else ('original_start_endpoint' if original_start else 'chsel'); e = row[key]
                    residual = row['original_contact_sdf_cm' if panel == 1 else 'selected_contact_sdf_cm']
                    text(draw, (x, 749), f'位置 {e["center_cm"]:.2f} cm / 旋转 {e["rotation_deg"]:.1f}°', 23)
                    text(draw, (x, 790), f'双向网格顶点距离 {e["symmetric_nearest_vertex_cm"]:.2f} cm', 22)
                    text(draw, (x, 831), '固定第0初值的终点；无大范围搜索' if panel == 2 and original_start else
                        f'接触到预测表面 {residual:.2f} cm', 23)
                    text(draw, (x, 872), f'质量 {state[3]:.3f} kg（质量、尺寸未修正）', 20)
                    if uses_floor:
                        z = row['minimum_world_z_m_prior_then_selected'][panel-1]*100
                        text(draw, (x, 900), f'完整预测网格最低点 {z:.2f} cm' + ('（穿地）' if z < -.1 else ''),
                             16, color=(180,45,45) if z < -.1 else TEAL)
            constraint = '接触与手内排除' if uses_free else '接触'
            text(ImageDraw.Draw(im), (24, 921), ('青色轮廓仅为评价真值；融合权重来自另外56个TRAIN时刻，按融合目标选择。固定8帧资格，非泛化或任意形状重建。' if uses_fusion else
                ('青色轮廓仅为评价真值；右侧固定原初值局部终点，未按真值挑选。TRAIN消融，非完整QD、泛化或形状重建结果。' if original_start else
                f'青色轮廓仅为评价真值；候选仅按当前{constraint}约束选择。TRAIN验证，不代表泛化、任意形状重建或搬运策略成功。')), 16)
            path = out / f'episode_{ep}_frame_{frame}.png'; im.save(path); paths.append(path)
            print('RENDERED_QUALIFICATION', ep, frame, flush=True)
    finally:
        renderer.delete()
    sheet = Image.new('RGB', (1600, 480*4), BG)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.resize((800, 480)), ((i % 2)*800, (i // 2)*480))
    sheet.save(out / 'all_eight.jpg')
    (out / 'RESULT.json').write_text(json.dumps(dict(complete=True, actual_mesh_images=len(paths),
        images=[str(p) for p in paths], scope='Eight still frames; not a continuous carry video',
        optimizer_updates=0, physics_steps=0), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
    ap.add_argument('--original-start', action='store_true', help='Render predeclared raw original-start local endpoint')
    main(ap.parse_args())
