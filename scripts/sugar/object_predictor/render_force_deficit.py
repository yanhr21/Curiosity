"""Actual saved geometry with explicitly validation-only coverage diagnostics."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyrender

from .render_device import configure_retained_egl
from .render_object_state import World, load, true_state, canvas, text, PW, PH, AMBER
from .render_force_observability import arrow


def main():
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); args = ap.parse_args()
    root = Path(args.output); p = json.loads((root/'PROTOCOL.json').read_text())
    configure_retained_egl(); world = World(); renderer = pyrender.OffscreenRenderer(PW, PH)
    for primitive in world.object_node.mesh.primitives:
        primitive.material.baseColorFactor = np.array([.1, .7, .75, .28])
        primitive.material.alphaMode = 'BLEND'; primitive.material.doubleSided = True
    records = []
    try:
        for ep in p['visual_episodes']:
            a = load(Path(p['source'])/'cases'/f'episode_{ep}'/f'episode_{ep}.npz')
            d = load(root/f'episode_{ep}.npz'); frame = p['visual_frame']
            idx, = np.flatnonzero(d['frame']==frame)
            c, q, dims, mass = true_state(a, frame)
            im = canvas('承重信号主要漏在哪里：同一真实搬运时刻',
                        f'案例 {ep} | {a["timestamp_s"][frame]:.2f} 秒 | 真值 {mass:.3f} kg | 仅传感器覆盖诊断，无新模型预测',
                        ['现有掌侧剪切观测', '全接触表面剪切（验证专用）', '完整求解器支持力（验证专用）'])
            record = dict(episode=ep, frame=frame, values=[])
            for panel, key in enumerate(('observed_support', 'full_field_shear_support', 'full_solver_support')):
                fz = float(d[key+'_w'][idx, 2])
                world.set(a['hand_pose_w'][frame], c, q, dims, a['contact_position_w'][frame], a['normal_load_n'][frame])
                world.camera_at(c+[.75, -1.25, .60], c+[0., 0., .06])
                nodes = arrow(world, c+[-.25, 0., .10], fz, AMBER)
                nodes += arrow(world, c+[.25, 0., .18], -mass*p['gravity_m_s2'], (90, 100, 115))
                rgb, _ = renderer.render(world.scene, flags=0)
                if rgb.std() < 5: raise RuntimeError('Constant rendering')
                im.paste(Image.fromarray(rgb), (8+panel*532, 144)); draw = ImageDraw.Draw(im); x = 24+panel*532
                equivalent = fz/p['gravity_m_s2']
                text(draw, (x, 747), f'竖向支持分量 {fz:+.3f} N', 25, AMBER)
                text(draw, (x, 795), f'支持等效质量 {equivalent:.3f} kg', 24)
                text(draw, (x, 838), f'质量相对误差 {abs(equivalent/mass-1)*100:.1f}%', 22)
                text(draw, (x, 879), '橙：支持分量；灰：真实重力', 19)
                record['values'].append(dict(mode=key, force_z_n=fz, equivalent_mass_kg=equivalent))
                for node in nodes: world.scene.remove_node(node)
            text(ImageDraw.Draw(im), (24, 925), '后两列包含模型不可见的参考量，不能当作观测或预测改进。箭头统一0.05米/牛，平移展示，不代表真实作用点。', 18)
            file = f'coverage_{ep}.png'; im.save(root/file); record['image'] = file; records.append(record)
    finally:
        renderer.delete()
    (root/'RENDER.json').write_text(json.dumps(dict(complete=True, records=records), indent=2)+'\n')
    print('FORCE_DEFICIT_RENDER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
