"""All four actual canonical-approach fixture trajectories, fixed still clocks.

This renders recorded physics only. It contains no object predictor output.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyrender

from .render_device import configure_retained_egl
from .render_object_state import World, load, true_state, text
from .retained_execution import require_active_resource


def main(root):
    require_active_resource()
    result = json.loads((root / 'RESULT.json').read_text())
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    episodes = [5008, 5012, 5020, 5000]
    if not result['complete'] or [r['episode'] for r in result['cases']] != episodes:
        raise ValueError('All four fixed attempts must finish before rendering')
    if protocol['study'] != 'canonical_approach_fixture_v1':
        raise ValueError('Unexpected physical protocol')
    configure_retained_egl()
    out = root / 'renders'
    out.mkdir(exist_ok=False)
    renderer = pyrender.OffscreenRenderer(784, 620)
    world = World(ground_height=0.)
    records = []
    try:
        for case in result['cases']:
            episode = case['episode']
            data = load(root / 'cases' / f'episode_{episode}' / f'episode_{episode}.npz')
            if len(data['timestamp_s']) != 2400:
                raise ValueError('Incomplete recorded trajectory')
            center = true_state(data, 0)[0] + [0, 0, .1]
            for frame in (1181, 1581, 2381):
                im = Image.new('RGB', (1600, 820), '#f2f6f9')
                draw = ImageDraw.Draw(im)
                text(draw, (16, 12), f'{episode} · 固定 X 方向夹持采集 · {data["timestamp_s"][frame]:.2f} 秒', 27)
                text(draw, (16, 54), '同一真实完整网格的两个视角；无物体预测；失败也展示', 22)
                position, quat, dimensions, _ = true_state(data, frame)
                world.set(data['hand_pose_w'][frame], position, quat, dimensions,
                          data['contact_position_w'][frame], data['normal_load_n'][frame])
                for i, direction in enumerate(([.75, -1.2, .55], [-.9, -.8, .7])):
                    world.camera_at(center + direction, center)
                    rgb, depth = renderer.render(world.scene, flags=0)
                    if rgb.std() < 5 or not np.isfinite(depth).all() or np.count_nonzero(depth) < 100:
                        raise RuntimeError('Invalid actual full-mesh render')
                    im.paste(Image.fromarray(rgb), (8 + i * 800, 94))
                loads = data['normal_load_n'][frame].reshape(2, 27).sum(1)
                text(draw, (16, 731), f'离地 {data["validation_full_mesh_min_z_m"][frame]*100:.2f} cm；掌侧载荷 {loads[0]:.2f} / {loads[1]:.2f} N', 24)
                text(draw, (16, 773), '初始接近角明确改为 0°，不能证明原斜向抓取已修复。', 21)
                filename = f'episode_{episode}_frame_{frame}.png'
                im.save(out / filename)
                records.append(dict(episode=episode, frame=frame, image=filename, actual_mesh=True))
    finally:
        renderer.delete()
    (out / 'RESULT.json').write_text(json.dumps(dict(complete=True, expected_stills=12,
        actual_stills=len(records), records=records, model_predictions=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
