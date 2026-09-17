"""All TEST sensing-fixture episodes with frozen learned estimates, actual meshes.

Rendering consumes recorded hand/object poses and saved batch1 predictions.
Truth is an evaluator overlay only. PNG generation precedes FFmpeg encoding.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrender
from PIL import Image, ImageDraw

from .render_object_state import World, load, true_state, state_from_saved, canvas, text, mat, PW, PH, BG, TEAL, AMBER
from .report_grip_transfer import errors
from .render_device import configure_retained_egl


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    configure_retained_egl()
    root = Path(args.root); protocol = json.loads((root / 'PROTOCOL.json').read_text()); data_root = Path(protocol['data'])
    cases = [c for c in json.loads((data_root / 'PROTOCOL.json').read_text())['configurations'] if c['split'] == 'test']
    arms = ['centroid_force', 'surface_force_airborne_mass_guard']
    saved = [load(root / 'visual' / arm / 'predictions.npz') for arm in arms]
    episode_errors = [errors(values['prediction'], values['target']) for values in saved]
    for key in ('episode', 'frame', 'target', 'mass_label_weight'):
        if not np.array_equal(saved[0][key], saved[1][key]):
            raise ValueError('Visual arms have different source clocks or labels')
    # Original primary clocks must retain the exact batch1 estimates. Extra
    # rendering clocks do not select or modify any prediction.
    for arm, visual in zip(arms, saved):
        primary = load(root / 'batch1' / arm / 'predictions.npz')
        selected = (visual['frame'] - 31) % 25 == 0
        if not all(np.array_equal(visual[k][selected], primary[k]) for k in ('episode', 'frame', 'target')):
            raise ValueError('Original primary clocks missing from visual readback')
        if np.max(abs(visual['prediction'][selected] - primary['prediction'])) > 1e-5:
            raise ValueError('Original batch1 predictions changed in visual readback')
    out = root / 'renders'; out.mkdir(exist_ok=True); sequence = out / 'frames'; sequence.mkdir(exist_ok=True)
    world = World(); renderer = pyrender.OffscreenRenderer(PW, PH)
    paths = []; posters = []; case_rows = []
    try:
        for case_index, case in enumerate(cases):
            episode = case['episode']; source = data_root / 'cases' / f'episode_{episode:04d}'
            data = load(source / f'episode_{episode:04d}.npz')
            camera_target = true_state(data, 0)[0] + np.array([0., 0., .10])
            expected = np.arange(31, len(data['timestamp_s']), 5)
            lookups = []
            for values in saved:
                indices = np.flatnonzero(values['episode'] == episode)
                if not np.array_equal(values['frame'][indices], expected):
                    raise ValueError('Incomplete declared TEST visual clock grid')
                lookups.append({int(values['frame'][i]): int(i) for i in indices})
            frames = list(range(1, len(data['timestamp_s']), 5))
            case_rows.append(dict(episode=episode, source=str(source), video_start_s=len(paths) / 40., frames=len(frames)))
            for frame in frames:
                path = sequence / f'episode_{episode:04d}_frame_{frame:04d}.png'
                paths.append(path)
                if path.exists():
                    with Image.open(path) as previous:
                        rgb = np.asarray(previous)
                        if rgb.shape != (960, 1600, 3) or any(rgb[144:724, 8+p*532:528+p*532].std() < 5 for p in range(1 if frame < 31 else 3)):
                            raise ValueError('Invalid saved render; preserve it for diagnosis')
                    continue
                truth = true_state(data, frame); c, q, dims, mass = truth
                t = float(data['timestamp_s'][frame])
                im = canvas('物体状态估计 · 完整 Utonia 匹配测试',
                    f'TEST {case_index+1}/8  |  案例 {episode}  |  实际记录回放  |  4× 速度  |  包含全部采集失败',
                    ['实际箱体与双手', '原接触中心 / 学习预测', '接触面 + 重量监督 / 学习预测'])
                for panel in range(3):
                    for primitive in world.object_node.mesh.primitives:
                        primitive.material.baseColorFactor = [*(np.array(TEAL if panel == 0 else AMBER) / 255), 1.]
                    index = None if panel == 0 or frame < 31 else lookups[panel-1][frame]
                    state = truth if index is None else state_from_saved(saved[panel-1]['prediction'][index], data['hand_pose_w'][frame])
                    world.set(data['hand_pose_w'][frame], *state[:3], data['contact_position_w'][frame], data['normal_load_n'][frame])
                    world.camera_at(camera_target + [.8, -1.3, .5], camera_target)
                    if panel and index is None:
                        image = Image.new('RGB', (PW, PH), BG)
                        text(ImageDraw.Draw(image), (80, 255), f'积累历史 {frame+1} / 32 帧', 25)
                    else:
                        rgb, _ = renderer.render(world.scene, flags=0)
                        if rgb.std() < 5:
                            renderer.delete(); renderer = pyrender.OffscreenRenderer(PW, PH)
                            rgb, _ = renderer.render(world.scene, flags=0)
                            if rgb.std() < 5:
                                raise RuntimeError(f'Constant RGB at {episode}/{frame}/{panel}')
                        image = Image.fromarray(rgb)
                        if panel:
                            world.extent_guide(image, c, q, dims)
                    im.paste(image, (8+panel*532, 144)); draw = ImageDraw.Draw(im)
                    if panel == 0:
                        text(draw, (24, 750), f'记录时间 {t:.2f} s / 真值 {mass:.3f} kg', 25)
                        text(draw, (24, 795), f'完整网格离地 {data["validation_full_mesh_min_z_m"][frame]*100:.1f} cm', 26)
                        loads = data['normal_load_n'][frame].reshape(2, 27).sum(1)
                        text(draw, (24, 840), f'掌侧载荷 L {loads[0]:.1f} / R {loads[1]:.1f} N', 23)
                    elif index is not None:
                        e = episode_errors[panel-1]
                        text(draw, (24+panel*532, 749), f'预测质量 {state[3]:.3f} kg', 30, AMBER)
                        text(draw, (24+panel*532, 791), f'质量误差 {e["mass_pct"][index]:.1f}%', 25)
                        text(draw, (24+panel*532, 833), f'位置 {e["center_cm"][index]:.1f} cm / 旋转 {e["rotation_deg"][index]:.1f}°', 22)
                        text(draw, (24+panel*532, 870), f'尺寸误差 {e["size_pct"][index]:.1f}%', 22)
                text(ImageDraw.Draw(im), (24, 920), '完整双手夹具，非全身策略；青色真值轮廓仅用于显示。预测未平滑、未用真值修正；已知网格的位姿与尺度估计。', 17)
                im.save(path)
            posters.append(paths[-1]); print('RENDERED_CASE', episode, flush=True)
    finally:
        renderer.delete()
    movie = out / 'surface_state_comparison.mp4'
    with imageio.get_writer(movie, fps=40, codec='libx264', quality=8, pixelformat='yuv420p', macro_block_size=1) as writer:
        for path in paths:
            writer.append_data(imageio.imread(path))
    reader = imageio.get_reader(movie); count = 0
    for rgb in reader:
        if rgb.shape != (960, 1600, 3) or rgb[144:724, 8:528].std() < 5:
            raise ValueError('Invalid encoded RGB')
        count += 1
    fps = reader.get_meta_data()['fps']; reader.close()
    if count != len(paths) or fps != 40.:
        raise ValueError('Encoded frame count or rate differs')
    sheet = Image.new('RGB', (1600, 4*480), BG)
    for i, path in enumerate(posters):
        with Image.open(path) as image:
            sheet.paste(image.resize((800, 480)), ((i % 2)*800, (i // 2)*480))
    sheet.save(out / 'all_test_posters.jpg')
    (out / 'RENDER_RESULT.json').write_text(json.dumps(dict(complete=True, frames=count, fps=fps, seconds=count/fps,
        all_frames_decoded=True, cases=case_rows, predictions='frozen batch1, no smoothing or truth correction',
        new_physics_steps=0, optimizer_updates=0), indent=2))
    comparison = json.loads((root / 'COMPARISON.json').read_text())
    rows = []
    for arm, label in [('centroid_force', '接触中心'), ('surface_force', '完整接触面'), ('surface_force_airborne_mass_guard', '接触面 + 离地重量监督')]:
        result = comparison['summaries'][arm]['airborne_hold']; values = result['equal_episode']
        display = ['无标签' if values[k] is None else f'{values[k]:.3f}' for k in ('center_cm', 'rotation_deg', 'size_pct', 'mass_pct')]
        rows.append('| '+label+' | '+' | '.join(display)+f' | {result["episodes_covered"]}/{result["total_episodes"]} |')
    report = '\n'.join(['# 完整 Utonia 接触面状态预测对照', '',
        '[实际网格渲染视频](renders/surface_state_comparison.mp4) · [播放页](index.html) · [完整数值](COMPARISON.json)', '',
        '三组均使用完整官方 Utonia、相同初始化与实际四配置批次，各 2000 更新。质量监督守卫只影响损失，不参与输入或抽样。', '',
        '| 输入与监督 | 中心 cm | 旋转 ° | 尺寸 % | 质量 % | 离地保持案例覆盖 |',
        '|---|---:|---:|---:|---:|---:|', *rows, '',
        f'预先声明的整体科学验收：{"通过" if comparison["scientific_acceptance"] else "未全部通过"}。具体阈值、未覆盖案例、所有窗口和力置零对照均保留在完整数值文件。', '',
        '视频包含全部八个 TEST 配置与实际失败，4× 播放；没有新物理执行或额外训练。显示的是已知扫描网格的预测位姿与尺寸，尚非任意形状重建。', '',
        '局限：同一 CarryBox 家族、理想模拟触觉、部分手部传感覆盖。离地保持标签不保证完整受力可观测。未证明 native 迁移、材质估计、真实机器人或 demo following 收益。', '',
        '![主指标](primary_comparison.png)', '', '![训练过程](learning_curves.png)', ''])
    (root / 'REPORT.md').write_text(report)
    buttons = ''.join(f'<button onclick="document.querySelector(\'video\').currentTime={r["video_start_s"]}">案例 {r["episode"]}</button>' for r in case_rows)
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Utonia 物体状态估计</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;background:#f2f6f9;color:#1b2736;font:18px/1.7 system-ui}video,img{width:100%}button{margin:4px;padding:8px}a{color:#087e88}</style><h1>完整 Utonia：触觉几何与物体状态估计</h1><p>全部 TEST 配置，4× 播放；左为实际状态，中为接触中心基线，右为接触面与离地重量监督。不是全身策略或任意形状重建。</p><video controls preload="metadata" src="renders/surface_state_comparison.mp4"></video><p>'+buttons+'</p><p><a href="REPORT.md">完整报告与失败项</a> · <a href="COMPARISON.json">全部数值与验收</a></p><img src="primary_comparison.png"><img src="renders/all_test_posters.jpg">')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); main(parser.parse_args())
