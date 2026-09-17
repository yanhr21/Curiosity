"""Actual full-mesh comparison for the fixed occupancy-sampling repair.

No model/physics execution, smoothing, welding, fitted alignment, or mesh repair.
Three panels show truth, previous static coverage-v2 and dynamic extraction-grid overfit.
Both prediction columns use the same declared grid-coordinate correction. A rotating camera is not a physical
rollout. Failed/missing reconstructions get text placeholders, never a fake mesh.
--check imports real CPU dependencies only, without creating an EGL context.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .qualify_michelangelo_ae import OBJECT_IDS, CHECKPOINT_SHA, ABC_TO_VAE, digest
from .michelangelo_geometry_coordinates import correct_extracted_coordinates


def render(root):
    from .retained_execution import require_active_resource
    require_active_resource()
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import pyrender
    import trimesh
    from .render_device import configure_retained_egl
    from .render_object_state import canvas, text, material, look, PW, PH, BG, TEAL, AMBER

    result = json.loads((root/'RESULT.json').read_text())
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    if protocol.get('sampling_revision') != 'dynamic_extraction_grid_v1':
        raise ValueError('Require the reviewed dynamic extraction-grid sampling protocol')
    if (tuple(r['object_id'] for r in result['cases']) != OBJECT_IDS
            or result['scope'] != 'full_surface_native_ae_dynamic_grid_overfit_only'
            or result['initial_checkpoint_sha256'] != CHECKPOINT_SHA
            or result['model_updates'] != 1000
            or digest(root/'endpoint.pt') != result['checkpoint_sha256']):
        raise ValueError('Require actual full-model 1000-step endpoint and all four cases')
    baseline = root.parent/'michelangelo_coverage_overfit_v2'
    old_result = json.loads((baseline/'RESULT.json').read_text())
    old_protocol = json.loads((baseline/'PROTOCOL.json').read_text())
    if (old_result['scope'] != 'full_surface_native_ae_coverage_overfit_only'
            or old_protocol.get('sampling_revision') != 2
            or old_result['initial_checkpoint_sha256'] != CHECKPOINT_SHA
            or old_result['model_updates'] != 1000
            or old_result['checkpoint_sha256'] != digest(baseline/'endpoint.pt')):
        raise ValueError('Previous static coverage-v2 full-model endpoint differs')
    old_rows = {r['object_id']:dict(corrected=r['metrics'],original=r['metrics'],source_sha256=r['case_output_sha256'],checks=r['checks']) for r in old_result['cases']}
    bindings = {}
    for oid in OBJECT_IDS:
        old_path=baseline/f'{oid}.npz'
        if digest(old_path) != old_rows[oid]['source_sha256']:
            raise ValueError('Previous geometry/metric source changed')
        bindings[oid]=dict(previous_case_sha256=digest(old_path), endpoint_case_sha256=digest(root/f'{oid}.npz') if (root/f'{oid}.npz').exists() else None)
    out = root/'renders'; out.mkdir(exist_ok=False)
    device = configure_retained_egl()
    renderer = pyrender.OffscreenRenderer(PW, PH)
    movie = out/'native_ae_dynamic_grid_comparison.mp4'
    records, stills, frame_count = [], [], 0
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        with imageio.get_writer(movie, fps=30, codec='libx264', quality=8,
                                pixelformat='yuv420p', macro_block_size=1) as writer:
            for row in result['cases']:
                oid = row['object_id']; saved = root/f'{oid}.npz'
                vertices = np.load(row['source_vertices'], allow_pickle=False)
                faces = np.load(row['source_faces'], allow_pickle=False)
                with np.load(baseline/f'{oid}.npz',allow_pickle=False) as old:
                    if not np.array_equal(old['truth_vertices_canonical'],vertices) or not np.array_equal(old['truth_faces'],faces):
                        raise ValueError('Frozen truth mesh differs')
                    corrected=correct_extracted_coordinates(old['reconstruction_vertices_vae_raw_official'])/ABC_TO_VAE
                    if not np.array_equal(old['reconstruction_vertices_canonical'],corrected):
                        raise ValueError('Previous mesh coordinate binding differs')
                    frozen=(old['reconstruction_vertices_canonical'],old['reconstruction_faces'])
                meshes = [(vertices, faces), frozen, None]
                if saved.exists():
                    if digest(saved) != row['case_output_sha256']:
                        raise ValueError('Saved endpoint mesh source differs from evaluated case')
                    with np.load(saved, allow_pickle=False) as z:
                        if not np.array_equal(z['truth_vertices_canonical'], vertices) or not np.array_equal(z['truth_faces'], faces):
                            raise ValueError('Saved AE truth differs from original full mesh')
                        corrected=correct_extracted_coordinates(z['reconstruction_vertices_vae_raw_official'])/ABC_TO_VAE
                        if not np.array_equal(z['reconstruction_vertices_canonical'],corrected):
                            raise ValueError('Displayed mesh differs from declared grid correction')
                        pred = (z['reconstruction_vertices_canonical'], z['reconstruction_faces'])
                    meshes[2] = pred
                scenes, cameras = [], []
                for panel, mesh in enumerate(meshes):
                    if mesh is None:
                        scenes.append(None); cameras.append(None); continue
                    v, f = mesh
                    if (not np.isfinite(v).all() or f.ndim != 2 or f.shape[1] != 3
                            or not len(f) or f.min() < 0 or f.max() >= len(v)):
                        raise ValueError('Invalid raw full mesh')
                    scene = pyrender.Scene(bg_color=[*np.asarray(BG)/255, 1], ambient_light=[.45]*3)
                    shader = material(TEAL if panel == 0 else AMBER); shader.doubleSided = True
                    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, f, process=False),
                                                        material=shader, smooth=False))
                    for eye, power in (([.6, -.8, .9], 2.2), ([-.6, .5, .5], 1.2)):
                        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=power),
                                  pose=look(np.array(eye), np.zeros(3)))
                    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(43), znear=.01, zfar=10))
                    scenes.append(scene); cameras.append(camera)
                records.append(dict(object_id=oid, reconstruction_available=meshes[2] is not None,
                                    video_start_s=frame_count/30, frames=96))
                for frame in range(96):
                    im = canvas('完整官方 Michelangelo AE · 动态网格错误监督',
                                f'{oid} | 真值表面 XYZ+法向输入 / 两轮各自完整模型1000步 / 仅改变volume采样 | 相机环绕，非物理 rollout',
                                ['真实完整网格', '静态覆盖v2：1000步', '动态网格监督：1000步'])
                    draw = ImageDraw.Draw(im)
                    angle = -np.pi/3 + 2*np.pi*frame/96
                    eye = np.array([.9*np.cos(angle), .9*np.sin(angle), .45])
                    for panel, (scene, camera) in enumerate(zip(scenes, cameras)):
                        x = 24 + 532*panel
                        if scene is None:
                            text(draw, (x, 360), '无可用 AE 重建', 28)
                            text(draw, (x, 410), '保留真实失败；未补造网格', 20)
                        else:
                            scene.set_pose(camera, look(eye, np.zeros(3)))
                            render_flags = flags
                            rgb, depth = renderer.render(scene, flags=render_flags)
                            if rgb.std() < 5 or not np.isfinite(depth).all() or np.count_nonzero(depth) < 100:
                                raise RuntimeError(f'Blank actual mesh render: {oid}/{frame}/{panel}')
                            im.paste(Image.fromarray(rgb), (8+532*panel, 144))
                        score = row.get('metrics', {}) if panel==2 else dict(old_rows[oid]['corrected'],occupancy_iou=old_rows[oid]['original']['occupancy_iou'])
                        if panel == 0:
                            text(draw, (x, 752), f'完整资产：{len(vertices)} 顶点 / {len(faces)} 面', 21)
                            text(draw, (x, 797), '真实表面采样4096点及面法向输入AE', 20)
                            text(draw, (x, 842), '本阶段允许GT输入，专测表示能力', 20)
                        elif score:
                            text(draw, (x, 752), f'CD×9000 {score["cd_x9000"]:.4f} / F@.01 {score["fscore"]:.1%}', 21)
                            text(draw, (x, 797), f'面积 {score["area_ratio"]:.2f}× / IoU {score["occupancy_iou"]:.1%}', 21)
                            text(draw, (x, 842), f'逐面7探针最大误差 {score["maximum_face_probe_distance"]:.4f}', 20)
                        else:
                            text(draw, (x, 797), '评价未完成，保留失败', 23)
                        if panel:
                            text(draw, (x, 883), '数值资格：'+('PASS' if (row['checks']['passed'] if panel==2 else old_rows[oid]['checks']['passed']) else 'FAIL'), 20)
                    text(draw, (24, 924),
                         '仅四物体完整表面输入的重建微调；不是触觉或泛化成功。两预测均修正提取网格坐标；未补洞、平滑或拟合对齐。', 16)
                    if frame in (0, 32, 64):
                        name = f'{oid}_view{frame:02d}.png'; im.save(out/name); stills.append(name)
                    writer.append_data(np.asarray(im)); frame_count += 1
                print('MICHELANGELO_RAW_RENDER', oid, frame_count, flush=True)
    finally:
        renderer.delete()
    decoded = 0
    with imageio.get_reader(movie) as reader:
        fps = reader.get_meta_data()['fps']
        for rgb in reader:
            if rgb.shape != (960, 1600, 3):
                raise ValueError('Bad decoded frame dimensions')
            decoded += 1
    if decoded != 384 or frame_count != 384 or fps != 30:
        raise ValueError('Incomplete four-case encoded render')
    actual_cases = sum(r['reconstruction_available'] for r in records)
    report = dict(complete=True, sampling_revision=protocol['sampling_revision'],
                  baseline_scope=old_result['scope'], baseline_checkpoint_sha256=old_result['checkpoint_sha256'],
                  checkpoint_sha256=result['checkpoint_sha256'], initial_checkpoint_sha256=CHECKPOINT_SHA, bindings=bindings, training_model_updates=1000, records=records,
                  stills=stills, actual_reconstruction_cases=actual_cases,
                  actual_mesh_stills=3*actual_cases, placeholder_stills=3*(4-actual_cases),
                  frames=frame_count, all_frames_decoded=True, fps=fps, renderer=device,
                  scope='Native full-surface full-model volume sampling comparison, camera turntable; not tactile prediction',
                  model_forwards=0, model_updates=0, physics_controls=0,
                  visual_inspection_required=True, representation_qualified=False)
    (out/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    html = ['<!doctype html><meta charset="utf-8"><title>Michelangelo 原生AE资格</title>',
            '<style>body{max-width:1600px;margin:25px auto;background:#f2f6f9;font:18px/1.6 system-ui}video,img{width:100%}</style>',
            '<h1>完整官方 AE：全部四个网格</h1><p>完整真实表面输入；静态覆盖v2与动态网格错误采样对比；两轮均从官方权重重置并完整训练1000步。相机环绕，不是触觉搬运。</p>',
            '<video controls src="renders/native_ae_dynamic_grid_comparison.mp4"></video>']
    for name in stills:
        html.append(f'<p>{name}</p><img loading="lazy" src="renders/{name}">')
    (root/'index.html').write_text('\n'.join(html))
    return 0 if actual_cases == 4 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        import pyrender, trimesh, imageio, PIL
        from .render_device import configure_retained_egl
        from .render_object_state import canvas, text, material, look
        print(json.dumps(dict(real_render_dependencies_imported=True, egl_created=False,
                              rendered_frames=0, model_forwards=0)))
        return 0
    if args.root is None:
        parser.error('--root required for separately scheduled rendering')
    return render(args.root)


if __name__ == '__main__':
    raise SystemExit(main())
