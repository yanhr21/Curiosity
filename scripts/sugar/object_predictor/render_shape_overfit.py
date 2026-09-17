"""Actual full-mesh before/after render of every fixed shape-overfit input.

Camera turntable, not a simulated rollout. No model inference or mesh fitting.
The separate carry/fixture render must show hands and physical trajectories.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def render(root):
    from .retained_execution import require_active_resource
    require_active_resource()
    os.environ['PYOPENGL_PLATFORM']='egl'
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import pyrender
    import trimesh
    from .render_device import configure_retained_egl
    from .render_object_state import canvas, text, material, look, PW, PH, BG, TEAL, AMBER

    result=json.loads((root/'RESULT.json').read_text())
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if not result['complete'] or not result['checkpoint_reload']['passed']:
        raise ValueError('Require completed fixed training and actual checkpoint reload')
    cases=protocol['cases']
    if len(cases)!=8 or len({r['name'] for r in cases})!=8:
        raise ValueError('Render all eight fixed inputs without selection')
    steps=int(protocol['fixed_updates'])
    continuation=protocol.get('continuation')
    update_label=(f'完整官方模型累计 {steps} 次更新 / 本次 {protocol["new_optimizer_updates"]} 次'
                  if continuation else f'完整官方模型 {steps} 次更新')
    if continuation:
        if (steps!=4000 or protocol['new_optimizer_updates']!=3000
                or result.get('optimizer_updates')!=3000 or result.get('total_optimizer_updates')!=4000
                or not result.get('resume_replay',{}).get('passed')):
            raise ValueError('Require declared actual 1000+3000 continuation with replay')
    before=json.loads((root/'step_0000/METRICS.json').read_text())
    after=json.loads((root/f'step_{steps:04d}/METRICS.json').read_text())
    metrics=[]; geometry_metrics=[]
    for saved in (before,after):
        metrics.append({r['name']:r['metrics'] for r in saved['records'] if r['condition']=='observed'})
        geometry_metrics.append({r['name']:r.get('geometry_metrics') for r in saved['records'] if r['condition']=='observed'})
    out=root/'renders';out.mkdir(exist_ok=False)
    device=configure_retained_egl()
    renderer=pyrender.OffscreenRenderer(PW,PH)
    native=result['scope']=='native_only'
    title='官方原生触摸' if native else 'SUGAR 实际触摸'
    if protocol.get('geometry_repair'):
        title += ' / 几何约束修复'
    movie=out/'shape_before_after.mp4'
    stills=[];records=[];frame_count=0
    try:
        with imageio.get_writer(movie,fps=30,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1) as writer:
            for row in cases:
                name=row['name'];saved=[]
                for stage in ('step_0000',f'step_{steps:04d}'):
                    with np.load(root/stage/f'{name}.npz',allow_pickle=False) as z:
                        saved.append({key:z[key] for key in z.files})
                a,b=saved
                for key in ('truth_vertices_canonical','truth_faces','input_touch_charts','global_faces'):
                    if not np.array_equal(a[key],b[key]):raise ValueError('Before/after input or label changed: '+name+'/'+key)
                meshes=[(a['truth_vertices_canonical'],a['truth_faces']),
                        (a['observed_vertices_canonical'][:1824],a['global_faces']),
                        (b['observed_vertices_canonical'][:1824],b['global_faces'])]
                scenes=[];cameras=[]
                for panel,(vertices,faces) in enumerate(meshes):
                    if not np.isfinite(vertices).all() or faces.max()>=len(vertices):raise ValueError('Invalid complete mesh')
                    scene=pyrender.Scene(bg_color=[*np.asarray(BG)/255,1],ambient_light=[.45,.45,.45])
                    shader=material(TEAL if panel==0 else AMBER);shader.doubleSided=True
                    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices,faces,process=False),material=shader,smooth=False))
                    for eye,power in (([.6,-.8,.9],2.2),([-.6,.5,.5],1.2)):
                        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=power),pose=look(np.array(eye),np.zeros(3)))
                    camera=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(43),znear=.01,zfar=10))
                    scenes.append(scene);cameras.append(camera)
                records.append(dict(name=name,video_start_s=frame_count/30,frames=96))
                for frame in range(96):
                    im=canvas(f'{title} · 完整形状 overfit 前后',
                        f'{name} | 同一固定输入 / {update_label} | 相机环绕，非物理 rollout',
                        ['真实完整网格','预训练初始预测','固定 overfit 终点预测'])
                    angle=-np.pi/3+2*np.pi*frame/96
                    eye=np.array([.9*np.cos(angle),.9*np.sin(angle),.45])
                    draw=ImageDraw.Draw(im)
                    for panel,(scene,camera) in enumerate(zip(scenes,cameras)):
                        scene.set_pose(camera,look(eye,np.zeros(3)))
                        rgb,depth=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
                        if rgb.std()<5 or not np.isfinite(depth).all() or np.count_nonzero(depth)<100:
                            raise RuntimeError(f'Empty actual mesh render {name}/{frame}/{panel}')
                        im.paste(Image.fromarray(rgb),(8+532*panel,144));x=24+532*panel
                        if panel==0:
                            text(draw,(x,752),f'原始完整资产 {len(meshes[0][0])} 顶点',22)
                            text(draw,(x,797),'固定 canonical 坐标；未配准',22)
                            count=int((a['input_touch_charts'][...,3]==2).sum())
                            text(draw,(x,842),f'五次尝试中接触顶点：{count}/125',21)
                        else:
                            score=metrics[panel-1][name]
                            text(draw,(x,752),f'全局 CD×9000：{score["global_cd_x9000"]:.4f}',22)
                            text(draw,(x,797),f'F@0.01：{score["global_fscore"]:.1%}',22)
                            text(draw,(x,842),f'官方全表面 loss：{score["official_cd_x9000"]:.4f}',21)
                        geometric = geometry_metrics[panel-1].get(name) if panel else None
                        if geometric is not None:
                            text(draw,(x,883),f'面积 {geometric["area_ratio"]:.2f}× / 面探针最大误差 {geometric["maximum_face_probe_label_distance"]:.4f}',17)
                        else:
                            text(draw,(x,883),'全局预测显示 1824 顶点 / 2304 面' if panel else '真值只用于监督、评价与本栏渲染',17)
                    footer = ('误差使用 canonical 单位；每面7探针不保证整面误差上界。保留原始预测，无平滑、配准或显示修正。'
                              if protocol.get('geometry_repair') else
                              '固定小样本形状验证；不是未知物体泛化、材质、质量或搬运成功。所有八组输入保留；没有真值对齐或显示时修正预测。')
                    text(draw,(24,924),footer,16)
                    if frame in (0,32,64):
                        filename=f'{name}_view{frame:02d}.png';im.save(out/filename);stills.append(filename)
                    writer.append_data(np.asarray(im));frame_count+=1
                print('SHAPE_OVERFIT_RENDER',name,frame_count,flush=True)
    finally:
        renderer.delete()
    decoded=0
    with imageio.get_reader(movie) as reader:
        fps=reader.get_meta_data()['fps']
        for rgb in reader:
            if rgb.shape!=(960,1600,3):raise ValueError('Bad decoded video dimensions')
            for panel in range(3):
                if rgb[144:724,8+532*panel:528+532*panel].std()<5:raise ValueError('Blank decoded panel')
            decoded+=1
    if decoded!=768 or decoded!=frame_count or fps!=30:raise ValueError('Incomplete encoded turntable')
    summary=dict(complete=True,actual_mesh_stills=len(stills),frames=frame_count,fps=fps,
        all_frames_decoded=True,records=records,stills=stills,renderer=device,
        checkpoint_sha256=result['checkpoint_sha256'],
        scope='Camera turntable of actual saved shape predictions, not a physical hand rollout',
        new_model_forwards=0,new_physics_controls=0,new_optimizer_updates=0)
    if continuation:
        summary.update(training_total_optimizer_updates=steps, training_new_optimizer_updates=protocol['new_optimizer_updates'],
                       rendered_endpoint=f'step_{steps:04d}', initial_snapshot='copied original released step_0000')
    (out/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>形状 overfit 前后</title>',
        '<style>body{max-width:1600px;margin:25px auto;background:#f2f6f9;font:18px/1.6 system-ui}video,img{width:100%}</style>',
        f'<h1>{title}：全部八组输入</h1><p>实际网格相机环绕。此视频不是手部物理 rollout；原生形状与 SUGAR 迁移单独验收。</p>',
        '<video controls src="renders/shape_before_after.mp4"></video>']
    for filename in stills:html.append(f'<p>{filename}</p><img loading="lazy" src="renders/{filename}">')
    (root/'index.html').write_text('\n'.join(html))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    render(parser.parse_args().root)
