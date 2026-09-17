"""Render actual full hands and the original decoder's predicted mesh surfaces."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyrender
import trimesh

from .render_device import configure_retained_egl
from .render_object_state import World, load, true_state, canvas, text, material, PW, PH, TEAL, AMBER, BG


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Retained explicit compute step required')
    report=json.loads((root/'RESULT.json').read_text())
    assert report['complete'] and len(report['cases'])==16
    configure_retained_egl()
    out=root/'renders'; out.mkdir(exist_ok=False)
    records=json.loads(Path('experiments/object_predictor_v1/response_surface_dataset_v1/COLLECTION_RESULT.json').read_text())['records']
    records={r['episode']:r for r in records}
    inputs=Path(report['protocol']['inputs'])
    world=World(ground_height=0.)
    original_mesh=world.object_node.mesh
    renderer=pyrender.OffscreenRenderer(PW,PH)
    images=[]
    try:
        for row in report['cases']:
            ep,frame,arm=row['episode'],row['frame'],row['arm']
            trace=load(Path(records[ep]['source'])/f'episode_{ep}.npz')
            prediction=load(root/row['output_file'])
            sensed=load(inputs/f'episode_{ep}_frame_{frame}.npz')
            truth=true_state(trace,frame)
            target=true_state(trace,0)[0]+np.array([0.,0.,.10])
            title='官方 Active3D · 实际触觉形状重建诊断'
            subtitle=f'{arm} | TRAIN {ep} | {trace["timestamp_s"][frame]:.2f} s | {row["actual_charts"]} 张局部触觉网格'
            im=canvas(title,subtitle,['实际物体与双手','完整模型 · chart 置零','完整模型 · 实测 chart'])
            for panel,condition in enumerate((None,'empty','observed')):
                world.object_node.mesh=original_mesh
                world.set(trace['hand_pose_w'][frame],*truth[:3],trace['contact_position_w'][frame],trace['normal_load_n'][frame])
                if condition:
                    mesh=trimesh.Trimesh(prediction[condition+'_vertices_world_m'],prediction['faces'],process=False)
                    shader=material(AMBER); shader.doubleSided=True
                    world.object_node.mesh=pyrender.Mesh.from_trimesh(mesh,material=shader,smooth=False)
                    world.scene.set_pose(world.object_node,np.eye(4))
                world.camera_at(target+[.8,-1.3,.5],target)
                rgb,depth=renderer.render(world.scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
                if rgb.std()<5 or not np.isfinite(depth).all() or np.count_nonzero(depth)<100:
                    raise RuntimeError(f'Empty or invalid render {ep}/{frame}/{arm}/{condition}')
                image=Image.fromarray(rgb)
                if condition:
                    world.extent_guide(image,*truth[:3])
                im.paste(image,(8+panel*532,144))
                draw=ImageDraw.Draw(im); x=24+panel*532
                if condition:
                    metrics=row['metrics'][condition]
                    text(draw,(x,752),f'双向表面距离 {metrics["sampled_symmetric_surface_cm"]:.2f} cm',23)
                    text(draw,(x,797),f'1 cm F-score {metrics["fscore_1cm"]:.3f}',23)
                    text(draw,(x,842),f'预测表面最低点 {metrics["minimum_world_z_m"]*100:.2f} cm',21)
                    text(draw,(x,879),'1824 顶点 / 2304 面，网络直接输出',19)
                else:
                    text(draw,(x,752),f'实际质量 {truth[3]:.3f} kg',25)
                    text(draw,(x,797),f'完整网格离地 {trace["validation_full_mesh_min_z_m"][frame]*100:.1f} cm',23)
                    text(draw,(x,842),'真值只用于评价与显示',23)
                    text(draw,(x,879),f'已知网格 Utonia 参考 {row["known_mesh_utonia_reference"]["sampled_symmetric_surface_cm"]:.2f} cm',18)
            text(ImageDraw.Draw(im),(24,920),'青色轮廓为评价真值。两条件均用含触觉的 Utonia 位姿/尺度；中栏仅将形状解码器的 chart 置零。',16)
            path=out/f'{arm}_episode_{ep}_frame_{frame}.png'; im.save(path); images.append(path)
            print('RENDERED_OFFICIAL_SHAPE',arm,ep,frame,flush=True)
    finally:
        renderer.delete()
    sheet=Image.new('RGB',(1600,480*8),BG)
    for i,path in enumerate(images):
        with Image.open(path) as image:
            sheet.paste(image.resize((800,480)),((i%2)*800,(i//2)*480))
    sheet.save(out/'all_sixteen.jpg')
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,images=[str(p) for p in images],
        actual_mesh_stills=len(images),scope='16 fixed-frame renders, not a continuous video'),indent=2)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>Active3D 实际重建</title>',
          '<style>body{max-width:1600px;margin:24px auto;background:#f2f6f9;color:#172e40;font:18px/1.6 system-ui}img{width:100%}a{color:#087e88}</style>',
          '<h1>官方 Active3D：真实接触／空触觉对照</h1>',
          '<p>固定 8 个 TRAIN 时刻 × 两个完整官方模型。左右对照共用预测位姿/尺度；全部案例保留。当前为输入适配诊断。</p>',
          '<p><a href="REPORT.md">实验报告</a> · <a href="RESULT.json">全部指标</a></p>']
    for path in images:
        html.append(f'<h2>{path.stem}</h2><img loading="lazy" src="renders/{path.name}">')
    (root/'index.html').write_text('\n'.join(html)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
