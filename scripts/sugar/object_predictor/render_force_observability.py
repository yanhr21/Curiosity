"""Actual saved meshes with diagnostic vertical-force arrows; no prediction."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
import trimesh,pyrender
from .render_device import configure_retained_egl
from .render_object_state import World,load,true_state,canvas,text,mat,material,PW,PH,AMBER
from .audit_force_observability import MODES


def arrow(world,start,force_z,color):
    length=abs(force_z)*.05
    if length<1e-6:return []
    direction=np.array([0.,0.,1. if force_z>=0 else -1.])
    quat=Rotation.align_vectors(direction[None],np.array([[0.,0.,1.]]))[0].as_quat()
    pieces=[(trimesh.creation.cylinder(radius=.003,height=length),start+direction*length*.5),
            (trimesh.creation.cone(radius=.009,height=.022),start+direction*length)]
    return [world.scene.add(pyrender.Mesh.from_trimesh(mesh,material=material(color)),pose=mat(pos,quat)) for mesh,pos in pieces]


def main():
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.output);p=json.loads((root/'PROTOCOL.json').read_text());spec=json.loads((root/'VISUAL_SCOPE.json').read_text())
    configure_retained_egl();world=World();renderer=pyrender.OffscreenRenderer(PW,PH);records=[]
    for primitive in world.object_node.mesh.primitives:
        primitive.material.baseColorFactor=np.array([.1,.7,.75,.28]);primitive.material.alphaMode='BLEND';primitive.material.doubleSided=True
    try:
        for ep in spec['episodes']:
            a=load(Path(p['source'])/'cases'/f'episode_{ep}'/f'episode_{ep}.npz');d=load(root/f'episode_{ep}.npz')
            frame=spec['frame'];idx=np.flatnonzero(d['frame']==frame)
            if len(idx)!=1:raise ValueError('Unmatched render clock')
            idx=int(idx[0]);c,q,dims,mass=true_state(a,frame)
            im=canvas('实际已搬起，但几何法向不等于受力方向',
                      f'案例 {ep} | {a["timestamp_s"][frame]:.2f} 秒 | 真值 {mass:.3f} kg | 离地 {a["validation_full_mesh_min_z_m"][frame]*100:.2f} cm | 力学诊断，无学习模型',
                      ['仅掌侧剪切力','CAD 法向载荷＋剪切力','拟合界面法向载荷＋剪切力'])
            rec=dict(episode=ep,frame=frame,values=[])
            for panel,mode in enumerate(MODES):
                fz=float(d[mode+'_on_object_w'][idx,2]);world.set(a['hand_pose_w'][frame],c,q,dims,a['contact_position_w'][frame],a['normal_load_n'][frame])
                world.camera_at(c+[.75,-1.25,.60],c+[0.,0.,.06])
                nodes=arrow(world,c+[-.25,0.,.10],fz,AMBER)
                nodes+=arrow(world,c+[.25,0.,.18],-mass*p['gravity_m_s2'],(90,100,115))
                rgb,_=renderer.render(world.scene,flags=0)
                if rgb.std()<5:raise RuntimeError('Constant frame')
                im.paste(Image.fromarray(rgb),(8+panel*532,144));draw=ImageDraw.Draw(im);x=24+panel*532
                eq=fz/p['gravity_m_s2']
                text(draw,(x,747),f'计算竖向支持力 {fz:+.3f} N',25,AMBER)
                text(draw,(x,795),f'支持等效质量 {eq:+.3f} kg',24)
                text(draw,(x,838),f'相对真实质量误差 {abs(eq-mass)/mass*100:.1f}%',22)
                text(draw,(x,879),'橙箭头：支持分量；灰箭头：真实重力',19)
                rec['values'].append(dict(mode=mode,force_z_n=fz,signed_support_mass_kg=eq))
                for node in nodes:world.scene.remove_node(node)
            text(ImageDraw.Draw(im),(24,925),'各面板均为同一真实网格与手姿态。箭头统一0.05米/牛，平移到物体两侧便于观察，不代表真实作用点。',18)
            image=root/f'force_balance_{ep}.png';im.save(image);rec['image']=str(image);records.append(rec)
    finally:renderer.delete()
    (root/'RENDER.json').write_text(json.dumps(dict(complete=True,records=records,selection=spec),indent=2))
    print('FORCE_RENDER_COMPLETE',json.dumps(records),flush=True)


if __name__=='__main__':main()
