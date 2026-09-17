"""Full-mesh contact layout at fixed saved frames; no learned inference."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
import trimesh,pyrender
from .render_device import configure_retained_egl
from .render_object_state import World,load,true_state,canvas,text,mat,PW,PH


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Use retained compute step')
    configure_retained_egl()
    source=Path(args.source);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    ep=json.loads((source/'RESULT.json').read_text())['episode']
    data=load(source/f'episode_{ep}.npz');field=load(source/'contact_surface.npz');frame=args.frame
    lo,hi=field['offset'][frame:frame+2];hand=field['hand'][lo:hi];pad=field['pad'][lo:hi]
    area=field['area_m2'][lo:hi];pressure=field['normal_pressure_pa'][lo:hi]
    pos=field['position_hand_frame_m'][lo:hi];poses=data['hand_pose_w'][frame]
    world=World();renderer=pyrender.OffscreenRenderer(PW,PH)
    try:
        assigned=pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1,radius=.0004),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,.02,.01,1],emissiveFactor=[.4,0,0]))
        uncovered=pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1,radius=.0004),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,.7,0,1],emissiveFactor=[.3,.15,0]))
        nodes=[]
        for side in [0,1]:
            select=(hand==side)&(pressure>0)
            points=Rotation.from_quat(poses[side,3:]).apply(pos[select])+poses[side,:3]
            for point,p in zip(points,pad[select]):nodes.append(world.scene.add(assigned if p>=0 else uncovered,pose=mat(point)))
        c,q,dims,mass=true_state(data,frame)
        im=canvas(f'{ep} · 实际接触布局',f'保存轨迹 {data["timestamp_s"][frame]:.2f} 秒 | 红：记录掌侧接触，黄：未覆盖接触（仅诊断）', ['实际箱体与双手','左手完整掌面 / 隐去物体','右手完整掌面 / 隐去物体'])
        for panel in range(3):
            world.set(poses,c,q,dims,data['contact_position_w'][frame],np.zeros(54))
            if panel==0:world.camera_at(c+[.7,-1.2,.5],c)
            else:
                side=panel-1;rot=Rotation.from_quat(poses[side,3:]);center=poses[side,:3]+rot.apply([.065,0,0])
                eye=center+rot.apply([.015,-.30 if side==0 else .30,.045])
                world.camera_at(eye,center)
                world.scene.set_pose(world.object_node,mat([0,0,-20]))
                for name,node,_ in world.robot:
                    if name.startswith('right' if side==0 else 'left'):world.scene.set_pose(node,mat([0,0,-20]))
            rgb,_=renderer.render(world.scene,flags=0)
            if rgb.std()<5:raise RuntimeError('Constant RGB')
            im.paste(Image.fromarray(rgb),(8+panel*532,144));draw=ImageDraw.Draw(im)
            if panel==0:
                text(draw,(24,755),f'网格离地 {100*data["validation_full_mesh_min_z_m"][frame]:.2f} cm',25)
                text(draw,(24,800),'完整原始手与物体网格',24)
            else:
                side=panel-1;sel=(hand==side)&(pressure>0)&(pad>=0)
                loads=data['normal_load_n'][frame].reshape(2,27)[side]
                total=(area[(hand==side)]*pressure[(hand==side)]).sum();recorded=loads.sum()
                text(draw,(24+panel*532,755),f'有效区域 {int((loads>.01).sum())} / 掌侧 {recorded:.2f} N',25)
                text(draw,(24+panel*532,798),f'面积 {1e6*area[sel].sum():.2f} mm²',24)
                text(draw,(24+panel*532,840),f'未覆盖的分配载荷 {total-recorded:.2f} N',23)
        text(ImageDraw.Draw(im),(24,914),'点标记半径 0.4 mm，仅便于查看；不代表接触面积。黄色接触与物体真值不进入 predictor。',21)
        im.save(out);print('RENDERED',str(out),flush=True)
    finally:renderer.delete()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--output',required=True);main(p.parse_args())
