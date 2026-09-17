"""Render real archived meshes/poses and frozen predictions using OpenGL EGL.

No dynamics, interpolated predictions, learned shape generator, or policy update.
The cyan dashed cuboid is explicitly an oriented extent guide, not a replacement
for the complete scanned mesh, which is rendered without decimation.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import pyrender
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/object_predictor_v1'
OUT = BASE/'rendered_carry_v1'
NATIVE = ROOT/'experiments/isaaclab_g1_anatomical27_object_demos/carrybox_plain_longx1p6_native_v1'
URDF = ROOT/'SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TEAL = (19, 161, 171)
AMBER = (231, 143, 43)
INK = (27, 39, 54)
BG = (242, 246, 249)
W, H = 1600, 960
PW, PH = 520, 580


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def mat(position, quat=None, scale=None):
    p=np.eye(4)
    if quat is not None:p[:3,:3]=Rotation.from_quat(quat).as_matrix()
    if scale is not None:p[:3,:3]=p[:3,:3]@np.diag(scale)
    p[:3,3]=position
    return p


def look(eye, target):
    z=np.asarray(eye)-target;z=z/np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    p=np.eye(4);p[:3,:3]=np.stack([x,y,z],axis=1);p[:3,3]=eye
    return p


def material(color):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*(np.array(color)/255),1],metallicFactor=.12,roughnessFactor=.65)


def mesh_node(scene, mesh, color):
    return scene.add(pyrender.Mesh.from_trimesh(mesh,material=material(color),smooth=False))


def text(draw, xy, value, size=23, color=INK):
    draw.text(xy, value,font=ImageFont.truetype(FONT,size),fill=color)


class World:
    def __init__(self, full=False, names=None, color=TEAL, ground_height=None):
        self.scene=pyrender.Scene(bg_color=[*np.array(BG)/255,1],ambient_light=[.43,.43,.43])
        self.camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(39),znear=.01,zfar=30)
        self.camera_node=self.scene.add(self.camera)
        self.robot=[]
        for link in ET.parse(URDF).getroot().findall('link'):
            name=link.attrib['name']
            if not full and name not in ('left_rubber_hand','right_rubber_hand'):continue
            if full and name not in names:continue
            for visual in link.findall('visual'):
                m=visual.find('geometry/mesh')
                if m is None:continue
                mesh=trimesh.load(URDF.parent/m.attrib['filename'],force='mesh',process=False)
                mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '))
                origin=visual.find('origin');tf=np.eye(4)
                if origin is not None:
                    tf[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                    tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                mc=visual.find('material');dark=mc is not None and mc.get('name')=='dark'
                shade=(65,78,93) if dark else (187,198,209)
                if 'rubber_hand' in name:shade=(64,104,153)
                node=mesh_node(self.scene,mesh,shade)
                self.robot.append((name,node,tf))
        obj=load(OUT/'object_mesh.npz');v=obj['vertices'];self.dims=np.ptp(v,axis=0)
        v=v-(v.max(0)+v.min(0))/2
        object_mesh=trimesh.Trimesh(v,obj['faces'],process=False)
        self.object_node=mesh_node(self.scene,object_mesh,color)
        # The released scan is oblique in its asset axes. An axis-aligned label
        # cuboid is much larger than the visible box. Use a tight oriented
        # mesh-bound guide for visual comparison, retaining original dimensions
        # and complete mesh for the actual object and learned scale transform.
        bounds=object_mesh.bounding_box_oriented
        self.guide_vertices=np.asarray(bounds.vertices)
        self.guide_edges=bounds.face_adjacency_edges[bounds.face_adjacency_angles>.2]
        ground_center_z = -.025 if ground_height is None else ground_height - .015/2
        grid_center_z = -.016 if ground_height is None else ground_height - .0004
        ground=trimesh.creation.box(extents=[5,5,.015]);ground.apply_translation([.6,.3,ground_center_z])
        mesh_node(self.scene,ground,(220,227,232))
        # Ground grid and cast shadows make the actual lift visible.
        for axis in (0,1):
            for t in np.arange(-1.5,2.6,.25):
                ext=[4,.003,.001] if axis==0 else [.003,4,.001]
                p=[.5,t,grid_center_z] if axis==0 else [t,.5,grid_center_z]
                line=trimesh.creation.box(extents=ext);line.apply_translation(p)
                mesh_node(self.scene,line,(191,204,214))
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.4),pose=look([2,-3,5],[.5,.3,.5]))
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.1),pose=look([-2,1,3],[.5,.3,.5]))
        self.dots=[]
        for _ in range(54):
            n=mesh_node(self.scene,trimesh.creation.icosphere(subdivisions=1,radius=.005),(238,75,64))
            self.dots.append(n)
        self.pose=None

    def set(self, hand, center, quat, dims, sites, loads, bodies=None, names=None):
        for name,node,tf in self.robot:
            if bodies is not None:
                pose=bodies[names.index(name),:7];q=pose[[4,5,6,3]]
                p=mat(pose[:3],q)
            else:
                pose=hand[0 if name.startswith('left') else 1];p=mat(pose[:3],pose[3:])
            self.scene.set_pose(node,p@tf)
        self.scene.set_pose(self.object_node,mat(center,quat,dims/self.dims))
        for j,node in enumerate(self.dots):
            pos=sites[j] if loads[j]>.001 else np.array([0,0,-20])
            self.scene.set_pose(node,mat(pos,scale=np.full(3,1+min(loads[j],8)*.13)))

    def camera_at(self, eye, target):
        self.pose=look(eye,np.asarray(target));self.scene.set_pose(self.camera_node,self.pose)

    def render(self, renderer):
        color,depth=renderer.render(self.scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        if not np.isfinite(depth).all() or np.count_nonzero(depth)<100:raise ValueError('Empty render')
        return Image.fromarray(color)

    def extent_guide(self, image, center, quat, dims):
        world=Rotation.from_quat(quat).apply(self.guide_vertices*(dims/self.dims))+center
        view=np.linalg.inv(self.pose)
        clip=(self.camera.get_projection_matrix(PW,PH)@view@np.c_[world,np.ones(8)].T).T
        xy=(clip[:,:2]/clip[:,3:]+1)*np.array([PW/2,PH/2]);xy[:,1]=PH-xy[:,1]
        d=ImageDraw.Draw(image)
        for a,b in self.guide_edges:
            for t in np.arange(0,1,.10):
                u=xy[a]*(1-t)+xy[b]*t;v=xy[a]*(1-min(1,t+.055))+xy[b]*min(1,t+.055)
                d.line([tuple(u),tuple(v)],fill=TEAL,width=2)


def true_state(data,i):
    p=data['object_pose_w'][i]
    c=Rotation.from_quat(p[3:]).apply(data['object_local_center_m'][i])+p[:3]
    return c,p[3:],data['object_dimensions_m'][i],float(data['object_mass_kg'][i])


def dimensions(d):return ' × '.join(f'{v*100:.1f}' for v in d)+' cm'


def canvas(title,subtitle,labels):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(24,12),title,34)
    text(d,(24,62),subtitle,20,(88,101,116))
    for k,label in enumerate(labels):text(d,(16+k*532,104),label,25,TEAL if k<2 else AMBER)
    return im


def carry(preview=False):
    data=load(BASE/'isaac_saved_r1/episode_1000.npz')
    native=load_native()
    pred=load(OUT/'carry_predictions.npz');errors=load(OUT/'carry_errors.npz')
    names=native['robot_body_names'].tolist()
    worlds=[World(True,names),World(),World(color=AMBER)]
    renderer=pyrender.OffscreenRenderer(PW,PH)
    samples=[0,31,50,79] if preview else list(range(80))
    writer=None if preview else imageio.get_writer(OUT/'carry_state_estimation.mp4',fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1)
    try:
        for i in samples:
            c,q,dims,mass=true_state(data,i)
            im=canvas('双手搬运 · 物体状态估计', '真实官方策略记录回放  |  当前 Utonia 冻结模型  |  跨后端测试  |  0.25× 慢放',
                      ['实际机器人 / 完整网格','真实物体 / 手部近景','模型估计 / 同一视角'])
            for k,world in enumerate(worlds):
                est=i>=31 and k==2
                j=i-31
                wc,wq,wd=(pred['center_w_m'][j],pred['object_rotation_w_xyzw'][j],pred['dimensions_m'][j]) if est else (c,q,dims)
                world.set(data['hand_pose_w'][i],wc,wq,wd,data['contact_position_w'][i],data['normal_load_n'][i],
                          native['robot_body_state_w'][i] if k==0 else None,names)
                if k==0:
                    world.camera_at(np.array([1.65,-1.85,1.55]),[.72,.32,.7])
                else:world.camera_at(c+np.array([.92,-1.28,.62]),c)
                if k==2 and i<31:
                    panel=Image.new('RGB',(PW,PH),BG);pd=ImageDraw.Draw(panel)
                    text(pd,(70,240),'正在积累 32 帧历史',26)
                    text(pd,(110,283),f'{i+1} / 32 帧',25)
                else:
                    panel=world.render(renderer)
                    if k==2:world.extent_guide(panel,c,q,dims)
                im.paste(panel,(8+k*532,144))
            draw=ImageDraw.Draw(im)
            elapsed=float(data['timestamp_s'][i]-data['timestamp_s'][0])
            lift=float(data['object_pose_w'][i,2]-data['object_pose_w'][0,2])
            text(draw,(24,737),f'记录时间 {elapsed:.2f} s   |   抬升 {lift*100:.1f} cm   |   帧 {i+1}/80',24)
            text(draw,(24,778),'真值重量',20,TEAL);text(draw,(24,809),f'{mass:.3f} kg',36,TEAL)
            text(draw,(250,778),'真值尺寸（资产坐标轴范围）',20,TEAL);text(draw,(250,816),dimensions(dims),23,TEAL)
            if i>=31:
                text(draw,(800,778),'预测重量',20,AMBER);text(draw,(800,809),f"{pred['mass_kg'][j]:.3f} kg",36,AMBER)
                text(draw,(1050,778),'预测尺寸（资产坐标轴范围）',20,AMBER);text(draw,(1050,816),dimensions(pred['dimensions_m'][j]),23,AMBER)
                text(draw,(800,867),f"位置误差 {errors['center_cm'][j]:.1f} cm   旋转 {errors['rotation_deg'][j]:.1f}°   重量 {errors['mass_pct'][j]:.1f}%",22,(184,65,54))
            text(draw,(24,867),f"记录的双手法向载荷  L {data['normal_load_n'][i,:27].sum():.1f} N / R {data['normal_load_n'][i,27:].sum():.1f} N",21)
            text(draw,(24,914),'青色：真值    橙色：预测    红点：触觉接触    青色虚线：真实网格定向包围轮廓；预测未平滑、未用真值修正',19,(85,99,113))
            if i in (0,31,50,79):im.save(OUT/f'carry_frame_{i:03d}.png')
            if writer is not None:
                for _ in range(2):writer.append_data(np.asarray(im))
            print('RENDER_CARRY',i,flush=True)
    finally:
        if writer:writer.close()
        renderer.delete()
    return dict(source_frames=80,video_frames=160,fps=25,duration_s=6.4,playback_speed=.25,robot_visual_meshes=len(worlds[0].robot))


def load_native():
    with np.load(NATIVE/'whole_hand_trace.npz') as z:
        return {k:z[k] for k in ('robot_body_state_w','robot_body_names')}


def state_from_saved(p,hand):
    from .shape_metric import rotation
    rh=Rotation.from_quat(hand[0,3:]).as_matrix()
    return rh@p[:3]+hand[0,:3],Rotation.from_matrix(rh@rotation(p[3:9])).as_quat(),np.exp(p[9:12]),float(np.exp(p[12]))


def support(preview=False):
    ev=BASE/'evaluation_mixed_geometry_mass_surface_v1/mixed'
    saved=[load(ev/m/'predictions.npz') for m in ('geometry','geometry_contact_force')]
    worlds=[World(),World(color=AMBER),World(color=AMBER)]
    renderer=pyrender.OffscreenRenderer(PW,PH)
    writer=None if preview else imageio.get_writer(OUT/'support_mass_comparison.mp4',fps=10,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1)
    count=0
    try:
        for ep in range(30,36):
            data=load(BASE/f'mixed_geometry_mass_surface_v1/episode_{ep:04d}.npz')
            indices=np.flatnonzero(saved[0]['episode']==ep)
            if preview:indices=indices[[-1]]
            for idx in indices:
                i=int(saved[0]['frame'][idx]);assert saved[1]['episode'][idx]==ep and saved[1]['frame'][idx]==i
                truth=true_state(data,i);c,q,dims,mass=truth
                states=[truth]+[state_from_saved(a['prediction'][idx],data['hand_pose_w'][i]) for a in saved]
                im=canvas('托举测试 · 触觉是否帮助估计重量？',f'全部 6 个 TEST 物体依次播放  |  当前第 {ep-29}/6 个  |  固定双手托举夹具，非全身搬运策略  |  1×',
                          ['真实物体 / 实际双手','只用手部几何 / 估计','几何 + 接触 + 力 / 估计'])
                for k,(world,state) in enumerate(zip(worlds,states)):
                    wc,wq,wd,wm=state
                    world.set(data['hand_pose_w'][i],wc,wq,wd,data['contact_position_w'][i],data['normal_load_n'][i])
                    world.camera_at(c+np.array([1.1,-1.3,-.20]),c+np.array([0,0,-.08]))
                    panel=world.render(renderer)
                    if k:world.extent_guide(panel,c,q,dims)
                    im.paste(panel,(8+k*532,144))
                    draw=ImageDraw.Draw(im)
                    text(draw,(24+k*532,747),'真实重量' if k==0 else '预测重量',23,TEAL if k==0 else AMBER)
                    text(draw,(24+k*532,784),f'{wm:.3f} kg',42,TEAL if k==0 else AMBER)
                    text(draw,(24+k*532,847),f'重量误差  {abs(wm/mass-1)*100:.1f}%' if k else f'记录时间 {data["timestamp_s"][i]:.2f} s',23)
                draw=ImageDraw.Draw(im)
                text(draw,(24,909),'实际完整箱体网格 + 双手网格    红点：有载荷的触觉位置    青色虚线：真实范围    逐个展示全部测试例，包含失败',20,(85,99,113))
                if i==int(saved[0]['frame'][np.flatnonzero(saved[0]['episode']==ep)[-1]]):im.save(OUT/f'support_{ep:04d}.png')
                if writer:writer.append_data(np.asarray(im));count+=1
            print('RENDER_SUPPORT',ep,flush=True)
    finally:
        if writer:writer.close()
        renderer.delete()
    return dict(episodes=list(range(30,36)),video_frames=count,fps=10,sampling='All 34 original saved evaluation windows per episode; one frame each; 100 ms apart')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--kind',choices=['carry','support','both'],default='both');args=ap.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Use retained compute step')
    result={}
    if args.kind in ('carry','both'):result['carry']=carry(args.preview)
    if args.kind in ('support','both'):result['support']=support(args.preview)
    if not args.preview:
        result.update(renderer='pyrender OpenGL EGL; full released URDF visual meshes and complete scanned positive outer object shell',physics_steps=0,optimizer_updates=0,predictions_interpolated=False,
                      object_mesh_vertices=15626,object_mesh_faces=31248)
        (OUT/f'RENDER_{args.kind}.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
