"""Actual matched rollout: per-hand quotas versus narrowly triggered shared total CoP travel.

Full original hands/object, fixed failure/regression trio. No predictor output.
"""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse
import json
import hashlib
import trimesh
from scipy.spatial.transform import Rotation
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import pyrender

from .render_device import configure_retained_egl
from .render_object_state import World,load,true_state,text


def main(root):
    from .retained_execution import require_active_resource
    require_active_resource()
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    result=json.loads((root/'RESULT.json').read_text()); assert result['complete']
    if protocol['controller_intervention'] != 'shared_budget_cop_v1' or [r['episode'] for r in result['cases']] != [5014,5012,5000]:
        raise ValueError('Require the fixed completed pressure-center alignment trio')
    baseline=Path(protocol['baseline_root'])
    baseline_protocol=json.loads((baseline/'PROTOCOL.json').read_text())
    if baseline_protocol['controller_intervention']!='allocated_floor_cop_v1':
        raise ValueError('Require immediate complete allocated-floor baseline')
    if not json.loads((baseline/'RESULT.json').read_text())['complete']:
        raise ValueError('Incomplete allocated-floor baseline')
    saved={}
    for episode in (5014,5012,5000):
        saved[episode]=[load(source/'cases'/f'episode_{episode}'/f'episode_{episode}.npz') for source in (baseline,root)]
        for data in saved[episode]:
            if len(data['timestamp_s'])!=2400 or not np.allclose(data['timestamp_s'],.02*np.arange(1,2401),rtol=0,atol=1e-8):
                raise ValueError('Incomplete original 2400 control clock')
        if not np.array_equal(saved[episode][0]['timestamp_s'],saved[episode][1]['timestamp_s']):
            raise ValueError('Mismatched comparison clocks')
    mesh_paths=[Path('SUGAR/descriptions/robots/g1/meshes')/(side+'_rubber_hand.STL') for side in ('left','right')]
    meshes=[np.asarray(trimesh.load(p,force='mesh').vertices,np.float32).astype(float) for p in mesh_paths]
    heights={};bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in mesh_paths}
    for episode,datasets in saved.items():
        heights[episode]=[]
        for panel,data in enumerate(datasets):
            poses=data['hand_pose_w'];minimum=np.zeros((2400,2))
            for side,vertices in enumerate(meshes):
                axes=Rotation.from_quat(poses[:,side,3:]).as_matrix()[:,2,:]
                for lo in range(0,2400,100):minimum[lo:lo+100,side]=(axes[lo:lo+100]@vertices.T).min(1)+poses[lo:lo+100,side,2]
            companion=(baseline,root)[panel]/'cases'/f'episode_{episode}'/'HAND_FLOOR_SUBSTEPS.npz'
            sub=load(companion)
            if not np.array_equal(sub['actual_hand_pose_w'][7::8],poses) or not np.allclose(sub['actual_min_z_m'][7::8],minimum,rtol=0,atol=1e-12):raise ValueError('Displayed hand-floor minima differ from actual substep evidence')
            bindings[str(companion)]=hashlib.sha256(companion.read_bytes()).hexdigest()
            source=(baseline,root)[panel]/'cases'/f'episode_{episode}'/f'episode_{episode}.npz'
            bindings[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            heights[episode].append(minimum)
    configure_retained_egl()
    out=root/'renders';out.mkdir(exist_ok=False)
    frame_dir=out/'frames';frame_dir.mkdir()
    world=World(ground_height=0.);renderer=pyrender.OffscreenRenderer(532,560)
    frames=list(range(0,2400,10));paths=[];records=[]
    try:
        for case in result['cases']:
            episode=case['episode'];datasets=saved[episode]
            target=true_state(datasets[0],0)[0]+[0,0,.08]
            # Side-on to the initial hand separation: expose both hands instead
            # of hiding the farther hand behind the opaque object. Shared camera
            # is derived only from the baseline initial hand poses.
            separation=datasets[0]['hand_pose_w'][0,1,:3]-datasets[0]['hand_pose_w'][0,0,:3]
            separation=separation.copy();separation[2]=0.
            if np.linalg.norm(separation)<1e-6:raise ValueError('Degenerate initial hand separation')
            tangent=np.cross(separation/np.linalg.norm(separation),[0.,0.,1.])
            eye=target+1.3*tangent+[0.,0.,.55]
            for frame in frames+[2399]:
                im=Image.new('RGB',(1088,928),'#f2f6f9');draw=ImageDraw.Draw(im)
                text(draw,(16,12),f'{episode} · 共享调整行程：固定三例对照',28)
                text(draw,(16,53),'真实完整网格回放｜双手总行程上限仍200mm｜地面、载荷及截止不变｜无模型预测',19)
                timestamp=float(datasets[0]['timestamp_s'][frame])
                for panel,data in enumerate(datasets):
                    x=8+panel*544
                    text(draw,(x+8,91),'原每手100mm配额' if panel==0 else '受阻手配额转给安全手',22)
                    center,quaternion,dimensions,_=true_state(data,frame)
                    world.set(data['hand_pose_w'][frame],center,quaternion,dimensions,
                              data['contact_position_w'][frame],data['normal_load_n'][frame])
                    world.camera_at(eye,target)
                    rgb,depth=renderer.render(world.scene,flags=0)
                    if rgb.std()<5 or not np.isfinite(depth).all() or np.count_nonzero(depth)<100:
                        raise RuntimeError(f'Invalid render {episode}/{frame}/{panel}')
                    im.paste(Image.fromarray(rgb),(x,126))
                    loads=data['normal_load_n'][frame].reshape(2,27).sum(1)
                    gains=data['validation_controller_response_gain_m_per_ns'][frame]
                    gain_text=' / '.join(f'{g:.2e}' for g in gains)
                    text(draw,(x+8,694),f'{timestamp:05.2f}s  手载荷 {loads[0]:.2f} / {loads[1]:.2f} N',20)
                    text(draw,(x+8,728),f'完整物体离地 {data["validation_full_mesh_min_z_m"][frame]*100:.2f} cm',20)
                    travel=data['validation_controller_cop_travel_m'][frame]*1000.
                    text(draw,(x+8,762),f'调整行程 L/R {travel[0]:.2f} / {travel[1]:.2f} mm',19)
                    ready_text=f'持续就绪 {data["validation_controller_ready_seconds"][frame]:.2f} s'
                    if panel:
                        if data['validation_controller_cop_valid'][frame].all():
                            ready_text+=f' / 接触错位 {data["validation_controller_cop_line_angle_deg"][frame]:.1f}°'
                        else:
                            ready_text+=' / 接触错位不可用'
                    text(draw,(x+8,796),ready_text,19)
                    h=heights[episode][panel][frame]*1000.
                    text(draw,(x+8,830),f'全手最低点 L/R {h[0]:.3f} / {h[1]:.3f} mm',19,color=(180,40,40) if (h<0).any() else (27,39,54))
                    blocked=bool(data['validation_controller_floor_blocked'][frame])
                    count=int(data['validation_controller_floor_blocked_count'][frame])
                    if panel:
                        borrowed=data['validation_controller_shared_borrowed_travel_m'][frame]*1000.
                        total=float(data['validation_controller_shared_total_travel_m'][frame])*1000.
                        status=f'总 {total:.2f}mm / 借用 {borrowed[0]:.2f},{borrowed[1]:.2f}mm'
                    else:
                        changed_count=int(data['validation_controller_cop_allocation_changed_count'][frame])
                        status=f'分配 {changed_count} / 整步保持 {count}'
                    text(draw,(x+8,862),status,18)
                text(draw,(16,904),'载荷与最低点为实际步后值；两侧保存全部8子步几何。失败与回退均保留，未放宽原门槛。',16)
                if frame==2399:
                    im.save(out/f'episode_{episode}_final.png')
                else:
                    path=frame_dir/f'{len(paths):04d}.png';im.save(path);paths.append(path)
                    records.append(dict(episode=episode,frame=frame,timestamp_s=timestamp))
                if frame%400==0:print('SHARED_BUDGET_RENDER',episode,frame,flush=True)
    finally:
        renderer.delete()
    movie=out/'shared_budget_rollout.mp4'
    with imageio.get_writer(movie,fps=10,codec='libx264',quality=8,macro_block_size=16) as writer:
        for path in paths:writer.append_data(imageio.imread(path))
    reader=imageio.get_reader(movie);count=0
    for rgb in reader:
        assert rgb.shape==(928,1088,3)
        for panel in (0,1):
            assert rgb[126:686,8+panel*544:540+panel*544].std()>5
        count+=1
    reader.close();assert count==len(paths)==720
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,frames=count,fps=10,seconds=72,
        all_frames_decoded=True,playback='2x',cases=[r['episode'] for r in result['cases']],records=records,
        scope='All six complete48s physical recordings; no model estimates. Original12 plus late plus new actual8substep full-hand floor gate.',bindings=bindings),indent=2)+'\n')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>共享调整行程对照</title>',
          '<style>body{max-width:1200px;margin:28px auto;font:18px/1.7 system-ui;background:#f2f6f9}video,img{width:100%}</style>',
          '<h1>共享未执行行程：三例实际回放</h1>',
          '<p>两侧使用相同完整手网格地面约束、原力和5度起升门槛、40秒截止与4mm/s速度。左侧每手固定100mm；右侧仅在一手配额截断、另一手非零调整因地面受阻时，允许安全手使用未执行配额。双手累计仍不超过200mm。真实全部三例，2倍速，失败完整保留，无模型预测。</p>',
          '<p><a href="RESULT.json">全部三例、原物体门槛与新增全手门槛</a></p>',
          '<video controls preload="metadata" src="renders/shared_budget_rollout.mp4"></video>']
    for row in result['cases']:
        html.append(f'<h2>{row["episode"]}，48秒</h2><img src="renders/episode_{row["episode"]}_final.png">')
    (root/'index.html').write_text('\n'.join(html)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
