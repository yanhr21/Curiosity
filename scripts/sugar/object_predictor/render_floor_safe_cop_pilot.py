"""Actual matched rollout: original qualified CoP versus full-hand floor constraint.

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
    if protocol['controller_intervention'] != 'floor_safe_qualified_cop_v1' or [r['episode'] for r in result['cases']] != [5014,5012,5000]:
        raise ValueError('Require the fixed completed pressure-center alignment trio')
    baseline=Path(protocol['baseline_root'])
    baseline_protocol=json.loads((baseline/'PROTOCOL.json').read_text())
    if baseline_protocol['controller_intervention']!='qualified_cop_response_v1':
        raise ValueError('Require immediate complete qualified-response baseline')
    if not json.loads((baseline/'RESULT.json').read_text())['complete']:
        raise ValueError('Incomplete qualified-response baseline')
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
            if panel:
                companion=root/'cases'/f'episode_{episode}'/'HAND_FLOOR_SUBSTEPS.npz'
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
                text(draw,(16,12),f'{episode} · 全手地面约束：固定三例对照',28)
                text(draw,(16,53),'真实完整网格回放｜同初始朝向、载荷与截止｜新增全手运动约束｜无模型预测',19)
                timestamp=float(datasets[0]['timestamp_s'][frame])
                for panel,data in enumerate(datasets):
                    x=8+panel*544
                    text(draw,(x+8,91),'原接触对齐控制' if panel==0 else '增加全手地面约束',22)
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
                    text(draw,(x+8,762),f'实际增益 {gain_text} m/(N·s)',19)
                    ready_text=f'持续就绪 {data["validation_controller_ready_seconds"][frame]:.2f} s'
                    if panel:
                        if data['validation_controller_cop_valid'][frame].all():
                            ready_text+=f' / 接触错位 {data["validation_controller_cop_line_angle_deg"][frame]:.1f}°'
                        else:
                            ready_text+=' / 接触错位不可用'
                    text(draw,(x+8,796),ready_text,19)
                    h=heights[episode][panel][frame]*1000.
                    text(draw,(x+8,830),f'全手最低点 L/R {h[0]:.3f} / {h[1]:.3f} mm',19,color=(180,40,40) if (h<0).any() else (27,39,54))
                    status='原轨迹：无全手地面约束'
                    if panel:
                        blocked=bool(data['validation_controller_floor_blocked'][frame])
                        count=int(data['validation_controller_floor_blocked_count'][frame])
                        status=f'本步：{"拒绝不安全运动" if blocked else "执行原运动"} / 累计拒绝 {count}'
                    text(draw,(x+8,862),status,18)
                text(draw,(16,904),'载荷与最低点为实际步后值；右侧另保存全部8子步几何。所有未起升、穿地与回退均保留。',16)
                if frame==2399:
                    im.save(out/f'episode_{episode}_final.png')
                else:
                    path=frame_dir/f'{len(paths):04d}.png';im.save(path);paths.append(path)
                    records.append(dict(episode=episode,frame=frame,timestamp_s=timestamp))
                if frame%400==0:print('FLOOR_SAFE_RENDER',episode,frame,flush=True)
    finally:
        renderer.delete()
    movie=out/'floor_safe_rollout.mp4'
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
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>全手地面约束对照</title>',
          '<style>body{max-width:1200px;margin:28px auto;font:18px/1.7 system-ui;background:#f2f6f9}video,img{width:100%}</style>',
          '<h1>完整手几何地面约束：三例实际回放</h1>',
          '<p>左侧为原qualified接触对齐控制；右侧保持相同初始手朝向，新增完整手网格平移与旋转全过程地面约束。安全指令完整执行，不安全指令保持原姿态并回滚运动进度。原载荷、截止和物体门槛不变；新增全部8子步全手地面检查。两侧都是真实物理，2倍速，保留全部失败，无模型预测。</p>',
          '<p><a href="RESULT.json">全部三例、原物体门槛与新增全手门槛</a></p>',
          '<video controls preload="metadata" src="renders/floor_safe_rollout.mp4"></video>']
    for row in result['cases']:
        html.append(f'<h2>{row["episode"]}，48秒</h2><img src="renders/episode_{row["episode"]}_final.png">')
    (root/'index.html').write_text('\n'.join(html)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
