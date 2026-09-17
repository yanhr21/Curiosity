"""Actual matched rollout: same qualified CoP servo, common-up initial hand frames.

Full original hands/object, fixed failure/regression trio. No predictor output.
"""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse
import json
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
    if protocol['controller_intervention'] != 'common_up_cop_alignment_v1' or [r['episode'] for r in result['cases']] != [5014,5012,5000]:
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
                im=Image.new('RGB',(1088,864),'#f2f6f9');draw=ImageDraw.Draw(im)
                text(draw,(16,12),f'{episode} · 初始手朝向修复：固定三例对照',28)
                text(draw,(16,53),'真实完整网格回放｜同配置与物理，仅初始手切向朝向改变｜无模型预测',19)
                timestamp=float(datasets[0]['timestamp_s'][frame])
                for panel,data in enumerate(datasets):
                    x=8+panel*544
                    text(draw,(x+8,91),'修复响应估计：原始手朝向' if panel==0 else '共同向上：初始手朝向',22)
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
                text(draw,(16,838),'载荷为当帧步后观测；增益/就绪/接触错位依据前帧观测。两侧保留全部失败。',16)
                if frame==2399:
                    im.save(out/f'episode_{episode}_final.png')
                else:
                    path=frame_dir/f'{len(paths):04d}.png';im.save(path);paths.append(path)
                    records.append(dict(episode=episode,frame=frame,timestamp_s=timestamp))
                if frame%400==0:print('COMMON_UP_RENDER',episode,frame,flush=True)
    finally:
        renderer.delete()
    movie=out/'common_up_rollout.mp4'
    with imageio.get_writer(movie,fps=10,codec='libx264',quality=8,macro_block_size=16) as writer:
        for path in paths:writer.append_data(imageio.imread(path))
    reader=imageio.get_reader(movie);count=0
    for rgb in reader:
        assert rgb.shape==(864,1088,3)
        for panel in (0,1):
            assert rgb[126:686,8+panel*544:540+panel*544].std()>5
        count+=1
    reader.close();assert count==len(paths)==720
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,frames=count,fps=10,seconds=72,
        all_frames_decoded=True,playback='2x',cases=[r['episode'] for r in result['cases']],records=records,
        scope='All six complete48s physical recordings; no model estimates'),indent=2)+'\n')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>初始手朝向对照</title>',
          '<style>body{max-width:1200px;margin:28px auto;font:18px/1.7 system-ui;background:#f2f6f9}video,img{width:100%}</style>',
          '<h1>初始手朝向修复：固定三例完整回放</h1>',
          '<p>两侧采用同一已修复的力响应估计、接触中心对齐动作、夹力目标和原准入门槛。右侧只调整初始双手绕接近法向的朝向，使手的长轴共同向上；初始接近法向与支持三角中心不变。完整手姿态在仿真开始前设置，不在首个控制步突然翻转。左右均为实际物理轨迹，2倍速。所有失败保留，无模型预测。</p>',
          '<p><a href="RESULT.json">全部结果与原12项门槛</a></p>',
          '<video controls preload="metadata" src="renders/common_up_rollout.mp4"></video>']
    for row in result['cases']:
        html.append(f'<h2>{row["episode"]}，48秒</h2><img src="renders/episode_{row["episode"]}_final.png">')
    (root/'index.html').write_text('\n'.join(html)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
