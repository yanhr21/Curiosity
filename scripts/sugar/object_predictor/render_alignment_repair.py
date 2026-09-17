"""Actual matched rollout: continued alignment versus post-lift alignment freeze.

Full original hands/object, fixed failure/regression pair. No predictor output.
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
    configure_retained_egl()
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    result=json.loads((root/'RESULT.json').read_text()); assert result['complete']
    if protocol['controller_intervention'] != 'freeze_postlift_alignment_v1' or [r['episode'] for r in result['cases']] != [5012,5000]:
        raise ValueError('Require the fixed completed freeze-alignment pair')
    out=root/'renders';out.mkdir(exist_ok=False)
    frame_dir=out/'frames';frame_dir.mkdir()
    world=World(ground_height=0.);renderer=pyrender.OffscreenRenderer(532,560)
    frames=list(range(0,2400,10));paths=[];records=[]
    try:
        for case in result['cases']:
            episode=case['episode']; datasets=[]
            for source in (Path(protocol['baseline_root']),root):
                datasets.append(load(source/'cases'/f'episode_{episode}'/f'episode_{episode}.npz'))
            assert np.array_equal(datasets[0]['timestamp_s'],datasets[1]['timestamp_s'])
            target=true_state(datasets[0],0)[0]+[0,0,.08]
            eye=target+[.75,-1.2,.55]
            for frame in frames+[2399]:
                im=Image.new('RGB',(1088,864),'#f2f6f9');draw=ImageDraw.Draw(im)
                text(draw,(16,12),f'{episode} · 抬起后对齐控制：固定单因素对照',28)
                text(draw,(16,53),'真实完整网格回放｜同配置、同种子、同物理｜无模型预测',19)
                timestamp=float(datasets[0]['timestamp_s'][frame])
                for panel,data in enumerate(datasets):
                    x=8+panel*544
                    text(draw,(x+8,91),'抬升后继续调整对齐' if panel==0 else '抬升后冻结对齐调整',22)
                    center,quaternion,dimensions,_=true_state(data,frame)
                    world.set(data['hand_pose_w'][frame],center,quaternion,dimensions,
                              data['contact_position_w'][frame],data['normal_load_n'][frame])
                    world.camera_at(eye,target)
                    rgb,depth=renderer.render(world.scene,flags=0)
                    if rgb.std()<5 or not np.isfinite(depth).all() or np.count_nonzero(depth)<100:
                        raise RuntimeError(f'Invalid render {episode}/{frame}/{panel}')
                    im.paste(Image.fromarray(rgb),(x,126))
                    loads=data['normal_load_n'][frame].reshape(2,27).sum(1)
                    valid=data['validation_controller_fit_valid'][frame]
                    angles=data['validation_controller_alignment_error_deg'][frame]
                    angular=' / '.join(f'{a:.1f}°' if v else '无效' for a,v in zip(angles,valid))
                    text(draw,(x+8,694),f'{timestamp:05.2f}s  手载荷 {loads[0]:.2f} / {loads[1]:.2f} N',20)
                    text(draw,(x+8,728),f'完整物体离地 {data["validation_full_mesh_min_z_m"][frame]*100:.2f} cm',20)
                    text(draw,(x+8,762),f'控制器平面错角 {angular}',19)
                    text(draw,(x+8,796),f'持续就绪 {data["validation_controller_ready_seconds"][frame]:.2f} s',19)
                text(draw,(16,838),'载荷为当帧步后观测；平面/就绪值依据前帧观测。无效角度不显示为数值。',16)
                if frame==2399:
                    im.save(out/f'episode_{episode}_final.png')
                else:
                    path=frame_dir/f'{len(paths):04d}.png';im.save(path);paths.append(path)
                    records.append(dict(episode=episode,frame=frame,timestamp_s=timestamp))
                if frame%400==0:print('ALIGNMENT_REPAIR_RENDER',episode,frame,flush=True)
    finally:
        renderer.delete()
    movie=out/'alignment_repair_rollout.mp4'
    with imageio.get_writer(movie,fps=10,codec='libx264',quality=8,macro_block_size=16) as writer:
        for path in paths:writer.append_data(imageio.imread(path))
    reader=imageio.get_reader(movie);count=0
    for rgb in reader:
        assert rgb.shape==(864,1088,3)
        for panel in (0,1):
            assert rgb[126:686,8+panel*544:540+panel*544].std()>5
        count+=1
    reader.close();assert count==len(paths)==480
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,frames=count,fps=10,seconds=48,
        all_frames_decoded=True,playback='2x',cases=[r['episode'] for r in result['cases']],records=records,
        scope='Both complete48s physical recordings; no model estimates'),indent=2)+'\n')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>抬升后对齐修复对照</title>',
          '<style>body{max-width:1200px;margin:28px auto;font:18px/1.7 system-ui;background:#f2f6f9}video,img{width:100%}</style>',
          '<h1>抬升后对齐：固定两例完整回放</h1>',
          '<p>只改变抬升准入之后是否继续对齐；力反馈和搬运路径不变；左右均为实际物理轨迹，2倍速。所有成功/失败结果保留，无模型预测。</p>',
          '<p><a href="RESULT.json">全部结果与原12项门槛</a></p>',
          '<video controls preload="metadata" src="renders/alignment_repair_rollout.mp4"></video>']
    for row in result['cases']:
        html.append(f'<h2>{row["episode"]}，48秒</h2><img src="renders/episode_{row["episode"]}_final.png">')
    (root/'index.html').write_text('\n'.join(html)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
