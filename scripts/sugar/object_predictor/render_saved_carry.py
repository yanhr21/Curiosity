"""Actual full meshes and raw frozen estimates at matched saved clocks."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pyrender,imageio.v2 as imageio
from .render_device import configure_retained_egl
from .render_object_state import World,load,true_state,state_from_saved,canvas,text,TEAL,AMBER,PW,PH
from .report_grip_transfer import errors


def main():
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--episodes',type=int,nargs='+');ap.add_argument('--snapshots-only',action='store_true');args=ap.parse_args()
    root=Path(args.output);protocol=json.loads((root/'PROTOCOL.json').read_text());rp=protocol['render'];out=root/'renders';out.mkdir(exist_ok=True)
    configure_retained_egl();renderer=pyrender.OffscreenRenderer(PW,PH);world=World();records=[]
    episodes=args.episodes if args.episodes is not None else protocol['episodes']
    try:
        for ep in episodes:
            case=Path(protocol['source'])/'cases'/f'episode_{ep}';data=load(case/f'episode_{ep}.npz');meta=json.loads((case/'RESULT.json').read_text())
            saved=[load(root/'evaluation'/f'episode_{ep}'/f'{mode}.npz') for mode in rp['models']]
            frames=rp['video_frames'] if ep in rp['video_episodes'] and not args.snapshots_only else rp['all32_snapshots']
            lookup=[{int(f):i for i,f in enumerate(a['frame'])} for a in saved]
            # One fixed identical camera covers true and predicted geometry over
            # the entire case. Prediction outliers remain visible and unmodified.
            low=[];high=[]
            for frame in protocol['frames']:
                states=[true_state(data,frame)]+[state_from_saved(a['prediction'][idx[frame]],data['hand_pose_w'][frame]) for a,idx in zip(saved,lookup)]
                for center,quat,dims,mass in states:
                    radius=np.linalg.norm(dims)/2;low.append(center-radius);high.append(center+radius)
                low.append(data['hand_pose_w'][frame,:,:3].min(0)-.18);high.append(data['hand_pose_w'][frame,:,:3].max(0)+.18)
            lo=np.min(low,axis=0);hi=np.max(high,axis=0);target=(lo+hi)/2;radius=np.linalg.norm(hi-lo)/2
            direction=np.array([.65,-1.3,.48]);direction/=np.linalg.norm(direction);eye=target+direction*max(1.25,3.8*radius)
            sequence=out/f'episode_{ep}';sequence.mkdir(exist_ok=True)
            for frame in frames:
                path=sequence/f'frame_{frame:04d}.png'
                if path.exists():continue
                truth=true_state(data,frame);t=float(data['timestamp_s'][frame]);phase=int(data['validation_controller_phase'][frame]);phase_name=['接近/夹紧','抬升','保持'][phase]
                im=canvas('完整 Utonia · 从双手观测估计物体',f'案例 {ep} | {t:.2f} 秒 | {phase_name} | 冻结旧模型，无新训练 | 视频5×', ['真实物体与双手','只用几何 / 冻结模型','几何＋接触＋力 / 冻结模型'])
                for panel in range(3):
                    j=None if panel==0 else lookup[panel-1][frame]
                    state=truth if panel==0 else state_from_saved(saved[panel-1]['prediction'][j],data['hand_pose_w'][frame])
                    for primitive in world.object_node.mesh.primitives:primitive.material.baseColorFactor=[*(np.asarray(TEAL if panel==0 else AMBER)/255),1.]
                    world.set(data['hand_pose_w'][frame],*state[:3],data['contact_position_w'][frame],data['normal_load_n'][frame]);world.camera_at(eye,target)
                    rgb,_=renderer.render(world.scene,flags=0)
                    if rgb.std()<5:raise RuntimeError(f'Constant RGB {ep}/{frame}/{panel}')
                    view=Image.fromarray(rgb)
                    if panel:world.extent_guide(view,*truth[:3])
                    im.paste(view,(8+panel*532,144));draw=ImageDraw.Draw(im);x=24+panel*532
                    if panel==0:
                        loads=data['normal_load_n'][frame].reshape(2,27).sum(1)
                        text(draw,(x,747),f'真实质量 {truth[3]:.3f} kg',28,TEAL)
                        text(draw,(x,795),f'掌侧载荷 {loads[0]:.1f} / {loads[1]:.1f} N',22)
                        text(draw,(x,838),f'实际离地 {data["validation_full_mesh_min_z_m"][frame]*100:.2f} cm',23)
                        text(draw,(x,879),f'原采集验收：{"通过" if meta["passed"] else "失败"}',22)
                    else:
                        a=saved[panel-1];err=errors(a['prediction'][j:j+1],a['target'][j:j+1])
                        text(draw,(x,747),f'预测质量 {state[3]:.3f} kg',28,AMBER)
                        text(draw,(x,795),f'质量误差 {err["mass_pct"][0]:.1f}%',23)
                        text(draw,(x,838),f'位置 {err["center_cm"][0]:.1f} cm / 旋转 {err["rotation_deg"][0]:.1f}°',22)
                        text(draw,(x,879),f'尺寸误差 {err["size_pct"][0]:.1f}%',22)
                text(ImageDraw.Draw(im),(24,925),'青色/虚线：真值；橙色：预测状态下的完整已知网格。无平滑、无真值校正；不是任意形状重建。',18)
                im.save(path)
            record=dict(episode=ep,snapshots=rp['all32_snapshots'],camera_eye=eye.tolist(),camera_target=target.tolist())
            if ep in rp['video_episodes'] and not args.snapshots_only:
                video=out/f'episode_{ep}.mp4'
                with imageio.get_writer(video,fps=rp['fps'],codec='libx264',quality=8,macro_block_size=1) as writer:
                    for frame in rp['video_frames']:writer.append_data(imageio.imread(sequence/f'frame_{frame:04d}.png'))
                decoded=0
                for rgb in imageio.get_reader(video):
                    if rgb.std()<5:raise RuntimeError(f'Constant decoded frame{ep}/{decoded}')
                    decoded+=1
                if decoded!=len(rp['video_frames']):raise ValueError('Incomplete video')
                record.update(video=str(video),decoded_frames=decoded,sha256=hashlib.file_digest(video.open('rb'),'sha256').hexdigest())
            records.append(record);print('RENDER_COMPLETE',ep,flush=True)
    finally:renderer.delete()
    if args.snapshots_only:
        (out/('PREVIEW_'+str(episodes[0])+'.json')).write_text(json.dumps(dict(records=records),indent=2));return
    if episodes!=protocol['episodes']:raise ValueError('Final report requires all original32')
    # Review sheets preserve every declared snapshot in episode/time order.
    # Full-resolution originals remain the interactive viewer's source.
    sheets=[]
    for start in range(0,len(episodes),4):
        sheet=Image.new('RGB',(1600,1920),'white')
        for row,ep in enumerate(episodes[start:start+4]):
            for col,frame in enumerate(rp['all32_snapshots']):
                with Image.open(out/f'episode_{ep}'/f'frame_{frame:04d}.png') as original:
                    sheet.paste(original.resize((800,480),Image.Resampling.LANCZOS),(col*800,row*480))
        path=out/f'review_{episodes[start]}_{episodes[min(start+3,len(episodes)-1)]}.jpg'
        sheet.save(path,quality=95);sheets.append(str(path))
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,records=records,all32_snapshots=True,review_sheets=sheets,raw_prediction=True,new_optimizer_updates=0),indent=2))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整Utonia冻结物体估计</title><style>body{max-width:1600px;margin:28px auto;padding:0 20px;background:#f2f6f9;color:#1b2736;font:18px/1.7 system-ui}img,video{width:100%}button,select{font:inherit;padding:6px;margin:6px}</style><h1>完整Utonia：原32条轨迹的冻结物体估计</h1><p>三个完整已有模型加力置零对照，全部原32例均计入指标。画面展示实际状态、几何模型、含力模型；预测为完整已知网格的位姿/尺寸，未实现任意形状重建。无新训练。</p><p><a href="REPORT.md">完整报告</a> · <a href="COMPARISON.json">全部分项</a></p><img src="comparison.png"><h2>预先固定的四类轨迹</h2>']
    labels={5000:'正常搬运',5005:'载荷不足',5011:'接触方向异常',5014:'倾倒失败'}
    for ep in rp['video_episodes']:parts.append(f'<h3>{ep}：{labels[ep]}</h3><video controls preload="metadata" src="renders/episode_{ep}.mp4"></video>')
    parts.append('<h2>全部32例：同一固定时刻</h2><select id="episode">'+''.join(f'<option>{ep}</option>' for ep in episodes)+'</select><select id="frame"><option value="1156">23.14秒</option><option value="2381">47.64秒</option></select><img id="snapshot"><script>const e=document.getElementById("episode"),f=document.getElementById("frame");function show(){document.getElementById("snapshot").src=`renders/episode_${e.value}/frame_${f.value.padStart(4,"0")}.png`;}e.onchange=show;f.onchange=show;show();</script></html>')
    (root/'index.html').write_text(''.join(parts));print('ALL32_RENDER_COMPLETE',flush=True)


if __name__=='__main__':main()
