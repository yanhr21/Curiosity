"""Matched exact-mesh rendering of all four Newton grip/weight cases."""
import argparse,json,os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import ImageDraw
from .render_object_state import World,load,true_state,state_from_saved,canvas,text,TEAL,AMBER,PW,PH
import pyrender
from .report_grip_transfer import errors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
    ap.add_argument('--start-episode',type=int,default=4000);ap.add_argument('--start-frame',type=int,default=1)
    ap.add_argument('--output-name',default='newton_grip_transfer.mp4')
    ap.add_argument('--frames-only',action='store_true');ap.add_argument('--sequence-offset',type=int,default=0);args=ap.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    root=Path('experiments/object_predictor_v1/grip_transfer_v1');out=root/'renders';out.mkdir(exist_ok=True)
    modes=['geometry','geometry_contact_force'];saved=[load(root/'evaluation'/m/'predictions.npz') for m in modes]
    # Reuse one scene and its GPU buffers for the three views. Repeatedly
    # switching three complete mesh graphs produced intermittent black RGB
    # on this EGL driver although depth remained valid; incidents are retained.
    renderer=pyrender.OffscreenRenderer(PW,PH)
    shared_world=World();worlds=[shared_world]*3
    writer=None if args.preview or args.frames_only else imageio.get_writer(out/args.output_name,fps=20,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1)
    if args.frames_only:(out/'frames').mkdir(exist_ok=True)
    count=args.sequence_offset
    try:
        for ep in range(args.start_episode,4004):
            data=load(root/'data_v1'/f'episode_{ep:04d}.npz');meta=json.loads((root/'data_v1'/f'episode_{ep:04d}.json').read_text())
            indices=[{int(a['frame'][j]):int(j) for j in np.flatnonzero(a['episode']==ep)} for a in saved]
            frames=[1,531,931,1106] if args.preview else list(range(1,1200,5))
            if ep==args.start_episode:frames=[f for f in frames if f>=args.start_frame]
            for frame in frames:
                truth=true_state(data,frame);c,q,dims,mass=truth
                t=float(data['timestamp_s'][frame]);phase='接近 / 夹紧' if t<16 else ('抬升 / 平移' if t<20 else '保持')
                im=canvas('Newton 夹持搬运 · 重量与夹紧力分离',f'案例 {ep-3999}/4  |  真值 {mass:.1f} kg  |  每手夹紧目标 {meta["grip_target_per_hand_n"]} N  |  完整双手夹具 / 非全身策略  |  2×',
                          ['实际物体与双手','只用几何 / 冻结旧模型','几何 + 接触 + 力 / 冻结旧模型'])
                for k,world in enumerate(worlds):
                    for primitive in world.object_node.mesh.primitives:
                        primitive.material.baseColorFactor=[*(np.asarray(TEAL if k==0 else AMBER)/255),1.]
                    j=None if frame<31 or k==0 else indices[k-1][frame]
                    state=truth if k==0 or j is None else state_from_saved(saved[k-1]['prediction'][j],data['hand_pose_w'][frame])
                    world.set(data['hand_pose_w'][frame],*state[:3],data['contact_position_w'][frame],data['normal_load_n'][frame])
                    world.camera_at(c+np.array([.65,-1.3,.48]),c)
                    panel=world.render(renderer)
                    if np.asarray(panel).std()<5:
                        # Retry exactly the same saved scene without the shadow
                        # framebuffer pass. No physics or prediction is rerun.
                        from PIL import Image
                        color,depth=renderer.render(world.scene,flags=0)
                        panel=Image.fromarray(color)
                        event=dict(episode=ep,frame=frame,panel=k,reason='constant RGB from shadow pass',
                                   action='same-state plain OpenGL pass',recovered=bool(np.asarray(panel).std()>=5))
                        if not event['recovered']:
                            from OpenGL.GL import glGetString,GL_RENDERER,glGetError
                            event['old_context_gl_error']=int(glGetError())
                            renderer.delete();renderer=pyrender.OffscreenRenderer(PW,PH)
                            event['fresh_gl_renderer']=str(glGetString(GL_RENDERER))
                            color,depth=renderer.render(world.scene,flags=0)
                            panel=Image.fromarray(color)
                            event['action']='fresh EGL context, same-state plain pass'
                            event['recovered']=bool(np.asarray(panel).std()>=5)
                        with (out/'RENDER_EVENTS.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
                        if not event['recovered']:raise RuntimeError(f'Constant RGB persists after context replacement at episode {ep} frame {frame} panel {k}')
                    if k and j is not None:world.extent_guide(panel,c,q,dims)
                    if k and j is None:
                        from PIL import Image
                        from .render_object_state import BG
                        panel=Image.new('RGB',(PW,PH),BG);pd=ImageDraw.Draw(panel);text(pd,(90,240),'正在积累 32 帧历史',24)
                    im.paste(panel,(8+k*532,144));draw=ImageDraw.Draw(im)
                    if k==0:
                        text(draw,(24,747),f'{phase}    {t:.2f} s',24,TEAL)
                        text(draw,(24,790),f'真实重量 {mass:.3f} kg',31,TEAL)
                        loads=data['normal_load_n'][frame].reshape(2,27).sum(1)
                        text(draw,(24,842),f'记录载荷 L {loads[0]:.1f} / R {loads[1]:.1f} N',22)
                        text(draw,(24,878),f'完整网格离地 {data["validation_full_mesh_min_z_m"][frame]*100:.1f} cm',22)
                    elif j is not None:
                        a=saved[k-1];e=errors(a['prediction'][j:j+1],a['target'][j:j+1])
                        text(draw,(24+k*532,747),f'预测重量 {state[3]:.3f} kg',30,AMBER)
                        text(draw,(24+k*532,791),f'重量误差 {e["mass_pct"][0]:.1f}%',24)
                        text(draw,(24+k*532,836),f'位置 {e["center_cm"][0]:.1f} cm  /  旋转 {e["rotation_deg"][0]:.1f}°',23)
                        text(draw,(24+k*532,875),f'尺寸误差 {e["size_pct"][0]:.1f}%',22)
                text(ImageDraw.Draw(im),(24,920),'青色：真值    橙色：预测已知网格    红点：接触    预测与画面使用同一记录时刻；无平滑、无新增训练',19)
                if frame in (1,531,931,1106):im.save(out/f'episode_{ep}_frame_{frame:04d}.png')
                if args.frames_only:im.save(out/'frames'/f'frame_{count:04d}.png')
                if writer:writer.append_data(np.asarray(im))
                count+=1
            print('RENDER_GRIP',ep,flush=True)
    finally:
        if writer:writer.close()
        renderer.delete()
    if not args.preview:(out/(Path(args.output_name).stem+'.json')).write_text(json.dumps(dict(frames=count,fps=20,seconds=count/20,source_hz=50,sampled_every=5,playback_speed=2,start_episode=args.start_episode,start_frame=args.start_frame,episodes=list(range(args.start_episode,4004)),full_original_mesh=True,model_updates=0,physics_replayed=False),indent=2))


if __name__=='__main__':main()
