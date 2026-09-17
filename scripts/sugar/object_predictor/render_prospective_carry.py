"""Actual fresh fixture replay; sparse estimates are held in WORLD coordinates."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrender
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .render_device import configure_retained_egl
from .render_object_state import World, canvas, text, PW, PH, BG, TEAL, AMBER
from .report_prospective_carry import read, world_state, world_truth, past_index, pose_error, minimum_z


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Render in the retained compute step')
    configure_retained_egl()
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    comparison = json.loads((root/'COMPARISON.json').read_text())
    assert comparison['complete']
    configs = protocol['configurations']; clocks = np.array(protocol['prediction_frames'])
    records = {r['episode']: r for r in json.loads((root/'COLLECTION_RESULT.json').read_text())['records']}
    mesh = read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    vertices = mesh['vertices'].astype(float); dims = np.ptp(vertices,axis=0)
    vertices -= (vertices.max(0)+vertices.min(0))/2
    out = root/'renders'; out.mkdir(exist_ok=False)
    sequence = out/'frames'; sequence.mkdir(exist_ok=False)
    world = World(ground_height=0.); renderer = pyrender.OffscreenRenderer(PW,PH)
    paths = []; posters = []; cases = []; fps = protocol['render']['video_fps']
    try:
        for number, config in enumerate(configs):
            ep = config['episode']; trace = read(Path(records[ep]['source'])/f'episode_{ep}.npz')
            states = [[],[]]; contact = []
            for frame in clocks:
                saved = read(root/'fusion'/f'episode_{ep}_frame_{frame}.npz')
                assert np.array_equal(saved['hand_pose_w'],trace['hand_pose_w'][frame])
                contact.append(len(saved['points_current_left_hand_m']))
                for arm,key in enumerate(('original_prediction','prediction')):
                    # Conversion happens ONCE using the inference-frame hand pose.
                    # The display-frame hand below never modifies these states.
                    states[arm].append(world_state(saved[key],saved['hand_pose_w'][0]))
            target = world_truth(trace,0)[0] + [0.,0.,.10]
            frames = list(range(1,2400,protocol['render']['recorded_frame_stride']))
            cases.append(dict(episode=ep, video_start_s=len(paths)/fps, frames=len(frames),
                              controller_passed=records[ep]['controller_passed']))
            for frame in frames:
                path = sequence/f'episode_{ep}_frame_{frame:04d}.png'; paths.append(path)
                k = past_index(clocks,frame)
                assert k < 0 or clocks[k] <= frame
                truth = world_truth(trace,frame); now = float(trace['timestamp_s'][frame])
                age = None if k < 0 else now-float(trace['timestamp_s'][clocks[k]])
                im = canvas('新配置搬运 · 冻结 Utonia 与接触／地面融合',
                    f'配置 {number+1}/16 · {ep} | 记录 {now:.2f}s | 4× 回放 | 离线 1Hz 估计，更新之间保持世界位姿',
                    ['实际箱体与双手','冻结 Utonia','先验 + 接触几何 + 已知地面'])
                draw = ImageDraw.Draw(im)
                for panel in range(3):
                    x = 24+panel*532
                    if panel and k < 0:
                        image = Image.new('RGB',(PW,PH),BG)
                        text(ImageDraw.Draw(image),(85,255),f'积累历史 {frame+1}/32 帧',25)
                        im.paste(image,(8+panel*532,144))
                        continue
                    state = truth if panel == 0 else states[panel-1][k]
                    center,Q,size,mass = state
                    quat = Rotation.from_matrix(Q).as_quat()
                    for primitive in world.object_node.mesh.primitives:
                        primitive.material.baseColorFactor = [*(np.array(TEAL if panel==0 else AMBER)/255),1.]
                    world.set(trace['hand_pose_w'][frame],center,quat,size,
                              trace['contact_position_w'][frame],trace['normal_load_n'][frame])
                    world.camera_at(target+[.8,-1.3,.5],target)
                    rgb,_ = renderer.render(world.scene,flags=0)
                    if rgb.std()<5:
                        raise RuntimeError(f'Constant rendered RGB: {ep}/{frame}/{panel}')
                    image = Image.fromarray(rgb)
                    if panel:
                        world.extent_guide(image,truth[0],Rotation.from_matrix(truth[1]).as_quat(),truth[2])
                    im.paste(image,(8+panel*532,144))
                    if panel==0:
                        loads = trace['normal_load_n'][frame].reshape(2,27).sum(1)
                        text(draw,(x,752),f'真实质量 {mass:.3f} kg',27)
                        text(draw,(x,794),f'完整网格离地 {trace["validation_full_mesh_min_z_m"][frame]*100:.1f} cm',23)
                        text(draw,(x,836),f'掌侧载荷 L {loads[0]:.1f} / R {loads[1]:.1f} N',22)
                        text(draw,(x,878),'本例采集：'+('通过' if records[ep]['controller_passed'] else '失败，完整保留'),21)
                    else:
                        error = pose_error(state,truth); z = minimum_z(vertices,dims,state)*100
                        text(draw,(x,752),f'预测质量 {mass:.3f} kg',27,AMBER)
                        text(draw,(x,794),f'位置 {error["center_cm"]:.1f} cm / 旋转 {error["rotation_deg"]:.1f}°',23)
                        text(draw,(x,836),f'预测网格最低点 {z:.2f} cm',22,
                             color=(180,45,45) if z<-.1 else TEAL)
                        text(draw,(x,875),f'距本次估计 {age:.2f}s · '+('无接触，保留原预测' if contact[k]==0 else f'当前估计用了 {contact[k]} 个接触点'),17)
                text(draw,(24,925),'真实网格与手部回放；青色轮廓仅用于评价。预测没有随手移动、使用未来帧或真值修正；已知网格，非任意形状重建。',16)
                im.save(path)
            posters.append(paths[-1])
            print('PROSPECTIVE_RENDER_CASE',ep,len(paths),flush=True)
    finally:
        renderer.delete()
    movie = out/'prospective_carry_comparison.mp4'
    with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1) as writer:
        for path in paths:
            writer.append_data(imageio.imread(path))
    reader = imageio.get_reader(movie); count = 0
    for rgb in reader:
        assert rgb.shape==(960,1600,3)
        decoded_frame = frames[count % len(frames)]
        for panel in range(1 if decoded_frame < clocks[0] else 3):
            assert rgb[144:724,8+panel*532:528+panel*532].std()>5
        count+=1
    encoded_fps = reader.get_meta_data()['fps']; reader.close()
    assert count==len(paths)==len(configs)*480 and encoded_fps==fps
    sheet = Image.new('RGB',(1600,480*((len(posters)+1)//2)),BG)
    for i,path in enumerate(posters):
        with Image.open(path) as im:
            sheet.paste(im.resize((800,480)),((i%2)*800,(i//2)*480))
    sheet.save(out/'all_cases.jpg')
    (out/'RESULT.json').write_text(json.dumps(dict(complete=True,frames=count,fps=fps,seconds=count/fps,
        all_frames_decoded=True,cases=cases,holding='Previous predicted WORLD pose; current hands rendered independently',
        estimate_clock_hz=1,real_time_claim=False,physics_steps=0,model_parameter_updates=0),indent=2)+'\n')
    buttons=''.join(f'<button onclick="document.querySelector(\'video\').currentTime={c["video_start_s"]}">案例 {c["episode"]}</button>' for c in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新配置搬运状态估计</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;background:#f2f6f9;color:#1b2736;font:18px/1.7 system-ui}video,img{width:100%}button{margin:4px;padding:8px}a{color:#087e88}</style><h1>新配置搬运：完整模型与接触几何融合</h1><p>全部16配置，4×回放。估计每1秒仿真时间离线计算一次，更新之间保持世界位姿；不是实时跟踪。包括所有采集失败。</p><video controls preload="metadata" src="renders/prospective_carry_comparison.mp4"></video><p>'+buttons+'</p><p><a href="REPORT.md">报告</a> · <a href="COMPARISON.json">全部数值和失败</a></p><img src="renders/all_cases.jpg">')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
