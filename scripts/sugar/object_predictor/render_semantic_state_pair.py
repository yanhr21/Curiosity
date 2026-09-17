"""Actual fixed-trajectory full-mesh replay with saved Utonia before/after states.

Every inference prediction is converted once at its own causal clock into world
coordinates. Current hand poses never drag or correct an older object estimate.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

CONDITIONAL_PROFILE = 'conditional_contact_v1'
FAILURE_AWARE_PROFILE = 'failure_aware_controlled_contact_v1'
SEMANTIC_PROFILE = 'semantic_state_coverage464_v1'
CONDITIONAL_PROFILES = (SEMANTIC_PROFILE,)
CONDITIONS = ('no_summary','observed_summary')
INFERENCE_FRAMES = tuple(sorted(set(range(31,2400,25)) | set(range(49,1200,50))))
EVIDENCE_CODES = {'0': 'PRIOR_UNKNOWN', '1': 'CONTACT_CANDIDATE_OBSERVABILITY_UNPROVEN'}
EVIDENCE_FIELDS = ('state_precision_eligible', 'state_contact_history_frames', 'state_evidence_status')
WORKER_THREAD_LIMITS = {key: '1' for key in (
    'LP_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
FFMPEG_SINGLE_THREAD_PARAMETERS = [
    '-threads','1','-filter_threads','1','-filter_complex_threads','1']


def episode_shards(episodes, workers):
    """Keep complete episodes and their fixed order; never split a timeline."""
    episodes = tuple(episodes)
    if workers not in (1, 4) or len(episodes) != 16 or len(set(episodes)) != 16:
        raise ValueError('Expected fixed sixteen episodes and one or four workers')
    width = len(episodes) // workers
    return tuple(episodes[start:start+width] for start in range(0, len(episodes), width))


def worker_environment(environment):
    result = dict(environment)
    result.update(WORKER_THREAD_LIMITS)
    return result


def validate_decoded_panel_frame(rgb, index, frames):
    if rgb.shape != (960, 1600, 3):
        raise ValueError('Unexpected video shape')
    frame = frames[index % len(frames)]
    for panel in range(1 if frame < 31 else 3):
        if rgb[144:724, 8+532*panel:528+532*panel].std() < 5:
            raise ValueError('Blank decoded panel')


def decode_movie(movie, frames, expected_count, fps, *, single_thread=False):
    import imageio.v2 as imageio
    options = {'input_params': FFMPEG_SINGLE_THREAD_PARAMETERS} if single_thread else {}
    decoded = 0
    with imageio.get_reader(movie, **options) as reader:
        actual_fps = reader.get_meta_data()['fps']
        for rgb in reader:
            validate_decoded_panel_frame(rgb, decoded, frames)
            decoded += 1
    if decoded != expected_count or actual_fps != fps:
        raise ValueError('Incomplete expected video')
    return decoded


def validate_state_evidence(protocol, arms):
    """Validate saved observation evidence without GL/model/GT-derived decisions."""
    profile = protocol.get('supervision_profile', 'all_state')
    if profile == 'all_state':
        return False  # Preserve the original artifact/display contract.
    if profile not in CONDITIONAL_PROFILES or protocol.get('collection_study') != 'controlled_fixture16_v1':
        raise ValueError('Unknown or mismatched conditional render scope')
    if protocol.get('state_evidence_status_codes') != EVIDENCE_CODES:
        raise ValueError('Conditional evidence status mapping differs')
    if len(arms) != 2:
        raise ValueError('Require real saved before/after predictions')
    ordered = []
    for arm in arms:
        n = len(arm['episode'])
        if n == 0 or any(key not in arm for key in EVIDENCE_FIELDS):
            raise ValueError('Conditional predictions lack their saved evidence fields')
        if any(np.asarray(arm[key]).shape != (n,) or not np.isfinite(arm[key]).all()
               for key in EVIDENCE_FIELDS):
            raise ValueError('Invalid conditional evidence array shape/value')
        status = np.asarray(arm['state_evidence_status'])
        count = np.asarray(arm['state_contact_history_frames'])
        eligible = np.asarray(arm['state_precision_eligible'])
        if (not np.issubdtype(status.dtype, np.integer) or
                not np.issubdtype(count.dtype, np.integer) or
                not np.isin(status, (0, 1)).all() or not np.isin(eligible, (0., 1.)).all()
                or np.any((count < 0) | (count > 32)) or
                not np.array_equal(status, eligible) or not np.array_equal(status > 0, count > 0)):
            raise ValueError('Conditional evidence/precision/history flags disagree')
        keys = np.column_stack((arm['episode'], arm['frame']))
        if keys.shape != (n, 2) or len(np.unique(keys, axis=0)) != n:
            raise ValueError('Duplicate or malformed conditional prediction clocks')
        order = np.lexsort((arm['frame'], arm['episode']))
        ordered.append({key: np.asarray(arm[key])[order]
                        for key in ('episode', 'frame', *EVIDENCE_FIELDS)})
    if any(not np.array_equal(ordered[0][key], ordered[1][key]) for key in ordered[0]):
        raise ValueError('Before/after saved state-evidence clocks differ')
    return True


def held_evidence(clocks, timestamps, frame, status, contact_frames):
    """Use only the last causal prediction's evidence and its physical age.

    No hand pose, truth or predicted state enters this function. The caller keeps
    the already world-bound prediction unchanged while displaying this context.
    """
    clocks = np.asarray(clocks)
    timestamps = np.asarray(timestamps)
    if (clocks.ndim != 1 or not len(clocks) or not np.issubdtype(clocks.dtype, np.integer)
            or np.any(np.diff(clocks) <= 0) or clocks[0] < 0 or clocks[-1] >= len(timestamps)
            or frame < 0 or frame >= len(timestamps)
            or len(status) != len(clocks) or len(contact_frames) != len(clocks)):
        raise ValueError('Invalid held-prediction clock/evidence grid')
    index = int(np.searchsorted(clocks, frame, side='right') - 1)
    if index < 0:
        return dict(index=-1, prediction_frame=None, prediction_time_s=None,
                    age_s=None, status=None, contact_history_frames=None)
    prediction_frame = int(clocks[index])
    prediction_time = float(timestamps[prediction_frame])
    age = float(timestamps[frame] - prediction_time)
    if not np.isfinite(age) or age < 0:
        raise ValueError('A held prediction must never come from a future clock')
    return dict(index=index, prediction_frame=prediction_frame, prediction_time_s=prediction_time,
                age_s=age, status=int(status[index]), contact_history_frames=int(contact_frames[index]))


def evidence_label(context):
    if context['index'] < 0:
        return '积累历史', '尚无完整32帧预测'
    if context['status'] == 0:
        return '先验/未知', '无接触证据；非精确状态估计'
    if context['status'] == 1:
        return '接触候选', '有接触证据；可辨识性未证明'
    raise ValueError('Unknown saved evidence status')


def bind_saved_world_states(predicted, clocks, hand_poses):
    """Bind each saved state to its own hand clock, before replay begins."""
    from .report_prospective_carry import world_state
    return [world_state(prediction, hand_poses[int(frame), 0])
            for prediction, frame in zip(predicted, clocks, strict=True)]


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def predictions(root,condition):
    arm=root/condition
    parts=[read(arm/'fit_02120.npz'),read(arm/'same_trajectory_interpolation.npz')]
    keys=set(parts[0])
    if set(parts[1])!=keys:raise ValueError('Fit/development prediction fields differ')
    return {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}


def verify_pair(root):
    from .semantic_state_training import verify_artifacts
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    for condition in CONDITIONS:
        arm=root/condition
        if json.loads((arm/'PROTOCOL.json').read_text())!=dict(protocol,condition=condition):
            raise ValueError('Arm protocol differs from matched pair source/objective')
        verify_artifacts(arm)
        result=json.loads((arm/'RESULT.json').read_text())
        if (result.get('execution_complete') is not True or result.get('condition')!=condition
                or result.get('full_endpoint_reload_max_abs')!=0
                or result.get('source_optimizer_updates')!=2100
                or result.get('optimizer_updates')!=20 or result.get('total_optimizer_updates')!=2120
                or result.get('fit_items')!=464 or result.get('interpolation_items')!=1440
                or result.get('endpoint_prediction_file')!='fit_02120.npz'):
            raise ValueError('Require complete matched semantic20-update endpoints')
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if protocol.get('supervision_profile')!=SEMANTIC_PROFILE:
        raise ValueError('Require explicit semantic observation objective profile')
    return protocol


def render(root, shard_index=None):
    from .retained_execution import require_active_resource
    require_active_resource()
    verify_pair(root)
    os.environ['PYOPENGL_PLATFORM']='egl'
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    import pyrender
    from scipy.spatial.transform import Rotation
    from .data import target_at
    from .overfit_data import FIXED_EPISODES
    from .render_device import configure_retained_egl
    from .render_object_state import World,canvas,text,PW,PH,BG,TEAL,AMBER
    from .report_prospective_carry import world_truth,pose_error,past_index

    protocol=verify_pair(root)
    source=Path(protocol['source_data'])
    records={r['episode']:r for r in json.loads((source/'COLLECTION_RESULT.json').read_text())['records']}
    arms=[predictions(root,condition) for condition in CONDITIONS]
    conditional=validate_state_evidence(protocol,arms)
    for arm in arms:
        if len(arm['episode'])!=1904:raise ValueError('Require all fit and dense causal clocks')
    if shard_index is None:
        episodes=episode_shards(FIXED_EPISODES,1)[0]
        out=root/'renders'
    else:
        if shard_index not in range(4):raise ValueError('Invalid internal worker shard')
        episodes=episode_shards(FIXED_EPISODES,4)[shard_index]
        out=root/'renders'/'parts'/f'part_{shard_index:02d}'
    out.mkdir(exist_ok=False)
    device=configure_retained_egl();world=World(ground_height=0.)
    renderer=pyrender.OffscreenRenderer(PW,PH)
    frames=list(range(1,2400,10));fps=20
    movie=out/'fixed16_semantic_observation_comparison.mp4'
    video_cases=[];stills=[];count=0
    try:
        writer_options={'ffmpeg_params':FFMPEG_SINGLE_THREAD_PARAMETERS} if shard_index is not None else {}
        with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1,**writer_options) as writer:
            for episode in episodes:
                trace=read(Path(records[episode]['source'])/f'episode_{episode}.npz')
                clocks=None;states=[];probabilities=[];evidence=[]
                for arm in arms:
                    selected=np.flatnonzero(arm['episode']==episode)
                    selected=selected[np.argsort(arm['frame'][selected])]
                    actual=arm['frame'][selected]
                    if not np.array_equal(actual,np.asarray(INFERENCE_FRAMES)):raise ValueError('Causal clock grid changed')
                    if clocks is not None and not np.array_equal(actual,clocks):raise ValueError('Before/after clocks differ')
                    clocks=actual
                    expected=np.stack([target_at(trace,int(f)) for f in clocks])
                    if not np.array_equal(arm['target'][selected],expected):raise ValueError('Saved labels differ from actual physical replay')
                    states.append(bind_saved_world_states(arm['prediction'][selected],clocks,trace['hand_pose_w']))
                    probabilities.append(arm['availability_probability'][selected])
                    if conditional:
                        evidence.append({key:arm[key][selected] for key in EVIDENCE_FIELDS})
                camera_target=world_truth(trace,0)[0]+[0,0,.10]
                video_cases.append(dict(episode=episode,video_start_s=count/fps,frames=len(frames),
                    controller_passed=records[episode]['controller_passed']))
                if conditional:
                    status=evidence[0]['state_evidence_status']
                    video_cases[-1].update(prior_unknown_prediction_clocks=int((status==0).sum()),
                        contact_candidate_prediction_clocks=int((status==1).sum()))
                for frame in frames:
                    k=past_index(clocks,frame);truth=world_truth(trace,frame)
                    contexts=[]
                    labels=['实际物体与双手','无明确观测分支','力汇总与手高度分支']
                    title='完整 Utonia · 明确观测输入对照'
                    if conditional:
                        contexts=[held_evidence(clocks,trace['timestamp_s'],frame,
                            item['state_evidence_status'],item['state_contact_history_frames']) for item in evidence]
                        if any(context['index']!=k for context in contexts):raise ValueError('Evidence and world-state clocks differ')
                        labels=['实际物体与双手', '无观测分支 · '+evidence_label(contexts[0])[0],
                                '明确观测 · '+evidence_label(contexts[1])[0]]
                        title='受控训练轨迹 · 同预算明确观测对照'
                    im=canvas(title,
                        f'{episode} | 仿真 {trace["timestamp_s"][frame]:.2f}s | 4×回放 / 固定因果时刻离线估计 / 世界位姿保持',
                        labels)
                    draw=ImageDraw.Draw(im)
                    for panel in range(3):
                        x=24+532*panel
                        if panel and k<0:
                            image=Image.new('RGB',(PW,PH),BG)
                            text(ImageDraw.Draw(image),(90,260),f'积累历史 {frame+1}/32 帧',24)
                            im.paste(image,(8+532*panel,144));continue
                        state=truth if panel==0 else states[panel-1][k]
                        center,Q,size,mass=state;quat=Rotation.from_matrix(Q).as_quat()
                        for primitive in world.object_node.mesh.primitives:
                            primitive.material.baseColorFactor=[*(np.array(TEAL if panel==0 else AMBER)/255),1]
                        world.set(trace['hand_pose_w'][frame],center,quat,size,
                            trace['contact_position_w'][frame],trace['normal_load_n'][frame])
                        world.camera_at(camera_target+[.8,-1.3,.5],camera_target)
                        rgb,depth=renderer.render(world.scene)
                        if rgb.std()<5 or not np.isfinite(depth).all():raise RuntimeError('Invalid actual mesh render')
                        image=Image.fromarray(rgb)
                        if panel:world.extent_guide(image,truth[0],Rotation.from_matrix(truth[1]).as_quat(),truth[2])
                        im.paste(image,(8+532*panel,144))
                        if panel==0:
                            loads=trace['normal_load_n'][frame].reshape(2,27).sum(1)
                            text(draw,(x,752),f'真实质量 {mass:.3f} kg',25)
                            text(draw,(x,794),f'实际最低点 {trace["validation_full_mesh_min_z_m"][frame]*100:.1f} cm',22)
                            text(draw,(x,836),f'掌侧载荷 {loads[0]:.1f} / {loads[1]:.1f} N',22)
                            text(draw,(x,878),'本例控制资格：'+('通过' if records[episode]['controller_passed'] else '失败'),21)
                        else:
                            p=float(probabilities[panel-1][k]);error=pose_error(state,truth)
                            if conditional:
                                context=contexts[panel-1]
                                text(draw,(x,726),f'H32接触 {context["contact_history_frames"]}/32 · '+evidence_label(context)[1],16)
                            text(draw,(x,752),f'模型质量输出 {mass:.3f} kg',24,AMBER)
                            text(draw,(x,794),f'上次质量可用性 {p:.1%} · '+('判为可用' if p>=.5 else '暂不确定'),20)
                            text(draw,(x,836),f'位置 {error["center_cm"]:.1f} cm / 原始角误差 {error["rotation_deg"]:.1f}°',19)
                            age=trace['timestamp_s'][frame]-trace['timestamp_s'][clocks[k]]
                            if conditional:
                                text(draw,(x,878),f'预测时刻 {context["prediction_time_s"]:.2f}s · 已保持 {context["age_s"]:.2f}s',18)
                            else:
                                text(draw,(x,878),f'距估计 {age:.2f}s；形状为已知完整网格',19)
                    note=('受控TRAIN：无接触为先验/未知，有接触仍仅候选；完整已知网格。青虚线仅GT评价；误差含保持延迟，无未来、GT配准或跟手修正。'
                          if conditional else '固定16训练配置；青色虚线=真值评价参考。画面误差含估计保持延迟，区别于同刻验收；无未来帧、真值配准或跟手修正。')
                    text(draw,(24,924),note,16)
                    if frame in (31,1181,1581,2381):
                        filename=f'episode_{episode}_frame_{frame:04d}.png';im.save(out/filename);stills.append(filename)
                    writer.append_data(np.asarray(im));count+=1
                print('SEMANTIC_STATE_RENDER',episode,count,flush=True)
    finally:
        renderer.delete()
    if count!=len(episodes)*len(frames):raise ValueError('Incomplete expected episodes')
    decode_movie(movie,frames,count,fps,single_thread=shard_index is not None)
    rendered_result=dict(complete=True,frames=count,fps=fps,
        all_frames_decoded=True,cases=video_cases,stills=stills,actual_mesh_stills=len(stills),
        renderer=device,playback_speed=4,simulation_frames_per_second=5,inference_hz=None,inference_schedule="119 fixed causal clocks per48s case",
        holding='Previous prediction in world coordinates; current hand poses independent',
        displayed_errors='Held prior-clock WORLD prediction versus current-frame truth; includes hold latency, unlike same-clock numerical gates',
        truth_guide='Cyan dashed extent guide is GT for evaluation only, never used to correct prediction',
        availability_clock='Learned probability at last inference clock, not current-frame ground truth',
        source_optimizer_updates=2100, endpoint_optimizer_updates=2120,
        matched_updates_per_arm=20, matched_training_microbatches_per_arm=2320,conditions=list(CONDITIONS),
        new_physics_controls=0,new_optimizer_updates=0,new_model_forwards=0)
    if conditional:
        rendered_result.update(supervision_profile=protocol['supervision_profile'],state_evidence_status_codes=EVIDENCE_CODES,
            evidence_clock='Saved H32 evidence at the held prediction clock; never recomputed from current-frame truth or future touch',
            evidence_before_after_exact=True,
            unknown_display='Prior/unknown still renders the actual uncorrected model output and all errors; not a precise or confident state',
            contact_display='Contact candidate, observability unproven; state evidence is independent of learned mass availability',
            original_blind_or_all_state_success_claimed=False)
    if shard_index is not None:
        # A successful worker is not a successful full-sixteen delivery.
        rendered_result.pop('complete')
        rendered_result.update(shard_complete=True,render_scope='episode_shard',
            shard_index=shard_index,episode_order=list(episodes),
            physical_frame_clocks=frames,inference_frame_clocks=list(INFERENCE_FRAMES),
            worker_thread_limits=WORKER_THREAD_LIMITS,ffmpeg_threads=1)
        (out/'RESULT.json').write_text(json.dumps(rendered_result,indent=2)+'\n')
    else:
        write_delivery(root,out,rendered_result)


def write_delivery(root,out,rendered_result):
    """Publish the same full-run delivery only after complete-video validation."""
    conditional=rendered_result.get('supervision_profile') in CONDITIONAL_PROFILES
    video_cases=rendered_result['cases']
    (out/'RESULT.json').write_text(json.dumps(rendered_result,indent=2)+'\n')
    buttons=''.join(f'<button onclick="document.querySelector(\'video\').currentTime={r["video_start_s"]}">{r["episode"]}</button>' for r in video_cases)
    introduction='中列与右列均从完整2100步Utonia和Adam出发，在同464时刻、相同语义目标训练20次整批更新。中列观测分支置零，右列输入逐帧力汇总和真实手高度；两侧均保留完整官方骨干。固定训练轨迹及已查看的开发插值，不是未见物体测试。质量可用性为上次估计的学习概率；接触候选不代表状态已可辨识。全部失败与未知阶段保留。青色虚线只作真值评价，不修正模型输出；世界位姿保持包含推断延迟。视频无新物理或新模型推断。'
    page_title='明确观测输入对照';heading='固定16例：真实物体、无观测分支与明确观测分支'
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+page_title+'</title><style>body{max-width:1600px;margin:25px auto;background:#f2f6f9;font:18px/1.6 system-ui}video,img{width:100%}button{margin:5px;padding:8px}</style><h1>'+heading+'</h1><p>'+introduction+'</p><video controls src="renders/fixed16_semantic_observation_comparison.mp4"></video>'+buttons)


def merge_shard_results(reports, episodes):
    """Validate all four actual receipts; final video decoding remains required."""
    shards=episode_shards(episodes,4)
    if len(reports)!=4:raise ValueError('Require all four rendered shards')
    dynamic={'shard_complete','render_scope','shard_index','episode_order',
             'physical_frame_clocks','inference_frame_clocks','worker_thread_limits',
             'ffmpeg_threads','frames','cases','stills','actual_mesh_stills','all_frames_decoded'}
    common=None;cases=[];stills=[];count=0
    frames=list(range(1,2400,10));clocks=list(INFERENCE_FRAMES)
    for index,(report,shard) in enumerate(zip(reports,shards,strict=True)):
        if (report.get('complete') is not None or report.get('shard_complete') is not True
                or report.get('render_scope')!='episode_shard' or report.get('shard_index')!=index
                or report.get('episode_order')!=list(shard) or report.get('frames')!=960
                or report.get('fps')!=20 or report.get('all_frames_decoded') is not True
                or report.get('physical_frame_clocks')!=frames
                or report.get('inference_frame_clocks')!=clocks
                or report.get('worker_thread_limits')!=WORKER_THREAD_LIMITS
                or report.get('ffmpeg_threads')!=1):
            raise ValueError(f'Incomplete or mismatched render shard {index}')
        if [case['episode'] for case in report['cases']]!=list(shard):
            raise ValueError('Shard episode ordering differs')
        expected_stills=[f'episode_{episode}_frame_{frame:04d}.png'
                         for episode in shard for frame in (31,1181,1581,2381)]
        if report['stills']!=expected_stills or report['actual_mesh_stills']!=16:
            raise ValueError('Missing or reordered fixed still clocks')
        static={key:value for key,value in report.items() if key not in dynamic}
        if common is not None and static!=common:
            raise ValueError('Render shards used different display/evidence contracts')
        common=static
        for case_index,case in enumerate(report['cases']):
            if case['frames']!=240 or case['video_start_s']!=case_index*12:
                raise ValueError('Shard timeline differs')
            cases.append(dict(case,video_start_s=(count+case_index*240)/20))
        stills.extend(report['stills']);count+=report['frames']
    if count!=3840 or [case['episode'] for case in cases]!=list(episodes) or len(stills)!=64:
        raise ValueError('Incomplete fixed-sixteen delivery')
    return dict(common,frames=count,cases=cases,stills=stills,actual_mesh_stills=len(stills),
        parallel_workers=4,worker_thread_limits=WORKER_THREAD_LIMITS,ffmpeg_threads=1,
        physical_frame_clocks=frames,inference_frame_clocks=clocks,
        concatenation='Fixed episode order, ffmpeg stream copy; final complete decode required')


def render_parallel(root):
    """Four clean processes, one retained step and inherited lock, no parent GL."""
    from .retained_execution import require_active_resource
    from .overfit_data import FIXED_EPISODES
    require_active_resource();verify_pair(root)
    root=root.resolve();out=root/'renders';out.mkdir(exist_ok=False)
    parts=out/'parts';parts.mkdir()
    children=[];logs=[];receipts=[]
    try:
        for index in range(4):
            log=(parts/f'part_{index:02d}.log').open('wb');logs.append(log)
            command=[sys.executable,'-P','-m','scripts.sugar.object_predictor.render_semantic_state_pair',
                     '--root',str(root),'--shard-index',str(index)]
            child=subprocess.Popen(command,env=worker_environment(os.environ),pass_fds=(9,),
                                   stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            print('STATE_RENDER_WORKER_STARTED',index,child.pid,flush=True)
        # Wait for every own child even on a failed shard. Never signal the
        # retained step, its process group, allocation or sibling queued jobs.
        for index,child in enumerate(children):
            code=child.wait();receipts.append(dict(shard_index=index,pid=child.pid,returncode=code))
            print('STATE_RENDER_WORKER_EXIT',index,code,flush=True)
    except BaseException:
        # Only startup failure/interruption needs cleanup; normal child failures
        # take the all-terminal path above and preserve every partial artifact.
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        for log in logs:log.close()
    (parts/'WORKERS.json').write_text(json.dumps(receipts,indent=2)+'\n')
    if any(receipt['returncode']!=0 for receipt in receipts):
        raise RuntimeError('A render shard failed; all workers ended, partial results retained')
    paths=[parts/f'part_{index:02d}' for index in range(4)]
    reports=[json.loads((path/'RESULT.json').read_text()) for path in paths]
    merged=merge_shard_results(reports,FIXED_EPISODES)
    movies=[path/'fixed16_semantic_observation_comparison.mp4' for path in paths]
    if not all(movie.is_file() and movie.stat().st_size>0 for movie in movies):
        raise ValueError('A rendered shard movie is missing')
    listing=parts/'concat.txt'
    listing.write_text(''.join("file '"+str(movie).replace("'", "'\\''")+"'\n" for movie in movies))
    import imageio_ffmpeg
    movie=out/'fixed16_semantic_observation_comparison.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error',
                    '-threads','1','-f','concat','-safe','0','-i',str(listing),
                    '-map','0:v:0','-c','copy','-an','-threads','1',str(movie)],check=True)
    decode_movie(movie,merged['physical_frame_clocks'],3840,20,single_thread=True)
    for path,report in zip(paths,reports,strict=True):
        for name in report['stills']:
            if not (path/name).is_file():raise ValueError('Missing actual still file')
            shutil.copyfile(path/name,out/name)
    # Guard and source/artifact binding still hold at publication, after every
    # worker and ffmpeg process has exited. No additional allocation is taken.
    require_active_resource();verify_pair(root)
    merged.update(complete=True,all_frames_decoded=True,
                  worker_receipts='parts/WORKERS.json',parts_preserved=True)
    write_delivery(root,out,merged)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,4),default=1,
                        help='Four Mesa processes use complete-episode shards; default preserves serial replay')
    parser.add_argument('--shard-index',type=int,choices=range(4),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.shard_index is not None:
        if args.workers!=1:parser.error('Internal shard cannot launch more workers')
        import signal
        def terminate_worker(signum,_frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,terminate_worker)
        os.environ.update(WORKER_THREAD_LIMITS)
        render(args.root,args.shard_index)
    elif args.workers==4:render_parallel(args.root)
    else:render(args.root)
