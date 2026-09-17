"""Evaluate frozen fresh-config predictions, with separate update/held clocks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data import target_at
from .report_grip_transfer import errors
from .shape_metric import rotation


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def world_state(prediction, hand):
    """Bind a hand-relative prediction to the hand at its OWN inference time."""
    Rh = Rotation.from_quat(hand[3:]).as_matrix()
    return (Rh @ prediction[:3] + hand[:3], Rh @ rotation(prediction[3:9]),
            np.exp(prediction[9:12]), float(np.exp(prediction[12])))


def past_index(clocks, frame):
    """Return -1 before warmup, otherwise the last past (never future) clock."""
    return int(np.searchsorted(clocks, frame, side='right') - 1)


def world_truth(trace, frame):
    pose = trace['object_pose_w'][frame]
    Q = Rotation.from_quat(pose[3:]).as_matrix()
    return (pose[:3] + Q @ trace['object_local_center_m'][frame], Q,
            trace['object_dimensions_m'][frame], float(trace['object_mass_kg'][frame]))


def angle_deg(Q):
    return float(np.linalg.norm(Rotation.from_matrix(Q).as_rotvec()) * 180 / np.pi)


def pose_error(estimate, truth):
    return dict(center_cm=float(np.linalg.norm(estimate[0]-truth[0])*100),
                rotation_deg=angle_deg(estimate[1] @ truth[1].T))


def minimum_z(vertices, dims, state):
    center, Q, size, _ = state
    return float(((vertices*(size/dims)) @ Q.T + center)[:, 2].min())


def mean_dict(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else None


def main(root):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Evaluate in the retained compute step')
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    fitted = json.loads((root/'fusion/RESULT.json').read_text())
    inputs = json.loads((root/'inputs/RESULT.json').read_text())
    collection = json.loads((root/'COLLECTION_RESULT.json').read_text())
    assert fitted['complete'] and inputs['complete'] and collection['complete']
    configs = protocol['configurations']; clocks = np.array(protocol['prediction_frames'])
    expected = [(c['episode'], int(f)) for c in configs for f in clocks]
    assert [(r['episode'], r['frame']) for r in fitted['cases']] == expected
    lookup = {(r['episode'], r['frame']): r for r in fitted['cases']}
    records = {r['episode']: r for r in collection['records']}
    mesh = read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    vertices = mesh['vertices'].astype(float); dims = np.ptp(vertices, axis=0)
    vertices -= (vertices.max(0)+vertices.min(0))/2
    episodes = []; latencies = []
    for config in configs:
        ep = config['episode']; source = Path(records[ep]['source'])
        trace = read(source/f'episode_{ep}.npz')
        supervision = read(source/'supervision.npz')
        rows = [lookup[(ep, int(f))] for f in clocks]
        states = dict(utonia=[], chsel=[]); contact = []
        for frame, row in zip(clocks, rows):
            saved = read(root/'fusion'/f'episode_{ep}_frame_{frame}.npz')
            assert np.array_equal(saved['hand_pose_w'], trace['hand_pose_w'][frame])
            assert float(saved['timestamp_s']) == float(trace['timestamp_s'][frame])
            contact.append(len(saved['points_current_left_hand_m']) > 0)
            if not contact[-1]:
                assert np.array_equal(saved['prediction'], saved['original_prediction'])
            assert np.array_equal(saved['prediction'][9:], saved['original_prediction'][9:])
            for arm, key in [('utonia','original_prediction'), ('chsel','prediction')]:
                states[arm].append(world_state(saved[key], trace['hand_pose_w'][frame, 0]))
                # Independent world transform checks the solver's full-mesh floor diagnostic.
                z = minimum_z(vertices, dims, states[arm][-1])
                index = 0 if arm == 'utonia' else 1
                assert abs(z-row['minimum_world_z_m_prior_then_selected'][index]) < 1e-5
                target = target_at(trace, frame)
                checked = errors(saved[key][None], target[None])
                for key_, value in checked.items():
                    assert abs(float(value[0])-row[arm][key_]) < 1e-3
            if contact[-1]:
                latencies.append(row['seconds_including_sdf_and_checks'])
        contact = np.array(contact, bool)
        airborne = supervision['mass_label_weight'][clocks] > 0
        truth = [world_truth(trace, f) for f in clocks]
        item = dict(episode=ep, geometry_group=config['geometry_group'],
                    controller_passed=records[ep]['controller_passed'],
                    inference_clocks=len(clocks), contact_clocks=int(contact.sum()),
                    airborne_hold_clocks=int(airborne.sum()), arms={})
        for arm in states:
            z = np.array([minimum_z(vertices, dims, s) for s in states[arm]])
            subsets = {}
            for label, mask in [('all',np.ones(len(clocks),bool)), ('contact',contact),
                                ('no_contact',~contact), ('airborne_hold',airborne)]:
                selected = np.flatnonzero(mask)
                subsets[label] = dict(clocks=len(selected),
                    errors=mean_dict([rows[i][arm] for i in selected]),
                    penetration_beyond1mm_fraction=float(np.mean(z[mask] < -.001)) if len(selected) else None)
            deltas = []
            for i in range(1, len(clocks)):
                estimated_dc = states[arm][i][0]-states[arm][i-1][0]
                actual_dc = truth[i][0]-truth[i-1][0]
                estimated_dQ = states[arm][i][1] @ states[arm][i-1][1].T
                actual_dQ = truth[i][1] @ truth[i-1][1].T
                deltas.append(dict(translation_increment_error_cm=float(np.linalg.norm(estimated_dc-actual_dc)*100),
                                   rotation_increment_error_deg=angle_deg(estimated_dQ @ actual_dQ.T)))
            held = []; held_z = []
            for frame in range(1, 2400, protocol['render']['recorded_frame_stride']):
                k = past_index(clocks, frame)
                if k < 0:
                    continue
                assert clocks[k] <= frame and (k+1 == len(clocks) or clocks[k+1] > frame)
                held.append(pose_error(states[arm][k], world_truth(trace, frame)))
                held_z.append(minimum_z(vertices, dims, states[arm][k]))
            item['arms'][arm] = dict(subsets=subsets, world_increment=mean_dict(deltas),
                render_held_world_error=mean_dict(held), render_held_frames=len(held),
                render_held_penetration_beyond1mm_fraction=float(np.mean(np.array(held_z)<-.001)))
        episodes.append(item)
        print('PROSPECTIVE_EPISODE_REPORT', json.dumps(item), flush=True)
    aggregate = {arm: mean_dict([r['arms'][arm]['subsets']['all']['errors'] for r in episodes]) for arm in states}
    subset_summary = {}
    for label in ('all','contact','no_contact','airborne_hold'):
        subset_summary[label] = {}
        for arm in states:
            covered = [r['arms'][arm]['subsets'][label] for r in episodes
                       if r['arms'][arm]['subsets'][label]['clocks'] > 0]
            subset_summary[label][arm] = dict(episodes_covered=len(covered), total_episodes=len(episodes),
                clocks=sum(r['clocks'] for r in covered), equal_episode_errors=mean_dict([r['errors'] for r in covered]),
                equal_episode_penetration_beyond1mm_fraction=float(np.mean([
                    r['penetration_beyond1mm_fraction'] for r in covered])) if covered else None)
    groups = {str(g): {arm: mean_dict([r['arms'][arm]['subsets']['all']['errors'] for r in episodes if r['geometry_group']==g])
                      for arm in states} for g in sorted({c['geometry_group'] for c in configs})}
    joint_wins = sum(all(r['arms']['chsel']['subsets']['all']['errors'][k] < r['arms']['utonia']['subsets']['all']['errors'][k]
                         for k in ('center_cm','symmetric_nearest_vertex_cm')) for r in episodes)
    checks = dict(center_mean_improves=aggregate['chsel']['center_cm']<aggregate['utonia']['center_cm'],
        mesh_mean_improves=aggregate['chsel']['symmetric_nearest_vertex_cm']<aggregate['utonia']['symmetric_nearest_vertex_cm'],
        rotation_mean_nonregression=aggregate['chsel']['rotation_deg']<=aggregate['utonia']['rotation_deg'],
        center_and_mesh_episode_wins_min=joint_wins>=protocol['acceptance']['center_and_mesh_episode_wins_min'],
        contact_clocks_no_penetration_beyond1mm=all(
            r['arms']['chsel']['subsets']['contact']['penetration_beyond1mm_fraction'] in (None,0.) for r in episodes)
            and any(r['contact_clocks']>0 for r in episodes))
    forward_seconds = [r['model_forward_and_cpu_transfer_seconds'] for r in inputs['cases']]
    result = dict(complete=True, equal_episode_all_clocks=aggregate, geometry_groups=groups,
        subset_summary=subset_summary,
        equal_episode_world_increment={arm:mean_dict([r['arms'][arm]['world_increment'] for r in episodes]) for arm in states},
        equal_episode_render_held_error={arm:mean_dict([r['arms'][arm]['render_held_world_error'] for r in episodes]) for arm in states},
        episodes=episodes, center_and_mesh_episode_wins=joint_wins,
        relative_improvement_checks=checks, relative_improvement_passed=all(checks.values()),
        reliable_carry_state_estimation_established=False,
        acquisition_passes=sum(r['controller_passed'] for r in episodes), attempted_episodes=len(episodes),
        actual_airborne_hold_episodes=sum(r['airborne_hold_clocks']>0 for r in episodes),
        offline_registration_seconds=dict(count=len(latencies), median=float(np.median(latencies)) if latencies else None,
                                         maximum=float(max(latencies)) if latencies else None),
        frozen_model_forward_and_cpu_transfer_seconds=dict(count=len(forward_seconds),
            median=float(np.median(forward_seconds)), maximum=float(max(forward_seconds)),
            excludes='Input preprocessing, repeat validation forwards and contact registration; not end-to-end latency'),
        model_parameter_updates=0, scope=protocol['scope'], acceptance_scope=protocol['acceptance_scope'])
    (root/'COMPARISON.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines=['# 新配置连续搬运：冻结预测与接触/地面融合', '',
           '16 个配置来自 4 个几何/轨迹组 × 2 质量 × 2 握力；所有采集失败和零接触时刻均保留。', '',
           '| 方法 | 中心 cm | 旋转 ° | 网格距离 cm | 尺寸 % | 质量 % |', '|---|---:|---:|---:|---:|---:|']
    for arm,label in [('utonia','冻结 Utonia'),('chsel','固定先验/接触/地面融合')]:
        s=aggregate[arm]
        lines.append('| '+label+' | '+' | '.join(f'{s[k]:.3f}' for k in ('center_cm','rotation_deg','symmetric_nearest_vertex_cm','size_pct','mass_pct'))+' |')
    lines += ['', f'相对改善验收：{result["relative_improvement_passed"]}；中心和网格同时改善 {joint_wins}/16。',
              f'实际采集通过 {result["acquisition_passes"]}/16；有离地保持评价时刻 {result["actual_airborne_hold_episodes"]}/16。', '',
              '模型没有更新，质量和尺寸也没有经过融合修正。验收仅针对组合方法的相对改善；不证明任意形状、材质、真实触觉或 demo following 收益。', '',
              '每 1 秒仿真时间离线估计一次，预测时刻为 0.64–47.64 秒，不代表实时速度。视频在更新之间保持上次世界位姿，显示估计年龄，未跟随当前手或使用未来预测。', '',
              '[全部结果、四组汇总和失败项](COMPARISON.json) · [视频播放页](index.html)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
