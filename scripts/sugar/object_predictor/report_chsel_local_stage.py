"""Read saved official local trajectories; no registration, learning or physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .qualify_chsel_backend import read, predict_from_inverse
from .report_grip_transfer import errors


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    root = Path(args.root)
    result = json.loads((root / 'RESULT.json').read_text())
    assert result['complete']
    labels = read(root.parent / 'chsel_qualification_inputs/evaluation_only.npz')
    baseline = json.loads((root.parent / 'chsel_qualification_v1_r1/RESULT.json').read_text())
    fig, axes = plt.subplots(8, 3, figsize=(15, 24))
    records = []
    for i, row in enumerate(result['cases']):
        name = f'episode_{row["episode"]}_frame_{row["frame"]}.npz'
        saved, trace = read(root / name), read(root / ('trace_' + name))
        original = saved['original_prediction']
        assert labels['episode'][i] == row['episode'] and labels['frame'][i] == row['frame']
        R, T = trace['first_start_R'], trace['first_start_T']
        h = np.tile(np.eye(4), (len(R) + 1, 1, 1))
        h[:-1, :3, :3], h[:-1, :3, 3] = R, T
        h[-1] = trace['local_endpoints_hand_to_object'][0]
        predictions = np.stack([predict_from_inverse(x, original) for x in h])
        stats = errors(predictions, np.repeat(labels['target'][i:i+1], len(h), axis=0))
        cost = np.r_[trace['first_start_cost'], row['original_start_final_cost']]
        prior_shift = np.linalg.norm(predictions[:, :3] - original[:3], axis=1) * 100
        np.savez_compressed(root / ('evaluated_trace_' + name),
                            cost=cost, prior_center_shift_cm=prior_shift, **stats)
        base_row = baseline['cases'][i]
        cost_diff = abs(row['original_cost'] - base_row['original_cost'])
        old_saved = read(root.parent / 'chsel_qualification_v1_r1' / name)
        assert np.array_equal(original, old_saved['original_prediction'])
        record = dict(episode=row['episode'], frame=row['frame'], calls=len(R),
            original_cost_difference_vs_prior_full_qd_arm=cost_diff,
            selected_prediction_max_absolute_difference_vs_full_qd=float(np.max(abs(
                saved['prediction'] - old_saved['prediction']))),
            first_update_center_shift_cm=float(prior_shift[1]),
            endpoint_center_shift_cm=float(prior_shift[-1]),
            initial_center_error_cm=float(stats['center_cm'][0]),
            endpoint_center_error_cm=float(stats['center_cm'][-1]),
            initial_rotation_error_deg=float(stats['rotation_deg'][0]),
            endpoint_rotation_error_deg=float(stats['rotation_deg'][-1]),
            initial_observation_cost=float(cost[0]), endpoint_observation_cost=float(cost[-1]))
        records.append(record)
        for j, (series, title) in enumerate([(cost / cost[0], 'Observation cost / initial'),
                                            (stats['center_cm'], 'Center error (cm)'),
                                            (stats['rotation_deg'], 'Rotation error (deg)')]):
            ax = axes[i, j]
            ax.plot(np.arange(len(series)), series, lw=1.3)
            ax.axhline(series[0], color='gray', ls='--', lw=.8)
            ax.scatter([len(series)-1], [series[-1]], color='red', s=12)
            ax.set_title(f'{row["episode"]}/{row["frame"]} | {title}', fontsize=10)
            ax.set_xlabel('Actual cost calls; final point is re-scored endpoint', fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('Official CHSEL local stage: exact Utonia start, no QD\n'
                 'All 8 fixed TRAIN clocks; labels used only in this saved-result analysis', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .975]); fig.savefig(root / 'original_start_trajectories.png', dpi=120)
    plt.close(fig)
    report = dict(complete=True, cases=records, registration_calls=0, model_updates=0, physics_steps=0,
        recorder='One original loss call, detached copied snapshots; added synchronization affects wall time',
        result_sha256=hashlib.sha256((root/'RESULT.json').read_bytes()).hexdigest())
    (root / 'TRAJECTORY_ANALYSIS.json').write_text(json.dumps(report, indent=2))
    lines = ['# 官方 CHSEL 局部阶段固定消融', '',
        '相同八个 TRAIN 时刻、完整预测网格、30 个相同规则初值和官方局部优化默认值；只跳过 QD。'
        '第 0 初值精确为原 Utonia。它的原始终点预先固定，不按标签选择。所有拟合结束后才读取真值。', '',
        '| 输出 | 位置 cm | 旋转 ° | 网格距离 cm |', '|---|---:|---:|---:|']
    for key, label in [('utonia','原 Utonia'),('original_start_endpoint','原初值局部终点'),
                       ('original_start_or_prior','原初值终点与原预测按观测择优'),('chsel','30 局部终点与原预测按观测择优')]:
        m = result['mean'][key]
        lines.append(f'| {label} | {m["center_cm"]:.4f} | {m["rotation_deg"]:.4f} | {m["symmetric_nearest_vertex_cm"]:.4f} |')
    lines += ['', '质量和尺寸保持原预测。这里没有持续先验因子、未知形状重建、训练或新物理采集；'
              '八帧是机制消融，不能声称泛化改善。', '',
              '轨迹的每一点来自记录器在官方损失调用时复制的状态，末点为重新评分后的最终姿态。'
              '官方返回的 t_history 平移可能与后续优化共享存储，未把它当成真实历史。'
              '记录器增加同步开销，耗时不用于宣称加速。', '',
              '首次运行在第一帧优化后保存 t_history 时失败；已保留原日志、源码和状态，'
              '修正仅移除该不可靠历史保存。见上级 LOCAL_STAGE_SAVE_INCIDENT.json。', '',
              '![原初值轨迹](original_start_trajectories.png)', '',
              '实际完整网格静帧另见 renders_original_start；静帧不等于连续搬运视频。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CHSEL 局部阶段</title>'
        '<style>body{max-width:1500px;margin:32px auto;font:18px/1.7 system-ui}img{width:100%}</style>'
        '<h1>原 Utonia 初值出发的官方局部优化</h1><p>八个固定 TRAIN 时刻；右侧显示预先固定的第0初值终点。'
        '这是完整网格静帧，非连续搬运视频。</p><p><a href="REPORT.md">完整报告</a></p>'
        + ''.join(f'<img src="renders_original_start/episode_{r["episode"]}_frame_{r["frame"]}.png">'
                  for r in records) + '<img src="original_start_trajectories.png">')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); main(ap.parse_args())
