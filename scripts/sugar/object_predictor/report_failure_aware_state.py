"""Saved-only report: all sixteen cases, actual clocks, no model or physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(root):
    result = json.loads((root / 'RESULT.json').read_text())
    if not result['execution_complete']:
        raise ValueError('Require completed actual endpoint evaluation')
    paths = [root / 'initial_fit.npz'] + [root / f'fit_{step:05d}.npz' for step in (500, 1000, 1500, 2000)]
    curves = []
    for step, path in zip((0, 500, 1000, 1500, 2000), paths):
        with np.load(path, allow_pickle=False) as z:
            state = z['state_precision_eligible'] > .5
            mass = z['mass_available'] > .5
            negative = ~mass
            curves.append(dict(step=step, center_cm=float(z['center_cm'][state].mean()),
                mesh_cm=float(z['mesh_nn_cm'][state].mean()),
                mass_pct=100*float(z['mass_relative'][mass].mean()),
                force_rmse_n=float(np.sqrt(np.mean(z['force_rmse_n']**2))),
                false_confident_pct=100*float((z['availability_probability'][negative] >= .5).mean()),
                state_count=int(state.sum()), mass_count=int(mass.sum())))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, key, title in zip(axes.flat, ('center_cm', 'mesh_cm', 'mass_pct', 'force_rmse_n', 'false_confident_pct'),
            ('Contact candidate center error (cm)', 'Contact candidate mesh error (cm)',
             'Available mass error (%)', 'Observed force RMSE (N)', 'Unavailable mass: false confidence (%)')):
        ax.plot([r['step'] for r in curves], [r[key] for r in curves], marker='o')
        ax.set(title=title, xlabel='Actual full-model updates'); ax.grid(alpha=.25)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .9, 'All 16 attempts retained\n80 fixed fit clocks\n'
        f"{curves[-1]['state_count']} state candidates; {curves[-1]['mass_count']} mass labels\n"
        'Same-trajectory evaluation is not generalization\nPhysical failure 5014 remains a failure', va='top')
    fig.suptitle('Full official Utonia: fixed failure-aware overfit (saved outputs only)')
    fig.savefig(root / 'fit_learning_curves.png', dpi=160); plt.close(fig)
    text = ['# 完整 Utonia 固定小样本修复验收', '',
            f"实际执行完成；总体学习验收 **{'PASS' if result['acceptance_passed'] else 'FAIL'}**。固定2000更新，16例全部保留。",
            '原物理资格15/16，5014失败仍保留。此处仅测试已知网格状态拟合和同轨迹插值，不代表任意形状重建、泛化或触觉增益。', '',
            '初始对照为官方预训练 Utonia 骨干加新初始化的任务读出层，不是历史上已训练的最佳 predictor。因此前后改善只说明本轮学习有效，不证明优于旧模型。', '',
            '[训练曲线](fit_learning_curves.png)。视频由独立实际网格渲染阶段提供，其完成情况以 renders 内回执为准。', '']
    for title, name in (('80个固定拟合时刻', 'fit_02000'), ('1440个同轨迹插值时刻', 'same_trajectory_interpolation')):
        summary_path = root / (name + '.json'); paths.append(summary_path)
        summary = json.loads(summary_path.read_text())
        text += [f'## {title}', '', '|案例|有接触候选数|中心均值 cm|网格均值 cm|可用质量数|质量误差 %|力 RMSE N|验收|', '|---|---:|---:|---:|---:|---:|---:|---|']
        for episode, row in summary['per_case'].items():
            state = row['state_candidate_metrics']; mass = row['metrics']['available_mass_relative']
            center = f"{state['center_cm']['mean']:.3f}" if state['center_cm'] else 'N/A'
            mesh = f"{state['mesh_nn_cm']['mean']:.3f}" if state['mesh_nn_cm'] else 'N/A'
            mass_error = f"{100*mass['mean']:.2f}" if mass is not None else 'N/A'
            text.append(f"|{episode}|{row['state_candidate_rows']}|{center}|{mesh}|{row['available_mass_denominator']}|{mass_error}|{row['metrics']['force_rmse_n']:.3f}|{'PASS' if row['passed'] else 'FAIL'}|")
        text += ['', '质量 N/A 表示没有可用监督，不能记作质量通过；有接触只是状态回归候选条件，不是完整可辨识保证。所有原始预测和未筛选误差保存在对应 NPZ/JSON。', '']
    text += ['检查点重载、全参数更新、输入和标签一致性结果见 [RESULT.json](RESULT.json) 和 [PARAMETER_UPDATES.json](PARAMETER_UPDATES.json)。', '']
    (root / 'NUMERICAL_REPORT.md').write_text('\n'.join(text))
    paths += [root/'RESULT.json', root/'PARAMETER_UPDATES.json']
    (root / 'SAVED_REPORT_READBACK.json').write_text(json.dumps(dict(
        model_forwards=0, model_updates=0, physics_controls=0,
        sources={p.name:digest(p) for p in paths}, curves=curves,
        acceptance_passed=result['acceptance_passed']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
