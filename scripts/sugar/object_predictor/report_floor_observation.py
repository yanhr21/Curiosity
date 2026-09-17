"""Saved matched endpoints only; no model forward, training, physics or GL."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .render_state_floor_observation import comparison_baseline, read


def metrics(a):
    state = a['state_precision_eligible'] > .5
    mass = a['mass_available'] > .5
    predicted = a['availability_probability'] >= .5
    return dict(
        state_rows=int(state.sum()), mass_rows=int(mass.sum()),
        center_cm=float(a['center_cm'][state].mean()) if state.any() else None,
        mesh_cm=float(a['mesh_nn_cm'][state].mean()) if state.any() else None,
        mass_percent=100*float(a['mass_relative'][mass].mean()) if mass.any() else None,
        mass_p95_percent=100*float(np.quantile(a['mass_relative'][mass], .95)) if mass.any() else None,
        force_rmse_n=float(np.sqrt(np.mean(a['force_rmse_n']**2))),
        false_confident_percent=100*float(predicted[~mass].mean()) if (~mass).any() else None,
        false_confident_count=int((predicted & ~mass).sum()),
        availability_accuracy_percent=100*float((predicted == mass).mean()))


def report(root):
    from .overfit_floor_observation import verify_artifacts
    verify_artifacts(root)
    baseline = comparison_baseline(root)
    result = json.loads((root/'RESULT.json').read_text())
    if not result['execution_complete']:
        raise ValueError('No complete actual floor endpoint')
    sources = [root/'RESULT.json', root/'PROTOCOL.json', baseline/'RESULT.json']
    roles = {}
    label_fields = ('episode', 'frame', 'timestamp_s', 'target', 'force_target_n',
                    'mass_available', 'mass_status', 'contact_present',
                    'state_precision_eligible', 'state_contact_history_frames', 'state_evidence_status')
    text = ['# 地面高度观测：同预算实际对照', '',
            '两组均恢复同一原2000步完整Utonia和Adam，固定100次整批更新。中间差异为新增当前左手相对公开地面的高度观测与23个零初始化读出权重；不使用物体真值高度。', '',
            f"新条件原门槛验收：**{'PASS' if result['acceptance_passed'] else 'FAIL'}**。原物理15/16，5014失败保留；这是固定训练配置及同轨迹插值，不是泛化或触觉增益。", '']
    for stem in ('fit_02100', 'same_trajectory_interpolation'):
        paths = [base/(stem+'.npz') for base in (baseline, root)]
        arrays = [read(path) for path in paths]
        sources.extend(paths)
        if any(not np.array_equal(arrays[0][key], arrays[1][key]) for key in label_fields):
            raise ValueError('Matched endpoint labels or clocks differ: '+stem)
        rows = {}
        for episode in dict.fromkeys(arrays[0]['episode'].tolist()):
            rows[str(episode)] = [metrics({key:value[a['episode']==episode] for key,value in a.items()}) for a in arrays]
        roles[stem] = dict(all=[metrics(a) for a in arrays], per_case=rows,
                           clocks_targets_masks_exact=True)
        text += [f'## {stem}', '', '|条件|候选中心 cm|候选网格 cm|可用质量均值 %|可用质量p95 %|力 RMSE N|不可用误判 %|',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for title, m in zip(('未补高度', '补入高度'), roles[stem]['all']):
            values = [m[key] for key in ('center_cm', 'mesh_cm', 'mass_percent', 'mass_p95_percent', 'force_rmse_n', 'false_confident_percent')]
            text.append('|'+title+'|'+'|'.join('N/A' if v is None else f'{v:.4f}' for v in values)+'|')
        text += ['', '每例完整结果与分母见 MATCHED_COMPARISON.json；缺少可用质量标签为N/A，不计作成功。', '']
    text += ['[实际渲染](index.html)：未补高度2100端点对比补入高度2100端点，均保留原错误、未知阶段与物理失败。视频完成与否以 renders/RESULT.json 为准。', '',
             '本对照只验证这种线性加法观测接入。负结果不能证明地面信息无用，也不能证明所有误判都由此前高度丢失造成。', '']
    record = dict(complete=True, model_forwards=0, model_updates=0, physics_controls=0,
                  matched_baseline=str(baseline), roles=roles, acceptance_passed=result['acceptance_passed'],
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root/'MATCHED_COMPARISON.json').write_text(json.dumps(record, indent=2)+'\n')
    (root/'MATCHED_COMPARISON.md').write_text('\n'.join(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
