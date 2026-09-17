"""Report every matched acquisition outcome; geometry admission is not success."""
import argparse
import json
from pathlib import Path

import numpy as np

from .collect_response_corpus import write_json


def summarize(case, episode):
    result=json.loads((case/'RESULT.json').read_text())
    with np.load(case/f'episode_{episode}.npz',allow_pickle=False) as saved:
        a={k:saved[k] for k in ('timestamp_s','normal_load_n','validation_controller_fit_valid',
                              'validation_controller_ready_seconds','validation_controller_lift_start_s',
                              'validation_controller_response_gain_m_per_ns','validation_full_mesh_min_z_m')}
    loads=a['normal_load_n'].reshape(-1,2,27).sum(2)
    lift=a['validation_controller_lift_start_s'][-1]
    return dict(controller_passed=result['passed'],checks=result['checks'],values=result['values'],
                lift_start_s=float(lift),fit_valid_frames=a['validation_controller_fit_valid'].sum(0).tolist(),
                both_fit_fraction=float(a['validation_controller_fit_valid'].all(1).mean()),
                max_ready_seconds=float(a['validation_controller_ready_seconds'].max()),
                peak_hand_load_n=loads.max(0).tolist(),final_hand_load_n=loads[-1].tolist(),
                final_full_mesh_clearance_cm=float(a['validation_full_mesh_min_z_m'][-1]*100),
                max_response_gain_m_per_ns=a['validation_controller_response_gain_m_per_ns'].max(0).tolist())


def main(root):
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    collection=json.loads((root/'COLLECTION_RESULT.json').read_text())
    assert collection['complete'] and len(collection['records'])==2
    rows=[]
    for config in protocol['configurations']:
        episode=config['episode']
        pair={arm:summarize(Path(path)/'cases'/f'episode_{episode}',episode)
              for arm,path in [('baseline',protocol['baseline_root']),('candidate',str(root))]}
        if pair['baseline']['checks'].keys()!=pair['candidate']['checks'].keys():
            raise ValueError('Changed physical acceptance criteria')
        rows.append(dict(episode=episode,config=config,**pair))
    # Only the original complete physical criteria decide this diagnostic.
    accepted=all(row['candidate']['controller_passed'] for row in rows)
    result=dict(complete=True,diagnostic_accepted=accepted,cases=rows,
                changed_rule='Plane-fit minimum current measured load: 2.0 -> 0.2 N only',
                new_physics_controls=4800,reused_baseline_controls=4800,model_forwards=0,model_parameter_updates=0,
                scope='Two predeclared post-hoc acquisition cases. No unseen configuration, predictor accuracy, material or demo-following benefit claim.')
    write_json(root/'COMPARISON.json',result)
    lines=['# 平面准入：固定单因素采集对照','',
           '只把平面拟合的当前载荷准入从2.0N降为已有接触锁存门槛0.2N。几何质量、ready/lift、目标载荷、原物理与稳定性门槛保持不变。',
           '6000为已有成功例，6004为已诊断失败例；选择使用了既有结果，所以这是事后机制诊断，不是新的泛化测试。原轨迹复用，候选各新增2400控制步。','',
           '| 案例 | 条件 | 完整物理门槛 | 抬升开始 s | 左/右有效拟合帧 | 最终离地 cm |','|---|---|---|---:|---|---:|']
    for row in rows:
        for arm,label in [('baseline','原2N'),('candidate','候选0.2N')]:
            value=row[arm]
            lines.append(f'| {row["episode"]} | {label} | {value["controller_passed"]} | {value["lift_start_s"]:.2f} | {value["fit_valid_frames"]} | {value["final_full_mesh_clearance_cm"]:.2f} |')
    lines+=['',f'两例均通过全部原物理门槛：**{accepted}**。',
            '更多拟合帧或开始抬升本身不构成通过；峰值载荷、稳定性、最终保持等失败项全部保留在COMPARISON.json。',
            '本轮无模型推理或训练；实际网格并排回放另由render_plane_gate生成。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('PLANE_GATE_COMPARISON',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
