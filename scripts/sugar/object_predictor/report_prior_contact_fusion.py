"""Report a finished fixed fusion qualification; no new fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    root=Path(args.root); result=json.loads((root/'RESULT.json').read_text())
    protocol=json.loads((root/'PROTOCOL.json').read_text()); floor=bool(protocol.get('known_floor'))
    scales=json.loads((root.parent/'fusion_scales_v1/RESULT.json').read_text())
    gradient=json.loads((root.parent/('FLOOR_FUSION_GRADIENT_QUALIFICATION.json' if floor else
                                    'FUSION_GRADIENT_QUALIFICATION.json')).read_text())
    assert result['complete'] and len(result['cases'])==8 and gradient['complete']
    old=json.loads((root.parent/'chsel_local_stage_v1_r1/RESULT.json').read_text())
    b,f=result['mean']['utonia'],result['mean']['chsel']
    improvements={k:sum(r['chsel'][k]<r['utonia'][k] for r in result['cases'])
                  for k in ('center_cm','rotation_deg','symmetric_nearest_vertex_cm')}
    table=['| 方法 | 位置 cm | 旋转 ° | 双向最近顶点 cm |', '|---|---:|---:|---:|']
    for name,m in [('冻结 Utonia',b),('仅接触的原初值局部终点',old['mean']['original_start_endpoint']),
                   ('先验＋接触融合＋已知地面' if floor else '先验＋接触融合',f)]:
        table.append(f'| {name} | {m["center_cm"]:.4f} | {m["rotation_deg"]:.4f} | {m["symmetric_nearest_vertex_cm"]:.4f} |')
    rows=['| 案例/帧 | 位置 原→融合 cm | 旋转 原→融合 ° | 网格 原→融合 cm |', '|---|---:|---:|---:|']
    for r in result['cases']:
        x,y=r['utonia'],r['chsel']
        rows.append(f'| {r["episode"]}/{r["frame"]} | {x["center_cm"]:.3f} → {y["center_cm"]:.3f} | '
                    f'{x["rotation_deg"]:.2f} → {y["rotation_deg"]:.2f} | '
                    f'{x["symmetric_nearest_vertex_cm"]:.3f} → {y["symmetric_nearest_vertex_cm"]:.3f} |')
    checks=max(x['max_scaled_difference'] for r in gradient['cases'] for x in r['checks'])
    text=['# Utonia 持续先验与接触几何融合：固定八帧资格', '',
          '这是新增的本项目融合适配层：完整冻结官方 Utonia、完整官方三角面查询、官方 TorchLie 旋转对数、'
          '不变的官方 CHSEL 局部优化器。不是原 CHSEL 损失，也不是新训练或简化替代网络。', '',*table,'',
          f'位置/旋转/网格距离改善帧数：{improvements["center_cm"]}/8、{improvements["rotation_deg"]}/8、'
          f'{improvements["symmetric_nearest_vertex_cm"]}/8。原先声明的平均位置与网格均改善判据：'
          f'{result["geometry_improves_mean_train_qualification"]}。', '',*rows,'',
          f'质量误差仍为 {b["mass_pct"]:.4f}%，尺寸误差仍为 {b["size_pct"]:.4f}%；两者未经此后端修正。', '',
          '## 固定设置与验证', '',
          '八帧均为5000..5003的1206/2206时刻。56个尺度估计时刻来自另外28条TRAIN轨迹5004..5031。'
          '位置残差使用物体中心，旋转使用当前左手坐标的Log(Q Q0ᵀ)。接触项每手按面积求平均，再跨手求和，'
          '空手跳过。每帧原预测加30个官方局部终点，只按总融合目标选择；八帧拟合全部保存后才读评价标签。', '',
          '位置每轴经验RMS(m)：'+str(scales['pose_scales_m_rad'][:3])+'；旋转(rad)：'+str(scales['pose_scales_m_rad'][3:])+'.',
          '两手接触有效残差尺度(m)：'+str(scales['contact_scales_m'])+'；其中包括预测尺寸偏差。', '',
          f'全部八帧、两个固定姿态、六个物理方向的有限差分通过，最大缩放差异{checks:.6g}；'
          'TorchLie零角、近零、近π短弧和零角Jacobian检查通过。该预检覆盖Float64物理扰动，'
          '不等于实际Float32 rotation6d优化每个状态的全域梯度认证。', '',
          '## 结论边界', '',
          '这些仍是TRAIN机制资格帧。误差尺度也来自模型训练数据，不能称泛化验证或校准置信区间。'
          'Utonia与接触项使用相关观测，不能称独立观测的严格Bayesian MAP。此轮同时改变接触残差形式、'
          '面积归一化并加入先验，任何改善只能归于整个融合方案，不能单独归因于某一个因素。', '',
          '当前输出仍是已知完整网格的位姿与原尺寸、质量。尚未证明未知形状重建、质量改善、材质辨识或demo following收益。'
          '八张实际完整网格静帧保留全部案例；它们不是新连续搬运视频。', '',
          '[八帧实际渲染](index.html) · [完整数值](RESULT.json)']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    if floor:
        floor_data=json.loads((root/'SAVED_FLOOR_READBACK.json').read_text())
        prior=json.loads((root.parent/'prior_contact_fusion_v1/RESULT.json').read_text())['mean']['chsel']
        z=[r['fusion_mesh_min_z_cm'] for r in floor_data['cases']]
        floor_pass=all(x>=-.1 for x in z)
        for row,zrow in zip(result['cases'],floor_data['cases']):
            assert abs(row['minimum_world_z_m_prior_then_selected'][1]*100-zrow['fusion_mesh_min_z_cm'])<1e-4
        extra=['\n## 已知地面因子与独立完整网格回读\n',
               '前一组无地面融合的位置/旋转/网格均值为'
               f'{prior["center_cm"]:.4f}cm / {prior["rotation_deg"]:.4f}° / {prior["symmetric_nearest_vertex_cm"]:.4f}cm，'
               '但四个落地帧仍穿地4.63–7.34cm，因此保留为不满足物理要求的部分正结果。',
               '本组唯一新增：场景已知z=0地面，完整预测网格的最低点负高度除以1mm后平方、乘0.5。'
               '不使用物体真值或是否落地的标签，所有八帧均启用，同一权重、不扫描。它是软约束，不保证解析意义的绝对零穿透。',
               f'八个预测最低点(cm)：{z}。预先声明的穿地不超过1mm检查：{floor_pass}。',
               f'平均位置/网格改善且全部八帧地面检查共同通过：{floor_pass and result["geometry_improves_mean_train_qualification"]}。'
               '这仍不是全场景物理可行性或泛化验收。']
        with (root/'REPORT.md').open('a') as handle:handle.write('\n\n'.join(extra)+'\n')
    html='<!doctype html><meta charset="utf-8"><title>物体先验与接触融合</title><style>body{max-width:1600px;margin:30px auto;font:18px/1.7 system-ui}img{width:100%}table{border-collapse:collapse}td,th{padding:8px 20px;border:1px solid #ccc}</style><h1>Utonia持续先验＋接触几何融合</h1><p>固定8个TRAIN时刻；权重来自另外56个TRAIN时刻。已知网格状态估计，非泛化、任意形状或新连续搬运视频。</p><p><a href="REPORT.md">完整报告与限制</a></p>'
    html+='<table><tr><th>指标</th><th>原Utonia</th><th>融合</th></tr>'
    for label,k in [('位置 cm','center_cm'),('旋转 °','rotation_deg'),('网格 cm','symmetric_nearest_vertex_cm')]:
        html+=f'<tr><td>{label}</td><td>{b[k]:.4f}</td><td>{f[k]:.4f}</td></tr>'
    html+='</table>'
    if floor:
        html+='<p>同时使用公开场景的地面约束；最低点按完整预测网格计算，图片中明确显示。'
        html+=f'全部8帧穿地不超过1mm：{floor_pass}。仍是TRAIN资格，质量与尺寸未修正。</p>'
    for r in result['cases']:
        html+=f'<h2>{r["episode"]}/{r["frame"]}</h2><img src="renders/episode_{r["episode"]}_frame_{r["frame"]}.png">'
    (root/'index.html').write_text(html)
    sources={}
    for name in ['fusion_cost.py','calibrate_fusion_scales.py','qualify_chsel_backend.py']:
        p=Path('scripts/sugar/object_predictor')/name
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        (root/('source_'+name)).write_bytes(p.read_bytes())
    assert sources['scripts/sugar/object_predictor/fusion_cost.py']==gradient['adapter_sha256']
    (root/'REPORT_BINDINGS.json').write_text(json.dumps(dict(sources=sources,
        result_sha256=hashlib.sha256((root/'RESULT.json').read_bytes()).hexdigest(),
        scales_sha256=hashlib.sha256((root.parent/'fusion_scales_v1/RESULT.json').read_bytes()).hexdigest(),
        torchlie_download=json.loads((root.parent/'backend_wheels/torchlie_download.json').read_text()),
        improvements=improvements),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(ap.parse_args())
