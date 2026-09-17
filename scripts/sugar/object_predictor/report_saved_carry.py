"""Matched frozen predictor readback; preserve every case and missing-label count."""
import argparse,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .report_grip_transfer import errors

METRICS=('center_cm','rotation_deg','size_pct','mass_pct')


def main():
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    root=Path(args.output);p=json.loads((root/'PROTOCOL.json').read_text());rows=[];checks={}
    episodes=[e for e in p['episodes'] if (root/'evaluation'/f'episode_{e}'/'CONTEXT.json').exists()]
    if not args.partial and episodes!=p['episodes']:raise ValueError('Missing declared cases')
    for ep in episodes:
        case=root/'evaluation'/f'episode_{ep}';ctx=json.loads((case/'CONTEXT.json').read_text())
        source=json.loads((Path(p['source'])/'cases'/f'episode_{ep}'/'RESULT.json').read_text())
        t=np.array([r['timestamp_s'] for r in ctx]);valid=np.array([r['both_fit_valid'] for r in ctx]);opposed=np.array([r['opposition_angle_deg'] is not None and r['opposition_angle_deg']<=10 for r in ctx])
        masks=dict(all=np.ones(len(t),bool),prelift23=(t>=23)&(t<24),late44=(t>=44),airborne_hold=np.array([r['airborne_hold'] for r in ctx]),both_fit_opposed=valid&opposed,both_fit_nonopposed=valid&~opposed,fit_missing=~valid)
        reference=None
        for mode in p['evaluation_modes']:
            a=dict(np.load(case/f'{mode}.npz'));checks[f'{ep}/{mode}/clocks']=np.array_equal(a['frame'],p['frames'])
            if reference is None:reference=a
            else:checks[f'{ep}/{mode}/matched_targets']=np.array_equal(a['target'],reference['target'])
            values=errors(a['prediction'],a['target'])
            for name,mask in masks.items():
                rows.append(dict(episode=ep,mode=mode,group=name,physical_passed=source['passed'],samples=int(mask.sum()),
                                 metrics={k:float(values[k][mask].mean()) if mask.any() else None for k in METRICS},
                                 predicted_mass_kg=float(np.exp(a['prediction'][mask,12]).mean()) if mask.any() else None,
                                 true_mass_kg=float(np.exp(a['target'][mask,12]).mean()) if mask.any() else None))
    if not all(checks.values()):raise ValueError('Unmatched output clocks/targets')
    summary=[]
    for mode in p['evaluation_modes']:
        for group in masks:
            for physical in ['all','passed','failed']:
                eligible=[r for r in rows if r['mode']==mode and r['group']==group and (physical=='all' or r['physical_passed']==(physical=='passed'))]
                covered=[r for r in eligible if r['samples']]
                summary.append(dict(mode=mode,group=group,physical=physical,episodes_covered=len(covered),total_episodes=len(eligible),samples=sum(r['samples'] for r in eligible),
                    equal_episode={k:float(np.mean([r['metrics'][k] for r in covered])) if covered else None for k in METRICS}))
    report=dict(complete=len(episodes)==32,completed_episodes=episodes,checks=checks,all_matched=all(checks.values()),summary=summary,rows=rows,
                interpretation='Observed-controller fit/opposition categories are descriptive, not randomized interventions or true surface labels. Airborne errors cover only cases with labels; all32 remain in denominator.',new_optimizer_updates=0,new_physics=0)
    name='PARTIAL_COMPARISON' if args.partial else 'COMPARISON'
    (root/f'{name}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    headline=[r for r in summary if r['group'] in ['all','airborne_hold','prelift23'] and r['physical']=='all']
    print(json.dumps(dict(complete=report['complete'],cases=len(episodes),headline=headline),indent=2),flush=True)
    if args.partial:return
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    names=['geometry','contact','force','force zero'];colors=['#537895','#4b9b93','#c18740','#a6a6a6']
    for row,group in enumerate(['all','airborne_hold']):
        for col,key in enumerate(METRICS):
            items=[next(r for r in summary if r['mode']==mode and r['group']==group and r['physical']=='all') for mode in p['evaluation_modes']]
            ax=axes[row,col];values=[r['equal_episode'][key] for r in items];ax.bar(names,values,color=colors);ax.set_title(key);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
            ax.set_ylim(0,max(values)*1.2)
        axes[row,0].set_ylabel(f'{group}\ncovered {items[0]["episodes_covered"]}/{items[0]["total_episodes"]} cases')
    fig.suptitle('Frozen full Utonia: all32 saved TRAIN fixtures, matched95 clocks/case, batch1\nOriginal failure cases retained; airborne subset is not full acquisition coverage')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
    # Same four predeclared render cases, every saved prediction; no new forwards.
    video_episodes=p['render']['video_episodes']
    fig,axes=plt.subplots(len(video_episodes),2,figsize=(14,3*len(video_episodes)),squeeze=False)
    for row,ep in enumerate(video_episodes):
        case=root/'evaluation'/f'episode_{ep}';ctx=json.loads((case/'CONTEXT.json').read_text())
        t=np.array([r['timestamp_s'] for r in ctx]);hold=np.array([r['airborne_hold'] for r in ctx])
        for mode,label,color in zip(p['evaluation_modes'],names,colors):
            a=dict(np.load(case/f'{mode}.npz'))
            axes[row,0].plot(t,np.exp(a['prediction'][:,12]),label=label,color=color)
            axes[row,1].plot(t,errors(a['prediction'],a['target'])['center_cm'],label=label,color=color)
        axes[row,0].plot(t,np.exp(a['target'][:,12]),'k--',label='true mass')
        axes[row,0].set_ylabel(f'{ep}: mass (kg)');axes[row,1].set_ylabel('center error (cm)')
        for ax in axes[row]:
            ax.fill_between(t,0,1,where=hold,transform=ax.get_xaxis_transform(),color='#4b9b93',alpha=.10)
            ax.set_xlabel('recorded time (s)');ax.grid(alpha=.2)
    for ax in axes[0]:ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Fixed four rendered cases: raw frozen predictions, all95 saved clocks\nShading = actual airborne hold; no smoothing or truth correction')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/'prediction_timelines.png',dpi=140);plt.close(fig)
    lines=['# 原32条夹持轨迹：完整Utonia冻结预测','', '已有三组完整官方Utonia最终2000步模型，另对含力模型做力置零。原32条轨迹、每条95个固定时刻，batch1，32帧因果历史；模型、编码和归一化均沿用已有训练配置。没有新采集或训练，没有删除物理失败案例。', '', '这些轨迹属于原surface采集TRAIN配置，对已有旧模型是迁移诊断；不是新surface实验的held-out TEST验收。预测形状仍是对已知完整网格施加预测位置、旋转、三维尺寸，未实现任意形状重建。', '', '| 分组 | 模型 | 覆盖案例 | 位置cm | 旋转° | 尺寸% | 质量% |','|---|---|---:|---:|---:|---:|---:|']
    for r in headline:
        m=r['equal_episode'];lines.append(f"| {r['group']} | {r['mode']} | {r['episodes_covered']}/{r['total_episodes']} | {m['center_cm']:.3f} | {m['rotation_deg']:.3f} | {m['size_pct']:.3f} | {m['mass_pct']:.3f} |")
    lines+=['','为区分物理采集失败与预测失败，下面单独列出**原采集验收通过案例的离地保持阶段**。这只是上述完整结果的描述性分组，不替换全32例结果。','','| 模型 | 覆盖通过案例 | 位置cm | 旋转° | 尺寸% | 质量% |','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['group']=='airborne_hold' and r['physical']=='passed':
            m=r['equal_episode'];formatted=['缺失' if m[k] is None else f'{m[k]:.3f}' for k in METRICS]
            lines.append(f"| {r['mode']} | {r['episodes_covered']}/{r['total_episodes']} | "+' | '.join(formatted)+' |')
    lines+=['','![四条固定渲染轨迹的完整预测时间序列](prediction_timelines.png)','', '时间序列与四条视频使用相同预先固定案例和95个采样时刻，全部原始输出均保留；淡色背景表示实际离地保持。无力模型、含力模型与力置零均显示，便于区分持续偏差和随接触变化的预测。']
    lines+=['','所有数值为有样本案例的等案例平均。离地hold要求当前完整网格离地>1cm、两侧掌侧接触>0.01N、实际控制器处于hold；这是验证分组，未用它挑选模型输入或时刻。无标签案例在完整分项中以0样本保留。','', '![比较](comparison.png)', '', '接触方向分组使用同一时刻记录的控制器fit_valid和带方向法线（读取上一控制帧的触觉）；它是观测界面关系，不是物体真实法线或随机因果干预。10°沿用既有可视化平面兼容诊断，不作为新成功门槛。', '', '[完整分项](COMPARISON.json) · [运行前协议](PROTOCOL.json) · [完整模型加载记录](MODELS.json) · [输入缓存逐项验证](CACHE.json)', '', '原新surface训练仍受原4/8训练几何覆盖条件约束，已有采集只有2/8；本轮冻结结果不能替代该条件。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
