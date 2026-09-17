"""Saved-sensor force balance diagnostic, never a learned object predictor.

All candidate resultants consume only the existing observation adapter. Object
mass/clearance and solver forces are read afterwards for validation. No fitted
calibration, true contact normals, new physics, or model forward is used.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import OBSERVATION_KEYS
from .surface_data import SURFACE_KEYS, select_surface_frames, _surface_points

MODES = ('shear_only', 'cad_normal_plus_shear', 'fitted_normal_plus_shear')


def observed_resultants(obs, surfaces, cad_normals):
    """Return forces ON OBJECT; recorded shear is force ON HAND."""
    results = {key: [] for key in MODES}
    details = []
    for t, surface in enumerate(surfaces):
        sensed = _surface_points(obs, surface, t, cad_normals[t])
        shear = np.asarray(obs['shear_force_w'][t], dtype=np.float64).sum(0)
        cad_normal = (np.asarray(obs['normal_load_n'][t], dtype=np.float64)[:, None]
                      * np.asarray(cad_normals[t], dtype=np.float64)).sum(0)
        fitted_normal = (sensed['load'][:, None] * sensed['normal']).sum(0)
        # Keep identical pad shear in both direction comparisons, so only the
        # estimated normal contribution differs. Dense shear is audited below.
        results['shear_only'].append(-shear)
        results['cad_normal_plus_shear'].append(cad_normal - shear)
        results['fitted_normal_plus_shear'].append(fitted_normal - shear)
        dense_load = np.bincount(sensed['pad'], weights=sensed['load'], minlength=54)
        sparse = np.asarray(obs['normal_load_n'][t], dtype=np.float64)
        active = sparse > 1e-3
        details.append(dict(
            selected_faces=len(sensed['load']), fitted_faces=int(sensed['resolved'].sum()),
            fitted_load_fraction=float(sensed['load'][sensed['resolved']].sum() / sensed['load'].sum())
                if sensed['load'].sum() > 0 else None,
            dense_sparse_active_load_max_error_n=float(np.max(abs(dense_load[active]-sparse[active]))) if active.any() else 0.,
            dense_sparse_shear_component_max_error_n=float(np.max(abs(sensed['shear'].sum(0)-shear))),
            excluded_subthreshold_load_n=float(sparse[~active].sum())))
    return {key: np.asarray(value) for key, value in results.items()}, details


def main():
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.output);protocol=json.loads((root/'PROTOCOL.json').read_text())
    frozen=Path(protocol['frozen']);source=Path(protocol['source']);frames=np.asarray(protocol['frames'])
    rows=[];traces={};integration=[]
    for ep in protocol['episodes']:
        with np.load(frozen/'cache'/f'episode_{ep}.npz',allow_pickle=False) as saved:
            # Observation and validation fields remain separate at the call boundary.
            obs={key:saved[key][frames] for key in OBSERVATION_KEYS}
            cad=saved['hand_contact_surface_normals_w'][frames]
            validation={key:saved[key][frames] for key in (
                'object_mass_kg','validation_full_mesh_min_z_m','validation_controller_phase',
                'validation_hand_normal_vec_w','validation_hand_friction_vec_w')}
        case=source/'cases'/f'episode_{ep}'
        meta=json.loads((case/'RESULT.json').read_text())
        with np.load(case/'contact_surface.npz',allow_pickle=False) as surface:
            dense={key:surface[key] for key in ('offset',*SURFACE_KEYS)}
        sensed=select_surface_frames(dense,frames)
        force,details=observed_resultants(obs,sensed,cad)
        if not all(np.isfinite(v).all() for v in force.values()):
            raise FloatingPointError(ep)
        # Validation-only reference is not an observation candidate.
        solver=-(validation['validation_hand_normal_vec_w'].astype(float)
                 +validation['validation_hand_friction_vec_w'].astype(float)).sum(1)
        t=obs['timestamp_s'];mass=validation['object_mass_kg'].astype(float)
        loads=obs['normal_load_n'].reshape(-1,2,27).sum(2)
        hold=(validation['validation_full_mesh_min_z_m']>.01)&(loads>.01).all(1)&(validation['validation_controller_phase']==2)
        masks=dict(all=np.ones(len(t),bool),airborne_hold=hold,late44_airborne_hold=(t>=44)&hold,
                   grounded=validation['validation_full_mesh_min_z_m']<=.01)
        equivalents={k:v[:,2]/protocol['gravity_m_s2'] for k,v in force.items()}
        equivalents['validation_full_solver']=solver[:,2]/protocol['gravity_m_s2']
        for mode,eq in equivalents.items():
            for group,mask in masks.items():
                rows.append(dict(episode=ep,physical_passed=meta['passed'],mode=mode,group=group,
                    samples=int(mask.sum()),relative_error_pct=float(np.mean(abs(eq[mask]-mass[mask])/mass[mask])*100) if mask.any() else None,
                    signed_support_mass_kg=float(eq[mask].mean()) if mask.any() else None,
                    true_mass_kg=float(mass[mask].mean()) if mask.any() else None,
                    negative_support_samples=int((eq[mask]<0).sum())))
        integration.append(dict(episode=ep,
            max_active_load_error_n=max(v['dense_sparse_active_load_max_error_n'] for v in details),
            max_shear_component_difference_n=max(v['dense_sparse_shear_component_max_error_n'] for v in details),
            max_excluded_subthreshold_load_n=max(v['excluded_subthreshold_load_n'] for v in details)))
        trace=dict(episode=ep,frame=frames,timestamp_s=t,true_mass_kg=mass,airborne_hold=hold,
                   full_solver_on_object_w=solver,**{k+'_on_object_w':v for k,v in force.items()})
        np.savez_compressed(root/f'episode_{ep}.npz',**trace)
        if ep in protocol['visual_episodes']:traces[ep]=(t,mass,hold,equivalents)
        print('FORCE_BALANCE_CASE_COMPLETE',ep,flush=True)
    summary=[]
    for mode in (*MODES,'validation_full_solver'):
        for group in masks:
            for physical in ('all','passed','failed'):
                part=[v for v in rows if v['mode']==mode and v['group']==group and (physical=='all' or v['physical_passed']==(physical=='passed'))]
                covered=[v for v in part if v['samples']]
                summary.append(dict(mode=mode,group=group,physical=physical,total_episodes=len(part),episodes_covered=len(covered),
                    samples=sum(v['samples'] for v in part),negative_support_samples=sum(v['negative_support_samples'] for v in part),
                    equal_episode_relative_error_pct=float(np.mean([v['relative_error_pct'] for v in covered])) if covered else None))
    report=dict(complete=len(integration)==32,rows=rows,summary=summary,integration=integration,
                new_physics=0,model_forwards=0,new_optimizer_updates=0,
                interpretation='Fz/g is signed support-equivalent mass. It is an object-mass estimate only under complete force observation, no ground support, and negligible vertical acceleration. Observed estimates use partial pads and approximate directions/traction; no calibration or clipping.')
    (root/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    fig,axes=plt.subplots(4,1,figsize=(13,12))
    for ax,ep in zip(axes,protocol['visual_episodes']):
        t,mass,hold,eq=traces[ep]
        for mode in MODES:ax.plot(t,eq[mode],label=mode)
        ax.plot(t,eq['validation_full_solver'],':',color='gray',label='full solver (validation only)')
        ax.plot(t,mass,'k--',label='true mass')
        ax.fill_between(t,0,1,where=hold,transform=ax.get_xaxis_transform(),alpha=.12,color='green')
        ax.set_ylabel(f'{ep}: signed Fz/g (kg)');ax.grid(alpha=.25)
    axes[0].legend(ncol=2,fontsize=8);axes[-1].set_xlabel('recorded time (s)')
    fig.suptitle('Saved-observation force balance diagnostic; no learned model, fit, clipping or calibration\nAll95 original clocks; shading = actual airborne hold, grounded Fz/g is not object mass')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'force_balance_timelines.png',dpi=140);plt.close(fig)
    lines=['# 已有触觉观测的竖向力平衡诊断','','这是明确标注的力学诊断，不是新的神经网络或 Utonia 训练结果。32条原始轨迹、各95个原固定时刻，全部案例保留。','',
        '观测候选只用掌侧触觉、当前手姿态和已有CAD/接触面法向适配器。真实质量、离地高度、控制阶段、求解器力仅在计算完成后用于验证分组。完整求解器曲线不进入观测候选。','',
        '物体侧合力约定为 `Σ(法向载荷 × 手部向外法向) − Σ(手上剪切力)`；CAD法向和拟合接触面法向都只是方向近似。拟合沿用surface_data已有六点/平面性门槛与CAD回退，不读取真实界面法向。三种候选使用相同原掌侧剪切，单独记录稠密积分与原掌侧读数的差值。','',
        '**Fz/g称为有符号支持等效质量**：仅在承重力完整、无地面支持且竖向加速度可忽略时才等于物体质量。未校准、未裁剪负值、未过滤预测误差。','',
        '| 分组 | 物理验收 | 方法 | 有样本/全部案例 | 等案例相对误差% | 负支持样本 |','|---|---|---|---:|---:|---:|']
    for v in summary:
        if v['group'] in ('airborne_hold','late44_airborne_hold') and v['physical'] in ('all','passed'):
            err=v['equal_episode_relative_error_pct'];lines.append(f"| {v['group']} | {v['physical']} | {v['mode']} | {v['episodes_covered']}/{v['total_episodes']} | {err:.3f} | {v['negative_support_samples']} |")
    lines+=['','![原始力平衡时间序列](force_balance_timelines.png)','','接触前/地面支撑阶段的曲线用于显示物理含义，不作为可估重成功证据。所有缺失离地样本的案例保留在RESULT.json中。该诊断不修改原采集通过条件、训练覆盖门槛或已完成冻结模型结果。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE_FORCE_OBSERVABILITY',json.dumps([v for v in summary if v['group']=='late44_airborne_hold' and v['physical']=='passed']),flush=True)


if __name__=='__main__':main()
