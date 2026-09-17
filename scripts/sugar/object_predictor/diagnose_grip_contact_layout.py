"""Saved contact geometry and force observability; no controller/model forwards."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def distribution(values):
    a=np.asarray(values,float)
    a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)),min=float(a.min()),max=float(a.max())) if len(a) else dict(n=0,median=None,min=None,max=None)


def inspect(source, label):
    result=json.loads((source/'RESULT.json').read_text());ep=result['episode']
    keys=['timestamp_s','normal_load_n','contact_area_m2','shear_force_w','validation_resolved_normal_n',
          'validation_hand_normal_vec_w','validation_hand_friction_vec_w','validation_full_mesh_min_z_m',
          'validation_controller_fit_valid','validation_controller_alignment_error_deg']
    with np.load(source/f'episode_{ep}.npz') as z:a={k:z[k] for k in keys}
    with np.load(source/'contact_surface.npz') as z:
        field={k:z[k] for k in ['offset','pad','hand','area_m2','normal_pressure_pa','position_hand_frame_m']}
    t=a['timestamp_s'];load=a['normal_load_n'].reshape(-1,2,27)
    mg=result['mass_kg']*9.81
    start=result['lift_start_s'];pre_end=start if start>=0 else 40.
    windows={'prelift_last1s':(t>=pre_end-1)&(t<pre_end),'final4s':t>=44.}
    if start>=0:windows['first_lift1s']=(t>=start)&(t<start+1)
    rows=[]
    for name,mask in windows.items():
        ids=np.flatnonzero(mask);assert len(ids)
        total=load[ids].sum(2);full=a['validation_resolved_normal_n'][ids]
        normal_up=-a['validation_hand_normal_vec_w'][ids].sum(1)[:,2]
        friction_up=-a['validation_hand_friction_vec_w'][ids].sum(1)[:,2]
        shear_up=-a['shear_force_w'][ids].sum(1)[:,2]
        row=dict(episode=ep,mode=label,window=name,frames=len(ids),controller_passed=result['passed'],
                 mass_kg=result['mass_kg'],target_load_n=result['grip_target_per_hand_n'],
                 actual_clearance_cm=distribution(100*a['validation_full_mesh_min_z_m'][ids]),
                 full_normal_up_over_weight_mean=float(normal_up.mean()/mg),
                 full_friction_up_over_weight_mean=float(friction_up.mean()/mg),
                 full_total_up_over_weight_mean=float((normal_up+friction_up).mean()/mg),
                 observed_shear_up_over_weight_mean=float(shear_up.mean()/mg),
                 hands=[])
        for hand in [0,1]:
            local=load[ids,hand];den=total[:,hand]
            contact=den>.01
            ratio=np.divide(den,full[:,hand],out=np.full(len(ids),np.nan),where=full[:,hand]>1e-8)
            rms=[]
            for frame in ids:
                lo,hi=field['offset'][frame:frame+2]
                sel=(field['hand'][lo:hi]==hand)&(field['pad'][lo:hi]>=0)&(field['normal_pressure_pa'][lo:hi]>0)
                points=field['position_hand_frame_m'][lo:hi][sel].astype(float)
                area=field['area_m2'][lo:hi][sel].astype(float)
                if len(points)>=3 and area.sum()>0:
                    c=np.average(points,axis=0,weights=area);d=points-c
                    cov=(d.T*area)@d/area.sum()
                    rms.append(np.sqrt(np.maximum(np.linalg.eigvalsh(cov),0))*1000)
            rms=np.asarray(rms).reshape(-1,3)
            integrated=local.sum(0)
            row['hands'].append(dict(hand=hand,contact_frames=int(contact.sum()),
                load_n=distribution(den),full_normal_n=distribution(full[:,hand]),
                observed_full_scalar_load_ratio=distribution(ratio),
                active_pads=distribution((local>.01).sum(1)),
                palm_fraction=float(integrated[:12].sum()/integrated.sum()) if integrated.sum()>0 else None,
                pad_load_share=(integrated/integrated.sum()).tolist() if integrated.sum()>0 else None,
                area_mm2=distribution(a['contact_area_m2'][ids].reshape(-1,2,27)[:,hand].sum(1)*1e6),
                rms_small_tangent_mm=distribution(rms[:,1]),rms_large_tangent_mm=distribution(rms[:,2]),
                valid_plane_fraction=float(a['validation_controller_fit_valid'][ids,hand].mean())))
        rows.append(row)
    return rows


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    data=Path(args.data);out=Path(args.output);out.mkdir(exist_ok=False)
    configs=json.loads((data/'COLLECTION_RESULT.json').read_text())['records']
    assert len(configs)==32 and all(c['split']=='train' for c in configs)
    rows=[]
    for c in configs:
        rows.extend(inspect(Path(c['source']),'baseline'))
        print('INSPECTED',c['episode'],flush=True)
    for ep in [5014,5000]:rows.extend(inspect(Path(args.diagnostic)/f'cases/episode_{ep}','sustained'))
    report=dict(rows=rows,cases=34,new_physics=0,new_model_forwards=0,new_optimizer_updates=0,
        scope='All32 fixed TRAIN and both completed sustained-feedback cases; no held-out data or new controller trial.',
        geometry='Area-weighted covariance of assigned contact positions each frame in known hand frame; tangent RMS does not prove frictional wrench closure.',
        force='Full vector force on object is negative saved force on hands, validation only. Observed shear is pressure-distributed per-hand friction projected onto faces and restricted to assigned pads; not an independent per-face force measurement.',
        windows='Prelift last1s before recorded trigger; no-trigger uses39..40s. Final4s fixed44..48s for everyone. Firstlift1s only where triggered. No selection by outcome.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(16,10),sharex=True)
    pre=[r for r in rows if r['window']=='prelift_last1s'];x=np.arange(len(pre))
    for hand,offset,color in [(0,-.18,'#138f99'),(1,.18,'#b56739')]:
        axes[0].bar(x+offset,[r['hands'][hand]['active_pads']['median'] for r in pre],width=.36,color=color,label=['L','R'][hand])
        axes[1].bar(x+offset,[r['hands'][hand]['rms_large_tangent_mm']['median'] or 0 for r in pre],width=.36,color=color)
        axes[2].bar(x+offset,[r['hands'][hand]['observed_full_scalar_load_ratio']['median'] or 0 for r in pre],width=.36,color=color)
    axes[0].set_ylabel('Active assigned pads');axes[0].legend()
    axes[1].set_ylabel('Large tangent RMS (mm)');axes[2].set_ylabel('Assigned / full normal load')
    axes[2].set_xticks(x,[str(r['episode'])+(' S' if r['mode']=='sustained' else '') for r in pre],rotation=65)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.axvline(31.5,color='black',ls=':')
    fig.suptitle('Saved pre-lift contact layout; zero bars may be missing contact (see JSON)')
    fig.tight_layout();fig.savefig(out/'contact_layout.png',dpi=140);plt.close(fig)
    print('COMPLETE_CONTACT_LAYOUT',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);main(p.parse_args())
