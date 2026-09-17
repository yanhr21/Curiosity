"""Non-neural observation diagnostics; never a replacement pretrained model.

Contact centroid plus a TRAIN-only offset checks whether these observations carry
usable center information. Summed normal load / gravity is a quasi-static support
diagnostic, invalid during squeezing, impact, release or partial support.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .data import ContactDataset


def rows(dataset):
    features=[];targets=[];mass=[];contacts=[];episodes=[];frames=[]
    for i,(e,t) in enumerate(dataset.index):
        meta,a=dataset.episodes[e]
        hand=a['hand_pose_w'][t,0]
        Rh=Rotation.from_quat(hand[3:]).as_matrix()
        hit=a['normal_load_n'][t]>1e-3
        points=a['contact_position_w'][t,hit] if hit.any() else a['hand_sites_w'][t]
        features.append((points.mean(0)-hand[:3])@Rh)
        targets.append(dataset[i]['target'])
        mass.append(a['normal_load_n'][t-dataset.history+1:t+1].sum(1).mean()/9.81)
        contacts.append(dataset[i]['contact']);episodes.append(meta['episode']);frames.append(t)
    return np.asarray(features),np.asarray(targets),np.asarray(mass),np.asarray(contacts),np.asarray(episodes),np.asarray(frames)


def main(args):
    train=ContactDataset(args.train_data,'geometry_contact','train')
    test=ContactDataset(args.test_data,'geometry_contact','test')
    train_x,train_y,_,train_contact,*_=rows(train)
    x,y,mass,contact,episodes,frames=rows(test)
    offset=(train_y[:,:3]-train_x).mean(0)
    predicted=x+offset
    conditional=predicted.copy()
    conditional_offsets={}
    for flag in (False,True):
        fitted=(train_y[train_contact==flag,:3]-train_x[train_contact==flag]).mean(0)
        conditional[contact==flag]=x[contact==flag]+fitted
        conditional_offsets[str(flag)]=fitted.tolist()
    errors=dict(contact_centroid_train_offset_position_cm=np.linalg.norm(predicted-y[:,:3],axis=1)*100,
                contact_centroid_contact_conditioned_train_offset_position_cm=np.linalg.norm(conditional-y[:,:3],axis=1)*100,
                quasi_static_normal_sum_mass_relative=np.abs(mass-np.exp(y[:,12]))/np.exp(y[:,12]))
    report=dict(scope=__doc__,train_data=args.train_data,test_data=args.test_data,
                train_only_offset_m=offset.tolist(),train_only_contact_conditioned_offsets_m=conditional_offsets,metrics={})
    for group,mask in [('all',np.ones(len(y),bool)),('contact',contact),('no_contact',~contact)]:
        report['metrics'][group]={'samples':int(mask.sum())}
        if mask.any():
            for key,values in errors.items():
                report['metrics'][group][key]=dict(mean=float(values[mask].mean()),
                    equal_episode_mean=float(np.mean([values[mask & (episodes==e)].mean() for e in np.unique(episodes[mask])])),
                    p90=float(np.quantile(values[mask],.9)))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'predictions.npz',center=predicted,contact_conditioned_center=conditional,mass=mass,target=y,
                        episode=episodes,frame=frames,contact=contact,**errors)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('train-data','test-data','output'):
        ap.add_argument('--'+name,required=True)
    main(ap.parse_args())
