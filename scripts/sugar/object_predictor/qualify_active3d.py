"""Fixed real-chart/empty-chart comparison with both complete official models.

No learning, pose fitting, shape selection, or ground-truth canonical alignment.
Ground truth is opened only after all model outputs have been written.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import trimesh

from .official_active3d import OfficialActive3D
from .shape_metric import rotation


def sampled_mesh(vertices, faces, seed):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError('Invalid predicted surface area')
    return trimesh.sample.sample_surface(mesh, 16384, seed=seed)[0]


def surface_error(predicted, truth):
    pt = cKDTree(truth).query(predicted, workers=1)[0]
    tp = cKDTree(predicted).query(truth, workers=1)[0]
    return dict(sampled_symmetric_surface_cm=float((pt.mean()+tp.mean())*50),
                prediction_to_truth_cm=float(pt.mean()*100), truth_to_prediction_cm=float(tp.mean()*100),
                fscore_1cm=float(2*np.mean(pt<.01)*np.mean(tp<.01)/max(np.mean(pt<.01)+np.mean(tp<.01),1e-12)))


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Retained explicit compute step required')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(args.inputs)
    report = json.loads((source/'RESULT.json').read_text())
    expected = [(ep, frame) for ep in range(5000,5004) for frame in (1206,2206)]
    assert report['complete'] and [(r['episode'],r['frame']) for r in report['cases']] == expected
    protocol = dict(inputs=str(source), cases=expected, arms=['t_p','t_g'], conditions=['empty','observed'],
                    model_forwards=64, model_parameter_updates=0, physics_steps=0,
                    frame='Frozen Utonia predicted canonical frame; same frame for empty/observed. No GT alignment.',
                    empty_condition='Only Active3D charts are zeroed. Upstream Utonia pose/scale still comes from tactile input; not an end-to-end no-touch baseline.',
                    chart_adaptation=report['limitations'],
                    metric='Mean bidirectional nearest-neighbor distance between 16384 area-sampled mesh points per surface, cm; no rigid alignment',
                    acceptance='Diagnostic comparison only; report every case, no threshold/model selection or generalization claim')
    (root/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vendor=Path('experiments/object_predictor_v1/vendor/Active-3D-Vision-and-Touch')
    pretrained=Path('experiments/object_predictor_v1/checkpoints/active3d_official')
    loaded = {}; rows=[]
    # Labels are not read anywhere in this inference loop.
    for arm in protocol['arms']:
        model=OfficialActive3D(vendor,pretrained/arm)
        loaded[arm]=model.provenance
        if model.provenance['parameters'] != {'t_p':1033599,'t_g':2612199}[arm]:
            raise AssertionError('Expected complete official checkpoint')
        for case_index,case in enumerate(report['cases']):
            with np.load(source/case['input_file'],allow_pickle=False) as z:
                charts=z[arm+'_charts'].reshape(1,-1,4)
                center=z['center_w']; orientation=z['orientation_w']; scale=float(z['canonical_scale_m'])
            saved={}; timings={}
            for condition in protocol['conditions']:
                data=np.zeros_like(charts) if condition=='empty' else charts
                torch.cuda.synchronize(); start=time.perf_counter()
                prediction,mask=model.predict(data)
                prediction=prediction.cpu().numpy()[0]
                timings[condition]=time.perf_counter()-start
                repeat,_=model.predict(data)
                if not np.array_equal(prediction,repeat.cpu().numpy()[0]):
                    raise AssertionError('Full official forward repeat differs')
                canonical=prediction[:model.vision_vertex_count]
                saved[condition+'_vertices_canonical']=canonical
                saved[condition+'_vertices_world_m']=canonical*scale@orientation.T+center
                saved[condition+'_mask']=mask.cpu().numpy()[0]
            saved.update(faces=model.vision_faces,center_w=center,orientation_w=orientation,canonical_scale_m=scale)
            name=f'{arm}_episode_{case["episode"]}_frame_{case["frame"]}.npz'
            np.savez_compressed(root/name,**saved)
            rows.append(dict(arm=arm,episode=case['episode'],frame=case['frame'],output_file=name,
                             case_index=case_index,forward_seconds=timings,repeat_max_difference=0.,
                             actual_charts=min(case['accepted_charts'],5 if arm=='t_p' else 20)))
            print('OFFICIAL_SHAPE_PREDICTION',arm,case['episode'],case['frame'],flush=True)
        del model
        torch.cuda.empty_cache()
    (root/'INFERENCE.json').write_text(json.dumps(dict(complete=True,models=loaded,cases=rows,protocol=protocol),indent=2)+'\n')

    # Separate post-inference evaluator; actual object mesh is never a model input.
    collection=json.loads(Path('experiments/object_predictor_v1/response_surface_dataset_v1/COLLECTION_RESULT.json').read_text())
    records={r['episode']:r for r in collection['records']}
    with np.load('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz',allow_pickle=False) as z:
        reference=z['vertices'].astype(float); reference_faces=z['faces']
    dimensions=np.ptp(reference,axis=0); reference-=(reference.max(0)+reference.min(0))/2
    for row in rows:
        case=report['cases'][row['case_index']]; frame=row['frame']; ep=row['episode']
        with np.load(Path(records[ep]['source'])/f'episode_{ep}.npz',allow_pickle=False) as z:
            pose=z['object_pose_w'][frame]; true_rotation=Rotation.from_quat(pose[3:]).as_matrix()
            true_center=pose[:3]+z['object_local_center_m'][frame]@true_rotation.T
            true_size=z['object_dimensions_m'][frame]
        truth_vertices=(reference*(true_size/dimensions))@true_rotation.T+true_center
        seed=20260916+row['case_index']
        truth_points=sampled_mesh(truth_vertices,reference_faces,seed)
        with np.load(source/case['input_file'],allow_pickle=False) as z:
            prior=z['prediction']; hands=z['hand_pose_w']
        rh=Rotation.from_quat(hands[0,3:]).as_matrix()
        prior_vertices=((reference*(np.exp(prior[9:12])/dimensions))@rotation(prior[3:9]).T+prior[:3])@rh.T+hands[0,:3]
        row['known_mesh_utonia_reference']=surface_error(sampled_mesh(prior_vertices,reference_faces,seed+100),truth_points)
        with np.load(root/row['output_file'],allow_pickle=False) as z:
            row['metrics']={}
            for condition in protocol['conditions']:
                vertices=z[condition+'_vertices_world_m']
                points=sampled_mesh(vertices,z['faces'],seed+200)
                metrics=surface_error(points,truth_points)
                metrics.update(minimum_world_z_m=float(vertices[:,2].min()),surface_area_m2=float(trimesh.Trimesh(vertices,z['faces'],process=False).area))
                row['metrics'][condition]=metrics
    aggregate={}
    for arm in protocol['arms']:
        part=[row for row in rows if row['arm']==arm]
        aggregate[arm]={condition:{key:float(np.mean([r['metrics'][condition][key] for r in part]))
                                  for key in ('sampled_symmetric_surface_cm','fscore_1cm','minimum_world_z_m')}
                        for condition in protocol['conditions']}
        aggregate[arm]['observed_improves_cases']=sum(r['metrics']['observed']['sampled_symmetric_surface_cm']<r['metrics']['empty']['sampled_symmetric_surface_cm'] for r in part)
    result=dict(complete=True,protocol=protocol,models=loaded,cases=rows,mean=aggregate,
                scope='Fixed eight TRAIN input-adaptation diagnostic with complete official shape decoders. '
                      'Frozen predicted pose/scale and tiny current-contact charts; unknown-shape generalization, mass/material and policy benefit unproven.')
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 官方 Active3D：真实接触输入与空触觉对照','',result['scope'],'',
           '原网络、完整发布权重、0次参数更新。所有预测先保存，之后才读取真值。显示几何为网络输出的1824顶点／2304面，未套用目标物体网格。',
           '两条件坐标与尺度均沿用含触觉的冻结Utonia预测。“空触觉”仅将Active3D的chart置零，不是端到端无触觉基线。',
           '输入为实际触觉质心的小范围插值，远小于初始模板；官方光学CNN输出的实际尺度分布尚未测量。','',
           '| 模型 | 空触觉表面距离 cm | 真实触觉表面距离 cm | 改善帧 / 8 |','|---|---:|---:|---:|']
    for arm,value in aggregate.items():
        lines.append(f'| {arm} | {value["empty"]["sampled_symmetric_surface_cm"]:.3f} | {value["observed"]["sampled_symmetric_surface_cm"]:.3f} | {value["observed_improves_cases"]} |')
    lines+=['','指标是双向、面积采样的最近点距离（每个完整表面16384点），非官方训练用Chamfer的同数值复现；未做真值刚体对齐。全部逐帧值保存在RESULT.json。',
            '实际渲染另由render_active3d生成；此报告完成不代表渲染已完成。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('OFFICIAL_SHAPE_COMPARISON',json.dumps(aggregate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
