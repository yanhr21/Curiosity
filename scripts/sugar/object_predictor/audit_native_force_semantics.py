"""Saved native sensor vs solver force readback; never a model calibration."""
from pathlib import Path
import hashlib,json
import numpy as np
from sugar_newton.hand.patches import PATCH_SPECS


def main():
    base=Path('experiments/object_predictor_v1');source=Path('experiments/isaaclab_g1_anatomical27_object_demos/carrybox_plain_longx1p6_native_v1/whole_hand_trace.npz')
    with np.load(source) as z:
        names=z['robot_box_force_body_names'].tolist()
        ids=[names.index(f'{side}_anatomical_{p.name}_elastomer') for side in ('left','right') for p in PATCH_SPECS]
        solved_normal=z['robot_box_force_w'][:,ids].astype(float)
        solved_shear=z['robot_box_friction_force_w'][:,ids].astype(float)
        checks={'explicit_54_pad_body_mapping':len(set(ids))==54,
                'normal_force_matches_separate_pad_sensor':np.allclose(solved_normal,z['patch_box_force_w'].reshape(80,54,3),rtol=0,atol=1e-6),
                'friction_force_matches_separate_pad_sensor':np.allclose(solved_shear,z['patch_box_friction_force_w'].reshape(80,54,3),rtol=0,atol=1e-6)}
    with np.load(base/'isaac_saved_r1/episode_1000.npz') as z:raw=z['normal_load_n'].sum(1)
    normal=np.linalg.norm(solved_normal,axis=-1).sum(1)
    vertical=-(solved_normal+solved_shear).sum(1)[:,2]
    def summary(a):return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))
    result=dict(source=str(source),source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
                checks=checks,passed=all(checks.values()),range='Same49predictionframes31..79',
                tactile_sdf_normal_sum_n=summary(raw[31:]),solver_patch_normal_vector_norm_sum_n=summary(normal[31:]),
                ratio_tactile_to_solver_normal=summary((raw/np.maximum(normal,1e-12))[31:]),
                solver_upward_net_contact_force_n=summary(vertical[31:]),
                definitions={'tactile':'Official TacSL SDF-derived local forces, converted using saved contact normals',
                             'solver':'PhysX normal/friction contact force readback on the same54 elastomer bodies; normal value is sum of per-body vector norms, not sum of all individual contact magnitudes',
                             'newton_training':'SolverMuJoCo resolved normal/friction loads redistributed over contact surface; field.py source definition'},
                scope='Readback of an existing trace; differing geometry/contact discretization and force models confounded. No fitted scale, no sensor replacement, no prediction improvement claim.',
                physics_steps=0,model_forwards=0,optimizer_updates=0)
    out=base/'grip_transfer_v1';(out/'NATIVE_FORCE_SEMANTICS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['passed']


if __name__=='__main__':main()
