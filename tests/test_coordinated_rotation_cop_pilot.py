import ast
import copy
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.sugar.object_predictor import run_coordinated_rotation_cop_pilot as pilot
from scripts.sugar.object_predictor.coordinated_rotation_cop import CoordinatedRotationCoPController


class CoordinatedPilotTests(unittest.TestCase):
    def test_fixed_original_three_gates_and_direct_shared_baseline(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'shared_budget_cop_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(p['physics_substeps'],8)
        self.assertEqual(len(p['original_physical_checks']),12);self.assertEqual(p['late_mass_check']['frame'],2381)
        self.assertEqual(p['shared_budget']['total_actual_m'],.2)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.]);self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertFalse(p['automatic_expansion']);self.assertFalse(p['automatic_training'])

    def test_scene_collector_original_physics_ast_unchanged(self):
        p=Path('scripts/sugar/object_predictor')
        for old,new in [('shared_budget_cop_scene.py','coordinated_rotation_cop_scene.py'),('collect_shared_budget_cop.py','collect_coordinated_rotation_cop.py')]:
            before=(p/old).read_text().replace('SharedBudgetCoP','CoordinatedRotationCoP').replace('shared_budget_cop_scene','coordinated_rotation_cop_scene').replace('from .shared_budget_cop import','from .coordinated_rotation_cop import').replace('collect_shared_budget_cop','collect_coordinated_rotation_cop')
            self.assertEqual(ast.dump(ast.parse(before)),ast.dump(ast.parse((p/new).read_text())))

    def fixture(self):
        poses=np.array([[-.2,0.,.2,0.,0.,0.,1.],[.2,0.,.2,0.,0.,0.,1.]])
        centers=np.array([[0.,0.,.04],[0.,0.,-.04]])
        offsets=np.array([[0,y,z] for y in (-.01,0.,.01) for z in (-.01,.01)])
        tilt=Rotation.from_euler('y',-2.,degrees=True)
        field=dict(pos=np.concatenate([tilt.apply(offsets)+c for c in centers]),area=np.full(12,1e-5),
            pressure=np.full(12,100.),pad=np.repeat([0,27],6),patch=np.repeat([0,1],6))
        c=CoordinatedRotationCoPController(target_load_n=12.,force_gain=.000025,response_gain=True)
        c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)));c.observe(poses,field)
        target,_,record=c.command(36.,np.array([9.,10.]),.02)
        a={'timestamp_s':np.array([35.98,36.]),'hand_pose_w':np.stack([poses,target])}
        a.update({'validation_controller_'+('distance_m' if k=='distance' else k):np.stack([np.zeros_like(v),v]) for k,v in record.items()})
        a['validation_controller_lift_start_s'][0]=-1.
        return a,c.floor_vertices,poses,record['floor_observed_min_z_m'],record['floor_requested_swept_min_z_m'],field,c.support_normals

    def test_actual_selected_rotation_replayed_with_parent_transaction(self):
        a,m,p,h,q,f,n=self.fixture()
        minimum=pilot.allocation_row_readback(a,1,m,p,h,q,f,n)
        np.testing.assert_allclose(minimum,a['validation_controller_floor_executed_swept_min_z_m'][1],rtol=0,atol=1e-12)
        self.assertEqual(a['validation_controller_rotation_coord_selected_suppressed'][1].sum(),1)

    def test_forged_selection_prediction_parent_and_actual_pose_rejected(self):
        a,m,p,h,q,f,n=self.fixture()
        for key in ('rotation_coord_selected_suppressed','rotation_coord_candidate_eligible','actual_alignment_active'):
            bad=copy.deepcopy(a);bad['validation_controller_'+key][1]=~bad['validation_controller_'+key][1]
            with self.assertRaises(ValueError):pilot.allocation_row_readback(bad,1,m,p,h,q,f,n)
        for key in ('rotation_coord_original_predicted_angle_deg','rotation_coord_parent_target_pose_w','floor_executed_target_pose_w','cop_travel_m'):
            bad=copy.deepcopy(a);bad['validation_controller_'+key].reshape(2,-1)[1,0]+=.001
            with self.assertRaises(ValueError):pilot.allocation_row_readback(bad,1,m,p,h,q,f,n)


if __name__=='__main__':unittest.main()
