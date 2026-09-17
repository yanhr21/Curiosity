import ast
import copy
from pathlib import Path
import unittest
import numpy as np
from scripts.sugar.object_predictor import run_shared_budget_cop_pilot as pilot
from scripts.sugar.object_predictor.shared_budget_cop import SharedBudgetCoPController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests as Fixture


class SharedPilotTests(unittest.TestCase):
    def test_fixed_three_direct_baseline_and_all_original_physical_thresholds(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'allocated_floor_cop_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(p['physics_substeps'],8)
        self.assertEqual(len(p['original_physical_checks']),12);self.assertEqual(p['late_mass_check']['frame'],2381)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.])
        self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertEqual(p['shared_budget']['total_actual_m'],.2)
        self.assertIsNone(p['cop_rule']['per_hand_accumulated_travel_max_m'])
        self.assertFalse(p['automatic_expansion']);self.assertFalse(p['automatic_training'])

    def test_original_physics_and_collector_ast_only_explicit_new_binding(self):
        folder=Path('scripts/sugar/object_predictor')
        old=(folder/'allocated_floor_cop_scene.py').read_text().replace('from .allocated_floor_cop import AllocatedFloorCoPController',
            'from .shared_budget_cop import SharedBudgetCoPController').replace('AllocatedFloorCoPController','SharedBudgetCoPController').replace('AllocatedFloorCoPScene','SharedBudgetCoPScene')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse((folder/'shared_budget_cop_scene.py').read_text())))
        old=(folder/'collect_allocated_floor_cop.py').read_text().replace('collect_allocated_floor_cop','collect_shared_budget_cop').replace('allocated_floor_cop_scene','shared_budget_cop_scene').replace('AllocatedFloorCoPScene','SharedBudgetCoPScene')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse((folder/'collect_shared_budget_cop.py').read_text())))

    def fixture(self,before=(.0944,.1)):
        # Numeric row transaction fixture; not an invented physical trajectory.
        poses,field=Fixture().observation(centers=np.array([[0.,0.,.08],[0.,0.,-.08]]));poses[:,2]=.00002
        c=SharedBudgetCoPController(target_load_n=12.,force_gain=.000025,response_gain=True)
        c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)));c.observe(poses,field)
        c.cop_travel[:]=before
        target,_,rec=c.command(36.58,np.full(2,8.),.02)
        a={'timestamp_s':np.array([36.56,36.58]),'hand_pose_w':np.stack((poses,target))}
        a.update({'validation_controller_'+('distance_m' if k=='distance' else k):np.stack((np.zeros_like(v),v)) for k,v in rec.items()})
        a['validation_controller_cop_travel_m'][0]=before
        a['validation_controller_lift_start_s'][0]=-1.
        return a,c.floor_vertices,poses,rec['floor_observed_min_z_m'],rec['floor_requested_swept_min_z_m']

    def test_borrow_and_partial_final200mm_step_readback(self):
        for before in ((.0944,.1),(.0944,.10557)):
            a,m,p,h,q=self.fixture(before)
            actual=pilot.allocation_row_readback(a,1,m,p,h,q)
            self.assertTrue((actual>=1e-5).all())
            self.assertTrue(a['validation_controller_shared_borrow_applied'][1,1])

    def test_forged_witness_parent_state_travel_and_selector_rejected(self):
        a,m,p,h,q=self.fixture()
        for key in ('shared_floor_donor','shared_borrow_applied','cop_allocation_selected_mask'):
            bad=copy.deepcopy(a);bad['validation_controller_'+key][1]=~bad['validation_controller_'+key][1]
            with self.assertRaises(ValueError):pilot.allocation_row_readback(bad,1,m,p,h,q)
        for key in ('distance_m','cop_travel_m','shared_original_proposed_step_m','shared_candidate_steps_m'):
            bad=copy.deepcopy(a);bad['validation_controller_'+key][1].flat[0]+=.001
            with self.assertRaises(ValueError):pilot.allocation_row_readback(bad,1,m,p,h,q)


if __name__=='__main__':unittest.main()
