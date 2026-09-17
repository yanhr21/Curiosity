import ast
import copy
from pathlib import Path
import unittest
import numpy as np
from scripts.sugar.object_predictor import run_allocated_floor_cop_pilot as pilot
from scripts.sugar.object_predictor.allocated_floor_cop import AllocatedFloorCoPController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests as Fixture


class AllocationPilotTests(unittest.TestCase):
    def test_fixed_baseline_original_gates_and_new_scope(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'floor_safe_qualified_cop_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(len(p['original_physical_checks']),12)
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(p['physics_substeps'],8)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.])
        self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertEqual(p['cop_rule']['per_hand_accumulated_travel_max_m'],.1)
        self.assertEqual(p['late_mass_check']['frame'],2381)
        self.assertFalse(p['automatic_expansion']);self.assertFalse(p['allocation']['postlift_allocation'])

    def test_physics_and_collector_ast_identical_except_explicit_type_and_binding(self):
        folder=Path('scripts/sugar/object_predictor')
        old=(folder/'floor_safe_cop_scene.py').read_text().replace('from .floor_safe_cop import FloorSafeCoPController',
            'from .allocated_floor_cop import AllocatedFloorCoPController').replace('FloorSafeCoPController','AllocatedFloorCoPController').replace('class FloorSafeCoPScene','class AllocatedFloorCoPScene')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse((folder/'allocated_floor_cop_scene.py').read_text())))
        old=(folder/'collect_floor_safe_cop.py').read_text().replace('collect_floor_safe_cop.py','collect_allocated_floor_cop.py').replace('floor_safe_cop_scene','allocated_floor_cop_scene').replace('FloorSafeCoPScene','AllocatedFloorCoPScene')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse((folder/'collect_allocated_floor_cop.py').read_text())))

    def fixture(self):
        poses,field=Fixture().observation(centers=np.array([[0.,0.,.08],[0.,0.,-.08]]))
        poses[:,2]=.00002
        c=AllocatedFloorCoPController(target_load_n=12.,force_gain=.000025,response_gain=True)
        c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)));c.observe(poses,field)
        target,_,rec=c.command(24.,np.full(2,8.),.02)
        a={'timestamp_s':np.array([24.]),'hand_pose_w':target[None]}
        a.update({'validation_controller_'+('distance_m' if k=='distance' else k):np.asarray(v)[None] for k,v in rec.items()})
        return a,c.floor_vertices,poses,rec['floor_observed_min_z_m'],rec['floor_requested_swept_min_z_m']

    def test_actual_partial_transaction_replay_and_forged_choice_rejected(self):
        a,meshes,poses,initial,requested=self.fixture()
        out=pilot.allocation_row_readback(a,0,meshes,poses,initial,requested)
        self.assertTrue((out>=1e-5).all())
        a['validation_controller_cop_allocation_selected_mask'][0]=[True,False]
        with self.assertRaisesRegex(ValueError,'decision replay'):
            pilot.allocation_row_readback(a,0,meshes,poses,initial,requested)

    def test_forged_parent_state_and_suppressed_travel_rejected(self):
        a,meshes,poses,initial,requested=self.fixture()
        for key in ('distance_m','cop_servo_velocity_m_s','cop_allocation_proposed_step_m'):
            bad=copy.deepcopy(a);bad['validation_controller_'+key].flat[0]+=.001
            with self.assertRaises(ValueError):pilot.allocation_row_readback(bad,0,meshes,poses,initial,requested)


if __name__=='__main__':unittest.main()
