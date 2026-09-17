"""CPU-only protocol and saved-observation validation; no simulator or model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor import run_prelift_cop_pilot as pilot


class CoPPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=dict(load=12.,approach_angle_deg=0.,yaw_delta_deg=0.,lift_height_m=.18,lateral_xy=[0.,.04])
        c=pilot.PreliftCoPAlignmentController(target_load_n=12.,force_gain=.000025,
            response_gain=True,controller_revision='sensor_feedback_v2')
        turns=[Rotation.from_euler('y',sign*90,degrees=True)*frame[0]
               for sign,frame in zip((1,-1),c.frames)]
        poses=np.array([[-.2,0.,.3,*turns[0].as_quat()],[.2,0.,.3,*turns[1].as_quat()]])
        local=np.array([[0.,y,z] for y in (-.002,0.,.002) for z in (-.002,0.,.002)])
        field=dict(pos=np.concatenate([t.inv().apply(local) for t in turns]),
            area=np.full(18,1e-6),pressure=np.full(18,12./9e-6),
            pad=np.repeat([0,27],9),patch=np.repeat([0,1],9))
        records=[]
        for i in range(2400):
            observed=field if i else {k:v[:0] for k,v in field.items()}
            c.observe(poses,observed)
            _,_,record=c.command((i+1)*.02,np.full(2,12.) if i else np.zeros(2),.02)
            records.append(record)
        cls.arrays={'timestamp_s':.02*np.arange(1,2401),'hand_pose_w':np.broadcast_to(poses,(2400,2,7)).copy()}
        for key in (*pilot.TELEMETRY,'lift_start_s','ready_seconds','fit_valid','alignment_error_deg'):
            cls.arrays['validation_controller_'+key]=np.stack([r[key] for r in records])
        cls.surface=dict(offset=np.arange(2401)*18,
            **{key:np.concatenate([field[source]]*2400) for key,source in
               [('position_hand_frame_m','pos'),('area_m2','area'),('normal_pressure_pa','pressure'),('pad','pad'),('hand','patch')]})

    def test_fixed_three_original_configurations_and_original_thresholds(self):
        p=pilot.protocol();baseline=json.loads((pilot.BASELINE/'PROTOCOL.json').read_text())
        self.assertEqual(tuple(c['episode'] for c in p['configurations']),(5014,5012,5000))
        for c in p['configurations']:
            self.assertEqual(c,next(x for x in baseline['configurations'] if x['episode']==c['episode']))
        self.assertEqual(p['baseline_intervention'],'freeze_postlift_alignment_v1')
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(len(p['original_physical_checks']),12)
        self.assertEqual(p['late_mass_check']['criteria'],pilot.fixed_protocol()['mass_criteria'])
        self.assertFalse(p['automatic_training']);self.assertFalse(p['automatic_expansion'])

    def test_cpu_prepare_binds_all_original_sources_and_rejects_protocol_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'new';pilot.prepare(root);pilot.validate_protocol(root)
            p=json.loads((root/'PROTOCOL.json').read_text());p['configurations'][0]['load']+=1
            pilot.write(root/'PROTOCOL.json',p)
            with self.assertRaises(ValueError):pilot.validate_protocol(root)

    def test_actual_controller_telemetry_and_previous_observation_geometry_replay(self):
        summary=pilot.telemetry_readback(self.arrays,self.config)
        self.assertEqual(summary['admission_time_s'],24.)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);np.savez_compressed(folder/'contact_surface.npz',**self.surface)
            checked=pilot.geometry_readback(self.arrays,folder,self.config)
            self.assertEqual(checked['previous_observation_rows_replayed'],2399)
            self.assertLessEqual(checked['max_geometry_abs_difference'],1e-10)

    def test_postlift_translation_and_missing_sustained_gate_are_rejected(self):
        arrays=copy.deepcopy(self.arrays)
        arrays['validation_controller_cop_servo_velocity_m_s'][1500,0,0]=1e-4
        with self.assertRaises(ValueError):pilot.telemetry_readback(arrays,self.config)
        arrays=copy.deepcopy(self.arrays)
        arrays['validation_controller_cop_line_angle_deg'][1190]=10.
        arrays['validation_controller_cop_admission_ready'][1190]=False
        with self.assertRaises(ValueError):pilot.telemetry_readback(arrays,self.config)

    def test_travel_budget_and_changed_geometry_cannot_pass_saved_validation(self):
        arrays=copy.deepcopy(self.arrays);arrays['validation_controller_cop_travel_m'][50:,0]=.1001
        with self.assertRaises(ValueError):pilot.telemetry_readback(arrays,self.config)
        arrays=copy.deepcopy(self.arrays);arrays['validation_controller_cop_world_m'][20,0,0]+=.001
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);np.savez_compressed(folder/'contact_surface.npz',**self.surface)
            with self.assertRaises(ValueError):pilot.geometry_readback(arrays,folder,self.config)


if __name__=='__main__':unittest.main()
