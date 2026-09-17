"""CPU-only protocol and immediate-baseline checks for the angular pilot."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from .fixture_settling_pilot import PAIRS, protocol, validate_protocol, read_baseline
from .fixture_touch_scene import FixtureDrive


class AngularFixtureProtocolTests(unittest.TestCase):
    def test_default_settling_protocol_remains_exact_archived_declaration(self):
        old=json.loads(Path('experiments/object_predictor_v1/overfit_repair_v1/FIXTURE_SETTLING_PROTOCOL.json').read_text())
        for key,value in protocol().items():self.assertEqual(old[key],value,key)
        self.assertEqual(FixtureDrive().force_gain_m_ns,.001)
        self.assertEqual(FixtureDrive().angular_kp_nm_rad,8.)

    def test_only_angular_stiffness_differs_from_immediate_gain_baseline(self):
        old=protocol();new=protocol('fixture_angular_stiffness_pilot_v1')
        diff={k:(old['fixture_drive'][k],v) for k,v in new['fixture_drive'].items() if old['fixture_drive'][k]!=v}
        self.assertEqual(diff,{'angular_kp_nm_rad':(8.,32.)})
        cfg,attempts=validate_protocol(new)
        self.assertEqual(asdict(cfg),new['fixture_drive'])
        self.assertEqual(attempts,[dict(object_id=o,direction_index=d) for o,d in PAIRS])
        for k in ('controls_per_attempt','dt','substeps','sample_frame'):
            self.assertEqual(old[k],new[k])

    def test_no_gate_clock_case_or_other_drive_changes_are_accepted(self):
        original=protocol('fixture_angular_stiffness_pilot_v1')
        for key,value in [('controls_per_attempt',601),('dt',.01),('substeps',16),('sample_frame',500)]:
            altered=deepcopy(original);altered[key]=value
            with self.assertRaises(ValueError,msg=key):validate_protocol(altered)
        for key in ('angular_kd_nms_rad','angular_effort_limit_nm','force_gain_m_ns','target_load_n','sample_s','linear_kp_n_m'):
            altered=deepcopy(original);altered['fixture_drive'][key]*=2
            with self.assertRaises(ValueError,msg=key):validate_protocol(altered)
        altered=deepcopy(original);altered['attempt_order']=list(reversed(altered['attempt_order']))
        with self.assertRaises(ValueError):validate_protocol(altered)
        with self.assertRaises(ValueError):validate_protocol(dict(original,study='unknown'))

    def test_baseline_is_corrected_three_probe_result_and_incident_is_preserved(self):
        data,binding=read_baseline(protocol('fixture_angular_stiffness_pilot_v1'))
        self.assertTrue(binding['report'].endswith('/fixture_settling_pilot_v1/report_r1/RESULT.json'))
        self.assertEqual(len(data['cases']),3)
        self.assertEqual([c['qualification_passed'] for c in data['cases']],[True,True,False])
        self.assertIn('Shapely',binding['incident']['problem'])
        self.assertEqual(binding['incident']['new_physics'],0)
        self.assertEqual(len(binding['report_sha256']),64)


if __name__=='__main__':unittest.main()
