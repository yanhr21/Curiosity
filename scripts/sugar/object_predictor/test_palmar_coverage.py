"""Sensor-routing invariants; synthetic inputs are unit tests only."""
import unittest
import numpy as np
from .palmar_coverage import continuous_palmar_field, ContinuousPalmarField


class PalmarCoverageTests(unittest.TestCase):
    def field(self):
        return dict(pos=np.array([[0.,-.02,.004],[0.,.02,.004],[0.,.02,.004],[0.,-.02,.004],
                                  [.12,-.02,.007],[.12,.02,.007],[0.,0.,0.]]),
                    patch=np.array([0,1,0,1,0,1,0]),pad=np.array([-1,-1,-1,-1,20,47,-1]),
                    area=np.arange(1,8)*1e-6,pressure=np.arange(1,8)*100.,
                    traction_vec=np.arange(21).reshape(7,3).astype(float))

    def test_only_unassigned_palmar_faces_gain_coverage(self):
        f=self.field();r=continuous_palmar_field(f)
        self.assertTrue((r['pad'][:2]>=0).all())
        np.testing.assert_array_equal(r['pad'][2:],[-1,-1,20,47,-1])
        np.testing.assert_array_equal(r['pad'][:2]//27,[0,1])

    def test_preserves_source_labels_positions_and_physical_channels(self):
        f=self.field();saved={k:v.copy() for k,v in f.items()};r=continuous_palmar_field(f)
        for k in f:np.testing.assert_array_equal(f[k],saved[k])
        for k in ('pos','patch','area','pressure','traction_vec'):np.testing.assert_array_equal(r[k],f[k])
        np.testing.assert_array_equal(r['anatomical_pad'],f['pad'])
        r['pad'][0]=25
        self.assertEqual(f['pad'][0],-1)

    def test_no_validation_truth_required_or_used(self):
        f=self.field();a=continuous_palmar_field(f)
        b=continuous_palmar_field({**f,'normal':np.full((7,3),np.nan),'object_mass':float('nan')})
        np.testing.assert_array_equal(a['pad'],b['pad'])

    def test_empty_and_invalid_identity(self):
        f=self.field();empty={k:v[:0] for k,v in f.items()}
        self.assertEqual(len(continuous_palmar_field(empty)['pad']),0)
        f['pad'][0]=30
        with self.assertRaises(ValueError):continuous_palmar_field(f)

    def test_wrapper_preserves_raw_field_and_updates(self):
        class Raw:
            total=7
            capacity=100
            def to_numpy(self):return fixture
            def update(self,value):self.total=value
        fixture=self.field();raw=Raw();wrapped=ContinuousPalmarField(raw,(-1.,1.))
        self.assertEqual(wrapped.total,7);self.assertEqual(wrapped.capacity,100)
        self.assertEqual(raw.to_numpy()['pad'][0],-1)
        self.assertGreaterEqual(wrapped.to_numpy()['pad'][0],0)
        wrapped.update(8);self.assertEqual(raw.total,8)


if __name__=='__main__':unittest.main()
