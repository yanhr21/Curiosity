"""Numerical gain invariants, not simulated physical qualification."""
import unittest
import numpy as np
from .response_gain import ObservedForceGain


class ResponseGainTests(unittest.TestCase):
    def test_soft_response_rise_is_bounded_and_capped(self):
        c=ObservedForceGain();old=.000025
        for i in range(100):
            gain,upper,_=c.update(0,i*.02,2.+3000*i*2e-6,i*2e-6,.02,True)
            self.assertLessEqual(gain,old*1.05+1e-15);self.assertLessEqual(gain,.00015)
            if upper:self.assertLessEqual(gain*.02*upper,.1+1e-12)
            old=gain
        self.assertAlmostEqual(gain,.00015);self.assertAlmostEqual(upper,3000.)

    def test_hard_response_reduces_gain_without_delay(self):
        c=ObservedForceGain()
        for i in range(15):gain,upper,_=c.update(0,i*.02,2.+1e6*i*2e-6,i*2e-6,.02,True)
        self.assertAlmostEqual(gain,5e-6);self.assertAlmostEqual(upper,1e6)

    def test_no_excitation_retains_last_response_bound(self):
        c=ObservedForceGain()
        for i in range(15):c.update(0,i*.02,2.+1e6*i*2e-6,i*2e-6,.02,True)
        for i in range(15,200):gain,upper,n=c.update(0,i*.02,30.,28e-6,.02,True)
        self.assertEqual(n,0);self.assertAlmostEqual(upper,1e6);self.assertAlmostEqual(gain,5e-6)

    def test_invalid_geometry_never_accelerates(self):
        c=ObservedForceGain()
        for i in range(100):gain,_,_=c.update(0,i*.02,2.+3000*i*2e-6,i*2e-6,.02,False)
        self.assertLessEqual(gain,.000025)

    def test_missing_response_and_side_independence(self):
        c=ObservedForceGain()
        for i in range(100):gain,upper,n=c.update(0,i*.02,2.,i*2e-6,.02,True)
        self.assertEqual(upper,0.);self.assertEqual(n,0);self.assertEqual(gain,.000025)
        np.testing.assert_array_equal(c.upper,[0.,0.]);self.assertEqual(len(c.samples[1]),0)

    def test_bad_clock_rejected(self):
        c=ObservedForceGain();c.update(0,1.,2.,.01,.02,True)
        with self.assertRaises(ValueError):c.update(0,1.,2.,.01,.02,True)


if __name__=='__main__':unittest.main()
