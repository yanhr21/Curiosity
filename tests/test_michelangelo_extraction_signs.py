"""Pure CPU sign direction, ambiguity, exact-replay and call-accounting tests."""
import ast
import inspect
import unittest

import numpy as np
from scripts.sugar.object_predictor import audit_michelangelo_extraction_signs as audit


class ExtractionSignTests(unittest.TestCase):
    def test_fp_fn_boundary_and_zero_sign(self):
        logits=np.array([1.,-1.,1.,-1.,0.,-1.],np.float32)
        labels=np.array([False,True,True,False,False,True])
        boundary=np.array([False,False,False,False,False,True])
        arrays,m=audit.sign_arrays_and_metrics(logits,labels,boundary)
        self.assertEqual(m['false_positive'],2)
        self.assertEqual(m['false_negative'],1)
        self.assertEqual(m['true_positive'],1)
        self.assertEqual(m['true_negative'],1)
        self.assertEqual(m['valid_nodes'],5)
        self.assertEqual(m['ambiguous_raw_sign_disagreements'],1)
        np.testing.assert_array_equal(arrays['grid_false_positive'],[True,False,False,False,True,False])
        np.testing.assert_array_equal(arrays['grid_false_negative'],[False,True,False,False,False,False])
        self.assertFalse(m['all_nonboundary_signs_correct'])

    def test_exact_replay_does_not_allow_small_error_or_dtype_change(self):
        saved=np.array([0.,1.],np.float32)
        self.assertTrue(audit.exact_comparison(saved.copy(),saved)['exact'])
        changed=saved.copy();changed[1]=np.nextafter(changed[1],np.float32(2.))
        self.assertFalse(audit.exact_comparison(changed,saved)['exact'])
        self.assertFalse(audit.exact_comparison(saved.astype(np.float64),saved)['exact'])
        changed=saved.copy();changed[0]=-0.
        self.assertFalse(audit.exact_comparison(changed,saved)['exact'])

    def test_nonfinite_logits_refused(self):
        with self.assertRaises(ValueError):
            audit.sign_arrays_and_metrics(np.array([np.nan]),np.array([False]),np.array([False]))

    def test_call_attempt_and_completion_are_distinct(self):
        counts=audit.new_counts()
        self.assertEqual(audit.counted(counts,'encode',lambda:3),3)
        def failure():raise RuntimeError('diagnostic failure')
        with self.assertRaises(RuntimeError):audit.counted(counts,'encode',failure)
        self.assertEqual(counts['encode'],dict(attempted=2,completed=1))

    def test_all_case_exact_gate_precedes_original_extractor(self):
        tree=ast.parse(inspect.getsource(audit.run))
        # All dense work is guarded by the explicit all-case exact condition.
        guards=[x for x in ast.walk(tree) if isinstance(x,ast.If)
                and isinstance(x.test,ast.Name) and x.test.id=='exact']
        self.assertEqual(len(guards),1)
        calls=[x for x in ast.walk(guards[0]) if isinstance(x,ast.Call)
               and isinstance(x.func,ast.Name) and x.func.id=='counted'
               and len(x.args)>2 and isinstance(x.args[2],ast.Name) and x.args[2].id=='extract_geometry']
        self.assertEqual(len(calls),1)


if __name__=='__main__':
    unittest.main()
