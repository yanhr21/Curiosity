"""CPU state/clock tests, using synthetic parameter tensors, never a network."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from scripts.sugar.object_predictor.active3d_continuation import (
    digest, formal_continuation_gate, load_resume, nested_equal, restore,
)
from scripts.sugar.object_predictor.train_active3d_overfit import EXPERIMENT, load_cases, public_row


class ContinuationTests(unittest.TestCase):
    @staticmethod
    def parameters():
        # No model forward: this solely exercises Adam's full state transition.
        return torch.nn.ParameterDict({'a': torch.nn.Parameter(torch.tensor([.2, -.7])),
                                       'b': torch.nn.Parameter(torch.tensor([.5]))})

    @staticmethod
    def update(parameters, optimizer, step):
        torch.manual_seed(20260916+step)
        optimizer.zero_grad(set_to_none=True)
        loss = sum((p-torch.randn_like(p)).square().sum() for p in parameters.values())
        loss.backward()
        optimizer.step()

    def saved_state(self, directory):
        p = self.parameters(); o = torch.optim.Adam(p.parameters(), lr=3e-4)
        for step in range(1, 1001):
            self.update(p, o, step)
        directory = Path(directory)
        torch.save(p.state_dict(), directory/'model')
        torch.save(o.state_dict(), directory/'optim')
        resume = dict(model=directory/'model', optimizer=directory/'optim', receipt=dict(
            source_model_sha256=digest(directory/'model'), source_optimizer_sha256=digest(directory/'optim')))
        return p, o, resume

    def test_complete_adam_restore_preserves_next_absolute_seed_update_exactly(self):
        with TemporaryDirectory() as tmp:
            p, o, resume = self.saved_state(tmp)
            q = self.parameters(); qo = torch.optim.Adam(q.parameters(), lr=3e-4)
            restore(q, qo, resume)
            self.assertTrue(nested_equal(p.state_dict(), q.state_dict()))
            self.assertTrue(nested_equal(o.state_dict(), qo.state_dict()))
            self.update(p, o, 1001); self.update(q, qo, 1001)
            self.assertTrue(nested_equal(p.state_dict(), q.state_dict()))
            self.assertTrue(nested_equal(o.state_dict(), qo.state_dict()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_incomplete_or_wrong_clock_adam_is_rejected(self):
        with TemporaryDirectory() as tmp:
            _, _, resume = self.saved_state(tmp)
            original = torch.load(resume['optimizer'], weights_only=True)
            for mode in ('missing_moment', 'wrong_step'):
                state = torch.load(resume['optimizer'], weights_only=True) if mode == 'missing_moment' else original
                first = next(iter(state['state']))
                if mode == 'missing_moment':
                    del state['state'][first]['exp_avg']
                else:
                    state['state'][first]['step'] = torch.tensor(999.)
                torch.save(state, resume['optimizer'])
                resume['receipt']['source_optimizer_sha256'] = digest(resume['optimizer'])
                p = self.parameters(); o = torch.optim.Adam(p.parameters(), lr=3e-4)
                with self.assertRaises(RuntimeError):
                    restore(p, o, resume)

    def test_real_source_contract_and_actual_engine_binding(self):
        root = EXPERIMENT/'overfit_repair_v1'
        args = SimpleNamespace(resume_from=root/'active3d_native_geometry_v1', geometry_repair=True,
            seed=20260916, dataset=EXPERIMENT/'datasets/active3d_official',
            checkpoint=EXPERIMENT/'checkpoints/active3d_official/t_p',
            engine_verification=root/'native_engine_equivalence_v1/RESULT.json')
        rows, selection = load_cases(args.dataset)
        rows = [public_row(row) for row in rows]
        info = load_resume(args, rows, selection)
        self.assertEqual(info['receipt']['additional_updates'], 3000)
        self.assertTrue(info['receipt']['engine_equivalence_passed'])
        args.engine_verification = None
        with self.assertRaises(RuntimeError):
            load_resume(args, rows, selection)
        args.seed += 1
        with self.assertRaises(RuntimeError):
            load_resume(args, rows, selection, require_engine=False)

    def test_continuation_gate_keeps_all_scientific_and_visual_requirements(self):
        names = [f'{ident}_repeat{r}' for ident in ('18704', '11898', '15737', '13266') for r in (0, 1)]
        training = dict(complete=True, fixed_input_cases=8, optimizer_updates=3000, backwards=3000,
            total_optimizer_updates=4000, native_numeric_gate_passed=True, geometry_gate_passed=True,
            checkpoint_reload=dict(passed=True), checkpoint_sha256='synthetic', resume_replay=dict(passed=True),
            continuation=dict(start_absolute_step=1000, end_absolute_step=4000, additional_updates=3000,
                              engine_equivalence_passed=True))
        rendering = dict(complete=True, actual_mesh_stills=24, frames=768, all_frames_decoded=True,
                         records=[dict(name=n) for n in names], checkpoint_sha256='synthetic',
                         rendered_endpoint='step_4000', training_total_optimizer_updates=4000,
                         training_new_optimizer_updates=3000)
        inspection = dict(checkpoint_sha256='synthetic', cases=[dict(name=n, passed=True,
            images=[f'{n}_view{v:02d}.png' for v in (0, 32, 64)]) for n in names])
        self.assertTrue(formal_continuation_gate(training, rendering, inspection)['passed'])
        for key, value in [('optimizer_updates', 2999), ('backwards', 2999), ('total_optimizer_updates', 3000),
                           ('native_numeric_gate_passed', False), ('geometry_gate_passed', False),
                           ('resume_replay', dict(passed=False))]:
            self.assertFalse(formal_continuation_gate(dict(training, **{key: value}), rendering, inspection)['passed'])
        self.assertFalse(formal_continuation_gate(training, dict(rendering, checkpoint_sha256='old'), inspection)['passed'])
        self.assertFalse(formal_continuation_gate(training, dict(rendering, rendered_endpoint='step_1000'), inspection)['passed'])
        self.assertFalse(formal_continuation_gate(training, dict(rendering, training_new_optimizer_updates=1000), inspection)['passed'])
        missing = dict(rendering); missing.pop('rendered_endpoint')
        self.assertFalse(formal_continuation_gate(training, missing, inspection)['passed'])
        self.assertFalse(formal_continuation_gate(training, rendering)['passed'])


if __name__ == '__main__':
    unittest.main()
