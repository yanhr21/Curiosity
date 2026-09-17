"""Run the full official Utonia on qualified real tactile-surface input batches."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import MODES
from .model import ObjectPredictor, disable_stochastic_regularizers, loss_components


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.input)
    if not json.loads((root / 'INPUT_REPORT.json').read_text())['passed']:
        raise RuntimeError('Physical input qualification did not pass')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    released = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict']
    checks = {}; details = {}; readout_reference = None
    for mode in MODES:
        torch.manual_seed(310016); torch.cuda.manual_seed_all(310016)
        with np.load(root / (mode + '.npz')) as arrays:
            batch = {k: torch.from_numpy(arrays[k]).cuda() for k in arrays.files}
        model = ObjectPredictor(args.checkpoint, history=32, deterministic_pooling=True).cuda()
        state = model.backbone.state_dict()
        checks[mode + '/every_official_tensor_exact'] = all(torch.equal(
            state[k.replace('embedding.stem.linear.', 'embedding.stem.linear.original.')].cpu(), value)
            for k, value in released.items())
        checks[mode + '/full_parameter_count'] = model.original_parameter_count == 137253744 and sum(p.numel() for p in model.parameters()) == 138407503
        if readout_reference is None:
            readout_reference = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        checks[mode + '/matched_readout_initialization'] = all(torch.equal(v.detach().cpu(), readout_reference[k]) for k, v in model.head.state_dict().items())
        checks[mode + '/extra_affine_zero'] = not bool(torch.count_nonzero(model.backbone.embedding.stem.linear.extra.weight))
        model.train(); disable_stochastic_regularizers(model)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
        prediction = model(batch); parts = loss_components(prediction, batch['target']); sum(parts.values()).backward()
        torch.cuda.synchronize(); elapsed = time.monotonic() - start
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        checks[mode + '/only_official_unused_mask_token_has_no_gradient'] = missing == ['backbone.embedding.mask_token']
        checks[mode + '/every_used_gradient_finite'] = not bad
        for stage in range(5):
            checks[f'{mode}/stage{stage}_gradient_nonzero'] = any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                for n, p in model.named_parameters() if f'enc.enc{stage}.' in n)
        checks[mode + '/finite_output_loss'] = bool(torch.isfinite(prediction).all()) and all(bool(torch.isfinite(p)) for p in parts.values())
        model.eval()
        with torch.no_grad():
            first = model(batch); second = model(batch)
        checks[mode + '/frozen_repeat_exact'] = torch.equal(first, second)
        details[mode] = dict(points=int(len(batch['coord'])), input_frames=len(batch['offset']),
            histories=int(len(batch['target'])), feature_width=int(batch['feat'].shape[1]),
            full_forward_backward_s=elapsed, peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
            losses={k: float(v.detach()) for k, v in parts.items()}, missing_gradients=missing,
            nonfinite_gradients=bad, repeat_max_difference=float((first-second).abs().max()))
        print('FULL_MODEL', mode, json.dumps(details[mode]), flush=True)
        del model, state, batch, prediction, parts, first, second
        gc.collect(); torch.cuda.empty_cache()
    report = dict(passed=all(checks.values()), checks=checks, details=details,
                  optimizer_updates=0, scope='Compatibility and complete-backbone gradient qualification only. No new trained model or held-out accuracy result.')
    (root / 'FULL_MODEL_REPORT.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); parser.add_argument('--checkpoint', required=True)
    main(parser.parse_args())
