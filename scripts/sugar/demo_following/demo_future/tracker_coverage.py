"""Actual recovery-data and stateful-resume glue for the full official Tracker."""
import json
from pathlib import Path
import numpy as np
import torch


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def recovery_pool(base: Path, old_records, old_sources, arm):
    run = base / 'matched_tracker_future_plan'
    if not json.loads((run/'ON_STUDENT_STATE_READBACK.json').read_text())['interface_and_replay_passed']:
        raise RuntimeError('Student-state replay/interface failed')
    if not json.loads((run/'FUTURE_TRACKER_RECOVERY_READBACK.json').read_text())['passed']:
        raise RuntimeError('Actual recovery feasibility failed')
    sources = [run/'future_plan'/('teacher_handoff100_'+a) for a in ('original', 'alternate')]
    records = []
    for path in sources:
        result = json.loads((path/'RESULT.json').read_text())
        if not result['full_budget_without_reset'] or not result['at_least_ten_consecutive_lifted_frames']:
            raise RuntimeError('Incomplete recovery source')
        if not json.loads((path/'TEACHER_INTERFACE_AUDIT.json').read_text())['passed']:
            raise RuntimeError('Recovery teacher interface failed')
        record = dict(np.load(path/'TRACE.npz', allow_pickle=False))
        if record['done'].any() or record['planned_reference_command'].shape != (400,1,8,36):
            raise RuntimeError('Recovery source shape or reset changed')
        if np.max(abs(record['planned_reference_command'][:,:,0]-record['reference_command'])) > 1e-5:
            raise RuntimeError('Recovery plan/current command disagree')
        records.append(record)
    for key in ('teacher_observation', 'planned_reference_command', 'executed_action', 'robot_body_state_before_w'):
        if not np.array_equal(records[0][key][:158], records[1][key][:158]):
            raise RuntimeError('Recovery references differ before selection')
    common = old_records + records
    selected = old_records + (old_records if arm == 'replay_control' else records)
    selected_sources = old_sources + (old_sources if arm == 'replay_control' else sources)
    return selected, selected_sources, common, old_sources + sources


def restore_optimizer(alg, payload):
    alg.optimizer.load_state_dict(payload['optimizer_state_dict'])
    alg.update_step = int(payload['infos']['bcppo_update_step'])
    clocks = sorted({int(v['step']) for v in alg.optimizer.state.values() if 'step' in v})
    passed = equal_state(alg.optimizer.state_dict(), payload['optimizer_state_dict'])
    if not passed or alg.update_step != 256 or clocks != [5120]:
        raise RuntimeError('Full parent Adam moments/clocks or BCPPO stage failed restoration')
    if any(group['lr'] != 1e-4 for group in alg.optimizer.param_groups):
        raise RuntimeError('Parent optimizer learning rate differs')
    return dict(passed=True, optimizer_state_exact=True, bcppo_update_step=256,
                optimizer_clocks=clocks, newly_applied_updates=0,
                rng='Explicit new matched seed272042; parent did not store RNG state')
