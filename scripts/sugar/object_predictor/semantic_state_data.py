"""New fixed464 semantic-state preparation; original training profiles are frozen.

Only data and observation glue lives here. No model is constructed or trained.
The original full H32 encoder and failure-aware recorded-source checks are reused.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from .overfit_data import FIXED_EPISODES, FIXED_FRAMES, input_conflicts
from .conditional_overfit_data import state_history_evidence, precision_eligible_conflict_groups
from .failure_aware_overfit import FailureAwareOverfitDataset
from .overfit_supervision_conflicts import state_supervision_feasibility
from .observed_force_summary import summarize_observed_forces, summarize_support_projections, FORCE_FIELDS, SUPPORT_FIELDS
from .surface_data import collate_surface

STUDY = 'semantic_state_observation_v1'
PROFILE = 'semantic_state_coverage464_v1'
ADDED_FRAMES = tuple(range(49, 1200, 50))
FIT_FRAMES = tuple(sorted((*FIXED_FRAMES, *ADDED_FRAMES)))
EVALUATION_FRAMES = tuple(f for f in range(31, 2400, 25) if f not in FIXED_FRAMES)
HEIGHT_SCALE_M = .2
SUMMARY_FIELDS = (*FORCE_FIELDS, *SUPPORT_FIELDS, 'observed_left_hand_world_height_over_0p2m')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def source_contract(source):
    """Accept only actual nofloor2100, never the one-solve optimizer archive."""
    source = Path(source).resolve()
    p = json.loads((source/'PROTOCOL.json').read_text())
    r = json.loads((source/'RESULT.json').read_text())
    m = json.loads((source/'ARTIFACTS.json').read_text())
    if (p.get('study') != 'failure_aware_fullbatch_refinement_v1' or
            r.get('execution_complete') is not True or r.get('total_optimizer_updates') != 2100 or
            r.get('full_parameter_update_passed') is not True or r.get('full_endpoint_reload_max_abs') != 0 or
            m.get('schema') != 2 or m.get('complete') is not True or m.get('kind') != p['study']):
        raise ValueError('Require complete nofloor2100 full-model and original Adam source')
    for binding in (*m['sources'].values(), *m['artifacts'].values(), m['protocol']):
        path = Path(binding['path'])
        if not path.is_absolute(): path = source/path
        if sha(path) != binding['sha256']:
            raise ValueError('Frozen nofloor2100 binding changed: ' + str(path))
    if p['data']['history'] != 32 or p['data']['history_interval_s'] != .02:
        raise ValueError('Original physical-time H32 contract changed')
    return p, dict(source=str(source), total_optimizer_updates=2100,
        bindings={name:dict(path=str(source/name),sha256=sha(source/name)) for name in
                  ('model.pt','PROTOCOL.json','RESULT.json','ARTIFACTS.json','fit_02100.npz','same_trajectory_interpolation.npz')})


class SemanticStateDataset(FailureAwareOverfitDataset):
    def __init__(self, root, original_protocol, *, role='fit'):
        protocol = deepcopy(original_protocol)
        if role not in ('fit', 'development_interpolation'):
            raise ValueError('Only the fixed fit or already-viewed development clocks are allowed')
        # Respect the frozen five-clock guard. Use its existing additional-clock
        # path for the24 added frames, then compose the new explicitly scoped
        # dataset; never relabel an added window as one of the original80.
        super().__init__(root, protocol, interpolation_frames=EVALUATION_FRAMES if role != 'fit' else None)
        if role == 'fit':
            added = FailureAwareOverfitDataset(root, protocol, interpolation_frames=ADDED_FRAMES)
            self.rows.extend(added.rows)
            self.rows.sort(key=lambda r:(r['metadata']['episode'],r['metadata']['frame']))
            self.conflicts = input_conflicts(self.rows)
            self.qualified = self.qualified and added.qualified
        self.supervision_profile = PROFILE
        self.semantic_role = role
        self.expected_frames = EVALUATION_FRAMES if role != 'fit' else FIT_FRAMES
        self.summary_input_provenance = {}
        for episode in FIXED_EPISODES:
            path = Path(self.collection_records[episode]['source'])/f'episode_{episode}.npz'
            with np.load(path, allow_pickle=False) as archive:
                # This is observable proprioception, not object pose or min_z.
                hand = archive['hand_pose_w']
                times = archive['timestamp_s']
            for row in self.rows:
                if row['metadata']['episode'] != episode: continue
                indices = np.asarray(row['metadata']['history_indices'])
                if len(indices) != 32 or indices[-1] != row['metadata']['frame']:
                    raise ValueError('Original causal history indices changed')
                heights = hand[indices, 0, 2].astype(np.float32)
                if not np.isfinite(heights).all(): raise ValueError('Nonfinite observed hand height')
                row['observed_left_hand_world_height_m'] = heights[:, None].copy()
                row['observed_height_timestamp_s'] = times[indices].copy()
            self.summary_input_provenance[str(episode)] = dict(path=str(path.resolve()),sha256=sha(path),
                field='hand_pose_w[history_indices,0,2]',reference='Public world floor z=0',height_scale_m=HEIGHT_SCALE_M)


def full_observation_groups(rows):
    """All exact observation groups, even when13 targets are identical.

    This is stricter bookkeeping than the old target-conflict-only inventory:
    availability may disagree while the13 continuous targets are identical.
    Actual scaled historical height is an input; target/mask values never enter
    the grouping key. Summary10 is a deterministic function of these points.
    """
    groups = {}
    arrays = []
    for index,row in enumerate(rows):
        values = [np.ascontiguousarray(v) for k in ('coord','grid_coord','feat') for v in row['inputs'][k]]
        values.append(np.ascontiguousarray(row['observed_left_hand_world_height_m']/HEIGHT_SCALE_M))
        h = hashlib.sha256()
        for v in values:
            h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        groups.setdefault(h.digest(),[]).append(index);arrays.append(values)
    out=[]
    for members in groups.values():
        if len(members)<2:continue
        first=arrays[members[0]]
        if not all(len(arrays[i])==len(first) and all(np.array_equal(a,b) for a,b in zip(first,arrays[i])) for i in members):
            raise RuntimeError('Complete observation digest collision')
        out.append(dict(row_indices=members,samples=[rows[i]['metadata'] for i in members]))
    return out


def collate_semantic_observations(rows):
    """Only original encoded observations plus the actual historical hand z."""
    import torch
    points = collate_surface([r['inputs'] for r in rows])
    force = summarize_observed_forces(points)
    support = summarize_support_projections(points)
    heights = np.stack([r['observed_left_hand_world_height_m'] for r in rows])
    if heights.shape != (len(rows),32,1) or not np.isfinite(heights).all():
        raise ValueError('Actual historical left-hand heights must be [B,32,1]')
    summary = torch.cat((force, support, torch.from_numpy(heights/HEIGHT_SCALE_M)), dim=-1)
    return dict(points=points, observation_history=summary)


def qualification(dataset, fit_limits):
    expected = [(e,f) for e in FIXED_EPISODES for f in dataset.expected_frames]
    identities = [(r['metadata']['episode'],r['metadata']['frame']) for r in dataset.rows]
    full_groups = full_observation_groups(dataset.rows)
    state_groups = []
    for group in full_groups:
        members = [i for i in group['row_indices'] if dataset.rows[i]['supervision']['state_precision_eligible']>.5]
        if len(members)>1:state_groups.append(dict(group,row_indices=members))
    state = state_supervision_feasibility(dataset.rows, state_groups,
        center_limit_m=fit_limits['center_cm']/100., size_max_relative_limit=fit_limits['size_max_relative'])
    mass_conflicts = [g for g in full_groups if len({float(dataset.rows[i]['target'][12])
        for i in g['row_indices'] if dataset.rows[i]['supervision']['mass_available']>.5})>1]
    availability_conflicts = [g for g in full_groups if len({bool(dataset.rows[i]['supervision']['mass_available'])
        for i in g['row_indices']})>1]
    checks = dict(exact_fixed_identities=identities==expected,
        every_record_complete=dataset.collection_result.get('complete') is True,
        actual_recorded_source_checks=dataset.failure_source_validation['passed'],
        controlled_source_schema=dataset.controlled_source_validation['passed'],
        actual_com_labels_present=dataset.qualified,
        contact_masks_recomputed=all(all(row['supervision'][k]==v for k,v in state_history_evidence(row['inputs']).items()) for row in dataset.rows),
        no_incompatible_candidate_state=state['passed'], no_conflicting_available_mass=not mass_conflicts,
        fit_and_development_clocks_disjoint=not set(FIT_FRAMES)&set(EVALUATION_FRAMES))
    checks['no_identical_full_observation_availability_conflict']=not availability_conflicts
    ns=int(sum(r['supervision']['state_precision_eligible'] for r in dataset.rows))
    nm=int(sum(r['supervision']['mass_available'] for r in dataset.rows))
    return dict(passed=all(checks.values()),checks=checks,role=dataset.semantic_role,rows=len(dataset),
        state_candidate_rows=ns,prior_unknown_rows=len(dataset)-ns,mass_available_rows=nm,mass_unavailable_rows=len(dataset)-nm,
        physical_success_gate_passed=dataset.collection_result['qualification_passed'],
        original_physical_records={str(e):dataset.collection_records[e]['controller_passed'] for e in FIXED_EPISODES},
        original_point_input_conflicts=dataset.conflicts,full_observation_groups=full_groups,state_feasibility=state,
        available_mass_conflicts=mass_conflicts,unresolved_availability_conflicts=availability_conflicts,
        scope='New464 coverage preparation, original complete failed/successful recordings. Contact only admits a state candidate; passing necessary conflict bounds is not an observability or accuracy certificate.')


def task_denominators(dataset):
    ns=sum(float(r['supervision']['state_precision_eligible']) for r in dataset.rows)
    nm=sum(float(r['supervision']['mass_available']) for r in dataset.rows)
    total=len(dataset)
    if min(ns,nm,total-nm)<=0:raise ValueError('Declared global task/class denominators must be present')
    return dict(state_candidates=int(ns),available_mass=int(nm),availability_positive=int(nm),
        availability_negative=int(total-nm),current_contact_total=total,
        state_rule='Global sum over eligible rows / Ns for center,mesh,size; original physical scales/weights retained.',
        mass_rule='Global sum over available rows / Nm; no contribution or denominator from unavailable rows.',
        availability_rule='0.5*sum(BCE_positive)/Npos + 0.5*sum(BCE_negative)/Nneg.',
        contact_rule='sum(BCE_current_contact)/Nall; explicit secondary observation task.',
        legacy_force_rule='Report the original8 output errors; no force regression term or deterministic-summary learning claim.')
