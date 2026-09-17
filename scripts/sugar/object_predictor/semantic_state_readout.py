"""Observation-only linear glue around the ORIGINAL complete Utonia readouts.

No backbone, temporal network or alternative state predictor is implemented.
Original parameters/names/Adam objects stay intact. New352-column blocks have
their own Parameters and fresh Adam clocks, and are zero at installation.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

HISTORY=32
SUMMARY_WIDTH=11
POINT_KEYS=frozenset(('coord','grid_coord','feat','offset'))


def fit_observation_scales(train_history):
    """Only fixed464 TRAIN histories; no centering, labels or dense statistics."""
    import numpy as np
    values=np.asarray(train_history)
    if values.shape!=(464,HISTORY,SUMMARY_WIDTH) or values.dtype!=np.float32 or not np.isfinite(values).all():
        raise ValueError('Require the complete actual464 finite observed histories')
    rms=np.sqrt(np.mean(values[:,:,:10].astype(np.float64)**2,axis=(0,1)))
    scales=np.r_[np.where(rms>0,rms,1.),1.].astype(np.float32)
    normalized=values/scales
    l1=np.abs(normalized).sum(axis=(1,2))
    return scales,dict(force10_rms_n=rms.tolist(),scales=scales.tolist(),mean_centering=False,
        zero_rms_columns=np.flatnonzero(rms==0).tolist(),height_already_divided_by_public_0p2m=True,
        fit_rows=464,history_frames=14848,evaluation_statistics_used=False,
        normalized_l1=dict(minimum=float(l1.min()),median=float(np.median(l1)),maximum=float(l1.max())))


def scale_observation_history(history,scales):
    if history.ndim!=3 or history.shape[1:]!=(HISTORY,SUMMARY_WIDTH) or scales.shape!=(SUMMARY_WIDTH,):
        raise ValueError('Observation history/scales have the wrong declared shape')
    scales=scales.to(history)
    if not bool(torch.isfinite(scales).all()) or bool((scales<=0).any()) or float(scales[-1])!=1.:
        raise ValueError('Force RMS scales must be finite positive; public height scale remains fixed')
    return history/scales


class ObservationLinear(nn.Module):
    """Mathematical per-frame concatenation, old-width GEMM for exact zero start."""
    def __init__(self, original, *, classification_only=False):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.in_features%HISTORY or original.bias is None:
            raise ValueError('Require the existing full chronological linear readout')
        if classification_only and original.out_features!=10:
            raise ValueError('Original auxiliary has8force plus2classification rows')
        self.in_features=original.in_features
        self.out_features=original.out_features
        self.frame_features=original.in_features//HISTORY
        self.logical_augmented_in_features=HISTORY*(self.frame_features+SUMMARY_WIDTH)
        # Reuse actual Parameter objects under identical original names.
        self.weight=original.weight
        self.bias=original.bias
        self.classification_only=bool(classification_only)
        outputs=2 if classification_only else original.out_features
        self.summary_weight=nn.Parameter(original.weight.new_zeros((outputs,HISTORY*SUMMARY_WIDTH)))
        self._history=None
        self.zero_effect_enabled=True
        self.zero_effect_calls=0
        self.zero_effect_items=0

    def forward(self,x):
        if self._history is None:
            raise RuntimeError('Explicit observation context is required for the augmented readout')
        history=self._history
        if x.ndim!=2 or x.shape!=(history.shape[0],self.in_features):
            raise ValueError('Original per-sample chronological feature layout changed')
        original=F.linear(x,self.weight,self.bias)
        delta=F.linear(history.reshape(len(history),-1),self.summary_weight,None)
        if self.classification_only:
            value=torch.cat((original[:,:8],original[:,8:]+delta),dim=1)
        else:
            value=original+delta
        if self.zero_effect_enabled:
            if bool(torch.count_nonzero(self.summary_weight)) or not torch.equal(value,original):
                raise RuntimeError('Initial observation block is not exactly zero-effect')
            self.zero_effect_calls+=1;self.zero_effect_items+=len(x)
        return value

    def expanded_weight(self):
        """Review/test view only: [oldframe0,newframe0,oldframe1,newframe1,...]."""
        new=self.summary_weight.reshape(-1,HISTORY,SUMMARY_WIDTH)
        if self.classification_only:
            new=torch.cat((new.new_zeros((8,HISTORY,SUMMARY_WIDTH)),new),dim=0)
        return torch.cat((self.weight.reshape(self.out_features,HISTORY,self.frame_features),new),dim=2).reshape(
            self.out_features,self.logical_augmented_in_features)


def install_semantic_readouts(model):
    """Call AFTER strict nofloor2100 model/Adam restoration, before new forwards."""
    if isinstance(model.predictor.head,ObservationLinear) or isinstance(model.auxiliary,ObservationLinear):
        raise ValueError('Observation blocks already installed')
    before={n:id(p) for n,p in model.named_parameters()}
    model.predictor.head=ObservationLinear(model.predictor.head)
    model.auxiliary=ObservationLinear(model.auxiliary,classification_only=True)
    after=dict(model.named_parameters())
    if any(id(after[n])!=identity for n,identity in before.items()):
        raise RuntimeError('Original full-model parameter name or identity changed')
    added=set(after)-set(before)
    if added!={'predictor.head.summary_weight','auxiliary.summary_weight'}:
        raise RuntimeError('Only the two explicit observation blocks may be added')
    return dict(original_parameter_names_and_identities_exact=True,
        added_parameter_names=sorted(added),added_parameters=sum(after[n].numel() for n in added),
        old_readout_width=model.predictor.head.in_features,
        logical_augmented_width=model.predictor.head.logical_augmented_in_features,
        new_classification_outputs=2,legacy_force_new_columns_fixed_zero=True)


@contextmanager
def observation_context(model,history):
    modules=(model.predictor.head,model.auxiliary)
    if not all(isinstance(m,ObservationLinear) for m in modules):
        raise ValueError('Install the original-readout observation blocks first')
    if (not isinstance(history,torch.Tensor) or history.ndim!=3 or history.shape[1:]!=(HISTORY,SUMMARY_WIDTH)
            or history.dtype!=modules[0].weight.dtype or history.device!=modules[0].weight.device
            or history.requires_grad or not bool(torch.isfinite(history).all())):
        raise ValueError('Require finite non-gradient observed history[B,32,11] on model device')
    if any(m._history is not None for m in modules):raise RuntimeError('Reentrant observation context forbidden')
    try:
        for m in modules:m._history=history
        yield
    finally:
        for m in modules:m._history=None


def forward_semantic_observations(model,points,history):
    """The unchanged original complete-model forward receives its original4keys."""
    if set(points)!=POINT_KEYS:raise ValueError('Only the four original point observations are allowed')
    if len(points['offset'])!=len(history)*HISTORY:
        raise ValueError('Sample-major H32 observations/summary mapping differs')
    with observation_context(model,history):
        return model(points)


def add_observation_optimizer_groups(model,optimizer,*,state_lr,classification_lr):
    if min(state_lr,classification_lr)<=0:raise ValueError('Explicit positive new-block learning rates required')
    expected=('full_official_backbone','sensor_affine','state_readout','auxiliary_readout')
    if tuple(g.get('name') for g in optimizer.param_groups)!=expected:
        raise ValueError('Require the original restored nofloor2100 four-group Adam')
    old={id(p):optimizer.state.get(p) for g in optimizer.param_groups for p in g['params']}
    new=(model.predictor.head.summary_weight,model.auxiliary.summary_weight)
    if any(id(p) in old or p in optimizer.state for p in new):
        raise ValueError('New observation blocks must have fresh empty Adam state')
    for p,lr,name,source in zip(new,(state_lr,classification_lr),
            ('semantic_state_observation','semantic_classification_observation'),optimizer.param_groups[2:4]):
        group={k:v for k,v in source.items() if k not in ('params','name','lr')}
        optimizer.add_param_group(dict(group,params=[p],name=name,lr=lr))
    params=[p for g in optimizer.param_groups for p in g['params']]
    if len({id(p) for p in params})!=len(params) or {id(p) for p in params}!={id(p) for p in model.parameters()}:
        raise RuntimeError('Complete official model and new blocks must be bound exactly once')
    if any(optimizer.state.get(p) is not old[id(p)] for p in params if id(p) in old):
        raise RuntimeError('Original Adam state objects changed')
    return dict(original_state_objects_exact=True,new_blocks_have_no_state=True,
        new_parameter_clock_before_first_step=0,original_parameter_clock=2100,
        note='Separate newParameters avoid sharing old2100 bias correction; legacy force rows have no new observation block.')


def end_zero_effect_checks(model):
    report={}
    for name,module in (('state',model.predictor.head),('classification',model.auxiliary)):
        report[name]=dict(calls=module.zero_effect_calls,items=module.zero_effect_items)
        module.zero_effect_enabled=False
    return report
