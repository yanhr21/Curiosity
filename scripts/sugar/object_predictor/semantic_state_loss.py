"""Explicit GLOBAL task denominators for the new semantic464 proposal.

Summing these microbatch contributions equals the declared complete464
objective. Never divide again by number of microbatches. Original8 force loss is
not a semantic task; its predictions/errors remain legacy reporting only.
"""
import torch
from torch.nn import functional as F
from .overfit_model import LOSS_SCALES, state_vertices, symmetric_mesh_distance

WEIGHTS=dict(center=.5,mesh=1.,size=.5,mass=1.,availability=1.,contact=.25)


def semantic_loss_parts(output,batch,normalized_sample_vertices,denominators):
    state,target=output['state'],batch['target']
    ns=denominators['state_candidates'];nm=denominators['available_mass']
    npos=denominators['availability_positive'];nneg=denominators['availability_negative']
    nall=denominators['current_contact_total']
    if min(ns,nm,npos,nneg,nall)<=0 or nm!=npos or npos+nneg!=nall:
        raise ValueError('Malformed fixed global task/class denominators')
    selected=batch['state_precision_eligible']>.5
    available=batch['mass_available']>.5
    zero=state.sum()*0
    if bool(selected.any()):
        prediction,truth=state[selected],target[selected]
        center=(prediction[:,:3]-truth[:,:3]).norm(dim=1).sum()/(ns*LOSS_SCALES['center_m'])
        mesh=symmetric_mesh_distance(state_vertices(prediction,normalized_sample_vertices),
            state_vertices(truth,normalized_sample_vertices)).sum()/(ns*LOSS_SCALES['mesh_m'])
        size=torch.expm1(prediction[:,9:12]-truth[:,9:12]).abs().mean(dim=1).sum()/(ns*LOSS_SCALES['size_relative'])
    else:center=mesh=size=zero
    mass=(torch.expm1(state[available,12]-target[available,12]).abs().sum()/(nm*LOSS_SCALES['mass_relative'])
          if bool(available.any()) else zero)
    bce=F.binary_cross_entropy_with_logits(output['availability_logit'],batch['mass_available'],reduction='none')
    availability=.5*bce[available].sum()/npos+.5*bce[~available].sum()/nneg
    contact=F.binary_cross_entropy_with_logits(output['contact_logit'],batch['physics']['contact_present'][:,0],reduction='sum')/nall
    return dict(center=center,mesh=mesh,size=size,mass=mass,availability=availability,contact=contact)


def semantic_total_loss(parts):
    if set(parts)!=set(WEIGHTS):raise ValueError('Semantic objective excludes legacy force regression explicitly')
    return sum(WEIGHTS[k]*v for k,v in parts.items())
