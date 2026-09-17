"""Full official Utonia with independent overfit task readouts and metric losses.

The original ObjectPredictor forward, its complete pretrained backbone, and its
20-column sensor affine remain intact.  Extra linear readouts supervise existing
observed force information and learned availability; they are not replacement
backbones, temporal models, or shape-reconstruction networks.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F


FORCE_TARGET_FIELDS = (
    ('normal_load_by_encoded_side_n', 2),
    ('shear_on_hand_current_frame_n', 3),
    ('normal_on_object_approx_current_frame_n', 3),
)
FORCE_DIM = sum(size for _, size in FORCE_TARGET_FIELDS)
LOSS_SCALES = dict(center_m=.01, mesh_m=.01, size_relative=.05,
                   mass_relative=.05, force_n=.5)
LOSS_WEIGHTS = dict(center=.5, mesh=1., size=.5, mass=1., force=1.,
                    availability=1., contact=.25)


class FullUtoniaOverfit(nn.Module):
    def __init__(self, checkpoint, history=32):
        super().__init__()
        from .model import ObjectPredictor
        self.predictor = ObjectPredictor(checkpoint, history=history, deterministic_pooling=True)
        self.history = history
        self.auxiliary = nn.Linear(self.predictor.feature_dim * history, FORCE_DIM + 2)
        nn.init.normal_(self.auxiliary.weight, std=1e-4)
        nn.init.zeros_(self.auxiliary.bias)

    def forward(self, inputs):
        # Capture the original readout input; do not duplicate or replace the
        # original official-backbone/pooling/chronological-readout forward path.
        captured = []
        handle = self.predictor.head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            state = self.predictor(inputs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one original chronological readout')
        auxiliary = self.auxiliary(captured[0])
        return dict(state=state, force=auxiliary[:, :FORCE_DIM],
                    availability_logit=auxiliary[:, FORCE_DIM],
                    contact_logit=auxiliary[:, FORCE_DIM + 1])


def rotation_matrices(rot6):
    """Same six-dimensional coordinate conversion as the archived model helper."""
    a, b = rot6[..., :3], rot6[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def state_vertices(state, normalized_vertices):
    """Known full-mesh geometry in the current LEFT HAND frame, in meters."""
    local = normalized_vertices[None] * state[:, None, 9:12].exp()
    return local @ rotation_matrices(state[:, 3:9]).transpose(-1, -2) + state[:, None, :3]


def detached_neighbor_indices(predicted, target):
    """Exact CPU KDTree indices; torch below differentiates the actual distances.

    This is the declared sampled known-mesh geometric task loss, not official
    PyTorch3D Chamfer or the Active3D training backend.
    """
    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    forward, reverse = [], []
    for pred, truth in zip(pred_np, target_np, strict=True):
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise FloatingPointError('Nonfinite known-mesh vertices')
        forward.append(cKDTree(truth).query(pred, workers=1)[1])
        reverse.append(cKDTree(pred).query(truth, workers=1)[1])
    return (torch.as_tensor(np.stack(forward), device=predicted.device, dtype=torch.long),
            torch.as_tensor(np.stack(reverse), device=predicted.device, dtype=torch.long))


def symmetric_mesh_distance(predicted, target):
    """Per-item symmetric nearest-vertex mean distance, with piecewise gradients."""
    forward, reverse = detached_neighbor_indices(predicted, target)
    target_match = torch.gather(target, 1, forward[..., None].expand(-1, -1, 3))
    pred_match = torch.gather(predicted, 1, reverse[..., None].expand(-1, -1, 3))
    return .5 * ((predicted - target_match).norm(dim=-1).mean(1)
                 + (target - pred_match).norm(dim=-1).mean(1))


def force_target(physics):
    values = []
    for name, size in FORCE_TARGET_FIELDS:
        value = physics[name]
        if value.ndim != 2 or value.shape[1] != size:
            raise ValueError('Unexpected force target shape for ' + name)
        values.append(value)
    return torch.cat(values, dim=1)


def relative_mass_error(state, target):
    return torch.expm1(state[:, 12] - target[:, 12]).abs()


def masked_mass_loss(state, target, available):
    # Select before evaluating the error: unavailable different-mass examples
    # contribute exactly no precision gradient, even for extreme mass labels.
    selected = available > .5
    if bool(selected.any()):
        return relative_mass_error(state[selected], target[selected]).mean() / LOSS_SCALES['mass_relative']
    return state[:, 12].sum() * 0.


def task_losses(output, batch, normalized_sample_vertices):
    state, target = output['state'], batch['target']
    predicted_vertices = state_vertices(state, normalized_sample_vertices)
    target_vertices = state_vertices(target, normalized_sample_vertices)
    return dict(
        center=(state[:, :3] - target[:, :3]).norm(dim=1).mean() / LOSS_SCALES['center_m'],
        mesh=symmetric_mesh_distance(predicted_vertices, target_vertices).mean() / LOSS_SCALES['mesh_m'],
        size=torch.expm1(state[:, 9:12] - target[:, 9:12]).abs().mean() / LOSS_SCALES['size_relative'],
        mass=masked_mass_loss(state, target, batch['mass_available']),
        force=F.smooth_l1_loss(output['force'] / LOSS_SCALES['force_n'],
                               force_target(batch['physics']) / LOSS_SCALES['force_n'], beta=1.),
        availability=F.binary_cross_entropy_with_logits(output['availability_logit'], batch['mass_available']),
        contact=F.binary_cross_entropy_with_logits(output['contact_logit'], batch['physics']['contact_present'][:, 0]),
    )


def total_loss(parts):
    if set(parts) != set(LOSS_WEIGHTS):
        raise ValueError('Every declared loss component is required')
    return sum(LOSS_WEIGHTS[key] * value for key, value in parts.items())


def parameter_groups(model, *, backbone_lr=1e-5, sensor_lr=5e-4, readout_lr=1e-4):
    """All original and added parameters are optimized, without freezing layers."""
    backbone = [p for n, p in model.predictor.backbone.named_parameters() if '.extra.' not in n]
    sensor = list(model.predictor.backbone.embedding.stem.linear.extra.parameters())
    readout = [*model.predictor.head.parameters(), *model.auxiliary.parameters()]
    groups = [dict(params=backbone, lr=backbone_lr, name='full_official_backbone'),
              dict(params=sensor, lr=sensor_lr, name='sensor_affine'),
              dict(params=readout, lr=readout_lr, name='task_readouts')]
    bound = [id(p) for group in groups for p in group['params']]
    if len(bound) != len(set(bound)) or set(bound) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer must bind every complete-model parameter exactly once')
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError('No model parameter may be frozen in this qualification')
    return groups
