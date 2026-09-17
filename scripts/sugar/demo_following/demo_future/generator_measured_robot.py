"""Zero-column input adaptation of the existing official robot affine layer.

No hidden layers, tokens, denoiser blocks or normalizer statistics are added.
The split affine evaluation preserves the original36-column numerical path.
"""
import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_KEY='obs_encoder.robot_state_net.0.weight'
EXTRA_KEY='obs_encoder.robot_state_net.0.measured_weight'


class PartitionedRobotAffine(nn.Linear):
    """Same affine map with two leaf parameters, preserving AMP accumulation."""
    def forward(self, inputs):
        if inputs.shape[-1]!=68 or self.weight.shape!=(256,36) or self.measured_weight.shape!=(256,32):
            raise RuntimeError('Require original36 and added32 affine parameter blocks')
        return F.linear(inputs[...,:36].contiguous(),self.weight,self.bias)+F.linear(inputs[...,36:].contiguous(),self.measured_weight,None)


class PreservedRobotAffine(nn.Linear):
    def forward(self, inputs):
        if inputs.shape[-1]!=68 or self.weight.shape!=(256,68):
            raise RuntimeError('Require the declared36command+32measured robot input')
        original=F.linear(inputs[...,:36].contiguous(),self.weight[:,:36].contiguous(),self.bias)
        measured=F.linear(inputs[...,36:].contiguous(),self.weight[:,36:].contiguous(),None)
        return original+measured


def install_measured_robot_inputs(encoder, parameter_layout='wide68'):
    if not encoder.use_last_action or getattr(encoder,'use_measured_robot_state',False):
        raise RuntimeError('Extend the original previous-command encoder exactly once')
    original=encoder.robot_state_net[0]
    if type(original) is not nn.Linear or original.weight.shape!=(256,36):
        raise RuntimeError('Require the complete official36to256 robot affine layer')
    if parameter_layout=='partitioned_leaf':
        original.register_parameter('measured_weight',nn.Parameter(torch.zeros(256,32,device=original.weight.device,dtype=original.weight.dtype),requires_grad=original.weight.requires_grad))
        original.__class__=PartitionedRobotAffine
        original.in_features=68
        encoder.use_measured_robot_state=True
        return encoder
    if parameter_layout!='wide68':
        raise ValueError('Unknown robot-affine parameter layout')
    device=original.weight.device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        extended=PreservedRobotAffine(68,256,bias=original.bias is not None,device=device,dtype=original.weight.dtype)
    with torch.no_grad():
        extended.weight.zero_();extended.weight[:,:36].copy_(original.weight)
        if original.bias is not None:extended.bias.copy_(original.bias)
    extended.train(original.training)
    extended.weight.requires_grad_(original.weight.requires_grad)
    if original.bias is not None:extended.bias.requires_grad_(original.bias.requires_grad)
    encoder.robot_state_net[0]=extended
    encoder.use_measured_robot_state=True
    return encoder


def extend_robot_state_dict(state):
    if state[WEIGHT_KEY].shape!=(256,36):
        raise RuntimeError('Only extend an immutable original36-column checkpoint')
    result=dict(state)
    if EXTRA_KEY in state:raise RuntimeError('Measured input already exists in source state')
    result[EXTRA_KEY]=state[WEIGHT_KEY].new_zeros((256,32))
    return result


def original_robot_slice(name,value):
    return value[:,:36] if name==WEIGHT_KEY else value
