"""Causal command-buffer and tensor-routing glue, without a model or physics.

The caller supplies full official Generator output and actual Tracker state.
The48-D position packet remains explicitly known reference geometry. It is not
generated, and this adapter does not establish generated physical execution.
"""
import torch


class OriginalNumericPositionFeedback:
    """Known original-demo positions, using the unchanged official transforms.

    Native source clocks only. This removes the measured-Refiner position-bank
    dependency, but positions remain a declared original-demo input, not model
    predictions. It creates no trajectory and changes no physical world state.
    """

    def __init__(self, env, source_ids):
        import pickle
        from pathlib import Path
        import numpy as np
        from scripts.sugar.smp.sugar_g1_box_schema import SOURCE_BODY_NAMES
        root = Path(__file__).resolve().parents[4]
        command = env.command_manager.get_term('motion')
        if command.cfg.anchor_body_name != 'torso_link' or command.motion.num_motion != len(source_ids):
            raise RuntimeError('Original position source mapping or torso anchor differs')
        index = SOURCE_BODY_NAMES.index('torso_link')
        anchors, boxes = [], []
        for source in source_ids:
            folder = root / f'SUGAR/data/CarryBox/data_{source:03d}'
            with np.load(folder / 'robot_50hz.npz') as robot:
                anchors.append(torch.as_tensor(robot['body_pos_w'][:, index].copy(), device=env.device))
            with (folder / 'obj_motion_global_50hz.pkl').open('rb') as stream:
                boxes.append(torch.as_tensor(pickle.load(stream)['obj_trans'], device=env.device))
        lengths = [min(len(a), len(b)) for a, b in zip(anchors, boxes)]
        self.lengths = torch.tensor(lengths, device=env.device, dtype=torch.long)
        self.anchors = torch.zeros(len(source_ids), max(lengths), 3, device=env.device)
        self.boxes = torch.zeros_like(self.anchors)
        for i, count in enumerate(lengths):
            self.anchors[i, :count] = anchors[i][:count]
            self.boxes[i, :count] = boxes[i][:count]

    def __call__(self, env):
        from isaaclab.utils.math import quat_apply_inverse
        command = env.command_manager.get_term('motion')
        ids = command.motion_id[:, None]
        times = command.time_steps[:, None] + torch.arange(8, device=env.device)[None]
        if bool((times < 0).any() or (times >= self.lengths[command.motion_id, None]).any()):
            raise RuntimeError('Original position plan exceeds real source frames; no padding or extrapolation')
        origin = env.scene.env_origins[:, None]
        anchors = self.anchors[ids, times] + origin
        boxes = self.boxes[ids, times] + origin
        a = quat_apply_inverse(command.robot_anchor_quat_w[:, None].expand(-1, 8, -1),
                               anchors - command.robot_anchor_pos_w[:, None])
        b = quat_apply_inverse(command.obj_quat_w[:, None].expand(-1, 8, -1),
                               boxes - command.obj_pos_w[:, None])
        return torch.cat([a.flatten(1), b.flatten(1)], dim=1)


class IssuedCommandHistory:
    """Keep five actually issued36-D commands for the official10-Hz input lag.

    At handoff frame t, past_commands must be the already recorded commands at
    t-5..t-1. Subsequent updates contain commands actually supplied to Tracker,
    never labels read from a future reference or the29-D motor-action history.
    """

    def __init__(self, first_frame, past_commands):
        if past_commands.ndim != 3 or past_commands.shape[1:] != (5, 36):
            raise ValueError('Require five past issued full commands, shape Bx5x36')
        if first_frame < 5 or not torch.isfinite(past_commands).all():
            raise ValueError('A real five-frame prefix is required before generator handoff')
        self.next_frame = int(first_frame)
        self._commands = past_commands.detach().clone()

    def previous_10hz_command(self, control_frame):
        if int(control_frame) != self.next_frame:
            raise RuntimeError('Command history clock must match the current control call')
        return self._commands[:, :1].clone()

    def record_issued(self, control_frame, command):
        if int(control_frame) != self.next_frame or command.shape != self._commands[:, 0].shape:
            raise RuntimeError('Issued command frame or full36-D shape differs')
        if command.device != self._commands.device or command.dtype != self._commands.dtype or not torch.isfinite(command).all():
            raise RuntimeError('Issued command device, dtype or numeric state differs')
        self._commands = torch.cat([self._commands[:, 1:], command.detach()[:, None]], dim=1)
        self.next_frame += 1


def compose_generated_tracker_input(actual_observation, dense_plan, known_reference_positions,
                                    *, plan_frame, control_frame):
    """Route a newly generated complete plan into the full846-D actor input.

    Uses the official wrapper's36-frame interpolated output. Replanning must
    occur at this very control call: an aged8-point plan lacks the35-frame
    lookahead. The original numeric36-command prefix is completely replaced;
    all474 actual-state/history entries are preserved. Reference positions are
    a separately declared numeric-demo input and are never called generated.
    """
    if int(plan_frame) != int(control_frame):
        raise RuntimeError('Aged plan lacks the complete35-frame lookahead; generate a new full plan')
    batch = actual_observation.shape[0]
    if actual_observation.shape != (batch, 510) or dense_plan.shape != (batch, 36, 36) or known_reference_positions.shape != (batch, 48):
        raise ValueError('Require complete official510-state,36x36 dense command plan and48 known positions')
    for value in (dense_plan, known_reference_positions):
        if value.device != actual_observation.device or value.dtype != actual_observation.dtype:
            raise ValueError('All actor input tensors must have matching device and dtype')
    if not all(torch.isfinite(value).all() for value in (actual_observation, dense_plan, known_reference_positions)):
        raise RuntimeError('Nonfinite full actor input')
    current = torch.cat([dense_plan[:, 0], actual_observation[:, 36:]], dim=1)
    future = dense_plan[:, ::5].flatten(1)
    return torch.cat([current, future, known_reference_positions], dim=1)
