"""Official T-Rex tensor shape contract used by project glue code.

This module only defines shapes and validation helpers. It does not implement
T-Rex, a tactile encoder, a VQ-VAE, a policy, or a world model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when an observation/action tensor shape violates the contract."""


@dataclass(frozen=True)
class TactileDeformShape:
    fingers: int = 10
    channels: int = 1
    height: int | None = None
    width: int | None = None

    def as_json(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class TRexTensorContract:
    """Shape-only contract extracted from official T-Rex loader expectations."""

    action_dim: int = 62
    action_chunk: int = 16
    tactile_fingers: int = 10
    tactile_channels: int = 6
    slow_camera_key: str = "observation.images.head"
    wrist_right_camera_key: str = "observation.images.wrist_right"
    wrist_left_camera_key: str = "observation.images.wrist_left"
    state_key: str = "observation.state"
    action_key: str = "action"
    action_abs_key: str = "action_abs"
    tactile_f6_key: str = "observation.tactile_f6"
    tactile_deform_prefixes: tuple[str, ...] = (
        "observation.tactile_deform.l0",
        "observation.tactile_deform.l1",
        "observation.tactile_deform.l2",
        "observation.tactile_deform.l3",
        "observation.tactile_deform.l4",
        "observation.tactile_deform.r0",
        "observation.tactile_deform.r1",
        "observation.tactile_deform.r2",
        "observation.tactile_deform.r3",
        "observation.tactile_deform.r4",
    )

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["tactile_f6_shape"] = [self.tactile_fingers, self.tactile_channels]
        data["action_shape"] = [self.action_chunk, self.action_dim]
        data["state_shape"] = [self.action_dim]
        return data


DEFAULT_TREX_CONTRACT = TRexTensorContract()


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], (list, tuple)):
                inner = _shape(value[0])
                return (len(value), *inner)
            return (len(value),)
        raise SchemaValidationError(f"object has no shape: {type(value)!r}")
    return tuple(int(dim) for dim in shape)


def validate_state_shape(state: Any, contract: TRexTensorContract = DEFAULT_TREX_CONTRACT) -> None:
    observed = _shape(state)
    expected = (contract.action_dim,)
    if observed != expected:
        raise SchemaValidationError(f"state shape {observed} != expected {expected}")


def validate_action_chunk_shape(
    action: Any, contract: TRexTensorContract = DEFAULT_TREX_CONTRACT
) -> None:
    observed = _shape(action)
    expected = (contract.action_chunk, contract.action_dim)
    if observed != expected:
        raise SchemaValidationError(f"action chunk shape {observed} != expected {expected}")


def validate_tactile_f6_shape(
    tactile_f6: Any, contract: TRexTensorContract = DEFAULT_TREX_CONTRACT
) -> None:
    observed = _shape(tactile_f6)
    expected = (contract.tactile_fingers, contract.tactile_channels)
    if observed != expected:
        raise SchemaValidationError(f"tactile_f6 shape {observed} != expected {expected}")


def validate_tactile_deform_shape(
    tactile_deform: Any,
    deform_shape: TactileDeformShape,
) -> None:
    observed = _shape(tactile_deform)
    if len(observed) != 4:
        raise SchemaValidationError(f"tactile_deform must be rank 4, got {observed}")
    fingers, channels, height, width = observed
    if fingers != deform_shape.fingers or channels != deform_shape.channels:
        expected = (deform_shape.fingers, deform_shape.channels, deform_shape.height, deform_shape.width)
        raise SchemaValidationError(f"tactile_deform shape {observed} != expected prefix {expected}")
    if deform_shape.height is not None and height != deform_shape.height:
        raise SchemaValidationError(f"tactile_deform height {height} != expected {deform_shape.height}")
    if deform_shape.width is not None and width != deform_shape.width:
        raise SchemaValidationError(f"tactile_deform width {width} != expected {deform_shape.width}")
