"""Shape-only schema gates for official T-Rex data/payload formats.

These helpers validate keys and tensor shapes before Newton/Taccel samples are
passed into official T-Rex loaders or inference scripts. They do not implement
T-Rex, a tactile encoder, a VQ-VAE, a policy, or a learned world model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from newton_tactile_curiosity.data_schemas.trex_contract import (
    DEFAULT_TREX_CONTRACT,
    TRexTensorContract,
    SchemaValidationError,
    validate_action_chunk_shape,
    validate_state_shape,
    validate_tactile_f6_shape,
)


@dataclass(frozen=True)
class ImageShape:
    channels: int = 3
    height: int | None = None
    width: int | None = None


@dataclass(frozen=True)
class DeformVideoShape:
    channels: int = 3
    height: int | None = None
    width: int | None = None


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), *_shape(value[0]))
        return (len(value),)
    raise SchemaValidationError(f"object has no shape: {type(value)!r}")


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SchemaValidationError(f"missing required keys: {missing}")


def _validate_chw(name: str, value: Any, spec: ImageShape | DeformVideoShape) -> None:
    observed = _shape(value)
    if len(observed) != 3:
        raise SchemaValidationError(f"{name} must be CHW rank 3, got {observed}")
    channels, height, width = observed
    if channels != spec.channels:
        raise SchemaValidationError(f"{name} channels {channels} != expected {spec.channels}")
    if spec.height is not None and height != spec.height:
        raise SchemaValidationError(f"{name} height {height} != expected {spec.height}")
    if spec.width is not None and width != spec.width:
        raise SchemaValidationError(f"{name} width {width} != expected {spec.width}")


def _validate_head_sequence(name: str, value: Any, spec: ImageShape) -> None:
    observed = _shape(value)
    if len(observed) != 4:
        raise SchemaValidationError(f"{name} must be TCHW rank 4, got {observed}")
    frames, channels, height, width = observed
    if frames < 1:
        raise SchemaValidationError(f"{name} must contain at least one frame")
    if channels != spec.channels:
        raise SchemaValidationError(f"{name} channels {channels} != expected {spec.channels}")
    if spec.height is not None and height != spec.height:
        raise SchemaValidationError(f"{name} height {height} != expected {spec.height}")
    if spec.width is not None and width != spec.width:
        raise SchemaValidationError(f"{name} width {width} != expected {spec.width}")


def validate_lerobot_frame(
    frame: Mapping[str, Any],
    *,
    head_shape: ImageShape = ImageShape(),
    wrist_shape: ImageShape = ImageShape(),
    deform_shape: DeformVideoShape = DeformVideoShape(),
    include_wrist: bool = True,
    include_tactile: bool = True,
    include_action_abs: bool = True,
    contract: TRexTensorContract = DEFAULT_TREX_CONTRACT,
) -> None:
    """Validate one frame against official T-Rex LeRobot key/shape expectations.

    The head image key is a temporal sequence because official T-Rex loaders use
    the current head frame plus future FLARE frames. Wrist and deform entries are
    per-frame CHW tensors. Deform maps follow the official LeRobot storage form:
    ten per-finger video keys, left fingers 0-4 then right fingers 0-4.
    """

    required = (
        contract.slow_camera_key,
        contract.state_key,
        contract.action_key,
    )
    if include_action_abs:
        required += (contract.action_abs_key,)
    if include_wrist:
        required += (contract.wrist_right_camera_key, contract.wrist_left_camera_key)
    if include_tactile:
        required += (contract.tactile_f6_key, *contract.tactile_deform_prefixes)
    _require_keys(frame, required)

    _validate_head_sequence(contract.slow_camera_key, frame[contract.slow_camera_key], head_shape)
    validate_state_shape(frame[contract.state_key], contract)
    validate_action_chunk_shape(frame[contract.action_key], contract)
    if include_action_abs:
        validate_state_shape(frame[contract.action_abs_key], contract)
    if include_wrist:
        _validate_chw(contract.wrist_right_camera_key, frame[contract.wrist_right_camera_key], wrist_shape)
        _validate_chw(contract.wrist_left_camera_key, frame[contract.wrist_left_camera_key], wrist_shape)
    if include_tactile:
        validate_tactile_f6_shape(frame[contract.tactile_f6_key], contract)
        for key in contract.tactile_deform_prefixes:
            _validate_chw(key, frame[key], deform_shape)


def validate_trex_inference_payload(
    payload: Mapping[str, Any],
    *,
    mode: str = "slow_and_fast",
    require_wrist_left: bool = True,
    require_robot_state: bool = True,
    require_tactile_f6: bool = True,
    require_tactile_deform: bool = True,
    contract: TRexTensorContract = DEFAULT_TREX_CONTRACT,
) -> None:
    """Validate a payload for official `external/T-Rex/scripts/test.py`.

    Image values are intentionally checked only for key presence because the
    official server expects encoded image bytes and decodes them with PIL. This
    function verifies non-image numeric shapes: state `[62]`, tactile F6
    `[10, 6]` or history `[T, 10, 6]`, and tactile deform `[10, H, W]` or
    `[10, 1, H, W]`.
    """

    if mode not in {"slow", "fast", "slow_and_fast"}:
        raise SchemaValidationError(f"unknown T-Rex mode: {mode}")

    if mode in {"slow", "slow_and_fast"}:
        _require_keys(payload, ("image_head", "image_wrist_right"))
        if require_wrist_left:
            _require_keys(payload, ("image_wrist_left",))

    if require_robot_state and mode in {"slow", "slow_and_fast"}:
        _require_keys(payload, ("state_fast",))
        validate_state_shape(payload["state_fast"], contract)

    if require_tactile_f6:
        _require_keys(payload, ("tactile_f6",))
        observed = _shape(payload["tactile_f6"])
        if observed == (contract.tactile_fingers, contract.tactile_channels):
            return_f6_ok = True
        elif (
            len(observed) == 3
            and observed[1] == contract.tactile_fingers
            and observed[2] == contract.tactile_channels
        ):
            return_f6_ok = True
        else:
            return_f6_ok = False
        if not return_f6_ok:
            raise SchemaValidationError(
                "tactile_f6 must be [10, 6] or [T, 10, 6], "
                f"got {observed}"
            )

    if require_tactile_deform:
        if "tactile_deform" in payload:
            deform = payload["tactile_deform"]
        elif "tactile_image_deform" in payload:
            deform = payload["tactile_image_deform"]
        else:
            raise SchemaValidationError("missing required deform key: tactile_deform or tactile_image_deform")
        observed = _shape(deform)
        valid_rank3 = len(observed) == 3 and observed[0] == contract.tactile_fingers
        valid_rank4 = (
            len(observed) == 4
            and observed[0] == contract.tactile_fingers
            and observed[1] == 1
        )
        if not (valid_rank3 or valid_rank4):
            raise SchemaValidationError(
                "tactile_deform must be [10, H, W] or [10, 1, H, W], "
                f"got {observed}"
            )
