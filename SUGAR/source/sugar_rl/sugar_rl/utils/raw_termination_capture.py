"""Capture raw termination-term outputs before IsaacLab collapses their labels.

IsaacLab v2.3.0's :class:`TerminationManager` returns the correct union of all
termination terms, but its diagnostic ``_term_dones`` row keeps only the last
true term.  Reading ``get_term()`` after ``env.step()`` therefore cannot prove
that success and failure were mutually exclusive.

This module is evaluation-only glue.  It wraps the already-resolved term
callables without replacing ``TerminationManager.compute`` or changing any
returned value.  The wrappers copy each raw boolean vector while the manager
performs its real pre-reset computation.  Callers must then use
``snapshot_after_step`` to prove that exactly one complete manager computation
occurred and that the raw union is bitwise equal to the ``dones`` returned by
the environment.
"""

from __future__ import annotations

from typing import Any

import torch


class _CapturingTerminationTerm:
    """Transparent callable wrapper owned by :class:`RawTerminationCapture`."""

    def __init__(
        self,
        capture: RawTerminationCapture,
        index: int,
        name: str,
        original: Any,
    ) -> None:
        self._capture = capture
        self._index = index
        self._name = name
        self._original = original

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        value = self._original(*args, **kwargs)
        self._capture._record_term(self._index, self._name, value)
        return value

    def reset(self, env_ids=None):
        """Preserve reset behavior for class-based manager terms."""

        reset = getattr(self._original, "reset", None)
        if reset is None:
            return None
        return reset(env_ids=env_ids)

    def __getattr__(self, name: str):
        """Delegate optional manager-term attributes such as ``serialize``."""

        return getattr(self._original, name)


class RawTerminationCapture:
    """Fail-closed recorder for one environment's termination manager.

    Install this object after the environment and its managers have been fully
    constructed but before the first scored ``env.step``.  It deliberately
    relies on the configured term order and refuses missing, duplicate, or
    out-of-order calls.
    """

    protocol = "isaaclab_raw_pre_reset_termination_capture_v1"
    _manager_attribute = "_sugar_raw_termination_capture"

    def __init__(self, termination_manager: Any) -> None:
        if hasattr(termination_manager, self._manager_attribute):
            raise RuntimeError("Raw termination capture is already installed")

        term_names = tuple(termination_manager.active_terms)
        term_cfgs = tuple(termination_manager._term_cfgs)
        if not term_names:
            raise RuntimeError("Cannot capture an empty termination manager")
        if len(term_names) != len(term_cfgs):
            raise RuntimeError(
                "Termination manager name/config counts differ: "
                f"{len(term_names)} != {len(term_cfgs)}"
            )
        if len(set(term_names)) != len(term_names):
            raise RuntimeError(f"Termination term names are not unique: {term_names}")

        self.manager = termination_manager
        self.term_names = term_names
        self.num_envs = int(termination_manager.num_envs)
        self.device = termination_manager.device
        self._working = torch.zeros(
            (self.num_envs, len(self.term_names)),
            dtype=torch.bool,
            device=self.device,
        )
        self._latest = torch.zeros_like(self._working)
        self._next_index = 0
        self._completed_compute_count = 0
        self._snapshot_count = 0
        self._union_comparison_count = 0
        self._union_mismatch_count = 0
        self._last_snapshotted_compute_count = 0
        self._installed = True
        self._original_callables_restored = False
        self._original_term_cfgs = term_cfgs
        self._original_class_term_cfgs = tuple(
            getattr(termination_manager, "_class_term_cfgs", ())
        )
        self._original_functions: list[Any] = []
        self._wrappers: list[_CapturingTerminationTerm] = []

        changed_cfgs: list[tuple[Any, Any]] = []
        try:
            for index, (name, term_cfg) in enumerate(
                zip(term_names, term_cfgs, strict=True)
            ):
                original = term_cfg.func
                wrapper = _CapturingTerminationTerm(self, index, name, original)
                self._original_functions.append(original)
                self._wrappers.append(wrapper)
                term_cfg.func = wrapper
                changed_cfgs.append((term_cfg, original))
            setattr(termination_manager, self._manager_attribute, self)
        except BaseException:
            for term_cfg, original in reversed(changed_cfgs):
                term_cfg.func = original
            if getattr(termination_manager, self._manager_attribute, None) is self:
                delattr(termination_manager, self._manager_attribute)
            self._installed = False
            raise

    @property
    def completed_compute_count(self) -> int:
        """Number of complete, in-order manager computations observed."""

        return self._completed_compute_count

    @property
    def snapshot_count(self) -> int:
        """Number of successfully validated, non-duplicate step snapshots."""

        return self._snapshot_count

    @property
    def union_comparison_count(self) -> int:
        """Number of raw-union versus returned-done comparisons attempted."""

        return self._union_comparison_count

    @property
    def union_mismatch_count(self) -> int:
        """Number of failed raw-union comparisons."""

        return self._union_mismatch_count

    @property
    def original_callables_restored(self) -> bool:
        """Whether :meth:`restore` verified and restored every callable."""

        return self._original_callables_restored

    def _validate_manager_identity(self) -> None:
        """Reject manager/name/config/device replacement while capture is active."""

        if tuple(self.manager.active_terms) != self.term_names:
            raise RuntimeError("Termination-manager active term order changed")
        current_cfgs = tuple(self.manager._term_cfgs)
        if len(current_cfgs) != len(self._original_term_cfgs) or any(
            current is not original
            for current, original in zip(
                current_cfgs, self._original_term_cfgs, strict=True
            )
        ):
            raise RuntimeError("Termination-manager config objects changed")
        if int(self.manager.num_envs) != self.num_envs:
            raise RuntimeError("Termination-manager environment count changed")
        if str(self.manager.device) != str(self.device):
            raise RuntimeError("Termination-manager device changed")
        current_class_cfgs = tuple(
            getattr(self.manager, "_class_term_cfgs", ())
        )
        if len(current_class_cfgs) != len(self._original_class_term_cfgs) or any(
            current is not original
            for current, original in zip(
                current_class_cfgs, self._original_class_term_cfgs, strict=True
            )
        ):
            raise RuntimeError("Termination-manager class-term config list changed")
        if self._installed and any(
            term_cfg.func is not wrapper
            for term_cfg, wrapper in zip(
                current_cfgs, self._wrappers, strict=True
            )
        ):
            raise RuntimeError("Termination-manager wrapped callable changed")

    def metadata(self) -> dict[str, object]:
        """Return machine-readable capture counters and restoration state."""

        return {
            "protocol": self.protocol,
            "term_names": list(self.term_names),
            "num_envs": self.num_envs,
            "class_term_count": len(self._original_class_term_cfgs),
            "completed_compute_count": self._completed_compute_count,
            "snapshot_count": self._snapshot_count,
            "union_comparison_count": self._union_comparison_count,
            "union_mismatch_count": self._union_mismatch_count,
            "installed": self._installed,
            "original_callables_restored": self._original_callables_restored,
        }

    def _record_term(self, index: int, name: str, value: Any) -> None:
        if not self._installed:
            raise RuntimeError("Raw termination capture received a call after restore")
        if index != self._next_index:
            raise RuntimeError(
                "Termination terms were not called exactly once in configured order: "
                f"expected index {self._next_index}, got {index} ({name})"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Termination term {name!r} returned {type(value).__name__}, not Tensor"
            )
        if value.dtype != torch.bool or tuple(value.shape) != (self.num_envs,):
            raise RuntimeError(
                f"Termination term {name!r} returned dtype/shape "
                f"{value.dtype}/{tuple(value.shape)}, expected bool/{(self.num_envs,)}"
            )
        if value.device != self._working.device:
            raise RuntimeError(
                f"Termination term {name!r} returned device {value.device}, "
                f"expected {self._working.device}"
            )

        if index == 0:
            self._working.zero_()
        self._working[:, index].copy_(value.detach())
        self._next_index += 1

        if self._next_index == len(self.term_names):
            self._latest.copy_(self._working)
            self._completed_compute_count += 1
            self._next_index = 0

    def snapshot_after_step(
        self,
        dones: torch.Tensor,
        *,
        completed_compute_count_before_step: int,
    ) -> torch.Tensor:
        """Return the raw matrix for one step after validating its exact union.

        Args:
            dones: Net environment ``dones`` returned by the same step.
            completed_compute_count_before_step: Counter read immediately before
                that step.
        """

        if not self._installed:
            raise RuntimeError("Cannot snapshot after raw termination capture restore")
        expected_count = completed_compute_count_before_step + 1
        self._validate_manager_identity()
        if self._completed_compute_count != expected_count:
            raise RuntimeError(
                "Expected exactly one termination-manager computation during env.step: "
                f"before={completed_compute_count_before_step}, "
                f"after={self._completed_compute_count}"
            )
        if self._next_index != 0:
            raise RuntimeError(
                "Termination-manager computation ended with an incomplete raw row: "
                f"next_index={self._next_index}"
            )
        if self._completed_compute_count <= self._last_snapshotted_compute_count:
            raise RuntimeError(
                "The latest termination-manager computation was already snapshotted: "
                f"compute_count={self._completed_compute_count}"
            )
        if not isinstance(dones, torch.Tensor):
            raise TypeError(
                f"Returned dones has type {type(dones).__name__}, expected Tensor"
            )
        if dones.dtype != torch.bool or tuple(dones.shape) != (self.num_envs,):
            raise RuntimeError(
                "Returned dones dtype/shape does not match the official boolean "
                f"vector contract: {dones.dtype}/{tuple(dones.shape)}, expected "
                f"bool/{(self.num_envs,)}"
            )
        if dones.device != self._working.device:
            raise RuntimeError(
                f"Returned dones device {dones.device} does not match capture "
                f"device {self._working.device}"
            )
        raw_union = self._latest.any(dim=1)
        returned_union = dones
        self._union_comparison_count += 1
        if not torch.equal(raw_union, returned_union):
            mismatch_count = int(torch.count_nonzero(raw_union != returned_union).item())
            self._union_mismatch_count += 1
            raise RuntimeError(
                "Raw termination-term union differs from returned dones for "
                f"{mismatch_count} environments"
            )
        self._last_snapshotted_compute_count = self._completed_compute_count
        self._snapshot_count += 1
        return self._latest.detach().clone()

    def restore(self) -> None:
        """Restore the exact original term callables."""

        if not self._installed:
            return
        self._validate_manager_identity()
        if self._next_index != 0:
            raise RuntimeError("Cannot restore during an incomplete termination computation")
        for term_cfg, wrapper, original in zip(
            self.manager._term_cfgs,
            self._wrappers,
            self._original_functions,
            strict=True,
        ):
            if term_cfg.func is not wrapper:
                raise RuntimeError(
                    "Termination callable changed after raw capture installation"
                )
            term_cfg.func = original
        if getattr(self.manager, self._manager_attribute, None) is not self:
            raise RuntimeError("Termination-manager capture marker changed unexpectedly")
        delattr(self.manager, self._manager_attribute)
        self._installed = False
        self._original_callables_restored = True
