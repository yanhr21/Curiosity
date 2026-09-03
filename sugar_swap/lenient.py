"""Config classes that accept IsaacLab's USD/PhysX properties and record what Newton ignores.

SUGAR's environment configs are written against Isaac Sim's spawners, so they set properties
that describe how to author a USD prim -- `ArticulationRootPropertiesCfg`, rigid-body sleep
thresholds, per-shape contact offsets, and so on. Newton builds its model from the URDF and
its own solver settings instead, so most of those fields have no destination.

Rather than fork the configs to delete those fields, the shadow classes here accept anything
and remember it. Two consequences worth being explicit about:

* Fields Newton *does* consume are declared explicitly on the subclasses in `shadows.py` and
  read by the scene builder.
* Everything else is retained in `ignored_fields` and reported by `report_ignored()`. This is
  the honest accounting of where the swap is not faithful: a solver-iteration count or a
  contact offset that Isaac Sim would have honoured and Newton silently will not.

Reporting is the point. An earlier version of this port set physics properties that were
never read, and the resulting fidelity gap was only found by benchmark archaeology.
"""

from __future__ import annotations

import copy
from typing import Any

# Populated as configs are instantiated; keyed by "ClassName.field".
_IGNORED: dict[str, Any] = {}


class LenientCfg:
    """Base for shadowed IsaacLab config classes.

    Declared class attributes are treated as consumed by the Newton backend. Any other
    keyword is stored on the instance and recorded globally as ignored.
    """

    def __init__(self, **kwargs: Any):
        declared = self._declared_fields()
        for key, value in kwargs.items():
            setattr(self, key, value)
            if key not in declared:
                _IGNORED[f"{type(self).__name__}.{key}"] = value

    @classmethod
    def _declared_fields(cls) -> set[str]:
        fields: set[str] = set()
        for klass in cls.__mro__:
            if klass in (LenientCfg, object):
                continue
            fields.update(k for k in vars(klass) if not k.startswith("_"))
            fields.update(getattr(klass, "__annotations__", {}).keys())
        return fields

    def replace(self, **kwargs: Any) -> LenientCfg:
        """IsaacLab configs are copied with `.replace(...)`; SUGAR's cfgs rely on it."""
        out = self.copy()
        for key, value in kwargs.items():
            setattr(out, key, value)
        return out

    def copy(self) -> LenientCfg:
        """The other half of IsaacLab's `configclass` copy API.

        SUGAR's `MotionCommand` does `FRAME_MARKER_CFG.copy()` and then mutates the result's
        nested marker entries, so this has to be deep -- a shallow copy would write through
        to the module-level preset and leak between command terms.
        """
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_dict()})"


def ignored_fields() -> dict[str, Any]:
    """Every config field the Newton backend did not consume."""
    return dict(_IGNORED)


def report_ignored() -> str:
    """Human-readable summary of the unfaithful surface, for logging at startup."""
    if not _IGNORED:
        return "sugar_swap: no ignored config fields"
    lines = [f"sugar_swap: {len(_IGNORED)} IsaacLab config fields not consumed by Newton:"]
    lines.extend(f"    {key} = {value!r}" for key, value in sorted(_IGNORED.items()))
    return "\n".join(lines)
