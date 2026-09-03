"""IsaacLab's event terms, lifted verbatim out of a module we cannot import.

`isaaclab/envs/mdp/events.py` is the one file in IsaacLab's term library with real Isaac Sim
coupling: it imports `carb`, `omni.physics.tensors`, `isaacsim.core.utils` and `pxr` at module
scope, for terms that author USD or drive the replicator. SUGAR needs four terms out of that
file, and those four turn out to touch the simulator only through `asset.root_physx_view` --
which `sugar_swap.physx_view` implements on Newton.

So instead of importing the module (impossible) or transcribing the four terms (which would
drift from IsaacLab and quietly change the randomisation distributions), this module parses
the file and executes just those definitions. The bodies are IsaacLab's, byte for byte;
only the objects they act on are Newton's.

The trade-off is a hard dependency on those terms staying free of Isaac Sim coupling. That is
checked rather than assumed: `_FORBIDDEN` is scanned for in each extracted definition and a
match raises at import, so an IsaacLab upgrade that adds a USD call here fails loudly instead
of silently doing nothing.
"""

from __future__ import annotations

import ast
import pathlib
import types
from typing import Any

# Terms SUGAR references, plus the helper they share.
_WANTED = (
    "_randomize_prop_by_op",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_com",
    "randomize_rigid_body_material",
    "push_by_setting_velocity",
)

# Any of these in an extracted body means the term reaches Isaac Sim directly and the
# extraction is no longer sound.
_FORBIDDEN = ("physx.", "carb.", "pxr.", "isaacsim.", "get_current_stage", "enable_extension")


def _events_source_path() -> pathlib.Path:
    """Locate IsaacLab's events.py without importing anything from IsaacLab."""
    here = pathlib.Path(__file__).resolve().parent.parent
    candidate = (
        here / "IsaacLab" / "source" / "isaaclab" / "isaaclab" / "envs" / "mdp" / "events.py"
    )
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"sugar_swap: cannot find IsaacLab's events.py (looked at {candidate}). The verbatim "
        "event terms are extracted from it, so the IsaacLab checkout is required."
    )


def _build_namespace() -> dict[str, Any]:
    """Globals for the extracted terms: torch plus the shadowed IsaacLab objects."""
    import math

    import torch

    from isaaclab.managers import ManagerTermBase, SceneEntityCfg

    from .assets import Articulation, RigidObject, RigidObjectCollection

    return {
        "__name__": "sugar_swap.events",
        "annotations": None,
        "torch": torch,
        "math": math,
        "Articulation": Articulation,
        "RigidObject": RigidObject,
        "RigidObjectCollection": RigidObjectCollection,
        "SceneEntityCfg": SceneEntityCfg,
        "ManagerTermBase": ManagerTermBase,
    }


def _extract() -> dict[str, Any]:
    path = _events_source_path()
    source = path.read_text()
    tree = ast.parse(source)

    wanted = {name: None for name in _WANTED}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            segment = ast.get_source_segment(source, node) or ""
            for bad in _FORBIDDEN:
                if bad in segment:
                    raise RuntimeError(
                        f"sugar_swap: IsaacLab's `{node.name}` now references `{bad}`, so it can "
                        "no longer be reused verbatim on the Newton backend. Port it explicitly."
                    )
            wanted[node.name] = node

    missing = [name for name, node in wanted.items() if node is None]
    if missing:
        raise RuntimeError(f"sugar_swap: not found in IsaacLab's events.py: {missing}")

    namespace = _build_namespace()
    module = ast.Module(body=[wanted[name] for name in _WANTED], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def build() -> dict[str, types.ModuleType]:
    """Construct the `isaaclab.envs.mdp.events` shadow, exporting the verbatim terms."""
    namespace = _extract()
    mod = types.ModuleType("isaaclab.envs.mdp.events")
    for name in _WANTED:
        setattr(mod, name, namespace[name])
    # `from .events import *` must not re-export our scaffolding.
    mod.__all__ = [name for name in _WANTED if not name.startswith("_")]
    return {"isaaclab.envs.mdp.events": mod}
