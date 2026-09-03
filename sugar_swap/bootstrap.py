"""Point IsaacLab's import graph at Newton. Must run before importing anything from SUGAR.

The swap is done by substitution rather than modification. IsaacLab's package is split into
two kinds of module:

* Backend-agnostic -- `utils.math`, `utils.string`, the managers, the MDP term library. These
  are pure torch and are imported from the real IsaacLab, unmodified.
* Isaac Sim-specific -- `assets`, `sensors`, `scene`, `sim`, `envs`, `markers`, `terrains`,
  `actuators`. These are replaced in `sys.modules` by the Newton-backed equivalents in this
  package before anything imports them.

Because Python resolves `from isaaclab.assets import Articulation` through `sys.modules`
first, registering the substitutes up front means IsaacLab's own managers and term functions
transparently operate on Newton objects, and SUGAR needs no edits at all.

Ordering is load-bearing and enforced by `install()`:

1. Stub the Omniverse modules that only provide logging and event streams.
2. Register the Newton-backed replacements.
3. Register the verbatim-extracted event terms as `isaaclab.envs.mdp.events`, which must
   happen before step 4 imports that package and triggers `from .events import *`.
4. Import the real `isaaclab.envs.mdp`, which now binds against everything above.
"""

from __future__ import annotations

import sys

_INSTALLED = False


def install(verbose: bool = False) -> None:
    """Substitute the Newton backend into IsaacLab's import graph. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    if "isaaclab.assets" in sys.modules:
        raise RuntimeError(
            "sugar_swap: isaaclab.assets is already imported, so the Newton substitutes cannot "
            "take effect. Call sugar_swap.bootstrap.install() before importing IsaacLab or SUGAR."
        )

    from . import _stubs

    _stubs.install()

    from . import assets, env, events, scene, sensors, shadows, tasks_utils

    replacements: dict[str, object] = {}
    replacements.update(shadows.build())
    replacements.update(assets.build())
    replacements.update(sensors.build())
    replacements.update(scene.build())
    replacements.update(env.build())
    sys.modules.update(replacements)

    # `sugar_rl.tasks` discovers its task modules through this helper, whose real package
    # would boot Isaac Sim on import.
    sys.modules.update(tasks_utils.build())

    # Extracting the event terms imports isaaclab.managers, which needs the substitutes above
    # to already be registered, so this cannot be folded into the batch.
    sys.modules.update(events.build())

    # Bind the substitutes as attributes too, so `import isaaclab; isaaclab.assets...` works.
    import isaaclab

    for name, module in {**replacements}.items():
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "isaaclab":
            setattr(isaaclab, parts[1], module)

    # Force the real term library to bind now, while the substitutes are guaranteed in place.
    import isaaclab.envs.mdp  # noqa: F401

    _INSTALLED = True

    if verbose:
        from .lenient import report_ignored

        print(report_ignored(), flush=True)


def installed() -> bool:
    return _INSTALLED
