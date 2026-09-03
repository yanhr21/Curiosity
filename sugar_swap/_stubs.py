"""Minimal stand-ins for the Omniverse modules IsaacLab imports but barely uses.

IsaacLab's backend-agnostic code -- `utils.math`, the managers, the config dataclasses -- is
pure torch/Python except for a handful of Omniverse touch points:

* `omni.log.info/warn/error`, used only for logging.
* `omni.kit.app.get_app_interface()`, used only to subscribe debug-visualisation callbacks
  to the app's post-update event stream.
* `omni.timeline.get_timeline_interface()`, used only to defer scene-entity resolution to a
  timeline PLAY event. Our `SimulationContext.is_playing()` returns True, so
  `ManagerBase.__init__` resolves terms directly and never reaches this path.

Registering these in `sys.modules` lets that code be reused verbatim instead of forked. Any
attribute we did not anticipate raises rather than silently returning a mock, so a real
dependency on Omniverse surfaces as an error instead of wrong physics.
"""

from __future__ import annotations

import sys
import types


class _Subscription:
    """Handle returned by the event-stream subscribe calls; only unsubscribe() is used."""

    def unsubscribe(self) -> None:
        pass


class _EventStream:
    def create_subscription_to_pop(self, *_args, **_kwargs) -> _Subscription:
        return _Subscription()

    def create_subscription_to_pop_by_type(self, *_args, **_kwargs) -> _Subscription:
        return _Subscription()


class _AppInterface:
    def get_post_update_event_stream(self) -> _EventStream:
        return _EventStream()


class _TimelineInterface:
    def get_timeline_event_stream(self) -> _EventStream:
        return _EventStream()


class _TimelineEventType:
    PLAY = 0
    PAUSE = 1
    STOP = 2


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def install() -> None:
    """Register the stub modules. Idempotent, and a no-op if the real ones are present."""
    if "omni.log" in sys.modules and not getattr(sys.modules["omni.log"], "_is_sugar_swap_stub", False):
        # A real Omniverse is loaded (we are inside Isaac Sim); leave it alone.
        return

    def _log(*_args, **_kwargs) -> None:
        pass

    omni_log = _make_module(
        "omni.log", info=_log, warn=_log, error=_log, verbose=_log, _is_sugar_swap_stub=True
    )
    omni_kit_app = _make_module("omni.kit.app", get_app_interface=_AppInterface)
    omni_kit = _make_module("omni.kit", app=omni_kit_app)
    omni_timeline = _make_module(
        "omni.timeline",
        get_timeline_interface=_TimelineInterface,
        TimelineEventType=_TimelineEventType,
    )
    omni = _make_module("omni", log=omni_log, kit=omni_kit, timeline=omni_timeline)

    # carb is imported for settings access in code paths we do not enter.
    carb = _make_module("carb", settings=_make_module("carb.settings"))

    sys.modules.update(
        {
            "omni": omni,
            "omni.log": omni_log,
            "omni.kit": omni_kit,
            "omni.kit.app": omni_kit_app,
            "omni.timeline": omni_timeline,
            "carb": carb,
            "carb.settings": carb.settings,
        }
    )
    _install_pxr()


def _install_pxr() -> None:
    """Stub `pxr`, but only if USD is genuinely unavailable.

    IsaacLab's task-space and surface-gripper action terms import `pxr` at module scope.
    SUGAR uses neither, but `isaaclab.envs.mdp.actions` imports the whole package, so the
    name has to resolve for the terms we do want.

    The Newton environment, however, ships a real `usd-core`, and Newton's own USD importer
    does `from pxr import Usd` lazily at call time. Registering the stub unconditionally
    would shadow a working USD and turn any asset load into a confusing attribute error far
    from here -- SUGAR's carried box is a `.usd` file, so that path is live. Defer to the
    real package whenever it imports.
    """
    try:
        import pxr  # noqa: F401

        return
    except ImportError:
        pass

    pxr = _make_module(
        "pxr",
        UsdPhysics=_make_module("pxr.UsdPhysics"),
        Gf=_make_module("pxr.Gf"),
        Sdf=_make_module("pxr.Sdf"),
        UsdGeom=_make_module("pxr.UsdGeom"),
        Vt=_make_module("pxr.Vt"),
        Usd=_make_module("pxr.Usd"),
    )
    sys.modules.update(
        {
            "pxr": pxr,
            "pxr.UsdPhysics": pxr.UsdPhysics,
            "pxr.Gf": pxr.Gf,
            "pxr.Sdf": pxr.Sdf,
            "pxr.UsdGeom": pxr.UsdGeom,
            "pxr.Vt": pxr.Vt,
            "pxr.Usd": pxr.Usd,
        }
    )
