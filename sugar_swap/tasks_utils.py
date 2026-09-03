"""`isaaclab_tasks.utils`, loaded verbatim from source without importing its package.

`sugar_rl.tasks.__init__` calls `isaaclab_tasks.utils.import_packages` to auto-discover its
task modules, and SUGAR's scripts use `load_cfg_from_registry` to fetch an agent config from
the gym registry. Both helpers are plain Python -- importlib, pkgutil, gymnasium, yaml -- but
they live in a package whose `__init__` boots Isaac Sim.

Loading the two source files directly by path sidesteps that `__init__` while keeping the
helpers byte-identical, which matters because `import_packages` decides which task modules
get registered and a reimplementation could quietly register a different set.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


def _isaaclab_tasks_root() -> pathlib.Path:
    root = (
        pathlib.Path(__file__).resolve().parent.parent
        / "IsaacLab" / "source" / "isaaclab_tasks" / "isaaclab_tasks"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"sugar_swap: expected IsaacLab tasks package at {root}")
    return root


def _load_by_path(module_name: str, path: pathlib.Path) -> types.ModuleType:
    """Execute a single source file as `module_name`, bypassing its package `__init__`."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"sugar_swap: cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build() -> dict[str, types.ModuleType]:
    """Construct the `isaaclab_tasks` shadow with a working `utils` subpackage."""
    root = _isaaclab_tasks_root()

    # A bare package object, so the real __init__ (which boots Isaac Sim) never runs.
    pkg = types.ModuleType("isaaclab_tasks")
    pkg.__path__ = []
    utils = types.ModuleType("isaaclab_tasks.utils")
    utils.__path__ = []
    sys.modules["isaaclab_tasks"] = pkg
    sys.modules["isaaclab_tasks.utils"] = utils

    importer = _load_by_path("isaaclab_tasks.utils.importer", root / "utils" / "importer.py")
    utils.importer = importer
    utils.import_packages = importer.import_packages

    modules = {
        "isaaclab_tasks": pkg,
        "isaaclab_tasks.utils": utils,
        "isaaclab_tasks.utils.importer": importer,
    }

    # parse_cfg pulls in gymnasium and isaaclab.envs (already substituted). It is optional:
    # only the play/train scripts need it, so a failure here should not block training.
    try:
        parse_cfg = _load_by_path("isaaclab_tasks.utils.parse_cfg", root / "utils" / "parse_cfg.py")
    except Exception as exc:  # noqa: BLE001
        utils._parse_cfg_error = exc
    else:
        utils.parse_cfg = parse_cfg
        for name in ("load_cfg_from_registry", "parse_env_cfg", "get_checkpoint_path"):
            if hasattr(parse_cfg, name):
                setattr(utils, name, getattr(parse_cfg, name))
        modules["isaaclab_tasks.utils.parse_cfg"] = parse_cfg

    return modules
