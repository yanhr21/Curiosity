# SPDX-License-Identifier: BSD-3-Clause
"""Per-run identity that has to outlive a single SLURM allocation.

A training run is not one process. It is a chain of 4 h legs, each resuming from the highest
``model_*.pt`` in the run directory (``slurm/train_leg.sh``), so a 3-day run is ~18 processes.
rsl_rl's ``WandbSummaryWriter`` calls ``wandb.init(project=..., entity=..., name=...)`` with
no id and no resume (``rsl_rl/utils/wandb_utils.py:38``), and wandb mints a fresh run for
every such call -- 18 disconnected curves, none of which shows the run.

So the run directory, which already holds the checkpoints and the evaluation videos, holds the
run's identity too: ``run_meta.json`` beside them, written by the first leg and read by every
later one. ``wandb.init`` honours ``WANDB_RUN_ID`` and ``WANDB_RESUME`` from the environment,
so setting those before the runner is built steers rsl_rl's own init without monkeypatching it
or editing the installed package.

The id is minted here rather than read back from wandb. Discovering what wandb chose would
mean init'ing first and persisting second, which loses the race against a leg that dies in its
first minutes; minting it up front means the file is on disk before wandb is ever contacted,
and it is the same 8-character shape wandb would have produced.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

META_NAME = "run_meta.json"

_ID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def _new_run_id() -> str:
    return "".join(secrets.choice(_ID_CHARS) for _ in range(8))


def _load(path: Path) -> dict | None:
    """``None`` for absent *and* for unusable: a queued leg must not die on its metadata."""
    try:
        meta = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        print(f"[run] {path} unreadable ({type(exc).__name__}: {exc}); starting a new wandb run")
        return None
    if isinstance(meta, dict) and isinstance(meta.get("wandb_run_id"), str) and meta["wandb_run_id"]:
        return meta
    print(f"[run] {path} carries no usable wandb_run_id; starting a new wandb run")
    return None


def _save(path: Path, meta: dict) -> None:
    # Temp file plus os.replace, which is atomic within a directory. A leg killed by the wall
    # clock part-way through a plain write would leave a truncated file, and the next leg would
    # have to choose between crashing and silently forking the curve.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def bind_wandb_run(log_dir, *, project: str, stage: str, rank: int = 0) -> str | None:
    """Point this leg's wandb init at the run's persistent id, minting it on the first leg.

    Returns the id, or ``None`` on ranks that do not log. Call it *before* constructing
    ``OnPolicyRunner``: the runner builds its summary writer in ``__init__``
    (``on_policy_runner.py:63``) and ``wandb.init`` reads the environment once, at that point.

    Rank 0 alone touches the file. rsl_rl disables logging on every other rank
    (``on_policy_runner.py:51``), so there is nothing to bind there, and eight ranks writing one
    path would race for no gain.

    One consequence worth knowing before reading the curves. wandb discards any step at or
    below the high-water mark it has already recorded for a run, and rsl_rl logs at
    step=iteration. Checkpoints land every ``--save-interval`` iterations, so a leg killed by
    the wall clock rewinds to the last one and re-trains up to ``save_interval - 1`` iterations
    that wandb has already seen; those points are dropped, and the curve is flat from the kill
    point until the run passes its previous peak. That is expected, not a bug, and the only
    alternative is a fresh run per leg -- which is what this function exists to stop.
    """
    if rank != 0:
        return None

    log_dir = Path(log_dir)
    path = log_dir / META_NAME
    meta = _load(path)
    if meta is None:
        meta = {
            # An explicit WANDB_RUN_ID seeds the FIRST leg only, so a run wandb already created
            # can be adopted by exporting its id once. After that the file is the sole source
            # of truth, otherwise a stray environment variable could split the run again.
            "wandb_run_id": os.environ.get("WANDB_RUN_ID") or _new_run_id(),
            "wandb_project": project,
            "wandb_entity": os.environ.get("WANDB_USERNAME"),
            "run_name": log_dir.name,
            "stage": stage,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by_slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        _save(path, meta)
        print(f"[run] wandb run {meta['wandb_run_id']} created, recorded in {path}")
    else:
        print(f"[run] wandb run {meta['wandb_run_id']} continued, read from {path}")

    os.environ["WANDB_RUN_ID"] = meta["wandb_run_id"]
    # "allow", not "must": the first leg has nothing to resume, and a run whose wandb history
    # was deleted upstream should keep training rather than abort at init.
    os.environ["WANDB_RESUME"] = "allow"
    _allow_config_val_change()
    return meta["wandb_run_id"]


def _allow_config_val_change() -> None:
    """Let a resumed leg overwrite the config the first leg logged.

    Reusing one wandb run across legs -- the whole point of the id above -- means the second
    leg calls ``wandb.config.update({"runner_cfg": ...})`` against a run that already has a
    ``runner_cfg``, and the two differ: ``resume`` is False on the first leg and True on every
    one after. wandb's default is to refuse a changed value, so leg 2 dies with

        ConfigError: Attempted to change value of key "runner_cfg"

    after training has already started -- late enough to look like a training failure rather
    than a logging one.     rsl_rl calls ``wandb.config.update({"runner_cfg": ...})``
    (``wandb_utils.py:49``) with no ``allow_val_change``, from inside the writer it builds
    itself, so there is no argument to pass and no hook to override.

    Two plausible fixes do NOT work, so do not re-derive them. ``wandb.Settings`` has an
    ``allow_val_change`` field, but ``Config._sanitize`` (``wandb_config.py:277``) consults it
    only under Jupyter -- passing it to ``wandb.init`` changes nothing here. And wandb exposes
    no environment variable for it. What is left is defaulting the argument on ``Config.update``
    itself, which is what this does.

    Overwriting is the right resolution rather than a lesser evil: the later leg's config is
    the one that describes the process actually running.

    Idempotent: chained legs import this module once, but a re-entrant call must not stack
    wrappers.
    """
    try:
        from wandb.sdk import wandb_config
    except ImportError:  # tensorboard runs do not need any of this
        return
    if getattr(wandb_config.Config.update, "_rb_allow_val_change", False):
        return

    original = wandb_config.Config.update

    def update(self, d, allow_val_change=None):
        # Only supply the default; an explicit False from some other caller still means False.
        return original(self, d, True if allow_val_change is None else allow_val_change)

    update._rb_allow_val_change = True
    wandb_config.Config.update = update
