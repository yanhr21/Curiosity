#!/usr/bin/env python3
"""Run one module after bounded retries of import-only filesystem failures.

The shared Python package tree has intermittently returned partial Torch
imports.  Retrying the experiment itself would be scientifically invalid, so
this bootstrap retries only module import.  Once ``main`` begins, its exit
status is final and no training/evaluation work is replayed.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path


ATTEMPTS = 8
ATTEMPT_ENV = "PAPER_ZERO_WAM_IMPORT_ATTEMPT"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_module_with_import_retry.py MODULE [ARGS ...]")
    module_name = sys.argv[1]
    module_args = sys.argv[2:]
    attempt = int(os.environ.get(ATTEMPT_ENV, "1"))
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        print(
            f"import-only attempt {attempt}/{ATTEMPTS} failed for "
            f"{module_name}: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        if attempt >= ATTEMPTS:
            traceback.print_exc()
            raise
        # A failed Torch import can leave registrations inside its C extension;
        # deleting Python modules is insufficient.  Re-exec before main() gives
        # the next import a genuinely clean interpreter and cannot replay work.
        os.environ[ATTEMPT_ENV] = str(attempt + 1)
        time.sleep(float(attempt))
        os.execv(
            sys.executable,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                module_name,
                *module_args,
            ],
        )
        raise AssertionError("os.execv unexpectedly returned")
    if module is None or not callable(getattr(module, "main", None)):
        raise RuntimeError(f"module does not expose callable main(): {module_name}")
    os.environ.pop(ATTEMPT_ENV, None)
    # From this boundary onward there is deliberately no retry: model loading,
    # optimizer updates, evaluation and rendering may execute only once.
    sys.argv = [module_name, *module_args]
    module.main()


if __name__ == "__main__":
    main()
