# SPDX-License-Identifier: Apache-2.0
"""Per-dataset adapters: native layout → :class:`~scene_ingest.spec.SceneSpec`."""

from __future__ import annotations

from .sage import load_sage_scene

__all__ = ["load_sage_scene"]
