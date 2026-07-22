# SPDX-License-Identifier: Apache-2.0
"""Dataset → Newton scene ingestion (physics + tactile).

Adapters parse a dataset's native layout into a dataset-agnostic :class:`SceneSpec`;
:func:`build_newton_scene` turns it into a Newton ``ModelBuilder``. See
``claude_context/dataset_ingestion.md`` for the design.
"""

from __future__ import annotations

from .spec import (
    DoorSpec,
    MaterialSpec,
    ObjectSpec,
    RandomizationSpec,
    RobotSpec,
    RoomSpec,
    SceneSpec,
    SensorSpec,
    WallSpec,
)

__all__ = [
    "SceneSpec",
    "ObjectSpec",
    "MaterialSpec",
    "RoomSpec",
    "WallSpec",
    "DoorSpec",
    "RobotSpec",
    "SensorSpec",
    "RandomizationSpec",
]
