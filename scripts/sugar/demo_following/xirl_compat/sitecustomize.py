"""Narrow API compatibility for released XIRL dependencies on Python 3.11.

XIRL imports ``load_state_dict_from_url`` from the torchvision 0.8 module
``torchvision.models.utils``.  That module was removed upstream; the function
itself is still provided by PyTorch.  This shim restores only that import path
and does not alter the released XIRL architecture, weights, loss, or trainer.
Pymunk 5.6.0, which is pinned by the released XIRL requirements, also reads
``collections.Sequence`` removed in Python 3.10; restore that exact alias.
"""

from __future__ import annotations

import collections
import collections.abc
import sys
import types

from torch.hub import load_state_dict_from_url


if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence


if "torchvision.models.utils" not in sys.modules:
    module = types.ModuleType("torchvision.models.utils")
    module.load_state_dict_from_url = load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = module
