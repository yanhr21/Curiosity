# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Run the real panda_hydro example with an overridden hydroelastic stiffness kh,
# exporting USD. kh is forced on every ShapeConfig the example builds.

import newton
import newton.examples
from newton.examples.robot.example_robot_panda_hydro import Example

parser = Example.create_parser()
parser.add_argument("--kh", type=float, default=1e13)
viewer, args = newton.examples.init(parser)

# Force kh on every ShapeConfig the example constructs (incl. dataclasses.replace).
SC = newton.ModelBuilder.ShapeConfig
_orig_init = SC.__init__


def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.kh = args.kh


SC.__init__ = _init

print(f"[run_panda_hydro_kh] kh override = {args.kh:g}")
newton.examples.run(Example(viewer, args), args)
