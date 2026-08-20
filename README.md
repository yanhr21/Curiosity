# sugar_newton

Tactile sensing for the SUGAR G1 carry task, rebuilt on Newton.

Design: `Curiosity/PLAN/16_newton_tactile_rewrite/plan.md`.
Task list: `Curiosity/TODO/16_newton_tactile_rewrite/todo.md`.
The audit that motivated the rewrite: `Curiosity/claude_context/findings.md`.

This package **depends on** Newton; it does not live inside it. Plan 16 §3 — the
audit's sharpest finding was a local edit inside vendored IsaacLab
(`visuotactile_sensor.py:564-608`) that was indistinguishable from upstream by
inspection and caused the shear leak. `git diff` against upstream Newton must
stay empty.

## Status

Phase 1 (tactile core + analytic validator) is **running and passing**. Phases
0 and 2-5 have not started; Phase 0 is blocked on copying assets off the runtime
host.

## Running the validator

Newton runs on the OCI-ord **login node against the CPU device** — no container,
no GPU, no SUGAR asset. That is what makes this validator cheap enough to run on
every change.

```bash
NT=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
SP=$NT/.venv/lib/python3.12/site-packages
PY=<a python 3.12 interpreter>     # the venv's own python lives at /root/... inside the container

cd /lustre/fs12/.../robot_baby/sugar_newton
PYTHONPATH=$SP:$NT:$PWD $PY -m validation.incline
```

`$SP` supplies warp / mujoco_warp / torch; `$NT` supplies the editable `newton`
package (the venv's `newton.pth` is not honoured when `PYTHONPATH` is set by
hand). On GPU, run it inside the CUDA container per
`Curiosity_newton/renders/build_and_render.sh`.

## What the validator asserts

A block of mass `m` on a ramp of angle `theta` with friction `mu`:

| # | assertion | regime |
|---|---|---|
| 1 | `normal_load == m g cos(theta)` | seated |
| 2 | `utilization_mean == tan(theta) / mu` | sticking |
| 3 | **slip is zero while `tan(theta) < mu`** | sticking |
| 4 | `utilization_max <= 1` | sticking |
| 5 | slip velocity > 0 and gross-slip fraction > 0.5 | sliding |

**Assertion 3 is the reason this file exists.** Plan 15's tactile reported
friction utilization `0.622` on a *static* grasp — past its own `0.60`
incipient-slip trigger with nothing moving — because TacSL projected the total
contact force into a per-taxel frame, so off-centre contact leaked the normal
force into the shear channel. It survived a full training and evaluation
campaign. A static test this small would have caught it on day one and did not
exist.

Current output (CPU, `mu = 0.5`, critical angle 26.57°):

```
  theta  stick    N meas     N exp  u_mean   u_exp   u_max     slip d     slip v     |v| fd  gross
   5.00   True    4.8870    4.8863  0.1750  0.1750  0.1918  7.347e-06  9.381e-07  9.484e-07  0.000
  12.00   True    4.7976    4.7978  0.4251  0.4251  0.5394  1.750e-05  2.237e-06  2.227e-06  0.000
  20.00   True    4.6094    4.6092  0.7278  0.7279  1.0000  2.915e-05  4.139e-06  4.055e-06  0.000
  24.57   True    4.4607    4.4610  0.9143  0.9142  1.0000  3.627e-05  6.279e-06  6.159e-06  0.000
  31.57  False    4.2014    4.1793  0.1750  1.2287  0.1750  0.000e+00  1.194e-01  6.734e-01  1.000
  41.57  False    0.0364    3.6699  0.0250  1.7735  0.0250  0.000e+00  2.965e-02  3.332e+00  1.000
```

Normal load matches analytically to four decimals. `u_mean` matches `tan/mu` to
four decimals. `u_max` exceeds `u_mean` and saturates at 1 — correct: the
leading corner of a tilted block reaches the friction cone before the patch as a
whole does, which is exactly the per-contact detail a single patch-level ratio
would hide. Slip displacement and slip velocity are two independent estimates
and agree to ~2%.

## What the first run taught us

Recorded here because each one is a fact about the platform, not about this code.

1. **Contact indices survive `update_contacts`.** `SolverMuJoCo.update_contacts`
   (`solver_mujoco.py:4380-4411`) *replaces* the whole contact set — count,
   shapes, points, normal and force — with MuJoCo's own. `match_index` is
   computed by the Newton pipeline on the pre-solve ordering, so anchor
   propagation is only valid if the ordering round-trips. **It does**, verified
   by comparing contact *positions* and not just shape pairs and normals (in a
   single-pair scene those cannot distinguish a permutation). The validator
   fails loudly if this ever stops holding.

2. **Slip displacement alone hides gross slip.** The matcher breaks a match once
   a contact moves more than `contact_matching_pos_threshold` (0.5 mm) in a
   step, so a fully sliding patch re-anchors every frame and its anchor drift
   reads **exactly zero**. Anchor drift measures *incipient* slip — the
   pre-sliding micro-displacement regime — and nothing else. The
   `gross_slip_fraction` channel (re-anchor rate) covers the sliding regime and
   reads 0.000 / 1.000 across the transition with no threshold anywhere. This
   defect was in the first version of the reducer and the validator caught it.

3. **`add_shape_box` adds mass on top of `add_body(mass=...)`**, from
   `ShapeConfig.density` (default 1000 kg/m³). A 10 cm cube silently added 1.0 kg
   to a 0.5 kg body, and the sensor dutifully reported 3× the expected normal
   load. The sensor was right and the scene was wrong. Pass `density=0.0` and
   assert `model.body_mass` after `finalize()`.

4. **The compliant contact has a ballistic envelope.** Past ~40° at `dt = 1/240`
   with `ke = 1e5`, the block launches and chatters instead of sliding steadily,
   so `mg cos(theta)` stops being the right expectation. The validator reports
   this as a NOTE and still asserts the slip channels — it bounds the solver's
   envelope, not the sensor's.

## Open — do not assume these are done

- **The real-μ path is not yet exercised.** `rigid_contact_friction` is only
  allocated when the pipeline is built with a hydroelastic SDF config
  (`collide.py:896`), and the validator's box-on-box scene has none — so it ran
  on `TactileConfig.fallback_friction`. Repairing audit #4 depends on the
  per-contact path, and **it is still unverified**. Re-run with hydroelastic
  mesh shapes and a randomized μ before claiming that finding is fixed.
- Contact area and peak pressure (Plan 16 §4 channels 9-10) are not implemented;
  they need the hydroelastic contact surface.
- The quantitative sliding test should be a prescribed-velocity scene, where
  tangential velocity is an input rather than an outcome. The free-sliding
  assertions here are deliberately qualitative.
- Nothing has been run on GPU, and nothing has been run with more than one
  world.

## Layout

```
sugar_newton/tactile/reducer.py   PatchTactile — contacts to per-patch channels
validation/incline.py             analytic ground-truth validator
```
