# sugar_newton

Tactile sensing for the SUGAR G1 carry task, rebuilt on Newton.

Design: [`../PLAN/16_newton_tactile_rewrite/plan.md`](../PLAN/16_newton_tactile_rewrite/plan.md).
Task list: [`../TODO/16_newton_tactile_rewrite/todo.md`](../TODO/16_newton_tactile_rewrite/todo.md).
The audit that motivated the rewrite: [`../claude_context/findings.md`](../claude_context/findings.md).

Developed on branch `2026_8_19_sugar_newton` and integrated into `sugar` on 2026-08-27.
The merge keeps the newer `sugar` implementations under `SUGAR/`; this package supplies
only the Newton environment, tactile reducer, validators and glue around official BCPPO.

This package **depends on** Newton; it does not vendor or patch it. Plan 16 §3 —
the audit's sharpest finding was a local edit inside vendored IsaacLab
(`visuotactile_sensor.py:564-608`) that was indistinguishable from upstream by
inspection and caused the shear leak. `git diff` against upstream Newton must
stay empty.

## Status

The tactile core and its analytic validators run, the G1 CarryBox loop and official-BCPPO
execution smoke have run on Newton, and the required SUGAR assets/Refiner checkpoint are
present. The four contact-dependent Tracker reward terms are implemented from resolved
Newton forces with the official three-frame history and have passed the H200 environment-
level runtime/sign audit. The official Refiner rollout produced 912 endpoint-complete
Tracker clips; all are finite and aligned within one frame to their raw teacher motions.

The acting-Refiner admission queue is currently closed.  Fresh PPO, frame-zero anchoring,
official action anchoring, a frozen-expert residual, the six-layer causal temporal composer and
the released-Tracker-supervised frozen-Refiner adapter have all completed their fixed Newton gates.
The last topology uses the parameter-exact released Tracker only as a current-state action-label
source; its 510-D observation is synchronized with the deployed actor's 890-D observation, and the
teacher is absent at inference.  Fresh seed171717 passes the 64-update numerical gate with finite
parameters and `4/12288 = 0.03255%` divergences.  Both released experts retain exact-zero drift,
but frozen evaluation reaches only `1/20` 5-cm lifts and `0/20` strict completions, with mean
lift/contact `0.011805 m/0.195350`, below the fixed `16/20` rule and no better than the temporal
endpoint.  Downstream Tracker/BCPPO therefore has not launched.  Do not sweep teacher weight,
residual limit, LR, reward or update budget for this rejected topology.  Raw CarryBox motions
remain execution-smoke input only and are rejected for a formal run.

The next bounded diagnostic keeps the same serious frozen experts and deployed residual but uses
the repository's official BCPPO Stage-1 pure-distillation schedule, removing the simultaneous PPO
surrogate and critic updates that confounded the rejected mixed-objective endpoint.  Fresh
seed171718 has residual delta `0.00046239`, exact-zero critic drift, std delta `7.45e-10`, finite
tensors and exact-zero released-expert drift, but records one divergence in 384 transitions.  The
predeclared short gate required zero, so the diagnostic is rejected.  A seed171719 automatic launch
that mistakenly used the looser general ceiling was stopped after update 0 / 192 transitions; it
has no final result and must not be resumed.  No pure-distillation formal endpoint or downstream
Tracker/BCPPO is admitted.

The active bounded follow-up combines the same exact released Tracker action teacher with the
already-admitted six-layer causal temporal Refiner composer.  The Tracker remains training-only;
the deployed actor consumes the current 890-D state plus exact past `10 x 890` history.  Its
component and seed171720 zero-optimizer gates pass with exact-zero endpoint/expert/state drift.
Seed171721 then passes two Stage-1 updates / 384 transitions with zero divergence, composer delta
`0.00045949`, exact-zero critic drift and std delta `7.45e-10`.  A strict fixed-20 audit proves both
experts remain exact and the Tracker is absent at inference; all profiles are finite, while the
diagnostic physical result is `0/20` lift and `0/20` strict.  These structural gates automatically
admitted the single predeclared fresh seed171722 64-update run.  That endpoint passes its numerical
training contract with one divergence in 12,288 transitions (`0.00814%`), but fixed-20 evaluation
reaches only `8/20` lift and `0/20` strict.  Mean lift/contact are `0.071140 m/0.128391`; both experts
remain exact and all profiles are finite.  This is a real improvement over the short checkpoint,
not an admitted acting Refiner.  Tracker/BCPPO remains closed, and this topology must not be swept.

The active follow-up keeps the same official Refiner, released Tracker teacher and six-layer causal
Transformer but makes composition identifiable: deployed action is always the complete exact
Refiner action plus one bounded 29-D temporal correction.  The head starts at exact zero and the
Tracker remains training-only.  Fresh zero-optimizer and two-update gates precede a single fixed
64-update endpoint; the physical admission rule remains `16/20` lift and strict.

## Running the validator

Run simulation and GPU validation through a Slurm compute allocation. The current H200
runtime uses the exact Newton submodule plus the shared Python/Warp installations; no
training or simulation is launched on the login node.

```bash
srun --overlap --jobid=<H200_JOB_ID> --nodes=1 --ntasks=1 --pty bash -l
export NEWTON_PY=/public/home/yanhongru/envs/isaac_arena_py312/bin/python
export PYTHONPATH=/public/home/yanhongru/envs/newton_warp_114:$PWD/third_party/newton:$PWD
$NEWTON_PY -m sugar_newton.validation.incline
$NEWTON_PY -m sugar_newton.validation.hand_map --out sugar_newton/_gpu_out/hand_map
```

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

Reference output from the original CPU diagnostic (`mu = 0.5`, critical angle 26.57°):

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

## Friction: verified, after the test found a bug

`validation/friction.py` exists to close audit #4 — Plan 15's utilization divided
by a sensor-fixed `mu = 0.5` while training randomized the box's friction over
`U[0.2, 0.8]`, so the channel was blind to the quantity it was named after.

Writing the test found the same class of bug in this code. The reducer used
`rigid_contact_friction` **as** μ. It is not μ; it is a per-contact *scale*
(default 1.0) that hydroelastic reduction writes for moment matching, and which
MuJoCo multiplies the resolved material friction by. The pair itself combines by
elementwise **max**. The correct form is

    mu_contact = max(mu_a, mu_b) * friction_scale

The original incline test could not have caught it: both shapes had `mu = 0.5`,
so `max` coincided with the fallback constant and every number came out right
for the wrong reason.

The test is built to fail under each way of getting this wrong — the fallback is
set to an absurd `7.0`, and cases A and B are the same pair with μ *swapped
between the shapes*, which must read identically if the rule is really `max`:

```
case                        mu_ramp  mu_blk  mu_pair     util  util exp
A  high ramp / low block       0.80    0.30     0.80   0.2657    0.2657
B  low ramp / high block       0.30    0.80     0.80   0.2657    0.2657
C  both low                    0.30    0.30     0.30   0.7085    0.7085
D  both high                   0.90    0.90     0.90   0.2362    0.2362

PASSED — utilization tracks max(mu_a, mu_b); spread 0.4723 across mu 0.3-0.9
```

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

## Remaining validation items

- The hydroelastic friction-scale path has passed at scale 1.0, but a non-trivial
  moment-matching scale is still required to close that audit completely.
- More-than-one-world tactile validation and the composed friction video remain open.

The prescribed-velocity quantitative gate now passes on H200 in
`validation/hand_map.py`: exact kinematic carriage speed `0.0500 m/s`, load-weighted
reported slip `0.0502 m/s`, nonzero solved normal load and no hydroelastic overflow.
The free incline's sliding branch remains qualitative because its accelerating compliant
block is phase-dependent; the four sticking analytic cases retain their force,
utilization and exact-zero-slip assertions.

## Layout

```
sugar_newton/tactile/reducer.py      PatchTactile — contacts to per-patch channels
sugar_newton/validation/incline.py   analytic ground-truth validator
sugar_newton/validation/hand_map.py  prescribed H200 hand/plate slip validator
```
