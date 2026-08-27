# Plan 16: Newton Tactile Rewrite — Measured Normal, Friction and Slip

**Status:** integrated into `sugar` and under active Newton execution on 2026-08-27. All
simulation, validation and training run inside the retained H200 Slurm allocation; the login
node is not a compute path. This plan is an execution record, not an authorization boundary.
Plan 15 stays as the record of the PhysX experiment and its audit.

Plan 15 在 IsaacLab/PhysX 上得到 null result，随后的完整审计（115 条 findings，见
`claude_context/findings.md`）确认该 null result 不能作为科学结论：触觉通道本身测不到
它声称测的量。Plan 16 换到 Newton 重建，保留 assets、teacher 和 BCPPO，但触觉传感
完全重写。以下用英文书写以保证精度；findings log 仍是权威记录。

*Written in English for precision. Section numbering mirrors Plan 15.*

---

## 1. Why a rewrite rather than a fix

Plan 15's audit found four independent confounds and one asymmetric bug, all of which
trace to a single architectural decision: **tactile was computed in a read-only view
parallel to the solver, with its own frame conventions and its own friction constant.**
TacSL projected the *total* contact force into a per-taxel frame, so under stick the
"shear" channel reported a geometric leak of the normal force; and it divided by the
sensor's fixed `mu = 0.5` while training randomized the object's real friction over
`U[0.2, 0.8]`. Neither channel could carry the signal the experiment was about.

Newton removes the architectural cause rather than the symptoms. Its collision pipeline
produces the contacts the solver actually integrates, and every quantity below is read
off those same contacts.

Secondary motivations, in the order the user stated them: faster training (Plan 15 ran at
`num_envs = 4`; Newton replicates worlds on one `ModelBuilder`), faster rendering, and
easier tactile sensing.

## 2. What is kept and what is rebuilt

| | decision |
|---|---|
| **Assets** | Kept. G1 29-DoF, 54 anatomical patches, official CarryBox. |
| **Teacher** | Architecture and initialization kept — the exact official 890-D Refiner and `refiner_model10000.pt`; the released weights failed the Newton open-loop physical gate and are being adapted in Newton with fresh PPO state. |
| **Warm start** | Kept — the official Tracker checkpoint. |
| **Reference motion** | Kept — `data/CarryBox`. |
| **Learning algorithm** | Kept — `BCPPO` (`rsl_rl_bcppo.py`), unmodified. |
| **Tactile sensing** | **Rebuilt from zero.** No TacSL, no `PatchSlipDetector`, no channel scales. |
| **Reward** | **Rebuilt.** Plan 15's reward is anti-grasp (audit #1/#2/#3). |
| **Physics backend** | IsaacLab/PhysX → Newton (`SolverMuJoCo`, `solver="newton"`). |
| **Env framework** | IsaacLab manager-based → lean vec-env on Newton. |

`OFFICIAL_REFINER` is used **twice** by the current training contract
(`train_online_patch_mass_bcppo.py:95` and `:108`): as BCPPO's distillation target and as
the *acting* policy for the episode prefix in `OnlineTeacherHandoffVecEnvWrapper.step`.
Both roles must survive the port.

## 3. Where the code lives

`sugar_newton/` in this repo. Development from `2026_8_19_sugar_newton` was compared and
merged into `sugar` on 2026-08-27. It **depends on** the pinned Newton submodule rather than
living inside the Newton fork; the SUGAR source, exact official teacher architecture and
unmodified BCPPO implementation remain available to the port.

The audit's sharpest lesson is `IsaacLab/.../tacsl_sensor/visuotactile_sensor.py:564-608`
— a *local* modification inside vendored upstream code, indistinguishable from upstream by
inspection, which introduced the per-taxel frame that caused the shear leak (audit #5).
Upstream v2.3.2 uses one constant quaternion per sensor. Nobody could see the difference
because the vendor tree was also the edit surface.

Rule for Plan 16: **`git diff` against upstream Newton must stay empty.** Anything the
research needs that Newton does not provide is written in our package or contributed
upstream — never patched into the vendored engine. If a genuine engine change is
unavoidable it goes in a single, named patch file with a test, not an in-place edit.

## 4. The tactile contract — measured, not inferred

Per patch, per control step. Every quantity is a reduction over the rigid contacts whose
shape pair is (patch shape, object shape), read from the buffer the solver integrated.

| # | channel | definition | unit |
|---|---|---|---|
| 1 | `contact_count` | active contacts on the patch | count |
| 2 | `normal_load` | `Σ (f·n)` over contacts | N |
| 3 | `friction_load` | `‖Σ (f − (f·n)n)‖` | N |
| 4 | `friction_load_abs` | `Σ ‖f − (f·n)n‖` | N |
| 5 | `utilization_max` | `max ‖f_t‖ / (μ_c ‖f_n‖)` | — |
| 6 | `utilization_mean` | load-weighted mean of the same ratio | — |
| 7 | `slip_displacement` | load-weighted `‖tangential anchor drift‖` | m |
| 8 | `slip_velocity` | load-weighted `‖tangential relative velocity‖` | m/s |
| 9 | `contact_area` | hydroelastic contact-surface area on the patch | m² |
| 10 | `peak_pressure` | depth field scaled to integrate to `normal_load`, max over faces | Pa |

**Contact area and pressure (9-10) come from the surface, and pressure is not `kh·depth`.**
Reduction collapses a patch's surface into a few contacts that carry force but no
footprint, so both channels read the hydroelastic contact surface directly. An earlier
draft of this row specified `kh · max(0, −depth)`. That is the hydroelastic model's own
law, but under `use_mujoco_contacts=False` the normal force is MuJoCo's constraint solve
— the surface supplies geometry, not force. Measured on the incline scene,
`∫ kh·depth dA` came to **328.8 N against a true normal load of 4.886 N**, a factor of
67 that would drift with timestep and solver settings while looking plausible. So the
depth field supplies the shape and the solved load supplies the magnitude:

    p_i = penetration_i · normal_load / Σ_j (penetration_j · area_j)

which integrates to the measured load by construction. `depth < 0` is penetration, the
convention `Viewer.log_hydro_contact_surface(penetrating_only=True)` filters on.
`sugar_newton/validation/pressure.py` is what caught both errors and fails if either
returns.

**Normal and friction (channels 2-4).** `f_n = (f·n)n` against `rigid_contact_normal`, the
true contact normal; `f_t = f − f_n`. This is the decomposition `SensorContact` already
performs (`newton/_src/sensors/sensor_contact.py:90-95`). It is the physically correct one,
and it is the one TacSL did not do.

**Friction utilization (5-6) uses the real per-contact μ.**

    mu_contact = max(mu_a, mu_b) * friction_scale

`rigid_contact_friction` is **not** μ — it is a per-contact *scale*, default 1.0, that
hydroelastic contact reduction writes for moment matching when many surface faces collapse
to a few representative contacts (`contact_reduction_hydroelastic.py:885`). MuJoCo
multiplies the resolved material friction by it (`kernels.py:460-468`). The material
friction itself is combined across the pair by elementwise **max** (`kernels.py:165`),
MuJoCo's standard rule — not an average, and not shape0's value.

An earlier draft of this section claimed `rigid_contact_friction` *was* the coefficient,
and the reducer was written to match. That would have read ≈1.0 for every contact
regardless of material: a friction channel that cannot see friction, which is audit #4
again in a new costume. Corrected in both places, and
`sugar_newton/validation/friction.py` now fails if it ever regresses.

This is the direct repair of audit #4.

Both a max and a load-weighted mean are reported because Coulomb's condition is *per
contact point*; a single patch-level ratio hides a saturated point inside an unsaturated
patch.

**Slip (7-8) is measured against a persistent anchor.** With `contact_matching` enabled
(`newton/_src/sim/collide.py:505`) the pipeline populates `rigid_contact_match_index`,
giving frame-to-frame contact correspondence. We keep our own anchor buffer:

```
new contact  (match_index < 0):  anchor[i] = current world contact midpoint
matched      (match_index = m):  anchor[i] = anchor_prev[m]
slip displacement            :  tangential part of (current midpoint − anchor[i])
```

Anchor drift under a maintained contact **is** material slip. Tangential relative velocity
is computed independently from body states via `velocity_at_point` (Newton layout
`(linear, angular)`, `newton/_src/math/spatial.py:54`), giving two independent slip
estimates that must agree.

We use `contact_matching="latest"`, not `"sticky"`. Sticky mode replays the previous
frame's body-frame anchors *onto the contact points the solver reads*, which turns stick
into a tangential spring — a physics change. Plan 16 measures; it does not perturb.
(`"sticky"` is additionally marked experimental.)

**There is no slip detector.** No threshold latch, no evidence counter, no differencing
buffer, no `reset_mask`. Audit #6 (the GROSS latch) and audit #7 (the swallowed reset that
is the best single explanation for PS < P) describe state that no longer exists. Slip is a
measurement with a unit, not a score.

**No channel scale file.** Plan 15 baked p99.5 max-scaling into the encoder's persistent
buffer and therefore into every checkpoint, with nothing binding a scale file to the
channel definitions that produced it. Plan 16 normalizes with fixed physical constants
recorded in the config, so a checkpoint carries its own interpretation.

## 5. Validation — the gate that would have caught Plan 15

Before any of this touches a robot, the reducer is validated against analytic ground truth
on a block resting on an incline of angle `θ` with friction `μ`:

1. `normal_load` == `m g cos θ` (within solver tolerance)
2. `utilization_max` == `tan θ / μ`
3. **`slip_displacement` and `slip_velocity` are exactly zero for `tan θ < μ`**
4. above the critical angle, `slip_velocity` matches the finite-differenced block velocity
5. `utilization_max` never exceeds `1 + ε` while sticking

Test 3 is the one that matters. Plan 15's measured failure was utilization `0.622` at
**zero** relative motion — past the `0.60` slip trigger with nothing moving. A five-line
static test would have caught it on the first day and did not exist. It exists now, and it
runs on CPU with no GPU, no container and no SUGAR asset.

An additional runtime check belongs in the same file: `utilization > 1 + ε` under a
converged solve is proof of contamination by construction, and should raise rather than
be normalized away.

## 6. Phases

**Phase 0 — prerequisites.** Copy from the runtime host (`/public/home/yanhongru/Curiosity`,
none of it is on the OCI-ord filesystem): `SUGAR/descriptions/robots/g1/meshes`,
`data/CarryBox`, `refiner_model10000.pt`, the official Tracker checkpoint, the R15 gel USD,
the CarryBox asset. Bring up a Newton env on this cluster.

**Phase 1 — tactile core + validator.** §4 and §5. No SUGAR asset needed; runs on CPU.
*This phase is unblocked and starts immediately.*

**Phase 2 — asset and the throughput question.** G1 29-DoF from Newton's stock
`g1_29dof_with_hand_rev_1_0.usda` (`newton/examples/robot/example_robot_g1.py:45`), 54
anatomical patches as hydroelastic mesh shapes on `ModelBuilder`, then
`replicate(world_count)`. Benchmark worlds × patches × `sdf_max_resolution` → fps.

This benchmark decides whether the rewrite delivers its headline motivation. The existing
measurement is **82 fps with 2 pads in 1 world**, and this path is *collision-bound on SDF
queries* — 108 patch shapes across many worlds is unmeasured. Run it before building on it.

**Phase 3 — observations and the teacher gate.** The teacher's 890-D observation
(`official_refiner_nominal_teacher.py:35`) has 16 terms
(`base_refiner_env_cfg.py:219-243`): future reference frames and anchors, body pos/ori,
base lin/ang vel, joint pos/vel rel, last action, and object pos/ori/lin/ang vel in body
frame. **None of them touch PhysX** — each is a function of reference motion, articulation
state, or rigid-body state, all of which Newton provides. Port the 16 terms, load the
checkpoint, run it open-loop, and answer one question: **does the frozen Refiner still lift
the CarryBox under MuJoCo-Warp?**

The fixed 20-profile gate answered **no** for the released weights: all 20 profiles remained
finite, but only `1/20` lifted the box by at least 5 cm and `0/20` strictly completed; mean
peak lift was `0.01125 m`. The exact official architecture and checkpoint initialization are
therefore retained, while the acting teacher is adapted in Newton from the 100 CarryBox
reference motions with fresh optimizer and iteration state. No Tracker training is admitted
until the adapted teacher passes a frozen physical gate. The first transfer attempt exposed
two Newton reset/capacity defects: per-world `njmax=2048` overflowed and q/qd-only reset left
MuJoCo-Warp warm-start buffers contaminated after divergence. The corrected path uses 8192
constraints per world and Newton's official masked `SolverMuJoCo.reset`. Its fresh 64-update
run completes 12,288 transitions with one isolated divergence (`0.00814%`) and finite policy
parameters; this is a training-stability pass only, while the fixed 20-profile deterministic
physical gate remains the teacher-admission test.

The 64-update frozen gate is stable but not admitted: `1/20` profiles lift at least 5 cm and
`0/20` strictly complete. The continuous response is broad rather than one-outlier-only:
`16/20` profiles improve peak lift (median `+0.00786 m`) and `19/20` improve bilateral contact
(median `+0.02922`). The single predeclared longer run was therefore started fresh from the same
official checkpoint, not resumed from update 64. It completed all 256 updates / 49,152
transitions with `13/49152 = 0.02645%` divergence, finite policy parameters and a maximum
actor/critic parameter delta of `0.0270707`; its training gate passes. The fixed frozen physical
gate on `model_255.pt` is nevertheless decisively negative: all `20/20` profiles are finite, but
`0/20` lift by 5 cm and `0/20` strictly complete. Mean peak lift is only `0.0031307 m` and mean
bilateral contact is `0.152296`. This endpoint is worse than fresh-64 on the declared physical
metrics, so the Refiner PPO objective is rejected and must not receive another update-budget
extension or checkpoint-selection sweep.

The first post-rejection diagnostic isolates the initial-state distribution rather than changing
the reward or extending the failed run. Under the exact 300-step training sampler, frame zero has
only `1.787%` expected probability. The stronger hypothesis that training mostly starts with an
already lifted box is false: only `3.208%` of reset choices exceed 5 cm, across 29/100 motions, so
the reset-distribution audit records `reset_distribution_mismatch=false` under its predeclared
criterion. Paired frozen endpoints nevertheless show that the 256-update degradation is broad:
relative to fresh-64, lift and bilateral contact decrease on `19/20` profiles and mean termination
moves from step `226.1` to `203.95`; four profiles now fail early on end-effector position.

One bounded frame-zero-anchor diagnostic therefore changes exactly one variable: of eight Newton
training worlds, one always resets at frame zero while seven retain the original random-phase
coverage. It starts fresh from the exact official Refiner with seed 171702, unchanged reward,
optimizer stabilization and 64-update budget. Explicit evaluator starts override the anchor. A
16-reset H200 smoke verifies that the anchored world is always zero while the random world samples
nonzero phases. The same frozen 20-profile physical gate decides the endpoint; failure rejects this
reset intervention without an update extension.

That diagnostic is complete and negative. Seed 171702 finishes 64 updates / 12,288 transitions
with one divergence (`0.00814%`), finite policy parameters and the exact official topology. The
frozen endpoint remains `0/20` lift and `0/20` strict completion, with mean lift `0.007011 m`, mean
bilateral contact `0.179006` and mean termination step `218.75`. Relative to the original random
fresh-64 endpoint, only `1/20` profiles improve lift, `2/20` improve contact and `2/20` run longer;
all three aggregate metrics regress. Do not increase the number of anchored worlds, extend this
run or turn reset weighting into a sweep. The next acting-teacher method must preserve the exact
official Refiner while adding a serious action/behavior anchor during Newton adaptation rather
than relying on the same unanchored PPO objective.

The downstream Tracker path now fails closed on that decision. Its Newton VecEnv loads the
same checkpoint as a strict deterministic acting Refiner and as BCPPO's frozen distillation
teacher, verifies parameter equality, executes the Refiner until a no-reset 5 cm / 10-frame
physical handoff, and exposes a training-only post-handoff mask that never enters the 510-D
actor. The mask removes prefix PPO/value/entropy credit but deliberately retains SUGAR's
official full-trajectory teacher distillation. The runner rejects a missing, failed,
differently hashed or malformed frozen-gate
result and has no resume route. The student actor/critic/std load exactly from the released
CarryBox Tracker while its checkpoint optimizer and iteration are ignored, preserving the
declared warm start with fresh Newton BCPPO state. This implementation does not admit a launch by
itself. The fresh-256 endpoint failed the unchanged gate, and the automatic chain recorded
`AUTO_BCPPO_CHAIN_REJECTED_REFINER_GATE`; no acting-handoff smoke or Tracker optimizer update was
run.

For any future independently admitted acting checkpoint, one fixed 12-update / one-profile
runtime smoke must first demonstrate a real
handoff, nonzero student-controlled steps, finite policy state, frozen teacher parameters and
the original divergence bound. It then launches one fresh 3000-update, eight-world BCPPO run,
which reaches the complete official BC/critic/PPO curriculum rather than stopping inside pure
distillation. The fixed endpoint is automatically compared with the exact released Tracker on
twenty predeclared processed profiles. Both arms share the acting Refiner, frame-zero state,
physics and observation hashes; evaluator validity and physical advantage are separate
machine-readable decisions.

**Phase 4 — env and learning.** A vec-env implementing the `rsl_rl` VecEnv protocol
(torch↔warp interop following `newton/_src/solvers/kamino/examples/rl/`), with BCPPO
unmodified. Reward built correctly from the start:

- patches excluded from any undesired-contact penalty (audit #1 — in Plan 15 all 54 pads
  were counted at −0.02/body/step, and six in contact cancelled the entire achievable
  positive reward of 5.125);
- the contact-reward term pointed at bodies that actually have collision (audit #2 — Plan
  15's `hoi_contact` read hand links whose collision subtrees are deactivated at spawn, so
  it supplied no gradient at all);
- a term that rewards holding the box (audit #3 — Plan 15 had none).

**Phase 5 — experiment protocol.** Decisions that survive any engine:
train and evaluate on the same motion distribution (audit #8), report strict terminations
(audit #9), and a defensible interval — more than 3 seed clusters, or BCa, or both, with
multiplicity stated (audit #10).

## 7. What this rewrite does not buy

**Comparability.** Nine PhysX endpoints and 900 frozen rollouts do not transfer. The
`-0.2712` becomes a fact about an engine no longer in use. That is an acceptable trade —
the audit's conclusion was already that the result could not be published — but it should
be stated rather than discovered later.

**A result.** Plan 16 builds a correct instrument. Whether live whole-hand tactile improves
frozen physical behaviour under an online mass change remains open, and the claim stays
what Plan 15 §1 set: an *incremental* benefit over the deployable proprioception base.
Mass leaks into proprioception through joint sag and tracking error; "only tactile can
sense weight" is not a claim this design can support in any engine.

## 8. Standing rules carried forward

- Never let privileged signals into the actor: object pose, mass, mass factor, jump flag,
  reward, or future frames are evaluation labels only. Note Plan 15 violated this in
  substance because TacSL derived shear from `relative_velocity_world` (F-0008); in Plan 16
  slip velocity is computed from body states and is a **legitimate** tactile observable
  only insofar as a real sensor could infer it — the actor consumes patch channels, never
  the object's state.
- Never claim sim-to-real or GelSight calibration. These are high-fidelity simulated
  signals.
- A patch is the policy unit, never a contact point.
- Never report a number without naming the evaluation view.
