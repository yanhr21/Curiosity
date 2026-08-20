# Plan 16: Newton Tactile Rewrite — Measured Normal, Friction and Slip

**Status:** active design, opened 2026-08-19. Supersedes Plan 15 as the active plan;
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
| **Teacher** | Kept — `refiner_model10000.pt`, the same artifact training uses today. |
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

`sugar_newton/` in this repo, on branch `2026_8_19_sugar_newton` (branched from `sugar`).
It **depends on** Newton rather than living inside the Newton fork, and the SUGAR source,
the frozen teacher and BCPPO are all on the branch to port from.

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
| 10 | `peak_pressure` | `kh · max(0, −depth)` max over faces | Pa |

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

If yes, the teacher ports as-is. If no, a teacher is retrained in Newton from the reference
motion — which is why the motion data matters more than the checkpoint.

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
