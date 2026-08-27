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

## 9. Official-Refiner action-anchor transfer (2026-08-27)

The random-reset fresh-256 Refiner and the one-of-eight frame-zero fresh-64
intervention both fail the unchanged frozen physical gate, so neither checkpoint may
act for downstream Tracker training.  The next bounded method keeps the exact released
`890 -> 512/256/128 -> 29` Refiner as both student initialization and a separately loaded,
frozen BCPPO teacher.  Newton supplies the same current causal 890-D privileged group to
the teacher; no future state, outcome label, reset anchor or toy controller is added.
The official BCPPO mean-distillation term remains active with a stage-3 floor of `1.0`
while PPO starts immediately.  Training uses random reset, eight worlds, action std
`0.05`, fixed learning rate `1e-5`, and exactly 64 fresh updates.

The first two-update smoke with adaptive KL is rejected.  Its first update raised the
learning rate from `1e-5` to `1e-2`, producing actor/critic maximum parameter changes
`0.02166/0.06118` and action std `0.06406`.  A single protocol correction disables that
adaptive update (`desired_kl=None`) without changing the official BCPPO loss.  The fresh
seed171705 smoke keeps both recorded learning rates at `1e-5`, has zero divergence,
finite parameters, actor/critic changes `0.0001936/0.0002668`, and std `0.0500219`; it
passes the predeclared smoke limits (`actor <= 0.002`, std `<= 0.052`).  The earlier
zero-optimizer audit remains exact over 384 real Newton transitions.

The formal endpoint is evaluated on the unchanged fixed 20-profile Refiner physical
gate.  A numerical training pass is not admission: at least `16/20` profiles must satisfy
the existing strict lift/hold rule.  Failure stops before Tracker/BCPPO; success alone
allows the already implemented acting-teacher handoff chain to start automatically.

That formal seed171706 endpoint is complete and rejected.  All 12,288 transitions and
parameters are finite, divergence is `5/12288 = 0.0407%`, all 64 recorded learning rates
remain `1e-5`, and actor/critic maximum changes are `0.001884/0.010272`.  The frozen gate
has `20/20` finite profiles but only `2/20` 5-cm lifts and `0/20` strict completions; mean
peak lift/contact are `0.027440 m/0.211441`.  This is below the fixed `16/20` rule, and a
real `train_bcppo` entry call rejects the gate before constructing its environment or
taking an optimizer step.  Relative to the exact official pre-update Refiner on the same
profiles, lift improves in `13/20`, bilateral contact in `16/20`, and motion95 changes
from `0.05312` to `0.22117 m`; this is a directional Newton-adaptation response, not an
acting-teacher admission.  Do not extend the update budget or sweep anchor strength/LR.

The next parameter-free physics audit changes only the still-unmatched torso collision
topology.  Isaac's official URDF converter convex-hulls every mesh collider, while Newton
currently keeps the 51,410-triangle torso mesh.  Its derived convex hull has 2,586
triangles and `1.79x` the enclosed volume.  A one-profile execution smoke is finite and
replaces exactly two torso shapes; the fixed-20 official-Refiner comparison decides this
hypothesis without changing or training the policy.

That torso comparison is complete and negative.  Exact-official mesh versus torso-hull
both give `1/20` lift and `0/20` strict completion.  Mean lift changes only
`0.011250 -> 0.011333 m` and bilateral contact `0.193976 -> 0.194529`; matched mean lift
delta is `+0.0000834 m`, with `7/20` profiles improving and `9/20` worsening.  The largest
absolute per-profile lift change is only `0.00248 m`.  Do not enable the torso hull by
default, hull additional links, or run another collision-geometry sweep.  The next
acting-policy method must change controller topology while keeping the exact official
Refiner frozen, using only adapter code around that serious released component.

The frozen-expert residual topology now passes its pre-training gates.  It reuses the
repository's admitted `FrozenSelectedTrackerResidual` design: a parameter-exact frozen
official Refiner plus one trainable `890 -> 512/256/128 -> 29` residual whose final layer
is exact zero and whose deployed output is bounded by `tanh` with the existing limit
`1.0`.  The component audit gives exact zero composed-action delta, zero expert gradients,
and nonzero residual gradients.  A 384-transition zero-optimizer Newton smoke has zero
divergence and exact zero policy-state change.  The fresh two-update seed171709 smoke keeps
the expert bitwise exact, changes the residual by `0.0002166`, holds LR at `1e-5`, std at
`0.050024`, and remains finite with zero divergence.  The fixed seed171710 endpoint is now
complete.  All 64 logged learning rates are `1e-5`; training has 2 divergences in 12,288
transitions (`0.01628%`), finite parameters, residual delta `0.002876`, and exact-zero
embedded-expert weight/std drift.  The unchanged fixed-20 composed-action gate rejects it at
`0/20` lift and `0/20` strict completion versus required `16/20`.  Mean peak lift is
`0.009543 m` and bilateral contact `0.197050`.  Relative to the exact official Refiner,
matched mean lift decreases by `0.001706 m`; relative to the action-anchor endpoint it
decreases by `0.017897 m`.  This topology is physically rejected: do not sweep residual
limit, learning rate, reward weights or update budget, and do not launch Tracker from it.

The next fixed controller-topology diagnostic is a causal temporal composer around the same
parameter-exact official Refiner.  It reuses the repository's admitted six-layer, 384-D
Transformer transition core rather than introducing a smaller local model.  The deployed actor
receives the current official 890-D observation plus an explicit exact past `10 x 890` history;
the history is reset per Newton world and its final frame must be elementwise identical to the
current observation.  The frozen Refiner acts only on the current frame.  A zero-output head
initializes the composed action exactly to that endpoint, while the trainable output controls a
causal expert-retention scalar and bounded 29-D correction.  Future states, terminations and
outcome labels never enter the actor.  This method must first pass component, zero-optimizer and
two-update H200 gates.  Only those machine-readable passes admit one fresh 64-update endpoint;
there is no history-length/model-size/LR/residual-limit/update sweep.

Those pre-training gates now pass.  The component has 11,984,443 total and 11,360,286
trainable parameters; composed/endpoint, retention-from-one and bounded-residual deltas are all
exact zero, the official expert has zero gradients, the Transformer receives maximum gradient
`8.4493`, and a mismatched final history frame is rejected.  Fresh seed171712 executes 384 live
Newton transitions with zero optimizer steps, zero divergence, finite tensors and exact-zero
policy-state change.  Fresh seed171713 executes two optimizer updates with zero divergence,
fixed LR `1e-5`, composer delta `0.00021968`, final std `0.0500613`, and exact-zero embedded
expert drift.  Its live 9790-D observation audit also records exact-zero history-last/current
and composed/endpoint deltas.  These machine gates admit exactly one fresh seed171714,
eight-world, 64-update endpoint followed by the unchanged fixed-20 gate.

That formal endpoint is complete and rejected.  Seed171714 finishes all 64 updates / 12,288
transitions with three divergences (`0.02441%`), finite policy parameters, fixed final learning
rate `1e-5`, actor/critic maximum changes `0.003092/0.012734`, and final std `0.050575`.
The embedded official Refiner actor and std remain parameter-exact.  The fixed 20-profile Newton
gate has `20/20` finite profiles and no active rollout divergence, but only `1/20` reaches a
5-cm lift and `0/20` reaches strict completion versus the required `16/20`.  Mean peak lift and
bilateral contact are `0.012809 m/0.198697`.  Mean expert retention is `0.99999988` and the mean
absolute bounded correction is only `0.010058`, so the learned temporal controller remains
near the rejected official endpoint rather than producing the missing executable contact
behavior.  Do not sweep history length, Transformer size, residual limit, learning rate, reward
weights or update budget, and do not launch Tracker from this checkpoint.

The next bounded serious diagnostic changes the supervision source rather than scaling the same
PPO/distillation objective.  The released parameter-exact CarryBox Tracker has already executed
closed-loop in Newton and lifted the box by `0.21--0.30 m`; it may therefore provide a current-state
action label for an adapter around the frozen official Refiner.  The released Tracker and Refiner
remain frozen, Tracker actions are training labels only, and future states, rewards, outcomes and
terminations never enter the deployed actor.  Before any formal Newton run, the implementation
must prove exact released-checkpoint loading, causal 510-D/890-D state alignment, zero optimizer
state change in a live smoke, nonzero adapter-only gradients, and exact-zero frozen-expert drift.
This is an action-supervision diagnostic using serious released SUGAR components, not permission
to start downstream Tracker/BCPPO or to substitute a local teacher.

The released-Tracker-supervised adapter now passes its pre-training gates.  BCPPO receives a
separate exact 510-D Tracker observation from the same Newton action boundary as the student's
890-D Refiner observation; reading the paired groups changes neither `q` nor `qd`.  The deployed
actor is the existing frozen-official-Refiner `890 -> 512/256/128 -> 29` residual topology, while
the exact released Tracker is registered only as a frozen training teacher.  The component gate
starts at exact zero endpoint delta, gives zero Refiner and Tracker gradients, and gives residual
gradient `0.38765`.  Fresh seed171715 runs 48 live Newton transitions with zero optimizer updates,
zero divergence, finite tensors and exact-zero policy-state change.  Fresh seed171716 runs two
updates / 384 transitions with zero divergence, fixed LR `1e-5`, residual/critic deltas
`0.0004075/0.0004647` and final std `0.0500423`; the two frozen experts retain exact-zero
weight/std drift.  Distillation loss decreases from `0.2598` to `0.1961`, establishing a real
released-Tracker action signal rather than the earlier near-identity Refiner anchor.

A one-profile evaluator smoke strictly loads the resulting dual-expert training checkpoint,
proves that the Tracker teacher is unused at inference, and verifies the deployed composed action
exactly.  Its physical outcome is diagnostic-only and does not select a checkpoint.  These gates
admit exactly one fresh seed171717, eight-world, 64-update endpoint and the unchanged fixed-20
physical gate.  There is no teacher weight, residual limit, LR, reward or update-budget sweep.

That formal endpoint is now complete and rejected.  Fresh seed171717 finishes all 64 optimizer
updates / 12,288 Newton transitions with finite parameters, fixed LR `1e-5`, final std `0.0507304`
and `4/12288 = 0.03255%` synchronized divergences, below the fixed `0.5%` numerical ceiling.  The
residual actor and critic change by `0.0032824/0.0124748`.  Frozen evaluation strictly restores
the same `model_63.pt` (SHA256
`b8279bc6de12ff86edfc679d0cbfaae5696a12951fd56506d3c2ed222ee86e96`) over all 20 fixed profiles.
Both released experts retain exact-zero weight/std drift, the Tracker remains training-only, all
profiles are finite and the recorded composed action equals the deployed action.  Nevertheless,
only `1/20` profiles reach a 5 cm lift and `0/20` meet strict completion, versus the required
`16/20` for each.  Mean peak lift/bilateral contact are `0.011805 m/0.195350`, and the mean absolute
residual is `0.009393`.  This does not improve the rejected temporal endpoint's `1/20, 0/20`
physical result.  Downstream Tracker/BCPPO remains closed; do not sweep the teacher weight,
residual limit, LR, reward or update budget for this topology.

The next bounded diagnostic isolates a controller-objective confound rather than scaling that
rejected mixed objective.  The failed run entered BCPPO Stage 3 immediately, so every actor update
combined PPO surrogate gradients with the exact released-Tracker action loss while the critic was
also trained.  Repository BCPPO already implements a serious Stage-1 pure-distillation path.  Keep
the same parameter-exact released Refiner and Tracker, synchronized current 890-D/510-D state,
official-scale residual, fixed `1e-5` LR and `0.05` action std, but hold every declared update in
Stage 1.  The Tracker remains a training-only current-state label source and the deployed actor is
unchanged.  A fresh seed171718 two-update / 384-transition gate must show a nonzero residual update,
exact-zero critic/std and released-expert drift, finite live Newton tensors and no divergence.
Only that machine pass admits one fresh seed171719 eight-world, 64-update endpoint and the same
fixed-20 physical gate.  There is no stage-length, loss-coefficient, LR, residual-limit, reward or
update-budget sweep, and downstream Tracker/BCPPO remains forbidden below `16/20` lift and strict.

The pure-distillation pre-training diagnostic is complete but does not pass its predeclared gate.
Fresh seed171718 runs two official BCPPO Stage-1 updates / 384 live Newton transitions with finite
parameters, fixed LR `1e-5`, residual-only actor delta `0.00046239`, exact-zero critic drift and
action-std delta `7.45e-10` from `0.05`.  The distillation loss moves `0.2150 -> 0.2052`.  Strict
evaluation of `model_1.pt` restores both released experts with exact-zero weight/std drift, proves
the Tracker is unused at inference, completes all 20 profiles without active divergence and records
the deployed composed action exactly.  Its `1/20` lift and `0/20` strict result is diagnostic-only.
However, training records one synchronized divergence (`1/384 = 0.2604%`) while the predeclared
short gate required zero; the general `0.5%` training ceiling cannot retroactively weaken that
stricter admission rule.  An automatic chain incorrectly began seed171719 under the general ceiling;
it was stopped with `Ctrl+C` immediately after update 0 / 192 transitions, produced no
`TRAINING_RESULT.json`, is not a formal endpoint and must never be resumed or reported.  The valid
decision is rejection before formal training.  Do not rerun this pure-distillation objective with
a new seed or proceed to Tracker/BCPPO from it.

The next bounded topology combines two independently audited serious components rather than
repeating either rejected controller.  Keep the parameter-exact official Refiner inside the
admitted six-layer, 384-D, past-`10 x 890` causal temporal composer, and use the parameter-exact
released Tracker only as a training-time current-state action teacher from the synchronized 510-D
observation.  The deployed actor never queries the Tracker; future states, rewards and outcomes do
not enter its input.  Use repository BCPPO Stage-1 pure distillation so the causal composer receives
the action signal without PPO/critic gradient conflict.  First require an exact-zero endpoint and
expert drift audit, a fresh seed171720 zero-optimizer 48-transition live gate, then a fresh
seed171721 two-update / 384-transition gate with zero divergence, nonzero composer update and
exact-zero critic/std/expert drift.  Only all machine passes admit one fresh seed171722 64-update
endpoint and the unchanged fixed-20 gate.  Do not sweep history, Transformer size, teacher/loss
weight, LR, residual limit or update budget; Tracker/BCPPO remains closed below `16/20` lift and
strict.

The combined topology has now passed every pre-training gate.  It has 13,034,614 total parameters,
of which only the 11,360,286-parameter temporal composer is trainable.  Seed171720 completes 48
live Newton transitions with zero optimizer updates, zero divergence, finite tensors, exact-zero
initial endpoint delta and exact-zero policy-state and released-expert drift.  Seed171721 completes
the prescribed two Stage-1 updates / 384 transitions with zero divergence, composer delta
`0.00045949`, exact-zero critic drift, action-std delta `7.45e-10` and fixed LR `1e-5`; distillation
loss decreases from `0.1205` to `0.1103`.  Strict fixed-20 evaluation loads the combined checkpoint,
keeps both official experts parameter-exact, proves the Tracker is unused at inference and finishes
all profiles without active divergence.  Its diagnostic physical result is `0/20` lift and `0/20`
strict, with mean lift/contact `0.009796 m/0.192179`; this short endpoint is not the physical
selection point.  The machine gates therefore admit exactly the predeclared fresh seed171722
64-update endpoint, which is now running on the retained H200 allocation.  Its fixed-20 result alone
will decide whether the acting-Refiner queue reaches downstream Tracker/BCPPO.

The fixed seed171722 endpoint is now complete and rejected.  Training finishes all 64 updates /
12,288 transitions with one synchronized divergence (`0.00814%`), finite parameters, composer
delta `0.0162462`, exact-zero critic drift, action-std delta `7.45e-10` and fixed LR `1e-5`.
Frozen fixed-20 evaluation restores `model_63.pt` (SHA256
`141af2196e3e7335f58bd4626f7578ac232b78db7e064d501e640a7f1441826f`) and completes all profiles
without active divergence.  The official Refiner and training-only Tracker retain exact-zero
weight/std drift, and the Tracker is not called at inference.  Physical behavior improves from the
short endpoint's `0/20` lift to `8/20`, with mean peak lift `0.071140 m`, but remains `0/20` strict
against the fixed `16/20` requirements.  Mean bilateral-contact fraction is only `0.128391`, expert
retention falls to `0.822457`, mean bounded residual is `0.204223` and composed-endpoint delta is
`0.175667`.  This is meaningful action-teacher response, not an admitted acting Refiner.  Keep
downstream Tracker/BCPPO closed and do not sweep this topology's scale, capacity or budget.  The
next bounded diagnostic must change how the exact endpoint action is composed or retained, because
additional magnitude in the same unconstrained temporal residual is not justified by zero strict
completions and declining bilateral hold.

The next diagnostic removes that non-identifiable composition degree of freedom without changing
the serious components, data or objective.  The exact official Refiner remains fully present in
every deployed action; the same admitted six-layer 384-D past-`10 x 890` causal Transformer emits
only a bounded 29-D additive correction, and its exact-zero head makes initialization bitwise equal
to the released Refiner.  The released Tracker remains a parameter-exact, training-only
current-state label source and repository BCPPO stays entirely in Stage 1.  This tests whether the
lost endpoint retention, rather than missing action magnitude, caused the `8/20` lift but `0/20`
strict result.  First run seed171723 for 48 zero-optimizer live transitions, then seed171724 for
exactly two updates / 384 transitions with zero divergence, nonzero temporal update, exact-zero
critic/std/expert drift and a strict loading smoke.  Only all machine passes admit one fresh
seed171725 64-update endpoint and the unchanged fixed-20 physical gate.  Do not sweep residual
limit, history, Transformer size, teacher weight, LR or update budget.

The additive implementation and both short gates now pass.  Its component audit gives a 29-D head,
exact-zero endpoint delta, exact unit Refiner retention, zero Refiner gradient and composer gradient
`0.25147`.  Fresh seed171723 completes 48 live zero-optimizer transitions with finite tensors, zero
divergence and exact-zero policy-state change.  Fresh seed171724 completes two updates / 384
transitions with zero divergence, composer delta `0.00046440`, exact-zero critic drift, action-std
delta `7.45e-10` and fixed LR `1e-5`; distillation loss decreases `0.3349 -> 0.1788`.  A strict
one-profile checkpoint smoke restores both experts exactly, verifies endpoint-plus-residual action,
never calls the Tracker at inference and finishes without divergence.  Its `0/1` lift/strict result
is diagnostic-only.  The per-step retention tensor is constructed as exact ones; its accumulated
float32 mean is `0.99999994`, so the machine gate uses the structural identity check plus `1e-6`
reporting tolerance rather than an invalid exact equality on the reduction.  These passes admit the
single predeclared fresh seed171725 64-update run, now active on H200 with an automatic fixed-20
evaluator behind its unchanged numerical gate.

The seed171725 endpoint is complete and rejected.  Training passes all 64 updates / 12,288
transitions with two synchronized divergences (`0.01628%`), finite parameters, composer delta
`0.0145992`, exact-zero critic drift, action-std delta `7.45e-10` and fixed LR `1e-5`.  Frozen
fixed-20 evaluation restores `model_63.pt` (SHA256
`07a86510547dd2c967bdd19edd2d7452f7fff6d5d3d172c3d97f1786ec2e8a4f`), keeps both experts exact,
never calls the Tracker at inference and finishes all profiles without active divergence.  Exact
additive retention improves lift from `8/20` to `10/20` and bilateral contact from `0.128391` to
`0.170408`, but strict completion remains `0/20`; mean peak lift is `0.058460 m`.  All ten lifted
profiles terminate on `obj_pos`, and the full failure distribution is `obj_pos=13`, `ee_pos=6`,
`obj_ori=1`.  This is a real topology improvement but remains below both fixed `16/20` gates.  Do
not sweep this objective, scale, capacity or budget.

The next bounded topology addresses a remaining conditioning omission.  The released Tracker's
510-D actor input begins with an exact current 36-D reference command: 29 joint targets, root linear
and angular velocity, and contact.  The 890-D Refiner input contains the joint reference but not the
complete root-velocity/contact tuple, so current-state Tracker action labels are not fully
conditioned in the deployed student.  Add this exact current 36-D command to the same additive
six-layer causal composer, preserve the complete official Refiner endpoint and keep the Tracker
training-only.  This is a deployable selected-reference command, not a future state, reward or
outcome label.  First prove a non-mutating Newton command builder, elementwise synchronization with
the first 36 Tracker-observation coordinates, exact-zero initialized action delta and exact-zero
expert drift; only then predeclare fresh zero-optimizer and two-update live gates.

The command-builder prerequisite now passes on two live H200 worlds.  Calling
`tracker_command()` changes neither robot state nor any five-frame observation history (both maximum
deltas are exactly zero), returns finite `(2, 36)` tensors, and matches `observe()[:, :36]`
elementwise both immediately after reset and after one real Newton transition.  The remaining work
is therefore the command-conditioned composer wiring and its exact-zero component audit, not
reference-command reconstruction.

That component wiring now passes its pre-training CUDA gate at seed171726.  The deployed policy
input is exactly `current 890 + past 10 x 890 + current command 36 = 9826` coordinates.  The command
is projected as its own token beside the CLS token and ten causal history tokens in the same
six-layer, 384-D Transformer; no full 510-D Tracker state, future state, reward or outcome enters
the actor.  The 29-D output head is exact zero, initialized composed/endpoint delta is exactly zero,
retention is exactly one, official Refiner gradient is zero and composer gradient is `0.24515`.
Changing only the command changes the Transformer CLS representation by `0.09760`, so the new token
is functionally connected even though the zero head preserves the endpoint.

This gate predeclares one bounded sequence with no sweep.  Fresh seed171727 runs 48 live Newton
transitions and zero optimizer updates; it must retain exact state/parameter invariants, zero
divergence and elementwise command/Tracker-prefix synchronization.  Only that pass admits fresh
seed171728 for exactly two Stage-1 pure-distillation updates / 384 transitions, which must have zero
divergence, a nonzero composer update, exact-zero critic/std/expert drift (std tolerance `1e-7` around
`0.05`), fixed LR `1e-5`, finite tensors and strict checkpoint loading.  Only both short passes admit
one fresh seed171729 eight-world, 64-update endpoint and the unchanged fixed-20 physical gate.  The
acting Refiner still requires at least `16/20` lift and `16/20` strict completion before downstream
Tracker/BCPPO; command layout, Transformer size/history, loss, LR, residual limit and budget are not
tuning axes.

Both short gates now pass.  Seed171727 runs the prescribed 48 live transitions with zero optimizer
updates, zero divergence, finite tensors and exact-zero policy-state change.  Its live policy is
9826-D; history/current, appended-command/Tracker-prefix and composition-command deltas are all
exactly zero, while the command-token span across two worlds is `2.78275`.  Seed171728 runs exactly
two Stage-1 updates / 384 transitions with zero divergence, composer delta `0.00046290`, exact-zero
critic drift, action-std delta `7.45e-10`, fixed LR `1e-5` and finite parameters.  Strict one-profile
evaluation restores `model_1.pt` (SHA256
`e1dddcfb8fef26169394f323c1e3acde4ba251b31db4fd6167615c48ff8eb214`), keeps both official experts
parameter-exact, proves the Tracker is unused at inference, verifies the current-command and deployed
action contracts, and finishes without active divergence.  Its `0/1` lift/strict physical result is
short-checkpoint diagnosis only.  These machine passes admit exactly the predeclared fresh
seed171729 64-update endpoint and unchanged fixed-20 gate.

Seed171729 is now complete and rejected.  Its numerical training gate passes all 64 updates / 12,288
transitions with zero divergence, composer delta `0.0162573`, exact-zero critic drift, action-std
delta `7.45e-10`, fixed LR `1e-5` and finite parameters.  Frozen fixed-20 restores `model_63.pt`
(SHA256 `a7756a48137d2f6bfb069ef636d608bc8956fbba5ea8d5b2e8addf79014678e1`), preserves exact Refiner,
Tracker and current-command contracts, and finishes 20/20 profiles without divergence.  Physical
performance nevertheless regresses to `7/20` lift and `0/20` strict, with mean lift/contact
`0.044066 m/0.122728`, versus the unconditioned additive endpoint's `10/20` and
`0.058460 m/0.170408`.  Failure counts shift from `ee_pos=6, obj_pos=13, obj_ori=1` to
`ee_pos=11, obj_pos=9`, so the command token weakens bilateral hand tracking rather than repairing
object completion.  Do not sweep this command topology; downstream Tracker/BCPPO remains closed.

The next bounded action is a parameter-free teacher-ceiling audit, not another learned residual.
Every failed distillation topology uses the exact released CarryBox Tracker as its current-state
action label, but that exact 510-D actor has not yet faced this same fixed-20 raw-motion Newton gate.
Strictly load the official `510 -> 512/256/128 -> 29` Tracker, drive the same Newton resets through
the environment's once-per-step causal `observe()` tensor, and apply the unchanged `16/20` lift and
strict rules.  No optimizer, future state, physics change, threshold change or checkpoint sweep is
allowed.  If the Tracker ceiling fails, its action labels cannot justify further distillation; if it
passes, the next serious controller must address student closed-loop/expert composition rather than
teacher scale or command encoding.

The exact released Tracker ceiling is now complete and negative.  The official
`510 -> 512/256/128 -> 29` actor (SHA256
`ea406c98f622588e8537c14240c3101c4c3c75a333e74572d63b81f3fc391aa2`) completes all
twenty profiles with finite state/action tensors and no active divergence, but reaches only `2/20`
five-centimetre lifts and `0/20` strict completions.  Mean peak lift is `0.019740 m`, mean bilateral
contact is `0.054595`, and failures are `ee_pos=18`, `obj_pos=2`.  The exact label source is therefore
far below the unchanged `16/20` gates; the causal students' `7--10/20` lift came from retaining the
official Refiner, not from an admissible Tracker ceiling.  Stop Tracker action distillation and do not
sweep its checkpoint, command encoding, student capacity, residual limit or update budget.

The next bounded diagnosis changes the training objective rather than the official components.  It
keeps the parameter-exact released Refiner inside the admitted six-layer, 384-D, past-`10 x 890`
additive causal Transformer, starts at exact endpoint equality, and uses repository BCPPO/PPO with the
same fixed `1e-5` learning rate and `0.05` exploration standard deviation.  The actor remains exactly
9790-D.  A training-only physical-recovery score reads only the just-executed Newton transition:
object and maximum end-effector margins normalized by their existing `0.30 m` termination thresholds,
real bilateral hand-box contact, bilateral lift normalized by the fixed `0.05 m` gate, and current
failure.  It is blended equally with the unchanged official reward normalized by its documented
`5.125` positive scale.  No reward, contact label, future frame or outcome is appended to actor or
critic observations.

This objective first faces a fixed-context learnability diagnostic on frame-zero `data_000`, where the
exact Refiner currently lifts only `0.008450 m` and terminates on `obj_pos` at step 235.  Fresh
seed171731 passes the zero-optimizer H200 gate over 48 real transitions: all 48 reward calls occur,
maximum absolute combined reward is `0.711533`, every tensor is finite, divergence is zero and policy
state change is exact zero.  Fresh seed171732 is the single predeclared two-update gate; it must show
finite nonzero actor/critic learning, zero divergence, exact frozen-expert retention, all 384 reward
calls and strict checkpoint loading.  Only that pass admits one fresh fixed-context learnability
endpoint.  Task-wide training is allowed only if that endpoint both improves over `0.008450 m` and
reaches trajectory timeout without strict failure.  Objective weights, capacity, history, residual
limit, learning rate and update budget are not sweep axes.  The admitted fixed-context endpoint is
exactly 64 fresh updates; it is not an open budget axis.

The two-update gate and strict loading audit now pass.  Seed171732 completes all 384 transitions with
zero divergence, 384 physical-reward calls, maximum absolute reward `0.711787`, finite temporal-actor
and critic changes `0.0002143/0.0004684`, fixed learning rate `1e-5` and action std `0.0500774`.
Strict evaluation restores `model_1.pt` (SHA256
`0d67f9b29192295670d81ff6a42ef290d42a305bc8de116d5abede255bd3bd48`), keeps the official
Refiner parameter-exact and records only `0.001170` mean absolute correction.  Its `0.008810 m` peak
lift and strict failure are short-checkpoint diagnostics, not the selection point.  These numerical
and structural passes admit exactly one fresh seed171733, eight-world, frame-zero `data_000`,
64-update learnability endpoint followed by the same deterministic one-profile physical gate.

The fixed-context learnability result is negative.  Seed171733 finishes 64 updates / 12,288
transitions with 12,288 physical-reward calls, zero divergence, finite parameters, temporal-actor and
critic changes `0.003044/0.013657`, final std `0.050591` and maximum absolute combined reward
`0.794655`.  Frozen `data_000` evaluation restores `model_63.pt` (SHA256
`56a80b6423bfc7c333f304f010b72c81542e497f295de0016c555f011e8e2081`) and keeps the
official Refiner exact, but peak lift regresses from `0.008450` to `0.006923 m`, bilateral-contact
fraction from `0.2000` to `0.1872`, and both endpoints fail `obj_pos` at step 235.  The fixed-context
gate therefore rejects task-wide training.  Do not tune this objective, its equal blend, controller
capacity, history, residual limit, learning rate or update budget.

A parameter-free rerun also resolves the old claim that the released Tracker lifts `0.2--0.3 m` in
Newton.  The legacy single-world script was missing its derived actor asset; regenerating that asset
directly from the exact released checkpoint and running `data_000` at `mu=1.0` reproduces a late
`0.2442 m` lift over 400 frames.  It is not a contradiction with the strict ceiling.  The legacy
script has no termination logic: box height remains `0.207 m` at frame 200 and only rises by frame
250, while the formal exact-Tracker rollout has already failed `ee_pos` at frame 78 with
`0.001143 m` lift.  Its final object tracking error is `0.7718 m`.  The historical peak is
post-failure behavior and cannot admit an acting teacher.

Before another learned controller or reward, run one exact reference-joint-target ceiling.  At each
causal step convert the current reference joint pose through the same deployed Isaac action map,
`action = (q_ref - q_default) / action_scale`, and execute it in the unchanged fixed-20 Newton
environment.  This is not a learned or privileged replacement: it is a parameter-free feasibility
test of whether the supplied motion can be dynamically executed with the admitted PD gains, effort
limits, contact model and strict terminations.  Record action magnitude, lift, completion and failure
causes without clipping actions, changing physics or tuning thresholds.  A negative ceiling blocks
construction of another action teacher from this reference; a positive ceiling supports a serious
Newton-native inverse-dynamics/action-distillation route.

The exact reference-joint-target ceiling is complete and negative.  Across the fixed twenty profiles,
the inverse action map reconstructs every commanded joint target within `1.19e-7`, uses finite
unclipped actions (maximum magnitude `9.8803`) and produces finite Newton rollouts, but reaches
`0/20` lift and `0/20` strict completion.  Mean peak lift and bilateral contact are only
`0.002915 m/0.001515`; termination counts are `ee_pos=16`, `anchor_pos=5`, `anchor_ori=1` (some
profiles have multiple causes).  This does not prove that the complete reference is dynamically
impossible: it proves that reference joint poses alone are not a Newton action teacher, because they
do not supply the floating-base/contact corrections needed to realize that motion.  Do not distill
these actions.

The next fixed diagnostic moves learning to the actual failure frontier instead of diluting it over
the whole episode.  On frame-zero `data_000`, execute the parameter-exact official Refiner for 200
real Newton steps, preserve those observations in the admitted past-`10 x 890` history, and give the
sampled student zero PPO/value/entropy credit while its action is not executed.  Then hand off without
reset to the same additive six-layer temporal controller and train only the live post-prefix recovery
segment with the admitted causal physical-recovery objective plus the exact Refiner action anchor.
The deployed checkpoint still acts from frame zero; no prefix mask, reward, future state or outcome
enters its actor input.  First require a nine-horizon zero-optimizer execution/mask/expert audit and
one ten-update gate: with 24 steps per update, the first eight updates are necessarily still inside
the 200-step exact prefix and cannot diagnose student learning.  Only both passes admit one fresh
64-update fixed-context endpoint.  Prefix 200, objective
weights, model/history, residual limit, learning rate and budget are fixed rather than sweep axes.

The failure-frontier execution and optimization gates pass.  Fresh seed171734 executes nine
zero-optimizer horizons (`432` transitions): `400` exact-teacher and `32` student-control steps,
zero teacher/student execution error, `432/432` physical-reward calls, zero divergence and exactly
zero parameter change.  Fresh seed171735 then executes ten updates (`1,920` transitions): `1,640`
teacher and `280` student-control steps, exact execution in both segments, zero divergence and
nonzero actor/critic/std changes of `6.94e-4/7.50e-4/1.37e-4`.  Its frame-zero frozen diagnostic
improves the one-profile peak lift from the exact Refiner's `0.00845 m` to `0.01269 m`, with
`0.20851` bilateral-contact fraction, but remains below the `0.05 m` lift gate and has `0/1` strict
completion.  This is a valid learning-direction diagnostic, not a successful endpoint.  The one
predeclared fresh seed171736 64-update learnability run is now complete.  It passes the numerical
contract with `12,288/12,288` reward calls, `10,608/1,680` exact teacher/student execution steps,
zero execution error, zero divergence and nonzero actor/critic/std changes of
`0.00396/0.00579/0.000598`.  Frozen model63 remains negative from frame zero: peak lift is
`0.01223 m` and bilateral contact is `0.20851`, but the rollout still fails `obj_pos` at step 235,
exactly where the released Refiner fails, and therefore gives `0/1` lift and `0/1` strict completion.
The fixed failure-frontier objective improves small pre-failure lift but does not learn recovery.
Do not run a task-wide repeat, prefix/objective/budget sweep, Tracker or BCPPO from this checkpoint.

A final parameter-free deployment-topology audit rules out prefix-distribution contamination as the
explanation.  The same model63 is composed causally with the exact official Refiner for its first 200
steps, preserving the real temporal history, then executes 35 learned-recovery steps without reset.
The exact/student action counts are `200/35`, the handoff maximum action jump is only `0.00961`, and
all states remain finite.  Nevertheless it again fails `obj_pos` at step 235, reaches only
`0.01072 m` peak lift and has `0/1` strict completion.  The trained recovery fails even on its own
handoff distribution.  Reject this objective/topology and do not tune the prefix, reward or budget;
the next training method must change the Newton action target/controller rather than mask more of the
same released-Refiner rollout.

The next bounded action-objective diagnostic removes the failed Isaac-domain Refiner action anchor
only after the same exact prefix.  It keeps the parameter-exact official Refiner embedded, the fixed
200-step physical prefix, past-`10 x 890` six-layer temporal controller, residual limit, causal
physical reward, LR and budget unchanged, but sets post-handoff distillation weight to exactly zero
so Newton physics alone can select the recovery correction.  This is one Newton-native free-residual
test, not an anchor-weight sweep.  Require fresh seed171737 zero-optimizer and seed171738 ten-update
execution/learning gates before the single fresh seed171739 64-update endpoint; evaluate model63
first with the exact-prefix composition and then from frame zero only if recovery crosses step235.
No task-wide or downstream training is admitted without the existing `16/20` lift and strict gates.

The Newton-native free-residual diagnostic is complete and negative.  Seed171737 passes the
zero-optimizer contract with `400/32` teacher/student actions, `432/432` reward calls and exactly
zero parameter/execution error.  Seed171738 passes ten updates with `1,640/280` teacher/student
actions, zero divergence and nonzero actor/critic/std changes.  Fresh seed171739 then passes the
64-update numerical gate with `10,608/1,680` teacher/student actions, zero execution error,
`12,288/12,288` reward calls and actor/critic/std changes of
`0.00148/0.00682/0.000193`.  Exact-prefix frozen evaluation nevertheless executes only 35 recovery
steps before the same step235 `obj_pos` failure: peak lift is `0.01045 m`, bilateral contact is
`0.20426`, and the handoff action jump is `0.01576`.  Removing the action anchor increases the mean
residual to `0.00383` but does not recover.  The Refiner anchor is not the bottleneck; do not launch
frame-zero/task-wide evaluation or another anchor setting.
