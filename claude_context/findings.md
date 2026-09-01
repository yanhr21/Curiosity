# Verification findings

> **Scope: this audits the Plan 15 IsaacLab/PhysX line, not the current Newton line.**
> Every `Evidence` path below points into `scripts/sugar/`, `SUGAR/` or vendored `IsaacLab/`.
> It is nonetheless **required reading before touching any tactile channel or reward term**,
> because it is the reason Plan 16 exists and because most of these defects are portable — a
> naive port of `hoi_contact` or `undesired_contacts` reproduces two of them exactly. See
> `../README.md` § Rules that carry over for the short version, and `../AGENTS.md`
> § Audit addendum for the standing rules it produced.
>
> None of the defects here has been fixed *in the Plan 15 code*; Plan 16 rebuilds instead.

Everything found while building and auditing this page: claims checked against the
source they describe, plus anything tripped over on the way. Nothing is hand-waved
— each entry names the file and lines that settle it.

**Format** — the page parses `###` headings, so keep them exact:
`F-NNNN · VERDICT · variant · module`, then `Claim (page)` / `Reality` /
`Evidence` / `Action`. Verdicts: `CONFIRMED` (page is right) · `CORRECTED` (page
was wrong, now fixed) · `REFUTED` (wrong and unfixable from evidence, so the page
now says nothing) · `UNVERIFIED` (needs a run/checkpoint/dataset) · `OPEN` (gap
found, not yet closed).

Every entry below was read directly out of the source named in its `Evidence`
line. None has yet been checked by an independent auditor, and none has been
fixed — this round is a correctness audit of the experiment, not of the page.

---

## Round 1 — correctness audit of Plan 15 (2026-08-19)

Prompted by the null result: P shows no gain over Z, and PS is significantly
*worse* than P at 3× (paired hold difference `-0.2712`, 95% CI
`[-0.4655, -0.0667]`). Information reliably making a policy worse is the
signature of a wiring bug, so the sensing and training paths were read end to
end.

### F-0001 · OPEN · P, PS · patch tactile reduction
**Claim (plan/README):** the nine-channel patch record lets the policy observe the grasp's friction margin, and the friction sweep tests generalization to new friction.
**Reality:** `friction_utilization = ‖Σ shear‖ / (mu · Σ|F_n| + ε)` uses `mu` from the **TacSL sensor config**, pinned to `0.5` by `CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT`. TacSL has already capped the shear numerator with that same constant (`ft_norm = min(k_t·|v_t|, mu·|f_c|)`). Numerator and denominator therefore share the constant, and the channel is **mathematically invariant to the object's PhysX material friction**. The friction sweep varies the box's static/dynamic friction from 0.5 to 2.0 while every tactile channel keeps its definition unchanged.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_patch_tactile.py:144-146` · `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.py:1005-1007` · `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:155`
**Action:** none yet. The friction-generalization claim cannot be supported until utilization is computed against the actual contact material. Blocks any "tactile helps with new friction" result.

### F-0002 · OPEN · P, PS · patch tactile reduction
**Claim (plan §3):** signed XY shear is a physical grasp-load channel available to the policy each control step.
**Reality:** TacSL's tangential force is pure velocity damping with no tangential elasticity — `ft_world = -min(k_t·|v_t|, mu·|f_c|) · v̂_t`. Under stick (`|v_t| ≈ 0`) the shear goes to zero **regardless of how much tangential load the grasp is actually carrying**. Channels 4, 5, 6 and all three slip channels are therefore near-zero exactly during a successful hold, and only become informative once sliding is already underway. In practice "tactile" reduces to contact + normal load + pressure, i.e. a load cell.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:1005-1011`
**Action:** none yet. A tangential spring with stick memory is the physically correct fix and would live in vendored IsaacLab. Consistent with the repo's own evidence that mass is separable from the *continuous* channels in ~13 frames while slip needed a deliberately sliding calibration trace.

### F-0003 · OPEN · P, PS · patch tactile reduction
**Claim (plan §3):** `normal_load_n` and `signed_shear` are the patch's normal and tangential load.
**Reality:** TacSL projects the **total** contact force (`fc_world + ft_world`, normal penalty plus friction) into the taxel frame and labels local-z "normal" and local-xy "shear". The SDF contact normal is not aligned with the taxel frame's z-axis on curved anatomical patches, so a geometric slice of the *normal penalty force* lands in the shear channels. As a result `‖shear‖ / (mu · normal_z)` is **not** bounded by 1 and can diverge when the taxel frame is tilted.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:1013-1024`
**Action:** none yet. Resolve normal/tangential in the contact frame (the SDF normal is already computed as `normals_local`) rather than the taxel frame.

### F-0004 · OPEN · P, PS · channel scales
**Claim (plan §3):** common channel scales are frozen from a live leakage sweep and normalize each channel to a comparable range.
**Reality:** each scale is the 99.5th percentile of **nonzero magnitudes** over the sweep. Combined with F-0003, `friction_utilization` has no upper bound and no minimum-load gate — `contact` is `any(penetration > 0)`, which can be true while `normal_load ≈ 0`, giving `‖shear‖ / (0.5·0 + 1e-8)`. A small number of blow-up samples inflates `scales[5]`, and after `patches / channel_scales` the informative range of the channel is compressed toward zero.
**Evidence:** `scripts/sugar/native_tactile/fit_online_patch_channel_scales.py:26-32,55-67` · `online_patch_tactile.py:120,144`
**Action:** none yet. Gate the ratio on a minimum normal load before it is formed, and re-fit the scales. Note the unit tests use well-behaved synthetic data and cannot catch this (`tests/native_tactile/test_online_patch_channel_scales.py:22-33`).

### F-0005 · OPEN · PS · slip detector
**Claim (plan §7):** PS adds causal slip evidence on top of P.
**Reality:** the detector's primary incipient trigger is `friction_utilization >= 0.60`. Given F-0001/F-0003/F-0004 that input is a sensor-internal, geometry-contaminated, badly-scaled quantity, so `INCIPIENT` fires on light and tilted contacts rather than on slip. This is a coherent mechanism for **PS being worse than P** rather than merely no better: it injects three noisy channels into the actor.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/patch_slip.py:236-240,255-259`
**Action:** none yet. Depends on F-0001–F-0004 being fixed first; re-calibrate thresholds afterwards.

### F-0006 · OPEN · Z, P, PS · BCPPO
**Claim (plan §5):** teacher-prefix transitions do not enter PPO surrogate/value/entropy credit.
**Reality:** true for those three terms, which are all weighted by `active_weight` from `training_handoff_mask`. But `_reduce_distill_loss` returns a plain `per_sample_loss.mean()` — **the distillation loss is not masked**. It carries weight 1.0 through stages 1–2 and a floor of 0.25 in stage 3, so the dominant gradient is computed largely over teacher-driven *pickup* transitions rather than post-handoff load response.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py:77-82` (unmasked) vs `:255-282` (masked)
**Action:** none yet. Decide whether this is intended; if it is, say so in the plan, because the asymmetry currently reads as an oversight.

### F-0007 · OPEN · Z, P, PS · BCPPO
**Claim (plan §5):** the privileged critic is trained on the same transitions PPO credits.
**Reality:** `value_loss` is masked by `active_weight`, but GAE returns and advantages are computed by stock rsl_rl over the whole `num_steps_per_env = 24` window. A window that straddles the handoff bootstraps post-handoff advantages off value estimates for pre-handoff states the critic was never fit on.
**Evidence:** `rsl_rl_bcppo.py:259-282` · `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/agents/rsl_rl_bcppo_cfg.py:15`
**Action:** none yet. Bounded in frequency (one straddling window per episode per env) but it lands exactly on the transition of interest.

### F-0008 · OPEN · PS · slip detector
**Claim (plan §4, AGENTS §3):** the slip callable reads only current and past 54-patch signals; relative contact velocity is an evaluation label only.
**Reality:** violated in substance. TacSL derives the tangential force from `relative_velocity_world` between the taxel and the object's closest surface point, so `shear_xy_n` and `friction_utilization` — two of the detector's six inputs — are direct functions of the simulator's relative contact velocity. The detector is reading a laundered velocity oracle.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:988-1006` → `online_patch_tactile.py:498-506`
**Action:** none yet. This makes PS look *stronger* than a real tactile system would, not weaker, so it does not explain the null result — but it does mean the current PS is not the deployable quantity the plan claims.

### F-0009 · OPEN · P, PS · patch encoder
**Claim (plan §5):** a 3-layer pre-LN Transformer produces the 128-D patch embedding.
**Reality:** `nn.TransformerEncoder(layer, num_layers=3)` is constructed with `norm_first=True` and no `norm=` argument, so the stack has **no terminal LayerNorm**. A pre-LN stack leaves its residual stream unnormalized without one, so the embedding's scale relative to the 504-D base is uncontrolled. The `warm_start_tactile_gain = 0.01` on the first-layer patch columns masks this at initialization only.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/patch_tactile_encoder.py:82-92`
**Action:** none yet. Low severity next to F-0001–F-0005, but it is a free fix.

### F-0010 · OPEN · P, PS · patch encoder
**Claim (page/plan):** Z, P and PS share one encoder and differ only in which channels are live.
**Reality:** the encoder's attention mask is `patches.abs().amax(-1) > 0`, so "all nine channels exactly zero" means *padding*. A patch that is genuinely not in contact is masked out, and the policy cannot distinguish "no contact here" from "no such patch". It also creates a P/PS asymmetry: a patch that has just lost contact gets `slip_score = 1.0` from the `contact_loss` branch, so in PS the token stays active where the identical physical state is masked in P.
**Evidence:** `patch_tactile_encoder.py:126,140-150` · `patch_slip.py:222-232`
**Action:** none yet. Consider an explicit presence channel rather than overloading all-zero.

### F-0011 · OPEN · Z, P, PS · config plumbing
**Claim (page/plan):** the frozen channel scales are supplied per run by `--patch-scale-file`.
**Reality:** true, but fragile in a way worth recording. `patch_channel_scales: list[float] = _patch_channel_scales()` is a **class-body default evaluated at import time**, so it works only because the launcher sets `SUGAR_ONLINE_PATCH_CHANNEL_SCALES` before importing the agent-cfg module. It fails loudly today (nine `NaN`s → the actor raises), but any change of that sentinel to a plausible default turns the failure silent.
**Evidence:** `rsl_rl_online_patch_mass_bcppo_cfg.py:19-33,44` · `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:128-151`
**Action:** none yet. Read the scales inside `__post_init__` instead of at class-body evaluation.

### F-0012 · CONFIRMED · Z, P, PS · mass jump
**Claim (plan §2):** mass and inertia are written between two actor calls and read back.
**Reality:** correct. `_write_mass` scales the default inertia tensor by exactly `target_mass / default_mass` — the right transformation for a density change at fixed geometry — writes both, reads both back, and raises on any mismatch beyond `rtol=1e-6`. `apply_pending` is called after actor inference and before the physics substeps.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_mass_jump.py:122-150,215-241`
**Action:** none. This is the part of the experiment that is not in question.

### F-0013 · CONFIRMED · Z, P, PS · warm start
**Claim (plan §5):** all three branches start from the same official Tracker policy, and Z with zero patch input is exactly that policy.
**Reality:** correct and self-checking. `_audit_tracker_zero_patch_equivalence` builds the 510-D source and 632-D target forward passes and asserts the zero-patch embedding is exactly `0.0` and both actor and critic agree to `2e-6`, raising otherwise. The 510→504 column remap deliberately drops the contact label (35) and measured object pose (501:510).
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/online_patch_tactile_actor_critic.py:139-176,196-250`
**Action:** none.

### F-0014 · OPEN · Z · experiment design
**Claim (plan §5):** Z is the matched no-tactile control.
**Reality:** Z's observation term never touches `env.scene.sensors` at all. That is correct for the contract, but it means **any TacSL misconfiguration degrades P and PS while leaving Z untouched** — and the entire result is a P/PS-versus-Z comparison. A sensing bug and a genuine "tactile does not help" are indistinguishable from the outcome numbers alone.
**Evidence:** `online_patch_tactile.py:523-548`
**Action:** none yet. Given F-0001–F-0005, the per-channel variance and mass/friction mutual information during the *hold* phase should be measured on saved traces before any retraining is scheduled.

---

## Round 2 — structure (2026-08-19)

Three `site-auditor` agents (Opus, xhigh, read-only), one per branch, each given the
diagram's `<title>` node list and edge labels verbatim and asked to reconcile them
against the execution path for its own gym task id. IDs keep each auditor's own
block (`F-1xxx` PS, `F-2xxx` P, `F-3xxx` Z) so every entry stays traceable to the
agent that produced it.

**Headline: the Pass-1 draft diagram was wrong in load-bearing ways.** It spliced the
training loop and the frozen evaluator into one spine, drew the handoff as gating the
tactile stack and actor (it does not — they run from frame 0), pointed the slip node
the wrong way, and omitted the reward manager, the privileged critic, the channel-scale
normalization, the action pipeline and a second frozen Refiner. All corrections below
have been applied to the three diagrams unless the entry says otherwise.

### F-1002 · CORRECTED · PS · MISSING — reward manager and termination manager
**Claim (page):** The spine is Env → Teach → Hand → Tacsl → Enc → Actor → Jump → Out. `Out` is "hold / drop / fall".
**Reality:** Every control step runs `termination_manager.compute()` (6 terms: `trajectory_complete`, `anchor_ori`, `ee_body_pos`, `obj_pos`, `obj_ori`, `anchor_pos`) and `reward_manager.compute(dt)` (21 terms: 6 regularization + `undesired_contacts`, 11 motion-tracking, `obj2body_pos/ori`, `hoi_contact`). The reward is what PPO's surrogate optimizes from update 1000 onward, and the terminations are what actually end the episode — including `obj_pos`/`obj_ori`, which fire on reference deviation when the box is dropped. Neither appears as a node, an edge, or a panel line anywhere in `psflow`. The diagram therefore shows an RL pipeline with no reward and no episode boundary.
**Evidence:** `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:205,209`; `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/base_refiner_env_cfg.py:299-399,404-455`; `.../carry_box_refiner_env_cfg.py:84-116`; `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py:250-257,336-345`
**Action:** APPLIED — added an `Ev` node ("post-physics · reward 21 · term 6 · reset") to all three spines, and stated in the actor panel that the PPO surrogate is driven by the unchanged official SUGAR tracking reward, not by any hold/drop metric.

### F-1003 · CORRECTED · PS · MISSING — frozen channel-scale normalization
**Claim (page):** Edge `Tacsl → Enc` is labelled `[4,2,27,9]`; the `Enc` panel describes the encoder as "projected `9→128` by a bias-free linear".
**Reality:** Between the observation and the projection the encoder divides every channel by a frozen 9-vector: `normalized = patches / self.channel_scales`. Those scales are not learned and not computed online — they are read from `SUGAR_ONLINE_PATCH_CHANNEL_SCALES` in a class-body default evaluated at import time and fitted offline by `fit_online_patch_channel_scales.py` as the 0.995 quantile of nonzero magnitudes over saved sweep traces (channel 0 = 1.0, channels 3 and 4 share one shear scale, channels 7 and 8 = 1.0). This is the single stage that sets the numeric scale of every tactile channel entering the policy, and it is exactly the stage F-0004 claims is broken. It has no node and is not named in any panel.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/patch_tactile_encoder.py:54-69,138-140`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:18-32,44`; `scripts/sugar/native_tactile/fit_online_patch_channel_scales.py:26-69`
**Action:** APPLIED — the `Enc` node now leads with "÷ frozen patch_channel_scales (9)" and the panel records that the division precedes the bias-free projection and is fixed for the whole run.

### F-1004 · CORRECTED · PS · MISSING — action post-processing and the PD loop
**Claim (page):** Edge `Actor → Jump` is labelled "29-D action"; nothing appears between the actor and the physics.
**Reality:** The 29-D value the actor produces is a *raw* action. `OnlineMassJumpJointPositionAction.process_actions` calls `super().process_actions`, which computes `processed = raw * scale + offset`, where `scale` is the per-joint `0.25 * effort_limit / stiffness` dict and `offset` is `default_joint_pos` (`use_default_offset=True`). The processed joint target is then written as a PD position target once per physics substep — `decimation = 4` at `sim.dt = 0.005`, i.e. one 50 Hz action held across four 200 Hz PhysX steps.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_mass_jump_action.py:16-21`; `IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:170-179,181-199`; `.../actions_cfg.py:43-58`; `SUGAR/source/sugar_rl/sugar_rl/assets/robots/unitree.py:291-302`; `.../carry_box_refiner_env_cfg.py:134-143`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:183-193`
**Action:** APPLIED — added an `Act` node: "×joint scale + default pos · mass write · 4 PD substeps @ 200 Hz".

### F-1005 · CORRECTED · PS · MISSING — exploration noise; the executed action is a sample
**Claim (page):** `Actor` = "504-D base ⊕ 128-D → 512/256/128", edge out "29-D action".
**Reality:** During rollout the executed action is `Normal(mean, std).sample()`, not the MLP output. `std` is a learned parameter (`init_noise_std=0.5`, but overwritten by the Tracker checkpoint's own `std` at warm start) and is trained. The evaluator uses `get_inference_policy` → `act_inference`, the deterministic mean — so the training and evaluation action paths differ and the page showed only one.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/tactile_actor_critic.py:270-273`; `.../reference_only_tactile_actor_critic.py:159-169`; `.../online_patch_tactile_actor_critic.py:112-122`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:63`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:468-470`
**Action:** APPLIED — the `Actor` node now reads "sampled · σ from warm start" and the panel states training executes a sample while frozen evaluation executes the mean.

### F-1006 · CORRECTED · PS · MISSING — the privileged critic head
**Claim (page):** The only network nodes are `Enc` and `Actor`; "890-D privileged" appears as the label of the `Teach → Hand` edge.
**Reality:** `OnlinePatchTactileActorCritic` also carries a second live head: an 890-D → 512/256/128 → 1 MLP with `critic_tactile_encoder = nn.Identity()` and **no tactile input at all**, constructed in the parent class and asserted at 890-D. It is evaluated on every stored transition in `BCPPO.update` and its value loss is the *only* PPO term active in stage 2 (updates 500–999).
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/reference_only_tactile_actor_critic.py:90-110`; `.../online_patch_tactile_actor_critic.py:74-77`; `.../rsl_rl_bcppo.py:205,259-278,324-330`
**Action:** APPLIED — folded into the `Loss` node ("BCPPO · 890-D critic · 2nd Refiner") and spelled out in the actor panel, including that the critic sees no tactile in any branch.

### F-1007 · CORRECTED · PS · MISSING — BCPPO holds a *second* frozen Refiner
**Claim (page):** One `Teach` node, "frozen official Refiner / drives the pickup, in-episode". No edge from `Teach` to `Loss`.
**Reality:** The frozen Refiner enters this path **twice, as two different objects**. (1) The acting teacher: `FrozenOfficialRefinerTeacher` inside the wrapper, which builds its *own* `ObservationManager` over `BaseObservationsCfg().policy` and runs the full official `ActorCritic`. (2) The distillation target: `BCPPO.__init__` reloads the same checkpoint into a bare `MLP` (actor weights only, plus `std`) and evaluates it on the env's `teacher` observation group inside `update()`. That second copy produces `distill_loss`, which carries weight 1.0 through updates 0–999 and never falls below 0.25 — i.e. it is the dominant gradient for the entire run.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/official_refiner_nominal_teacher.py:85-113,134-154`; `.../rsl_rl_bcppo.py:35-71,284-308,317-345`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:57-61,71-91`
**Action:** APPLIED — the `Loss` node names the second Refiner copy and the panel records that the acting teacher and the distillation teacher are separate instantiations reading two separately-computed 890-D observations.

### F-1008 · CORRECTED · PS · MISSING — the `training_handoff_mask` observation group
**Claim (page):** `Hand` feeds `Tacsl` down the spine; the handoff/loss coupling is described only in prose.
**Reality:** The env computes a fifth observation group, `training_handoff_mask`, every control step, exposing `handoff_active` as a `[B,1]` float. It is not in `obs_groups`, so it never reaches the actor; `BCPPO.update` reads it straight out of `storage.observations` and out of each minibatch to build `active_weight`, which masks surrogate/value/entropy and the adaptive-KL statistic. This is the only mechanism connecting the handoff to the loss.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_teacher_handoff.py:143-151`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:166-172,195-203`; `.../rsl_rl_bcppo.py:95-108,150-160,222,255-282`
**Action:** APPLIED — named in the actor/loss panel as the fifth observation group and the sole handoff→loss coupling.

### F-1009 · CORRECTED · PS · PHANTOM — `Out`, "80 frames" and "motion 45" are evaluator-only
**Claim (page):** Under the spine, `Jump —80 frames→ Out` ("hold / drop / fall"), and `Env —motion 45 · 50 Hz→ Teach`.
**Reality:** None of `hold`, `drop`, `fall`, the 80-frame window or motion 45 exists in the training path the column documents. They live only in `evaluate_online_patch_mass_bcppo.py`, a separate entrypoint with its own seeds (152014/5/6) and the `play` env cfg. `--motion-id` defaults to 45 *there*, forced by monkeypatching `command._sample_init_state` with `fixed_start`. In training, `start_init_env_ratio = 1.0` puts every env on the protected branch, where `motion_id[env] = env_id % num_motion` and `time_steps = 0` — with `num_envs = 4` the four envs run motions 0,1,2,3, never 45. A grep of `SUGAR/source/` for `hold_success|robot_fall|post_jump_window` returns nothing.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:41,45,321,347-348,479-487`; `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py:252-257,1046-1063`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:241-244,258-259`
**Action:** APPLIED — `Out` is now separated from the training spine by a dashed eval boundary and labelled "frozen evaluator", and the input edge reads "motions 0–3 (train) · 50 Hz". The train/eval generalisation gap is called out in the results section.

### F-1010 · CORRECTED · PS · ORDER — `Slip` is downstream of the tactile reduction, not an input to it
**Claim (page):** The `Slip` side node was drawn joined to `Tacsl` by a dashed edge, reading as an input to the tactile reduction.
**Reality:** The order is strictly the reverse. `online_patch_tactile_with_slip_actor_history` first calls `_patch_history`, which reads the 54 TacSL sensors and runs `reduce_patch_taxels` to produce the six base channels; only then does `_online_patch_slip_history` run, consuming the *reduced* patch features of the newest history frame — `current = base_history[:, -1]`. `PatchSlipDetector` never touches a sensor. So `Slip` is a *consumer* of channels 0–5 and a *producer* of channels 6–8.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_patch_tactile.py:436-468,505-525`; `.../patch_slip.py:148-284`
**Action:** APPLIED — the edge is now directed `Tacsl → Slip` and labelled "6 ch in → 3 ch out"; the panel records that the detector only ever sees the newest reduced frame, never the history and never a taxel.

### F-1011 · CORRECTED · PS · ORDER — `Jump` is half downstream of `Actor`
**Claim (page):** `Actor —29-D action→ Jump`, i.e. the mass event is a consequence of the actor.
**Reality:** Two operations at two different points, and only one is downstream of the actor. (1) *Scheduling* (`OnlineMassJumpController.advance`) runs as an `interval` event **after** physics, upstream of the observation the next actor call reads; its trigger is `newly_qualified = (~qualified) & handoff.handoff_active` — it reads the handoff controller, not the actor. (2) *The write* (`apply_pending`) runs inside `process_actions`, after the actor's forward and the teacher multiplexer, before the four physics substeps. The jump fires **once per episode per env**. The interval-event order is `step_teacher_handoff` then `step_mass_jump` by configclass declaration order, so the scheduler sees the same-step handoff state.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_mass_jump.py:176-241,296-321`; `.../online_mass_jump_action.py:16-21`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:223-236`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:174,235`; `IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:204-229,337-341`
**Action:** APPLIED — the write now lives on the `Act` node ("mass write, pre-substeps"); the scheduler lives on the post-physics `Ev` node. The undocumented declaration-order dependency is recorded in the panel.

### F-1012 · CORRECTED · PS · ORDER — `Hand` executes at two points, the second *below* `Actor`
**Claim (page):** `Hand` sat between `Teach` and `Tacsl`, upstream of the encoder and actor.
**Reality:** Two distinct operations. (1) Gate state — `controller.advance()` runs as an interval event post-physics, before the observation compute: upstream of the actor, as drawn. (2) Action selection — the substitution is a per-env multiplexer in `OnlineTeacherHandoffVecEnvWrapper.step`, which *receives* the student's already-sampled action and only then computes the teacher action and does `torch.where(teacher_control[:, None], teacher_action, actions)`. That is **downstream** of `Actor`. Consequently `Teach` is also mis-ordered: the frozen Refiner's forward runs *after* the student's within a control step, reading a separately constructed `ObservationManager`. There is no data dependency at all from `Hand` to `Tacsl`.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/utils/online_teacher_handoff_wrapper.py:44-56`; `.../online_teacher_handoff.py:62-89,126-140`; `.../official_refiner_nominal_teacher.py:85-99,134-154`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:235,238`; `.../online_patch_tactile.py:505-525`
**Action:** APPLIED — added a `Mux` node ("teacher / student select · torch.where(~handoff_active)") between `Actor` and `Act`, fed by `Teach`; the lift gate moved to the post-physics `Ev` node.

### F-1013 · CORRECTED · PS · BOUNDARY — `Env` is the sensorized/SDF variant, not the official asset
**Claim (page):** `Env` = "G1 29-DoF · official CarryBox".
**Reality:** The scene is `ForceOnlyTrainingSceneCfg ⊂ OfficialRefinerAnatomicalWholeHandTacSLSceneCfg`, whose `robot` is the G1 plus 54 physical elastomer patch bodies and whose `obj` is `SMALLBOX_SDF_CFG`, the CarryBox with its collision approximation replaced by a cooked SDF mesh. The module's own docstring states the 54 compliant patches "affect contact dynamics". 54 `VisuoTactileSensorCfg` terms at `update_period=0.02`, force field on, camera tactile off, `tactile_array_size=(20,25)` = 500 taxels per patch. True of Z as well — the branch matching is real, but "official CarryBox" is not.
**Evidence:** `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:4-9,37,66-95,140-195`; `SUGAR/source/sugar_rl/sugar_rl/assets/objects/tactile_objects.py:218-228`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:29-37`
**Action:** APPLIED — `Env` relabelled "G1 29-DoF + 54-patch skin · SDF CarryBox", with the deltas from official SUGAR in the panel.

### F-1014 · CORRECTED · PS · BOUNDARY — `Tacsl` is three stages in one box
**Claim (page):** `Tacsl` = "TacSL R15 → 2 × 27 patches / 6 live channels + 3 slip · history 4".
**Reality:** Correct as a summary, but it collapses three separable stages, one of which is the page's own primary suspect: (a) 54 lazy `VisuoTactileSensor` updates, each computing an SDF penalty force per taxel and projecting the **total** `fc_world + ft_world` into the taxel frame; (b) `reduce_patch_taxels`, which sums `|normal|` over penetrating taxels, sums signed shear, divides by patch area, and forms `‖shear‖ / (mu·normal_load + 1e-8)` with `mu` taken from **the sensor's own cfg** (0.5) — the same `mu` TacSL already used to cap the numerator; (c) a 4-step ring buffer keyed on `common_step_counter`. Also, "R15" is only the render calibration cfg, which is disabled here; the physical unit is the 20×25 force-field grid.
**Evidence:** `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.py:993-1024`; `.../online_patch_tactile.py:85-162,202-275,363-406`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:77-89`
**Action:** APPLIED — node relabelled "54 TacSL force fields (20×25) → 2×27 patches / reduce → 6 ch · history 4"; the three sub-stages are enumerated in the panel and "R15" is qualified.

### F-1015 · CORRECTED · PS · BOUNDARY — `Actor` label omits its output layer and its sibling head
**Claim (page):** `Actor` = "504-D base ⊕ 128-D → 512/256/128".
**Reality:** The widths are right and asserted at runtime, but the node stopped at the last hidden layer — the head is `632→512→256→128→29`. The same module also owns the privileged critic and the learned `std`. "No object state" is true of the actor's 504-D base only; the same instance simultaneously consumes the 890-D privileged group, which contains `obj_pos_b`, `obj_ori_b`, `obj_lin_vel_b`, `obj_ang_vel_b`.
**Evidence:** `.../online_patch_tactile_actor_critic.py:70-106,140-146`; `.../tactile_actor_critic.py:234-237`; `.../base_refiner_env_cfg.py:222-255`
**Action:** APPLIED — node now reads "504 ⊕ 128 = 632 → 512/256/128 → 29" with the critic caveat in the panel.

### F-1016 · CORRECTED · PS · SHAPES — "890-D privileged" labelled the wrong end of the teacher
**Claim (page):** Edge `Teach → Hand` labelled "890-D privileged".
**Reality:** 890 is the teacher's **input** width, not its output. `FrozenOfficialRefinerTeacher` asserts `(num_envs, 890)` on its observation and `(num_envs, 29)` on the action it returns. What flows out of `Teach` is 29-D. The number itself is correct everywhere it is used.
**Evidence:** `.../official_refiner_nominal_teacher.py:35-36,92-99,139-152`; `.../online_patch_tactile_actor_critic.py:74-77`
**Action:** APPLIED — `Teach` now carries "890-D in" on the node and its outgoing edge to `Mux` is labelled "29-D teacher action".

### F-1017 · CORRECTED · PS · SHAPES — `[4,2,27,9]` never crosses that edge as a tensor
**Claim (page):** Edge `Tacsl → Enc` labelled `[4,2,27,9]`; the panel said "**Out:** `[B,4,2,27,9]`".
**Reality:** The observation term returns `actor.reshape(env.num_envs, -1)` — a flat `[B, 1944]`. The actor-critic constructor asserts exactly that. The `[4,2,27,9]` structure is reconstructed *inside* the encoder.
**Evidence:** `.../online_patch_tactile.py:64-65,525`; `.../online_patch_tactile_actor_critic.py:78-82`; `.../patch_tactile_encoder.py:122-137`
**Action:** APPLIED — edge relabelled `[B,1944]` and the panel corrected.

### F-1018 · CONFIRMED · PS · SHAPES — every remaining dimension and threshold on the spine
**Claim (page):** 504 / 890 / 632 / 1944 / 128 / 29; "6 live channels + 3 slip · history 4"; "2 × 27 patches"; "9→128 · 3 layers · 4 heads · FFN 256"; "lift ≥ 0.05 m for 10 frames"; "mass × {1, 1.5, 3, 6, 10}"; "inertia scaled · both read back"; "50 Hz".
**Reality:** All correct. 504 is the `TrackerCommandPolicyCfg` sum (29+3+3+5·3+5·29+5·29+5·29+5·3+3+1) and is asserted at construction; 890 asserted twice; 632 = 504+128 asserted; 1944 asserted; `num_actions != 29` raises. `PATCH_HISTORY_STEPS = 4`, 6 base + 3 slip channels, 27 patches per hand enforced at import (12 palm pads + 3 segments × 5 digits). Encoder geometry hard-frozen to `(4,2,27,9,128,3,4,256)`, `bias=False`, `dropout=0.0`, `norm_first=True`. `minimum_lift_m = 0.05`, `stable_lift_frames = 10`. `MASS_FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)`; inertia scaled by `target/default`; both read back at `rtol=1e-6`, raising on mismatch; `factor == 1.0` writes nothing. `decimation=4 × sim.dt=0.005` = 50 Hz, matching the sensors' `update_period=0.02` and the interval events' `interval_range_s=(0.02,0.02)`.
**Evidence:** `.../carry_box_online_patch_tactile_mass_env_cfg.py:53-54,92-116,223-236`; `.../online_patch_tactile.py:32-50`; `SUGAR/source/sugar_rl/sugar_rl/assets/robots/anatomical_whole_hand_tacsl_g1.py:155-174,225-289`; `.../patch_tactile_encoder.py:24-102,155-163`; `.../online_patch_tactile_actor_critic.py:51-82`; `.../online_mass_jump.py:14,122-150,215-241`; `.../carry_box_refiner_env_cfg.py:137-141`
**Action:** none needed.

### F-1019 · OPEN · PS · outside the path — precomputed and frozen inputs
**Claim (page):** The column was self-contained; the only external artefacts named were `--patch-scale-file` and the warm start.
**Reality:** Five things enter this run from outside the control loop. (1) `patch_channel_scales.json`, fitted offline, injected via env var, frozen into a class-body default at import. (2) `SUGAR/demo_ckpts/CarryBox/tracker.pt`, the released 510-D Tracker. (3) `experiments/.../refiner_model10000.pt`, the frozen Refiner, loaded twice, with a hard `iter == 10000` check but `expected_sha256=None` in the wrapper — **the SHA gate is skipped on the training path**. (4) `CURIOSITY_TACSL_CALIBRATION_DIR` and an optional preconverted G1 USD cache. (5) The compliant/TacSL physics constants — `normal_stiffness=20`, `tangential_stiffness=2`, `mu=0.5`, PhysX compliant stiffness 100 / damping 20 — env vars read at import time, and `mu=0.5` is the exact constant `friction_utilization` divides by. Additionally `MASS_FACTORS` multiply `NOMINAL_CARRYBOX_MASS_KG = 0.3023375869` while the spawn cfg declares `MassPropertiesCfg(mass=0.5)`; `SUGAR/descriptions/` is absent on this checkout, so whether `default_mass` equals the nominal is **UNVERIFIED** — if they differ, the inertia ratio at factor 1.0 is not 1.
**Evidence:** `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:14-51,95,108-125,129-150`; `.../online_patch_tactile_actor_critic.py:134-189,191-250`; `.../official_refiner_nominal_teacher.py:26-29,66-83,114-126`; `.../online_mass_jump.py:13,129-137`; `.../assets/objects/tactile_objects.py:218-228`
**Action:** Partially applied — a "frozen inputs" strip now lists the scale JSON, `tracker.pt`, `refiner_model10000.pt` (×2 uses) and the import-time physics constants, and records that the SHA pin is bypassed. **Still OPEN:** the `mass=0.5` vs `0.3023375869` question needs `SUGAR/descriptions/`, which is not on this checkout — resolve on the runtime host.

### F-2006 · CORRECTED · P · event manager — CarryBox friction is randomized in training
**Claim (page):** The scene panel stated nominal mass and asserted the branches are matched; no material randomization was shown anywhere.
**Reality:** `OnlineMassJumpEventCfg` disables only `obj_mass`. The inherited startup event `obj_physics_material` remains active and draws a per-environment CarryBox **static friction in `[0.2, 0.8]` and dynamic friction in `[0.2, 0.8]`** from 64 buckets, plus `robot_physics_material` on the non-elastomer G1 bodies in `[0.3, 1.6]` / `[0.3, 1.2]`. The evaluator, by contrast, can pin friction explicitly. So the box's real friction varies per training environment — **the exact quantity `friction_utilization` is blind to**, because it divides by the sensor's `mu`.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/base_refiner_env_cfg.py:262-295`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:206-212`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py:87-104`; `.../online_patch_tactile.py:237-247`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:544-567`
**Action:** APPLIED — added to the `Env` node panel and promoted into the "where it breaks" table. Independently re-verified by the caller: `obj_mass = None` is present in `OnlineMassJumpEventCfg`, `obj_physics_material` is not. This materially strengthens F-0001: the policies were trained under friction variation no tactile channel could observe.

### F-2011 · OPEN · P · `_patch_history` step guard — stale history across evaluation batches
**Claim (page):** The `Tacsl` node says "history 4", implying the 4-frame history tracks the current episode.
**Reality:** `_patch_history` caches per `int(env.common_step_counter)` and returns the cached entry unchanged when the counter has not advanced — the reset refill (`episode_length_buf == 0`) lives *inside* the recompute branch. `common_step_counter` is incremented only in `ManagerBasedRLEnv.step` and is never reset. The evaluator resets between profile batches and then recomputes observations twice at an unchanged counter (`env.reset()` → `observation_manager.compute`, then an explicit `observation_manager.compute(update_history=False)`), so **for every batch after the first, the first policy observation of the new batch carries the previous batch's terminal 4-frame patch history**. Affects P and PS, not Z.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_patch_tactile.py:378-406`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:76,203`; `.../manager_based_env.py:349,361-369`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:539-546,601-603`
**Action:** Caveat added to the tactile panel. **Still OPEN as a code bug:** the fix is to hoist the reset refill out of the step-counter guard. Training is unaffected because `_reset_idx` runs after the counter increments.

### F-2015 · CORRECTED · P · the load-bearing P ≡ PS claim
**Claim (page):** "P is identical to PS in every respect except that the three slip channels are exact zeros"; a lost-contact patch carries `slip_score = 1.0` in PS.
**Reality:** The first half is exactly right; the second is one channel short. Both terms call the *same* `_patch_history` with the *same* `PATCH_TERM_PARAMS`, so the cache key is identical; `_online_patch_slip_history` only *reads* `base_history[:, -1]` and writes into the detector's own buffers, never back into it; both then `torch.cat` (a copy) and `reshape(B, -1)`. So for the same physical state, elements `0..5` of every one of the 216 patch records are **bit-identical** and elements `6..8` are exact zeros in P. On the contact-loss frame the detector sets `score = 1.0` **and** `gross_evidence |= contact_loss` → `state = GROSS` → `gross_slip = True`, so PS carries `slip_score = 1.0` *and* `gross_slip = 1.0` — two of the three channels — which is what keeps the token unmasked. No other mask divergence is possible: contact ⇒ channel 0 = 1.0 in both branches.
**Evidence:** `.../online_patch_tactile.py:363-377,409-433,436-441,453-469,505-525`; `.../patch_slip.py:236-243,262-284`; `.../patch_tactile_encoder.py:138,147-163`
**Action:** APPLIED — panel now says "`slip_score = 1.0` **and** `gross_slip = 1.0`", and states that `_patch_history` is shared only between the P and PS terms, of which P registers exactly one. F-0010 updated accordingly.

### F-2016 · CORRECTED · P · evaluator outcome vocabulary
**Claim (page):** `Out` = "hold / drop / fall", three outcomes.
**Reality:** The evaluator emits five labels per profile plus two eligibility gates: `hold_success` (eligible ∧ height loss ≤ 0.05 ∧ ¬robot_fall), `strict_sugar_hold_success`, `drop` (height loss ≥ 0.15 ∨ min z ≤ initial z + 0.03), `safe_lower` (eligible ∧ ¬hold ∧ ¬drop ∧ ¬fall ∧ min z ≤ initial z + 0.08 ∧ min vz ≥ −0.35 ∧ max ref-ori error ≤ 0.8), `robot_fall`, and `reference_robot_deviation`. `eligible_post_jump_window` requires all 80 post-jump frames valid; `strict_sugar_eligible` additionally requires no SUGAR termination inside the window. `safe_lower` is aggregated into the summary alongside the hold counts.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:321-360,370-375,682-687,827-833`
**Action:** APPLIED — `Out` widened to "hold / safe-lower / drop / fall" with the 80-frame eligibility gate named in the panel.

### F-2021 · CORRECTED · P · transient `SpatialTactileEncoder` construction
**Claim (page):** implicit — the `Enc` node is the anatomical patch Transformer.
**Reality:** During construction the inherited `TactileActorCritic.__init__` first builds two `SpatialTactileEncoder`s (a 20×25 conv stack, `expected_flat_dim = 3000`, output 256) and a 760-wide actor MLP and a 1146-wide critic MLP; `ReferenceOnlyTactileActorCritic` then replaces the critic encoder with `nn.Identity` and the critic with `MLP(890,1)`, and `OnlinePatchTactileActorCritic` replaces the actor encoder with `AnatomicalPatchTactileEncoder` and the actor with `MLP(632, 29)`. None of the transient modules survive or ever run a forward pass (the printout is suppressed), but a reader following the class chain hits the 20×25 conv encoder first and may believe it is the deployed one.
**Evidence:** `.../tactile_actor_critic.py:169-192,216-219`; `.../reference_only_tactile_actor_critic.py:90-106`; `.../online_patch_tactile_actor_critic.py:28-31,84-95`
**Action:** APPLIED — one sentence added to the encoder panel.

### F-2007 · CONFIRMED · P · `PatchSlipDetector` is never called in P training
**Claim (page):** `Slip` drawn dormant, "never called", in the P column.
**Reality:** Correct for the P training path, and by construction rather than by guard. `PatchSlipDetector` is imported *inside* `_online_patch_slip_history`, reachable only from `online_patch_tactile_with_slip_actor_history`. The P env cfg registers only `online_patch_tactile_actor_history`; `OnlinePatchMassRobotEnvCfg` is the base class, so it inherits nothing from `OnlinePatchSlipMassRobotEnvCfg`. No slip attribute is ever set on a P env during training.
**Evidence:** `.../online_patch_tactile.py:409-433,436-451,505-525`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:130-140,175-182,239-247,267-269`; `SUGAR/scripts/sugar_rl/online_patch_mass_bcppo_task_registration.py:41-45`
**Action:** none for the training path — but see F-2008.

### F-2008 · CORRECTED · P · the evaluator runs the slip detector in *every* branch
**Claim (page):** `Slip` = "never called", unqualified, in a column whose terminal node is produced by the evaluator.
**Reality:** The evaluation run constructs a `PatchSlipDetector` per profile batch and calls `detector.update(...)` on every one of up to 450 steps, fed by a second, uncached call to `current_whole_hand_patch_features(base_env)` — **with no branch guard**. Its `slip.state` is written into the trace and becomes the reported `gross_slip_patch_fraction`. It never enters the policy, but "never called" is false for the pipeline that produces the `Out` node. The same path also makes the evaluator read all 54 TacSL sensors in the **Z** branch. Corroborated independently by the Z auditor (F-3009).
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:190-192,394-397,608,634-651,690-691`
**Action:** APPLIED — sublabel changed to "never in the actor path · evaluator runs it as a recorder" in both the P and Z columns.

### F-2009 · CORRECTED · P · the branch contract checks do not run in the formal task
**Claim (page):** implied — that P not calling the detector is a verified runtime property.
**Reality:** The runtime assertion `p_branch_did_not_call_slip` (`slip_updates == 0`) lives in `online_patch_preflight_runtime_report`, and `train.py` only invokes that report when `SUGAR_PLAN15_PREFLIGHT_BRANCH` is set — which the launcher sets **only** for task ids containing `-Preflight-`. Under the formal task the report never runs, so the property holds by config construction (F-2007), not by an executed check. The Z auditor found the identical situation for `zero_branch_never_read_tacsl` (F-3018).
**Evidence:** `.../online_patch_tactile.py:554-558,624-629,668`; `SUGAR/scripts/sugar_rl/train.py:620-639`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:89,96-102`
**Action:** APPLIED — the page now says the branch contracts are enforced by construction and by the `-Preflight-` tasks, not by an assertion in the formal run.

### F-2003 · CORRECTED · P · terminations and resets are routine in training
**Claim (page):** The `Hand → Tacsl` edge was labelled "no reset"; the diagram contained no termination or reset stage.
**Reality:** Six termination terms are active (`trajectory_complete` time-out, `anchor_ori`, `ee_body_pos`, `obj_pos`, `obj_ori`, `anchor_pos`) and `episode_length_s = 30.0`. Episodes terminate and reset routinely; `_reset_idx` fires `reset_teacher_handoff` (re-arms the lift gate) and `reset_online_mass_jump` (bumps the episode index, redraws delay+factor, and **writes nominal mass back to PhysX**), and `_patch_history` refills all four history slots for `episode_length_buf == 0`. "No reset" is true only of the *handoff*.
**Evidence:** `.../base_refiner_env_cfg.py:405-455`; `.../carry_box_refiner_env_cfg.py:136-138`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:205-222,234-238`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:212-222`; `.../online_mass_jump.py:152-174`
**Action:** APPLIED. Caller cross-check: the `episode_length_s = 1.0e9` values at `carry_box_online_patch_tactile_mass_env_cfg.py:277,285,293` are in the three ***Play*** cfgs (evaluator, `num_envs=1`); the three training cfgs do not override the inherited `30.0`. The auditor is right and the caller's earlier reading was wrong.

### F-2017 · CONFIRMED · P · dimension contract
**Claim (page):** flat width 1944, base 504, privileged 890, actor input 632, action 29, embedding 128, history 4, 2×27 patches, 6 live channels.
**Reality:** Every one holds and every one is enforced at runtime. The 504 decomposes as 29 + 3 + 3 + 5·(3+29+29+29+3) + 3 + 1. Duplicate of F-1018 from an independent agent.
**Evidence:** `.../online_patch_tactile.py:32-50,64-65,427-433`; `.../patch_tactile_encoder.py:38-53,67-68,122-137`; `.../online_patch_tactile_actor_critic.py:51-54,70-82,87-95,140-146`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:92-116`; `.../tactile_actor_critic.py:161-167,234-237`
**Action:** none needed.

### F-2019 · CONFIRMED · P · gym registry and configclass dispatch
**Claim (page):** The three branches are registered from one config module and differ only in which observation term populates `online_patch_tactile_history`.
**Reality:** Correct. Resolved: entry point `isaaclab.envs:ManagerBasedRLEnv`; env cfg `OnlinePatchMassRobotEnvCfg`; agent cfg `OnlinePatchMassBCPPORunnerCfg`; policy `OnlinePatchTactileActorCritic`; algorithm `BCPPO`; wrapper `OnlineTeacherHandoffVecEnvWrapper` (selected by `SUGAR_PLAN15_LIVE_HANDOFF=1`, not the plain `RslRlVecEnvWrapper`); action `class_type = OnlineMassJumpJointPositionAction`; scene `ForceOnlyTrainingSceneCfg` (all 54 patches force-field only, cameras `None`); events `OnlineMassJumpEventCfg` with `obj_mass = None`, `push_robot = None`, `push_object = None`. `obs_groups` gives `actor_base_groups = ["policy"]`, `critic_base_groups = ["critic"]`.
**Evidence:** `SUGAR/scripts/sugar_rl/online_patch_mass_bcppo_task_registration.py:9-18,41-45,58-67`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:94-95,155-161`; `SUGAR/scripts/sugar_rl/train.py:283-309`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:73-89,123-127,206-236,239-260`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:35-45,49-91`; `.../agents/rsl_rl_bcppo_cfg.py:6-12`; `.../tactile_actor_critic.py:161-167`
**Action:** none needed.

### F-2020 · CONFIRMED · P · BCPPO stage schedule and the unmasked distillation loss
**Claim (page):** "0–499 pure distillation, 500–999 add critic warmup, 1000–1999 ramp PPO authority, 2000–2999 steady full PPO with a 0.25 distillation floor"; distillation loss is an unmasked `.mean()`.
**Reality:** Correct. `bc_only_steps = 500`, `critic_warmup_steps = 1000`, `full_ppo_warmup_steps = 2000`, `max_iterations = 3000`, `stage3_distill_weight_floor = 0.25`, `training_mask_obs_group = "training_handoff_mask"`. Stage 3 uses `alpha = clip((step-1000)/1000, ≤1)` on surrogate and entropy with `distill_weight = max(1-alpha, 0.25)`. `schedule` flips from `"fixed"` to `"adaptive"` at update 500. Surrogate, value and entropy are reduced as `(· * active_weight).sum() / active_denom`; `_reduce_distill_loss` discards `obs_batch` and returns a plain `.mean()`. **This independently confirms F-0006.**
**Evidence:** `.../rsl_rl_bcppo.py:27-33,77-81,84-87,150-160,255-282,317-345`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:53-55,89-90`
**Action:** none needed; F-0006 stands.

### F-3002 · OPEN · Z, P, PS · reward manager × sensorized skin — the reward penalizes grasping
**Claim (page):** Z's reward is the official CarryBox reward, identical to and as intended as P/PS ("same physics, seeds and reward").
**Reality:** Two of the reward terms are structurally changed by the sensorized robot all three branches share, and neither effect was visible anywhere on the page.
1. `hoi_contact` (weight **+1.0**) reads `force_matrix_w_history` from ContactSensors whose `prim_path` is `{ENV_REGEX_NS}/Robot/left_rubber_hand` and `.../right_rubber_hand`. The robot spawner **deactivates the entire `collisions` subtree of both rubber-hand links** (`collision_root.SetActive(False)`, which raises if the subtree is missing), so those two bodies own no active collider and cannot generate a filtered contact pair with `Obj`. The 54 patches are *separate* rigid bodies attached by fixed joints, so all box contact lands on them, not on the hand link. `is_contact` is therefore permanently `False`, and the term degenerates to `(False == contact_label)`: **it pays +1.0 exactly when the reference says *no* contact, and 0 during the entire carry phase.**
2. `undesired_contacts` (weight **−1.0**) targets the `contact_forces` sensor (`prim_path="{ENV_REGEX_NS}/Robot/.*"`, i.e. every robot body) with body regex `^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_rubber_hand$)(?!right_rubber_hand$).+$`. The 54 `{side}_anatomical_{name}_elastomer` bodies and 2 `*_tip` bodies match that regex and are **not** excluded, and `undesired_contacts` sums unfiltered `net_forces_w_history` over the matched bodies above a 0.1 N threshold. **Every patch in contact with the box therefore contributes −1.0.**
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/assets/robots/anatomical_whole_hand_tacsl_g1.py:1093-1140` (deactivation), `:1872-1879` (called at spawn), `:1635,1637` (patch path `{side}_anatomical_{name}_elastomer`), `:474-492` (separate rigid body + fixed joint); `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/rewards.py:142-172`; `.../carry_box_refiner_env_cfg.py:86-98,99-112`; `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:260-268`; `.../base_refiner_env_cfg.py:71-73,93-108`
**Action:** APPLIED to the page as a first-class finding. **Still OPEN as a code/experiment issue, and it is the most consequential thing this audit found.** Independently re-verified by the caller: the deactivation raises if absent, `contact_forces` really is `Robot/.*`, and `hands_contact` really reads only the two rubber-hand sensors. The comparison across Z/P/PS survives — all three share it — but the RL objective is **anti-correlated with grasping**, which is an independent, sufficient explanation for why a policy given more grasp information would not do better and could do worse.

### F-3015 · CONFIRMED · Z · the TacSL sensors do not change the physics
**Claim (task):** If Z's simulated physics differed from P/PS in any way, the matched-control claim would break.
**Reality:** It does not differ. `ExactZeroPatchMassRobotEnvCfg` overrides **only** `observations`; `scene`, `actions`, `events`, `commands`, `rewards`, `terminations` and `sim` are inherited byte-for-byte. The physical deltas — 54 elastomer rigid bodies on fixed joints, the deactivated rubber-hand colliders, the `compliant_contact_stiffness=100 / damping=20` material, the SDF box — all come from the **robot and object asset configs in the shared scene**, not from the sensors. The `VisuoTactileSensor` objects only create read-only PhysX views (`create_rigid_body_view`, `create_sdf_shape_view`) and never write forces back: there is no `apply_force`/`set_force`/`add_force`/`write_*` call anywhere in `visuotactile_sensor.py`. Their per-step compute is skipped in Z anyway (`lazy_sensor_update = True`), and skipping a read cannot change the sim. Neither observation term consumes RNG, and the jump/handoff schedules are deterministic LCG/threshold logic. **Z's trajectory is matched to P/PS at the same seed.**
**Evidence:** `.../carry_box_online_patch_tactile_mass_env_cfg.py:262-265,239-260`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:144-150`; `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.py:283-342,649-686`; `IsaacLab/source/isaaclab/isaaclab/sensors/sensor_base.py:183-191`; `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:80`; `.../online_patch_tactile.py:528-551`; `.../online_mass_jump.py:104-120`
**Action:** none — this was the single biggest risk to the experiment's validity and it is clean. The matched-control claim holds.

### F-3016 · CONFIRMED · Z · the encoder is in the optimizer and its gradient is exactly zero
**Claim (page):** "the patch Transformer is instantiated and its parameters are in the optimizer, but no gradient reaches it".
**Reality:** Both halves hold, including through the `safe_active` fallback. The encoder is a submodule of the policy, the runner and its `BCPPO` are constructed **before** `configure_tactile_actor_finetune`, and upstream `rsl-rl-lib==3.0.1` `PPO.__init__` builds `optim.Adam(self.policy.parameters(), ...)` — every parameter, no `requires_grad` filter. All **41** encoder parameters are optimizer-managed. Executing the real module on an all-zero `[4,1944]` input in train mode and backpropagating a downstream MSE: output is bitwise `torch.zeros(4,128)`, and every one of the 41 parameters has a grad tensor that is present, finite and exactly all-zero. Five Adam steps at `lr=1e-3` leave every encoder parameter bitwise unchanged.
**Evidence:** `.../online_patch_tactile_actor_critic.py:84-86`; `.../patch_tactile_encoder.py:149-163`; `SUGAR/scripts/sugar_rl/train.py:312,315-332`; `.../rsl_rl_bcppo.py:398-412`; upstream `rsl_rl/algorithms/ppo.py` v3.0.1 *(external to this repo — `rsl_rl` is not installed on this node, so that one line is UNVERIFIED in-tree)*
**Action:** none. Panel now adds that the three transformer layers still execute every step; only the output is gated.

### F-3011 · CORRECTED · Z · the encoder's zero-output *mechanism* was described wrongly
**Claim (page):** "every input is zero: **the bias-free projection gives zero tokens** and an explicit final gate returns exact zeros".
**Reality:** The conclusion is right, the mechanism is wrong. `patch_projection` is bias-free so its output is exactly zero — but three learned identity embeddings (time, hand, patch) are then **added**, so the tokens entering the transformer are fully dense. Measured on the real module with an all-zero `[2,1944]` input: projection output has **0** non-zeros; tokens have **55296 / 55296** non-zeros with `max|t| = 0.152`; the 3-layer encoder runs to completion and produces `max|e| = 3.33`; only `torch.where(any_active[:,None], pooled, zeros_like(pooled))` on the last line makes the output zero. The `safe_active` fallback is what keeps this finite: with all tokens masked, `scores.masked_fill(~safe_active, -inf)` would softmax an all-`-inf` row into NaN and `0 * NaN` would poison the backward pass. Because index 0 is force-unmasked, `pooled` stays finite and the `where` gradient `grad * cond` is a clean zero.
**Evidence:** `.../patch_tactile_encoder.py:71-73,138-146,149-157,158-163`
**Action:** APPLIED — panel rewritten to name the `torch.where` gate as the only thing producing the zero, and to credit `safe_active` with keeping the masked softmax finite.

### F-3006 · OPEN · Z, P, PS · warm start silently overrides the configured learning rate
**Claim (page):** The Tracker warm start appeared only as an `Actor` subtitle.
**Reality:** The warm start is a five-step pre-iteration-0 stage with side effects the page never showed: `load_sugar_warm_start` (510→504 remap + critic/std copy), `_audit_tracker_zero_patch_equivalence` (raises on failure), `configure_tactile_actor_finetune` (sets `requires_grad=True` on the whole actor, the encoder and `std`, and — unlike the base class — installs **no** gradient mask on the actor's base columns), **an override of every optimizer `param_group["lr"]` from the configured `1e-3` to the Tracker checkpoint's own converged optimizer LR**, and a `model_pre_update.pt` snapshot. The LR override is load-bearing and silent: `BCPPOCfg.learning_rate = 1.0e-3` is never the LR actually used at update 0, and BCPPO keeps `schedule = "fixed"` for the first 500 updates, so it does not adapt away either.
**Evidence:** `SUGAR/scripts/sugar_rl/train.py:315-332,337-358`; `.../online_patch_tactile_actor_critic.py:112-132,134-189`; `.../rsl_rl_bcppo.py:84-87`; `.../agents/rsl_rl_online_patch_mass_bcppo_cfg.py:78`
**Action:** APPLIED to the page and to `operations.md`'s silent-failure list. **Still OPEN as a documentation/code issue:** the configured `learning_rate` in the agent cfg is misleading and should either be removed or respected.

### F-3017 · CORRECTED · Z · the warm-start equivalence is conditional
**Claim (page):** `Actor · ≡ the warm-started Tracker at init`; "the actor and critic match the released source to `2e-6`".
**Reality:** The audit passes, but the equivalence is conditional in two ways the label hid. (1) The 510→504 remap **drops** source column 35 (contact label) and columns 501:510 (measured object pose), and the audit builds its reference by constructing `source_actor_obs = zeros(2,510)` and filling only `[0:35]` and `[36:501]`. The proven statement is "target(base ⊕ 0) equals the released Tracker **evaluated with those 10 inputs set to zero**", not "equals the released Tracker on its own observation". (2) Target columns `500:504` (`base_lin_vel`, `motion_phase`) are new inputs with **zero first-layer weight** at init, so two of Z's 504 features have no authority at update 0. The `2e-6` and `zero_abs_max != 0.0` gates are genuine and do raise.
**Evidence:** `.../online_patch_tactile_actor_critic.py:155-172,174,191-250`
**Action:** APPLIED — sublabel changed to "≡ Tracker with contact-label + object-pose zeroed". F-0013's evidence range corrected to `:134-189` and `:191-250`.

### F-3018 · CORRECTED · Z · the "never read TacSL" check does not run in the formal task
**Claim (page):** The Z contract was presented as an enforced property.
**Reality:** The machine check `"zero_branch_never_read_tacsl": branch != "Z" or int(scalar("patch_sensor_reads")) == 0` lives in `online_patch_preflight_runtime_report`, which `train.py` invokes **only if `SUGAR_PLAN15_PREFLIGHT_BRANCH` is set** — and the launcher sets it only for `-Preflight-` task ids. For the formal Z run the check never executes. The contract is enforced only by construction of the observation term (which *is* watertight — it validates geometry and returns `torch.zeros`).
**Evidence:** `.../online_patch_tactile.py:554-559,604-609,632-634`; `SUGAR/scripts/sugar_rl/train.py:620-639`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:89,96-102`
**Action:** APPLIED. Same situation as F-2009 for the P contract.

### F-3010 · CORRECTED · Z · "never touches `env.scene.sensors`" was too broad
**Claim (page):** `Tacsl · exact zeros · never touches env.scene.sensors`, drawn as "present in the code but inert".
**Reality:** Three corrections. (1) **The 54 sensors are constructed and initialised in Z** — `ForceOnlyTrainingSceneCfg.__post_init__` calls `super().__post_init__()`, which sets 54 `VisuoTactileSensorCfg` attributes, before merely disabling the optical branch. `InteractiveScene._add_entities_from_cfg` instantiates each, and `_initialize_force_field` generates the 20×25 taxel grid and creates a rigid-body view and a PhysX SDF shape view per patch. They are live objects. (2) **The environment does touch `env.scene.sensors` every step in Z** — `feet_slide`, `feet_air_time_min_penalty`, `undesired_contacts` and `hands_contact` all index it. The true statement is narrower: *the Z patch-observation term* never reads *the 54 TacSL* sensors. (3) **The evaluator reads them in Z**, unconditionally, every eval step.
**Evidence:** `.../carry_box_online_patch_tactile_mass_env_cfg.py:73-89,241-244,263-264`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:188-195`; `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:767,795`; `IsaacLab/source/isaaclab_contrib/.../visuotactile_sensor.py:177-190,283-342`; `.../online_patch_tactile.py:165-168,202-275`; `.../mdp/rewards.py:153-154`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:634`
**Action:** APPLIED — sublabel narrowed to "the Z observation term never reads the 54 TacSL sensors", and the panel records that the sensors are constructed with live SDF views. F-0014 updated.

### F-3019 · CORRECTED · Z, P, PS · neither the robot nor the box is the official asset
**Claim (page):** `Env · G1 29-DoF · official CarryBox`.
**Reality:** Robot: `anatomical_whole_hand_tacsl_robot_cfg(UNITREE_G1_29DOF_MIMIC_CFG, ...)` — both `*_rubber_hand/collisions` subtrees deactivated, 54 elastomer rigid bodies + 2 camera-tip bodies added on fixed joints, a `compliant_contact_stiffness / compliant_contact_damping` material applied (100 / 20 under this launcher's env defaults, vs module defaults 10 / 1). Object: `SMALLBOX_SDF_CFG` (`SdfUsdFileCfg`, `solid_outer_shell_only=True`) rather than the official `SMALLBOX_CFG` (`UsdFileCfg`). MDP: `push_robot = None`, `push_object = None`, `obj_mass = None`, `init_with_ref = False`, `start_init_env_ratio = 1.0`, `rerender_on_reset = False`, plus a body-restricted `robot_physics_material` startup event that excludes all `*_anatomical_*` bodies.
**Evidence:** `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:144-150`; `.../anatomical_whole_hand_tacsl_g1.py:95-102,1872-1890`; `.../assets/objects/tactile_objects.py:220-229` vs `.../assets/objects/objects.py:53-66`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:212,249-259`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py:90-106`
**Action:** APPLIED — relabelled "G1 29-DoF + 54-patch skin · SDF CarryBox", with the deltas in the panel. **Z is matched to P/PS, not to official SUGAR** — the page previously conflated the two.

### F-3012 · CORRECTED · Z, P, PS · trained on motions 0–3, evaluated only on motion 45
**Claim (page):** The `Env → Teach` edge was labelled `motion 45 · 50 Hz`.
**Reality:** Motion 45 is **evaluation-only**. In training, `start_init_env_ratio = 1.0` puts all envs in the protected branch, and `_sample_init_state` assigns `motion_id[ids] = ids % num_motion` with `time_steps = 0`. With the config default `num_envs = 4` that is motions **0, 1, 2, 3**, never 45. Motion 45 appears only as `--motion-id` default 45 in the evaluator, forced by monkey-patching `command._sample_init_state = fixed_start`. **So the student is trained on motions 0–3 and scored exclusively on a motion it never trained on.** `50 Hz` is correct. Independently found by the PS and P auditors (F-1009, F-2010).
**Evidence:** `.../mdp/commands.py:1047-1067`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:241-244,258-259`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:41,479-487`; `.../carry_box_refiner_env_cfg.py:137-141`
**Action:** APPLIED — edge relabelled "motions 0–3 (train) · 50 Hz", and the train/eval generalisation gap is now stated in the results section. The page previously hid it.

### F-3013 · CORRECTED · Z, P, PS · the acting teacher's SHA-256 pin is bypassed
**Claim (page):** `890-D privileged` flowed out of `Teach`, implying the teacher consumes the environment's own observation.
**Reality:** `FrozenOfficialRefinerTeacher` builds its **own** `ObservationManager({"policy": BaseObservationsCfg().policy}, env)` with `enable_corruption = False`, and calls `compute()` fresh at `OnlineTeacherHandoffVecEnvWrapper.step()` entry — before `env.step()`, not from `env.obs_buf`. The env's `teacher` group (which feeds BCPPO's distillation teacher) is a different manager and code path. Also the wrapper constructs the teacher with `expected_sha256=None`, so the pinned `ACCEPTED_REFINER_SHA256` check is **disabled**; only path equality and `iter == 10000` remain.
**Evidence:** `.../official_refiner_nominal_teacher.py:62-83,85-99,134-154`; `.../online_teacher_handoff_wrapper.py:25-29,44-56`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:106-110`
**Action:** APPLIED — recorded in the teacher panel and in the frozen-inputs strip.

### F-3021 · CORRECTED · Z · four panel source citations were wrong
**Claim (page):** `panel-z-tactile` cited `online_patch_tactile.py:523-548`; `panel-z-scene` cited `carry_box_online_patch_tactile_mass_env_cfg.py:174-205`; `panel-z-enc` cited `patch_tactile_encoder.py:140-163` and `online_patch_tactile_actor_critic.py:196-250`.
**Reality:** `exact_zero_online_patch_tactile_actor_history` is at **528-551**. The "same config module, same registration" claim is anchored at **239-265** and **206-236**, not 174-205 (which are the observation-group containers). The encoder `active` mask starts at **138**, and the bias-free projection is declared at **71-73**. `_audit_tracker_zero_patch_equivalence` spans **191-250**.
**Evidence:** `.../online_patch_tactile.py:528-551`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:175-203,206-236,239-265`; `.../patch_tactile_encoder.py:71-73,138-163`; `.../online_patch_tactile_actor_critic.py:191-250`
**Action:** APPLIED — all four ranges replaced, in the panels and in F-0013/F-0014.

### F-3023 · OPEN · Z · bottom-up enumeration residue
**Claim (page):** implicitly, that the Z column covers what runs under `tasks/locomanip/` and `utils/`.
**Reality:** Modules that **run and were not on the diagram**: `online_patch_tactile.py::normalized_motion_phase` (a live 504-D *policy* feature, not a tactile one — its presence contradicts the "Z never uses this module" reading the dormant node invited); `mdp/rewards.py`; `mdp/terminations.py`; `mdp/commands.py::MotionCommand` (motion selection, `contact_label`, reference advance); `mdp/events.py::randomize_rigid_body_material`; the critic half of `reference_only_tactile_actor_critic.py`; `rsl_rl_bcppo.py::teacher_model`; `agents/rsl_rl_ppo_cfg.py::BasePPORunnerCfg` (used to rebuild the teacher's architecture). Does **not** run in Z training: `patch_slip.py`; all the live tactile functions in `online_patch_tactile.py`; `utils/parser_cfg.py`; the residual-teacher family in `official_refiner_nominal_teacher.py`; the audit scene's three raw ContactSensors (only its *event* class is inherited, despite `CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT=1` being exported — that variable gates a different registration script).
**Evidence:** `.../online_patch_tactile.py:74-82,202-275,278-360,363-406,409-433,436-502,505-525,528-551,554-674`; `.../tactile_actor_critic.py:169-192`; `.../reference_only_tactile_actor_critic.py:93-104`; `.../online_patch_tactile_actor_critic.py:84-100`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py:67-106`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:206-236`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:39`
**Action:** Partially applied — `normalized_motion_phase` is now named in the actor panel, and reward/termination/commands are represented by the `Ev` node. **Still OPEN:** `mdp/commands.py::MotionCommand` deserves its own treatment; it owns motion selection and `contact_label`, which F-3002 shows is load-bearing for the reward.

---

## Round 3 — depth (2026-08-19)

One `site-auditor` per major module. The sensing agent found a working Python with
torch (`/lustre/fsw/portfolios/nvr/users/shengzew/miniconda3/envs/caprl/bin/python`,
torch 2.8.0) and produced measured numbers rather than reasoning alone; its
reproduction script is at
`<scratchpad>/demo.py`. **It refuted F-0002 as written.**

### F-4001 · REFUTED · PS · tacsl_sensor + reduce_patch_taxels — F-0002 was wrong
**Claim (page, F-0002):** "TacSL's tangential force is pure velocity damping with no tangential elasticity. Under stick `|v_t| ≈ 0`, so shear collapses to zero no matter how much load the grasp carries" — and "shear, friction and all slip channels collapse to zero exactly while the grasp is holding".
**Reality:** The premise is right, the conclusion is wrong, and it contradicted the page's own F-0003 two paragraphs later. `ft_world` does vanish as `|v_t| → 0`, but the shear **channel** is not `ft`: line 1018 forms `tactile_force_world = fc_world + ft_world` and lines 1019-1024 emit the local-xy of that **total**. With `v_t = 0` the shear channel therefore equals the taxel-frame xy projection of the *normal penalty force*, `k_n·d·sinθ`, where θ is the angle between the taxel z-axis and the SDF normal. **That term is proportional to grasp load, not zero.** Measured with the real `reduce_patch_taxels` on a rigid cylindrical pad (R = 9 mm, taxel half-span 6.54 mm derived from the real `index_proximal` 18×17 mm spec), `k_n=20, k_t=2, mu=0.5`, at **exactly zero relative velocity**: symmetric contact → load 1.434 N, shear −5.6e−9 N, util 7.8e−9; 2.9° off-centre → 1.443 N, −0.0734 N, util 0.102; 8.6° → 1.417 N, −0.2153 N, util 0.304; **17.2° → 1.369 N, −0.4255 N, util 0.622**. At 17.2° off-centre with literally zero slip, `friction_utilization = 0.622` already exceeds the detector's incipient trigger of 0.60. Only for exactly aligned (flat/conformal) contact is stick shear exactly zero — the same code on a flat pad gives shear `(0,0)`, util `0`.
**Evidence:** `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.py:1004-1010,1012-1024,572-608,944-950`; `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_patch_tactile.py:118-151`; `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/patch_slip.py:52,245-248`; `SUGAR/source/sugar_rl/sugar_rl/assets/robots/anatomical_whole_hand_tacsl_g1.py:537-560,722-857`
**Action:** APPLIED — F-0002's wording replaced everywhere on the page. The corrected claim is *stronger*, not weaker: under stick the channel carries no tangential-traction information, and what remains is a geometry-dependent leak of the normal force that can trip the slip detector with zero relative motion.

### F-4004 · CORRECTED · PS · friction_utilization is a saturating slip-speed indicator
**Claim (page):** shear/friction channels are "silent" and become informative "once sliding is already underway".
**Reality:** On aligned contact the channel is a saturating ramp with a sub-millimetre-per-second knee, not a silence. `util_flat = min(k_t·|v_t| / (mu·k_n·d), 1)`, so the knee is `|v_t|* = mu·k_n·d/k_t = 5·d` — 1.5 mm/s at d = 0.3 mm, 0.25 mm/s at d = 0.05 mm. Measured on the flat 200-taxel pad at d = 0.3 mm: 0.1 mm/s → 0.0667; 1.0 mm/s → 0.667; **1.5 mm/s → exactly 1.0**; 5 mm/s → 1.0; 50 mm/s → 1.0. Above ~1.5 mm/s of relative velocity the channel is pinned at 1.0, above both slip thresholds, carrying **no magnitude information at all**. It is a soft-thresholded slip-speed indicator, not a traction measurement.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:1004-1010`; `.../online_patch_tactile.py:144-146`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:31-33`; `.../patch_slip.py:52-53`
**Action:** APPLIED — the two-regime description replaces "goes quiet under stick".

### F-4003 · CORRECTED · PS · the taxel frame is a LOCAL modification, not vendor code
**Claim (page):** the taxel-frame projection was attributed to "TacSL" throughout, and the remediation list said fixing shear "means patching vendored `IsaacLab/`".
**Reality:** The force equations are upstream-faithful, but **the taxel frame is not upstream**. Released IsaacLab v2.3.2 sets one constant quaternion for every taxel — `quat_from_euler_xyz(0,0,-π)` repeated `len(tactile_points)` times — so upstream local-z is the *elastomer body* z-axis. This checkout replaces that with a per-taxel basis bound to the sampled visual-mesh triangle normal, which is why "normal" can come out negative and why `reduce_patch_taxels` must take `abs`. That local change is the direct cause of the geometry-varying leak. It is also, on this asset, a *repair*: the custom patch body frame has `local_z = long_axis` (tangent to the pad), so unmodified upstream would have put ≈0 in "normal" and the entire contact force in "shear". **Both the leak and its fix live in local code.**
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:564-608`; `.../visuotactile_sensor_data.py:46-60`; `SUGAR/source/sugar_rl/sugar_rl/assets/robots/anatomical_whole_hand_tacsl_g1.py:400-415`; upstream v2.3.2 source confirmed from raw.githubusercontent.com
**Action:** APPLIED — this materially changes the remediation plan. Projecting onto the already-computed SDF normal instead of the taxel frame is a local fix; the vendor force equations need not be touched for F-0003.

### F-4009 · CORRECTED · PS · the shipped docstring contradicts the code
**Reality:** The `tactile_shear_force` docstring asserts "*This channel is the physical-tangent projection of TacSL's `F_t` only; normal pressure is never projected into or mixed with signed shear*". Line 1018 does exactly that mixing. The `tactile_normal_force` docstring is accurate. **A reader following the docstrings reaches the opposite conclusion from a reader following the code, and would clear F-0003 in error.**
**Evidence:** `IsaacLab/.../visuotactile_sensor_data.py:46-60` vs `.../visuotactile_sensor.py:1018-1024`
**Action:** APPLIED to the tactile panel.

### F-4010 · OPEN · PS · four silent-degradation paths in the sensing stack
**Reality:** (1) `torch.nan_to_num` on all three inputs: a NaN penetration or force becomes 0 and the taxel silently reports "no contact / no load" — verified, NaN normal + NaN shear on a penetrating taxel yields `[1,0,0,0,0,0]`. The downstream `torch.isfinite(output).all()` check is **vacuous** because sanitisation already happened. A `+inf` penetration survives as an *active* taxel. (2) `_update_force_field` returns early and leaves every force buffer at its previous value if `_contact_object_body_view is None`. (3) The camera branch is skipped whenever `_nominal_tactile is None`, and `get_initial_render()` is never called on any training or evaluation entry point — harmless for PS only because the training scene forces `enable_camera_tactile=False`; the *base* scene cfg turns it on for `palm_r1_c1`, where it would silently produce all-zero images. (4) The sensors observe **only** `{ENV_REGEX_NS}/Obj`, and only its first SDF child mesh: contact with the ground, the other hand or any other body produces exactly zero tactile output. **`contact = 0` means "not touching the box", not "not touching anything".**
**Evidence:** `.../online_patch_tactile.py:118-120,252-255`; `IsaacLab/.../visuotactile_sensor.py:344-400,671,759-760`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:82-89`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:83,97-109`
**Action:** Items (1) and (4) added to the tactile panel and to `operations.md`. **Still OPEN as code issues.**

### F-4011 · CONFIRMED · PS · every physics constant is a launcher env var, and every in-repo default disagrees
**Reality:** The quoted values (`k_n=20`, `k_t=2`, `mu=0.5`, compliant 100/20, 20×25, 50 Hz, camera off) all hold for the PS path — but each comes from an env var read at import time, and the in-repo defaults are wildly different: `VisuoTactileSensorCfg` defaults are `normal_contact_stiffness=1.0`, `friction_coefficient=2.0`, `tangential_stiffness=0.1`; the env-cfg fallbacks are `1.0 / 0.5 / 0.1`; `_COMPLIANT_STIFFNESS/_DAMPING` default to `10.0 / 1.0`. **Anyone running without `train_online_patch_mass_bcppo.py` gets a 20× softer normal stiffness and a 4× larger μ.**
**Evidence:** `IsaacLab/.../visuotactile_sensor_cfg.py:127-197,214-218`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:38,52-96`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:29-33`; `.../anatomical_whole_hand_tacsl_g1.py:95-111`
**Action:** APPLIED to the run section and `operations.md`.

### F-4005 · CONFIRMED · PS · friction_utilization is invariant to the object's material — F-0001 stands
**Reality:** Confirmed, and the invariance is stronger than stated. `current_whole_hand_patch_features` reads `sensor.cfg.friction_coefficient` and **discards the `friction_coefficient=0.5` term parameter entirely** (that param is a dead fallback). The cfg field is pinned to 0.5 by the launcher and the identical field caps the numerator inside TacSL. Nothing in the SDF force path reads a PhysX material. Concretely: the elastomer patches carry friction 0.5 with `friction_combine_mode="average"` while `obj_physics_material` draws box friction from `U[0.2,0.8]`, so **the real contact coefficient spans [0.35, 0.65] while every tactile channel keeps dividing by 0.5**.
**Evidence:** `.../online_patch_tactile.py:144-146,237-247`; `IsaacLab/.../visuotactile_sensor.py:1005-1007`; `.../anatomical_whole_hand_tacsl_g1.py:103-110,1880-1890`; `.../base_refiner_env_cfg.py:274-281`
**Action:** APPLIED — the concrete range [0.35, 0.65] added to the page.

### F-4002 · CONFIRMED · PS · the utilization ratio is unbounded — with measured magnitudes
**Reality:** Per taxel the ratio is exactly `tanθ/mu`, unbounded as θ→90°. Measured: grazing edge contact of the cylindrical pad gives util `1.347 / 1.600 / 1.750` at loads 0.298 / 0.083 / 0.0126 N, **at zero velocity**; a single taxel tilted 60°/80°/89°/90° at 0.1 mm penetration gives `3.46 / 11.3 / 114.5 / 2.0e5`. The `+1e-8` epsilon is the only ceiling. Nothing downstream clamps it — the encoder divides by the frozen scale and feeds a bias-free `nn.Linear`; the only guard is `torch.isfinite`, which such values pass.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:1012-1024`; `IsaacLab/source/isaaclab/isaaclab/utils/math.py:647-665`; `.../online_patch_tactile.py:144-151`; `.../patch_tactile_encoder.py:128-140`
**Action:** APPLIED — the measured ladder replaces the bare word "unbounded".

### F-4006 · CORRECTED · PS · the scale-inflation half of F-0004 is UNVERIFIED here
**Reality:** The no-minimum-load-gate half is confirmed: the only gate on utilization is the same `contact` predicate, and at fixed 45° tilt the ratio is a scale-free constant 2.0 down to loads of 1e-6 N. But the "inflates the frozen scale" half **cannot be closed on this checkout** — `fit_channel_scales` is indeed `np.quantile(|nonzero|, 0.995)`, but no sweep trace and no `patch_channel_scales.json` exists here, so the fitted `scales[5]` is unmeasurable. It is a mechanism, not a measurement.
**Evidence:** `.../online_patch_tactile.py:121-151`; `scripts/sugar/native_tactile/fit_online_patch_channel_scales.py:26-31,55-69`
**Action:** APPLIED — F-0004 now separates the confirmed half from the unverified half. **Resolving it needs the traces on the runtime host.**

### F-4008 · OPEN · PS · three unstated approximations in the patch reduction
**Reality:** (1) The signed shear sum adds x/y components expressed in **different frames** — each taxel's basis is bound to its own triangle normal — so `Σ shear` is not a vector in any single frame; coherence loss ≈2% over the active arc, ≈9% over the full 41.6° wrap, and a normal-force leak cancels to ~0 under symmetric contact while surviving fully off-centre. (2) `patch_area_m2` is the flat `width_m × length_m` rectangle, not the curved area of the conformal pad, so `mean_pressure_pa` is systematically high. (3) Taxels sit on a planar *projected* grid, so taxel density per unit true surface area falls as the pad slopes — with fixed per-taxel `k_n`, effective areal stiffness varies within a patch.
**Evidence:** `.../online_patch_tactile.py:28-31,125-126,143`; `IsaacLab/.../visuotactile_sensor.py:516-544,572-608`; `.../anatomical_whole_hand_tacsl_g1.py:722-857`
**Action:** APPLIED to the tactile panel. **Still OPEN as modelling issues.**

### F-4007 · CORRECTED · PS · the sensing contract was absent from the page
**Reality:** Per sensor (54, `tactile_array_size=(20,25)`, `num_tactile_points` hard-checked = 500): `penetration_depth [E,500] float32` metres ≥0; `tactile_normal_force [E,500] float32` newtons **signed**; `tactile_shear_force [E,500,2] float32` newtons signed; plus world-frame oracles zeroed outside penetrating taxels. Numeric scale: per taxel `F_n = 20 N/m × depth`; 500 taxels ⇒ full-contact patch stiffness ≤ 10 kN/m **independent of patch area**. Worked example (flat pad, 200/500 taxels at 0.3 mm, `index_proximal` area 3.06e-4 m²): `normal_load = 1.2 N`, `mean_pressure = 3921.6 Pa`. Patch areas range 1.182e-4 m² (`palm_r3_c1`) to 7.673e-4 m² (`palm_r1_c1`). `penetration_depth` is the **object's** SDF value at the taxel — cooked at `sdf_resolution=128` — not an elastomer deformation.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:557-562,612-642,912,958,1023-1024`; `.../online_patch_tactile.py:28-31,152-162,202-255,363-406,505-525`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py:38,71-96`; `.../tactile_objects.py:208-229`
**Action:** APPLIED — contract block added to the tactile panel.

### F-4012 · CONFIRMED · PS · sensor update scheduling
**Reality:** `update_period=0.02` equals the control step; lazy update on first `.data` access. Two details: the force path ignores the outdated-env mask and recomputes **all** envs whenever any row is stale (deliberate — advanced indexing would write to copies), and `_timestamp_last_update` is advanced only for the outdated subset, so a just-reset env reports timestamp 0 alongside freshly computed forces.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:649-686`; `IsaacLab/source/isaaclab/isaaclab/sensors/sensor_base.py:183-186,353-362`; `.../online_patch_tactile.py:278-313`
**Action:** none needed.

### F-4013 · CONFIRMED · PS · the force computation sequence
**Reality:** Ten steps with one branch, condensed to one line on the page. The branch is `if collision_mask.any() or cfg.visualize_sdf_closest_pts` — if no taxel in the whole batch penetrates, every buffer stays zero and the function ends. Non-penetrating taxels need no masking because `fc_norm = 0` forces both terms to 0. The three oracle tensors are the only quantities explicitly gated by `collision_mask`.
**Evidence:** `IsaacLab/.../visuotactile_sensor.py:734-787,894-1024`; `.../tactile_objects.py:208-229`
**Action:** APPLIED — the derivation of each output named in panel (a).

### F-4014 · CONFIRMED · PS · slip coupling
**Reality:** Confirmed exactly — `current = base_history[:, -1]`, fields sliced into `detector.update`, thresholds 0.60 / 0.90 combined with `|` against shear-rate and pressure-drop evidence. Given F-4001, that trigger fires at 0.622 on a 17°-off-centre curved contact with **zero relative velocity**.
**Evidence:** `.../online_patch_tactile.py:436-502`; `.../patch_slip.py:52-53,154-169,223-248`
**Action:** none needed.

### F-8002 · REFUTED · Z, P, PS · rsl_rl_bcppo — F-0007 was wrong
**Claim (page, F-0007):** "The value loss is masked by `active_weight` but GAE returns/advantages are computed by stock rsl_rl over the whole 24-step window, so windows straddling the handoff bootstrap post-handoff advantages off value estimates for states the critic was never fit on."
**Reality:** The premise (value masked, GAE not) is true, but **the stated consequence cannot occur**. `handoff_active` is **monotone within an episode** — `advance()` only ever sets entries `True`, and the sole `False` write is in `reset()`, wired as a `mode="reset"` event. GAE is a *backward* recursion, so `adv_t` depends on `V_{t+1}, V_{t+2}, …` — the future of `t`. For any transition with `mask_t = 1`, every later transition in the same episode also has `mask = 1`, so every value estimate its advantage bootstraps from **is** a post-handoff state the critic is fit on. The reverse case — a pre-handoff advantage bootstrapping off post-handoff values — does happen, and is discarded by the surrogate mask. Episode resets break the recursion via the `done` flag, so the mask cannot flip back inside a live GAE chain. The mask is also correctly phased: the wrapper reads `controller.handoff_active` at entry, i.e. the value written by `step_teacher_handoff` after the previous step's physics and before its observation compute — so `obs_t["training_handoff_mask"]` is exactly "the student's action was executed at step t". No off-by-one.
**Evidence:** `SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_teacher_handoff.py:55-60,62-89`; `.../online_teacher_handoff_wrapper.py:44-56`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:216-237`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:213-229`; `.../rsl_rl_bcppo.py:255-257,276-278`. Caveat: rsl-rl-lib 3.0.1 is not installed here, so the GAE recursion itself is asserted from the standard form, not read — UNVERIFIED on that one point; the monotonicity argument holds for any backward GAE with a `done` cut.
**Action:** APPLIED — F-0007 removed from the page in all three places. The real residual asymmetry, which is *not* an advantage-corruption bug: the critic is fit only on post-handoff states while `V` is still *queried* on pre-handoff states every rollout step, so pre-handoff `V` is an unconstrained extrapolation — harmless for the loss, but the logged value function is meaningless over the teacher prefix.

### F-8003 · CORRECTED · Z, P, PS · the unmasked distillation loss — sharper than F-0006 said
**Claim (page, F-0006):** the unmasked `.mean()` means "the dominant gradient is computed largely over teacher-driven pre-handoff pickup transitions".
**Reality:** The mechanism is exactly right; the consequence was unsupported and probably backwards. The mask is a per-env step function with one rising edge per episode, so with a 30 s episode budget the *steady-state* majority of transitions is post-handoff, not pre. **The sharp, source-derivable defect is different and worse: the unmasked distill term applies the frozen nominal-mass Refiner as a regression target during the post-jump window itself** — the exact regime the experiment exists to study. The teacher was trained under `obj_mass` log-uniform scaling in `[0.5, 2.0]`; the jump goes to `{1.0, 1.5, 3.0, 6.0, 10.0}`×, and this env sets `obj_mass = None`, so **the teacher has never seen 3×/6×/10×**. Weight is 1.0 for updates 0–1999 and 0.25 thereafter, against a surrogate that only reaches full weight at 2000.
**Evidence:** `.../rsl_rl_bcppo.py:77-81` vs `:222,255-282`; `:284-308`; `:317-345`; `.../online_mass_jump.py:14`; `.../base_refiner_env_cfg.py:285-295`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:211`
**Action:** APPLIED — F-0006's consequence rewritten on the page.

### F-8001 · CORRECTED · Z, P, PS · the LR does adapt back
**Claim (page, F-3006):** "the warm start overwrites every optimizer param group with the Tracker's LR, and BCPPO holds `schedule='fixed'` for the first 500 updates, **so it never adapts back**."
**Reality:** First two halves right; "never adapts back" is false. `update()` recomputes `self.schedule` on entry: `"fixed"` while `update_step < 500`, **`"adaptive"` from 500 onward**. From 500 the KL block runs **once per mini-batch** — 20× per update — writing a new scalar into every `param_group["lr"]`, ×1.5 / ÷1.5, clamped to `[1e-5, 1e-2]`. So the LR is pinned to the Tracker scalar for updates 0–499 only and is fully adaptive for the remaining 2500, starting *from* the Tracker scalar rather than from `1e-3`. The Tracker's numeric LR cannot be read here (`SUGAR/demo_ckpts/` is absent).
**Evidence:** `.../rsl_rl_bcppo.py:28,83-87,213,233-247`; `SUGAR/scripts/sugar_rl/train.py:337-358`; `.../rsl_rl_online_patch_mass_bcppo_cfg.py:78,80`
**Action:** APPLIED to the page and `operations.md`.

### F-8006 · CORRECTED · Z, P, PS · the distillation floor engages at 1750, not 2000
**Reality:** Stage boundaries are right; the floor's engagement point is not. `alpha = min((step-1000)/1000, 1)` and `distill_weight = max(1-alpha, 0.25)`, so `1-alpha` reaches 0.25 at **update 1750** — from there the floor binds and distillation is flat at 0.25 while the surrogate is still ramping. That is exactly the interval where the modified floor differs from official SUGAR (`stage3_distill_weight_floor = 0.0`), so the modification bites 250 updates earlier than stated. Two further omissions: in stage 2 the value loss is itself linearly ramped (`alpha × value_loss_coef × value_loss`), so at update 500 the critic's weight is exactly **0** and reaches 1.0 only at 999 — "critic warmup" is a ramp, not a switch; and in stage 3 the value loss is **not** ramped (full weight from 1000), only surrogate and entropy carry `alpha`.
**Evidence:** `.../rsl_rl_bcppo.py:317-319,324-330,335-345,28-30`; `.../rsl_rl_online_patch_mass_bcppo_cfg.py:89`; `.../rsl_rl_bcppo_cfg.py:11`
**Action:** APPLIED.

### F-8008 · REFUTED · Z, P, PS · the "24-step rollout can't contain an 80-frame outcome" reasoning
**Claim (caller's working assumption, never fully on the page):** "`num_steps_per_env = 24` vs an 80-frame outcome window means no rollout contains a full outcome, so all post-jump credit is mediated by the critic."
**Reality:** Wrong in three ways. (1) **The 80-frame window is an evaluator construct with no counterpart in training** — nothing in the training path references it; framing the objective in terms of it imports the eval process into the training process. (2) **The reward is dense, not terminal**: six per-step terms track the *measured* object against the reference every control step (`motion_obj_pos/ori/lin_vel/ang_vel` at 0.5, `obj2body_pos/ori` at 0.25), each `exp(-error/std²)`. A jump that makes the box sag degrades reward on the *first* post-jump frame. (3) **Terminations fire inside the window**: `obj_pos` ends the episode at 0.3 m reference error — from rest that is ~12–13 control steps of free fall at 50 Hz, well inside 24 — and it is not `time_out=True`, so it is a hard `done` with `V = 0`. What survives: γ is not the limiter (0.99²⁴ = 0.786), **truncation at 24 steps plus the bootstrap is** — so a slow *sag* that never trips a termination is credited only through the critic, which is privileged, tactile-blind, fit only on post-handoff samples, and ramped 0→1 over just 500 updates.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:45`; `.../base_refiner_env_cfg.py:367-395,408-412,433-439`; `.../mdp/rewards.py:74-77`; `.../mdp/terminations.py:54-59`; `.../carry_box_refiner_env_cfg.py:137-141`
**Action:** APPLIED — the claim is not on the page in its wrong form; the surviving version (drops are credited locally, slow sags only via the critic) is.

### F-9001 · CORRECTED · P, PS · the encoder output is 97 % constant — F-0009 stated the consequence backwards
**Claim (page, F-0009):** "no terminal LayerNorm, so the embedding's scale relative to the 504-D base is uncontrolled."
**Reality:** The structural half is right and the consequence is backwards. With pre-LN, each sub-block adds `attn(LN(x))` / `ff(LN(x))`, whose magnitude is set by the LayerNorm output and the layer weights — **not** by the input scale. Measured on CPU (torch 2.0.1, seed 7): tokens entering the stack rms 0.098; encoded tokens leaving rms 1.30 (×13–29 amplification); pooled output per-row L2 = 15.13. **Over 64 different random contact patterns the mean embedding has L2 = 15.12 and the mean deviation from it is L2 = 0.486 — only 3.2 % of the embedding's norm varies with the tactile input.** The other 96.8 % is a constant DC offset manufactured by the identity embeddings and the input-scale-independent LN branches. The output is not "uncontrolled/large" — it is *pinned* at rms ≈ 1.3 whatever the input, and the tactile signal is a 3 % perturbation riding on a constant. Strictly worse than stated.
The page also omitted the compensator. `warm_start_tactile_gain = 0.01` multiplies the 128 patch columns of `actor.0.weight` at construction and the warm start preserves them: at init the patch contribution is 0.7 % of the base contribution. **It does not persist.** Those columns are trainable with no gradient mask (F-9002), and Adam's per-step displacement is ≈ lr regardless of current magnitude. Measured, starting from the gained columns (std 2.295e-4 vs full-init 2.295e-2): at lr 1e-3, 20 steps → 0.75× full-init std, 100 steps → 2.03×. The schedule runs 20 optimizer steps per update, so **at the configured lr the gain is gone inside one BCPPO update.**
**Evidence:** `.../patch_tactile_encoder.py:83-94,76-82,98-102,155-163`; `.../online_patch_tactile_actor_critic.py:48,101-105,156-157,115-118`; `.../rsl_rl_online_patch_mass_bcppo_cfg.py:69,76-78`; `.../rsl_rl_bcppo.py:235-247`
**Action:** APPLIED. Base-side magnitude used a fresh kaiming first layer as stand-in — `SUGAR/demo_ckpts/` is absent, so the released Tracker's actual first-layer norm is UNVERIFIED.

### F-9002 · OPEN · P, PS · the finetune gate installs no gate — the warm-started base columns train from update 0
**Claim (page):** nothing; the page described the warm-started actor only as "the Tracker plus whatever BCPPO teaches it".
**Reality:** The parent class implements a real finetune gate: it freezes every actor parameter and `std`, unfreezes only the encoder and `actor.0.weight`, freezes encoder biases, and installs a backward hook `mask_base_columns` that zeroes the gradient of the first layer's **base** columns every step — the mechanism that would keep the warm-started Tracker columns bitwise fixed. `OnlinePatchTactileActorCritic.configure_tactile_actor_finetune` **overrides it without calling `super()`** and instead sets `requires_grad_(True)` on the actor, the encoder and `std`. Since nothing in the Plan-15 path ever set `requires_grad=False` (the warm start runs under `torch.no_grad()`, which does not change `requires_grad`), the method is **a no-op on module state plus a report dict**. No hook is installed; `_actor_base_gradient_hook` stays `None`. **The 504 warm-started base columns therefore train from update 0, and the "zero-patch actor ≡ Tracker" invariant holds only for the single forward pass inside the audit.** `train.py` calls this method "the tactile finetune gate" and writes that description into `tactile_finetune_resume.json`, persisting a wrong claim into the run artifacts.
**Evidence:** `.../online_patch_tactile_actor_critic.py:112-132`; `.../tactile_actor_critic.py:214,357-412` (the discarded freeze + `register_hook(mask_base_columns)` at :397); `SUGAR/scripts/sugar_rl/train.py:325-332,545-559`
**Action:** APPLIED to the page. **Still OPEN as a code defect** — either call `super()` or rename the method.

### F-9003 · CORRECTED · P, PS · the mask framing, and what it implies before first contact
**Claim (page, F-0010):** the mask "conflates *not in contact* with *padding*, so the policy cannot represent 'no contact here'."
**Reality:** The predicate is right, the framing is wrong in one direction and understated in another. **Wrong:** there is no padding — the geometry is hard-frozen to exactly 216 real patch records, so no token is ever filler; the code is deliberately reusing `src_key_padding_mask` as a contact gate. And exclusion from the pooled set *is* a representation, since each surviving token carries its own `patch_embedding` row. **Understated:** the real consequence is at the whole-hand level. With no patch in contact, `any_active` is false and `torch.where` returns exact zeros — **bitwise identical to Z's forced zero**. Pre-contact, P has all six live channels zero and PS additionally has all three slip channels zero. **So for the entire teacher prefix before first contact, P and PS feed the actor exactly Z's 632-D input — the three branches are literally the same policy until the hand touches the box.** The P/PS divergence claim is confirmed, and PS keeps its own 4-step slip ring buffer, so a contact-loss frame stays unmasked for **four** control steps, not one.
**Evidence:** `.../patch_tactile_encoder.py:38-53,138,147,152-154,160-163`; `.../patch_slip.py:236,240-241,247-250`; `.../online_patch_tactile.py:409-433,470-486`
**Action:** APPLIED.

### F-9004 · CONFIRMED · P, PS · after scaling, the binary channels dominate and shear is smallest
**Reality:** The scheme is p99.5 **max-scaling, not standardization** — no centering, no unit variance. Pooling x and y under one shear scale is correct (it preserves isotropy); 1.0 on channels 0/7/8 is correct (they are already {0,1}). But after division, on a plausible sparse-contact draw: contact **1.000**, normal load 0.148, pressure 0.154, **shear x 0.054, shear y 0.047**, friction utilization 0.258, slip score 1.000, slip flags 1.000. The binary indicators sit at exactly 1.0 and dominate; the continuous physical channels sit 4–20× below; **and the two channels the whole experiment is about are the smallest.** This is structural, not draw-dependent: p99.5 maps the tail to 1 and leaves the median wherever the distribution puts it, so heavier tails crush the typical value further — and F-4001/F-4002 make those tails heavier, pushing shear and utilization *further* toward zero.
**Evidence:** `scripts/sugar/native_tactile/fit_online_patch_channel_scales.py:13-32,55-66`; `.../patch_tactile_encoder.py:139`; `.../patch_slip.py:250`; `tests/native_tactile/test_online_patch_channel_scales.py:36-42`
**Action:** APPLIED.

### F-8004 · OPEN · Z, P, PS · the shape of one update was absent from the page
**Reality:** One BCPPO update is `num_steps_per_env = 24` × `num_envs = 4` = **96 transitions**, split into 4 mini-batches of 24 and iterated 5 epochs → **20 Adam steps per update, 60 000 over the run**. Also in force and never overridden: γ 0.99, λ 0.95, clip 0.2 (also the value clip), desired-KL 0.01, entropy 0.005, value 1.0, grad-norm 1.0, `normalize_advantage_per_mini_batch = False`.
**Evidence:** `.../rsl_rl_bcppo_cfg.py:15`; `.../rsl_rl_online_patch_mass_bcppo_cfg.py:55,71-91`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:240-244`; `SUGAR/scripts/sugar_rl/cli_args.py:62-97`
**Action:** APPLIED — 96 transitions per update is load-bearing for every masking claim and its absence made that discussion abstract.

### F-8005 · OPEN · Z, P, PS · whole updates land inside the teacher prefix
**Reality:** When no sample in a mini-batch is post-handoff, `active_count = 0`, `active_denom = 1.0`, and surrogate / value / entropy all evaluate to **exactly 0.0**, contributing no gradient — the update degenerates to pure behaviour cloning at the current `distill_weight`, i.e. in stage 3 a 4× smaller effective step. Not rare: `SUGAR_INIT_AT_RANDOM_EP_LEN=0` starts all 4 envs at episode step 0 in lockstep, and the repo's own preflight sizes a *single* update at 360 steps to "reach grasp, mass jump, and post-jump signal", so **the first 1 to 15 rollouts of every episode are 100 % pre-handoff in every mini-batch**. Two guards are correct: `kl_mean = 0.0` fails the `> 0.0` test so an empty batch cannot inflate the LR, and the masked advantage renormalization is skipped when `active_count <= 1`.
**Evidence:** `.../rsl_rl_bcppo.py:159-166,222,236,255-282`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:93`; `.../rsl_rl_online_patch_mass_bcppo_cfg.py:94-101`; `.../online_teacher_handoff.py:13-16,67-83`
**Action:** APPLIED. Exact fraction is structurally derivable but not numerically — it needs the handoff frame, which depends on Refiner pickup dynamics on motions 0–3 (`data/` and `demo_ckpts/` both absent here).

### F-8015 · OPEN · Z, P, PS · a logged `surrogate = 0.0` is ambiguous
**Reality:** `mean_surrogate_loss`, `mean_value_loss` and `mean_entropy` accumulate the **masked** scalars, so an update in which no env has handed off logs `surrogate = 0.0`, `value_function = 0.0`, `entropy = 0.0` — **indistinguishable in the log from a converged or collapsed policy**. The only signal that separates them is `alg.last_training_mask_report`, which is written on every `update()` but read **only** by the preflight path, never persisted during a formal 3000-update run. So formal training keeps no record of how many transitions were post-handoff.
**Evidence:** `.../rsl_rl_bcppo.py:95-110,255-282,417-421`; `SUGAR/scripts/sugar_rl/train.py:620-639`
**Action:** APPLIED to the page and `operations.md`. **Still OPEN** — persisting the mask report in formal runs is a one-line fix and would be worth a lot.

### F-8010 · OPEN · Z, P, PS · the distillation KL sets the student's exploration noise
**Reality:** The loss is `KL(teacher ‖ student)` — mode-covering. Because `σ_s` is a *learned, state-independent* parameter, the loss is minimized in `σ_s` at `σ_s² = σ_t² + (µ_t − µ_s)²`, so the distillation term **actively inflates the student's exploration noise** to the Refiner's std plus the squared mean gap. For updates 0–999 this is the only thing setting the student's std — there is no entropy term at all until update 1000. The teacher's std is read verbatim from the Refiner checkpoint; the student's std at update 0 is the *Tracker*'s std (copied by the warm start), not the configured `init_noise_std = 0.5`. `self.distill_loss_fn = nn.MSELoss()` is constructed and never used — dead code that will mislead anyone grepping for the BC loss.
**Evidence:** `.../rsl_rl_bcppo.py:32,70,297-308`; `.../reference_only_tactile_actor_critic.py:159-165`; `.../online_patch_tactile_actor_critic.py:169-172`
**Action:** APPLIED to the page.

### F-8011 · OPEN · Z, P, PS · the distillation teacher is rebuilt by shape inference, unvalidated
**Reality:** The acting teacher builds a real `ActorCritic` from config, verifies `class_name == "ActorCritic"`, loads `strict=True`, and checks `iter == 10000`. BCPPO's copy does none of that: it greps `state_dict` for `actor.*weight*` keys, sorts by index, **infers** `num_obs`, `num_actions` and `hidden_dims` from the shapes, then constructs a bare `MLP` with `activation="elu"` **hardcoded**, no observation normalizer, and no iteration or hash check. The activation happens to match, but nothing enforces it — a teacher trained with any other activation would be silently distilled as an ELU network. The inferred `num_obs` is never compared against the actual teacher-group width; a mismatch surfaces only as a matmul error deep in `update()`. One further asymmetry: the acting teacher forces `enable_corruption = False` while the env's `teacher` group inherits `enable_corruption = True`.
**Evidence:** `.../rsl_rl_bcppo.py:35-71,284-295`; `.../official_refiner_nominal_teacher.py:85-89,101-128`; `.../base_refiner_env_cfg.py:222-255`; `.../rsl_rl_ppo_cfg.py:17-22`
**Action:** APPLIED to the page.

### F-9010 · OPEN · P, PS · the warm-started actor runs on a cleaner input distribution than it was trained on
**Reality:** The released Tracker was trained on `TrackerCfg` with `enable_corruption = True` and explicit `Unoise` on 7 of its 11 terms (`ref_joint_pos` ±0.01, `root_lin_vel_b` ±0.05, `root_ang_vel_b` ±0.1, `base_ang_vel_history` ±0.2, `joint_pos_history` ±0.01, **`joint_vel_history` ±0.5**, `project_gravity` ±0.2). The Plan-15 `TrackerCommandPolicyCfg` sets `enable_corruption = False` and declares no noise on any term; the privileged critic group keeps corruption **on**. That is a real distribution shift between the source weights and the observations they now consume, on the same branch where the page asserts a 2e-6 equivalence. Shared by Z/P/PS so it does not confound the comparison, but it undercuts "the warm-started actor ≡ the Tracker" as an *operational* statement.
**Evidence:** `.../train_tracker/base_tracker_env_cfg.py:227-241`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:96-120`; `.../base_refiner_env_cfg.py:248-250`
**Action:** APPLIED.

### F-9009 · OPEN · P, PS · ~1.5 M parameters are built and discarded, consuming the RNG
**Reality:** F-2021 understated this by roughly 6×. The chain builds and drops **two** `SpatialTactileEncoder`s (~90 k each), a 760-wide actor MLP (~558 k) and a 1146-wide critic MLP (~752 k) before the real modules are installed. Every one consumes global RNG during init, so **the patch Transformer's weights depend on that consumption** — harmless for the branch comparison (all three build the identical throwaway set) but it means the encoder seed is not reproducible from the encoder code alone. A class attribute suppresses the log line that would reveal it. Also dead but reading as live machinery: `_tactile_enhanced_actor_forward` and the bounded-residual branch (both caps default `None` and are never passed), plus two fully-overridden warm-start audits.
**Evidence:** `.../tactile_actor_critic.py:169-203,216-219,296-355`; `.../reference_only_tactile_actor_critic.py:47-48,93-104,112-157,193-265`; `.../online_patch_tactile_actor_critic.py:31,33-49,84-100`
**Action:** APPLIED.

### F-9005 · CONFIRMED · P, PS · the configclass default is deep-copied and the NaN sentinel cannot survive
**Reality:** `@configclass` does **not** share one list — `_process_mutable_types` rewrites the member as `field(default_factory=_return_f(value))` and `_return_f._wrap` returns `deepcopy(f)` on every call; `__post_init__` deepcopies again. Each instance owns a distinct list, copied twice. The *values* are computed once per process at first import. The NaN sentinel cannot reach a run silently: the launcher sets the env var before the lazy gym-registry import, and if a NaN list did arrive the encoder raises — via the `torch.isfinite` half of the condition, since `NaN <= 0.0` is False. The real fragility is the sentinel *choice*: replacing `[nan]*9` with any plausible default would convert a hard crash into a silently mis-scaled run.
**Evidence:** `.../rsl_rl_online_patch_mass_bcppo_cfg.py:18-32,44`; `IsaacLab/source/isaaclab/isaaclab/utils/configclass.py:374-379,477,497,91-95,382-399`; `SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:128-153`; `.../patch_tactile_encoder.py:54-60`
**Action:** none needed — F-0011 stands as written.

### F-9012 · OPEN · P, PS · nothing binds a scale file to the channel definitions that produced it
**Reality:** The live configuration surface is one env var and three cfg knobs; five further cfg fields (`tactile_grid_shape`, `tactile_num_hands`, `tactile_channels_per_hand`, `tactile_encoder_channels`, `tactile_embedding_dim`) feed only the discarded encoders, and `init_noise_std = 0.5` is inert (overwritten by the warm start). Every failure surface is loud — nine `ValueError`/`RuntimeError` gates including a **per-forward non-finite scan over the full 1944-wide tensor**. **The one thing with no guard at all: nothing verifies that the fitted scales in the JSON were produced from the same channel definitions as the running sensor.** The scales are baked into the checkpoint buffer, so a checkpoint and a scale file can be recombined incorrectly without any error — which matters directly, because every remediation in the coda changes those definitions.
**Evidence:** `.../rsl_rl_online_patch_mass_bcppo_cfg.py:18-32,37-45,62-70`; `.../online_patch_tactile_actor_critic.py:48,51-54,70-82,139-152,169-172,240-244`; `.../patch_tactile_encoder.py:49-60,123-129`
**Action:** APPLIED to the coda and `operations.md`.

### F-9006 / F-9008 / F-9011 / F-9013 / F-8007 / F-8012 / F-8014 · CONFIRMED · consolidated
**Reality:** (F-9006) The 2e-6 equivalence is exactly as narrow as F-3017 said, and additionally is a **single forward pass at warm-start time only** — nothing re-checks it, and per F-9002 the base columns train from update 0; the tolerance was relaxed from the 1e-6 used by both parent audits. (F-9008) Every dimension checks out with exact decompositions: 504 = 29+3+3+15+145+145+145+15+3+1; 890 = 656 future + 42 body_pos + 84 body_ori + 93 proprio + 15 object; the 510→504 map lands on term boundaries on both sides. (F-9011) Exactly 41 encoder parameters, 402,944 scalars, ~27 % of the actor-side trainable model; `patch_projection.bias is None` confirmed at tensor level; `channel_scales` is a persistent buffer, in `state_dict` and every checkpoint, never in the optimizer. (F-9013) The Z-panel numbers reproduce independently (`max|t|` 0.144 vs the page's 0.152 — the same statistic at a different init seed; these should be labelled seed-dependent). (F-8007) The student-action-stored-while-teacher-executes claim is confirmed; the mask is load-bearing, not cosmetic. (F-8012) Advantages are normalized **twice** — globally by rsl-rl over the 96-transition rollout, then re-normalized inside BCPPO on post-handoff statistics only, skipped when fewer than two active samples. (F-8014) Stage 2 really is critic-only; the update contract and return keys are as documented.
**Evidence:** as cited in the individual agent reports above; all carry path:line.
**Action:** Seed-dependence labelled on the page; the rest need no page change.

### F-7001 · CORRECTED · PS · the reset-swallowing bug is ASYMMETRIC and hits PS on the evaluation path
**Claim (page, F-2011):** "the history cache is keyed on `common_step_counter` and the reset refill lives inside the recompute branch, so the first policy observation of every evaluation batch after the first carries the previous batch's terminal history. **Training is unaffected.**"
**Reality:** The cache description is right, but this **understates the PS-specific damage, and it is the one defect that is asymmetric between P and PS.** `_online_patch_slip_history` returns early on a step-counter match **before** calling `detector.update`, so on that call the detector never receives `reset_mask`. `ManagerBasedEnv.reset()` runs `_reset_idx` and then `observation_manager.compute()` **without incrementing `common_step_counter`** — the counter is bumped only inside `step()`. The evaluator calls `env.reset()` once per batch. So for batch ≥ 2 the reset is **silently swallowed**, and on the next `step()` `episode_length_buf` is already 1, so `reset_mask` is all-False again — the detector differences the new episode's frame 1 against the *previous* episode's terminal frame, with `dt = 0.02` so the monotonicity guard passes. Measured on the real module: episode A ending at `pressure 5000`, episode B starting at `pressure 250` with the reset dropped → `slip_score` saturates at **2.0** on the first frame instead of 0.778; and **a GROSS latch from episode A survives every frame of episode B**. P has no differencing state, so the identical cache bug costs P only a stale first observation.
**This is the strongest single candidate for PS being worse than P**, and it lives on the evaluation path that produced the −0.2712 hold difference.
**Evidence:** `.../online_patch_tactile.py:444-447,467,384-405`; `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:349,362`; `IsaacLab/.../manager_based_rl_env.py:202-203,221,237,394`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:543,546`; `.../patch_slip.py:187,191-192,263`
**Action:** APPLIED to the page. **OPEN as a code bug and as a re-run decision:** the fix is to hoist the reset out of the step-counter guard. Training resets are correct because `_reset_idx` runs *inside* `step()` before the observation compute at the already-incremented counter.

### F-7002 · OPEN · PS · GROSS is a latch that only an episode reset clears
**Reality:** `retained_gross = (self.state == GROSS) & incipient_evidence` is OR'd into `gross_evidence`. Once a patch has been GROSS once, **any** incipient evidence — including a perfectly static `utilization = 0.65` with zero relative velocity — holds it at GROSS indefinitely, with `gross_evidence_count` sitting at 0. Measured: two fast-shear steps reach GROSS, then six frames of byte-identical static input return `GROSS / gross_slip = 1 / score 0.722` on every frame. Combined with F-7001, a swallowed reset carries that latch across an episode boundary.
**Evidence:** `.../patch_slip.py:263-264,269-270`, cleared only at `:135`
**Action:** APPLIED to the page. **OPEN as a design question.**

### F-7003 · CORRECTED · PS · three OR'd incipient sources, and utilization can never reach GROSS
**Claim (page):** "Its primary incipient trigger is `friction_utilization ≥ 0.60`" — the page's entire description of the state machine.
**Reality:** There are **three** independent incipient sources OR'd together, and utilization is the only one that cannot ever produce GROSS. `incipient_evidence = (utilization ≥ 0.60) | (shear_rate ≥ 0.5) | (pressure_drop_rate ≥ 2.0)`, gated by `contact_now`. `gross_candidate = (shear_rate ≥ 3.0) | (pressure_drop_rate ≥ 6.0)` — **utilization is absent**. Measured boundaries at `dt = 0.02`, load 1.0 N: pressure drop **3.9 %/step → STICK, 4.0 % → INCIPIENT, 12.0 % twice → GROSS**; flat pressure, `|Δshear| = 0.0100 N/step → STICK, 0.0101 → INCIPIENT, 0.0601 twice → GROSS`. Also unstated: `temporal_valid` zeroes *both* rate channels whenever the previous frame was not in contact, so **on the first frame of any contact only utilization can fire** — a genuine touchdown slip is structurally invisible, and a tilted touchdown is INCIPIENT on frame 1.
**Evidence:** `.../patch_slip.py:54-57,199-208,209-221,245-249,250-256`
**Action:** APPLIED.

### F-7004 · CORRECTED · PS · `gross_friction_utilization = 0.90` is a normalizer, not a threshold
**Reality:** It is **never used as a state threshold**. Its only use is as the denominator of `utilization_score = clamp(utilization / 0.90, min=0)`. The constructor nevertheless validates it as if it were a threshold ("gross friction threshold must exceed incipient"), and the parameter name asserts the same. Measured: `utilization = 50.0`, static, in contact → state **INCIPIENT**, score clamped to 2.0, `gross_slip = 0`. The name and the validation disagree with the code; the code wins.
**Evidence:** `.../patch_slip.py:53,76-77,223-225,250-256`
**Action:** APPLIED.

### F-7005 · CONFIRMED · PS · utilization ≈ 2·tanθ under stick — exact geometric boundaries
**Reality:** Under stick `‖ft‖ = k_t·|v_t| → 0`, so utilization reduces to `2·tan θ` where θ is the angle between the object's SDF normal and the taxel z-axis (the numerator is a cancelling vector sum, the denominator uses per-taxel `abs`, so cancellation can only *lower* it). Therefore **INCIPIENT at θ ≥ 16.70° with literally zero relative motion; `slip_score = 1.0` at θ ≥ 24.23°; the 2.0 clamp at θ ≥ 41.99°**. Measured: static tilted contact at `util = 0.70` → INCIPIENT on six consecutive byte-identical frames. Exact boundaries with no rate evidence: `0.59 → STICK`, `0.60 → INCIPIENT`, `1.80 → INCIPIENT (2.0 clamp)`, `50.0 → INCIPIENT (2.0)`. **Corollary: under perfect frame alignment utilization is bounded by 1.0, so any observed value above 1.0 is proof of geometric contamination.** Which source dominates in the real 3× rollouts is UNVERIFIED — no trace survives in this tree.
**Evidence:** `.../online_patch_tactile.py:125-126,144-151,237-239`; `IsaacLab/.../visuotactile_sensor.py:958-959,1005-1010,1018-1024`; `.../patch_slip.py:223-249`
**Action:** APPLIED.

### F-7006 · OPEN · PS · the thresholds were calibrated on a flat patch and applied to curved ones
**Reality:** No call site overrides any threshold — all three construction sites pass only `batch_size` and `device`. The docstring says they were "calibrated on a controlled official-R15 stick-to-slide trace"; that trace is `run_isaaclab_r15_capsule_slip.py`, a **flat R15 capsule** rig (single patch, `patch_size_m=((0.023977, 0.032001),)`) — a geometry in which the taxel-frame misalignment that dominates the anatomical hand **does not exist**. **The 0.60 threshold was fitted on a surface where utilization is bounded by 1.0, and applied on 54 curved pads where it is not.**
**Evidence:** `.../patch_slip.py:38-43,45-61`; `.../online_patch_tactile.py:450`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:608`; `scripts/sugar/native_tactile/run_isaaclab_r15_capsule_slip.py:360-369`
**Action:** APPLIED. **OPEN** — recalibration must happen on the anatomical geometry.

### F-7007 · OPEN · PS · the two binary slip channels are the loudest inputs to the projection
**Reality:** Channels 7 and 8 (`incipient_slip`, `gross_slip`) are **hard-coded to scale 1.0**, not fitted; only channel 6 gets a percentile. So after normalization the two binary flags enter the bias-free `9→128` projection at magnitude exactly 1.0 while every continuous channel sits at a fraction of its 99.5th percentile (measured elsewhere: shear at ~0.05). **The three channels that PS adds are the loudest inputs to the shared projection** — a mechanism for PS degrading relative to P that the page did not name.
**Evidence:** `scripts/sugar/native_tactile/fit_online_patch_channel_scales.py:55-65`; `.../patch_tactile_encoder.py:71-73,139-140`
**Action:** APPLIED.

### F-7008 · OPEN · PS · no minimum-load floor on the rate or utilization denominators
**Reality:** Both rate channels and utilization are guarded only by `epsilon = 1e-8`. A patch in contact whose taxel-frame `Σ|f_z| → 0` — the 90°-misalignment limit — yields `utilization = ‖shear‖/1e-8 = O(1e8)` and `shear_rate` likewise: an instant INCIPIENT with score clamped at 2.0, and two such frames give GROSS. The only load floor anywhere is `contact_loss_min_load_n = 0.02 N`, and it applies **only** to the contact-loss branch.
**Evidence:** `.../patch_slip.py:60,102,203,206,223-225,241`
**Action:** APPLIED.

### F-7009 · OPEN · PS · the detector returns four fields; the policy sees three
**Reality:** `PatchSlipOutput` = `(state int64 ∈ {0,1,2,3}, slip_score float32 ∈ [0,2], incipient_slip bool, gross_slip bool)`. `features()` stacks only the last three. `state` — the field the diagram node is *named* after — goes only to diagnostics and the evaluator trace. **Within the three slip channels the policy cannot separate STICK from NO_CONTACT**: both are `(score, 0, 0)`.
**Evidence:** `.../patch_slip.py:13-16,19-34,243,279-284`; `.../online_patch_tactile.py:41-46,469,487-493`
**Action:** APPLIED — SVG node relabelled.

### F-7010 · CORRECTED · PS · the §4 contract fails literally as well as in substance
**Reality:** Two separate failures. (a) **Literally**, `update` takes seven arguments and two are environment state, not patch signals: `timestamp_s = common_step_counter × step_dt` and `reset_mask = episode_length_buf == 0`. Both benign, but the statement as written is false at the call boundary. (b) **In substance**, the docstring's "accepts no contact-relative velocity" is wrong, as F-0008 said. What it genuinely does not read is object *pose*, mass, jump flag, reward or any future frame.
**Evidence:** `.../patch_slip.py:38-43,148-157`; `.../online_patch_tactile.py:453-468`; `IsaacLab/.../visuotactile_sensor.py:922,988,1005-1024`
**Action:** APPLIED; F-0008 refined.

### F-7011 · CONFIRMED · PS · `dt`, ring-buffer staleness, and the parts of the panel that are right
**Reality:** `timestamp_s` is global but only differences are used, so the origin is harmless; on the first post-reset frame `dt` is forced to a dummy 1.0 and never consumed because `temporal_valid` requires `prior_initialized`. Caveat: the detector **assumes** the control period rather than measuring it — the module already computes the real per-sensor clock in `current_whole_hand_patch_timestamps_s` and the slip path never uses it. The ring buffer can go stale but cannot skip a step; the cache is in fact *protective* for `get_observations()`, without which the detector would be updated twice at one timestamp and raise. And the page's core description is exactly right: `patch_slip.py` imports only `dataclasses` and `torch`, holds no sensor reference, and consumes exactly `base_history[:, -1]`.
**Evidence:** `.../patch_slip.py:129-130,188-197,209-221,272`; `.../online_patch_tactile.py:278-313,382-406,444-447,453,460-468,470-486,505-525`
**Action:** none needed beyond the F-7001 rewrite.

### F-5002 · CORRECTED · Z, P, PS · `hoi_contact` is DEAD, not anti-grasp — the headline was half wrong
**Claim (page):** "The reward is anti-correlated with grasping"; "the two terms jointly pay the policy to let go."
**Reality:** Half right, and the wrong half was the headline. Because `is_contact` is identically `False`, `hands_contact` returns `(False == contact_label).float()`, and `contact_label` is `motion.contact_label[motion_id, time_steps]` — a lookup into a `.npy` loaded from disk. **The term is a pure function of (motion id, timestep) and is completely independent of what the policy does**: the policy cannot raise it by releasing the box and cannot lower it by gripping harder. It contributes **no behavioural gradient at all** — only a time-varying alive bonus of `+1.0 × step_dt = +0.02`/step on reference-no-contact frames, entering the return solely through episode length. "Can never fire" and "pays out exactly when the reference says no contact" are both correct; **"pays the policy to let go" attributed to `hoi_contact` is not**. The genuinely anti-grasp force is `undesired_contacts` alone.
**Evidence:** `.../mdp/rewards.py:167-172`; `.../mdp/commands.py:82,105,135,1744-1745`; `IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:143-153`
**Action:** APPLIED — the page now attributes the anti-grasp pressure to `undesired_contacts` and describes `hoi_contact` as a dead term supplying zero gradient.

### F-5004 · CORRECTED · Z, P, PS · `undesired_contacts` — −0.02/body/step, and six bodies cancel everything
**Reality:** (1) **Reduction:** the term reads `net_forces_w_history`, takes `max` over the 3-frame history of the force norm per body, compares to `threshold = 0.1` N, and returns `torch.sum(is_contact, dim=1)` — a **count of bodies over threshold**, not a force magnitude. Because `contact_forces` has no filter list, this is the **net** force from *anything* (box, ground, self-contact), not box-specific. (2) **Magnitude:** `RewardManager.compute` applies `func × weight × dt` with `dt = 0.02 s`, so each violating body costs **−0.02 per step**, not −1.0. But the maximum achievable *positive* reward per unit time is **5.125** (13 exp-kernel tracking terms + a perfect `hoi_contact`), so **six bodies simultaneously above 0.1 N cancel the entire theoretical maximum positive reward** — and during a bilateral carry far more than six of the 54 patches are in contact.
**Evidence:** `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:260-268`; `IsaacLab/.../reward_manager.py:143-153`; `.../carry_box_refiner_env_cfg.py:96,137,140`; `.../base_refiner_env_cfg.py:71-73,303-399`
**Action:** APPLIED.

### F-5007 · CONFIRMED · Z, P, PS · nothing in the reward asks the policy to hold the box up
**Reality:** There is **no** mass-aware, force-aware or hold-the-box term anywhere in the 21. The entire positive signal is reference tracking (11 motion terms + 2 obj2body terms) plus the dead `hoi_contact`; everything else is a penalty. The only coupling from the mass jump to the objective is indirect: sagging degrades `motion_obj_pos`/`motion_obj_lin_vel`/`obj2body_pos`, and at 0.3 m of object-position error the `obj_pos` termination ends the episode. `hold_success`, `drop` and `safe_lower` exist **only in the evaluator**.
**Evidence:** `.../base_refiner_env_cfg.py:298-399,433-439`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:339-360`; `.../online_mass_jump.py:273-308`
**Action:** APPLIED.

### F-5006 / F-5011 · CORRECTED · Z, P, PS · the full reward inventory, and one sign-inverted term
**Reality:** 21 reward terms, 6 termination terms, all inherited unmodified. **Regularization (7):** `joint_acc` −2.5e-7, `joint_torque` −1e-5, `action_rate_l2` −0.1, **`joint_limit` −10.0 (largest magnitude in the stack)**, `feet_slide` −0.1, `feet_air_time` +5.0, `undesired_contacts` −1.0. **Reference tracking (11):** `motion_joint_pos` 0.125/std 0.6; `motion_global_anchor_pos` 0.25/0.3; `motion_global_anchor_ori` 0.25/0.4; `motion_body_pos` 0.25/0.3; `motion_body_ori` 0.25/0.4; `motion_body_lin_vel` 0.25/1.0; `motion_body_ang_vel` 0.25/3.14; **`motion_obj_*` 0.5 each** (2× the body-tracking weight). **Interaction (3):** `obj2body_pos` 0.25, `obj2body_ori` 0.25, `hoi_contact` +1.0. **Terminations (6):** `trajectory_complete` (`time_out=True`), `anchor_ori` 0.8, `ee_body_pos` 0.3 m, `obj_pos` 0.3 m, `obj_ori` 0.8 rad, `anchor_pos` 0.3 m — none reads a sensor.
**`feet_air_time` is sign-inverted:** weight **+5.0** on `feet_air_time_min_penalty`, which returns `(last_air_time − 0.5).clamp_max(0.0) × first_contact` — always ≤ 0. A positive weight on a non-positive function makes it a **penalty** of up to −5.0 per touchdown, gated off for the first 50 env steps. It is the only term whose sign is inverted between weight and function.
**Evidence:** `.../base_refiner_env_cfg.py:298-399,319-326,404-455`; `.../carry_box_refiner_env_cfg.py:84-116`; `.../mdp/rewards.py:19-138,243-269,313-327`; `.../mdp/terminations.py:26-79`
**Action:** APPLIED — full table added to the page.

### F-5009 · CORRECTED · Z, P, PS · four terms read the sensors; five ContactSensors are read by nothing
**Reality:** Four, not three — `feet_air_time` also reads `contact_forces` every step. Complete map: `feet_slide`, `feet_air_time`, `undesired_contacts` → `contact_forces`; `hoi_contact` → `left_hand_forces` + `right_hand_forces`. No termination reads any sensor. The scene declares **7** `ContactSensorCfg`s; the other five (both feet, both hips, pelvis) are constructed and stepped but read by **nothing** in training. The working patch↔box sensors live in the **audit** scene, which training does not use — so the only channel in the training scene that could ever have reported hand–box contact to the reward is the dead pair.
**Evidence:** `.../mdp/rewards.py:153-154,243-269,313-327`; `.../base_refiner_env_cfg.py:71-133,311-326`; `.../carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py:30-76`
**Action:** APPLIED.

### F-5012 · OPEN · Z, P, PS · IsaacLab `activate_contact_sensors` writes the sleep threshold to the wrong prim
**Reality:** The function is documented and evidently intended to zero each rigid body's sleep threshold, but line 527 reads `PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPrimPath())` — the loop's **outer** `prim` (the robot root) instead of `child_prim`. The contact-report API is applied correctly to `child_prim`; only the sleep threshold is misdirected, written 91 times onto the robot root and never onto any actual body. Upstream IsaacLab, shared by all branches — but it means sleeping bodies are a live possibility for contact reporting rather than the excluded case intended.
**Evidence:** `IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py:521-538` (bug at `:527`, correct `child_prim` use at `:529-535`)
**Action:** APPLIED as a footnote. **OPEN as an upstream defect.**

### F-6001 · OPEN · Z, P, PS · every reported number came from `--physical-outcome-view`, which disables terminations
**Reality:** **Every number on the page was produced with `--physical-outcome-view`**, which monkey-patches `termination_manager.compute`. The replacement calls the original (so each SUGAR term is still evaluated and retained as a *label*), then zeroes `_terminated_buf`/`_truncated_buf` and returns all-`False`. `reset_env_ids` is therefore empty and `_reset_idx` is never called: **the rollout keeps running through anchor / end-effector / object reference violations for the full 450 steps.** The evaluator also stubs `_reset_idx` to a no-op and guards `done_latched` behind `if not args.physical_outcome_view`, so `valid_frame` is a **constant-True array** and the obs/action zeroing is dead code. Consequence: **`eligible` no longer means "the rollout stayed inside the SUGAR contract" — it reduces to "the mass jump landed by frame 370".** The evaluator emits stricter `strict_sugar_eligible` / `strict_sugar_hold_success` labels that the page never reported. Both downstream tools *require* the flag.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:494-508,547-548,607,729-730,740`; `IsaacLab/.../manager_based_rl_env.py:205-222`; `scripts/sugar/native_tactile/run_plan15_frozen_seed.sh:63`
**Action:** APPLIED to the results section and glossary. **OPEN** — the strict-view numbers exist in the summaries and have never been reported.

### F-6002 · CORRECTED · Z, P, PS · the 59 denominator, and why it cannot bias the comparison
**Reality:** One profile of the 60 fails `eligible_post_jump_window`. Crucially the eligible set is **branch-invariant and factor-invariant by construction**: the mass controller qualifies off the *frozen Refiner's* pickup, the delay is a closed-form function of `(seed, episode_index, env_id)` only (`10 + code % 41`), the wrapper executes the teacher for every pre-handoff step, and `apply_pending` fires unconditionally once the delay elapses — a student that has already dropped the box still gets its jump. **So there is no "a branch that drops earlier loses more profiles" bias.** Confirmed arithmetically by the page's own statistic: 33/59 − 49/59 = **−16/59 = −0.27119 → the reported −0.2712**; 22/59 − 8/59 = **+14/59 = 0.23729 → +0.2373**. Both land exactly on 59, so P and PS had *identical* eligible sets. Which profile is dropped is UNVERIFIED — no summaries exist in this checkout.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:299-330`; `.../online_mass_jump.py:104-120,152-174,195-241`; `.../online_teacher_handoff.py:62-89`
**Action:** APPLIED. This is a genuine *strengthening* of internal validity and belongs on the page alongside the criticisms.

### F-6003 · CORRECTED · Z, P, PS · the CI is optimistic — 3 clusters, percentile, 180 uncorrected intervals
**Reality:** A **paired two-level cluster bootstrap with a percentile interval — not BCa, not studentized, no bias correction**. Pairing is at the **profile** level; clustering at the **training-seed** level (level 1 draws 3 seeds with replacement, level 2 draws profiles within each). Ways it is narrower than the design justifies: (1) **three clusters** — only 10 distinct seed multisets exist, so the 2.5/97.5 percentiles are resolved by essentially nothing and coverage at this cluster count is well below nominal; (2) **percentile, not BCa**, on a bounded, discrete, skewed mean of ±1/0 differences; (3) **no multiplicity control** — one invocation writes **12 metrics × 5 factors × 3 comparisons = 180 intervals** and the page promoted one to "significant"; (4) the 20 within-cluster profiles are near-replicates (F-6012).
**Evidence:** `SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py:15-34,107-205`
**Action:** APPLIED — now presented as a strong point estimate with an optimistic interval, not a controlled 5 % test.

### F-6012 · OPEN · Z, P, PS · a "profile" is a near-replicate, not a fresh scenario
**Reality:** The 20 profiles inside one (seed, factor) run are near-replicates. Every profile starts from **motion 45 frame 0**; `init_with_ref = False` and `start_init_env_ratio = 1.0` remove start-frame randomization; `push_robot`/`push_object`/`obj_mass` are `None`; observation corruption is off on all five groups. Only two things differ: the deterministic jump delay `10 + code % 41`, and the **startup** friction draws — which run *once per process*, so the four env slots keep four fixed friction values that **repeat across all five batches**. Profiles p and p+4 share a box friction and differ only in jump timing. This is what the bootstrap treats as 20 exchangeable within-cluster units.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:443-452,479-487,538,549-582`; `.../carry_box_online_patch_tactile_mass_env_cfg.py:212,253-259`; `.../base_refiner_env_cfg.py:262-284`
**Action:** APPLIED.

### F-6016 · CORRECTED · Z, P, PS · 1× writes no mass at all
**Reality:** At `--mass-factor 1.0` **no mass or inertia write happens**: `apply_pending` selects `changed_ids = ids[target_factor != 1.0]`, which is empty, so `_write_mass` is never called and `mass_changed` stays False for the whole rollout; only the `jump_applied` marker and the eligibility clock fire. **The 1× column is a pure no-perturbation control, not a "1× jump".**
**Evidence:** `.../online_mass_jump.py:152-174,215-241`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:701-703`
**Action:** APPLIED.

### F-6007 / F-6008 · CORRECTED · Z, P, PS · `drop` has a second disjunct, and there is a fifth unnamed bucket
**Reality:** `drop = height_loss >= 0.15` **or** `min(z over window) <= initial_z + 0.03`, where `initial_z` is the object's height on the *first recorded frame of the batch* — the table height, not the jump height. The second disjunct is a "returned to the table" test and is what makes the 6×/10× columns saturate; it also makes hold and drop **not strictly mutually exclusive** for very low-lift jumps. And `safe_lower = eligible ∧ ¬hold ∧ ¬drop ∧ ¬fall ∧ min z ≤ initial_z + 0.08 ∧ min vz ≥ −0.35 ∧ max ref-ori error ≤ 0.8`. **A profile that is eligible, not a hold, not a drop, and not a controlled lowering is in none of the four labels** — the Z row shows it: at 3×, 52 + 2 = 54 of 59, leaving **5 unaccounted**.
**Evidence:** `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:297,339-360,827-833`
**Action:** APPLIED.

### F-6009 / F-6011 / F-6014 · OPEN · Z, P, PS · latched labels, unchecked provenance, unclamped reference clock
**Reality:** (F-6009) `TerminationManager` only rewrites a `_term_dones` row on frames where it fires, and `reset()` does not clear the buffer — so with resets suppressed the traced termination flags are **latched, not per-frame**. `reference_robot_deviation` is `.any()` over the post-jump window, so a deviation that fired during the *teacher-driven pickup* still reads True inside the window — and `compare` uses that field as a headline metric. (F-6011) `compare` enforces the 3×5×20 design, the seed pairing and the evaluation view, but **does not reject camera-enabled or friction-overridden runs**, and does not require 20 eligible profiles — a degraded denominator passes silently. (F-6014) With resets suppressed, `_update_command` increments `time_steps` with **no clamp**, and motion tensors are zero-filled past each clip's length — so if motion 45 is shorter than ~451 frames the rollout continues against an **all-zero reference**, voiding both reference-error columns and making `safe_lower` structurally unreachable. UNVERIFIED here (`SUGAR/data/` absent).
**Evidence:** `IsaacLab/.../termination_manager.py:126-188`; `SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py:344-346,547-548,607,661-675`; `.../compare_online_patch_mass_sweeps.py:30,66-104`; `.../mdp/terminations.py:69-79`; `.../mdp/commands.py:84-121`
**Action:** APPLIED to the page and `operations.md`.

### F-6005 / F-6010 / F-6013 / F-6015 / F-6017 · CONFIRMED / CORRECTED · consolidated
**Reality:** (F-6005) The evaluator's `PatchSlipDetector` is a **second, independent instance** from PS's observation-term detector, fed with a different timestamp origin (`(step+1)·step_dt` vs `step·step_dt`) and reset mask — the traced `slip_state` **re-derives** PS's slip rather than recording it. (F-6010) The friction table's hold/drop labels are **not** the branch table's: they come from `preflight_online_patch_mass_jump.py` at seed 150814, computed over *every frame after the jump* rather than an 80-frame window, with no robot-fall gate and no return-to-table disjunct. (F-6013) One run emits `frozen_evaluation_trace.npz` (23 arrays, leading shape `[450, 20, …]`) and `summary.json` (schema `plan15_frozen_online_patch_mass_evaluation_v3_live_handoff`); a jump-less profile gets a **reduced 14-key record** whose bilateral-contact field changes name *and* units, absorbed silently downstream. (F-6015) Full flag set: `--motion-id 45 --profiles 20 --num-envs 4 --max-steps 450 --post-jump-window 80 --physical-outcome-view --headless`; friction overrides and `--record-world` were **not** used for the reported numbers. (F-6017) The launchers write one root per branch containing 15 run subdirectories.
**Evidence:** as cited in the agent reports; all carry path:line.
**Action:** APPLIED where page-facing.
