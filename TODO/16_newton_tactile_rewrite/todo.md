# TODO 16 — Newton tactile rewrite

Design: [`PLAN/16_newton_tactile_rewrite/plan.md`](../../PLAN/16_newton_tactile_rewrite/plan.md).
Audit that motivated it: [`claude_context/findings.md`](../../claude_context/findings.md).

Code lives in this repo at `sugar_newton/`, on branch `2026_8_19_sugar_newton` (branched
from `sugar`, so the SUGAR source, the teacher and BCPPO are all present to port from).
Plan 16 §3 — the package *depends on* Newton and never vendors or patches it; `git diff`
against upstream Newton must stay empty.

---

## A. Phase 0 — prerequisites (blocking, not code)

- [x] **Assets fetched — from the public SUGAR release, not the runtime host.** The premise
      here was wrong: they are not only at `/public/home/yanhongru/Curiosity` (a path on a
      different cluster, not mounted on OCI-ord). SUGAR publishes them from its README's
      "download data" section as three Google Drive archives. `bash SUGAR/_downloads/fetch_assets.sh`
      pulls and unpacks all three into the already-gitignored `SUGAR/{descriptions,data,demo_ckpts}`.
      Provenance and every hash: `SUGAR/_downloads/MANIFEST.md`. Got 4 of 6:
      - [x] `SUGAR/descriptions/robots/g1/meshes` — 165 files / 137 MB, with
            `g1_29dof_rev_1_0_with_rubber_hand.urdf`
      - [x] `SUGAR/data/CarryBox` — 100 clips / 101 MB (all six tasks: 565 MB)
      - [x] the official Tracker checkpoint — `demo_ckpts/CarryBox/tracker.pt`
            (`generator.ckpt` came with it, not on the original list)
      - [x] the official CarryBox asset — `descriptions/objects/{big_box,small_box}`
      - [ ] **teacher `refiner_model10000.pt` — NOT in the public release.** `demo_ckpts/`
            ships tracker + generator only. The refiner is step 1 of SUGAR's own `train.sh`
            (`Sugar-G129dof-CarryBox-Refiner`, 4096 envs, 30001 iters), so that filename was
            an artifact of our run, not a published file. Its **training config is present**
            (`train_refiner/carry_box_refiner_env_cfg.py` + the `..._tacsl_audit_` variant).
            Either copy it off the runtime host or retrain — §D already budgets for retraining.
      - [ ] `gelsight_r15_finger.usd` — a TacSL asset, not a SUGAR one, so it is not in this
            release. §C says do not port the TacSL sensing half; confirm it is needed at all
            before chasing it.
- [x] **SHA-256 recorded for every archive and for the key individual assets** —
      `SUGAR/_downloads/MANIFEST.md`. Plan 15's teacher pin was disabled
      (`expected_sha256=None`); this is the thing that stops a repeat.
- [ ] **Newton env on this cluster.** Container recipe is in
      `Curiosity_newton/renders/build_and_render.sh` (uv sync inside an interactive CUDA
      container). A CPU-only login-node env also works for Phase 1 — see §B.

## B. Phase 1 — tactile core + validator (**passing**, see `sugar_newton/README.md`)

- [x] Confirm Newton imports and runs on the OCI-ord **login node, CPU device** — no
      container, no GPU. `warp 1.15.0.dev20260612`, `newton 1.4.0.dev0`, `mujoco_warp`.
- [x] `PatchTactile` reducer: rigid contacts → per-patch channels, as warp kernels with
      atomic reduction over contacts. `sugar_newton/tactile/reducer.py`.
- [x] Anchor propagation across frames via `rigid_contact_match_index`
      (`contact_matching="latest"` — **not** `"sticky"`, which perturbs the solve).
- [x] **Incline validator** with the analytic assertions of Plan 16 §5.
      `sugar_newton/validation/incline.py`, exits 0.
      Normal load matches `mg cos θ` to 4 decimals; load-weighted utilization matches
      `tan θ / μ` to 4 decimals; **slip is zero while sticking**; the two independent
      slip estimates (anchor drift, relative velocity) agree to ~2%.
- [x] **Index-alignment question resolved: the ordering round-trips.**
      `SolverMuJoCo.update_contacts` (`solver_mujoco.py:4380-4411`) replaces the entire
      contact set with MuJoCo's own, but hands it back in the same order. Verified by
      comparing contact *positions*, not just shape pairs and normals — in a
      single-pair scene those cannot distinguish a permutation. The validator now fails
      loudly if this stops holding.
- [x] Runtime reporting of `utilization > 1 + ε` via `PatchTactile.utilization_overflow`.
- [x] **`gross_slip_fraction` channel added** after the validator caught a silent zero:
      the matcher breaks a match once a contact moves more than
      `contact_matching_pos_threshold` (0.5 mm) per step, so a fully sliding patch
      re-anchors every frame and its anchor drift reads **exactly 0.0**. Anchor drift
      measures *incipient* slip only. The re-anchor fraction reads 0.000/1.000 across
      the stick→slide transition with no threshold anywhere.

### B-open — do not mark Phase 1 done until these close

- [x] **Material-μ half verified** — `sugar_newton/validation/friction.py`, exits 0 on CPU.
      Utilization tracks `max(mu_a, mu_b)` to 4 decimals across μ = 0.3…0.9 (spread
      0.4723), and swapping μ between the two shapes leaves the reading unchanged, which
      is what proves the pair rule is MAX rather than one shape's value. The fallback is
      set to an absurd 7.0 so any silent fall-through fails the test.
- [x] **Bug found and fixed by writing that test.** The reducer used
      `rigid_contact_friction` *as* μ. It is not μ — it is a per-contact **scale**
      (default 1.0) written by hydroelastic reduction for moment matching
      (`contact_reduction_hydroelastic.py:885`); MuJoCo multiplies the resolved material
      friction by it (`kernels.py:460-468`), and the pair itself combines by elementwise
      max (`kernels.py:165`). Correct form: `mu_contact = max(mu_a, mu_b) * scale`.
      The original incline test could not catch this because both shapes had μ = 0.5, so
      `max` coincided with the fallback. Plan 16 §4 corrected too.
- [x] **Scale half run on GPU — passes.** A100-SXM4-80GB, in the CUDA container via
      `sugar_newton/gpu_run.sh`. All four cases match, normal load 4.7979 N across the
      board, A == B to 3 decimals.
      Two things this took, both worth remembering:
      **(a) `njmax` must be sized for hydroelastic.** At 64 MuJoCo warned
      `nefc overflow - please increase njmax to 93` and silently dropped constraint rows;
      the symptom was an inflated normal load and a case reading 0.7848 vs 0.7085, not an
      error. Now 1024/512.
      **(b) Geometry must be sized to the SDF.** An 8 m ramp at `sdf_max_resolution=64`
      gives 12.5 cm voxels against a 1 cm narrow band, so hydroelastic found no contact
      surface at all — every case read zero contacts. Hydroelastic now defaults to a
      0.25 m ramp. Also dropped `gap`, which gates MuJoCo contact activation.
- [ ] **The scale is still only confirmed as a no-op.** `rigid_contact_friction` reads
      exactly `1.0` for every contact in this scene, so `mu * scale` is verified to not
      corrupt the value, and **not** verified against a non-trivial factor. Moment-matching
      only produces scale != 1 when a contact patch is large enough for hydroelastic
      reduction to actually reduce — which the 54-pad hand will produce naturally.
      Re-check there, and do not describe audit #4 as fully closed until then.
- [ ] **`validation/incline.py` currently exits 1** on the one sliding case
      (theta = critical + 5): time-averaged normal load reads 6.01 N against 4.18 N
      expected. This is the *measurement*, not the sensor — the 40-step window is far too
      short to average a bounce cycle, so it is biased by whatever phase it lands in. The
      four sticking cases pass to 4 decimals. Fix the window (or drive the sliding case
      at prescribed velocity) rather than loosening the tolerance.
- [x] **Contact area and peak pressure (channels 9-10) implemented and validated.**
      `reduce_contact_surface_kernel` + `finalize_pressure_kernel` in the reducer;
      `validation/pressure.py` exits 0 on GPU. Seated on a ramp: contact area reads
      **96.90 cm² against the block's 100.00 cm² face**, and peak/mean pressure climbs
      1.570 → 1.654 → 2.005 across θ = 5/12/20°, which is the gravity moment shifting
      load to the downhill edge.
      **Plan §4's definition was wrong twice and this test caught both.** `kh·depth` is
      the hydroelastic law, but with `use_mujoco_contacts=False` the force is MuJoCo's
      constraint solve — `∫ kh·depth dA` measured **328.8 N against a true 4.886 N**.
      And the sign was inverted: `depth < 0` is penetration. Channel 10 now scales the
      depth field to integrate to the solved `normal_load`. Plan §4 corrected to match.
      The surface only reaches the reducer when the pipeline is built with
      `HydroelasticSDF.Config(output_contact_surface=True)` — off by default.
- [ ] Prescribed-velocity sliding scene for a *quantitative* slip test; the free-slide
      assertions are deliberately qualitative because a bouncing block's finite-differenced
      speed and an instantaneous tactile reading are not the same quantity.
- [x] Run on GPU — done, A100. Sticking cases reproduce the CPU numbers to 4 decimals.
- [ ] Run with more than one world.
- [ ] **Video.** `validation/render_friction.py` (pass 1, in-container) +
      `validation/compose_friction_video.py` (pass 2, login node: matplotlib + ffmpeg).
      Sweeps mu down through the critical value at a fixed 20 deg so the clip shows
      stick -> incipient -> gross slip with nothing else in the scene changing.
      Two traps found: `render_env.sh` exports its own `OUT`, and `G1_XVFB=1` means a
      *windowed* GLX context -- setting `pyglet.options['headless']` forces EGL, which
      has no usable device here. Camera is workable but still frames the ramp too wide.

## C. Phase 2 — asset and throughput

- [ ] G1 29-DoF from Newton's stock `g1_29dof_with_hand_rev_1_0.usda`
      (`newton/examples/robot/example_robot_g1.py:45`) — same 29-D action space.
- [ ] Port the 54 anatomical patches onto `ModelBuilder` as hydroelastic mesh shapes.
      Source geometry: `anatomical_whole_hand_tacsl_g1.py` (27/hand: palm 4×3, plus
      proximal/middle/distal on five digits). Do **not** port the TacSL sensing half.
- [ ] **Throughput benchmark: worlds × patches × `sdf_max_resolution` → fps.** This
      decides whether the rewrite delivers its headline motivation. Known baseline: 82 fps
      with **2** pads in **1** world, and the path is collision-bound on SDF queries.
      Run this before building anything on top of it.
- [ ] Decide the patch-count/resolution operating point from that curve, and record it.

## D. Phase 3 — observations and the teacher gate

- [ ] Port the 16 observation terms of the 890-D teacher observation
      (`base_refiner_env_cfg.py:219-243`) onto Newton state. All are functions of
      reference motion, articulation state or rigid-body state — none touch PhysX.
- [ ] Motion command manager (future frames, anchors) on the ported `data/CarryBox`.
- [ ] Load `refiner_model10000.pt`, run open-loop. **Gate: does it still lift the box
      under MuJoCo-Warp?** Record the answer either way.
- [ ] If it does not: retrain a teacher in Newton from the reference motion. Budget for
      this — it is the most likely schedule risk after the throughput benchmark.

## E. Phase 4 — env and learning

- [ ] Vec-env implementing the `rsl_rl` VecEnv protocol; torch↔warp interop following
      `newton/_src/solvers/kamino/examples/rl/`.
- [ ] Wire `BCPPO` unmodified (`rsl_rl_bcppo.py`). Keep both teacher roles: distillation
      target *and* acting policy for the episode prefix.
- [ ] **Reward, built correctly from the start:**
      - [ ] patches excluded from any undesired-contact penalty (audit #1)
      - [ ] contact reward pointed at bodies that actually have collision (audit #2)
      - [ ] a term that rewards holding the box (audit #3)
      - [ ] verify each term's sign and gradient with a unit test — Plan 15 shipped
            `feet_air_time` at weight `+5.0` on a function that is always ≤ 0
- [ ] Port the mass-jump event (the one part of Plan 15 the audit found sound: written at
      the action boundary, inertia scaled by exactly `target/default`, both values read
      back).
- [ ] Confirm gradient isolation for any zero-tactile control branch by **measurement**,
      as Plan 15 did (all encoder params in the optimizer, every gradient exactly zero).

## F. Phase 5 — experiment protocol

- [ ] Train and evaluate on the same motion distribution (audit #8). Plan 15 trained on
      motions 0-3 and evaluated only on motion 45; every headline number was
      out-of-distribution and nothing flagged it.
- [ ] Report strict terminations alongside any physical-outcome view (audit #9).
- [ ] Interval that survives scrutiny (audit #10): more than 3 seed clusters, or BCa, or
      both — and state multiplicity.
- [ ] Profiles that are not near-replicates: vary initial frame, add push and observation
      randomization, not just a deterministic jump delay.

## G. Carried-forward rules

- [ ] Never patch vendored Newton in place (Plan 16 §3). A named patch file with a test,
      or upstream, or in our package.
- [ ] Never commit checkpoints, traces, videos or logs.
- [ ] Never claim "only tactile can sense mass" — mass leaks into proprioception through
      joint sag and tracking error.
- [ ] A patch is the policy unit, never a contact point.
