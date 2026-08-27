# TODO 16 — Newton tactile rewrite

Design: [`PLAN/16_newton_tactile_rewrite/plan.md`](../../PLAN/16_newton_tactile_rewrite/plan.md).
Audit that motivated it: [`claude_context/findings.md`](../../claude_context/findings.md).

Code lives in this repo at `sugar_newton/`. It was developed on
`2026_8_19_sugar_newton` and integrated into `sugar` on 2026-08-27 after comparison with
the newer SUGAR/BCPPO code. Newton execution is active on the retained H200 Slurm
allocation; simulation and training are launched only through its tmux/srun panes.
Plan 16 §3 — the package *depends on* Newton and never vendors or patches it; `git diff`
against upstream Newton must stay empty.

---

## A. Phase 0 — prerequisites (blocking, not code)

- [x] **Assets fetched — from the public SUGAR release, not the runtime host.** The premise
      here was wrong: they are not only at `/public/home/yanhongru/Curiosity` (a path on a
      different cluster, not mounted on OCI-ord). SUGAR publishes them from its README's
      "download data" section as three Google Drive archives. `bash SUGAR/_downloads/fetch_assets.sh`
      pulls and unpacks all three into the already-gitignored `SUGAR/{descriptions,data,demo_ckpts}`.
      Provenance and every hash: `SUGAR/_downloads/MANIFEST.md`. Got 5 of 6:
      - [x] `SUGAR/descriptions/robots/g1/meshes` — 165 files / 137 MB, with
            `g1_29dof_rev_1_0_with_rubber_hand.urdf`
      - [x] `SUGAR/data/CarryBox` — 100 clips / 101 MB (all six tasks: 565 MB)
      - [x] the official Tracker checkpoint — `demo_ckpts/CarryBox/tracker.pt`
            (`generator.ckpt` came with it, not on the original list)
      - [x] the official CarryBox asset — `descriptions/objects/{big_box,small_box}`
      - [x] **teacher `refiner_model10000.pt` — RECOVERED 2026-08-21, Phase 0 is now
            complete.** Never in the public release (`demo_ckpts/` ships tracker +
            generator only) because it was step 1 of SUGAR's own `train.sh` and therefore
            an artifact of *our* run. It was in our archived workspace on the Hub:
            `Railgun526/curiosity-workspace-large-files-20260708`, under
            `experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/`.
            Downloaded to the exact path the scripts expect
            (`SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py:16`); 14 957 503 B,
            sha256 `a398a729…16124c3`. Verified real: `iter = 10000`, actor
            `890 -> 512 -> 256 -> 128 -> 29`, matching critic. **This makes the §D teacher
            gate runnable instead of a retrain budget.**
      - [ ] `gelsight_r15_finger.usd` — a TacSL asset, not a SUGAR one, so it is not in this
            release. §C says do not port the TacSL sensing half; confirm it is needed at all
            before chasing it.
- [x] **SHA-256 recorded for every archive and for the key individual assets** —
      `SUGAR/_downloads/MANIFEST.md`. Plan 15's teacher pin was disabled
      (`expected_sha256=None`); this is the thing that stops a repeat.
- [x] **Newton env on this cluster.** H200 job 262332 on server63 uses the exact
      `third_party/newton` submodule with the shared Python 3.12 environment and Warp
      1.15 checkout through `srun --overlap`; no simulation runs on the login node.

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
- [x] **Free-slide phase bias separated from the quantitative gate.** On H200 the
      critical+5 case reads 9.2076 N against 4.1793 N over its phase-local 40-frame
      window; extending the window lets the block leave the ramp, so this accelerating,
      chattering case is not a seated-equilibrium measurement. `incline.py` now reports
      its normal load as a note while retaining the qualitative slip assertions. The
      quantitative gate is the prescribed scene below; no force tolerance was loosened.
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
- [x] **Prescribed-velocity quantitative slip gate passes on H200.** The first version
      used `add_body` plus two extra prismatics, creating unsupported loop joints; it
      falsely passed with 0/27 contacts. `validation/hand_map.py` now uses Newton's
      supported kinematic-root prescribed motion against a constrained dynamic load-cell
      plate and fails on zero contact/load. Exact carriage speed is 0.0500 m/s and the
      load-weighted tactile slip is 0.0502 m/s, with 4/27 patches contacting, 5074.82 N
      peak normal load, exact position tracking and no hydroelastic buffer overflow.
- [x] Run on GPU — done, A100. Sticking cases reproduce the CPU numbers to 4 decimals.
- [ ] Run with more than one world.
- [ ] **Video.** `validation/render_friction.py` (pass 1, in-container) +
      `validation/compose_friction_video.py` (pass 2, login node: matplotlib + ffmpeg).
      Sweeps mu down through the critical value at a fixed 20 deg so the clip shows
      stick -> incipient -> gross slip with nothing else in the scene changing.
      Two traps found: `render_env.sh` exports its own `OUT`, and `G1_XVFB=1` means a
      *windowed* GLX context -- setting `pyglet.options['headless']` forces EGL, which
      has no usable device here. Camera is workable but still frames the ramp too wide.
- [x] **Channels 1-10 as a continuous FIELD, not a per-link table.**
      `sugar_newton/tactile/field.py` (`ContactField`) samples pressure, tangential
      traction and slip velocity once **per contact-surface triangle**, in a chosen body
      frame, alongside the per-patch reduction and off the same solved loads.
      `validation/compose_allegro_field.py` renders it as three continuous maps
      (area-weighted Gaussian splat, one sequential hue each) over a silhouette of the
      hand, with the loaded links direct-labelled.
      Two of the three channels are genuinely per-face and one is not, and the module
      says so: **traction is the patch's measured friction load distributed across its
      faces in proportion to pressure**, which reduces to `tau_i = p_i * F_t / F_n`.
      It integrates to the measured friction load by construction; it is an estimate of
      the *shape*, not a per-face measurement. Newton's contact buffer has no per-triangle
      friction to read -- reduction collapses a patch to a few contacts before the solve.
      Trap while writing it: **`wp.float32` is not a numpy dtype.** `np.zeros(0, wp.float32)`
      silently yields an OBJECT array, `np.concatenate` keeps it, and the failure only
      surfaces at `np.load` as "Object arrays cannot be loaded when allow_pickle=False" --
      one whole recording later. (`wp.vec3f` happens to work, via ctypes.) Spell numpy
      dtypes out when building empty arrays to match a warp buffer.
- [x] **The Allegro scene was never holding the cube properly, and the cause was one
      missing line.** The stock example overwrites the root joint's parent rotation with
      `hand_rotation = normalize(quat(0.21643, 0.706218, -0.648166, 0.185191))`, which
      turns the palm **up**; our scene skipped it, so the hand sat in the USD's own tilted
      pose and the cube balanced on the palm edge, out of the thumb's reach and sunk into
      the proximal links. Every one of ~20 drive settings swept before that line was added
      dropped the cube on the floor (`reach` from the palm ~1 m). With it, the cube is
      cradled and worked: **45 deg of rotation, 92 mm of travel, contact on 2-5 links for
      the whole clip**, penetration 1.0-1.7 mm steady on a 50 mm cube.
      Three other numbers are load-bearing and were each established by
      `validation/allegro_grasp_sweep.py`, not by taste:
      **(a) `kh` 1e8 -> 1e10.** At 1e8 the contact is soft enough that the cube sinks
      through the fingers and escapes regardless of the drive -- the stiffness is not
      cosmetic, it is what makes the grasp exist.
      **(b) the digits must ease into the grasp.** A 0.2 rad step on 16 dofs at
      `ke = 150` on frame 0 is a snap-close, and it flicks the cube out.
      **(c) the swing rides on the flexion dofs only.** Driving each digit's `_0` dof
      (spread, and the thumb's opposition) with the same sinusoid unpicks the thumb from
      the grasp. The earlier "manipulate harder" patch (amplitude 0.22, rate 2.2) did
      neither and **threw the cube 1.15 m in 3 s with contact on 1 frame out of 150**.
- [x] **`joint_label` is per JOINT and must not be indexed by dof.** It includes the fixed
      root, mount and biotac-tip joints, which carry no coordinates, so the stock example's
      `for i in range(joint_dof_count - 6): ... joint_label[i][-2:] == "_0"` lands on the
      wrong joint -- it sets 0.6 on `index_joint_2`, `middle_joint_1`, `ring_joint_2`, not
      on the spread dofs it means. Bridge with `joint_q_start` instead.
- [x] **`HydroelasticSDF.Config(buffer_fraction=0.4)` overflows on this scene**
      ("iso subblock L1 overflow: 1789 > 1670"). An overflow silently DROPS contacts,
      i.e. punches holes in the field. Now 1.0 with `buffer_mult_iso=2`.
- [ ] **Penetration is ~1 mm and contact stiffness does not fix it.** Measured on 8 GPUs
      in parallel, `ke` from 1e3 to 1e6 with matched damping gives 1.16 / 1.20 / 1.23 /
      1.27 mm (p99 of the deepest face) -- flat. An earlier single run that read 1.22 ->
      0.40 mm was noise, not signal. Typical depth over a clip is lower than the p99
      suggests: median 0.74 mm, mean 1.01 mm, p99 2.06 mm on a 50 mm cube. What to try
      next is `solimp`, not `ke`.
- [ ] **Contact stiffness and damping must still be raised together.** `ke` (not `kh`) is
      the stiffness the solve sees, but `convert_solref` (`mujoco/kernels.py:192`, called at `:432`) yields
      a damping ratio of `(kd/2)*sqrt(factor/ke)` -- so raising `ke` alone divides the
      damping by `sqrt(ke)`. Going 1e3 -> 1e5 at a fixed `kd = 1e2` takes the ratio from
      ~1.5 to ~0.15, and the springy contact threw the cube **several metres**. Scale `kd`
      as `sqrt(ke)`. Also: **`--substeps 4` is unstable** -- verified standalone at two
      stiffnesses, the cube is flung ~1.1 m; 16 dropped it too. 8 is the operating point,
      and it is ~20 fps.
- [x] **The viewer was rendering on the CPU, and one missing file was why.** The image
      ships `libEGL_nvidia.so.0`, but `/usr/share/glvnd/egl_vendor.d/` contained only
      `50_mesa.json`, so EGL enumerated zero NVIDIA devices and `render_env.sh` worked
      around it with `LIBGL_ALWAYS_SOFTWARE=true` + Xvfb + mesa swrast -- the A100 idle
      while the CPU drew every frame. Writing `10_nvidia.json` makes
      `gl_info.get_renderer()` report `NVIDIA A100-SXM4-80GB`. Now in
      `setup_container.sh`; use `renders/render_env_egl.sh` (no Xvfb, so parallel render
      jobs no longer fight over `DISPLAY :99`).
- [x] **PNG encoding cost more than the physics.** 106 ms/frame against a 56 ms step.
      JPEG is 8.3 ms. `--image-format jpg`.
- [ ] **ViewerRTX is not the fast path, and is not installable here.** `ovrtx` fails to
      build in this container. It is also a path tracer: for a 19-shape scene GL
      rasterisation is 8.8 ms/frame against a 56 ms physics step, so rendering is already
      6x cheaper than the thing it would be waiting for. RTX is worth revisiting for
      *image quality* (domain randomisation), not for throughput.
- [ ] **End to end, 300 frames: ~28 s of process time for sim+render+encode, ~88 s to
      composite the figure.** The composite is ~265 ms/frame and is 90 % matplotlib's
      `canvas.draw`. It is a debugging artefact, not part of a training loop -- optimise
      it only if clips are being made constantly. Reusing artists instead of rebuilding
      the figure was already 7x; freezing tick locators another 1.4x.
- [ ] **Speed is the MuJoCo solve, not the tactile stack.** Profiled: `solve` 95 %,
      `collide` 4 %, reducer + field together 0.4 %. At `iterations=100` an EMPTY scene
      still costs 4.3 ms per substep, i.e. kernel-launch overhead, not arithmetic -- one
      environment cannot amortise it. ~20 fps at `--substeps 8`. CUDA-graph capture is
      implemented (`AllegroTactileScene.capture`) and needs
      `mjw_model.opt.graph_conditional = False` on this cluster, whose driver is 12.2
      while conditional graph nodes need 12.4.
- [ ] **Third law holds.** With the drive off, the tactile sensor weighs the cube:
      2.1187 N measured against a true 2.1190 N (ratio 0.9999). Note the SIGN -- the
      reducer orients force onto the patch, so it reports the force the cube exerts on the
      hand, along `m g`, not against it. `validation/allegro_static.py`.
- [ ] **This scene is chaotic and not bit-reproducible across GPU nodes.** The same
      config run on batch-block1 and batch-block7 gives 45.4 vs 52.4 deg of cube rotation
      and 2.17 vs 2.70 mm peak penetration. Report ranges, not single figures, and do not
      treat a one-run number as a regression signal.
- [ ] **`AllegroTactileScene.reset` does not reset the solver.** The sweep reuses one
      `SolverMuJoCo` across candidates, and its warm-start/contact caches carry over: a
      setting that survives 300 frames in its own process was scored as dropped when it
      ran fifth in a row. The sweep is a screen; re-run winners standalone. Fix by
      rebuilding the solver in `reset`, or accept and document.

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

- [x] Port the 16 observation terms of the 890-D teacher observation
      (`base_refiner_env_cfg.py:219-243`) onto Newton state. All are functions of
      reference motion, articulation state or rigid-body state — none touch PhysX.
      `obs_890.py` asserts the exact 890-D layout. The 2026-08-27 merge additionally
      separates aligned Refiner-rollout student motion from raw teacher motion; numerical
      parity against an Isaac 890-D dump remains an evaluation item, not a completed gate.
- [x] Motion command manager (future frames, anchors) on the ported `data/CarryBox`.
      It matches SUGAR's `data_{motion_id}_{env_id}` → `data_{motion_id}` alignment and
      rejects missing teacher IDs or a length mismatch greater than two frames.
- [x] Load `refiner_model10000.pt` and run the fixed 20-profile open-loop gate. The exact
      checkpoint SHA and `890 -> 512 -> 256 -> 128 -> 29` actor load strictly. All `20/20`
      rollouts are finite, but only `1/20` lifts at least 5 cm and `0/20` strictly complete;
      mean peak lift is `0.0112498 m`, mean bilateral-contact fraction is `0.193976`, and
      failures are 19 object-position plus one end-effector-position termination. Therefore
      the released weights do **not** port as an acting Newton teacher.
- [x] Run the first bounded adaptation of the exact official Refiner in Newton from the 100
      CarryBox reference motions.
      Seed 171701 uses eight worlds, 64 fresh PPO updates, the official 890-D observation and
      512/256/128 topology. Initialization audit proves exact source-parameter equality and
      no optimizer or learning iteration is loaded. Frozen physical evaluation follows the
      endpoint automatically. The first direct-Isaac-hyperparameter attempt was rejected at
      update 5 after episode collapse and non-finite states; no checkpoint from it is reused.
      The investigation found two Newton transfer defects rather than hiding them as policy
      noise: `njmax=2048` overflowed at 2205 and q/qd reset left MuJoCo-Warp warm-start and
      applied-force buffers contaminated after divergence. Capacity is now fixed at 8192 per
      world and every environment reset calls Newton's official masked `SolverMuJoCo.reset`.
      A zero-optimizer 1536-transition audit then reduced divergence from 437 to three isolated
      terminations with exact zero parameter change. The subsequent eight-update smoke passed
      its fixed `0.5%` training-divergence gate at `1/1536`, kept all parameters finite and did
      not collapse episode length. The formal run started again from the exact source, not from
      either smoke, and completed all 64 updates / 12,288 transitions. Its exact divergence rate
      is `1/12288 = 0.00814%`, all policy parameters are finite, actor/critic maximum parameter
      delta is `0.0110703`, and `TRAINING_RESULT.json` passes. The fixed 20-profile deterministic
      frozen physical gate on `model_63.pt` remains negative: `20/20` finite and zero divergence,
      but only `1/20` lifts at least 5 cm and `0/20` strictly complete. Mean peak lift improves
      from `0.01125` to `0.02836 m` and bilateral contact from `0.19398` to `0.22297`; paired
      audit finds lift improvement in `16/20` profiles (median `+0.00786 m`) and contact
      improvement in `19/20` (median `+0.02922`). This is broad learnability, not teacher
      admission.
- [x] Run the one predeclared longer fresh adaptation from the same exact source for 256 updates;
      `model_63.pt` was not resumed. Seed 171701 completed 256 updates / 49,152 transitions with
      `13/49152 = 0.02645%` divergence, all policy parameters finite, maximum actor/critic delta
      `0.0270707` and final action-std mean `0.0525262`; the training gate passes. The exact
      `model_255.pt` SHA is `56b200dfc974b0017c4c654ee535128c2257ce5626cc3de5337a271f9e91fd4e`.
      Its fixed 20-profile physical gate fails: `20/20` are finite, but lift is `0/20` and strict
      completion is `0/20` against required `16/20`; mean peak lift is `0.0031307 m` and mean
      bilateral contact is `0.152296`. The endpoint is worse than fresh-64 on both continuous
      metrics. Reject this Refiner PPO objective; do not extend its budget or select another
      checkpoint from the run.
- [x] Run one fixed frame-zero-anchor learnability diagnostic, not an extension of the rejected
      objective. The exact reset audit finds only `1.787%` expected frame-zero exposure, but also
      rejects the stronger already-lifted explanation: prelifted starts are only `3.208%` and its
      predeclared `reset_distribution_mismatch` decision is false. Paired fresh-256 versus fresh-64
      lift/contact both decrease on `19/20` profiles and mean termination advances from `226.1` to
      `203.95` steps. Seed 171702 therefore starts fresh from the official Refiner for 64 updates
      with exactly one of eight worlds anchored to frame zero and seven retaining random phases;
      reward, topology and stabilization are unchanged. The 16-reset H200 runtime smoke passes.
      Evaluate only with the unchanged 20-profile physical gate and do not extend this diagnostic
      if it fails. The run completes 64 updates / 12,288 transitions with one divergence
      (`0.00814%`) and finite parameters. Its frozen gate is negative: `0/20` lift, `0/20` strict,
      mean lift `0.007011 m`, mean bilateral contact `0.179006` and mean termination step `218.75`.
      Against random fresh-64, only `1/20`, `2/20` and `2/20` profiles improve lift, contact and
      duration respectively. Reject the reset intervention; do not increase its anchored-world
      count or update budget.
- [ ] Audit and implement a serious official-action-anchored Refiner transfer method. It must keep
      the exact released `890 -> 512/256/128 -> 29` Refiner as initialization and frozen behavior
      target, use the existing official SUGAR/BCPPO machinery where compatible, and retain the
      unchanged Newton physical gate. Do not substitute a local toy residual/controller or launch
      Tracker training before the resulting acting checkpoint passes.

## E. Phase 4 — env and learning

- [x] Vec-env implementing the `rsl_rl` VecEnv protocol; torch↔warp interop following
      `newton/_src/solvers/kamino/examples/rl/`.
- [x] Wire `BCPPO` unmodified (`rsl_rl_bcppo.py`) with both required teacher roles:
      distillation target **and acting policy for the episode prefix**. The H200 runtime
      successfully imports SUGAR's compatible pure-Python `rsl_rl 3.0.1`, and the
      distillation-only execution path ran, but it did not execute the Refiner as the acting
      prefix policy. That run is rejected rather than counted as a completed handoff. The corrected
      acting/distillation implementation, exact official Tracker warm start and automatic launch
      chain are now present; runtime training remains fail-closed on the physical Refiner gate.
- [x] **Reward contact terms implemented and H200-audited from live resolved forces:**
      - [x] ankles, rubber hands and box are excluded from undesired-contact bodies
            (audit #1)
      - [x] hand reward filters the actual rubber-hand/box collision pair (audit #2)
      - [x] bilateral hold consistency is rewarded against the official contact label
            (audit #3)
      - [x] `validation/contact_rewards.py` verifies every term body and weighted sign.
            It observes 105-194 N live net force and nonzero box-filtered hand force,
            crosses false/true labels, and independently reproduces `hoi_contact` every
            frame. A controlled `last_air_time=0.2 s` two-foot landing returns `-0.6`,
            hence the official `+5.0` weight contributes `-3.0` as intended.
      - [x] Collision/solver capacity is now one contract: pipeline allocation is
            `num_envs * nconmax`. Before this audit, solver `8192` versus pipeline `2618`
            made `update_contacts` fail before any contact reward could run. The later learned
            contact-rich audit also raised per-world `njmax` from 2048 to 8192 after an observed
            2205-constraint overflow.
- [x] **Official Refiner-rollout Tracker data generated and audited on H200.** The
      parameter-exact recovered `refiner_model10000.pt` ran in the official SUGAR
      Refiner-Rollout environment with 1000 worlds. All environments completed; 912
      trajectories reached the natural endpoint and the 88 early failures were excluded
      by the official collector. `process_refiner_rollout.py` processed all 912. The
      resulting dataset is finite, has exact `(T,29)` joints and `(T,14,3)` configured
      Tracker bodies, and every student/teacher length differs by exactly one frame.
      Successful data cover 95/100 source motions; IDs `18/40/48/60/66` have no admitted
      Refiner trajectory and are a recorded coverage limitation, not silently fabricated.
- [x] **The formal student/teacher body-layout confound is removed.** Processed Tracker
      data contain the 14 configured bodies, while raw teacher clips contain all 35 URDF
      bodies. `load_clips` now admits only those two explicit layouts, gives student and
      teacher separate body indices, and fails before environment construction on missing,
      misaligned, nonnumeric or non-finite clip data.
- [x] **Reject and stop the first formal Newton BCPPO attempt.** Although it used eight
      worlds, all 912 admitted clips and the official distillation target, Newton executed
      student actions from the first frame; the Refiner was never the acting prefix policy.
      It was stopped at iteration 12, is invalid for the handoff contract and must never be
      resumed or reported as Tracker progress.
- [x] **Implement the fail-closed acting-teacher handoff adapter.** The same strict
      `890 -> 512/256/128 -> 29` checkpoint is loaded independently as the deterministic
      Newton acting policy and BCPPO distillation teacher, then checked bitwise equal.
      Refiner actions execute until the box remains at least 5 cm above its reset height for
      ten consecutive frames; there is no fallback timer. A training-only 1-D mask is zero
      on every teacher-controlled transition and never enters the exact 510-D actor input.
      It masks PPO/value/entropy credit while retaining official full-trajectory Refiner
      distillation, so the student can imitate the prefix without pretending it acted there.
      The student loads the released CarryBox Tracker actor, critic and std exactly, while
      explicitly discarding its optimizer and iteration; Newton BCPPO therefore starts from
      the serious released skill with fresh update state rather than a random placeholder or
      a resumed rejected run.
      The VecEnv now reuses the policy observation returned by the physics step instead of
      pushing each frame into the causal history twice. Formal launch additionally requires
      a passing fixed-gate JSON whose checkpoint SHA matches the acting checkpoint; the
      failed source and fresh-64 results are rejected by test.
- [x] **Predeclare the matched Tracker frozen evaluator and automatic endpoint chain.**
      Twenty fixed motion-ID-disjoint processed profiles are reset at frame zero. The
      official Tracker warm start and learned `model_2999.pt` use the same admitted acting
      Refiner and one Newton environment whose solver, contact matcher, state, causal 510-D
      history, 890-D critic and teacher observations are reset and hashed per arm. The
      evaluator records handoff timing, actual teacher/student control steps, strict
      failures, lift, bilateral contact, reward, root loss and action bounds. It reports
      evaluation validity separately from physical advantage; it cannot turn a finite run
      into an improvement claim. The automatic chain runs it only after the formal-3000
      training gate passes.
- [x] Resolve the conditional acting-teacher/Tracker launch automatically from the fresh-256 gate.
      The gate reports training pass but physical fail (`0/20` lift, `0/20` strict, required
      `16/20`), and the chain records `AUTO_BCPPO_CHAIN_REJECTED_REFINER_GATE`. Therefore the
      physical handoff smoke and fresh Tracker BCPPO run were correctly not started; no failed
      checkpoint exercised BCPPO. A future run requires a different serious acting-teacher method,
      not another update-budget extension of this objective.
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

## H. Official-Refiner action-anchor transfer (2026-08-27)

- [x] Reject the adaptive-KL two-update smoke: learning rate reached `1e-2`, actor and
      critic drifted by `0.02166/0.06118`, and std rose to `0.06406`.
- [x] Disable KL learning-rate adaptation for this method while retaining official BCPPO,
      the exact frozen official Refiner teacher, mean-only distillation, stage-3 floor
      `1.0`, random resets, std `0.05`, and configured learning rate `1e-5`.
- [x] Pass the fresh seed171705 fixed-rate two-update smoke: both logged rates `1e-5`,
      zero divergence, all finite, actor/critic deltas `0.0001936/0.0002668`, std
      `0.0500219`.  The zero-optimizer audit is still exact over 384 transitions.
- [x] Run one fresh seed171706, eight-world, 64-update action-anchor endpoint on H200:
      finite, `5/12288` divergences, fixed LR `1e-5`, actor delta `0.001884`.
- [x] Run the unchanged fixed-20 frozen Refiner gate.  It is rejected at `2/20` lift and
      `0/20` strict versus required `16/20`; a real Tracker entry call fails closed before
      environment construction or any optimizer update.  Matched lift/contact improve in
      `13/20` and `16/20`, so record a directional response but no acting-teacher pass.
- [x] Generate and smoke the isolated Isaac-style torso convex hull: 51,410 mesh triangles
      become 2,586 hull triangles, volume ratio `1.79x`, exactly two torso shapes replaced,
      and the one-profile rollout is finite.
- [x] Complete the fixed-20 exact-official-Refiner torso-hull causal comparison.  It is
      negative: mesh/hull are both `1/20` lift and `0/20` strict; mean lift changes only
      `0.011250 -> 0.011333 m`, matched mean delta `+0.0000834 m`.  Keep mesh as default;
      do not hull more links or run another collision-geometry sweep.
- [ ] Implement the next topology change as adapter-only control around a parameter-exact
      frozen official Refiner.  It must retain the official `512/256/128` scale, consume
      only the causal current 890-D Newton observation, start at exact zero action delta,
      and pass a zero-optimizer plus bounded-update smoke before one fixed physical gate.
- [x] Automatically reject before
      Tracker unless at least `16/20` profiles satisfy strict lift/hold; launch the
      existing acting-teacher Tracker chain only on a machine-readable pass.  The
      action-anchor endpoint failed, so no Tracker run was launched.

---

## E. Phase 3 — the tracker in the Newton loop (2026-08-22)

- [x] **The original SUGAR environment runs here.** Not a missing IsaacLab patch, which
      was the earlier wrong call. IsaacLab's `.kit` sets the asset roots correctly; Kit
      requested the right S3 URL and timed out after 53 s (`Could not get Sdf layer`)
      while plain `curl` to the same URL returns 200 in milliseconds — so `omni.client`,
      not the network. `ISAACLAB_GROUND_PLANE_USD` is a red herring: it appears nowhere in
      IsaacLab v2.3.0, only SUGAR's scripts *set* it. Fix is three staged assets plus a
      Kit setting, no code change: see `isaac/RUNNING_ISAAC_HERE.md`.
- [x] **Ground truth captured from SUGAR's own dumper**, `--task
      Sugar-G129dof-CarryBox-Tracker-Rollout --rollout_dir`. 7 trajectories with per-step
      action, joint_pos/vel, root/anchor/body/object poses.
- [x] **The 510-D observation is validated offline**, `validation/verify_tracker_obs.py`.
      Rebuild the obs from Isaac's recorded state, run the official actor, compare with
      the action Isaac applied: correlation 0.970-0.987, RMSE 0.088 against an action std
      of ~1.3, with SUGAR's own Unoise accounting for 0.080 of it (ratio 1.11).
      Read RMSE, not correlation — the 29 reference joint angles dominate the prediction,
      so a wrong convention only costs 0.969 -> 0.945.
- [x] **`validation/g1_carrybox_policy.py`: floating base, legs on, `tracker.pt` closed
      loop at 50 Hz.** The G1 stands on its own feet for all 481 frames, walks in, squats,
      grips and lifts. Lift 0.21-0.30 m against a reference 0.63-0.69 m on three clips,
      reference joints tracked to ~8 deg. Video:
      `isaac/newton_carrybox.mp4`.
- [ ] **The remaining gap is unexplained.** Three hypotheses tested and refuted:
      friction (saturates by mu=1.0), the convex-hull hand collider (the `--hull-hands`
      ablation *lowers* the lift, 0.23 -> 0.07 m), contact compliance (`ke=1e4` is already
      the optimum; 1e3 and 1e5 are both worse). What is measured and stands: Newton
      demands far more wrist torque than Isaac — wrists at their effort limit 11.7% of
      frames (37.4% during the carry) against Isaac's 1.9%, while every non-wrist joint
      agrees at 0.1% vs 0.0%. The discrepancy is localised to the hand-box interface and
      the wrists are the weakest actuators by 10x (5 N.m against 50-139).
      Next untested candidate: Isaac hulls *every* link, and the reference carries the box
      against the chest, so the torso collider — which `--hull-hands` leaves alone — may
      form the shelf. Caveats: these are PD estimates from position error, not measured
      joint torques, and two runs at identical parameters gave 0.23 and 0.30 m, so
      sub-0.1 m single-run differences are not effects.
