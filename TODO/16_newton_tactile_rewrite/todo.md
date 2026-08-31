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
- [x] Implement the next topology change as adapter-only control around a parameter-exact
      frozen official Refiner.  It must retain the official `512/256/128` scale, consume
      only the causal current 890-D Newton observation, start at exact zero action delta,
      and pass a zero-optimizer plus bounded-update smoke before one fixed physical gate.
- [x] Reuse the admitted frozen-expert residual implementation at 890-D Refiner scale.
      Component audit: exact-zero composed delta, frozen expert with no gradients, nonzero
      residual gradients.  Zero-optimizer: 384 finite transitions, zero divergence and
      exact-zero state change.  Two-update seed171709: expert delta `0`, residual delta
      `0.0002166`, fixed LR `1e-5`, std `0.050024`, zero divergence.
- [x] Run one fresh seed171710, eight-world, 64-update frozen-expert residual endpoint and
      its unchanged fixed-20 composed-action physical gate.  Training passes with `2/12288`
      divergences, 64 fixed `1e-5` learning rates, residual delta `0.002876`, and exact-zero
      official expert weight/std drift.  The physical gate rejects it at `0/20` lift and
      `0/20` strict versus required `16/20`; mean lift/contact are `0.009543 m/0.197050`.
      Matched mean lift is `0.001706 m` below the official endpoint and `0.017897 m` below
      action-anchor.  Do not sweep residual limit, LR, reward weights or update budget.
- [x] Automatically reject before
      Tracker unless at least `16/20` profiles satisfy strict lift/hold; launch the
      existing acting-teacher Tracker chain only on a machine-readable pass.  The
      action-anchor and frozen-residual endpoints failed, so no Tracker run was launched.
- [x] Implement and audit the fixed causal temporal Refiner composer: exact frozen official
      actor/std; explicit current plus past `10 x 890` observation contract; admitted six-layer,
      384-D Transformer; exact-zero output head; causal expert-retention plus bounded 29-D
      correction; no future/outcome actor input.  Component audit: 11.984M total/11.360M
      trainable parameters, exact-zero endpoint delta, zero expert gradient, nonzero composer
      gradient and fail-closed mismatched-history rejection.
- [x] Pass one fresh zero-optimizer and one fresh two-update H200 gate.  Require exact history
      reset/last-frame equality, exact-zero pre-update composed-action delta, exact-zero official
      expert drift, finite tensors, zero diagnostic divergence, fixed `1e-5` LR and nonzero
      temporal-composer gradient/parameter movement.  Seed171712 gives 384 transitions with zero
      optimizer/state change and zero divergence.  Seed171713 gives composer delta `0.00021968`,
      expert delta `0`, two fixed `1e-5` LR records and zero divergence.
- [x] If and only if both gates pass, run one fresh eight-world, 64-update endpoint and its
      unchanged fixed-20 physical gate.  Do not sweep history length, Transformer size, residual
      limit, LR, reward weights or update budget; do not launch Tracker below `16/20` strict
      lift/hold.  Seed171714 passes the numerical training gate with `3/12288` divergences and
      finite parameters, but the physical gate rejects it at `1/20` lift and `0/20` strict.
      Mean lift/contact are `0.012809 m/0.198697`; expert retention is `0.99999988` and mean
      absolute correction only `0.010058`.  Tracker was not launched.
- [x] Implement the next bounded released-Tracker action-supervision diagnostic around the
      parameter-exact frozen Refiner.  The exact released Tracker supplies current-state action
      labels only; its 510-D observation and the Refiner's 890-D observation must be built from
      the same Newton state, and no future state/outcome may enter the actor.  The deployed actor
      remains the official-scale frozen-Refiner residual; the Tracker is training-only.
- [x] Pass component, zero-optimizer and short-update gates before any formal endpoint: strict
      loading of both released experts, nonzero adapter-only gradient/update, exact-zero expert
      actor/std drift, finite live Newton transitions and a machine-checked causal alignment.
      Component residual gradient is `0.38765` with zero expert gradients.  Seed171715 gives
      48 zero-update transitions with zero divergence/state change.  Seed171716 gives two updates,
      384 transitions, zero divergence, fixed LR `1e-5`, residual/critic deltas
      `0.0004075/0.0004647`, and exact-zero Refiner/Tracker weight/std drift.  A one-profile
      evaluator smoke strictly loads the checkpoint and proves the Tracker is unused at inference.
- [x] Run exactly one fresh seed171717, eight-world, 64-update supervised-adapter endpoint and
      the unchanged fixed-20 physical gate.  Training passes with 64 updates / 12,288 transitions,
      finite parameters, fixed LR `1e-5`, residual/critic deltas `0.0032824/0.0124748` and
      `4/12288 = 0.03255%` divergences.  The fixed-20 gate rejects the endpoint at `1/20` lift and
      `0/20` strict, with mean lift/contact `0.011805 m/0.195350`.  Both released experts retain
      exact-zero weight/std drift, the Tracker is unused at inference and all profiles are finite.
      Downstream Tracker was not launched.  Do not sweep teacher weight, residual limit, LR,
      reward or update budget for this topology.
- [x] Run fresh seed171718 for exactly two updates / 384 transitions using repository BCPPO
      Stage-1 pure distillation with the same released-Tracker current-state teacher and frozen
      official Refiner residual.  Require finite tensors, zero divergence, nonzero residual update,
      exact-zero critic/std drift, exact-zero drift of both released experts and strict evaluator
      loading before a formal endpoint is admitted.  Residual delta is `0.00046239`, critic drift is
      exactly zero, std delta is `7.45e-10`, all tensors are finite and strict evaluation restores
      both experts exactly.  The gate nevertheless rejects the run because it has one synchronized
      divergence (`1/384 = 0.2604%`) rather than the required zero.
- [x] Enforce the failed short gate before formal training.  An automatic chain initially used the
      looser general `0.5%` ceiling and began seed171719; it was stopped with `Ctrl+C` after update
      0 / 192 transitions.  It has no `TRAINING_RESULT.json`, is not a formal endpoint and must not
      be resumed or reported.  The pure-distillation formal run and downstream Tracker/BCPPO are
      rejected; do not rerun this objective with another seed or sweep its settings.
- [x] Combine the admitted six-layer causal temporal Refiner composer with the exact released
      Tracker as a training-only current-state action teacher.  Prove exact-zero composed/endpoint
      delta, exact-zero frozen Refiner/Tracker drift, synchronized 9790-D/510-D causal observations,
      Tracker absence from inference and a fresh seed171720 48-transition zero-optimizer pass.  The
      13,034,614-parameter implementation passes: its 11,360,286-parameter temporal composer starts
      at exact zero, both released experts retain exact-zero weight/std drift, and all 48 live
      Newton transitions are finite with zero divergence and zero policy-state change.
- [x] Run fresh seed171721 for exactly two Stage-1 pure-distillation updates / 384 transitions.
      Require zero divergence, finite tensors, nonzero temporal-composer update and exact-zero
      critic/std/expert drift.  Only all passes admit one fresh seed171722 64-update endpoint and
      unchanged fixed-20 physical gate; do not sweep this topology or launch Tracker below the
      `16/20` lift and strict rule.  The short gate passes with zero divergence, composer delta
      `0.00045949`, exact-zero critic drift, std delta `7.45e-10` and distillation loss
      `0.1205 -> 0.1103`.  Its strict fixed-20 loading audit preserves both experts exactly, never
      calls the Tracker at inference and finishes `20/20` finite profiles without divergence; the
      diagnostic physical result is `0/20` lift and `0/20` strict.
- [x] Finish the automatically admitted fresh seed171722 64-update endpoint and its unchanged
      fixed-20 physical gate.  Training passes with 64 updates / 12,288 transitions, one divergence
      (`0.00814%`), composer delta `0.0162462`, exact-zero critic drift and std delta `7.45e-10`.
      Frozen evaluation improves to `8/20` lift but remains `0/20` strict, versus the required
      `16/20` for each.  Mean peak lift/bilateral contact are `0.071140 m/0.128391`; all profiles are
      finite, both released experts remain exact and the Tracker is unused at inference.  Reject the
      endpoint and keep downstream Tracker/BCPPO closed.  Do not sweep this topology's history,
      Transformer size, teacher/loss weight, LR, residual limit or update budget.
- [x] Replace the rejected non-identifiable `retention * Refiner + residual` rule with an exact
      additive causal correction.  Preserve the parameter-exact official Refiner, exact released
      Tracker training teacher and admitted six-layer 384-D past-`10 x 890` Transformer; initialize
      the 29-D output head at exact zero so deployed action is bitwise the official Refiner.  The
      component audit proves output dimension 29, exact-zero endpoint delta, exact unit retention,
      zero Refiner gradient and nonzero composer gradient `0.25147`.
- [x] Run fresh seed171723 for 48 zero-optimizer transitions, then fresh seed171724 for exactly two
      Stage-1 updates / 384 transitions.  Require zero divergence, finite tensors, nonzero temporal
      update, exact-zero critic/std/expert drift, exact unit Refiner retention and strict checkpoint
      loading.  Only all machine passes admit one fresh seed171725 64-update endpoint plus the
      unchanged fixed-20 gate; no scale, capacity, LR, loss or budget sweep is allowed.  Seed171723
      passes with zero state/parameter change and zero divergence.  Seed171724 passes with zero
      divergence, composer delta `0.00046440`, exact-zero critic drift, std delta `7.45e-10` and
      distillation loss `0.3349 -> 0.1788`; the one-profile loading smoke passes every structural
      check and keeps both experts exact.  Its diagnostic result is `0/1` lift/strict.
- [x] Finish the automatically admitted fresh seed171725 64-update endpoint and unchanged fixed-20
      physical gate.  Training passes with 64 updates / 12,288 transitions, two divergences
      (`0.01628%`), composer delta `0.0145992`, exact-zero critic drift and std delta `7.45e-10`.
      Frozen evaluation gives `10/20` lift and `0/20` strict, mean lift/contact
      `0.058460 m/0.170408`, exact unit endpoint retention and 20/20 finite profiles.  Reject the
      endpoint and keep Tracker/BCPPO closed; do not sweep this additive topology.
- [x] Add the missing current 36-D Tracker command to the additive causal composer while preserving
      the exact Refiner and training-only Tracker.  The command is the deployable current reference
      tuple `29 joint target + 3 root linear velocity + 3 root angular velocity + 1 contact`; it is
      not a future/outcome label.  Require exact observation synchronization, zero-start action and
      expert drift before any fresh live Newton training.  The new non-mutating command builder now
      passes its two-world H200 gate: state/history deltas are exact zero and both reset-time and
      post-step maximum differences from `observe()[:, :36]` are exact zero.  The serious composer
      keeps the same six-layer 384-D past-`10 x 890` Transformer and adds the current command as one
      independent token.  Seed171726's CUDA component gate proves a 29-D exact-zero output head,
      exact-zero initialized endpoint delta and Refiner gradient, exact unit retention, nonzero
      composer gradient `0.24515`, and command-induced hidden-state delta `0.09760`.
- [x] Run exactly one fresh seed171727 zero-optimizer gate for 48 live Newton transitions.  Require
      finite policy/state tensors, zero divergence, exact-zero policy and frozen-expert drift,
      a `9826 = 890 + 10 x 890 + 36` policy tensor, history/current equality, and elementwise
      equality between the appended command and the first 36 coordinates of the synchronized
      released-Tracker observation.  All checks pass: `48/48` transitions are finite, divergence and
      policy-state delta are zero, all three history/command alignment deltas are exactly zero, and
      the two live worlds have command-token representation span `2.78275`.
- [x] Only if seed171727 passes, run exactly one fresh seed171728 two-update / 384-transition
      Stage-1 pure-distillation gate.  Require zero divergence, nonzero composer update, exact-zero
      critic and released-expert drift, action std within `1e-7` of `0.05`, fixed LR `1e-5`, finite
      parameters, and strict loading of the resulting command-conditioned checkpoint.  Do not sweep
      the command representation, history, capacity, loss, LR, residual limit or update budget.
      Training passes with zero divergence, composer delta `0.00046290`, exact-zero critic drift,
      std delta `7.45e-10` and fixed LR `1e-5`.  The one-profile strict-load smoke restores
      `model_1.pt` (SHA256 `e1dddcfb8fef26169394f323c1e3acde4ba251b31db4fd6167615c48ff8eb214`),
      keeps both experts exact, verifies the current-command contract and finishes finite without
      active divergence.  Its `0/1` lift/strict outcome is diagnostic-only.
- [x] Only if both short gates pass, run one fresh seed171729 eight-world, 64-update endpoint and the
      unchanged fixed-20 physical evaluator.  Admit downstream Tracker/BCPPO only at at least
      `16/20` lift and `16/20` strict completion; otherwise reject this topology automatically and
      continue to the next serious controller diagnosis.  Training passes all 64 updates / 12,288
      transitions with zero divergence, composer delta `0.0162573`, exact-zero critic drift, std
      delta `7.45e-10`, fixed LR `1e-5` and finite parameters.  Frozen fixed-20 strictly restores
      `model_63.pt` (SHA256 `a7756a48137d2f6bfb069ef636d608bc8956fbba5ea8d5b2e8addf79014678e1`),
      keeps both experts exact and finishes 20/20 finite profiles, but reaches only `7/20` lift and
      `0/20` strict.  Mean lift/contact are `0.044066 m/0.122728`; failures are `ee_pos=11` and
      `obj_pos=9`.  Reject the topology and keep downstream Tracker/BCPPO closed.
- [x] Before another learned controller, run one parameter-free fixed-20 ceiling audit of the exact
      released CarryBox Tracker on the same raw-motion Newton resets and termination rules.  Strictly
      load its official `510 -> 512/256/128 -> 29` actor, consume only `CarryBoxEnv.observe()` and
      preserve the unchanged `16/20` lift/strict rule.  This determines whether the current action
      label source itself is physically admissible; do not train, alter physics, tune thresholds or
      sweep Tracker checkpoints.  The exact released checkpoint is finite on `20/20` profiles but
      reaches only `2/20` lift and `0/20` strict completion, with mean lift/contact
      `0.019740 m/0.054595`; failures are `ee_pos=18`, `obj_pos=2`.  Its action labels are rejected
      for any further Newton Refiner distillation.
- [x] Implement the next controller diagnosis as a fixed-context causal physical-recovery objective,
      not another action label.  Preserve the parameter-exact official Refiner and the admitted
      six-layer, 384-D, past-`10 x 890` additive Transformer.  Blend `0.5` normalized unchanged
      official reward with `0.5` current-rollout physics: object/end-effector margins normalized by
      their existing `0.30 m` terminations, real bilateral hand-box contact, bilateral lift normalized
      by the existing `0.05 m` gate, and current failure.  These are reward labels only; actor input
      remains 9790-D and no future/outcome tensor is exposed.
- [x] Pass the fresh seed171731 zero-optimizer fixed-context gate on H200: `48/48` physical-reward
      calls, maximum absolute combined reward `0.711533`, zero divergence, finite tensors and exact
      zero policy-state change.  No reward label augments the actor.
- [x] Finish the fresh seed171732 two-update gate and strict one-profile load audit.  Training has
      `384/384` physical-reward calls, zero divergence, finite actor/critic deltas
      `0.0002143/0.0004684`, fixed LR `1e-5` and maximum absolute reward `0.711787`.  Strict loading
      preserves the official Refiner exactly; the diagnostic checkpoint gives `0.008810 m` peak lift
      and `0/1` strict, with only `0.001170` mean absolute correction.
- [x] Run one fresh 64-update frame-zero `data_000` learnability endpoint.  Admit a task-wide formal
      run only if the
      fixed context improves over the exact Refiner's `0.008450 m` peak lift and reaches trajectory
      timeout without strict failure.  Do not sweep objective weights, controller capacity, history,
      residual limit, LR or update budget.  Seed171733 passes the numerical training contract with
      `12,288/12,288` reward calls, zero divergence, actor/critic deltas
      `0.003044/0.013657` and finite parameters.  The frozen endpoint is negative: peak lift regresses
      `0.008450 -> 0.006923 m`, bilateral contact `0.2000 -> 0.1872`, and both endpoints fail on
      `obj_pos` at step 235.  Do not run the task-wide objective or tune it.
- [x] Resolve the apparent contradiction with the legacy single-world Tracker claim.  Regenerating
      `tracker_actor.npz` directly from the exact released checkpoint and rerunning `data_000` at
      `mu=1.0` reproduces `0.2442 m` late lift, but that script has no strict termination: box height
      is still only `0.207 m` at frame 200 and rises at frame 250, whereas the formal exact Tracker
      fails `ee_pos` at frame 78 with only `0.001143 m` lift.  The historical lift occurs after the
      trajectory is already invalid and is not an acting-teacher result.
- [x] Before another optimizer run, test the parameter-free Newton reference-joint-target ceiling.
      Convert each current reference joint pose through the exact Isaac action map
      `(q_ref - q_default) / action_scale`, run the unchanged fixed-20 strict environment, and record
      lift, strict completion, action magnitude and termination causes.  No policy training, physics
      change, threshold change or action clipping is allowed.  This decides whether the reference is
      dynamically executable with the admitted actuators before constructing a Newton-native action
      teacher.  The exact inverse map has maximum reconstruction error `1.19e-7` and all twenty
      rollouts are finite, but it gives `0/20` lift, `0/20` strict, mean lift/contact
      `0.002915 m/0.001515` and maximum action `9.8803`; failures include `ee_pos=16`,
      `anchor_pos=5`, `anchor_ori=1`.  Reference joint poses alone are not a valid Newton action
      teacher because they omit the floating-base/contact dynamics needed to realize the motion.
- [x] Implement the next fixed failure-frontier diagnostic around the parameter-exact official
      Refiner.  On frame-zero `data_000`, execute the exact Refiner for a fixed 200-step physical
      prefix with zero PPO/value/entropy credit, then train the same serious additive temporal
      controller only on the live post-prefix recovery segment using the admitted physical-recovery
      objective and an exact Refiner action anchor.  Actor history must contain the real prefix,
      future/outcome labels remain absent, and frozen evaluation must still deploy one checkpoint
      from frame zero.  Require a nine-horizon zero-optimizer gate and a ten-update gate (the first
      eight 24-step updates are necessarily inside the 200-step prefix) before one 64-update learnability
      endpoint; do not sweep prefix, objective weights, model, LR, residual or budget.  Seed171734
      passes the zero-optimizer gate with `400/32` exact teacher/student execution steps, `432/432`
      reward calls, zero divergence and zero parameter change.  Seed171735 passes the ten-update gate
      with `1,640/280` teacher/student steps, exact execution, no divergence and nonzero
      actor/critic/std changes `6.94e-4/7.50e-4/1.37e-4`.  Its frame-zero one-profile audit improves
      peak lift `0.00845 -> 0.01269 m` and gives `0.20851` bilateral contact, but remains `0/1` lift
      and strict.  Fresh seed171736 completes the single admitted 64-update endpoint with
      `10,608/1,680` teacher/student steps, zero execution error/divergence and nonzero
      actor/critic/std changes `0.00396/0.00579/0.000598`.  Frozen model63 reaches `0.01223 m` lift
      and `0.20851` bilateral contact but still fails `obj_pos` at step 235, giving `0/1` lift and
      strict.  It improves only small pre-failure lift, not recovery; task-wide training and
      downstream Tracker/BCPPO remain closed, with no prefix/objective/budget sweep.
- [x] Run one no-training exact-prefix deployment audit to distinguish frame-zero distribution drift
      from a failed recovery.  Compose the same model63 with 200 parameter-exact official Refiner
      actions, preserve real causal history, then execute the learned controller without reset.  The
      audit executes `200/35` exact/student actions with only `0.00961` maximum handoff action jump
      and no non-finite state, but again fails `obj_pos` at step 235 with `0.01072 m` lift and `0/1`
      strict completion.  Recovery fails on its training handoff distribution; do not tune prefix,
      reward or budget, and do not open downstream Tracker/BCPPO.
- [x] Run one fixed Newton-native free-residual diagnostic.  Preserve the exact Refiner, prefix200,
      serious past-`10 x 890` six-layer controller, residual limit, physical objective, LR and
      update budget, but remove the failed post-handoff Refiner action distillation entirely.  This
      changes the action objective rather than sweeping its weight.  Require fresh seed171737
      zero-optimizer and seed171738 ten-update gates before one seed171739 64-update model63; audit
      exact-prefix recovery first and frame-zero deployment only if it crosses step235.  Keep the
      task-wide `16/20` lift/strict and downstream Tracker/BCPPO gates closed meanwhile.  Seed171737
      zero-opt and seed171738 ten-update gates pass.  Seed171739 also passes its 64-update numerical
      contract with `10,608/1,680` exact/student actions, zero execution error/divergence and
      actor/critic/std changes `0.00148/0.00682/0.000193`.  Exact-prefix model63 still fails
      `obj_pos` at step235 after 35 student actions: lift/contact are `0.01045 m/0.20426`, handoff
      jump is `0.01576`, and strict completion is `0/1`.  The action anchor is not the bottleneck;
      frame-zero/task-wide evaluation and other anchor settings remain rejected.
- [x] Reject the first seed171740 one-candidate-per-world CEM despite its nominal `0.12082 m` lift,
      `29/32` surviving candidates and full bilateral contact.  The eight step200 worlds have
      `0.06533` q/qd spread, so candidate identity was confounded with world identity; independent
      best replay also differed by `0.00353 m` lift.  This is an existence signal, not a training
      target.
- [x] Generate one serious Newton-native recovery action target with corrected matched seed171740 CEM
      shooting.  Replay the exact Refiner to step200, then optimize the complete 35-step, 29-D
      correction as seven five-step knots with population/elite/generations `32/8/4` and bound one.
      Capture the complete causal/solver frontier once and evaluate every candidate over the same
      eight restored worlds.  Rank by all-world crossing, worst-case survival, strict object margin,
      lift, bilateral contact, end-effector margin and correction energy.  Require the embedded zero
      correction to reproduce the exact step235 `obj_pos` failure in `8/8`, tensor-exact frontier
      restore, unchanged official Refiner and best-candidate replay within the pre-search fixed
      `5e-4` continuous-metric tolerance; categorical outcomes remain exact.  Future shooting states
      are training labels only; admit controller supervision only if the best candidate stays strict
      valid after all 35 recovery steps in all eight worlds.  The search produces a strong `8/8`
      strict target with worst/mean lift `0.11027/0.11707 m`, full bilateral contact and positive
      margins, but independent replay fails the fixed tolerance at `0.006412` maximum delta
      (`0.002880 m` peak-lift delta).  Two triangle-buffer warnings also occur in the final candidate
      pool.  Preserve this as existence evidence but reject it for supervision.
- [x] Run fresh seed171742 robust matched-v3 CEM.  Preserve every v2 physics/search constant, but
      execute every candidate twice, rank overflow-free fixed-tolerance pairs before the unchanged
      worst-case physics ordering, and require the final selected action to match both executions in
      a third unselected replay.  Read Newton's public triangle-pair counter every step and reject any
      candidate exceeding the fixed one-million capacity.  Do not relax `5e-4` or change the CEM
      budget/bound.  This is negative: all-world candidates rise
      `5/32 -> 15/32 -> 18/32 -> 30/32`, while fully robust candidates fall `3 -> 1 -> 0 -> 0`.
      The selected stable candidate is only `0/8` strict with `0.01563/0.01702 m` worst/mean lift.
      Parameters/frontier tensors are exact and no candidate overflows the triangle-pair capacity,
      so reject both v2/v3 for distillation.
- [x] Run one simultaneous matched-world replay-topology smoke before any new shooting search.
      Duplicate the eight matched profiles into paired world blocks in one Newton environment,
      restore both blocks tensor-exactly and execute zero correction plus the saved v2 correction in
      matched control steps.  Keep the official Refiner exact, the fixed `5e-4` continuous tolerance,
      exact categorical outcomes and the public triangle-pair capacity.  Admit a fresh search only
      if every saved-v2 pair is stable and strict; otherwise close action shooting as supervision.
      The saved correction is `16/16` strict with `0.10077/0.10980 m` worst/mean lift, but identical
      pairs differ by `0.02915` and triangle pairs overflow at `1321029/1000000`; even zero pairs
      differ by `0.02857`.  Close action shooting as supervision.
- [x] Do not run the conditional seed171741 shooting-target distillation because no shooting target
      passes the fixed replay/capacity gate.  The predeclared architecture remains serious, but its
      required supervision label is rejected.
      If a future independent admitted target exists, train exactly one fresh seed171741 serious
      shooting-target
      distillation.
      Train only the existing six-layer 384-D past-`10 x 890` additive temporal composer around the
      parameter-exact official Refiner.  Use matched worlds `0--5` for training and `6--7` for
      validation, balance the final 64 zero-correction prefix frames against all 35 recovery labels,
      and fix AdamW at 3000 steps, LR/weight decay `1e-4/1e-4`, batch 128.  Require an exact frozen
      expert, causal last-frame equality, and 8/8 strict survival plus 5 cm lift on the fixed frontier
      before running the unchanged task-wide 20-profile gate.  Do not tune this fit or launch
      Tracker/BCPPO unless that later gate reaches `16/20` lift and `16/20` strict completion.
- [x] Implement and audit the serious causal Refiner reference-phase controller.  Preserve the
      released Refiner/std exactly; add an independent policy-reference phase used only by
      `obs_890`, while physical `env.t`, reward, termination and timeout retain the original clock.
      Reuse the six-layer 384-D past-`10 x 890` Transformer and output one phase offset bounded to
      `+/-7` frames with a fixed envelope that is zero at physical steps 200 and 235.  Seed171744
      zero-optimizer smoke must prove zero-offset observation/action identity, isolated nonzero
      reference-feature effects, unchanged physical/termination inputs, exact RNG and no
      future/outcome actor labels.
      Seed171744 passes with exact official parameters, physical state/suffix and CPU/CUDA RNG;
      only the 656-D reference prefix changes and the frozen action responds by `0.64942`.
- [x] If the component audit passes, run fresh seed171745 ten-update learning smoke and exactly one
      seed171746 64-update fixed-frontier endpoint with the unchanged physical-recovery reward.
      Require physical step235, policy phase235, strict validity and at least 5 cm lift before the
      unchanged task-wide `16/20` lift and `16/20` strict evaluation.  Do not tune phase bound,
      envelope, prefix, reward or budget; keep Tracker/BCPPO closed below the gate.
      Seed171745 passes its ten-update numerical contract.  Seed171746 passes 64-update training
      with `12288` finite transitions, zero divergence, 101 retimed calls and actor/critic deltas
      `0.000983/0.006365`.  Frozen seed181746 model63 still fails `obj_pos` at step235 with
      `0.00870 m` lift, `0.19574` bilateral contact, `0/1` strict completion and zero integer phase
      change.  The task-wide gate is not launched and Tracker/BCPPO stays closed; do not tune this
      topology.

- [x] Implement the causal latched transition-plan controller.  Keep the released Refiner exact;
      at step200 use the existing six-layer 384-D past-`10 x 890` Transformer once to emit seven
      bounded 29-D correction knots, hold each for five physical controls, then return to the exact
      Refiner at step235.  Mask all action samples except the handoff plan, expose no future/outcome
      actor input, and never use the rejected shooting correction as a supervision label.  Fix
      bound/std/LR at `1/0.35/1e-5` with one 256-update budget and no sweep.
- [x] Run fresh seed171747 zero-optimizer component/runtime gates, then seed171748 ten-update smoke
      and exactly one seed171749 256-update training only if preceding machine checks pass.  Frozen
      seed181748 must reach 5 cm lift and strict validity at step235 before the unchanged task-wide
      `16/20` gate; otherwise close this topology and keep Tracker/BCPPO closed.
      The zero/runtime and ten-update gates pass.  Seed171749 completes 49,152 transitions and 208
      plan latches with zero divergence; actor/critic deltas are `0.00200/0.00970`.  Frozen
      seed181748 verifies exact official control outside the 35-step chunk and one latch, but data000
      lifts only `0.01112 m`, has `0.20851` bilateral contact, fails `obj_pos` at step235 and reaches
      `0/1` strict completion.  The deterministic maximum correction is `0.06366`.  The fixed gate
      fails, so no task-wide evaluation or Tracker/BCPPO launch is admitted and this topology is
      closed without a sweep.

- [x] Repair the action-chunk semi-Markov return boundary without changing its controller or reward.
      The rejected runner used 24-step storage for a 35-step latched action and masked value loss on
      every execution row, so each handoff return was necessarily truncated.  Fix episode length and
      `num_steps_per_env` together at 235, retaining prefix200, `7 x 5` knots, bound/std/LR
      `1/0.35/1e-5`, the same physical reward and exact frozen Refiner.
- [x] Run fresh seed171750 for one zero-optimizer 235-step horizon and fresh seed171751 for exactly
      two updates / 3,760 transitions.  Only both machine passes admit fresh seed171752 for one
      32-update / 60,160-transition endpoint and frozen seed181750 data000 evaluation.  Require 5 cm
      lift plus strict step235 validity before task-wide `16/20`; otherwise close the credit-aligned
      topology and keep Tracker/BCPPO closed.
      Both short gates pass.  Seed171752 completes 60,160 transitions and 256 plans with zero
      divergence and actor/critic/std deltas `0.002175/0.007671/0.002226`.  Frozen seed181750 remains
      finite and exact outside the chunk; deterministic correction improves `0.06366 -> 0.08505`,
      lift improves `0.01112 -> 0.02214 m`, and bilateral contact improves
      `0.20851 -> 0.22034`, but it still fails `obj_pos` after 236 steps and reaches `0/1` strict.
      No task-wide gate or Tracker/BCPPO launch is admitted; close this family without an extension.

- [x] Run the single predeclared causal receding-knot Refiner diagnostic.  Keep the exact released
      Refiner, past `10 x 890` six-layer 384-D Transformer, prefix200, 35-step interval,
      bound/std/LR `1/0.35/1e-5` and the same physical reward.  Replan one 29-D correction only at
      steps `200/205/.../230`, hold each for five controls, mask PPO to those seven decisions, and
      retain matched 235-step episode/storage.  Future/outcome labels, shooting targets, Tracker
      and BCPPO remain absent.
- [x] Gate it automatically with fresh seed171753 for one zero-optimizer horizon and fresh
      seed171754 for exactly two updates / 3,760 transitions.  Only exact execution, seven latches
      per world, finite tensors, zero divergence and the required zero/nonzero parameter contracts
      admit fresh seed171755 for exactly 32 updates / 60,160 transitions.  Freeze model31 and use
      seed181752 on data000; require 5 cm lift plus strict step235 validity before task-wide
      `16/20`.  Otherwise close the topology with no sweep and keep Tracker/BCPPO closed.
      Seed171753 passes 1,880 zero-optimizer transitions with 56 exact latches, zero divergence and
      zero parameter change.  Seed171754 passes two updates / 3,760 transitions with 112 latches,
      zero divergence and actor/critic/std deltas `0.000223/0.000407/0.0000477`.  Fresh seed171755
      completes 32 updates / 60,160 transitions with 1,792 latches and zero divergence.  Frozen
      seed181752 is finite and exact outside the correction interval, but data000 reaches only
      `0.01005 m` lift, `0.20851` bilateral contact and `0/1` strict completion.  The automatic
      task-wide branch is skipped; close this topology without a sweep and keep Tracker/BCPPO
      closed.

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
