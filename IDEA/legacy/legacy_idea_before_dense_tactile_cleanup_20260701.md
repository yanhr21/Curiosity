# Newton-Native Curiosity Adaptation

## Superseding Reset On 2026-07-01: Dense Tactile Infant, Not Contact-Count Curiosity

This section is the active project direction. Older Phase 00/01 notes below
remain useful only as historical evidence unless they are explicitly carried
forward here.

The previous Newton-native curiosity route is now legacy negative evidence and
engineering-chain review only. It must not be described as the current main
solution or as curiosity-training success. That route used a base
grasp/lift/hold controller, logged low-dimensional rollout state such as object
height, contact count, controller phase, mass, and friction labels, trained a
GRU forward model to predict next object motion, contact count, and
slip/contact-loss risk, computed learning-progress scores from forward-model
error change, and used those scores as weights for supervised residual
fine-tuning. Each candidate then trained for about one hour and was evaluated
on held-out cells.

The core failure is conceptual: this was offline learning-progress scoring plus
supervised residual fine-tuning, not closed-loop curiosity. The intrinsic signal
did not drive online rollout exploration, did not change the data distribution
through policy optimization, and did not make the agent actively choose probing,
regrasping, grip-force adjustment, pressure balancing, or shear-minimizing
actions. It also used scalar contact proxies, not tactile data. The old fields
`newton.panda.rigid_contact_count` and
`candidate.modality.contact_available_mask` are low-dimensional contact proxies
only. The old 10-dimensional model input did not contain left/right pad
pressure maps, compression maps, `Fn`, `Ft`, shear direction, contact area,
penetration/compression, marker flow, or tactile images.

The old result is negative evidence. Five real one-hour curiosity policy
candidates failed the strongest-baseline comparison, with
`positive_curiosity_result=false` and `safety_regression_cell_count=4`. A
checkpoint, a completed run, a nonblank video, or lower auxiliary loss is not
evidence of curiosity success. Do not run a sixth old-style one-hour residual
training attempt unless the user explicitly resets that stop gate.

The active target is now a reference-video-aligned dense tactile infant:

1. First build tactile-rich simulator/base evidence. Before restarting
   curiosity, there must be a base controller/model that completes
   grasp/lift/hold while exporting synchronized dense visual and tactile
   mechanics: visual scene, left/right tactile pad maps, pressure/compression
   heatmaps, normal force `Fn`, tangential/shear force `Ft`, shear direction,
   contact area, center of pressure, penetration/compression,
   material/friction/stiffness statistics, and grip/shear/contact time-series.
2. Upgrade the tactile representation from scalar proxies to dense pad-resolved
   fields:
   - `left_pad.pressure: [T, H, W]`
   - `left_pad.compression: [T, H, W]`
   - `left_pad.shear_u/v: [T, H, W]`
   - `left_pad.contact_mask: [T, H, W]`
   - `left_pad.Fn/Ft: [T]` or `[T, H, W]`
   - the same fields for `right_pad`.
3. Preserve provenance for candidate Newton/MJWarp fields. Use names such as
   `candidate.newton_mjw.Fn`, `candidate.newton_mjw.Ft`,
   `candidate.newton_mjw.area_proxy`,
   `candidate.newton_mjw.marker_flow`, and
   `candidate.newton_mjw.contact_normal`. Do not rename these proxies into
   official tactile semantics.
4. Treat Gate 00F as a low-priority final semantic-validation/comparison-gap
   track. UniVTAC, TaCauchy, and IsaacLab TacSL remain valuable final
   references, but they are not the high-priority active blocker for the
   Newton-only dense tactile Phase 00 reset.
5. Future curiosity must be true closed-loop dense visuo-tactile prediction and
   active probing. The forward model must predict tactile/contact/mechanics,
   not only object height or contact count. The policy must have meaningful
   exploratory actions such as grip-force adjustment, regrasping, pressure
   balancing, and shear-minimizing probing.

Strict reporting language from this reset:

- Old contact-count curiosity pipeline = legacy negative evidence.
- Current target = reference-video-aligned dense tactile environment plus base
  grasp/lift/hold.
- Success claim condition = harder held-out tasks beat the strongest baseline
  without safety regression.

Strict forbidden shortcuts:

- Do not call `rigid_contact_count` tactile; it is only a low-dimensional
  contact proxy.
- Do not describe old Phase 01 results as curiosity success.
- Do not call sample reweighting closed-loop curiosity.
- Do not use toy models as T-Rex, VQ-VAE, tactile encoder, Transformer, or
  world-model substitutes.
- Do not map proxy fields into official tactile keys:
  `area_proxy != real contact area`,
  `marker_flow render != photometric GelSight marker output`,
  `contact_count != tactile map`, and
  `candidate Fn/Ft != validated official tactile force field`.
- Do not let strong baselines hide the problem. If base grasp/lift/hold is too
  easy, move to harder tasks or finer metrics.
- Do not evaluate only success/fail. Required metrics include lift, hold
  duration, slip, drop, contact loss, object acceleration, force/contact cost,
  and safety regression against the strongest baseline.

## Active Reset On 2026-07-01: Reference-Video-Aligned Tactile Infant

The active direction is reset by the reference video
`0780e5ec3fdb26b63ae63de0f49f07c4.mp4`. The prior Phase 00/01
contact-count pipeline is now legacy evidence and negative evidence only. It
must not be treated as the active tactile standard.

The new target is to build a tactile-rich simulated infant that can first
grasp and report dense contact mechanics, then use those signals for curiosity
training. The reference video defines the minimum tactile standard:

- synchronized visual scene and tactile diagnostics;
- left/right pad tactile maps;
- contact pressure or compression heatmaps;
- normal force `Fn` and tangential/shear force `Ft`;
- shear direction vectors;
- contact area and center/proxy statistics;
- mean/max penetration or compression;
- material-specific contact response, starting with rigid steel/metal;
- time-series plots for grip force, shear, contact area, and compliance;
- high-frequency rigid contact output comparable to the 82 FPS diagnostic
  shown in the video, with lower-rate soft-body FEM allowed only when clearly
  labeled as soft/deformable simulation.

The immediate Phase 00 is therefore not "make more cup videos." It is:

1. Rebuild the simulator scene and tactile instrumentation so metal/steel
   stress change, contact normal, tangential friction, shear, contact area,
   penetration/compression, and per-pad force statistics are generated as real
   simulator outputs.
2. Prepare a basic grasping base model/controller that can complete grasp,
   lift, and hold while producing dense tactile/mechanics evidence at the
   reference-video level.
3. Only after those gates pass, restart curiosity training as closed-loop
   active probing over dense visuo-tactile prediction, not offline reweighting
   over scalar contact counts.

Transition update on 2026-07-01: the Newton `8c501...` candidate chain now has
enough engineering evidence to begin a Newton-only dense tactile curiosity
training track. This is not a Gate 00F pass. Official UniVTAC/TaCauchy/
IsaacLab TacSL semantic validation remains a pending comparison gap because
dependency-complete official runtimes are still unavailable. Phase 01 training
may proceed under that explicit boundary: use Newton dense tactile/mechanics
signals, keep full tactile-mask and baseline requirements, and do not claim
final reference-video tactile validation until official semantic validation
passes or the remaining blockers are separately resolved.

Current Phase 00 evidence update on 2026-07-01: latest official Newton main at
`a217e55fab3d373a08fba374cc5cafc1826cf27f` now meets the base runtime target
with `p00_bench_main_20260701_035529` (`92.6 FPS`) and exports synchronized
steel-spec calibrated tactile diagnostics in
`p00_calib_view_v1_20260701_040715`. The same latest-main base also exports
official Panda hydro scene/USD evidence in
`p00_main_usd_v1_20260701_041900` (`panda_hydro.usd`, `6903124` bytes).
These are positive environment/base assets, but they do not complete the
reference-video tactile gate because direct hydro `Ft`, scene+tactile raster
fusion, and validated gel/marker tactile rendering are still missing.

Source freshness correction on 2026-07-01: a later remote refresh found that
Newton upstream `main` has advanced to
`d58e70266be0db803261f3e46a2f7d923a43db37`. The current active evidence
worktree `external/newton_main` remains at
`a217e55fab3d373a08fba374cc5cafc1826cf27f`, so `a217e55...` should now be
called the active evidence base, not the latest upstream Newton main. A fresh
latest-main update and compute-side sanity are required before using
`d58e702...` for base-model or tactile evidence claims.

Latest Newton code preparation update on 2026-07-01:
`external/newton_d58` now exists as a detached latest-upstream worktree at
`d58e70266be0db803261f3e46a2f7d923a43db37`. This preserves
`external/newton_main` at the existing active evidence commit. The new worktree
has now run compute-side runtime benchmark and candidate tactile export, but
still needs reference-video comparison and Gate review before any base/tactile
gate claim.

Latest Newton d58 runtime update on 2026-07-01:
`p00_bench_d58_v1_20260701_070459` measured `70.8 FPS` and did not meet the
82 FPS target. A follow-up hot/longer run
`p00_bench_d58_hot_v1_20260701_070611` measured `82.7 FPS` on H200 and meets
the runtime target for latest upstream Newton `d58e702...`. This is runtime
sanity only, not tactile/base completion.

Latest Newton d58 tactile export update on 2026-07-01:
`p00_mjw_d58_marker_v1_20260701_071248` ran on latest upstream Newton
`d58e702...` inside Curiosity Slurm job `160467` and produced a nonblank
candidate direct-force tactile video, contact sheet, and NPZ time series. Manual
inspection of the sheet shows synchronized scene frames plus left/right marker
flow, Fn/Ft heatmaps, normal/shear overlays, force curves, and contact-area
proxy activation during grasp/lift/hold. Key observed values are `240` frames,
`147` frames with pad-object contacts, max lift `0.22254392504692078 m`, max
candidate Fn sum `40.08497619628906`, max candidate Ft sum `12.025492668151855`,
left/right marker-flow norms `3.722446918487549` and `3.3947927951812744`, and
steel-candidate material overrides `mu=0.3`, `kh=1e12` observed in Newton. This
upgrades d58 from runtime-only to candidate dense tactile/mechanics evidence,
but `direct_tactile_claim_allowed=false`: photometric marker semantics, true
contact-area semantics, reference-video comparison, and Gate review are still
open.

Latest Newton d58 reference/Gate update on 2026-07-01:
`p00_refcmp_d58_marker_v1_20260701_071521` generated reference-vs-candidate
assets for the d58 tactile video and passed asset-level comparison
(`pass_reference_comparison_assets`). `p00_chan_d58_marker_v1_20260701_071757`
then passed the conservative channel/layout audit with scene, marker-render,
Fn/Ft, area-proxy, mechanics-curve, and reference layout checks all detected.
The full d58 Gate review `p00_gate_d58_marker_v1_20260701_071843` is still
`open_not_curiosity_ready`: passed checks include `official_newton_runtime_82_fps`,
`base_grasp_lift_final_test`, `steel_spec_material`, `candidate_direct_fn_ft`,
`sensorcontact_alignment`, `normal_and_area_proxy_overlay`,
`candidate_gel_marker_render`, `reference_comparison_assets`,
`channel_semantic_layout_audit`, `semantic_reference_matrix_available`,
`semantic_bridge_spec_available`, and `reference_asset_reuse_plan_available`.
Failed checks are `reference_env_availability`, `reference_asset_availability`,
`univtac_official_reference_sanity`, and `tacauchy_official_reference_sanity`.
Curiosity training remains disallowed.

Latest Phase 00 visual-fusion update on 2026-07-01:
`p00_fused_cam_v1_20260701_043900` now fuses official Newton main
`SensorTiledCamera` head/right-wrist/left-wrist scene frames with calibrated
left/right `Fn`, shear-vector, deformation tactile maps and mechanics curves in
one synchronized diagnostic video/contact sheet. This removes the earlier
schematic-scene limitation for the active Newton hydro diagnostic. It remains
environment/base evidence only, not curiosity success: direct solver `Ft`,
direct pad-resolved shear force, and validated gel/marker tactile rendering
are still required before restarting curiosity training claims.

Latest Phase 00 reference-video comparison update on 2026-07-01:
`p00_refcmp_v3_20260701_065300` decodes the user reference MP4
(`720` frames, `30 FPS`, `2846x1510`) and the current steel-spec candidate
direct-force tactile AVI (`240` frames, `30 FPS`, `1180x820`), samples both,
and writes side-by-side comparison sheets. The candidate is nonblank and now
contains real Newton scene views plus synchronized candidate direct `Fn`/`Ft`
maps, shear arrows, object-z/force curves, steel-spec material evidence, and
compatible-scene `SensorContact` alignment. Manual comparison shows the
reference video is still richer: it has gel/marker-style tactile camera
diagnostics, denser multi-panel tactile fields, more tactile channel overlays,
and more complete time-series/mechanics panels. Therefore this is positive
Phase 00 reference-alignment evidence and a concrete gap list, not Gate 00D or
Gate 00E completion and not curiosity success.

Latest Phase 00 normal/area overlay update on 2026-07-01:
`p00_mjw_normarea_v1_20260701_071900` extends the steel-spec candidate
direct-force tactile video with contact-normal overlays from MJWarp
`contact.frame` and a candidate contact-area proxy from pad-object point-contact
density. It passed the official Panda hydro final test, used the same
steel-spec material override (`mu=0.3`, `kh=1e12`), had `147` pad-object
contact frames, max object lift `0.22243636846542358` m, max candidate `Fn`
sum `40.0997428894043`, max candidate `Ft` sum `12.027881622314453`, left/right
area-proxy cell ratios `0.2900390625` / `0.279296875`, and zero read errors.
`p00_refcmp_normarea_v2_20260701_073000` then reran the reference comparison and
updates the gate checklist: the candidate now has direct visual normal overlay
and contact-area proxy overlay. The remaining tactile gaps are gel/marker-style
tactile rendering, validated marker/deformation tracking, validated real
contact-area semantics beyond the proxy, channel-by-channel semantic matching,
and final Gate 00D/00E review.

Latest Phase 00 candidate marker-render update on 2026-07-01:
`p00_mjw_marker_v1_20260701_074200` adds a blue gel-like candidate
marker/deformation rendering derived from the already exported candidate
`Fn`/`Ft`/normal/contact-area-proxy fields. It passed the official Panda hydro
final test with zero read errors, `146` pad-object contact frames, max object
lift `0.2225111573934555` m, max candidate `Fn` sum `41.90861511230469`, max
candidate `Ft` sum `12.294239044189453`, and nonzero left/right marker-flow
norms `4.690944671630859` / `3.1349213123321533`. The follow-up comparison
`p00_refcmp_marker_v1_20260701_074900` confirms the active candidate channel
set now includes gel/marker-style rendering derived from direct-force fields.
This is still candidate visualization only, not validated Taccel or hardware
photometric marker output.

Latest Phase 00 gate review update on 2026-07-01:
`p00_gate_review_v2_20260701_080800` reviews Gate 00D/00E against the current
evidence chain. It passes all current evidence checks: official Newton runtime
above `82 FPS`, base grasp/lift final test, steel-spec material settings,
candidate direct `Fn`/`Ft`, `SensorContact` alignment, normal/area proxy
overlay, candidate gel/marker render, and nonblank reference-comparison assets.
The gate status is still `open_not_curiosity_ready`: Gate 00D is
`open_reference_semantics_blocked` and Gate 00E is
`open_tactile_validation_blocked`. Curiosity training remains disallowed until
validated gel/marker photometric semantics, validated deformation-marker
tracking, validated real contact-area semantics, and channel-by-channel
reference-video semantic matching are solved.

Latest Phase 00 channel-audit update on 2026-07-01:
`p00_chan_audit_v1_20260701_082100` adds a channel-level visual layout audit
between the user reference video and the current marker candidate. It passes
scene, marker-render, force-heatmap, area-proxy, mechanics-curve, and reference
scene/tactile/mechanics layout checks, and writes a boxed audit sheet. The
follow-up `p00_gate_review_v3_20260701_082600` consumes this audit and adds
`channel_semantic_layout_audit` to the passed checks. Gate 00D/00E still remain
open because the audit is layout-level only, not validated photometric or
physical semantic equivalence.

Latest 2026 tactile-reference scan and local source audit on 2026-07-01:
newly relevant reference paths include UniVTAC, Tacmap, TaCauchy, and
ControlTac. UniVTAC was cloned to `external/UniVTAC` at official commit
`05bcd3edb92237107efa40105292a24f1a9fd761`; it supports Isaac Lab/TacEx
visuo-tactile manipulation tasks, data collection, ACT/ViTAL policy baselines,
and left/right tactile `rgb`, `rgb_marker`, `depth`, and `marker` fields.
TaCauchy was cloned to `external/TaCauchy` at official commit
`c228cfe9050904cd5d71d64f6eb5104768d4cbda`; it is an Isaac Sim/Lab plus UIPC
FEM tactile reference for Cauchy stress, normal pressure, tangential traction,
adaptive mesh refinement, force-field visualization, and tactile RGB images.
Tacmap emphasizes geometry-consistent penetration/deform maps, and ControlTac
is a force/position-conditioned tactile image generation reference, but no
local official code was obtained for them in this audit. These are
comparison/validation references for the open semantic blockers, not
replacements for the current Newton base without a separate official sanity and
compatibility check.

Latest Gate 00F enforcement update on 2026-07-01:
`p00_ref_univtac_sanity_v1_20260701_054900` and
`p00_ref_tacauchy_sanity_v1_20260701_054900` ran inside Curiosity held Slurm
job `160450` on `server02` only as official reference sanity/blocker probes.
Both official repositories matched their expected commits, but both probes
recorded `blocked_missing_prebuilt_environment` because no executable prebuilt
UniVTAC or TaCauchy Python environment exists under the approved local
environment paths. The follow-up `p00_gate_review_v4_20260701_055100` now
reviews Gate 00D/00E/00F together: mechanical/runtime/candidate-layout checks
still pass, but `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity` fail, so Gate 00F remains
`open_official_semantic_validation_blocked` and curiosity training remains
disallowed.

Latest Gate 00F bridge-spec update on 2026-07-01:
`semantic_bridge_spec_v1.json` now maps current candidate Newton channels
(`Fn`, `Ft`, marker flow, area proxy, contact normal, scene RGB) to official
UniVTAC and TaCauchy target semantics. `p00_gate_review_v5_20260701_060100`
ran inside Curiosity Slurm job `160454` on `server02` and verifies that the
gate review now checks both `semantic_reference_matrix_available` and
`semantic_bridge_spec_available`. The bridge-spec check passes, but UniVTAC and
TaCauchy official sanity still fail with `blocked_missing_prebuilt_environment`;
therefore Gate 00D/00E/00F remain open and curiosity training remains
disallowed.

Latest Gate 00F environment and asset update on 2026-07-01:
`reference_env_location_audit_v1.json` confirms that no approved target
UniVTAC/TaCauchy Python environment exists under the active paths, even though
project-local `envs/taccel/miniforge/bin/conda` is available as an env-creator
candidate. `reference_asset_availability_v1.json` adds a second Gate 00F
blocker: UniVTAC bundled TacEx has useful GelSight/GF225/shape assets, but
TaCauchy has only partial placeholder assets and lacks required full sensor
USD/calibration files, valid Franka UIPC assets, and
`Props/tactile_test_shapes`. `reference_asset_reuse_plan_v1.json` records a
candidate local asset reuse path from the UniVTAC bundled TacEx asset tree into
TaCauchy, but this was not executed and cannot be treated as official asset
completion. `phase00_gate_review.py` and its Slurm/tmux runners now accept
asset availability and asset reuse evidence as Gate 00F inputs; the updated
code has only passed syntax checks and has not been rerun in compute after the
asset-check change.

Latest post-approval Gate 00F/source update on 2026-07-01:
After the user said `全都允许继续`, TaCauchy asset file presence was repaired by
approved local reuse from UniVTAC bundled TacEx. `Sensor.usd` is present and
the tactile test shape USD count is `21`. Base Python env prefixes now exist:
`envs/univtac/conda/bin/python` is `Python 3.10.20`, and
`envs/tacauchy/conda/bin/python` is `Python 3.11.15`. This clears file-level
asset/env availability only; it is not official dependency readiness or
official sanity. The current Gate 00F readiness result remains
`gate00f_ready=false` with
`reason=blocked_official_sanity_or_gate_review_not_passed`; the effective
remaining failed checks are `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity`. Official readiness is now blocked by
the dependency-install location policy: heavy Isaac/TacEx/UIPC installs cannot
run silently on the login node, and dependency installation/builds are
forbidden on compute nodes. Continuing requires a prebuilt Curiosity reference
environment or a compliant non-login env-prep workflow.

Latest official source recheck V3 on 2026-07-01:
Newton upstream `main` is now
`8c501b47847569fecdda97a9f7f01205c6f7964f`, and a latest code worktree exists
at `external/newton_8c501`. This does not replace the d58 runtime/tactile
evidence by itself; its H200 runtime benchmark has now been tried and reached
`80.1`/`80.8 FPS`, which is acceptable for continuing dense tactile export,
reference comparison, channel audit, and Gate review. The old `82 FPS` number
is historical reference only and must not block progress. Correct
official source URLs are now recorded: Taccel
`https://github.com/Taccel-Simulator/Taccel.git`, TaCauchy
`https://github.com/figsama/TaCauchy.git`, HydroShear
`https://github.com/MMintLab/hydroshear.git`, IsaacLabTactile
`https://github.com/UM-ARM-Lab/IsaacLabTactile.git`, and TacEx
`https://github.com/DH-Ng/TacEx.git`. `external/TacEx` was cloned at
`adceed41afb7cb48f9ec1f66a662fb8e5a06627f`; `external/IsaacLabTactile` was
cloned at `21bcb476b27ceedccccd63afef6bbd822adc2b2b` with
`GIT_LFS_SKIP_SMUDGE=1` and blob filtering. `git-lfs` is unavailable on the
current PATH, so IsaacLabTactile asset completeness and official sanity are not
verified. Source acquisition is not Gate 00F completion and not curiosity
readiness.

Newton `8c501...` compute handoff on 2026-07-01:
`experiments/reports/phase00/ref_tactile/newton_8c501_sanity_handoff.md` and
`experiments/configs/phase00/ref_tactile/newton_8c501_sanity_handoff_v1.json`
record exact tmux-held Slurm commands for runtime benchmark, dense tactile
export, reference comparison, channel audit, and Gate review. The runtime
benchmark stage has been executed and is acceptable around 80 FPS; the
downstream 8c501 dense tactile export/reference/Gate stages have not been
executed and should proceed when a Curiosity tmux-held Slurm allocation is
available.

Newton `8c501...` runtime sanity result on 2026-07-01:
job `160854` ran two official Panda hydro null-viewer H200 benchmarks on
`external/newton_8c501`. The first measured `80.1 FPS` over `30.01s`; the
second measured `80.8 FPS` over `60.00s`. Both executed successfully but
should be treated as acceptable runtime evidence. Do not use the old `82 FPS`
reference to block `8c501...` dense tactile export. The remaining blockers are
tactile semantic completeness, official reference sanity/blocker evidence, and
base grasp/lift/hold validation.

Latest additional reference-code recheck on 2026-07-01:
FreeTacMan was cloned to `external/FreeTacMan` at official commit
`9285740a5d33385d3a9cf5ccdb185e3387b547bd` as a secondary 2026 real
visuo-tactile data and tactile-pretraining reference. DiffTactile was cloned
to `external/DiffTactile` at official commit
`c4bf43d44071758aea68a5c7ae125fc8257bb8e1` as a secondary differentiable
tactile simulator reference with FEM tactile sensors, marker extraction,
contact-rich manipulation tasks, and CMA-ES/PPO/SAC/RNN baselines.
Tacmap and ControlTac remain code-unavailable comparison gaps after project/
paper review and common GitHub remote probes. This does not change Gate 00F:
UniVTAC and TaCauchy remain the mandatory official semantic-validation
references before curiosity can restart.

Latest active-curiosity reference audit on 2026-07-01:
APPLE was cloned to `external/APPLE` at official commit
`4b1d71fadb786d865d4ee29a184ab408b9605083`, and Tactile MNIST was cloned to
`external/tactile-mnist` at official commit
`9e4e59139e9349ab361a3b9297f4815724ad6387`. APPLE is an ICLR 2026 active
perception via reinforcement learning codebase with SAC/CrossQ/PPO/random/grid
baselines and ViT-based tactile image configurations. Tactile MNIST provides
active GelSight Mini tactile perception environments, tactile-only
observations, sensor movement actions, train/test/holdout mesh splits, and
real/synthetic tactile image datasets. These are Gate 00G design references for
future closed-loop active probing and tactile-mask training. They are not
Newton-native grasping infant checkpoints, not Gate 00D/00E/00F completion,
and not evidence of current curiosity success.

Latest policy/photometric reference audit on 2026-07-01:
Reactive Diffusion Policy was cloned to `external/reactive_diffusion_policy` at
`824c5e8de1fd1811106907a04b5f0186e0138c0b`; ImplicitRDP was cloned to
`external/ImplicitRDP` at `4c90646df17787e31c88838106c4a0323ddefb4a`; and
Tactile Diffusion was cloned to `external/Tactile-Diffusion` at
`16868fb96d19d93dc5837600c26b48415632e4f6`. RDP/ImplicitRDP are future serious
visual-tactile or visual-force policy baselines with official dataset/checkpoint
links. Tactile Diffusion is a future photometric tactile-image generation
reference. None of them replaces Newton/Taccel base evidence or UniVTAC/TaCauchy
Gate 00F semantic validation.

The base implementation must start from current serious codebases and official
paths where possible. As of 2026-07-01, the current source-backed shortlist is:

- Newton: primary physics/runtime target. Upstream has moved beyond the local
  checked-out version; latest observed upstream main is
  `a217e55fab3d373a08fba374cc5cafc1826cf27f`, with tag `v1.3.0`.
- Taccel: primary tactile-simulation reference for high-performance
  vision-based tactile sensors, IPC/ABD, robot/sensor/object modeling, and
  tactile examples. Local checkout is at upstream main
  `cb23bc251b531ba6908a3788c2f91423cd543149`.
- T-Rex: primary tactile-reactive policy/model reference, especially the
  temporal tactile VQ-VAE and high-frequency tactile expert. Upstream main is
  `43ff632259d76f08373c085c53111825060d029b`; local checkout has unrelated
  dirty changes and must not be overwritten silently. A clean latest source
  snapshot now exists at `external/T-Rex_43ff`, and the full-pipeline branch is
  separately cloned at `external/T-Rex_full_b23`
  (`b23eafe564a1457cd4eacb889aaf6fbf29a29034`). Official released checkpoints
  are `miniFranka/T-Rex_pretrain_mecka22k_epoch1` and
  `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`; the midtrain
  checkpoint embeds the tactile VQ-VAE and is the strongest future
  tactile-reactive policy starting point after Gate 00F and a faithful
  Newton-to-T-Rex data contract.
- HydroShear and IsaacLab TacSL/IsaacLabTactile are reference paths for
  hydroelastic shear, force fields, and visuo-tactile simulation design, not a
  replacement for the Newton/Taccel mainline unless a recorded blocker forces
  a platform decision.
- Official Isaac Lab main TacSL is now tracked separately from the earlier
  UM-ARM-Lab/IsaacLabTactile clone. `external/IsaacLab_official` is sparsely
  cloned at `b4c321024792976150ca55fddb26fa34480d974e` and exposes
  `tactile_depth_image`, `tactile_rgb_image`, `penetration_depth`,
  `tactile_normal_force`, and `tactile_shear_force` fields plus normal,
  tangential, friction, tactile-array, and SDF contact-object configuration.
  It is a Gate 00F semantic-validation candidate, not completion evidence until
  official TacSL sanity runs in an approved environment.
- UniVTAC and TaCauchy are mandatory Gate 00F semantic-validation references.
  FreeTacMan is a secondary real visuo-tactile data/pretraining reference.
  DiffTactile is a secondary differentiable tactile simulation reference.
  Tacmap and ControlTac are comparison gaps until official code is available.
- APPLE and Tactile MNIST are secondary Gate 00G active tactile perception
  references for future curiosity design, not base grasp checkpoints and not
  semantic-validation replacements.
- Reactive Diffusion Policy and ImplicitRDP are secondary future policy
  baselines/checkpoint references. Tactile Diffusion is a secondary photometric
  tactile-image reference. They cannot be used as current success evidence
  without official environment/checkpoint/schema sanity.
- FTP-1 is now tracked at `external/ftp1-policy`, commit
  `dd7cda66c7e97a170e0435fc6c4428b350cbdcc0`, as a 2026 generalist
  foundation tactile policy/checkpoint reference. AnyTouch2 is now tracked at
  `external/AnyTouch2`, commit
  `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`, as an ICLR 2026 optical tactile
  representation reference. These are future serious baselines/encoders after
  Gate 00D/00E/00F, not current base-model completion and not a shortcut around
  official tactile semantic validation.

The new tactile representation must not be reduced back to
`rigid_contact_count`, `contact_available_mask`, or scalar slip-risk labels.
Those are allowed only as auxiliary low-dimensional summaries derived from the
dense tactile/mechanics record.

Curiosity must be redesigned around dense prediction:

- predict future tactile fields and mechanics, not only object `z`;
- give intrinsic reward for bounded learning progress over tactile pressure,
  shear, contact area, force balance, slip precursor, and object motion;
- use tactile-mask and vision-mask training so the policy remains balanced:
  vision+tactile, tactile-only masked vision, vision-only ablation, and noisy
  tactile ablation must all be evaluated;
- require held-out improvement over the strongest non-curiosity base model
  without safety regression before any curiosity success claim.

## Core Idea

This project studies embodied curiosity for contact-rich manipulation. The
agent should not learn grasping from raw pixels alone, and it should not be
forced to imitate T-Rex's full data format before the simulator can honestly
produce the same physical signals.

The current direction is:

1. Use Newton as the primary simulator for closed-loop manipulation.
2. Use reliable Newton signals first: object pose, end-effector pose, robot
   state, action targets, contact counts, contact proxies, and camera views.
3. Add Taccel tactile-marker or deformation evidence only when it is real,
   nonzero, visually inspected, and kept under explicit provenance namespaces.
4. Treat T-Rex as a strong reference policy and future bridge, not as the
   immediate gate for all progress.
5. Evaluate whether curiosity and contact prediction help a grasping system
   adapt to object properties such as mass, friction, fill level, compliance,
   slip, and unexpected force response.

The useful research question is:

Can a robot with a basic grasping prior improve closed-loop manipulation by
actively testing physical hypotheses about objects, then adapting grip force,
lift speed, regrasp timing, and stabilization based on prediction errors over
object motion, contact, and tactile/contact evidence?

## Non-Negotiable Final Target

The final target is not a toy curiosity pipeline and not a quick proof that a
small forward model can train. The final target is complete closed-loop
curiosity training that improves manipulation on harder tasks.

Required final evidence:

- complete closed-loop curiosity training or adaptation, not only replay
  scoring and not only supervised reweighting;
- harder task evaluation beyond the first easy cup lift/hold benchmark;
- full rollout visualization videos, plus frame browsers/contact sheets for
  inspection;
- strict metrics for lift, hold, slip, drop, contact loss, acceleration,
  excessive-force or safety cost, adaptation speed, and failure modes;
- direct comparison against no-adaptation, scripted feedback, no-curiosity
  residual adaptation, curiosity ablations, and serious/mainstream reference
  methods or official checkpoints when available;
- clear evidence that curiosity is better than the declared baseline on the
  harder-task metrics before any success claim is allowed.

If a run only shows that the code path executes, only matches the baseline, or
improves one metric while hiding a safety regression, it must be recorded as
intermediate or negative evidence. It must not be used to claim that curiosity
training is complete. If a mainstream method or official checkpoint is
unavailable or incompatible, that must be documented as a comparison gap or
blocker rather than used to lower the bar.

Harder training is a non-negotiable continuation requirement. The project must
not stop after a negative, weak, or merely runnable curiosity result. If the
current curiosity policy does not beat the strongest declared baseline on the
harder held-out tasks, the next work item is to continue with a faithful fix:
stronger closed-loop data collection, a corrected policy/adaptation objective,
required ablations, a baseline audit, or a clearly documented blocker. The
claim must never be downgraded to "the model trained", "a checkpoint exists",
"a video rendered", or "curiosity scores were computed".

The next training standard is deliberately harder than the current Phase 03
pipeline. It must not be reduced back to the easiest original cup cells, a
single held-out-like run, an offline curiosity score, or supervised-only sample
weighting. A valid hard-training attempt must start from a declared task family
such as Phase 07 variable water-cup weight/fill, preserve train/validation/
held-out separation, run real curiosity-driven policy/adaptation updates, and
then compare against the declared baselines with videos and strict safety
metrics. If the result is negative, unstable, or blocked, the correct outcome
is a recorded negative result or blocker, not a downgraded success claim.

User reaffirmation on 2026-06-27: this harder-training bar is a durable project
requirement, not a one-turn preference. Future work must not repeat the failure
mode of lowering the task difficulty, treating a toy or weak run as complete,
or exiting quickly after negative evidence. Until a harder-task curiosity policy
beats the strongest declared baseline without safety regression, the project
state remains incomplete and the next action must be a faithful continuation:
objective repair, stronger data collection, ablations, baseline audit,
mainstream-method comparison, or a clearly documented blocker.

Continuation lock from the same user reaffirmation: every future result must be
classified before any completion language is used. A run is incomplete if it
lacks real closed-loop curiosity updates, harder held-out evaluation, full
videos, strict safety metrics, required ablations, or a faithful comparison
against serious/mainstream methods. A run is negative evidence if curiosity
fails to beat the strongest declared baseline without safety regression. Either
case requires continued faithful work or a documented blocker; neither case may
be converted into a downgraded success claim or quick exit.

Harder-training persistence lock from 2026-06-27: this requirement must remain
written in the project memory files and must be treated as part of the research
definition, not as a temporary preference. The intended target is deliberately
harder than a runnable toy pipeline: complete closed-loop curiosity training,
harder held-out manipulation, full rollout videos, strict safety metrics,
ablations, and faithful serious-method comparison or documented blocker. A
future agent must not lower the target to make progress look complete. A weak,
negative, resource-limited, or inconvenient result must be recorded honestly
and followed by continued faithful training/evaluation, objective repair,
baseline audit, ablation, stronger data collection, official-method setup, or a
documented blocker.

User reaffirmation on 2026-06-28: this requirement must also remain written in
the idea, agent, plan, and todo files so future work cannot repeat downgrade or
quick-exit behavior. A training run, ablation queue, checkpoint, video, or
negative comparison is not completion unless the declared harder-task contract
passes. The project must continue toward a closed-loop curiosity policy that
beats the strongest declared baseline on harder held-out tasks without safety
regression, with faithful serious-method comparison or a documented blocker.
Until then, the correct state is incomplete or negative evidence followed by
continued faithful work.

Latest enforcement lock on 2026-06-28: the harder-training requirement is part
of the research definition, not a negotiable implementation detail. Future
work must not downgrade it into an easy task, toy substitute, smoke test,
offline score, validation-only threshold repair, queue completion, checkpoint
artifact, single video, or negative result described as success. A completion
claim requires harder closed-loop held-out evaluation, strongest-baseline
comparison, strict safety metrics, full videos, ablations, and faithful
serious-method comparison or a documented blocker. If that evidence chain is
not present and positive, the project remains incomplete or records negative
evidence, then continues with faithful objective/data/model repair or the next
approved evaluation/training step.

Evidence challenge reaffirmation on 2026-06-28: every future claim that
curiosity training is complete must point to a complete evidence chain, not just
to a finished script. Required evidence includes the exact training command and
log, checkpoint, official sanity check, held-out harder-task metrics, strongest
baseline comparison, safety comparison, full rollout videos, manual visual
inspection, and faithful serious-method comparison or blocker. Validation-only
threshold repair, diagnostic rollouts, negative held-out comparison, or a
rendered video is not evidence of completed curiosity learning.

Harder-training persistence reaffirmation on 2026-06-28 after the V3
rank-calibrated negative result: real one-hour training, improved validation
active loss, a written checkpoint, complete held-out videos, and nonblank
visual inspection are still not enough if the harder held-out all-cell
baseline/safety gates fail. This project must not repeat the previous downgrade
or quick-exit pattern. Negative evidence must be recorded as negative, the
state must remain incomplete, and the next action must be faithful objective
repair, stronger data/model repair, official-method setup, or a documented
blocker.

Closed-loop repair direction on 2026-06-28: after the V3 rank-calibrated
held-out failure, the next attempt must move beyond offline curiosity-score
reweighting. The implemented repair path records scripted corrective teacher
labels under `candidate.teacher.*` while the current learned residual policy
actually controls Newton on train/validation cells. This creates an on-policy
DAgger-style closed-loop source distribution for the next real one-hour policy
update, then evaluates the resulting checkpoint on the harder held-out cells
with full rollout videos. This is still not a final success claim until it
beats the strongest baseline and mainstream/official comparison gates.

Repair update on 2026-06-28 after the closed-loop teacher held-out failure:
the evidence shows dense corrective labels caused over-intervention. Future
training must not repeat dense teacher imitation unless paired evidence proves
the intervention beats no-adaptation. The immediate repair direction is
advantage-gated residual data: compare paired no-adaptation and intervention
rollouts, admit only cells with nonnegative hold/lift gain and no safety
regression, and fail the preflight instead of producing a convenient but harmful
training CSV. In parallel, use official Newton object-family probes such as the
`pen` scene only as source-selection diagnostics to find a harder distribution;
these probes are not success claims and are not a substitute for the final
closed-loop curiosity result.

## Infant Analogy

The intended analogy is not a newborn learning all motor control from scratch.
The intended agent is closer to an infant that already has a basic grasping
ability and then learns how the world pushes back.

For example:

- The agent expects a cup to be heavy because it appears full.
- It lifts the cup and observes that the acceleration, contact response, and
  force requirement are lower than expected.
- That mismatch becomes useful prediction error, not just noise.
- The agent adjusts grip force, lift speed, and stabilization.
- Over repeated encounters, it learns that visually similar cups can have
  different mass, friction, fill level, and deformation response.

The same pattern should apply to slippery objects, deformable pouches, handles,
thin cards, fragile objects, and objects whose apparent geometry does not fully
predict contact behavior.

This is the "养" stage: start from a basic manipulation prior, then use
closed-loop curiosity and physical feedback to become better adapted to
different objects.

## Why T-Rex Is No Longer The Immediate Gate

T-Rex remains important, but it is not the current main bottleneck to solve
first.

The official T-Rex tactile-reactive data was collected on a real bimanual
Dexmate Vega-1 robot with two Sharpa Wave hands. Its training data expects
real synchronized fields such as bimanual state/action, head and wrist cameras,
10 fingertip force/torque streams, and 10 tactile deformation streams.

Newton can simulate robots, including bimanual setups, but current workspace
evidence does not yet provide a faithful T-Rex-equivalent episode source:

- current Newton Panda sources are single-arm and not T-Rex bimanual Sharpa;
- current bimanual Allegro evidence gives a different state and tactile-sensor
  contract;
- current Taccel marker evidence is real and nonzero, but it is marker-flow
  evidence, not calibrated T-Rex `[10,6]` F6;
- current dense tactile render attempts did not pass the nonuniform deformation
  visual gate;
- current source packages still lack the synchronized 62D state/action,
  accepted cameras, calibrated F6, and 10 dense tactile deformation streams
  required for strict T-Rex promotion.

Therefore, generating shape-compatible simulated data is not enough. If the
robot embodiment, action semantics, tactile calibration, and camera semantics
are wrong, a T-Rex loader may run while the experiment remains scientifically
invalid.

T-Rex should now be used as:

- a reference architecture for tactile-reactive policies;
- a checkpoint/reference baseline where official data is available;
- a future bridge once a faithful or explicitly accepted equivalent simulator
  source exists;
- not the required format for every Newton curiosity experiment.

Current reference-checkpoint status as of 2026-06-27: the staged official
T-Rex midtrain assets passed checkpoint integrity and official model-load
sanity in the Curiosity allocation. Evidence is recorded in
`experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
This keeps T-Rex available as a reference checkpoint, but it does not solve the
Newton-to-T-Rex data-contract gap and does not replace the Newton scripted
infant prior.

## Current Mainline

The mainline is Newton-native closed-loop curiosity adaptation:

1. Build manipulation tasks with object property variation:
   mass, friction, fill-level proxy, compliance, shape, handle geometry, and
   fragility/safety tags.
2. Provide a basic grasping prior:
   scripted controller, behavior cloning, diffusion-policy-style policy, or
   another serious manipulation baseline.
3. Add intrinsic objectives over physical prediction:
   object motion, lift success, contact change, slip proxy, force/contact
   proxy, tactile-marker response, and safety cost.
4. Compare against baselines:
   no adaptation, scripted adaptation, contact-aware prediction, curiosity
   reward, and tactile/marker-aware variants.
5. Only later attempt T-Rex bridge work if the Newton-native mechanism proves
   useful and the required data contract can be satisfied without padding or
   renaming.

## Signals

Allowed current source namespaces:

```text
newton.state.*
newton.action_target.*
newton.object.*
newton.contact.*
newton.camera.*
taccel.marker.*
taccel.ftac.*
candidate.*
```

Forbidden promotions without strict evidence:

```text
observation.state
action
action_abs
observation.images.*
observation.tactile_f6
observation.tactile_deform.*
```

The project may use Newton and Taccel source data directly under their own
names. It must not rename partial evidence into official T-Rex fields.

## First Target Task Family

The first task family should be lift-and-hold adaptation:

- objects: cups, boxes, cylinders, pouches, cards, and simple handled objects;
- variation: mass, friction, fill-level proxy, compliance, and initial pose;
- initial skill: basic grasp and lift;
- challenge: detect that the expected physical response was wrong;
- adaptation: adjust grip force, lift speed, regrasp timing, or stabilization;
- metrics: lift success, slip/drop rate, excessive-force rate, object motion
  prediction error, contact prediction error, adaptation speed, and safety cost.

This directly matches the cup example: the agent may expect a water-filled cup
to be heavy, observe that it is lighter, and adapt force without crushing or
dropping it.

## Baseline Policy

The baseline should be serious and explicit:

- scripted or impedance grasp controller as a control baseline;
- behavior cloning or diffusion-policy-style baseline from generated
  demonstrations;
- contact-aware ICM or learning-progress curiosity as the intrinsic baseline;
- T-Rex compatibility diagnostics only as reference, not as the main baseline
  unless a faithful bridge exists.

Do not write a toy VQ-VAE, toy Transformer, toy world model, or toy T-Rex clone.
Any small diagnostic model must be labeled as a diagnostic and not represented
as faithful T-Rex progress.

## Training Strategy

The first learned system should not learn grasping from scratch. It should use
a reliable basic grasp-and-lift prior, then learn residual adaptation around
that prior. The intended "infant" has primitive manipulation ability already:
it can approach, close the gripper, lift, and hold, but it still needs to learn
how different objects push back.

Current decision: the short-term infant prior is the official Newton Panda
hydro scripted grasp/lift path, not a pretrained checkpoint. Web and local
source checks did not identify a directly usable Newton-native Panda
grasp/lift checkpoint. OpenPI DROID/Franka, Isaac Lab Mimic, and Isaac Sim
Franka policy assets remain future audit candidates, but they are not the
active short-term route.

User-approved route as of 2026-06-27: use this short-term stable method now.
The project should not wait for a Newton-native pretrained grasp checkpoint
before building the first infant baseline, feedback-adaptation baseline, and
residual-learning path. Checkpoint audits can continue later, but they are
secondary to the Newton scripted-prior route until a compatible official policy
is proven through the same visual and metric gates.

Short-term stable route: keep the official Newton Panda hydro scripted
grasp/lift controller as the non-learned infant prior, make the baseline
physics honest first, then learn residual adaptation around that prior. The
first learning target is not an end-to-end grasp policy. It is a small
controller-parameter or residual policy that changes grasp/lift/hold behavior
based on object motion, contact, slip, and later tactile evidence. This route
is selected because it already gives reliable primitive manipulation behavior
and avoids pretending that an embodiment-mismatched checkpoint is a
Newton-native grasping model.

The immediate implementation must prefer a pre-finalization Newton
mass/inertia/friction adapter in the official Panda hydro builder path, or a
documented Newton model-update API that passes the camera/export/visual gate.
Do not continue the runtime model-array mutation path that repeatedly produced
Warp CUDA illegal memory access during SensorTiledCamera/export cleanup.
As of 2026-06-27, the pre-finalization builder adapter is the active short-term
route. It has passed fresh official Newton sanity, SensorTiledCamera export,
automated visual validation, and manual visual inspection, and it has already
produced real cup variants for `empty_medium`, `half_medium`, and
`full_medium`. All three medium-friction variants lift and hold the cup with
low slip and no drop, but all fail the strict baseline only on the
object-acceleration threshold. That failure is useful: it identifies the first
residual-adaptation target as gentler stabilization of the lift/hold trajectory
rather than end-to-end grasp discovery.
The low-friction axis has also completed its non-held-out empty and half cells
(`empty_low`, `half_low`); both show the same pattern: real low friction is
applied in Newton, lift/hold/slip/drop/contact gates pass, and the strict
failure remains object acceleration. `full_low` remains a held-out
generalization cell.
The ordinary high-friction axis has completed with `half_high` and
`full_high`; both apply real Newton friction, pass lift/hold/slip/drop/contact
gates, and fail only on the strict object-acceleration threshold. The held-out
`full_low` and `empty_high` cells have also been evaluated as no-adaptation
evidence and must remain labeled as held-out generalization evidence for later
learned-adaptation comparisons.
The scripted feedback adaptation baseline has been configured as
`CONTROLLER_MODE=lift_hold_feedback` around the same official Newton scripted
prior. It uses real object-motion and contact-proxy feedback to adjust lift
duration, hold target, and stabilization timing. It is not a learned policy and
its nominal cup gate has passed fresh official Newton sanity, camera export,
visual validation, and manual visual inspection. Shared metrics still mark the
nominal feedback run as fail only on the strict object-acceleration threshold;
lift, hold, slip, drop, and contact gates pass. The nominal run did not trigger
feedback, which is acceptable because the rule should not perturb stable
nominal behavior without a detected mismatch.
The first ordinary feedback grid cell, `empty_low`, has also completed with
real Newton mass/friction applied. It passes lift/hold/slip/drop/contact gates
and fails only on the strict object-acceleration threshold. Feedback did not
trigger, so this is an honest scripted-feedback baseline result rather than a
claim that adaptation improved behavior.
The second ordinary feedback grid cell, `empty_medium`, shows the same pattern:
real mass/friction applied, visual and contact gates pass, strict metrics fail
only on object acceleration, and feedback does not trigger.
The third ordinary feedback grid cell, `half_low`, also completes with real
Newton mass/friction provenance. It passes visual, lift, hold, slip, drop, and
contact gates, fails only on the strict object-acceleration threshold, and does
not trigger feedback. This keeps the scripted-feedback result honest: it is a
runnable controller-parameter feedback baseline, not a learned adaptation or
curiosity result yet.
The fourth ordinary feedback grid cell, `half_medium`, completes with the same
validated pattern: real Newton half-mass and medium-friction settings are
applied, visual/lift/hold/slip/drop/contact gates pass, strict metrics fail
only on object acceleration, and feedback does not trigger.
The fifth ordinary feedback grid cell, `half_high`, also completes. Real
Newton half-mass and high-friction settings are applied, visual and task gates
pass, strict metrics fail only on object acceleration, and feedback remains
inactive.
The sixth ordinary feedback grid cell, `full_medium`, completes with real
Newton full-mass and medium-friction settings. It passes visual, lift, hold,
slip, drop, and contact gates, fails only on object acceleration, and does not
trigger feedback.
The seventh and final ordinary feedback grid cell, `full_high`, completes the
ordinary scripted-feedback mass/friction grid. It applies real Newton full-mass
and high-friction settings, passes visual/lift/hold/slip/drop/contact gates,
fails only on object acceleration, and does not trigger feedback. `full_low`
and `empty_high` remain held-out cells for later learned-adaptation comparison.
The first held-out scripted-feedback evaluation cell, `full_low`, has now been
run as held-out evidence rather than ordinary/training evidence. It applies
real Newton full-mass and low-friction settings, passes visual/lift/hold/slip/
drop/contact gates, fails only on object acceleration, and does not trigger
feedback.
The second held-out scripted-feedback evaluation cell, `empty_high`, also
completes. Both held-out cells now pass visual/lift/hold/slip/drop/contact
gates with real Newton physics provenance, both fail only on the strict
object-acceleration threshold, and neither triggers feedback. This completes
the scripted-feedback evaluation grid, but it still does not justify an
adaptation-improvement claim.

As of 2026-06-27, the short-term stable method is explicitly selected for the
next work: do not wait for an unverified Newton-native checkpoint, and do not
train a placeholder policy. Continue from the official Newton Panda hydro
scripted infant prior, collect residual controller-parameter labels only from
ordinary cells, and promote a label source only if it has nonzero feedback
while passing official Newton sanity, automated/manual visual inspection,
lift, hold, drop, and contact gates. The first two nonzero residual diagnostics
(`residual_label_source_sensitive_feedback_half_low_20260627_030145` and
`residual_label_sweep_half_low_contact58_20260627_0310`) prove that nonzero
`candidate.controller.*` labels can be generated, but both failed the formal
hold-duration gate and therefore remain diagnostic-only. The next immediate
step is a less disruptive ordinary-cell threshold sweep, not learned-adapter
training.

The less disruptive contact58 gentle sweeps then showed that nonzero feedback
labels can preserve lift, hold, drop, contact-loss, automated visual
validation, and manual visual inspection. The repeated strict acceleration
failure was traced by peak analysis to an initial recorded settling artifact:
the non-warmup top peak occurred at step 2, phase 0, before feedback was
active. Adding `PRE_RECORD_WARMUP_STEPS=15` removes that artifact from the
recorded metric window while preserving the official Newton rollout path and
nonzero feedback labels. The first promoted source candidate is
`residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`,
with metrics status pass, `feedback_trigger_count=241`, lift height
`0.15815936028957367` m, hold duration `2.5333309173583984` s, and
`max_object_accel_m_s2=0.5063306543767194`. The next step is no longer blind
threshold sweeping; it is to build the formal residual-label source runner
around `experiments/configs/residual_label_source_manifest_v1.json`, collect
additional ordinary cells, and keep held-out `full_low` and `empty_high` for
evaluation.

The formal source runner now exists and has passed on five ordinary cells:
`half_low`, `empty_low`, `half_medium`, `full_high`, and `empty_medium`. The
final runner `residual_label_source_runner_v1_20260627_0455` reran fresh official Newton
sanity inside tmux-held allocation `154142`, produced
`data/processed/residual_label_source_runner_v1_20260627/manifest.json`, and
validated `1800` records from `5` source runs with `failures=[]`,
`generated_trex_fields=[]`, `schema_promotion=blocked`, and
`training_started=false`. This clears the source-runner blocker but still does
not create a learned adapter. Source availability should no longer be treated
as the active gate; the main next step is a reviewed learned residual-adapter
runner that consumes these sources while preserving held-out split enforcement.

The residual-adapter training preflight now also exists and passed on compute
run `residual_adapter_training_preflight_v1_20260627_0523`. It reran fresh
official Newton sanity inside allocation `154142`, consumed the five-source
runner output, and wrote
`data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
The split is train=`1440` records from `half_low`, `empty_low`, `half_medium`,
and `full_high`, validation=`360` records from `empty_medium`, with held-out
`full_low` and `empty_high` still excluded. The preflight manifest has
`failures=[]`, `generated_trex_fields=[]`, `schema_promotion=blocked`,
`training_started=false`, and `no_model_created=true`. The active next gate is
the actual residual-adapter trainer implementation and review, not source
collection or split construction.

The residual-adapter trainer smoke now also passes. The final smoke run
`residual_adapter_trainer_v1_smoke_20260627_0539` used separate local venvs:
`envs/newton/.venv` for fresh official Newton sanity and
`envs/residual_adapter/.venv` for PyTorch/CUDA. It ran on `cuda:0` / NVIDIA
H200, executed 3 optimizer steps, produced validation metrics, wrote no
checkpoint, and recorded `real_training_result=false`,
`generated_trex_fields=[]`, and `schema_promotion=blocked`. The next active
gate is a real `RUN_MODE=train` run that satisfies the one-GPU one-hour rule,
monitors GPU utilization, writes a checkpoint, and is followed by held-out
visual/metric evaluation.

The training path is:

1. Start with the official Newton Panda hydro scripted grasp/lift path as the
   non-learned infant prior.
2. Generate Newton rollouts across mass, friction, fill-level proxy, and pose
   randomization.
3. Train only a small residual adapter or controller-parameter policy at first:
   gripper closure target, lift velocity scale, hold height target, regrasp
   trigger threshold, and stabilization duration.
4. Train object/contact/tactile forward models on the same rollouts before
   using curiosity for policy adaptation.
5. Add intrinsic reward only after the forward-model diagnostics show useful
   prediction of object motion, contact, slip, and tactile/contact response.
6. Evaluate on held-out mass/friction cells before claiming adaptation.

The data unit is a synchronized episode, not an isolated image. Required
episode fields include robot joint state, end-effector pose, object pose and
velocity, contact count or contact proxy, camera RGB-D, controller phase,
controller command parameters, success/failure labels, and later real tactile
evidence under `taccel.marker.*` or other explicit source namespaces.

OpenPI, pi0/pi0-FAST, diffusion policy, ACT-style policies, and T-Rex-style
tactile architectures may be considered only as serious baselines or reference
architectures. They must enter through documented source code, documented
checkpoints, and explicit observation/action adapters. They are deferred until
the scripted Newton infant prior has produced stable baseline rollouts and a
checkpoint audit is worth the extra integration cost. The immediate Newton
mainline must not depend on pretending that a mismatched checkpoint is a solved
grasping policy.

## Curiosity Mechanism

Curiosity should be driven by physical learning progress, not raw pixel
novelty. The agent should be rewarded for actions that improve predictions of
task-relevant physical consequences:

```text
intrinsic_reward =
  learning_progress(object/contact/tactile prediction)
+ controllable_disagreement
+ bounded_useful_change
- safety_penalty
- no_op_penalty
- excessive_force_penalty
```

The prediction targets are:

- object pose delta and velocity;
- lift response under expected mass;
- contact count or contact-proxy change;
- slip or contact-loss risk;
- tactile-marker flow, active marker count, or deformation response when real
  tactile evidence is available;
- success/failure risk under the current controller parameters.

Raw prediction error alone is not sufficient because it can reward chaotic
collisions, dropped objects, or visual noise. Learning progress and bounded
useful change should be preferred: the agent should seek interactions that
make its model better while staying inside force, drop, and stability limits.

Required ablations:

- no curiosity;
- random intrinsic reward;
- object-motion-only curiosity;
- contact-only curiosity;
- tactile-only curiosity;
- vision+tactile curiosity;
- shuffled tactile;
- delayed tactile.

Current diagnostic status as of 2026-06-27: the Phase 03 contact-aware
curiosity replay evaluator has passed on the full 3x3 no-adaptation
mass/friction grid, including held-out `full_low` and `empty_high`. The output
is `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`
with `status=pass` and `rollout_count=9`. This result is useful reward-shape
evidence only. It uses diagnostic replay predictors, does not train a learned
world model, does not update a policy, and uses `newton.contact_proxy_only`
rather than real tactile-marker evidence.

Important correction: the completed residual-adapter training and held-out cup
evaluation do not complete curiosity training. Complete curiosity training
still requires a learned forward model, a real learning-progress or
controllable-disagreement signal, intrinsic-reward-driven policy/adaptation
training, and held-out evaluation against the residual adapter without
curiosity. These gates are now tracked in
`PLAN/03_curiosity_reward/plan.md` and `TODO/03_curiosity_reward/todo.md`.

After the current cup benchmark is stable, the project must move to harder
tasks instead of overfitting the easy setup. The next task progression is
tracked in `PLAN/07_harder_task_progression/plan.md` and
`TODO/07_harder_task_progression/todo.md`, starting with variable water-cup
weight/fill variants and then slippery, deformable, handled/off-center, and
fragile/safety-constrained objects.

Curiosity is considered useful only if it improves held-out mass/friction
adaptation without hiding drop, slip, or excessive-force failures.

## Vision-Tactile Fusion And Masking

Touch must be a first-class online signal, not a late feature concatenated only
for reporting. The planned model structure is:

```text
vision_encoder(rgb, depth) -> z_v
tactile_encoder(contact_proxy, marker_flow, deform) -> z_t
proprio_encoder(joint, ee, gripper, phase) -> z_p
action_encoder(controller_params) -> z_a

fusion(z_v, z_t, z_p, z_a, masks) -> z
policy_head(z) -> residual controller params
forward_model(z, action) -> next object/contact/tactile prediction
```

The tactile stream enters both the policy and the curiosity forward model. This
ensures touch can change actions online and can also drive exploration through
prediction learning progress.

Current source status as of 2026-06-27: the first tactile/contact source is a
Newton contact-proxy manifest, not real tactile F6 or dense deformation. The
manifest at
`data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
has `status=pass`, `source_run_count=10`, `record_count=3600`,
`generated_trex_fields=[]`, and `schema_promotion=blocked`. This can support
Newton-native contact-aware diagnostics and future residual-adapter input
audits, but it must not be described as T-Rex tactile evidence.

Current training-preparation status as of 2026-06-27: the residual adapter and
forward-model target contract is defined in
`docs/residual_adapter_forward_model_contract_v1.md` and
`experiments/configs/residual_adapter_forward_model_contract_v1.json`. It adds
controller-parameter residual outputs, object/contact prediction targets,
Newton contact-proxy tactile/contact input, modality masks, post-contact
pure-touch windows, held-out cells, and required ablations. Its status is
`target_contract_ready_training_not_started`; it is not a trained adapter and
not a learned world model.

Current residual-adapter training readiness as of 2026-06-27: training is
blocked, not started. The readiness audit is recorded in
`experiments/reports/2026-06-27_phase04_residual_adapter_training_readiness_v1.md`.
The original blocker was that all scripted-feedback evaluations had
`feedback_trigger_count=0`. The first ordinary-cell sensitive-feedback
diagnostic,
`residual_label_source_sensitive_feedback_half_low_20260627_030145`, now proves
the official Newton path can emit nonzero residual controller corrections
(`feedback_trigger_count=241`). It is not promoted to training data because the
threshold is too aggressive and fails hold-duration/object-acceleration
metrics. A follow-up acceleration-sensitive diagnostic preserves lift/hold/
drop/contact behavior but produces `feedback_trigger_count=0`, so scalar
threshold tuning alone did not solve it. The best current candidate is
`residual_label_sweep_half_low_contact58_gentle_20260627_0345`: it produces
nonzero residual labels and preserves lift/hold/drop/contact/visual/manual
gates, but strict metrics still fail on object acceleration. The next step is
to reduce object acceleration around that candidate on ordinary cells, not a
toy policy or no-op adapter.

Current residual-correction collection plan as of 2026-06-27: the first
diagnostic collection route is defined in
`experiments/configs/residual_correction_collection_plan_v1.json` and
`experiments/reports/2026-06-27_phase04_residual_correction_collection_plan_v1.md`.
It proposes an acceleration-sensitive ordinary-cell diagnostic to produce
nonzero feedback residual fields while preserving held-out `full_low` and
`empty_high`. This is not training and not an adaptation-improvement claim.

Current nonzero residual diagnostic status as of 2026-06-27: run
`residual_label_source_sensitive_feedback_half_low_20260627_030145` produced
nonzero residual feedback fields (`feedback_trigger_count=241`) on ordinary
`half_low`, but it is rejected as a training label source because the formal
metrics failed the hold-duration gate. The diagnostic proves residual fields
can be generated; it does not provide usable training labels yet.

Vision and touch must be balanced through training-time modality masking,
cross-modal prediction, and explicit ablations. The policy should see:

- both vision and touch;
- vision masked, touch visible;
- touch masked, vision visible;
- partial vision mask;
- partial tactile mask.

After contact, the curriculum should include pure tactile windows. This matches
the guitar-playing analogy: early approach still needs vision, but once contact
is established the agent should be able to stabilize, detect slip, and adjust
without continuously looking.

Initial masking policy:

```text
p(mask_vision | post_contact) = 0.3 -> 0.6 curriculum
p(mask_tactile | post_contact) = 0.1 -> 0.2
p(both_visible) remains nonzero
```

Pure tactile success is not enough. The required evidence is that multimodal
vision+touch outperforms vision-only and touch-only, while shuffled or delayed
tactile degrades performance. That is the test that touch is real, online, and
causally useful.

Current ablation-reporting status as of 2026-06-27: the Phase 05
contact-proxy ablation report is recorded in
`experiments/reports/2026-06-27_phase05_contact_proxy_ablation_report_v1.md`.
It summarizes existing Phase 03 replay diagnostics for object-motion-only,
contact-proxy-only, object+contact, shuffled-contact, and delayed-contact
ablations across 9 mass/friction rollouts. This is not yet proof of a trained
policy using touch; it is the current auditable baseline for later residual
adapter evaluation.

## T-Rex Bridge Criteria

Revisit strict T-Rex promotion only when a source can provide all of the
following from the same synchronized episode:

```text
observation.state [62]
action [16,62]
action_abs [62]
observation.images.head
observation.images.wrist_right
observation.images.wrist_left
observation.tactile_f6 [10,6]
observation.tactile_deform.l0..l4
observation.tactile_deform.r0..r4
```

Every field must be real, synchronized, visually inspected where applicable,
and generated from a documented embodiment/controller contract. Padding,
shape-only projection, unrelated stream composition, or marker-to-F6 renaming
is not acceptable.

## Historical Archive

Pre-pivot experiments, reports, logs, and compatibility attempts are archived
under:

```text
legacy/2026-06-26_pre_pivot_archive/
```

They remain useful evidence, especially for understanding why strict T-Rex
promotion is blocked. They are no longer the primary planning surface.

## Post-Pivot Guarded Objective Gate

The first post-pivot Newton-native curiosity objective is now a guarded
contact/camera/object-change objective, not raw RGB curiosity and not a learned
world model. The relevant design artifact is archived under:

```text
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/curiosity_v4_newton_objective_spec_20260626/
```

The objective suppresses simulator-settling artifacts by assigning
`actionable_score=0` to transitions with `sample_step_a < 10`. This moves the
selected segment away from the raw high-scoring `cube_dense_1->2` transient and
selects `guarded_cube_dense_173->174`.

A compute-node target-window visual gate was run around steps `169..178` in
the existing tmux-held allocation `154023`. It reran fresh official Newton
sensor-contact sanity, exported 10 SensorTiledCamera frames, passed visual
validation, and passed manual inspection of the contact sheet plus frames
`0173`, `0174`, and `0178`.

Post-pivot evidence final location after automatic archive:

```text
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/newton_panda_hydro_camera_cube_objective_v4_guarded_169_178_20260626_0005_manual_visual_inspection.json
legacy/2026-06-26_pre_pivot_archive/experiments/outputs/newton_panda_hydro_camera_cube_objective_v4_guarded_169_178_20260626_0005_downstream_gate_cleared.json
legacy/2026-06-26_pre_pivot_archive/experiments/reports/2026-06-27_objective_v4_target_window_visual_gate.md
```

This is source prioritization and visual gating only. It does not claim learned
ICM, world-model training, policy success, calibrated F6, T-Rex schema
promotion, or physical Newton/Taccel synchronization.

## Phase 01 Concrete Task: Variable-Mass Cup Lift-And-Hold

The current concrete Newton task is a variable-mass cup lift-and-hold
adaptation benchmark. This is the first executable task family for the
Newton-native curiosity line.

The task is defined in:

```text
docs/lift_hold_variable_mass_cup_task_spec.md
experiments/configs/lift_hold_variable_mass_cup_task_v1.json
experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py
experiments/reports/2026-06-27_lift_hold_variable_mass_cup_task_spec.md
TODO/01_newton_task_definition/todo.md
```

The task uses the official Newton Panda hydro example as the first scene
entry point:

```text
external/newton/newton/examples/robot/example_robot_panda_hydro.py
```

The local official cup asset selected for the next gate is:

```text
external/newton-assets-cache/newton-assets_manipulation_objects_cup_f7f64ec3_8e8df07d/manipulation_objects/cup/model.usda
```

The experiment grid varies fill/mass proxy, friction, and initial pose. The
first mass levels are empty, half, and full cup proxies. The first friction
levels are low, medium, and high. Held-out combinations include a full
low-friction cup and an empty high-friction cup. This is meant to test whether
the policy can adapt to wrong expectations about object response, not merely
memorize a single cup.

Allowed observations remain Newton-native and explicitly namespaced:

```text
newton.panda.*
newton.object.*
newton.contact.*
newton.camera.*
candidate.controller.*
```

No T-Rex fields are promoted in this phase. The validator requires
`generated_trex_fields=[]`, `schema_promotion=blocked`, and
`no_model_or_training=true`.

## Phase 01 First Visual Gate Result

The first Phase 01 official visual gate has passed. It reused the existing
tmux-held allocation `154023`, reran fresh official Newton `sensor_contact`
sanity on the compute node, exported 9 SensorTiledCamera frames, and passed
manual visual inspection.

Run tag:

```text
lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021
```

Key evidence paths:

```text
logs/newton/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021.log
logs/newton/phase01_lift_hold_task_validation_20260627_0021.log
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_summary.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_visual_validation.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_manual_visual_inspection.json
experiments/outputs/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021_downstream_gate_cleared.json
experiments/visuals/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021/frame_browser.html
experiments/visuals/lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021/contact_sheet.png
```

The inspected frames showed nonblank head, right-wrist, and left-wrist camera
panels with the Panda robot, table, official grasped cube, and cup placement
context visible. This clears only the official Panda hydro scene, camera
export, validation, and reporting path. It does not claim cup grasp success,
learned curiosity, policy adaptation, T-Rex compatibility, or tactile F6.

## Immediate Next Step

Do not wait for strict T-Rex schema compatibility before moving. The next
completed step adapted the grasped object path from the official Panda hydro
example to the local official cup asset, reran fresh official Newton sanity,
exported camera frames, manually inspected the result, and recorded the cup
asset gate result.

Cup asset run tag:

```text
lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105
```

The cup-asset gate passed for retarget and visual evidence:

```text
tracked_object=existing_cup_asset
adapter=retarget_existing_official_cup_asset_as_object
generated_trex_fields=[]
schema_promotion=blocked
no_model_or_training=true
```

Manual inspection confirmed that the cup is visible and tracked. The numeric
summary reports `max_lift=0.15901388227939606`, but the final inspected frame
shows the cup tilted/fallen. Therefore this is not a stable cup grasp success
yet.

If cup geometry, collision, mass, or grasp initialization needs adjustment,
record it as a Newton cup-asset adaptation issue and keep iterating inside the
Phase 01 task definition. Do not substitute a toy model or pretend the result
is T-Rex-style data.

The next concrete step was to distinguish a real cup-grasp issue from the
official controller's short release cycle. A second gate,
`lift_hold_variable_mass_cup_v1_existing_cup_hold_gate_20260627_0145`, added
`final_hold_duration=999.0`. It passed official sanity and visual validation,
and manual inspection showed the cup still elevated at frame `0239`.

Numeric result:

```text
final_object_z=0.30836987495422363
max_lift=0.15986861288547516
```

This clears the extended-hold visual gate, but it still does not claim the full
two-second hold metric because the 240-frame diagnostic window covers only
about one second after reaching the high hold pose. The next step is a longer
metric gate, for example 360 frames or more, with explicit success/failure
metric extraction.

That longer metric gate has now passed:

```text
run_tag=lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210
num_steps=420
success_all_worlds=true
longest_hold_s=4.1
max_lift=0.16001205146312714
drop_from_max=0.0
failure_reasons=[]
```

Manual inspection of frames `0240`, `0360`, and `0419` confirmed that the cup
remains elevated through the sampled hold window. This clears the Phase 01
scripted cup lift-and-hold metric gate, but it still does not claim learned
curiosity, policy adaptation, T-Rex compatibility, tactile F6, or training.

## Phase 04 Real Residual Adapter Training Status

The first Newton-native residual controller-parameter adapter has now been
trained. This is deliberately not an official T-Rex method and not a T-Rex
schema result. It is a residual controller adapter around the official Newton
Panda hydro scripted infant prior, using the compute-verified residual-label
split.

Real training run:

```text
run_tag=residual_adapter_trainer_v1_train_20260627_0548
checkpoint=checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt
summary=experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_summary.json
report=experiments/reports/2026-06-27_phase04_residual_adapter_training_v1.md
```

The run reused the existing tmux-held Curiosity GPU allocation `154142`,
passed fresh official Newton sanity, trained for `3600.0302035808563` seconds
on an NVIDIA H200, wrote a checkpoint, completed `32685` optimizer steps, and
passed GPU utilization monitoring with mean utilization
`99.08333333333333%`. Validation loss was
`6.241170922294259e-05` on ordinary validation cell `empty_medium`.

This completes the training gate but not the science claim. The checkpoint has
not yet been connected back into the Newton closed-loop controller, has not
produced trained-policy visual/browser rollouts, and has not been evaluated on
held-out `full_low` or `empty_high`. Therefore the next step is to evaluate
the trained checkpoint, not to keep waiting for exact T-Rex data-schema
alignment.

## 2026-07-01 Reference-Video Tactile Reset Milestone

The active direction is now Phase 00 reference-video-aligned dense tactile
environment work, not the legacy contact-count curiosity path.

Major current milestone:

```text
newton_main_commit=a217e55fab3d373a08fba374cc5cafc1826cf27f
benchmark_run=p00_bench_main_20260701_035529
benchmark_fps=92.6
meets_82_fps_target=true
main_tactile_run=p00_main_f6_v1_20260701_035926
lift_success=true
hold_frames_above_lift_threshold=71
max_object_lift_m=0.22340291738510132
grid_and_f6_tactile_export=true
calibrated_view_run=p00_calib_view_v1_20260701_040715
raw_fn_nonzero_cell_ratio=0.03515625
calibrated_fn_nonzero_cell_ratio=0.236328125
```

Interpretation: latest official Newton main now meets the user's 82 FPS base
runtime target and remains compatible with the current steel-spec grid/F6
tactile diagnostic exporter. The calibrated-view run improves tactile panel
readability while preserving raw maps. This is real Phase 00 environment/base
progress, not curiosity-training success.

Still open before curiosity training may restart:

- direct hydro `Ft` and direct pad-resolved shear force:
  `p00_mjw_force_audit_v2_20260701_045700` proves bottom-level MJWarp EFC
  normal/tangent force arrays are readable and include pad-object force during
  the 240-frame official Panda hydro grasp/lift sequence, but this is still a
  candidate direct-force path. `p00_mjw_direct_v1_20260701_052900` now maps
  that candidate MJWarp force into synchronized left/right pad `Fn`/`Ft`
  heatmaps, shear arrows, and real Newton `SensorTiledCamera` scene video; it
  passed the official 240-frame Panda hydro final test and produced nonblank
  visual evidence. `p00_mjw_align_v1_20260701_055200` validates the same
  candidate EFC frame mapping against official `SensorContact` on a compatible
  MuJoCo-contact scene with force relative RMSE `3.2491620810680347e-08`,
  friction relative RMSE `2.0018143688320552e-07`, and mean cosine `1.0` for
  both. `p00_mjw_direct_steel_v1_20260701_060500` then applies the validated
  sign convention in the active hydro diagnostic with steel-spec material
  override (`mu=0.3`, `kh=1e12`), `material_notify_status=pass`, official
  Panda hydro final test pass, synchronized real scene views, and candidate
  direct `Fn`/`Ft` tactile maps. `p00_refcmp_v3_20260701_065300` now compares
  it directly against the reference MP4 and confirms it is nonblank and
  materially closer, but still below the reference video's gel/marker tactile
  density and channel richness;
- reference-video-level dense tactile richness;
- validated gel/marker photometric semantics and deformation-marker tracking
  beyond the current candidate rendering;
- validated real contact-area semantics beyond the current point-contact-density
  proxy;
- validated channel-level semantic equivalence beyond the current layout audit;
- USD/photoreal visual scene fusion with tactile panels;
- tactile-mask/vision-mask training and closed-loop curiosity evaluation.
