# Phase 00 Dense Tactile Infant Plan

Status: active after the 2026-07-01 reset.

Legacy archive:

- `PLAN/legacy_20260701_pre_dense_tactile_reset/00_ref_tactile_env/`
- `PLAN/legacy_20260701_pre_dense_tactile_reset/01_newton_only_curiosity/`
- matching task records under
  `TODO/legacy_20260701_pre_dense_tactile_reset/`

## Active Position

The old Newton-native contact-count curiosity pipeline is legacy negative
evidence and engineering-chain review only. It used low-dimensional rollout
state, a GRU forward model, learning-progress scoring, and supervised residual
fine-tuning. It was not closed-loop curiosity because intrinsic reward did not
drive online exploration or create a new data distribution.

The old tactile fields were scalar contact proxies:

- `newton.panda.rigid_contact_count`
- `candidate.modality.contact_available_mask`

They are not tactile maps. They do not encode where contact happens, pressure
magnitude, shear, left/right imbalance, imminent slip, contact area,
penetration/compression, marker flow, or tactile images. Five old real one-hour
curiosity candidates failed strongest-baseline comparison
(`positive_curiosity_result=false`, `safety_regression_cell_count=4`), so no
sixth old-style run is allowed unless the user explicitly resets that gate.

## Current Target

Build a reference-video-aligned dense tactile infant:

1. A base controller/model completes grasp, lift, and hold.
2. The same rollout exports synchronized dense visual and tactile mechanics.
3. Dense tactile evidence is available before curiosity is restarted.
4. Later curiosity becomes closed-loop active probing over dense
   visuo-tactile prediction.

Required base evidence:

- visual scene;
- left/right tactile pad maps;
- pressure and compression heatmaps;
- normal force `Fn`;
- tangential/shear force `Ft`;
- shear direction;
- contact area and center of pressure;
- penetration/compression;
- material, friction, and stiffness statistics;
- grip, shear, contact, and safety time-series.

## Dense Representation Contract

The active target representation is pad-resolved:

- `left_pad.pressure: [T, H, W]`
- `left_pad.compression: [T, H, W]`
- `left_pad.shear_u/v: [T, H, W]`
- `left_pad.contact_mask: [T, H, W]`
- `left_pad.Fn/Ft: [T]` or `[T, H, W]`
- equivalent `right_pad.*` fields.

Candidate Newton/MJWarp fields must preserve provenance:

- `candidate.newton_mjw.Fn`
- `candidate.newton_mjw.Ft`
- `candidate.newton_mjw.area_proxy`
- `candidate.newton_mjw.marker_flow`
- `candidate.newton_mjw.contact_normal`

Do not promote these candidate proxy fields into official tactile semantics.
`area_proxy != real contact area`, `marker_flow render != photometric GelSight
marker output`, `contact_count != tactile map`, and `candidate Fn/Ft !=
validated official tactile force field`.

## Gate 00F Priority

Gate 00F is low-priority final semantic validation/comparison-gap work.
UniVTAC, TaCauchy, and IsaacLab TacSL remain useful final references, but
current Phase 00 must not get stuck there. The priority is to make Newton
dense tactile/base evidence real, provenance-preserving, and visually
inspectable.

## Execution Plan

1. Archive and reset active records.
   Confirm the old active `PLAN/TODO` entries are under
   `legacy_20260701_pre_dense_tactile_reset/`, and active records point only to
   this phase.
2. Build or select the tactile-rich base environment.
   Use current serious Newton/Taccel paths and approved compute workflow. No
   login-node simulation, rendering, validation, dataset conversion, or
   training.
3. Produce base grasp/lift/hold evidence with dense mechanics.
   Evidence must include visual scene, dense pad maps, `Fn`, `Ft`, shear,
   contact area/center, penetration/compression, material stats, and time
   series.
4. Write the dense data schema and manifest.
   Keep source provenance explicit. Candidate proxy namespaces remain
   candidate namespaces.
5. Define closed-loop curiosity only after dense tactile/base evidence exists.
   The future model must predict tactile/contact/mechanics, and the policy
   must support active probing, regrasping, grip-force adjustment,
   pressure-balancing, and shear-minimizing actions.
6. Define stronger baselines and harder held-out tests.
   If base grasp/lift/hold is too easy, increase task difficulty or use finer
   metrics before any curiosity claim.
7. Run Phase01 dense closed-loop training preflight.
   Validate that the dense tactile/base evidence, closed-loop curiosity design,
   baseline set, modality-mask requirements, and success/non-claim boundaries
   are all ready before any real training attempt. This preflight is not
   training and not curiosity success.
8. Start real closed-loop dense curiosity training only after the preflight
   passes.
   Every counted attempt must be at least one hour in a Curiosity-owned
   tmux-held Slurm allocation, tracked in the attempt ledger, and evaluated
   against the strongest baseline and safety metrics.

## Success Conditions

Do not claim success until harder held-out tasks beat the strongest baseline
without safety regression. Required metrics include lift, hold duration, slip,
drop, contact loss, object acceleration, force/contact cost, and safety
regression.

Video evidence format rule:

- Newly generated rollout/visualization videos must be MP4 only.
- Do not generate AVI as active evidence. If an upstream tool emits AVI, convert
  or replace it with MP4 before recording the artifact.

Current Phase01 preflight evidence:

- failed diagnostic: `p01_dense_preflight_20260701_2051`;
- passing preflight: `p01_dense_preflight_20260701_2052`;
- summary:
  `experiments/outputs/phase01/dense/preflight/p01_dense_preflight_20260701_2052/dense_training_preflight_summary.json`;
- report:
  `experiments/reports/phase01/dense/preflight/p01_dense_preflight_20260701_2052/dense_training_preflight.md`;
- status: `pass_preflight_training_contract_ready`;
- boundary: not training, not a checkpoint, and not curiosity success.

Current Phase01 closed-loop probe smoke evidence:

- short interface smoke: `p01_dense_clprobe_smoke_20260701_2100`;
- contact/lift smoke: `p01_dense_clprobe_smoke240_20260701_2101`;
- summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/smoke/p01_dense_clprobe_smoke240_20260701_2101/dense_closed_loop_probe_summary.json`;
- report:
  `experiments/reports/phase01/dense/closed_loop_probe/smoke/p01_dense_clprobe_smoke240_20260701_2101/dense_closed_loop_probe.md`;
- status: `pass_smoke_closed_loop_dense_probe`;
- evidence: closed-loop action changed, dense candidate tactile/mechanics
  features were used, max lift reached about `0.2225 m`, and hold frames were
  nonzero;
- boundary: smoke only, not a real one-hour training attempt, not baseline
  comparison, and not curiosity success.

Current Phase01 eval smoke evidence:

- eval smoke: `p01_dense_clprobe_eval_smoke_20260701_2106`;
- summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/eval_smoke/p01_dense_clprobe_eval_smoke_20260701_2106/dense_closed_loop_eval_summary.json`;
- report:
  `experiments/reports/phase01/dense/closed_loop_probe/eval_smoke/p01_dense_clprobe_eval_smoke_20260701_2106/dense_closed_loop_eval.md`;
- status: `pass_eval_smoke_metrics_ready`;
- finding: on the easy cube smoke, `base_zero_action` already lifts and holds;
  checkpoint delta was tiny (`delta_max_lift` about `0.00034 m`,
  `delta_hold_frames=0`, no safety regression). This is evaluation plumbing,
  not evidence of curiosity success.
- next implication: the first real one-hour attempt must use harder held-out
  cells or finer force/shear/safety metrics so the strongest baseline leaves
  measurable room for improvement.

Current harder-cell probe evidence:

- saturated probe: `p01_dense_hardcell_pen_mu015_eval_20260701_2109`;
  `base_zero_action` still reached about `0.2152 m` lift and `85` hold frames,
  so `pen + mu=0.15` is not hard enough for a first success claim.
- useful harder-cell candidate:
  `p01_dense_hardcell_pen_mu002_eval_20260701_2111`; `base_zero_action`
  reached only about `0.0268 m` lift and `0` hold frames, while the smoke
  checkpoint reached about `0.0381 m` lift and `0` hold frames without safety
  regression.
- implication: `scene=pen`, `override_mu=0.02`, `override_kh=1e12`,
  `num_frames>=240` is the current candidate for the first real dense
  closed-loop training attempt because the strongest baseline has measurable
  room to improve.
- boundary: this is task selection/eval smoke only, not a one-hour training
  result and not curiosity success.

Current real-attempt readiness evidence:

- early failed launch:
  `p01_dense_clprobe_attempt001_pen_mu002_20260701_2117`; this failed before
  the one-hour gate with `RuntimeError: Failed to allocate 284 bytes on device
  'cuda:0'`, produced no valid training summary/checkpoint, and is invalid/not
  counted.
- cancelled diagnostic:
  `p01_dense_clprobe_diag_memfix_pen_mu002_20260701_2121`; cancelled before
  summary/eval and not counted.
- memory-fix diagnostic:
  `p01_dense_clprobe_diag_memfix2_pen_mu002_20260701_2130`; completed about
  `613.6 s`, `80` episodes, and `40` generations without repeating the CUDA
  allocation failure.
- diagnostic eval finding: base lift was about `0.0268 m`, checkpoint lift was
  about `0.0490 m`, `delta_hold_frames=0`, and no safety regression.
- boundary: this is still below one hour and has no hold improvement, so it is
  not a real training attempt and not curiosity success.
- implication: restart the first counted real attempt on
  `scene=pen`, `override_mu=0.02`, `override_kh=1e12`, `num_frames=240`,
  `target_duration_s>=3600`, with GPU-utilization evidence and automatic
  validation.

First counted real attempt result:

- counted attempt:
  `p01_dense_clprobe_attempt001_pen_mu002_retry1_20260701_2140`;
- runtime: about `3617.6 s`, `376` episodes, `188` generations, inside
  Curiosity Slurm allocation `161009` on `server51`;
- checkpoint:
  `checkpoints/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt001_pen_mu002_retry1_20260701_2140/dense_closed_loop_probe_checkpoint.npz`;
- training summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt001_pen_mu002_retry1_20260701_2140/dense_closed_loop_probe_summary.json`;
- eval summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt001_pen_mu002_retry1_20260701_2140_eval/dense_closed_loop_eval_summary.json`;
- result: negative counted attempt 1/5. The checkpoint improved max lift only
  slightly (`base_mean_max_lift` about `0.0289 m`, checkpoint about
  `0.0337 m`, delta about `0.0048 m`) but `delta_hold_frames=0` and
  `success_claim_allowed=false`. The eval checkpoint also had high
  `drop_after_lift` about `0.1345`, so the slight lift increase is not stable
  manipulation progress.
- safety: no safety regression in this eval (`base_mean_safety_cost=0`,
  `checkpoint_mean_safety_cost=0`).
- implication: do not repeat the same one-hour run unchanged. Repair the
  objective/action/search/curriculum so attempt 2 prioritizes lift-to-hold
  stability and drop-after-lift reduction, not raw lift alone.

Post-attempt-1 repair evidence:

- code repair: closed-loop `lift_z_delta`, `lateral_y_delta`, and
  `probe_y_delta` now shift the IK waypoint target during the rollout instead
  of being logged but not injected. CEM also keeps a sigma floor instead of
  collapsing after a single elite.
- hard-cell check:
  `p01_dense_clprobe_repair_smoke_pen_mu002b_20260701_2258` on
  `pen + mu=0.02` still failed to produce hold (`best_max_lift` about
  `0.0140 m`, `best_hold_frames=0`), so that cell is too hard for attempt 2.
- repair candidate:
  `p01_dense_clprobe_repair_smoke_pen_mu005_20260701_2301` on
  `pen + mu=0.05` reached `best_max_lift` about `0.2051 m`,
  `best_hold_frames=97`, and zero safety cost in a short diagnostic.
- eval smoke:
  `p01_dense_clprobe_repair_eval_pen_mu005_20260701_2304` compared base and
  checkpoint on the same `pen + mu=0.05` cell. Base reached about `0.0698 m`
  lift and `0` hold frames; checkpoint reached about `0.2051 m` lift and `82`
  hold frames, with no safety regression.
- boundary: these are short diagnostics, not one-hour training and not
  curiosity success.
- implication: attempt 2 should use `scene=pen`, `override_mu=0.05`,
  `override_kh=1e12`, `num_frames=240`, `population_size=4`, `elite_count=2`,
  stronger drop penalty, and sigma floor. It must still run at least one hour
  and pass the normal strongest-baseline/safety gate before any success claim.

Second counted real attempt result:

- counted attempt:
  `p01_dense_clprobe_attempt002_pen_mu005_20260701_2310`;
- runtime: about `3624.0 s`, `372` episodes, `93` generations, inside
  Curiosity Slurm allocation `161009` on `server51`;
- checkpoint:
  `checkpoints/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt002_pen_mu005_20260701_2310/dense_closed_loop_probe_checkpoint.npz`;
- training summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt002_pen_mu005_20260701_2310/dense_closed_loop_probe_summary.json`;
- eval summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt002_pen_mu005_20260701_2310_eval/dense_closed_loop_eval_summary.json`;
- result: positive counted attempt. Base reached about `0.0700 m` lift and
  `0` hold frames; checkpoint reached about `0.2164 m` lift and `84` hold
  frames, with no safety regression.
- boundary: this is not final curiosity success yet. It still needs replicated
  evaluation, stronger/held-out baseline comparison, ablations, and MP4 visual
  rollout evidence.
- next step: generate MP4-only rollout evidence and run replicated validation
  before making any stronger claim.

Attempt 2 replicated validation:

- replicated eval:
  `p01_dense_clprobe_attempt002_repeval5_pen_mu005_20260702_0005`;
- repetitions: `5`;
- result: base mean lift about `0.0702 m`, base mean hold `0`; checkpoint mean
  lift about `0.1934 m`, checkpoint mean hold `69`; no safety regression.
- boundary: this strengthens the positive attempt-2 evidence but still does
  not satisfy final success because MP4 rollout evidence, ablations, and
  held-out/stronger baseline comparisons are still pending.

Attempt 2 MP4 rollout evidence:

- MP4 video:
  `experiments/visuals/phase01/dense/closed_loop_probe/attempt002_mu005/base_vs_checkpoint_rollout.mp4`;
- summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/mp4/attempt002_mu005/dense_rollout_mp4_summary.json`;
- decode validation:
  `experiments/outputs/phase01/dense/closed_loop_probe/mp4/attempt002_mu005/mp4_decode_validation.json`;
- result: MP4 decode passed with `240` frames, `30 FPS`, nonblank sampled
  frames, and `avi_generated=false`.
- rollout metrics in the MP4 summary: base max lift about `0.0699 m`, base
  hold `0`; checkpoint max lift about `0.2166 m`, checkpoint hold `84`.
- boundary: this satisfies immediate MP4 rollout evidence for the positive
  attempt, but final success still requires held-out/stronger baseline
  comparison and ablations.

Attempt 2 held-out/ablation evidence:

- held-out/ablation eval:
  `p01_dense_clprobe_heldout_ablation_attempt002_mu005_fix1_20260702_0029`;
- summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/heldout_ablation/p01_dense_clprobe_heldout_ablation_attempt002_mu005_fix1_20260702_0029/dense_heldout_ablation_summary.json`;
- result: mixed/negative for final claim. Attempt 2 remains strong on
  train-like `mu=0.05` and held-out `mu=0.06`, but lower-friction transfer is
  weak: `mu=0.04` has only about `+0.0062 m` lift delta and `0` hold delta,
  and `mu=0.03` is worse than base by about `-0.0055 m`.
- ablation finding: noisy tactile strongly damages performance, which is a
  useful sign that the current policy depends on tactile/mechanics features.
  However `vision_only_proxy` is not true camera vision, and this eval does not
  replace mandatory tactile-mask training inside policy optimization.
- boundary: this is not training and not final success. It blocks any final
  success claim and motivates a low-friction policy/action/search repair.

Third counted real attempt result:

- counted attempt:
  `p01_dense_clprobe_attempt003_curriculum_mu035045055_20260702_0045`;
- runtime: about `3649.3 s`, `384` episodes, `32` completed generations,
  inside Curiosity Slurm allocation `161068` on `server56`;
- training curriculum: `train_mu_values=[0.035, 0.045, 0.055]`, validation
  cell `mu=0.04`;
- checkpoint:
  `checkpoints/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt003_curriculum_mu035045055_20260702_0045/dense_closed_loop_probe_checkpoint.npz`;
- training summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt003_curriculum_mu035045055_20260702_0045/dense_closed_loop_probe_summary.json`;
- eval summary:
  `experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt003_curriculum_mu035045055_20260702_0045_eval/dense_closed_loop_eval_summary.json`;
- result: negative counted attempt 2/5. Validation at `mu=0.04` improved max
  lift only from about `0.0599 m` to `0.0661 m`, with `delta_hold_frames=0`
  and no safety regression. It does not satisfy the stronger-baseline held-out
  condition.
- implication: do not repeat this same 8-parameter low-friction curriculum
  unchanged. The next step should repair the policy/action space or objective
  so low-friction cells can produce stable hold instead of isolated single-cell
  lift.

Tail-hold objective repair evidence:

- diagnostic:
  `p01_dense_clprobe_diag_tailhold_curriculum_20260702_0150`;
- finding: the previous objective allowed "lift then drop" candidates to look
  good. Adding final-lift and last-60-frame tail-hold terms produced a short
  diagnostic with about `0.197 m` final lift and `60/60` tail-hold frames on
  train cells, but validation at exact `mu=0.04` remained weak. This is a
  useful repair signal, not success.
- invalid interrupted run:
  `p01_dense_clprobe_attempt004_tailhold_mu035040045055_20260702_0202` was
  stopped before one hour because predictor overflow caused a NaN aggregate
  score. It is invalid/not counted.
- repair applied: predictor update and score aggregation now have finite
  guards, and clean stable-objective runs should set `INTRINSIC_WEIGHT=0` so
  CEM is driven by final-lift/tail-hold/drop metrics rather than unstable
  intrinsic predictor overflow.
- next target: clean counted Attempt4 with
  `train_mu_values=[0.035, 0.04, 0.045, 0.055]`,
  `score_final_lift_weight=8.0`, `score_tail_hold_weight=0.08`,
  `score_drop_weight=12.0`, `stable_tail_frames=60`, and
  `INTRINSIC_WEIGHT=0`.

Valid reporting language:

- old contact-count curiosity pipeline = legacy negative evidence;
- current target = reference-video-aligned dense tactile environment plus base
  grasp/lift/hold;
- success claim condition = harder held-out tasks beat strongest baseline
  without safety regression.
