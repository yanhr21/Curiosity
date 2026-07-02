# Phase 00 Dense Tactile Infant TODO

Status: active after the 2026-07-01 reset.

## Active Rules

- [x] Mark previous active plan/todo directories as legacy.
      - Archived plan paths:
        `PLAN/legacy_20260701_pre_dense_tactile_reset/00_ref_tactile_env/`
        and
        `PLAN/legacy_20260701_pre_dense_tactile_reset/01_newton_only_curiosity/`.
      - Archived todo paths:
        `TODO/legacy_20260701_pre_dense_tactile_reset/00_ref_tactile_env/`
        and
        `TODO/legacy_20260701_pre_dense_tactile_reset/01_newton_only_curiosity/`.
- [x] Create new active plan/todo directories:
      `PLAN/00_dense_tactile_infant/` and
      `TODO/00_dense_tactile_infant/`.
- [x] Record that the old Newton-native contact-count curiosity pipeline is
      legacy negative evidence only.
- [x] Record that `rigid_contact_count` and
      `contact_available_mask` are scalar contact proxies, not tactile maps.
- [x] Record the stop rule: do not run a sixth old-style one-hour
      contact-count residual curiosity training attempt unless the user
      explicitly resets that stop gate.

## Immediate Documentation Tasks

- [x] Update `IDEA/idea.md` with the dense tactile infant reset.
- [x] Update `AGENTS.md` with the same active rule and forbidden shortcuts.
- [x] Update `PLAN/README.md` and `TODO/README.md` so the only active entry is
      `00_dense_tactile_infant/`.
- [x] Record the reset report and machine-readable reset config.
- [x] Run lightweight text/JSON validation only.
      - `jq empty` passed for
        `experiments/configs/phase00/dense_tactile_infant/reset_20260701_v1.json`.
      - `git diff --check` passed for the edited reset files.
      - `PLAN/README.md` and `TODO/README.md` now list only
        `00_dense_tactile_infant/` as active.
- [x] Clean important memory files so old experiment narratives cannot mislead
      future full training.
      - Rebuilt `IDEA/idea.md` as an active-only dense tactile infant idea.
      - Rebuilt `AGENTS.md` as active-only execution rules.
      - Archived the prior full idea text at
        `IDEA/legacy/legacy_idea_before_dense_tactile_cleanup_20260701.md`.
      - Split the old contact-count route into
        `IDEA/legacy/contact_count_curiosity_negative_evidence_20260701.md`.

## Dense Tactile/Base Evidence Tasks

- [x] Select the current serious Newton/Taccel simulator path without using
      scalar contact count as tactile.
      - Selected active path: Newton `external/newton_8c501` at
        `8c501b47847569fecdda97a9f7f01205c6f7964f`.
      - Record:
        `experiments/configs/phase00/dense_tactile_infant/simulator_path_selection_20260701_v1.json`.
      - Report:
        `experiments/reports/phase00/dense_tactile_infant/simulator_path_selection_20260701.md`.
      - Boundary: this is candidate dense tactile/base evidence path
        selection, not official tactile semantic validation and not curiosity
        training.
- [x] In a Curiosity-owned tmux-held Slurm allocation, produce or rerun the
      base grasp/lift/hold evidence. Do not run simulation/rendering/training
      on the login node.
      - Active run:
        `p00_dense_8c501_base_cop_20260701_2030` in Slurm job `160989` on
        `server64`.
      - Status: `pass_candidate_direct_force_export`,
        official final test `pass`, `240` frames, `146` pad-object contact
        frames, max lift `0.22242838144302368 m`.
      - Evidence manifest:
        `experiments/configs/phase00/dense_tactile_infant/active_evidence_manifest_20260701_v1.json`.
- [x] Export synchronized visual scene evidence.
      - Pre-MP4-rule legacy video from the original Phase00 base run:
        `experiments/visuals/phase00/dense_tactile_infant/newton_8c501/base/p00_dense_8c501_base_cop_20260701_2030/candidate_mjw_direct_tactile.avi`.
      - This path is historical evidence only. It must not be used as the
        format precedent for new active visualizations; all new videos must be
        MP4.
      - Reference comparison:
        `p00_dense_refcmp_8c501_base_cop_20260701_2032`.
- [x] Export left/right tactile pad fields.
      - Candidate left/right pad maps are present in the active NPZ and visual
        sheet under provenance-preserving candidate fields.
- [x] Export pressure and compression heatmaps.
      - Candidate `Fn`/`Ft` force heatmaps are present, but true compression
        heatmaps remain missing or unvalidated.
      - Active hydro compression/penetration runner is prepared:
        `experiments/configs/phase00/dense_tactile_infant/run_hydro_compression_in_alloc.sh`.
      - Active hydro run:
        `p00_dense_hydro_compress_8c501_20260701_2040`, status `pass`.
      - Boundary: compression is Newton hydro deform/penetration proxy, not
        final official tactile semantic validation.
- [x] Export `Fn` normal force.
      - Max pad-object candidate `Fn` sum:
        `41.91702651977539`.
- [x] Export `Ft` tangential/shear force.
      - Max pad-object candidate `Ft` sum:
        `12.221211433410645`.
- [x] Export shear direction.
      - Candidate shear/marker-flow and normal maps are present in the active
        NPZ and visible in the contact sheet.
- [x] Export contact area and center-of-pressure evidence.
      - Candidate area proxy and candidate force-map center-of-pressure proxy
        are present; real contact area and validated hardware/canonical CoP
        remain missing or unvalidated.
- [x] Export penetration/compression.
      - Hydro NPZ contains `max_penetration.npy` plus left/right deform proxy
        maps.
      - Active manifest records this as proxy evidence, not final official
        tactile semantic validation.
- [x] Export material, friction, and stiffness statistics.
      - Steel candidate material update passed with requested `mu=0.3` and
        `kh=1e12`.
- [x] Export grip, shear, contact, and safety time-series.
      - Object/contact/force time-series are present in the active NPZ and
        report; safety remains candidate/base-evidence level, not training
        evaluation.
- [x] Keep visual outputs grouped under short phase directories such as
      `experiments/visuals/phase00/dense_tactile_infant/<family>/<variant>/`.
      - Active outputs now use
        `experiments/visuals/phase00/dense_tactile_infant/newton_8c501/...`.

## Dense Representation/Provenance Tasks

- [x] Define and validate the dense pad schema:
      `left_pad.pressure [T,H,W]`,
      `left_pad.compression [T,H,W]`,
      `left_pad.shear_u/v [T,H,W]`,
      `left_pad.contact_mask [T,H,W]`,
      `left_pad.Fn/Ft [T]` or `[T,H,W]`, and the same `right_pad.*` fields.
      - Active schema:
        `experiments/configs/phase00/dense_tactile_infant/dense_pad_schema_v1.json`.
      - Validation: `jq empty` passed on 2026-07-01.
- [x] Store candidate Newton/MJWarp fields under provenance-preserving keys:
      `candidate.newton_mjw.Fn`,
      `candidate.newton_mjw.Ft`,
      `candidate.newton_mjw.area_proxy`,
      `candidate.newton_mjw.marker_flow`, and
      `candidate.newton_mjw.contact_normal`.
      - Active manifest records these as candidate/proxy fields and preserves
        the non-claim boundary.
- [x] Add explicit schema checks that prevent proxy promotion:
      `area_proxy != real contact area`,
      `marker_flow render != photometric GelSight marker output`,
      `contact_count != tactile map`, and
      `candidate Fn/Ft != validated official tactile force field`.
      - Active schema:
        `experiments/configs/phase00/dense_tactile_infant/dense_pad_schema_v1.json`.
      - Active evidence manifest keeps `direct_tactile_claim_allowed=false`.

## Closed-Loop Curiosity Restart Tasks

- [x] Do not restart curiosity until dense tactile/base evidence is available.
      - Dense tactile/base candidate evidence now exists in
        `experiments/configs/phase00/dense_tactile_infant/active_evidence_manifest_20260701_v1.json`.
      - Boundary: this clears design precondition only, not curiosity success.
- [x] Design the future forward model to predict tactile/contact/mechanics, not
      only object height or contact count.
      - Design:
        `experiments/configs/phase00/dense_tactile_infant/closed_loop_curiosity_design_v1.json`.
- [x] Design the policy action space to include active probing, regrasping,
      grip-force adjustment, pressure balancing, and shear-minimizing probing.
      - Design includes grip-force, width, lift, hold, wrist micro-probe,
        pressure-balance, regrasp, and shear-minimizing residual actions.
- [x] Ensure intrinsic reward affects online rollout exploration and policy
      optimization. Sample reweighting alone is not closed-loop curiosity.
      - Design requires intrinsic reward to enter policy optimization and
        marks sample reweighting alone as invalid.
- [x] Include tactile-mask training before any success claim:
      vision+tactile, tactile-only masked vision, vision-only ablation, and
      noisy or mismatched tactile ablation.
      - Contract includes vision+tactile, tactile-only, vision-only,
        noisy/delayed/shuffled/mismatched tactile.
- [x] Run Phase01 dense closed-loop training preflight in a Curiosity-owned
      tmux-held Slurm allocation.
      - Failed diagnostic:
        `p01_dense_preflight_20260701_2051`, job `160998` on `server64`;
        failure was a preflight-script naming check mismatch
        (`vision_only` in design vs. `vision_only_ablation` expected by the
        script), not a dense evidence failure and not training.
      - Passing preflight:
        `p01_dense_preflight_20260701_2052`, same allocation job `160998` on
        `server64`, status `pass_preflight_training_contract_ready`.
      - Summary:
        `experiments/outputs/phase01/dense/preflight/p01_dense_preflight_20260701_2052/dense_training_preflight_summary.json`.
      - Report:
        `experiments/reports/phase01/dense/preflight/p01_dense_preflight_20260701_2052/dense_training_preflight.md`.
      - Boundary: this is not training, not a checkpoint, not a real attempt,
        and not curiosity success.
- [x] Implement and smoke-test the Phase01 Newton-native dense closed-loop
      controller/probe training entry point.
      - Script:
        `src/newton_tactile_curiosity/phase01_dense_closed_loop_probe.py`.
      - Smoke config and launchers:
        `experiments/configs/phase01/dense/closed_loop_probe/`.
      - Short interface smoke:
        `p01_dense_clprobe_smoke_20260701_2100`, job `161003` on `server51`,
        status `pass_smoke_closed_loop_dense_probe`,
        `closed_loop_action_changed_any=true`.
      - Contact/lift smoke:
        `p01_dense_clprobe_smoke240_20260701_2101`, same allocation job
        `161003` on `server51`, status `pass_smoke_closed_loop_dense_probe`,
        max lift about `0.2225 m`, hold frames `89`, intrinsic score nonzero,
        safety cost `0.0`.
      - Summary:
        `experiments/outputs/phase01/dense/closed_loop_probe/smoke/p01_dense_clprobe_smoke240_20260701_2101/dense_closed_loop_probe_summary.json`.
      - Checkpoint:
        `checkpoints/phase01/dense/closed_loop_probe/smoke/p01_dense_clprobe_smoke240_20260701_2101/dense_closed_loop_probe_checkpoint.npz`.
      - Boundary: smoke only, not a counted one-hour real training attempt, not
        a strongest-baseline comparison, and not curiosity success.
- [x] Implement and smoke-test base-vs-checkpoint evaluation plumbing for the
      dense closed-loop probe.
      - Script:
        `src/newton_tactile_curiosity/phase01_dense_closed_loop_eval.py`.
      - Eval smoke:
        `p01_dense_clprobe_eval_smoke_20260701_2106`, job `161006` on
        `server51`, status `pass_eval_smoke_metrics_ready`.
      - Summary:
        `experiments/outputs/phase01/dense/closed_loop_probe/eval_smoke/p01_dense_clprobe_eval_smoke_20260701_2106/dense_closed_loop_eval_summary.json`.
      - Finding: easy cube smoke is too easy for a success claim;
        `base_zero_action` already gets max lift about `0.22243 m` and
        checkpoint policy gets about `0.22278 m`, with `delta_hold_frames=0`
        and no safety regression.
      - Boundary: this is eval plumbing only, not held-out harder-task success
        and not curiosity success.
- [x] Probe harder cells before the first real one-hour attempt so the strong
      base does not hide the research question.
      - `pen + mu=0.15` smoke:
        `p01_dense_hardcell_pen_mu015_eval_20260701_2109`, job `161007` on
        `server51`; base still reached about `0.2152 m` lift and `85` hold
        frames, so this is still too easy.
      - `pen + mu=0.02` smoke:
        `p01_dense_hardcell_pen_mu002_eval_20260701_2111`, same job `161007`;
        base reached only about `0.0268 m` lift and `0` hold frames, checkpoint
        smoke policy reached about `0.0381 m` lift and `0` hold frames, no
        safety regression.
      - Current candidate real-training cell:
        `scene=pen`, `override_mu=0.02`, `override_kh=1e12`,
        `num_frames>=240`.
      - Boundary: harder-cell probing is not a real training attempt and not
        curiosity success.

## Evaluation Tasks

- [x] Define strongest available baselines before training.
      - Baseline contract:
        `experiments/configs/phase00/dense_tactile_infant/baseline_eval_contract_v1.json`.
- [x] Enforce MP4-only video evidence for future visualization generation.
      - Newly generated rollout/visualization videos must be `.mp4`.
      - Do not generate `.avi` as active project evidence; convert or replace
        any upstream AVI output before recording it.
- [x] If the base model/controller already solves easy grasp/lift/hold, define
      harder held-out tasks or finer metrics so curiosity has room to improve.
      - Contract requires lower friction, mass/fill mismatch, off-center
        grasp, shape change, deformable/compliant object, fragile force limit,
        or held-out material/stiffness progression if base is easy.
- [x] Track lift, hold duration, slip, drop, contact loss, object acceleration,
      force/contact cost, and safety regression.
      - Metrics are defined in the baseline/evaluation contract.
- [ ] Require harder held-out improvement over strongest baseline without
      safety regression before any success claim.
      - Not complete because real closed-loop training/evaluation has not
        started.
      - Phase01 dense preflight passed and the next required step is real
        closed-loop training in a tmux-held Slurm allocation with attempt
        ledger evidence.
      - Eval smoke shows the easy cube setting is too saturated; the first real
        attempt must use harder held-out cells or stricter force/shear/safety
        metrics before any improvement claim.
      - Harder-cell smoke selects `pen + mu=0.02` as the first candidate cell
        because base lift/hold is weak enough to leave measurable room.
      - Early real-attempt launch
        `p01_dense_clprobe_attempt001_pen_mu002_20260701_2117` failed before
        one hour with a CUDA allocation error and is invalid/not counted.
      - Cancelled diagnostic
        `p01_dense_clprobe_diag_memfix_pen_mu002_20260701_2121` has no
        summary/eval and is not counted.
      - Memory-fix diagnostic
        `p01_dense_clprobe_diag_memfix2_pen_mu002_20260701_2130` completed
        about `613.6 s` without repeating the CUDA allocation failure; eval
        showed lift delta about `0.0222 m`, `delta_hold_frames=0`, and no
        safety regression. This is not a one-hour real attempt and not
        curiosity success.
      - Next action: restart the first counted real attempt on
        `scene=pen`, `override_mu=0.02`, `override_kh=1e12`,
        `num_frames=240`, `target_duration_s>=3600`, with GPU utilization,
        checkpoint/failure record, and automatic strongest-baseline/safety
        validation.
      - First counted attempt
        `p01_dense_clprobe_attempt001_pen_mu002_retry1_20260701_2140`
        completed about `3617.6 s` with `376` episodes and `188` generations.
        It is counted as negative attempt 1/5: checkpoint lift improved only
        from about `0.0289 m` to `0.0337 m`, `delta_hold_frames=0`, no safety
        regression, `success_claim_allowed=false`, and eval
        `drop_after_lift` was about `0.1345`.
      - Next action changed after attempt 1: do not repeat the same one-hour
        run unchanged. Inspect metrics and repair objective/action/search or
        curriculum so attempt 2 targets lift-to-hold stability and
        drop-after-lift reduction rather than raw lift alone.
      - Repair completed: closed-loop `lift_z_delta`, `lateral_y_delta`, and
        `probe_y_delta` now shift IK waypoint targets during rollout, and CEM
        keeps a sigma floor instead of collapsing after one elite.
      - Correct hard-cell repair smoke on `pen + mu=0.02`
        (`p01_dense_clprobe_repair_smoke_pen_mu002b_20260701_2258`) still had
        only about `0.0140 m` lift and `0` hold frames, so this cell is too
        hard for attempt 2.
      - Repair smoke/eval on `pen + mu=0.05`
        (`p01_dense_clprobe_repair_smoke_pen_mu005_20260701_2301` and
        `p01_dense_clprobe_repair_eval_pen_mu005_20260701_2304`) found a
        viable attempt-2 cell: base lift about `0.0698 m` and `0` hold frames;
        checkpoint lift about `0.2051 m` and `82` hold frames; no safety
        regression. This remains diagnostic-only, not success.
      - Next action: run counted attempt 2 on `scene=pen`,
        `override_mu=0.05`, `override_kh=1e12`, `num_frames=240`,
        `population_size=4`, `elite_count=2`, stronger drop penalty, and
        sigma floor, then validate against base/safety metrics.
      - Counted attempt 2
        `p01_dense_clprobe_attempt002_pen_mu005_20260701_2310` completed about
        `3624.0 s` with `372` episodes and `93` generations. It is the first
        positive counted attempt: base lift about `0.0700 m` and `0` hold
        frames; checkpoint lift about `0.2164 m` and `84` hold frames; no
        safety regression.
      - Boundary after attempt 2: do not call final curiosity success yet.
        Still required: replicated validation, stronger/held-out baseline
        comparison, ablations, and MP4-only rollout video evidence.
      - Replicated validation
        `p01_dense_clprobe_attempt002_repeval5_pen_mu005_20260702_0005`
        passed with `5` repetitions: base mean lift about `0.0702 m`, base
        hold `0`; checkpoint mean lift about `0.1934 m`, checkpoint hold `69`;
        no safety regression.
      - MP4-only rollout evidence
        `p01_dense_clprobe_attempt002_mp4_pen_mu005_20260702_0020` passed:
        `experiments/visuals/phase01/dense/closed_loop_probe/attempt002_mu005/base_vs_checkpoint_rollout.mp4`
        has `240` decoded frames at `30 FPS`, nonblank sampled frames, and
        `avi_generated=false`.
      - Held-out/ablation eval
        `p01_dense_clprobe_heldout_ablation_attempt002_mu005_fix1_20260702_0029`
        completed with `3` repetitions across train-like `mu=0.05` and
        held-out `mu=0.04/0.06/0.03`. It is mixed/negative for final claim:
        Attempt 2 remains strong on `mu=0.05` and `mu=0.06`, but `mu=0.04`
        has only about `+0.0062 m` lift delta and `0` hold delta, while
        `mu=0.03` is worse than base by about `-0.0055 m`. No safety
        regression. Noisy tactile damages performance, but this is
        evaluation-only and not tactile-mask training.
      - Counted attempt 3
        `p01_dense_clprobe_attempt003_curriculum_mu035045055_20260702_0045`
        ran about `3649.3 s` with `384` episodes and `32` generations using
        `train_mu_values=[0.035, 0.045, 0.055]`. Validation on `mu=0.04`
        improved lift only from about `0.0599 m` to `0.0661 m`, with
        `delta_hold_frames=0` and no safety regression. This is counted as
        negative real attempt 2/5.
      - Next action: do not repeat the same 8-parameter low-friction
        curriculum unchanged. Repair the policy/action space or objective for
        low-friction stable hold before another one-hour attempt.
      - Tail-hold diagnostic
        `p01_dense_clprobe_diag_tailhold_curriculum_20260702_0150` confirmed
        the objective issue: rewarding final lift and last-60-frame hold can
        produce train-cell final lift about `0.197 m` with `60/60` tail-hold
        frames in a short diagnostic. But validation at exact `mu=0.04`
        remained weak (`delta_max_lift` about `0.0097 m`, `delta_hold=0`),
        so it is diagnostic only, not success.
      - Interrupted Attempt4 candidate
        `p01_dense_clprobe_attempt004_tailhold_mu035040045055_20260702_0202`
        was stopped before one hour because predictor overflow caused a NaN
        aggregate score. It is invalid/not counted.
      - Repair applied after the invalid Attempt4 candidate: predictor update
        and score aggregation now have finite guards, and the runner can pass
        `INTRINSIC_WEIGHT`. The clean tail-hold Attempt4 must run with
        `INTRINSIC_WEIGHT=0` so objective repair is not polluted by predictor
        overflow.

- [ ] Repair low-friction closed-loop policy/action/search before the next real
      one-hour attempt.
      - Required because Attempt 2 failed lower-friction held-out transfer and
        Attempt 3's curriculum did not produce `mu=0.04` hold improvement.
      - Candidate repair directions: staged grip-before-lift objective,
        explicit hold-stability reward across all curriculum cells, stronger
        drop-after-lift penalty tied to final height, or additional residual
        actions for regrasp/pressure balancing rather than only the current
        8-parameter reactive policy.
      - Do not start Attempt 4 by simply rerunning
        `train_mu_values=[0.035, 0.045, 0.055]` with the same policy and
        objective.
      - Current clean Attempt4 target:
        `train_mu_values=[0.035, 0.04, 0.045, 0.055]`,
        `score_final_lift_weight=8.0`, `score_tail_hold_weight=0.08`,
        `score_drop_weight=12.0`, `stable_tail_frames=60`, and
        `INTRINSIC_WEIGHT=0`.

## Gate 00F Priority

- [x] Keep UniVTAC, TaCauchy, and IsaacLab TacSL as low-priority final
      semantic-validation/comparison-gap references.
      - Recorded in `IDEA/idea.md`, `AGENTS.md`, active plan, and active
        design contracts as final semantic/comparison references only.
- [x] Do not cycle on Gate 00F runtime/container work unless the user
      explicitly reopens that track.
      - Current Phase 00 work proceeded through Newton dense tactile/base
        evidence instead of blocking on Gate00F runtime/container setup.
