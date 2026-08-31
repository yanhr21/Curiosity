# TODO 17: Official Zero-WAM selected-video following

## A. Official audit

- [x] Read Zero-WAM paper v2, project page and official repository README completely.
- [x] Record the current repository boundary: main `5a8a2da` is project-page/README material only;
      code, model and data are announced for release before 2026-09-15.
- [x] Record the faithful architecture and objectives: Wan-2.2-TI2V-5B, 3072-D/30-layer video
      branch, 3072-D action branch, MoT factorization, shared Wan VAE, cached video prefix, action
      flow matching and training-only four-head stride-two IFP.
- [x] Record the scale boundary: 74.2K HumanGen pairs over 8.6K tasks, more than 6K task-balanced
      robot tasks, about 400K robot trajectories per epoch and 15,360 GPU-hours of pre-training.
- [x] Record the evidence boundary: `46.95%` on seven unseen RoboTwin tasks, IFP ablation
      `28.55% -> 46.95%`, stationary-tabletop emphasis and no exact-trajectory claim.
- [x] Implement and run a fail-closed canonical-release auditor: full main commit, recursive tree,
      real method code/entrypoints/schema, official checkpoint, strict load, official example,
      hashes and no-local-substitute must all pass machine-readable checks.
- [x] Reconfirm canonical main `5a8a2da069392c1974ee98941ada13a5208b0ca5` on 2026-08-31:
      five blobs, zero Python files, zero entrypoints/schema paths and
      `release_available=false`.
- [x] Stage the exact public Wan2.2 source commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`
      plus all 22 Wan2.2-TI2V-5B files (`34,203,123,632` bytes) in an isolated runtime; keep this
      explicitly below the Zero-WAM admission boundary.
- [x] Strict-load the exact public Wan base on H200 in BF16 with zero missing/unexpected/mismatched
      keys: `4,999,787,712` parameters, `10,263,348,736` peak bytes and a `0.5462 s` minimal-valid
      forward through all 30 released DiT blocks; record exact 22-file snapshot SHA256s.
- [x] Implement and self-test a fail-closed training-admission gate that marks the immutable SUGAR
      corpus ready only for the bounded two-task post-training audit, forbids 5B foundation
      pre-training on 199 trajectories and requires official release/prompt/29-DoF adapter gates.
- [x] Add a retained-H200 monitor that recomputes canonical release and training admission every
      fixed interval, exits automatically when official artifacts change and contains no approval
      or manual sentinel state.
- [ ] Detect the first official code/model/data release, freeze its full commit and checkpoint,
      and write `OFFICIAL_RELEASE_AUDIT.json` without asking for user authorization.
- [ ] Strict-load the official model on H200 and reproduce one released example plus documented
      tensor shapes, prompt caching and parameter/config identities.
- [ ] Measure official two-frame inference memory/time on H200; do not shrink the model after OOM.

## B. SUGAR paired-data contract

- [x] Verify the historical XIRL prompt corpus dimensions: 100 Carry + 99 Kick motions, 64 RGB
      frames each, `320 x 320` and fixed source-ID-disjoint train/valid/test split.
- [x] Reject that historical corpus for Plan 17 selected-demo training after visual inspection found
      neighboring tiled environments inside the 2.5 m-spacing camera frustum.
- [x] Complete the isolated Plan 17 prompt re-render with 30 m spacing, 20 m far clip and a fixed
      first-frame-only centering transform; retain the constant-frame gate.
- [x] Identify the required executable target chain as complete Generator+Tracker pairs with
      current 36-D command, 510-D Tracker observation and actual 29-D executed action.
- [x] Inventory the action-grounded half for all 199 motions: 139,300 official Generator+Tracker
      transitions, exact 36-D command/510-D observation/29-D executed-action equality, finite
      pre/post states, no reset and bitwise reproduction of the historical same-seed collector.
- [x] Build one immutable manifest row per complete pair with task/source/split, prompt/target frame
      paths, timestamps, command, observation, action and exact released expert identities.
- [x] Prove source-ID split disjointness, physical-time alignment, finite arrays and actual-action
      replay; reject prompt/target rows that reuse identical rendered pixels.
- [x] Build fixed wrong-task, reversed-frame and same-task-alternate prompt indices while holding
      robot history, target, diffusion noise and disabled-language state identical.
- [x] Emit the passing v2 machine-readable manifest gate: 199 rows, 139,300 actions, 27,860
      video-action intervals, 22,400 train intervals, 12,736 normalized pixel comparisons and zero
      identical prompt/robot pairs or complete streams.
- [x] Audit effective training diversity on the exact immutable manifest: Carry/Kick each contribute
      11,200 non-overlapping five-action chunks; all 22,400 action and observation chunks are unique,
      held-out exact overlap is zero, all 29 action dimensions vary, covariance rank is 29 and every
      task/phase bin is exactly balanced. Bind this result into formal training admission.

## C. Frozen official prompt gate

- [x] Freeze all 39 held-out motions at ten fixed causal phase anchors with matched, wrong-task,
      reversed, same-task-alternate and masked conditions: 390 matched-noise groups / 1,950 score
      instances, with source motion as the primary statistical unit.
- [x] Implement the fail-closed trajectory-level evaluator with one-sided exact sign tests, one
      eight-comparison Holm family, per-task direction checks, exact official provenance and
      pre-action-decoder future-change checks; pass positive and reject wrong-task-better fixtures.
- [ ] Run the released checkpoint on teacher-forced SUGAR robot histories with matched noise and
      official next-video flow loss.
- [ ] Evaluate matching versus wrong-task prompts on validation and test.
- [ ] Evaluate matching versus identical reversed prompts on validation and test.
- [ ] Evaluate matching versus different same-task source prompts on validation and test.
- [ ] Require positive paired mean margin, win rate above 0.5 and Holm-corrected exact sign
      `p < 0.05` on both splits for task, order and selected-motion identity.
- [ ] Require matching prompts to beat prompt masking and to change the predicted robot future
      before the action decoder under otherwise identical inputs.
- [ ] If any level fails, close SUGAR policy adaptation from that checkpoint without threshold,
      guidance, IFP-weight, chunk or model-size sweeps.

## D. Official G1 adaptation

- [x] Freeze a model-free evidence auditor for the official 29-DoF path: same passed prompt
      checkpoint, official adapter/config hashes, exact zero-update parameters, exact official
      two-update trainable scope, frozen VAE and joint video/action/IFP losses.
- [x] Freeze the serious fixed-data learnability contract at exactly 32 train motions (16/16), at
      chunk anchors `21/49/77/105` (exactly 128 causal samples) and `<=0.5x` tail-median
      video/action flow-loss ratios; pass complete
      synthetic evidence and reject action-only, frozen-scope drift and undocumented-interface cases.
      Fix each task's motion IDs to `2/7/14/21/26/33/40/45/52/57/64/71/76/83/90/95` before release.
- [x] Freeze the full formal-training coverage floor: seed271500, ten distinct complete epochs,
      every one of 160 train trajectories exactly once per epoch, chronological 140 intervals / 700
      actions per trajectory, prefix-balanced Carry/Kick, 224,000 interval and 1,120,000 action
      exposures, with exact schedule-file hash enforced by admission.
- [x] Freeze the formal run-completion evidence auditor: bind the real official run to the passed
      admission/commit/checkpoint and exact schedule; require H200 Slurm execution, at least ten
      complete epochs, finite strictly positive video/action/IFP gradients at every optimizer step,
      lower last-epoch median video/action loss, non-worse IFP, exact official trainable scope
      changes, bitwise-frozen VAE/scopes and hash-verified logs/checkpoint. Its contract tests pass
      complete evidence and reject nine epochs, order drift, action-only or single-joint-step
      gradients, flat full-data losses, VAE drift and held-out leakage; fixtures are not model evidence.
- [x] Replace trajectory-summary-only data evidence with a full-scale atomic-consumption contract:
      require exactly ordered records for all 224,000 ten-epoch intervals / 1,120,000 actions,
      immutable source/prompt/robot/action identities, contiguous single-trajectory packing, actual
      official-forward consumption and joint video/action/IFP targets. Pass the full-size contract
      fixture and reject nine epochs, duplication, reordering, cross-trajectory packing, held-out
      leakage, action-only targets and collapsed forward inputs; require 22,400 distinct forward
      fingerprints per epoch and bind the result/log hashes into checkpoint completion.
- [x] Bind every atomic-consumption optimizer-step ID to the joint optimizer trace: require batch
      and step indices monotonic and contiguous from zero, exact consumed-step min/max/count, no
      missing or extra optimizer record, finite video/action/IFP losses and strictly positive
      gradients at every step, plus first/last complete-epoch median progress. Reject a
      consumption-step gap, optimizer-trace gap, single-joint-step trace and flat losses in tests.
- [x] Prevent forward-only pseudo-training: require each packed sample to remain in one batch/step,
      every batch to map to one step, at most 140 atomic intervals per optimizer update and therefore
      a 1,600-update coverage minimum. Formal completion takes the maximum of that minimum, the
      paper's 4,000-step post-training reference and any larger released recipe. The 224,000-row
      fixture executes 7,000 updates and rejects both coverage-only 1,600-step and one-update-per-epoch
      evidence; no model or optimizer substitute is introduced.
- [x] Replace the self-reported budget boolean with structured evidence: require exact 4,000-step
      paper floor, positive released-recipe step count, effective configured step count no smaller
      than either, and at least the same number of contiguous consumed/optimizer-trace steps. Reject
      a configured 3,999-step fixture even when all other execution evidence passes.
- [x] Prove that every counted step is a real parameter update, not only a backward call: require
      positive effective LR, applied update, no AMP/scaler skip, exact before/after update indices,
      positive video/action/IFP parameter-update norms and zero non-finite trainable parameters after
      every step. Reject one skipped/zero-update row, one post-update non-finite row and booleans
      masquerading as numeric update evidence.
- [x] Bind optimizer contents, not only IDs: derive every step's single epoch, exact Carry/Kick
      interval counts, action exposures, forward batches and packed samples from the atomic log, then
      require field-exact optimizer-trace equality. Reject cross-epoch steps and a forged task
      composition while preserving the official model/optimizer boundary.
- [x] Prevent aggregate-loss masking: require task-conditioned official video/action/IFP losses only
      for tasks actually consumed by each step, then require Carry and Kick independently to improve
      last-epoch video/action medians with non-worse IFP. Reject loss records for absent tasks and a
      flat Kick trace even when aggregate losses improve.
- [ ] Inspect the released embodiment/action adapter and verify that 29-DoF continuous action
      chunks and SUGAR causal state can be represented without changing Zero-WAM semantics.
- [ ] If no documented adapter path exists, record an unsupported-embodiment blocker; do not write a
      replacement action Transformer.
- [ ] Run a zero-update audit for strict official modules, cached prompt-only swaps, causal masks,
      future-target isolation and exact-zero parameter change.
- [ ] Run a fixed two-update optimizer smoke and require finite nonzero changes only in modules
      selected by the official configuration.
- [ ] Run one fixed-data serious overfit using both official video-flow and 29-D action-flow targets;
      action MSE alone is insufficient.
- [ ] Admit formal post-training only if the joint overfit and frozen prompt gates pass.
- [ ] Run the exact admitted official recipe over the frozen full-data schedule and admit its final
      checkpoint to Stage E only if `audit_zero_wam_bounded_posttraining_run.py` passes.

## E. Closed-loop physical gates

- [x] Freeze the executable same-checkpoint SMALLBOX audit at seed281500: 20 matched profiles x two
      prompts x adapted/exact-endpoint baselines x 650 frames, 80 hash-unique traces, exact initial
      state/history and official endpoint-file hashes. Recompute safe Carry, strict-v4 safe Kick and
      root-height/root-tilt falls; require 16/20 per matched prompt, topology-specific advantage and
      no fall regression. Prove predicted future -> decoder -> executed 29-D action at every frame
      with no teacher future/target/router/demo reward. Contract tests reject checkpoint/state/cache,
      causal-chain/action, endpoint-identity, fall and 15/20 failures; fixtures are not model results.
- [x] Freeze the motion-disjoint test case set before outcomes: all 19 test motions, ten deterministic
      paired physics profiles and matched/reversed/same-task-alternate/wrong-task prompts, yielding
      190 groups / 760 adapted rollouts plus 380 identical-initial-state exact endpoint baselines,
      totaling 1,140 traces / 741,000 frames. All prompt sources remain test-only, evaluation
      targets stay outside deployed inputs and exact manifest SHA256 is
      `4be15b98fc8b0b71792dee053e657e39bdeaf0f8dd68840514c5b2d08f1d05e5`. Require per-source
      `8/10` prompted-task safety plus separate Holm-corrected order and identity gates.
- [x] Freeze the motion-disjoint result evaluator: re-audit all 1,140 full traces and endpoint file
      hashes, bind every prompt fingerprint to one official cached encoding, require exact initial
      physics/history and prompt-only condition swaps. Score every one of 190 matched causal traces
      at fixed full-horizon steps `49/99/.../649`: 2,470 matched-noise rows / 7,410 official flow
      comparisons. Aggregate physical safety/falls per source and order/identity after anchor then
      ten-profile reduction with one Holm family. Reject missing traces/anchors, `7/10`, fall
      regression, order/identity reversal, mismatched noise and prompt re-encoding in contract tests.

- [ ] Freeze one adapted checkpoint and restore elementwise-identical SMALLBOX physics/history for
      Carry45 and Kick21 prompt-only swaps.
- [ ] Require at least `16/20` strict Carry with 5 cm lift under Carry45 and `16/20` strict-v4 Kick
      under Kick21, with no fall-count regression against exact released endpoints.
- [ ] Prove that each deployed action is decoded from the generated next robot video, not a
      teacher-forced future, future target, router or demo reward.
- [ ] Archive synchronized prompt/predicted-future/actual/action traces and machine-readable checks.
- [ ] Only after SMALLBOX passes, run all held-out test-motion prompts and report physical task,
      fall, order and selected-motion identity gates separately.
- [ ] Only after the official HumanGen pipeline and all prior gates pass, run one faithful generated
      human-prompt cross-embodiment follow-up.

## F. Newton closure and hygiene

- [x] Close seed171755 receding-knot Newton Refiner training at the sole model31 endpoint: 32
      updates, 60,160 transitions, 1,792 latches and zero divergence.
- [x] Record frozen seed181752 data000 failure: `0.01005 m` lift, `0.20851` bilateral contact and
      `0/1` strict; task-wide was automatically skipped.
- [x] Forbid another Newton residual, knot, update, bound, reward, prefix, std or LR sweep as the
      response to this failure.
- [x] Keep datasets/checkpoints/videos under ignored `experiments/`; commit only source and docs.
