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
- [ ] Detect the first official code/model/data release, freeze its full commit and checkpoint,
      and write `OFFICIAL_RELEASE_AUDIT.json` without asking for user authorization.
- [ ] Strict-load the official model on H200 and reproduce one released example plus documented
      tensor shapes, prompt caching and parameter/config identities.
- [ ] Measure official two-frame inference memory/time on H200; do not shrink the model after OOM.

## B. SUGAR paired-data contract

- [x] Verify the existing clean prompt corpus: 100 Carry + 99 Kick motions, 64 RGB frames each,
      `320 x 320`, clean-frame contract and fixed source-ID-disjoint train/valid/test split.
- [x] Identify the required executable target chain as complete Generator+Tracker pairs with
      current 36-D command, 510-D Tracker observation and actual 29-D executed action.
- [ ] Inventory complete action-grounded robot-video targets for all 199 source motions; record
      missing or non-finite motions instead of synthesizing actions.
- [ ] Build one immutable manifest row per complete pair with task/source/split, prompt/target frame
      paths, timestamps, command, observation, action and exact released expert identities.
- [ ] Prove source-ID split disjointness, physical-time alignment, finite arrays and actual-action
      replay; reject prompt/target rows that reuse identical rendered pixels.
- [ ] Build fixed wrong-task, reversed-frame and same-task-alternate prompt indices while holding
      robot history, target, diffusion noise and disabled-language state identical.
- [ ] Emit a machine-readable manifest gate; do not start model adaptation when it fails.

## C. Frozen official prompt gate

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

## E. Closed-loop physical gates

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
- [ ] Keep datasets/checkpoints/videos under ignored `experiments/`; commit only source and docs.
