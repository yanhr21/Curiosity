# Global Agent Rules

## Highest-Priority Online 54-Patch Tactile Mass-Adaptation Training (2026-08-14)

- This section supersedes every older prohibition or execution queue concerning
  policy training. The only active plan and TODO are
  `PLAN/15_online_patch_tactile_mass_adaptation/plan.md` and
  `TODO/15_online_patch_tactile_mass_adaptation/todo.md`. Plan 14 and all older
  plans are read-only legacy. Do not resume RGB, demo following, ICM/Curiosity,
  Newton simulation, deformable demos or any unrelated training while Plan 15
  is active.
- The scientific question is whether live whole-hand tactile improves frozen
  physical behavior when a complete SUGAR G1 has already lifted CarryBox and
  the PhysX mass changes online without changing geometry or appearance. Start
  from `0.3023375869 kg` and audit/train/evaluate the predeclared
  `1.5x/3x/6x/10x` sweep. Change mass and inertia between control actions,
  continue the same episode, and never reset/replay the sensor history at the
  jump.
- Policy tactile space is exactly bilateral `27-patch` anatomy, never taxels:
  palm `4 x 3`, then proximal/middle/distal on thumb, index, middle, ring and
  little finger. Official R15 taxels remain the physical sensor backend and
  raw audit source, but each control step must reduce them online to one record
  per patch. Every patch record contains TacSL-derived contact, normal load,
  mean pressure, signed local-XY shear and friction utilization. Do not replace
  any field with `hands_contact_label`, ordinary ContactSensor output, object
  state, generated values or an offline trace.
- Add one causal, batch-stateful IsaacLab callable
  `PatchSlipDetector.update(...)`. It operates on current/past 54-patch
  contact/pressure/shear/friction plus timestamps and reset masks, and returns
  per-patch slip evidence and `NO_CONTACT/STICK/INCIPIENT/GROSS` state. Object
  motion, relative contact velocity, mass factor, jump flag, reward and future
  frames are evaluation labels only and may never enter the callable or actor.
- Official Refiner `890-D` observation contains measured object state such as
  `obj_lin_vel_b`, and robot `joint_pos/joint_vel` can also leak load after the
  hands sag. The deployed actor must therefore use the existing `504-D`
  no-measured-object-state Tracker-command/proprioception contract. The
  official `890-D` Refiner and privileged critic are training-only. Before any
  policy training, run the Plan-15 time-resolved leakage audit and report
  object-state, proprio-only, patch-tactile and patch-tactile-plus-slip onset.
  Never claim that only tactile senses mass unless the evidence supports it;
  otherwise test and report incremental benefit over proprioception.
- Run exactly three serious matched branches, serially: `Z` uses exact-zero
  patch/slip tensors and neither its actor observation nor the mass scheduler
  reads TacSL; `P` uses live patch
  contact/load/shear/friction with zero slip fields; `PS` uses the same live
  patch signal plus the causal slip callable. They share the anatomical
  patch-token encoder, official Tracker warm start, unchanged SUGAR
  `512/256/128` actor, official frozen Refiner teacher, repository BCPPO,
  optimizer, reward, physics, mass sampling, seeds and update budget. No toy
  MLP, offline tactile replay or taxel-CNN substitute is allowed.
- Repository BCPPO gives the actor no task-reward PPO before update 1000:
  updates 0--499 are pure distillation, 500--999 add only critic warmup, and
  1000--1999 ramp PPO authority. Updates 2000--2999 retain full PPO authority
  and a shared `stage3_distill_weight_floor=0.25` Refiner BC anchor. This is the
  existing repository BCPPO mechanism, not a new reward or teacher action at
  deployment. The zero-floor replacement-Z endpoints are frozen negatives:
  seed `151014` distillation loss rose from `0.3404` at update 2000 to
  `35.8202` at update 2999, the pre-handoff teacher/student action L2 grew from
  about `0.9` to `5.4--5.9`, and the endpoint failed roughly seven frames after
  handoff before any mass event. The same update-2000 checkpoint produced
  three real `1.5x` jumps and survived `65/38/74` post-jump frames, establishing
  behavior forgetting rather than a sensor or handoff-wrapper failure. Resume
  all three Z seeds from their update-2000 checkpoints with the shared `0.25`
  floor and the original fixed update-2999 endpoint; apply the identical floor
  to P and PS. Each formal run remains exactly 3000 updates. The withdrawn
  512-update Plan-15 draft cannot answer tactile training benefit.
- Never extend a Plan-15 run past the fixed `model_2999.pt` endpoint and never
  automatically chain the next formal seed after an endpoint. Freeze training
  there and first inspect checkpoint finiteness, live handoff, mass readback,
  the fixed 80-frame physical window, action continuity and synchronized
  world/54-patch video. If the behavior is invalid or ambiguous, run one
  fixed-condition overfit diagnostic with the same serious SUGAR policy,
  live Refiner handoff and online physics before spending another formal
  3000-update budget. Such an overfit is a diagnostic, not a formal Z/P/PS
  result. Do not continue long training while the endpoint behavior is
  unknown.
- All three anchored Z seeds `151014/151015/151016` have now stopped at the
  fixed finite `model_2999.pt` endpoint, each with 59 model tensors and 58
  optimizer states and no later checkpoint. Seed `151016` was externally
  interrupted with job `239098` after iteration 2943, resumed only from its
  complete `model_2750.pt` at iteration 2751 on `240173/server07`, and exited
  normally at 2999. Its 100-rollout camera-free frozen audit has physical holds
  `20/20,20/20,20/20,0/20,0/20` and drops
  `0/20,0/20,0/20,20/20,20/20` for `1x/1.5x/3x/6x/10x`; robot falls are all
  zero. All five `450 x 20` traces are finite, all 100 mass events read back
  the requested mass, and all have ten bilateral-contact frames before the
  event. Z remains frozen and must not be extended. After the user explicitly
  requested the unfinished tactile arms to train on 2026-08-15, formal
  `P/seed151014` started from the official Tracker on retained job
  `240170/server44` and has now completed its fixed `model_2999.pt` endpoint.
  The scheduler later marked job `240170` `CANCELLED by 0` after the trainer
  printed iteration 1734; this was not a training exception or voluntary GPU
  release. The last complete checkpoint was `model_1500.pt`. P resumed on
  retained `231256/server64`, which was also externally `CANCELLED by 0` after
  iteration 2458; its last complete checkpoint was `model_2250.pt`. It then
  resumed on retained `240922/server07` at BCPPO step and runner iteration 2251
  and exited normally at iteration 2999. Its finite endpoint contains 59 model
  tensors, 42 patch-encoder tensors and 58 optimizer states, with no later
  checkpoint. Unsaved intervals from both cancelled jobs are excluded.

  The paired `151014->152014` camera-free frozen evaluation is complete at 20
  profiles per factor. One shared profile fails before handoff, leaving 19
  eligible profiles per factor. P holds are `19,19,17,0,0` and drops are
  `0,0,2,19,19` at `1x/1.5x/3x/6x/10x`; paired Z holds are `19,19,16,1,0`
  and drops are `0,0,2,18,19`. This single seed shows only a mild 3x indication
  and no 6x benefit, so it does not prove tactile benefit. Two server07 camera
  starts failed before scene execution with `VK_ERROR_DEVICE_LOST`; this does
  not invalidate the camera-free rollout statistics. A separate render
  allocation was requested. After this numerical endpoint review,
  `P/seed151015` started from scratch on retained `240922/server07`, which was
  externally `CANCELLED by 0` after iteration 347. Its last complete checkpoint
  is `model_250.pt`; unsaved iterations 251--347 are excluded. It resumed on
  retained `241217/server59` at BCPPO step and runner iteration 251 with the
  original adaptive-KL learning rate and unchanged 3000-update contract. That
  allocation was externally cancelled after printed iteration 1961; its last
  complete checkpoint is `model_1750.pt`, so unsaved iterations 1751--1961 are
  excluded. It resumed on retained `241298/server59` at BCPPO step and runner
  iteration 1751 and completed normally at `model_2999.pt`. The finite endpoint
  contains 59 model tensors, 42 patch-encoder tensors and 58 optimizer states at
  learning rate `1e-5`, with no later checkpoint.

  Its paired `151015->152015` evaluation completed all 100 live rollouts. P
  holds are `20,20,18,0,0`, drops are `0,0,0,20,20` and robot falls are all
  zero at `1x/1.5x/3x/6x/10x`; paired Z holds are `20,20,16,0,0`, drops are
  `0,0,0,20,20` and robot falls are `0,0,0,4,3`. At 3x the paired hold
  discordance is four P-only versus two Z-only profiles. This is an indication,
  not a benefit result. `P/seed151016` continued from finite `model_500.pt`
  through two exact recoveries: `242239/server23` produced `model_750.pt`, and
  `242242/server06` produced `model_1250.pt`. Both short allocations ended by
  scheduler time limit; the earlier `241811/server28` unsaved 501--747 interval
  and the later 751--913 and 1251--1320 intervals are all excluded.
  Retained job `242229/server23` continued from `model_1250.pt` until the
  scheduler externally cancelled it after printed iteration 2546; its last
  complete checkpoint is `model_2500.pt`, so the unsaved 2501--2546 segment is
  excluded. Job `242660/server07` resumed at runner/BCPPO iteration 2501 and
  completed normally at finite `model_2999.pt`. The endpoint contains 59 model
  tensors, 42 patch-encoder tensors and 58 finite optimizer states at learning
  rate `1e-5`, with no later checkpoint. Its paired `151016->152016` evaluation
  completed all 100 rollouts: P holds are `20,20,14,0,0`, drops are
  `0,0,6,20,20`, and robot falls are `0,0,0,1,0`.

  Across all three paired seeds, P holds are `59,59,49,0,0` versus Z
  `59,59,52,1,0`, while P drops are `0,0,8,59,59` versus Z
  `0,0,2,58,59`. The 3x paired hierarchical-bootstrap interval for the P-Z
  hold difference crosses zero, so P does not establish tactile benefit and
  trends worse at 3x. Formal PS seed `151014` has now started from scratch on
  retained `242660/server07` with `OnlinePatchSlipMassRobotEnvCfg`. The valid Z
  mild/boundary/heavy behavior means no Z overfit is currently needed.
- The seed-`151015` endpoint has now passed the required frozen numerical and
  human-visible review without more training. Its 450-frame synchronized H.264
  evidence separates a `1.5x` hold, a `3x` bilateral-contact boundary with
  about `0.052 m` sag, and `6x/10x` box drops. The `6x` camera replay also
  destabilizes the robot after the drop; report that replay honestly rather
  than claiming frame-exact agreement with the camera-free trace. This review
  did not trigger overfit. It was the historical review gate before the now
  completed P seeds; formal PS subsequently started only after the complete
  three-seed P review above.
- Formal frozen evaluation uses at least 450 control frames, not 420. With the
  observed handoff near frame 297, the declared 50-frame maximum mass delay
  and 80-frame outcome window require coverage through at least frame 427.
  The old 420-frame horizon silently excluded late-jump profiles and is
  withdrawn for formal statistics; 420-frame camera videos remain valid visual
  records when their selected profile already contains a complete window.
- Formal training seeds are `151014/151015/151016`. Pair their endpoint
  checkpoints one-to-one with disjoint frozen-evaluation seeds
  `152014/152015/152016` in the same order. Each checkpoint/seed pair receives
  20 profiles for each of five mass conditions, hence exactly
  `3 x 5 x 20 = 300` rollouts per Z/P/PS arm. Do not evaluate every checkpoint
  on all three evaluation seeds, which would silently change the design to 900
  rollouts per arm. Never add profiles to only one arm.
- Schedule the mass event from sustained object lift and a matched random
  delay, not from tactile. Separately require live P/PS traces to show
  bilateral patch contact throughout the ten frames before the event. This
  keeps the event clock matched while preserving Z as a no-TacSL-read arm.
- The original frame-zero student-control training path is now a frozen
  negative and may not be resumed. Two independent 3000-update Z endpoints
  produced zero box-contact frames in all eight 1.5x profiles; a separate
  nominal 1.0x check reproduced the same four termination frames as 1.5x.
  Thus the failure precedes both contact and the mass event. Seed `151016` was
  stopped at iteration 226 by targeting only its recorded child process group;
  retained job `238355` remains alive.
- Every replacement training and frozen-evaluation episode must use a live
  official-Refiner handoff. The exact frozen Refiner controls the same complete
  G1 from motion-45 frame 0 until the box has remained at least `0.05 m` lifted
  for 10 consecutive control frames. Then, without reset, teleport, replay or
  sensor-history replacement, the matched Z/P/PS actor takes control in that
  same PhysX episode. The mass scheduler waits its matched `10--50` frames only
  after this physical hold qualification. Pre-handoff actions are online
  teacher actions and are excluded from PPO surrogate/value/entropy credit;
  they may retain the unchanged official teacher-distillation target. The
  handoff mask is training/evaluation infrastructure and may not enter the
  deployed actor.
- Re-run the Z/P/PS live one-update preflights with this handoff before any new
  formal training. Each preflight must show teacher-controlled continuous
  pickup, a no-reset handoff, student-controlled post-handoff steps and a real
  mass event. Z must still perform zero TacSL reads; P/PS must fill their four-
  frame histories online during the same teacher-controlled prefix. Do not
  start new P or PS formal training until the replacement Z endpoints contain
  eligible post-jump behavior.
- All three replacement one-update preflights now pass on retained
  `238253`/`server59`. Each ran 1440 transitions, completed four live Refiner
  handoffs and two real mass changes, and assigned policy credit only to the
  `142/143` post-handoff transitions while masking the `1298/1297` teacher
  transitions. Z made 364 exact-zero observation calls and zero TacSL reads.
  P and PS each made 361 online feature updates and exactly 19,494 official
  patch reads, observed bilateral contact in 363 env-samples, and P made zero
  slip calls. PS made 361 causal slip calls and returned nonzero incipient and
  gross patch states. These results admit new formal Z training only; they do
  not prove tactile benefit or authorize P formal training before valid Z
  endpoints.
- The first zero-floor replacement handoff-Z pass is a frozen training
  negative. Seed `151014` and `151015` completed finite `model_2999.pt`
  endpoints, but seed `151014`'s four-profile `1.5x` gate had zero eligible
  profiles: three handoffs failed about seven frames later before the mass
  event, and one profile ended before handoff. Stage diagnostics on the exact
  same four profiles showed update 1000 produced three real jumps and survived
  `18/21/60` post-jump frames, while update 2000 survived `65/38/74` frames;
  neither reached the fixed 80-frame eligibility window. This directly tracks
  the zero-floor distillation collapse described above. Do not use or resume
  the zero-floor update-2999 endpoints as formal Z results. Replacement Z now
  resumes from update 2000 with the shared `0.25` BC anchor; P/PS remain
  forbidden until anchored Z endpoints yield eligible post-jump profiles.
  The first anchored recoveries passed their runtime contracts: seed `151014`
  restored BCPPO step 2001 with optimizer LR `1e-5`, produced a finite
  `model_2250.pt` with 59 model tensors and 58 optimizer states, and retained
  simultaneous full-PPO updates plus distillation weight `0.25`. Seed `151016`
  restored step 751 and produced a finite `model_1000.pt`. Job `239105` was
  externally cancelled after seed `151014` printed iteration 2337. On retained
  `239106`, only seed `151016`'s recorded child PGID was then stopped after its
  update-1000 checkpoint so seed `151014` could resume from 2250; that job was
  externally cancelled after iteration 2304. At that time the restart points
  were anchored `151014/model_2250.pt` and `151016/model_1000.pt`; unsaved
  iterations were excluded. This scheduling snapshot is superseded by the
  completed `151014/151015/151016` anchored endpoints below. Do not call either
  external cancellation a training failure, and do not restart from older
  checkpoints when an allocation begins. Historical allocation details follow.
  Seed `151014` has a
  complete `model_1500.pt`; its allocation
  `238253` was externally `CANCELLED by 0` after iteration 1711. Seed `151015`
  has a complete `model_2250.pt`; its allocation `238620` was externally
  `CANCELLED by 0` after iteration 2339. Both restart checkpoints contain 59
  model tensors and 58 optimizer states and are finite. Unsaved iterations are
  excluded. Eight-hour backfill job `239105` on `server35` has restored seed
  `151014` at iteration 1501, and job `239106` on `server44` has restored seed
  `151015` at iteration 2251; both checkpoint, BCPPO-stage and optimizer-
  learning-rate synchronizations passed. Seed `151015` then completed normally
  at `model_2999.pt`; the endpoint contains 59 model tensors and 58 optimizer
  states and is finite. The same retained job immediately started replacement
  seed `151016` from the official Tracker at iteration 0 with a fixed 3000-
  update budget. Seed `151014` had passed its finite `model_2000.pt` and was
  continuing in steady full-PPO at that time. These were Z training milestones,
  not mass-adaptation results; current endpoint status is recorded below.
- Anchored Z seed `151014` has now completed exactly 3000 iterations at finite
  `model_2999.pt` (59 model tensors, 58 optimizer states); no later update was
  run. Its final distillation loss is `2.0839` with the fixed `0.25` anchor.
  The corrected 450-frame camera-free four-profile `1.5x` frozen gate has two
  eligible 80-frame holds. Profile 0 terminates 39 frames after the jump on
  `obj_pos`; profile 1 terminates before handoff on `ee_body_pos`; profiles 2
  and 3 complete the 80-frame window and later terminate on `obj_ori` at frames
  423 and 410. The old 420-frame count of one eligible profile was a horizon
  truncation, not policy failure. The pre-handoff ten-frame
  student/teacher action L2 is `1.04--1.12`, versus `5.51--5.86` for the
  withdrawn zero-floor endpoint. A repeated camera rollout of eligible profile
  3 fully decoded 420 H.264 frames, changed mass from `0.302` to `0.454 kg` at
  frame 308 and retained the box through the 80-frame window while displaying
  both 27-patch hands. The separately recorded camera profile 0 also retained
  the box until frame 396, 59 frames after its jump, at about `+0.823 m` lift;
  it was ineligible because of reference-tracking termination, not a physical
  drop or robot fall. A no-learning physical-continuation diagnostic that
  disables only `obj_pos/obj_ori` termination also has two eligible
  holds; profile 0 then reaches `anchor_pos` at frame 383 while still carrying
  with bilateral contact. This proves that anchoring repaired catastrophic
  handoff forgetting, but two favorable profiles are not tactile benefit. This
  review admitted one additional Z endpoint only; it did not authorize P or
  PS. Those branches remain unstarted.
- The anchored seed-151014 four-profile mass audit now uses the corrected
  450-frame horizon and exact termination labels. Eligible holds are
  `1/4, 2/4, 2/4, 0/4, 0/4` for `1.0x, 1.5x, 3x, 6x, 10x`. At `6x`, profile 2
  drops `0.194 m`; at `10x`, profiles 0/2 drop `0.165/0.216 m`. Thus this Z
  endpoint has both mild successes and heavy failure space and does not need a
  single-profile overfit merely to prove learnability. The four-profile audit
  is diagnostic only and must not be presented as a monotonic mass curve or
  tactile benefit. Jump frames match exactly across factors. Pre-jump actions
  and object positions are exact for `1.5x/3x/6x`; `10x` profile 2 has a small
  two-frame pre-event closed-loop divergence (`0.0146` action max, `0.23 mm`
  object position) while mass remains nominal, so never claim bitwise pairing
  for every closed-loop rollout.
- Anchored Z seed `151015` then completed exactly 3000 iterations and stopped
  at finite `model_2999.pt` (59 model tensors, 58 optimizer states); no later
  update and no automatic next seed were run. Its 450-frame, four-profile
  physical-outcome audit keeps the original SUGAR termination terms as labels
  while continuing the same PhysX episode. Physical hold counts for
  `1.0x/1.5x/3x/6x/10x` are `4/4,4/4,4/4,0/4,0/4`; all four `6x` and all four
  `10x` profiles physically drop, and one `10x` profile also meets the declared
  robot-fall definition. Strict SUGAR reference-window hold counts on those
  same camera-free trajectories are `0/4,1/4,3/4,0/4,0/4`. Therefore reference
  deviation and physical failure must be reported separately. This is a small
  endpoint audit, not the formal 20-profile-per-factor result and not tactile
  benefit. The clear mild-pass/heavy-failure region means no overfit is needed
  now; freeze training for review. If later behavior is invalid or ambiguous,
  use the declared fixed-condition serious overfit before another formal
  budget. P and PS must not auto-start.
- The formal frozen audit of anchored Z seed `151015` is now complete at exactly
  `20` profiles for each of `1.0x/1.5x/3x/6x/10x`, or 100 live PhysX rollouts.
  Physical hold counts are `20/20,20/20,16/20,0/20,0/20`; drop counts are
  `0/20,0/20,0/20,20/20,20/20`; robot-fall counts are
  `0/20,0/20,0/20,4/20,3/20`. The four non-hold `3x` profiles sagged by about
  `0.054--0.078 m` but did not meet the drop criterion, so `3x` is a boundary
  condition rather than a binary failure condition. All five `450 x 20`
  traces are finite, all 100 rollouts read back the requested mass, and all 100
  satisfy the ten-frame bilateral patch-contact gate before the event. The
  matched post-handoff delay sequence is identical profile-by-profile across
  factors; live handoff and event times vary by at most four frames. Do not
  claim bitwise-identical closed-loop trajectories. This is a serious frozen Z
  endpoint result, not tactile benefit. It gives a valid mild/boundary/heavy
  evaluation range, so no overfit is needed now. Training remains frozen; if
  later human-visible review invalidates the behavior, use the already declared
  serious fixed-condition overfit before any new formal budget.
- The matching formal frozen audit of anchored Z seed `151014` is also complete:
  `20` profiles for each of `1.0x/1.5x/3x/6x/10x`, with one identical
  profile-1 official-Refiner failure before handoff in every factor. Excluding
  that pre-handoff teacher failure, the `19` eligible post-jump profiles per
  factor have physical hold counts `19/19,19/19,16/19,1/19,0/19` and drop
  counts `0/19,0/19,2/19,18/19,19/19`; eligible robot-fall counts are
  `0/19,0/19,3/19,0/19,3/19`. All five traces are finite, every real event has
  correct mass readback and ten preceding bilateral-contact frames, and the
  matched delay sequence is identical across factors. The teacher-prefix
  failure is infrastructure coverage loss, not a student outcome and not a
  reason to extend or overfit the student.
- The formal frozen audit of anchored Z seed `151016` is complete at exactly 20
  profiles per factor. Physical holds are
  `20/20,20/20,20/20,0/20,0/20`; drops are
  `0/20,0/20,0/20,20/20,20/20`; robot falls are all zero. All five
  `450 x 20` traces are finite and all 100 events pass mass readback and the
  ten-frame bilateral-contact gate.
- Seed `151016` now also has two fully decoded 450-frame H.264 camera rollouts.
  The `3x` profile-0 video holds with bilateral contact; the `6x` profile-0
  video drops by `0.5619 m`. Both show the complete G1/CarryBox world view and
  both 27-patch hands on one clock. They are evidence of their own camera-enabled
  rollouts, not frame-exact replays of the authoritative camera-free traces.
- Across all three anchored Z checkpoint/seed pairs, eligible physical holds
  are therefore `59/59,59/59,52/59,1/59,0/59` for
  `1.0x/1.5x/3x/6x/10x`; drops are `0/59,0/59,2/59,58/59,59/59`. This is a
  valid frozen Z baseline with mild, boundary and heavy conditions, so no
  overfit and no further Z training are currently warranted. A fixed-condition
  serious overfit is used only if later review shows invalid or ambiguous
  behavior, never as automatic continuation beyond `model_2999.pt`.
- Live world-camera evaluation is outcome-sensitive near the boundary and is
  not the source of formal statistics. Seed `151014`, `6x` profile 7 is an
  exact repeatable camera-free hold at handoff/jump `297/307`, but the matching
  camera-enabled rollout drops by `0.279 m`. Conversely, formal camera-free
  `3x` profile 0 drops and the robot falls at handoff/jump `297/337`, while its
  camera-enabled rollout holds through the declared window. A fresh
  camera-free four-profile repeat completed with exit code 0 and all 23 trace
  fields exactly equal to profiles 0--3 of the formal run, including profile
  0's `0.452212 m` loss, drop and robot fall. Therefore the flip is a measured
  camera-path perturbation, not ordinary no-camera rerun variance. Report the
  camera-free 20-profile traces as authoritative and label every video as
  evidence of its own camera-enabled rollout; never claim that such a video is
  a frame-exact replay of the formal outcome.
- The paired frozen-Z reaction-window audit now covers all three endpoint pairs
  and all `59` eligible profiles per mass factor. Each heavy trace is
  event-aligned with the same profile's `1x` trace; patch channels use the
  already fixed scales, and onset is the first two post-jump samples above the
  profile's ten-frame pre-jump paired-difference maximum. Continuous
  load/pressure/signed-shear/friction is separated from contact binary. Among
  all `119` box drops, continuous patch change precedes drop in `119/119` with
  median lead `21` frames (`0.42 s`), versus `15` frames for contact binary;
  slip precedes `118/119` with median lead `11` frames (`0.22 s`). Normal load
  and pressure individually precede all `119` drops, each with median lead
  `20` frames, so the result is
  not driven only by friction or a changed contact bit. More importantly,
  continuous patch change precedes all `133` profiles that sag at least `0.02 m`,
  with median lead `7` frames, while contact binary precedes sag in only `81/133`
  with median lead `3` frames. This establishes an information advantage over
  binary contact and a usable sensing opportunity, not tactile-policy benefit.
  The Z action has a detectable onset for 117 drops and already diverges before
  111 of them, so proprioceptive/closed-loop leakage is behaviorally relevant
  and the eventual claim must remain incremental tactile benefit.
- The reaction-window result has a fully decoded 450-frame H.264 at
  `experiments/online_patch_tactile_mass_adaptation/frozen_reaction_window_v1/`
  `videos/6x_camera_and_formal_reaction_window_v1.mp4`. The left pane is the
  already admitted camera-enabled complete-G1/CarryBox/54-patch `6x` drop; the
  right pane animates the `38` formal camera-free `6x` drop profiles. The video
  states on-frame that these are separate rollouts and that formal counts use
  camera-free traces only. Do not erase this distinction when presenting it.
- The formal evaluator's multi-batch path now resets inside inference mode,
  clears IsaacLab's latched termination-reason buffer after each evaluation
  reset, preserves uncaught main exceptions instead of allowing application
  close to return success, and requires both trace and summary outputs before
  advancing to the next factor. An admitted two-batch eight-profile diagnostic
  and the subsequent five-factor formal audit show no inherited frame-zero
  termination labels. This is an evaluator correctness fix, not a new training
  gate.
- Retained job `242660/server07` is the current formal `PS/seed151014`
  allocation. Job `242229` was externally cancelled after P printed iteration
  2546; P recovered from finite `model_2500.pt`, completed, and was evaluated
  on `242660`. Jobs `231256`, `238054`,
  `239098`, `240170`, `240173`, `240922`, `241217`, `241298`, `241811`,
  `242239` and `242242` ended through scheduler enforcement and are no longer
  retained. Ending an audit or this agent turn is not permission to exit
  `242660` or any subsequently granted allocation.
- Do not weaken the lift gate merely to make an early one-update training
  preflight emit a mass event. The admitted continuous-action full-G1 collector
  is the mass/inertia-event physics gate. A stochastic warm-start policy may
  terminate before lift; its one-update preflight must preserve and report that
  fact while judging only the declared Z/P/PS online observation and update
  path.
- A positive result requires matched frozen-policy improvement in physical
  post-jump hold/recovery/safe-lower behavior with nominal no-jump behavior
  reported separately. Gradients, training loss, predicted reward, nonzero
  action difference or a single favorable video prove signal use only. Final
  H.264 evidence must show complete G1/CarryBox motion and both readable
  27-patch maps with contact, pressure, signed shear and slip on the same
  clock; the mass/jump overlay is evaluator-only and must be labeled as hidden
  from the actor.
- All sensing and slip inference used by training must be generated inside the
  current IsaacLab rollout before the next actor call. Saved traces may be used
  for audit and rendering only. Complete the Plan-15 documentation and leakage
  audit before training, execute one branch at a time, keep retained GPU
  allocations alive, and do not add routine hashes or defensive version
  ladders.
- Live runtime is restored as of 2026-08-14. Official AppLauncher on retained
  job `238128` completed a 420-frame full-G1/54-patch CarryBox rollout by
  reusing the already converted official G1 USD instead of rerunning the URDF
  importer during scene construction. At frame 299 PhysX changed and read back
  mass from `0.3023376` to `0.9070128 kg`; bilateral contact was present for the
  preceding ten frames, all 54 official TacSL clocks remained synchronized and
  strictly advanced, and the lifted box later dropped. This is live online
  evidence, not replay. The subsequent `3 seeds x 5 factors` fixed-action
  leakage sweep is complete: all 15 trajectories have exact paired actions and
  event frames, correct live mass readback, bilateral pre-event contact and
  synchronized advancing 54-patch clocks. At the event, patch-contact binary
  is unchanged for every mass pair while continuous patch load/pressure and
  the `504-D` proprio contract both respond. The scientific claim is therefore
  incremental tactile benefit over proprioception, never tactile-exclusive
  mass sensing. Exploratory leave-one-seed-out probes first become reliably
  mass-discriminative at 13 frames for patch tactile versus 35 frames for
  proprio, but three seeds are not a policy-benefit result.
- The original slip thresholds are withdrawn: across the 6300-frame CarryBox
  sweep they labeled all 14 oracle STICK samples GROSS because friction
  utilization saturates during ordinary loaded motion. The replacement stays
  causal and tactile-only: friction utilization marks INCIPIENT, while GROSS
  requires two consecutive high shear-rate or pressure-drop samples; contact
  loss after load remains a gross alert. In a 240-frame controlled official-R15
  trace it separated fixed contact, `0.006 m/s` slow slide, `0.03 m/s` fast
  slide and `0.01 m/s` return as STICK/INCIPIENT/GROSS/INCIPIENT. State
  confusion was `109/111` STICK, `109/109` INCIPIENT and `19/20` GROSS;
  incipient onset delay was zero and gross onset delay was one 50-Hz frame.
  Held-out object speed was used only as an evaluation label. A detached R15 is
  allowed only for this calibration and does not satisfy the G1 object-demo or
  policy-training result. The same callable then completed a separate
  420-frame full-G1 CarryBox `3x` live run: 107 bilateral-contact frames, mass
  event at frame 328 after ten bilateral frames, zero 54-patch clock skew and
  a strictly advancing official sensor clock. Against evaluation-only active-
  taxel velocity it had contact-supported precision `1.0`, recall `0.9971`,
  median delay zero and p95 delay one frame; all 28 loaded contact losses
  produced gross alerts. PS may proceed to its live one-update preflight, but
  none of these detector results prove policy benefit. Do not resume generic
  Vulkan diagnosis.
- Keep active documents and admitted experiments outside `legacy/`. Superseded
  documents belong in their directory-local `legacy/`; rejected or no-longer-
  needed local experiment packages belong under repository-root
  `legacy/experiments/`. Every directory named `legacy` is ignored by Git and
  may not be cited as an active result.
- The old frame-zero Z/P/PS one-update training-path preflights are historical
  wiring evidence only and must be repeated after live handoff. Z executed 364
  exact-zero observations with zero TacSL reads. P and PS each executed 361
  online feature updates and exactly `361 x 54 = 19,494` official patch reads;
  P made zero slip calls and PS made 361 causal slip calls. All three completed
  a real BCPPO optimizer update. The current warm start terminates before the
  CarryBox contact window, so these reports retain zero contact/load/event;
  nonzero bilateral tactile and the lift-gated mass event remain supported by
  the admitted continuous-action full-G1 collectors, not fabricated in the
  one-update rollout. The old formal Z used frozen seeds
  `151014/151015/151016` with the same 4-env, 24-step, 3000-update
  configuration; P and PS remained unstarted. Seeds `151014/151015` passed
  update 500, and each readable
  `model_500.pt` contains the anatomical patch encoder, serious SUGAR
  actor/critic and 58 optimizer-state entries while both trainings continue in
  critic warmup. Both resumed runs have now also produced readable
  `model_1000.pt` files with 42 patch-encoder tensors and 58 optimizer states,
  and have entered the task-reward PPO authority ramp. Treat these only as
  recoverable-training/stage-transition milestones, not as a tactile-benefit
  result. These old checkpoints do not authorize starting P.
- Z seeds `151014/151015` produced readable update-2000 checkpoints and
  entered the final 1000-update steady full-PPO stage. Seed `151015` has now
  completed the full 3000-update loop with `exit_code=0`; the trainer's
  zero-based final file is `model_2999.pt`, containing 59 model tensors and 58
  optimizer-state entries. This is a completed training endpoint, not yet a
  valid mass-adaptation result. Seed `151014` has also completed normally at
  the same `model_2999.pt` endpoint with the same 59/58 state counts. The
  fixed motion-45/1.5x four-profile check on seed `151015`'s checkpoint
  terminates at frames `96/48/201/194`; all four
  profiles still have zero box contact and no mass event. This is evidence that
  the PPO ramp changed survival duration but not yet a valid mass-adaptation
  baseline. Final endpoint evaluation then established the frame-zero
  structural negative described above; do not resume this path.
- The frozen evaluator must restore CarryBox motion 45/frame 0 and refresh the
  official motion-relative command buffer before the first policy observation.
  The corrected single-profile check with the intermediate Z update-1000
  checkpoint reaches frame 63 before robot-pose termination, rather than the
  invalid frame-0 termination from a stale command buffer. It still fails
  before box contact or the mass event, so it validates evaluator startup only
  and is not an endpoint Z outcome.
- A matched frozen-Refiner feasibility diagnostic now separates the mass
  conditions. With the same seed and frame-299 live jump, `1.5x` retains
  bilateral patch contact for 118 of the 121 post-jump frames and continues
  lifting. `3x`, `6x` and `10x` lose bilateral contact at frames 354, 321 and
  315 and fall by 0.213, 0.222 and 0.226 m from the jump height. This proves a
  recoverable mild condition and current-controller failure conditions; it
  does not prove that no stronger action can recover `3x+`.
- The predeclared mass-independent fixed response is also complete. At
  absolute frame 300 it adds symmetric `0.10 rad` inward shoulder targets and
  a small bilateral hip/knee/ankle lowering target, with zero reads of mass,
  jump state, tactile or object state. It does not recover `6x` or `10x`:
  bilateral contact ends at frames 320 and 315 and the objects still fall by
  0.171 and 0.213 m. Report this as an insufficient simple response, never as
  proof that all 29-DoF policies must fail.
- Scheduler job `238128` was externally `CANCELLED by 0` after seed `151014`
  printed update 651; this was neither a training exception nor a voluntary
  allocation release. Its last complete checkpoint is update 500. The run has
  resumed from that file on retained job `238055`: BCPPO stage state was
  reconstructed at update step 501 and the runner started iteration 501. At
  the next complete checkpoint 750, its recorded child was stopped and the run
  migrated to five-day retained job `238250`/`server23`; it restored BCPPO and
  runner iteration 751 with a fixed endpoint of 3000. Seed `151015`
  was subsequently also externally `CANCELLED by 0` after printing update 784;
  its last complete checkpoint is update 750. It has now resumed on retained
  job `238355`/`server07`: the checkpoint restores BCPPO update step 751, the
  runner starts learning iteration 751, and the fixed endpoint remains 3000
  with 2249 updates remaining. That resumed run subsequently completed
  normally at iteration 2999, i.e. all 3000 loop iterations. Seed `151014`
  likewise completed normally on retained job `238250`. Seed `151016` started
  from scratch on the same retained allocation, then was stopped at iteration
  226 after the two completed endpoints proved the shared frame-zero entry
  invalid. Do not count either unsaved 501--651 or 751--784 segment twice, and
  do not start P until handoff-Z produces eligible post-jump endpoints.

## Highest-Priority Full-G1 IsaacLab Object-Demo Reset (2026-08-13)

- H200 is an accepted IsaacLab runtime on this cluster: retained job `231928`
  on `server13` completed the full 660-frame CarryBox collection and rendering.
  Do not reject H200 from a generic support table. The Reflex cluster note is
  a SAPIEN recipe; its empty `DISPLAY` remains applicable, but selecting
  `/etc/vulkan/icd.d/nvidia_icd.json` explicitly is not an IsaacLab fix by
  itself. On 2026-08-13 both explicit-ICD and loader-default IsaacLab camera
  starts failed before scene construction on two `server13` H200s, and a later
  loader-default start failed identically on `server53`, while the same
  physical H200/driver/software environment had completed the earlier run.
  Treat this as a current cross-node Kit/Vulkan runtime failure; do not call
  the H200 architecture unsupported or misattribute a pre-scene crash to TacSL.
  A separate clean Isaac Sim 5.1 runtime with fresh Kit binaries, extension
  caches and official base components also failed in retained job `237668`.
  On 2026-08-14 retained job `238022` then reran the unchanged collector that
  had succeeded for 660 frames in job `231928`; it mounted all 54 patches and
  again failed with `VK_ERROR_DEVICE_LOST` while starting simulation, before
  the first physics or tactile step. This isolates the current blocker to the
  Kit/Vulkan runtime rather than Plan-15 mass, observation or slip code. A
  second physical H200 in retained job `238055` reproduced the same pre-scene
  failure with the force-only collector and with the exact camera/rendering
  command route that had succeeded on 2026-08-11. A complete 25.7 GB copy of
  the Python/Isaac runtime on server13 node-local storage also failed at the
  same `Simulation App Starting` boundary, so shared-filesystem reads are not
  the cause of this Vulkan crash. On 2026-08-14 retained job `238092` supplied
  a third physical H200 on `server38`; the minimal single-GPU SimulationApp
  canary again ended with `VK_ERROR_DEVICE_LOST` before returning from app
  construction. This is now a real cross-node reproduction, not a server13-
  local result. This historical outage was superseded later on 2026-08-14 by
  the successful job `238128` official-AppLauncher run recorded in the active
  Plan-15 section above; do not resume generic Vulkan diagnosis while that
  recovered path remains usable.
- Every active object tactile demo must run in IsaacLab/PhysX with the complete
  SUGAR G1 physically moving the object with its hands. Detached R15 fixtures,
  standalone plates, schematic hands, or kinematically moved sensor supports
  are diagnostics only and cannot satisfy the object-demo goal.
- Each G1 hand must use exactly 27 physical anatomical TacSL patches: palm
  `4 x 3`, plus proximal/middle/distal patches on thumb, index, middle, ring,
  and little fingers. The main H.264 must show the full G1/object world motion
  and both readable 27-patch maps on the same clock.
- Use the existing successful sensorized G1 CarryBox trace and collector as the
  foundation. First finish varied rigid-object pickups and physical failure
  samples, then another rigid shape. Investigate deformable assets only after
  those rigid G1 cases pass numerical and human-visible checks.
- Newton may supply USD/mesh geometry only. Do not start another Newton
  simulation for the active demos. Do not start policy training.
- The previously retained detached dual-R15 cup/block/soft-body runs are
  fixture diagnostics, not completed G1 pickup, failure, or soft-object demos.
  Do not cite them as satisfying this reset.

## Highest-Priority IsaacLab-Only Tactile Demo Expansion (2026-08-12)

- Every new tactile demonstration requested after this reset must execute in
  IsaacLab. Assets may be imported from the cloned `Newton/` repository, but
  the Newton simulator may not be used for these new demonstrations. Existing
  Newton results remain historical evidence only.
- Execute the expansion serially: first complete varied rigid-object pickups
  with large contact areas, then physical rigid-object failure cases such as
  post-lift slip or drop, then another rigid shape. Investigate deformable or
  soft-body pickup only after the rigid cases are complete.
- IsaacLab demonstrations continue to use the official v2.3.2
  `VisuoTactileSensor` and `GELSIGHT_R15_CFG` native fields. Do not replace
  them with rigid-contact proxies, generated taxels, labels, or object state.
- For native IsaacLab deformables, retain the official R15 taxel geometry,
  frames, data contract and TacSL normal/friction law, but query the current
  PhysX `SoftBodyView` collision-tetrahedron boundary instead of a rigid SDF.
  This must be labeled as the local deformable-surface TacSL extension, not as
  upstream v2.3.2 deformable support. A hidden rigid core remains forbidden.
- These are sensor/generalization demonstrations, not policy-training runs.
  Do not start training while this expansion is active.

## Highest-Priority Newton/IsaacLab Universal Tactile Reset (2026-08-11)

- This section supersedes the Plan-13 tactile-training execution queue. The
  only active plan and TODO are
  `PLAN/14_newton_isaaclab_universal_tactile/plan.md` and
  `TODO/14_newton_isaaclab_universal_tactile/todo.md`. Plan 12 remains the
  IsaacLab representation foundation and Plan 13 is read-only history. Do not
  start or resume tactile, zero, RGB, fusion, demo, ICM, or other policy
  training until the user explicitly reopens training.
- The active result must be one reusable native-signal tactile contract that
  works in both the cloned `Newton/` branch and the current IsaacLab tree, on
  CarryBox/box manipulation and at least one non-box scene per engine.
- IsaacLab tactile must come from the official v2.3.2
  `VisuoTactileSensor`/official R15 path and preserve raw taxel position,
  orientation, penetration, signed local-Z force, signed local-XY shear,
  timestamps, and available RGB/depth. Rigid-contact force, contact labels,
  object state, or generated taxels may not replace it.
- Newton tactile must come from solved native `Contacts.force` wrenches and
  native effective contact geometry after solver contact update. A spatial
  grid may conservatively serialize those raw samples, but `kh * depth`, an
  aggregate `SensorContact` wrench, hand-painted taxels, or a synthetic image
  may not replace the solved spatial force field.
- Preserve raw backend samples beside the common raster, signed force
  directions, geometry-fixed patch frames, counterpart identity, sequence,
  source timestamp and elapsed time. Any derived grid must conserve the raw
  force vector. Missing modalities are marked unavailable rather than filled
  with fabricated values; Newton has no native GelSight optical stream.
  Common world quaternions are `xyzw`: reorder official IsaacLab `wxyz`
  explicitly, while preserving the represented rotation. Force and optical
  clocks remain independent even when a scene samples them synchronously.
- Slip detection is causal and tactile-only. It may consume current/past
  signed normal/shear/penetration fields and timestamps. Object motion,
  relative contact velocity, SDF normals, outcome labels and rewards are
  simulator-only evaluation labels and may never enter the detector. Slip is
  separate from ICM, reward and policy training.
- Completion requires real Newton and IsaacLab runtime evidence: box and
  non-box scenes, controlled stick-to-slide intervals, numerical sign/force/
  clock checks, and separately playable synchronized H.264 videos. Unit tests,
  interface smoke tests, plans, or offline plots alone are not completion.
- Use the shortest implementation path and no routine hashes. Keep the
  retained GPU allocations alive and use them for in-scope simulation,
  validation and rendering; completion of a child process never authorizes
  releasing an allocation.

## Highest-Priority No Over-Defensive Design or Routine Hashing Rule

- Strictly prohibit over-defensive design. Use the shortest direct path from
  the stated scientific question to the requested runnable result and visible
  evidence. Do not invent extra gates, proof layers, version ladders, wrapper
  protocols, redundant controls, or defensive abstractions that the method or
  the user did not require.
- Strictly prohibit new routine hash checks, SHA-256 checks, hash manifests,
  and source-hash bookkeeping. Do not calculate or validate hashes merely to
  make an experiment look rigorous. Existing historical hashes may remain as
  provenance, but do not extend that system.
- A hash is allowed only when the user explicitly requests it or when an
  upstream download/package interface requires integrity verification. In
  that exceptional case, keep it local to that requirement and do not turn it
  into another admission gate.
- For the active native-tactile task, rigor means correct official sensor
  signals, correct physics, synchronized success/failure videos, numerical
  sanity checks that answer the actual tactile question, and human-visible
  correspondence. It does not mean accumulating audits or hashes.

## Highest-Priority Retained GPU Allocation Rule

- This rule overrides routine cleanup, task-completion, failure-handling, and
  end-of-report conventions. Finishing a phase is never authorization to end
  its usable GPU allocation.
- GPU allocations are scarce. Never actively release, cancel, or exit a
  retained GPU allocation merely because one command, smoke test, diagnostic,
  render, training segment, or nominal task has finished or failed. Keep the
  allocation after the immediate task completes so that audits, reruns, human
  review fixes, and follow-up modifications can use the same resource.
- Request `salloc` or an equivalent tmux-retained allocation with enough wall
  time to cover the full task plus expected debugging, reruns, audits,
  rendering, and subsequent user-requested modifications. Prefer one long
  retained allocation over repeated short `srun` jobs.
- Run subsequent commands inside the same retained allocation. If a command
  fails, return to its allocation shell, diagnose, patch, and rerun without
  ending the allocation.
- Only the user may authorize voluntarily releasing a usable GPU allocation.
  Otherwise it may end only through scheduler enforcement or an unavoidable
  safety condition. Ending an agent turn, writing a final report, reaching a
  scientific stop gate, or waiting for human review is never permission to
  terminate the allocation or its useful liveness workload. Never report
  routine cleanup as “GPU resource released.”
- While useful in-scope work remains, keep valid GPU utilization above the
  cluster's 30 percent liveness threshold with real project workloads,
  validation, or rendering. Never fabricate meaningless load merely to occupy
  the GPU.
- Never send an undirected terminal `Ctrl+C` to a tmux pane whose lifetime is
  the retained Slurm allocation. A foreground child that appears active may
  already be inside shutdown, allowing the signal to reach and cancel the
  allocation step itself. Launch long children as their own `setsid` process
  group in the retained shell, write the exact child PID/PGID to the run
  record, and leave the allocation shell at its prompt. If termination is
  required, target only that recorded child process group with `kill`; first
  verify that the target is not the allocation shell or `srun`. Generic tmux
  `C-c`, pane exit, shell exit, and `scancel` remain forbidden without explicit
  user authorization.

## Highest-Priority Native Tactile Training Extension (2026-08-11)

- The user's active objective now extends the completed representation work
  into testing whether the current tactile signal helps training and how it
  should be fused into the serious SUGAR policy. This section supersedes the
  Plan-12 prohibition on training. Plan 12 remains the sensor/visualization
  foundation; the active execution queue is Plan/TODO 13.
- The first comparison is no-RGB and no measured-object-state at the actor.
  Both arms retain robot proprioception, last action, motion phase, and the
  official deployable 35-D Tracker command. A training-only privileged critic
  and frozen official Refiner teacher are allowed and must never be described
  as deployed actor inputs.
- Compare the current four-frame, bilateral 27-patch native TacSL normal and
  signed-XY-shear tensor against an exact-zero, no-sensor-read arm with the
  same model, optimizer, task, teacher, seed, physics, and tensor width. Do not
  add simulator contact velocity or rigid-contact proxy to this first tactile
  experiment.
- Reuse the existing serious SUGAR `SpatialTactileEncoder`, unchanged SUGAR
  `512/256/128` actor MLP, official Refiner checkpoint, and repository-native
  BCPPO teacher-student schedule. Do not substitute a small local MLP or an
  offline classifier.
- Execute serially: live one-update tactile preflight, matched zero preflight,
  matched training/evaluation, then RGB and RGB+tactile fusion. A gradient or
  training-loss difference proves only signal use; tactile benefit requires a
  frozen-policy physical behavior difference on matched CarryBox conditions.

## Highest-Priority Native Tactile Reset (2026-08-10)

- This section is the retained sensor/visualization foundation and is
  superseded for execution by the 2026-08-11 training extension above. It had
  superseded every older experiment queue, including the 2026-08-03 Plan-11
  reset below. The only active plan and TODO are now
  `PLAN/legacy/13_native_tactile_training_fusion/plan.md` and
  `TODO/legacy/13_native_tactile_training_fusion/todo.md`.
- User correction on 2026-08-10: the final visualization must run on the
  sensorized SUGAR G1 CarryBox scene and show both complete hands. Detached
  dual-R15 fixtures are diagnostics only and cannot satisfy the goal. Each
  hand must show twelve palm patches plus thumb, index, middle, ring, and
  little fingers with proximal/middle/distal patches, all arranged readably in
  the same main video.
- The current objective is solely to build this reusable IsaacLab-native
  whole-hand tactile representation and synchronized visualization for both
  successful and failed CarryBox grasps. Do not study whether tactile improves
  training. Do not resume demo following, ICM/Curiosity, mass OOD, contact
  velocity, or policy training.
- Use only the official
  `isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor` and official
  `GELSIGHT_R15_CFG` for tactile output. Preserve raw per-taxel penetration,
  local normal force, signed local XY shear, GelSight RGB/depth, poses, and
  timestamps. Rigid-contact proxies, hand-written taxels, labels, aggregate
  wrenches, or object state may not replace or enter the tactile
  representation.
- Contact-object relative velocity, SDF normals, object pose, and outcome
  labels are simulator-only diagnostic fields. Keep them separate
  and never present them as deployed tactile feedback.
- Successful and failed grasp samples must use the same generic collector,
  tensor contract, derived representation, clock, and fixed rendering scales.
  Both must contain genuine native tactile contact. The failure must arise
  physically, not by zeroing, shuffling, or fabricating sensor data.
- Required human evidence is separate synchronized H.264 videos for a
  successful grasp and physical failures. Each must show continuous SUGAR G1 CarryBox world
  motion across the top and both complete anatomical hands below it. All 27
  patches per hand must remain distinct: palm `4 x 3`, five finger columns,
  and three segments per finger. Explicit human review is mandatory before
  completion; do not add an independent reconstruction layer to this visual
  task.
- Plans and TODOs 04--11 are read-only under `PLAN/legacy/` and `TODO/legacy/`.
  Their artifacts may be curated as history but their queues may not resume.
- Across the historical categories (1) tactile effect on training, (2) demo
  following, and (3) original ICM/Curiosity, retain at most five experiment
  packages in total. Active native-tactile representation evidence is outside
  that five-package quota, but must also be pruned to distinct scientific
  cases rather than version ladders. Move other artifacts into the single
  `/public/home/yanhongru/Curiosity_archive` tree; do not create new archive
  roots and do not discard provenance.

## Superseded Legacy Experiment Record (2026-08-03)

- This entire section is a read-only record of the former Plan-11 queue. Every
  statement below that calls a Plan-11 arm or work package "active" is
  superseded by the 2026-08-10 native-tactile reset above and must not trigger
  execution.

- The user stopped every older unfinished experiment queue and replaced it
  with the five-work-package mainline in
  `PLAN/11_demo_tg_icm_mass_ood_contact_velocity/plan.md` and
  `TODO/11_demo_tg_icm_mass_ood_contact_velocity/todo.md`. This section has
  priority over older Plan 06/07/08/09/10 execution queues. Older artifacts
  remain read-only evidence and implementation inputs; do not resume their
  training, parameter sweeps, version ladders, or failed follow-ups.
- Work package 1 is a matched online demonstration-conflict experiment using
  the frozen serious 11.9M selected-demo predictor. Compare correct demo,
  deliberately goal-conflicting official demo, zero demo, and task-only arms.
  Keep task reward, policy initialization, seeds, physics, optimizer, and
  update budget matched. Render the wrong-demo and zero-demo final behavior
  with synchronized task reward, predicted demo loss, demo feedback, box
  motion, and contacts. Do not substitute an offline prediction plot for the
  requested policy behavior.
- Diagnose the existing zero-tactile neutrality with contact-conditioned
  occlusion, shuffling, gradient/activation, data-coverage, and normalization
  tests. Preserve the frozen architecture. A retrain is allowed only with the
  same serious architecture and a predeclared contact-balanced dataset; no toy
  predictor is allowed.
- The completed frozen-predictor visualization must remain the concrete
  explanation for the old aggregate tie: all `4,693` motion-disjoint test base
  rows are shown once, only `48` contain any four-frame tactile history, only
  `29` contain current contact, and the split contains zero right-hand and zero
  bilateral tactile histories. Full and explicit-zero predictions are bitwise
  equal on the other `4,645` rows, while all `48/48` supported rows change.
  Keep both fully decoded H.264 records and their independent `15/15` audit;
  they diagnose sparse withdrawn dual-palm data and never substitute the
  requested final correct/wrong/zero/task-only policy rollouts.
- The stale Plan-11 V1 online demo-conflict draft is a frozen negative because
  it combined rejected H2R1 dual-palm input with an incomplete mid-trajectory
  restore.  The active V2 is an explicitly non-tactile four-arm control: every
  tactile-derived actor, critic, ICM, slip, strategy, external-reward, and
  predictor channel is exact zero at its declared width; no tactile sensor is
  read and no rigid-contact proxy may replace it.  Restore official source
  state 103 and preceding source action 102, and require the first live frozen
  Refiner action to match source action 103 within `2e-6` maximum absolute
  error.  Correct, motion-96 wrong, zero-demo, and task-only arms otherwise
  remain matched for serious actor, official teacher/SMP, original ICM, task
  reward, physics, initialization, seed, and 512 updates.  Since direct tactile
  failure does not exist in this control, teacher authority remains fixed; do
  not invent a proxy release.  This comparison may establish demo-conditioning
  policy influence only, never tactile usefulness or whole-hand admission.
- The V2 runner's generic native-authority postcheck contains one condition
  that is contradictory for this explicit-zero control: it asks manipulation
  residual authority to span `0.05--1.0`, while the same run correctly has no
  tactile failure, fixed teacher coefficient `1.0`, and both residual scales
  fixed at `0.05`. Never edit a completed runner proof. The independent
  `POSTCHECK_ADMISSION.json` may admit a 512-update arm only if this exact
  legacy span condition is its sole failure and all positive config,
  no-sensor, fixed-authority, original-ICM, frozen-predictor and checkpoint
  checks reconstruct. Frozen evaluation must bind both the original proof and
  this admission record.
- All four V2 demo arms have completed the exact
  `1/128/512` schedule and pass their separate positive postchecks `19/19`;
  the immutable reward-arm runner proofs retain only the contradictory legacy span check
  as false. The zero arm has `242,324` valid transitions and `512` independent
  original-ICM updates with no tactile read or proxy. Their final policies
  differ in `17/45` policy tensors (L2 `3.3325`), which proves conditioning
  entered learning. The complete matched frozen evaluation passes `14/14`,
  and all four separate 191-frame H.264 videos full-decode. The predeclared
  wrong-demo conflict criterion is not met: alignment improvement versus
  task-only is `-0.06581` with 95% CI `[-0.13156, 0.01769]`, while task-reward
  change is `+0.14458` with 95% CI `[-0.05557, 0.27302]`. Report that demo
  conditioning affects behavior but this run resolves neither wrong-demo
  preference at task cost nor task-reward dominance. All displayed profile-0
  arms terminate by unsafe fall at transition 189.
- On 2026-08-05 the user reopened the demonstration experiment as the highest
  priority because the old videos did not show what each demo was and did not
  make a visible four-arm trajectory difference. Treat V2 as an underpowered
  negative, not a completed positive demo-following result. The exact correct
  reference is official SUGAR `data_045`/motion 45 (`660` frames); the exact
  conflicting reference is official SUGAR `data_096`/motion 96 (`531`
  frames); zero demo is an exact normalized tensor with no physical reference
  trajectory; task-only loads predictor telemetry but adds no demo feedback to
  PPO. The independently reconstructed online normalized MAE is
  `0.52156/0.47884/0.44400` for correct/wrong/zero, whereas PPT Question 5's
  held-out offline values are `0.18727/0.38850/0.44809`. They use the same
  normalized error definition on different trajectory distributions, and the
  offline ranking does not transfer online. New H.264 comparison videos must
  always show the physical reference (or explicit zero card), actual
  conditioned rollout, task-only rollout, true future alignment, predictor
  estimate, and physical trajectory difference together.
- The active V3 authority rework is frozen before observing V3 behavior. It
  retains the serious `11,907,912`-parameter predictor, official SUGAR actor,
  frozen Refiner, official TinyMDM/SMP, original ICM, task reward, physics,
  optimizer, `512` updates, exact-zero tactile/no-read control, and common
  random numbers. The only shared authority changes are residual scale
  `0.05 -> 0.5` and potential scale `eta 2 -> 10`; eta 10 was selected from V2
  reward telemetry to make the demo channel comparable to the approximately
  `0.14` mean non-demo policy reward, not from V3 outcomes. Run exactly one
  arm at a time in order `correct_demo`, `wrong_demo`, `zero_demo`,
  `task_only`. A predicted reward difference cannot pass: correct must reduce
  reconstructed motion-45 alignment versus task-only, wrong must reduce
  reconstructed motion-96 alignment versus task-only, and task outcome must
  be reported separately. Zero remains an ablation, never a followable demo.
- The V3 correct-demo arm completed all 512 updates on 2026-08-05 and passes
  its independent positive admission. The immutable generic runner proof is
  deliberately retained with exactly two expected V2-only checks false:
  post-release residual authority span and eta-2 reward reconstruction. The
  independent V3 audit instead reconstructs fixed residual scale `0.5`, eta
  `10`, the frozen predictor, exact-zero/no-read tactile, official first
  action, all three checkpoints, and all 512 original-ICM updates. Wrong-demo
  then completed all 512 updates and passed the same independent admission
  with the specified official motion-96 condition. On 2026-08-05 the user
  cancelled task-only and requested immediate correct/wrong visualization.
  Zero-demo was interrupted after its update-1 checkpoint and remains paused;
  never call it complete. The active deliverable is exactly two H.264 movies,
  each showing the exact official numeric demo and its corresponding frozen
  policy rollout simultaneously. Without task-only, report direct behavior
  and demo correspondence only; do not claim an improvement over task-only.
- The requested two-arm video endpoint now passes independent audit. The
  correct/motion-45 frozen rollout has 140 frames and terminates `unsafe_fall`;
  the wrong/motion-96 rollout has 66 frames and also terminates `unsafe_fall`.
  Their actual box-relative XY displacements are respectively
  `[-0.2906,+0.5736]` and `[-0.5896,+0.2666] m`, which do not follow the
  official demo displacements `[-2.3351,+0.8095]` and
  `[+2.0019,+1.6067] m`. Each 1280x720 H.264 places the exact official numeric
  demo left and actual frozen IsaacLab behavior right; both fully decode and
  all source hashes/displayed numbers reconstruct. This is a negative direct-
  behavior result. Nonzero predicted/demo feedback does not establish demo
  following, and the cancelled task-only arm leaves no relative baseline.
- The later unrelated-demo intervention is a fixed-teacher experiment, never
  teacher-free: CarryBox remains the task, while both the official Refiner
  teacher reference and internal selected-demo condition are official KickBox
  `data_021`, with teacher coefficient exactly `1.0`. Do not rerun the existing
  correct arm. The unrelated arm is frozen at update `1216` after nineteen
  exact 64-update endpoints. All `380/380` frozen evaluation profiles terminate
  by unsafe fall with zero success; every endpoint has zero bilateral contact
  frames, update 1216 has zero lift, and canonical env 0 falls at step 7.
  The isolated `1.405 mm` maximum at update 1152 is not a grasp or lift. Every
  endpoint has its own two-panel H.264 showing the exact KickBox input beside
  actual CarryBox behavior. Do not extend this identical arm without a new
  scientific question.
- In the frozen 60-environment demo evaluation, gate reproduction of official
  source action 103 only on canonical source env 0 and keep the unchanged
  `2e-6` tolerance. Replicas translated to other world origins have measured
  float32 local-observation cancellation up to `1.89245e-6`; incorrectly
  maximizing their action error against the one unshifted source action gives
  `2.38419e-6`, while canonical env 0 is `1.01328e-6`. Archive all per-env
  observation/action errors and require the independent auditor to reconstruct
  the canonical gate. Never hide the replica drift or relax the canonical
  threshold.
- The user froze Plan-11 execution to a serial order on 2026-08-04. Finish the
  complete demo four-arm result and its four behavior videos first; then resume
  Tactile Genesis; then run the ICM policy-credit, 1--10/13 kg, and contact-
  velocity policy experiments one at a time. Do not advance a later work
  package in parallel. Demo, Tactile Genesis, and original-ICM policy credit
  were complete before the 2026-08-05 Demo rework. The user has now made the
  V3 Demo authority rework the only active experiment; the partially started
  mass arm is paused with its update-1/update-128 checkpoints preserved and
  may not resume until all four V3 arms, matched frozen evaluation, independent
  online-target audit, and reference-visible H.264 videos finish. Retained
  five-day Slurm allocation `221922` on `server59` remains active and must not
  be released. For TG provenance, the hash-bound resume
  started at update 4001 rather than repeating update 4000; unsaved telemetry
  rows 4001--4599 from the interrupted process remain archived but excluded.
  The corrected exact Stage-2 run is complete at update 5999. Final checkpoint
  SHA-256 is `be2100bc0cf926834fb8423cb36405346ca4c54cf0041bd261bb5b9e0b66eb96`;
  its independent audit passes `25/25`, optimizer step is `360000`, all 12
  released teacher tensors remain bitwise unchanged, and the checkpoint-bound
  trace contains exactly 6000 updates across the pause/resume boundary. The
  final three-seed/eight-environment frozen evaluation passes `125/125` and the
  predeclared native tactile-usefulness rule: full wins `20/24` against zero
  and `23/24` against shuffled tactile. All arms still have zero strict task
  success, and full beats the state-teacher upper bound on only `6/24`; report
  this as lower orientation error while held, never successful manipulation.
  The final Stage-3 no-learning import passes `17/17` plus independent `16/16`.
  Keep the allocation and all checkpoints alive.
- Work package 2 continues official Tactile Genesis. Four-finger support is a
  physically admissible result; palm or thumb inactivity is not by itself a
  failure. Audit sensor-local/world transforms and the sign convention for
  force on sensor versus force on object before judging the visualization.
  Keep `KinematicTaxel` force/torque and `ElastomerTaxel` marker displacement
  separate, and preserve raw vectors behind every rendered arrow or color.
  Its mandatory final human-review format is the PPT "Question 9" layout: the
  upper half is one continuous articulated bilateral CarryBox world video and
  the lower half is the synchronized left/right anatomical hand map. Produce
  separate H.264 files for Kinematic force/torque and Elastomer signed marker
  displacement, preserve inactive regions, and bind every frame to the same
  source/action clock. A native XHand small-object view, a one-hot layout
  audit, an aggregate plot, or the old historical negative PPT videos cannot
  substitute for this CarryBox presentation.
  The fresh required presentation now exists under
  `carrybox_question9_v1/videos/` and passes independent `14/14` video audit.
  Its measured result is negative for complete coverage: Kinematic is bilateral
  on `55/120` frames, Elastomer on `107/120`, and Elastomer reaches
  `10.361 mm`, beyond the declared `5 mm` stability bound. Preserve this as the
  completed CarryBox coverage experiment; do not convert it into a positive
  sensor-admission claim.
- The official native Stage-2 path is now fixed as the released
  `in_hand_repose-xhand1` state teacher distilled into the released
  `tac_mlp` student with `tactile_convrnn`, proprioceptive `rnn`, and
  `med-hand/force_torque`. Its student sees `36-D` proprioception, `13-D`
  goal, and `5970-D` spatial tactile history; its teacher sees the released
  privileged `79-D` state. The missing `pt_tnn` import resolves only through
  the same-organization official `neuroagents-lab/PyTorchTNN` repository at
  commit `50713d9800c5659ace4916af23c1273497451d27`. Never hand-write a
  replacement recurrent cell. The first exact one-update path passes its
  optimizer audit but is withdrawn as spatial evidence: the released
  `postprocess_generic` emits probe-major force then torque blocks, while
  `TactileLayout` reshapes them directly as channel-major `F,H,W`. The frozen
  sentinel audit finds `0/199` exact physical-probe vectors and a median of
  `15` source probes mixed into one encoder grid cell. Never reuse the
  unadapted one-update or interrupted formal checkpoints. The only admitted
  repair is the separately hash-bound serialization adapter
  `plan11_tg_force_torque_layout_adapter_v1`, which changes no value, physics,
  model, history, or loss and restores `199/199` probe vectors bitwise. A new
  corrected one-update gate now passes independent `30/30`, including runtime
  adapter identity, all tactile-branch gradients, frozen teacher, and checkpoint
  serialization. The completed exact 6000-update Stage-2 run used only this
  corrected path and did not resume either unadapted checkpoint. The associated
  `199`-frame H.264 moving-one-hot audit is the human-readable spatial evidence:
  the released bright cell is misplaced on `172/199` frames, while the strict
  full-vector audit remains `0/199` released versus `199/199` corrected. The
  four-arm frozen evaluator may record tactile only after cloning it at `s_t`
  before `env.step()`; direct live-sensor reconstruction must equal that tensor
  bitwise on every frame and the world renderer must show env0 only. The current
  two-frame/one-update all-environment structural preflight passes `43/43`,
  with hash-bound collector/renderer/auditor implementations, exact official
  anatomical order/probe counts and recurrent reset masks, and with usefulness
  explicitly unevaluated. It archives all eight behavior rollouts per seed
  while rendering env0 only; the final three-seed result uses all 24 paired
  rollouts rather than silently judging only three videos.
  The post-training launcher must recheck the current collector, renderer,
  auditor, and spatial-layout-adapter hashes against that passed preflight;
  a numerically passing stale preflight may not authorize modified code.
  The actor consumes five force/torque substeps per taxel, so videos must show
  the exact
  five-substep force envelope (maximum `|F|` and the vector at that substep),
  not only the final substep; otherwise valid contact can appear falsely empty.
- The official Stage-2 to Stage-3 handoff is also frozen: pass the completed
  distillation file only as `actor_checkpoint` with
  `load_actor_distribution=false`. The final no-learning runtime audit passes
  `17/17` and its separate source/file audit passes `16/16`: every
  non-distribution student tensor enters the Stage-3 PPO actor
  bitwise, the optimizer is empty, the iteration remains zero, the actor sees
  only `proprio + goal + tactile_sensors`, and mass remains available only to
  the privileged critic through `priv_obj_props`. The final import is bound to
  `model_5999.pt` hash
  `be2100bc0cf926834fb8423cb36405346ca4c54cf0041bd261bb5b9e0b66eb96`.
  It validates the handoff only and does not authorize or claim Stage-3 PPO
  training.
- Work package 3 makes original ICM behavior concrete. Visualize and audit the
  forward feature-prediction error for familiar, newly encountered,
  repeatedly learned, action-shuffled, next-state-shuffled, and uncontrollable
  noise transitions. ICM remains independent self-supervised discovery;
  success, slip, weight, reward conflict, or a new grasp may not be relabeled
  as curiosity.
- Its completed visitation comparison is a non-tactile exact-zero control.
  Both matched arms completed 512 updates with the complete serious
  original-ICM architecture, exact `12000-D` zero tactile width, no sensor
  read, and no proxy; only the ICM coefficient entering PPO differs (`1.0`
  versus `0.0`). Their initial policy, first-rollout actions, and update-1 ICM
  learner subtree are bitwise equal. The independent frozen evaluation at
  updates `1/128/512` passes `21/21`. At update 512, policy credit changes
  action and box trajectories in `20/20` matched profiles, and the shared
  familiar frozen ICM reports higher forward error in `19/20`
  (`p=4.01e-5`) and greater support distance in `18/20` (`p=4.02e-4`). It does
  not establish broader visitation: occupied box-state cells are
  `177.4` versus `174.4`, but the predeclared sign test gives `p=0.8145`;
  feature/action cell counts and task reward are also unresolved, and every
  profile has zero success. The two separate profile-0 H.264 videos pass full
  decode and end in unsafe fall. Report policy influence and more surprising
  visited transitions, never broader exploration, task improvement, tactile
  usefulness, or a successful new strategy.
- Work packages 4 and 5 train with a stratified 1--10 kg CarryBox mass domain
  and test the frozen policies at the OOD 13 kg condition. Mass is never an
  actor or ICM input. Compare tactile plus contact velocity, tactile without
  contact velocity, zero tactile, and state-only controls. Contact velocity
  must come from the released simulator's contact-point relative-velocity
  callable/output, not from hidden mass or a hand-written success oracle.
  The temporal sensor/logging layout is `[history,hand,channel,row,column]`,
  while the convolutional actor requires
  `[hand,history,channel,row,column]`. The separately audited actor adapter
  must perform that explicit permutation before flattening; a same-size direct
  reshape is forbidden because it interleaves hands and history frames.
  The staged whole-hand extension adds the mandatory patch axis rather than
  collapsing it: logging is
  `[history,hand,27 patches,6 channels,20,25]`, actor serialization is
  `[hand,history,patch,channel,row,column]`, and the existing serious SUGAR
  spatial encoder receives exactly `648` channels per hand. Its unregistered
  CPU contract passes `32/32`; the separately declared full, velocity-zero,
  exact-zero/no-read, and branch-removed state controls pass a static `16/16`
  config audit. A separate no-learning live anatomical-27 scene read now
  resolves all 54 real sensor objects and passes `32/32` tensor checks for the
  exact `648,000` width, live field shapes, cache, causal shift, serialization
  and controls. Its static reset has zero active taxels, so it is not active-
  contact or tactile-usefulness evidence. A separate motion-45 source-103
  snapshot now passes the active callable sub-gate with bilateral `160/562`
  active taxels, six active patches, `0.05887888 m/s` maximum active speed,
  exact inactive zeros and `7.45e-9` raw-to-actor error. The next recorded
  action immediately terminates and auto-resets the anatomical scene, so
  contact retention remains failed and this cannot be called a continuous
  grasp. Both the class and groups explicitly
  remain unregistered and unauthorized for policy execution. Do not add them
  to the task registry, launch a policy with them, or call them a
  tactile/mass-policy result until the immutable whole-hand sensor and human-
  review gates pass.
- For this explicitly authorized Plan 11 experiment only, official Tactile
  Genesis spatial signals and its contact-relative velocity may be consumed by
  a policy before the separate TacSL 27-patch standard passes. This does not
  waive or satisfy the TacSL standard and may not be called TacSL, GelSight,
  hardware tactile, or sim-to-real evidence. Every actor-visible channel and
  every simulator-only diagnostic must be listed separately.
- Completion requires runnable code, exact configs, retained-allocation run
  records, independent reconstruction, numeric comparisons, and H.264 videos.
  Writing a plan, passing an interface smoke test, or producing an offline
  sensor plot is not completion.

## No Degraded Placeholder Model Rule

- Never write downgraded placeholder MLP/VAE/Transformer/world-model
  implementations and present them as T-Rex-style, XIRL-style,
  RoboCLIP-style, VQ-VAE-style, SUGAR-style, SMP-style, ICM-style, or
  world-model progress.
- For SUGAR work, use the official repository, official released data,
  official descriptions, official checkpoints, official task registration, and
  the matching IsaacLab stack. Only write adapter or cluster glue needed to run
  that official code in this workspace.
- If official weights, code, assets, or runtime requirements are unavailable
  or incompatible, record the blocker exactly. Do not silently substitute a
  smaller homemade model or local controller to make an experiment run.
- Any reduced run must be explicitly labeled as a smoke test or diagnostic and
  must still use official SUGAR code and official SUGAR assets.
- For SMP, use the official MimicKit `TinyMDMModel`, EMA, scheduler, ensemble
  SDS, adaptive normalization, and GSI source wherever the representation
  permits. Do not write a smaller local diffusion model and call it SMP.
- For curiosity, preserve the original ICM forward/inverse self-supervised
  learning formulation. ICM rewards not-yet-predicted controllable
  transitions; task success, lift progress, slip reduction, contact counts,
  or strategy-descriptor distance are not ICM and may not be relabeled as it.
  PPO/A3C is the policy optimizer, not the definition of curiosity.
- Never present contact labels, thresholded contact forces, aggregate body
  wrenches, object state, or hand-written toy taxel arrays as tactile progress.
  Tactile work must use the official IsaacLab/TacSL sensor path described below
  and preserve its spatial pressure/shear signal.
- Whole-hand tactile is governed exclusively by
  `DOCS/legacy/sugar_whole_hand_tactile_non_degradation_standard_20260729.md`.
  The raw topology is frozen at 27 physical load-bearing elastomer patches per
  hand, 20 x 25 normal plus signed-XY-shear taxels per patch, and one
  geometry-fixed official R15 RGB/depth palm module per hand. Never reduce it
  to dual palms, a whole-hand hull/atlas/blob, collision-neutral or shadow
  sensors, proxy contact, generated taxels, interpolated gaps, aggregate-only
  plots, or still images. No policy/reward/slip use is allowed until every
  controlled, optical, correspondence, force-balance, continuous CarryBox,
  held-out-physics, independent-audit, video, and explicit human-review gate
  in that standard passes together. A failure is a blocker, not permission to
  relax the topology or threshold.

## Superseded Legacy Workspace Scope

The material in this section predates the 2026-08-10 native-tactile reset. It
is read-only method provenance, not an active execution queue. The only active
plan and TODO are Plan/TODO 12 declared above.

- Ideas 06--09 remain historical notes under `IDEA/`.
- Plans 04--11 are under `PLAN/legacy/`.
- TODOs 04--11 are under `TODO/legacy/`.
- The former completion ledger, whole-hand standard, and from-scratch reset
  protocol under `DOCS/` are historical evidence only for the active Plan-12
  task.
- Official SUGAR, IsaacLab, MimicKit/SMP, original ICM, T-REX, XIRL,
  RoboCLIP, and Tactile Genesis source trees remain available as pinned method
  sources or dependencies. Their presence does not reactivate old training.
- Active scripts: `scripts/sugar/`.
- Active mainline source trees at the workspace root:
  - `SUGAR/`
  - `IsaacLab/`
- Read-only official Tactile Genesis sources:
  - `external/tactile-genesis/` at commit
    `de2bcc998dce45aaac93c6817912380c5954ab38`;
  - `external/PyTorchTNN/` at commit
    `50713d9800c5659ace4916af23c1273497451d27`, used only as the exact
    same-organization `pt_tnn` dependency imported by the official temporal
    tactile encoder. The repository currently contains no license file, so do
    not redistribute or generalize its licensing status.
- Read-only official SMP upstream source:
  - `MimicKit/` at commit
    `2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69`.
- Read-only original ICM source:
  - `external/noreward-rl/` at commit
    `3e220c2177fc253916f12d980957fc40579d577a`.
- Read-only official T-REX source:
  - `external/ICML2019-TREX/` at commit
    `44f92b61ca6c79ac22d468382d4f2fbee164fb7a`.
- Read-only official XIRL source:
  - sparse `external/google-research-xirl/` at audited current commit
    `ec7c3d346277b737bc2decffcd1b533d4b7ec105`; inspected core method paths are
    semantically equivalent to official CoRL-release commit
    `2c5e1990bd60f2fddddb8604396ddce51b2d60ff`.
- Read-only official RoboCLIP source:
  - `external/RoboCLIP/` at commit
    `2d3f779033f1f3adf307a64080742e158caafe67`, with pinned
    `S3D_HowTo100M` submodule
    `b8cd0bbfd16fe41629d1b15e0cf384d75f56101a`.
- Curated reproduction outputs:
  `experiments/sugar_reproduction/outputs/final/`.
- Active reproduction record: `DOCS/sugar_carrybox_reproduction_full_record.md`.
- Active local report:
  `experiments/reports/curiosity_sugar_full_status_20260723.md`
  (ignored; never commit or push it).
- Active workspace-cleanup record:
  `DOCS/legacy/curiosity_workspace_cleanup_20260729.md`.
- Active dependency cache: `external/wheelhouse` for local dependency cache
  only. `external/noreward-rl`, `external/ICML2019-TREX`, and
  `external/google-research-xirl`, and `external/RoboCLIP` are pinned
  read-only method sources, not dependency caches.

`SUGAR/` and `IsaacLab/` must remain at the workspace root because both are
active mainline source trees, not external baselines. Curated checkpoints,
datasets, visualizations, videos, audits, and stop records belong under
`experiments/sugar_reproduction/outputs/final/` and are local-only. Transient
new-run logs may be created below `experiments/sugar_reproduction/`, but must
not be treated as final artifacts. Transitional symlinks at `external/SUGAR`,
the legacy IsaacLab link under `external/`, and `SUGAR/outputs` remain for
historical commands and the prebuilt environment's editable installs; new
scripts and docs must use the canonical root paths above. The editable-install
finders currently record old absolute paths below `external/`, so do not remove
those compatibility links until that environment is rebuilt or reinstalled in
an approved compute allocation.

## Highest-Priority True Whole-Hand Contact Reset (2026-07-31)

- Plan 10 now owns the physical hand/contact repair. Plan 06 remains the
  SUGAR/SMP/ICM research objective and frozen-control record, but it may not
  launch another fixed-rubber-hand TacSL installation/calibration sequence.
  Do not create V90/V91-style successors to V89 as an alleged solution to
  sparse CarryBox load.
- The frozen control is official SUGAR G1-29 with its two single-rigid-body
  rubber hands, plus the retained V83/V87/V88 evidence. It is valid for
  official SUGAR behavior comparison, but it cannot establish independently
  conforming five-finger mechanics.
- The primary repair arm is the official Unitree G1-29 plus Inspire articulated
  five-finger hand from `unitreerobotics/unitree_sim_isaaclab`, with the
  separately licensed official Unitree USD asset package. Runtime USD readback,
  not source-code expectations, must prove the exact body, joint, limit,
  collision, mass, and actuator topology. The official MIT BrainCo RevoLab
  Revo3 hand is a secondary source/fallback only; the unlicensed
  `brainco-description` assets may be inspected but not integrated.
- The primary high-fidelity deformable-contact source is the official MIT
  Taccel Warp-IPC/ABD implementation. Use its released arbitrary-URDF robot,
  joint, tetrahedral gel, contact, marker, depth, and RGB paths. Do not
  hand-write a toy soft skin, distribute PhysX wrench values into taxels, or
  claim that Tactile Genesis marker history deforms the rigid-body physics.
  TacEx and IsaacIPC remain paper-only candidates unless usable official code
  and an explicit license are actually available.
- Execute without learning in this order: exact source/license/hash and asset
  acquisition audit; IsaacLab v2.3.2 asset import/runtime topology audit;
  bilateral palm-plus-thumb-plus-four-finger reachability; actual box support,
  lift, hold, and raw object-wrench balance; official Taccel soft-gel response
  on the same articulated link topology; synchronized bilateral world/contact
  video; then the complete immutable tactile standard and human review.
- The first two Plan-10 gates now pass. The exact `1,305,090,539`-byte Unitree
  asset archive and all seven selected G1-Inspire files are hash-bound; live
  runtime readback has 53 joints, 65 bodies and 12 finger joints per hand with
  five digit groups on both sides. The accepted foundation movie uses a
  declared gravity-disabled, root-held visualization only and passes the
  independent 19/19 topology/trace/H.264 audit. It proves articulation and
  visibility, not standing, CarryBox mechanics, tactile sensing or balance.
- The retained fixed-rubber-hand V87 control is a negative force-balance
  boundary. Its reviewed middle frame loads only 6/54 distal-finger patches,
  with no palm center or thumb. Raw PhysX dynamic force/torque residual medians
  are `0.1532/0.0296`, but integrated TacSL residual medians are
  `3.8881/0.4802`, with zero qualifying quasi-static frames. Never report that
  tactile force balance as satisfied merely because the world trajectory is
  complete or raw PhysX contacts explain the dynamics.
- The next allowed runtime work is a no-learning articulated G1-Inspire
  bilateral palm/thumb/four-finger reachability and real-box support/lift/hold
  gate. Do not begin policy integration, training or another sensor display
  format before same-run contact anatomy and object force/torque close.
- A broad-looking render is never enough. The positive dense-grasp gate needs
  physical load and spatial soft-gel response on palm, thumb, index, middle,
  ring, and little anatomical groups on both hands in the declared hold
  interval, plus object force/torque closure. It does not require inventing
  load on every patch or filling legitimate unloaded gaps.
- Cross-engine pose replay may be a clearly labelled diagnostic, but it cannot
  satisfy same-run CarryBox mechanics or tactile admission. If the exact
  official articulated asset cannot execute in the selected soft-contact
  engine, record the compatibility blocker; do not replace it with a homemade
  hand, a kinematic animation, or generated tactile values.
- Policy, slip, learned reward, SMP, ICM, CHORD, recovery, and alternative-
  strategy experiments stay closed until Plan 10's no-learning physical and
  tactile gates pass and the user has inspected the synchronized videos.

## Experiment Git Exclusion Rule

- The entire root-level `experiments/` tree must remain ignored by Git.
- Never stage, commit, or push any file or directory below `experiments/`,
  including reports, logs, checkpoints, datasets, videos, and visualizations.
- Never use `git add -f`, a `.gitignore` negation rule, or any other mechanism
  to force `experiments/` content into Git history.
- Keep experiment artifacts on the shared filesystem. Put any concise result
  summary that must be version-controlled under `DOCS/`, `PLAN/`, or `TODO/`
  instead of `experiments/`.
- The experiment-root whitelist is exactly `sugar_reproduction/`,
  `sugar_smp_exploration/`, `sugar_demo_reward/`,
  `sugar_chord_paper_reproduction/`, and `reports/`; see
  `experiments/README.md`. Do not accumulate failed, invalid, repeated,
  timestamped, or human-review experiment roots beside them.
- `/public/home/yanhongru/Curiosity_archive/` is the single archive root.
  Never create another top-level `Curiosity_archive_*`,
  `Curiosity_failed_*`, `Curiosity_invalid_*`, `Curiosity_legacy`, or
  `Curiosity_superseded_*` sibling. Preserve original directory names below
  the single archive when provenance matters.
- The retained CHORD package is the positive public-paper-formula
  representation result. CHORD itself was not rejected; its old policy-effect
  runs were archived because their tactile installation/input evidence was
  later rejected.

Old dense-tactile Curiosity materials were archived outside the repository at:

```text
/public/home/yanhongru/Curiosity_archive/workspace_siblings/Curiosity_archive_20260702_pre_video_guided_carrying/
```

Old non-SUGAR plans, TODOs, experiments, logs, scripts, docs, top-level
artifacts, and external repos from this workspace are stored outside the
repository under:

```text
/public/home/yanhongru/Curiosity_archive/workspace_siblings/Curiosity_legacy/20260712_pre_sugar_workspace_cleanup/
```

The root-level `legacy/` path must remain absent and ignored. Never move this
archive back into Curiosity, and never stage, commit, or push any legacy
content. The former `Curiosity_legacy` sibling now lives under the single
archive root and is not repository source.

Do not treat archived Curiosity, AGILE, MuJoCo, tactile, prismatic, or failed
rendering results as current success evidence. They may only be used as
historical negative evidence or for comparison after the SUGAR reproduction is
complete.

## Highest Priority Research Mainline: SUGAR + SMP + ICM + Tactile

- The accepted official SUGAR CarryBox reproduction is the frozen control:
  `SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid
  Loco-Manipulation Learning Framework`.
- SUGAR is the closest public baseline because it combines human-video-driven
  loco-manipulation, IsaacLab, G1-style humanoids, and CarryBox-like tasks.
- The active research mainline uses SUGAR's G1 CarryBox action/physics/data,
  an official-architecture SMP/SDS motion prior, original-ICM curiosity, and
  direct spatial tactile sensing to explore alternative carrying strategies
  after the nominal bilateral side clamp slips or cannot lift the box.
- Current SUGAR exact reference tracking is not the exploration task: it would
  terminate or punish materially different successful poses. The new task must
  keep official SUGAR action/robot/assets/physics while using a goal-based
  objective and comparing against the frozen accepted control.
- SMP constrains natural/carrying-like motion. ICM is an independently learned
  intrinsic discovery model. Slip, task success, repeated original clamp, and
  safety remain separate external objectives/constraints with separate logs.
- A learned demo-conditioned compatibility reward may be investigated as a
  fifth, separately logged mechanism. It is learned imitation/style, not ICM
  curiosity and not automatically the frozen SMP score. Following a selected
  video requires a declared demo condition; runtime scoring must be causal and
  cannot read future reference or privileged outcomes.
- The 2026-07-27 causal-bootstrap audit withdraws the posture-adaptive V1
  capacity/formal/grid chain from both training and behavior evidence. Its
  mid-trajectory motion-45/frame-103 reset omitted official Refiner
  `last_action` 102 and repeated one TacSL frame instead of restoring real
  frames 100--103; the first alleged zero-residual teacher action has L2 error
  `12.2222967` against source action 103. Jobs `201895`/`201925`, their
  checkpoints, and partial grid traces must not be reused. The next allowed
  experiment is the V2 two-update gate in
  `DOCS/legacy/sugar_causal_contact_bootstrap_v2_protocol_20260727.md`, followed only
  on a pass by fresh matched no-demo/internal-reward initialization. Slip stays
  a direct-TacSL policy belief and separate external constraint; the frozen
  demo predictor stays a pre-failure potential; original ICM stays
  independently learned and ungated.
- Official XIRL is admitted only as a separate RGB task-progress control
  `r_xirl_goal`: current-frame embedding distance to a mean terminal
  demonstration embedding. It is not selected-demo conditioned, not a
  state/direct-TacSL sequence predictor, and not the final `r_demo_pred`.
  Because its reward path requires live RGB, do not train or integrate it
  while the RGB branch remains stopped without a new explicit user
  instruction.
- Official RoboCLIP is admitted only as the separate selected-video semantic
  control `r_roboclip_demo`: an unscaled S3D dot product between one demo video
  and the completed robot rollout, returned sparsely at episode termination.
  It is not a dense causal prefix predictor and has no state/direct-TacSL
  input. Its parent repository has no top-level license, and its linked S3D
  PyTorch weights/dictionary are not checked or hash-pinned locally. Their
  official Drive fallback identities are frozen as
  `s3d_howto100m.pth`/`125,031,128` bytes and
  `s3d_dict.npy`/`5,830,040` bytes, but no artifact-specific license or
  published content hash exists. Do not adapt, install, download weights,
  execute, or integrate it without artifact-license clarification and a new
  explicit user instruction reactivating an RGB reward path. Its released
  MetaWorld wrapper omits the S3D-documented `[0,1]` normalization while
  Kitchen applies it conditionally; any future released-code reproduction and
  S3D-spec normalization control must remain separate.
- DeepMind's versioned `https://tfhub.dev/deepmind/mil-nce/s3d/1` endpoint is
  admitted only as `deepmind_tfhub_v1`, a separate official S3D backend
  control. It is not the exact `roboclip_released_pytorch` backend until the
  frozen matched-input 512-D embedding and raw-dot-product equivalence suite
  passes. Its artifact-specific official Kaggle API resolves `Apache-2.0`,
  version ID `1444`, and exactly four files totaling `131,583,009` bytes; do
  not infer that result from the TensorFlow tutorial's code-sample license.
  The API publishes no content hashes, and the RoboCLIP-linked PyTorch
  artifacts remain separately unresolved. Before equivalence, never mix the
  two backends' scores, label datasets, calibration fits, student checkpoints,
  or policy cells.
- The S3D equivalence implementation must remain glue around the exact pinned
  PyTorch `S3D` class and official `tensorflow_hub.load` SavedModel interface.
  Keep the backends in separate isolated runtimes and let only the independent
  verifier decide embedding/raw-dot-product equality. Every execution requires
  an explicit user-authorization reference, exact raw official metadata,
  resolved artifact-license evidence, complete content hashes, a shared
  preprocessed tensor manifest, a passing independent
  source-redecode/bitwise-input audit, a retained Slurm allocation, and ignored
  `experiments/` outputs. The TFHub tree must match all four published paths
  and sizes before loading; the PyTorch artifacts must match both published
  filenames and sizes, and its frozen unresolved-license flag must first be
  replaced by artifact-specific evidence. Common inputs must use the official
  S3D preprocessing contract and explicit frozen SUGAR frame indices. Never
  add a local S3D replacement or run these entry points on a login node.
- Official S3D artifact acquisition must use the frozen hash-bound request
  pre-stage. It requires separate acquisition/model/network authorization,
  retained Slurm allocation, artifact-specific license evidence, exact
  metadata and byte counts, per-file hashes, failure cleanup, and atomic
  publication below ignored `experiments/`. Never create an acquisition
  request from a general instruction to continue. The TFHub/Kaggle path may
  run only after explicit authorization; the RoboCLIP-linked PyTorch path must
  fail before network access until its binary-artifact license is resolved.
- The S3D terminal-teacher to causal-prefix bridge is offline dataset glue,
  not a reward model or policy result. Every build must declare one official
  backend, preserve one exact float32 raw dot product per completed rollout,
  split before prefix expansion, keep source identities audit-only, and pass
  independent bitwise-score plus exact-row reconstruction. Do not generate
  labels while the admitted support lacks faithful release-to-recontact
  recovery or stable alternative-strategy success. A static pass never
  implies an embedding, label, student, reward, or policy result.
- Strategy-support evidence may open that label gate only through the frozen
  independent admission protocol. It must recompute same-episode bilateral
  failure, release/re-arm, later direct-TacSL recontact, changed object-frame
  topology, actual lift, stable goal, and unchanged safety from official
  SUGAR traces preserving `2 x 3 x 20 x 25` normal/signed-shear taxels.
  Binary contact and palm/object geometry remain non-tactile audit fields.
  One alternative family must repeat across at least two rollouts, seeds, and
  genuinely distinct physics tuples, with synchronized world RGB, spatial
  pressure/shear, and GelSight RGB/depth evidence. Current recovery and
  repeatable-alternative counts remain `0`; never fabricate a request or open
  the gate from a general instruction to continue.
- Strategy-support source provenance must first bind a passing frozen
  collection/readiness audit. Current C4 is not schema-compatible: it lacks
  complete per-step goal/safety/exact-palm-center fields and dual GelSight
  RGB/depth, and may not be retroactively upgraded. Future evidence may use
  only per-episode provenance: each episode trace, raw camera archive, world
  video, rollout identity, seed, and observed physics tuple must exactly
  cross-bind to the same passed collection audit and fresh single-pass record.
  One global source record may not authorize unrelated episode rows. Future
  evidence may use
  the unregistered one-environment coherent-task official-R15/world-camera
  config through the guarded deterministic frozen-policy single-pass
  collector. The single-pass output must preserve one world frame per source
  step, raw-frame hashes, and bilateral GelSight event windows without
  PPO/SMP/ICM/reward-model updates. Its independent audit must bind the exact
  collector run record, original authorized request, frozen checkpoint,
  physics readback, official runtime assets, source hashes, output files, and
  minimum trace length. Exact visual-only replay is a future contract and is
  not executable until a producer plus complete provenance verifier is
  implemented; it must preserve initial state, physics, and applied actions
  bitwise, pass strict state/direct-TacSL equivalence, and never count as
  another policy success. Images are never actor/reward inputs. Collection
  Future collection execution still requires a new explicit experiment
  authorization and retained Slurm allocation.
- The first authorized strategy-support runtime probe is a frozen negative,
  not supervision. Self-binding the unchanged global-v3 R15 mount offsets
  restores bilateral direct spatial pressure/signed-shear on exact motion-45
  contact states, but the 23-step frozen Stage-H candidate has zero direct
  contact, no closed failure/later attempt/goal success, and unsafe
  termination. The faithful official v2.3.2 compliant-contact `10/1` path and
  camera freshness audit now pass an independent bilateral `0.25--4 mm`
  normal-press pressure/shear/GelSight RGB-depth response test. Camera-fresh
  exact nominal replay is still optically static at only `0.005138395 N`
  maximum integrated normal force. This closes only the simulated
  sensor-response blocker. Contact-seeded H2R1 and C4-P3 global-90 both start
  from a verified bilateral motion-45 frame, lose one hand on their first
  action, and produce zero post-action bilateral direct/optical-response
  frames. The active gate is learned two-hand contact-topology/load retention,
  with any external retention objective logged separately from original ICM.
  The canonical one-update retention integration now passes: its weaker-hand
  direct-TacSL load term reconstructs exactly on valid transitions, every
  reset uses the hash-locked bilateral motion-45 source, and original ICM
  remains nonzero when retention is zero. Only `3/480` valid environment-steps
  have positive retention reward, so this is wiring/reset evidence and not a
  learned-retention result. The bounded eight-update
  reward-enabled/reward-disabled comparison and frozen held-out TacSL/GelSight
  audit are now negative. Excluding automatic-reset observations, reward-on
  has `26` valid bilateral transitions versus `34` for control, while the
  strict fresh-PhysX first-rollout bitwise gate fails. Both final policies lose
  left direct contact on the first held-out action and preserve bilateral
  direct/optical contact on `0/32` frames; both independent trace audits pass
  with the candidate gate false. Do not extend the current `0.04 N`,
  `0.02`-scaled, pre-first-failure term. The next allowed experiment is a new
  predeclared, source-normalized nominal contact-foundation pair selected from
  frozen reward/advantage-scale statistics. It remains external to ICM and
  must switch off before post-failure alternative discovery.
  That nominal foundation is now implemented and rejected. Its canonical
  one-update direct-taxel reconstruction and ICM-separation audit passes, but
  the eight-update enabled arm has only `33` valid bilateral transitions
  versus `31` for control, closes three more failures, and retains the failed
  strict fresh-PhysX first-rollout bitwise gate. Frozen seed-`4253` evaluation
  gives `0/32` bilateral post-action direct/direct-plus-optical frames in both
  arms; both lose left contact on the first action and keep right contact for
  only two frames. Both independent visual audits pass. Do not long-train this
  foundation. Before another PPO segment, the only allowed active experiment
  at this boundary is a no-learning causal reset/action-history audit that
  compares the frozen action with the official SUGAR reference continuation
  and a separately labeled hold diagnostic. Do not substitute a hand-written
  controller or claim contact learning, recovery, alternative strategy, or
  ICM progress.
  That no-learning audit now passes. In one synchronous four-branch PhysX
  step, cold H2R1 and H2R1 with the correct source action-102 plus
  TacSL-100--103 history both lose left direct contact; a current-joint-target
  hold loses both hands. The exact recorded official SUGAR action 103 alone
  preserves bilateral direct TacSL with left/right `0.004663/0.042392 N`,
  `31/150` active taxels, and signed shear on both hands. Independent arrays
  and the pressure/shear/GelSight/world render pass audit. This is a bootstrap
  diagnosis, not learned contact. The next allowed nominal experiment is a
  faithful accepted-official-Refiner teacher-action prefix with its exact
  reference observation and native action mapping. Any residual or behavior-
  cloned goal-policy adapter must be explicitly predeclared and zero-change
  audited. Teacher imitation/action anchoring is separate from SMP and ICM and
  must relax or turn off after tactile-confirmed nominal failure before
  alternative discovery. Do not substitute another scalar contact reward,
  a hand-written controller, or an alleged official-checkpoint warm start with
  incompatible observations.
  The exact recorded official action prefix 103--106 now preserves bilateral
  direct TacSL and signed shear across all five seed/post records in the
  unchanged goal scene, reaching `0.06139/0.22731 N` and `146/293` active
  taxels. Its independent audit and manual render inspection pass. The
  accepted Refiner checkpoint is hash-bound as the source-action generator but
  is not loaded, and state drift means this is not exact trajectory replay.
  Do not call it live teacher inference. The next gate is exact online
  reconstruction of the official Refiner reference observation and frozen
  action/contact equivalence before any residual or behavior-cloning adapter.
  That live gate now passes: the exact official uncorrupted 890-D observation
  plus frozen accepted `model_10000.pt` reproduces the first same-state action
  to L2 `1.94e-6`/maximum absolute `9.54e-7`, and four live goal-scene steps
  retain bilateral direct pressure and two-axis shear with no reset. The
  independent audit and synchronized render pass. This admits only a nominal
  action teacher. The next allowed comparison is an explicitly predeclared
  zero-residual official-teacher interface versus an offline behavior-cloned
  causal goal-policy bootstrap. Teacher anchoring must remain separate from
  SMP and original ICM and relax or turn off after direct-TacSL-confirmed
  nominal failure.
  Recovery and repeatable-alternative counts remain `0/0`; do not create an
  admission request, teacher label, reward predictor, or policy result from
  this episode or the diagnostic press.
- The Plan-08 exact-zero interface now passes: the reusable frozen official
  Refiner adapter changes no checkpoint state and exact-zero residuals
  reproduce four executed teacher actions bitwise while five seed/post records
  retain bilateral direct pressure and signed two-axis shear. Its independent
  audit is `19/19`. This does not yet pass PPO residual-variable
  sampling/log-probability integration, construct or train a residual policy,
  or select a residual scale.
- The Plan-08 offline-BC causal contract passes `11/11` with existing official
  sources: `74/8/10` motion-disjoint train/validation/test motions and
  `36,519/3,605/4,873` eligible rows. The admitted model is the existing
  `SugarNativeZeroPreservingTactileActorCritic`; no smaller BC model is
  allowed. All `40,124` train+validation rows are now materialized into 82
  motion shards and reconstruct bitwise in the independent `10/10` audit. The
  subsequent serious-actor BC fit reaches one-shot held-out test MSE
  `0.006995`, but zero tactile is marginally better than full TacSL and both
  retain bilateral contact for one live step. A fresh predeclared four-step
  gate rejects BC as the stable nominal foundation: step-4 candidate/control
  load ratios reach `12.78/4.65` and active-taxel ratios `3.04/1.65`. The
  negative run passes its independent `23/23` audit and synchronized render.
  Keep BC only as a bounded diagnostic/control; do not call it tactile
  advantage, stable behavior, recovery, or strategy discovery.
- The Plan-08 no-learning failure-release audit passes. Teacher influence is
  driven only by the direct-TacSL runtime's `initial_strategy_failed`,
  same-step `failure_closed`, and reset mask. Immediate release is zero and
  the frozen four-step linear schedule is `0.75,0.5,0.25,0`; only reset
  rearms it.
- The Plan-08 integrated residual path now passes `22/22` producer and
  `16/16` independent checks. The serious SUGAR-native direct-TacSL actor
  starts at exact-zero residual mean; upstream RSL-RL PPO 3.0.1 stores and
  log-scores the residual while the environment applies
  `alpha * frozen_official_teacher + 0.05 * residual` and the native processed
  joint target reconstructs exactly. A real tactile-confirmed failure releases
  the teacher to zero. Original ICM scores actual applied actions, receives no
  teacher/outcome gate, stays positive after bootstrap, and updates
  independently on 91 valid transitions. The teacher and official TinyMDM
  stay frozen and the combined checkpoint reloads exactly.
- The frozen integrated-checkpoint camera replay passes `11/11` producer and
  `15/15` independent checks with synchronized world RGB, bilateral direct
  pressure/signed shear, and official GelSight RGB/depth. It loses contact
  after release and later resets. Therefore this is only a one-update
  interface/checkpoint result; stable behavior, recovery, and repeatable
  alternative-strategy counts remain `0/0`.
- The Plan-08 fixed 64-update residual follow-up passes every producer check
  and the independent `14/14` audit with 30,386 valid original-ICM
  transitions, but it does not improve frozen behavior: update 8 and update
  64 both fail/reset at steps 3/17 with zero bilateral recontact. A
  same-chain common-random-number stochastic control produces an
  approximately `0.48 m` single-hand-looking lift at both update 1 and 64,
  with identical `2/36/27` bilateral/unilateral/no-contact frames. Sampled
  residual L2 is approximately `5.306`, versus `0.0383` for the update-64
  learned mean, and the teacher never releases. Treat this as Gaussian
  exploration/noise evidence, not learned ICM policy progress, post-failure
  recovery, or an alternative strategy.
- The fixed 20-pair common-random-number audit now rejects learned
  task/contact robustness at update 64: it adds two failures/resets, avoids
  none, lowers mean final height by `0.04317 m`, and remains sampled-noise
  dominated. The separate exact original-ICM attribution audit passes `22/22`
  checks but does not admit increased discovery: frozen ICM-update-1 paired
  error difference has mean `-0.06389` and 95% interval
  `[-0.21107,0.04297]`, although frozen-feature support distance rises
  `0.02498`. This is independent of task success and must not relabel failures
  as non-discoveries. Do not continue the same training contract merely for
  more updates; first run matched ICM-policy-weight-zero and
  SMP-policy-weight-zero credit controls while keeping the exact ICM learner
  active and separately logged.
- The matched 64-update policy-credit ablation was executed and is rejected;
  it is not a completed result.
  Full, SMP-policy-weight-zero, and ICM-policy-weight-zero arms all fail the
  frozen original-ICM discovery gate; every forward-error confidence interval
  crosses zero. Original ICM was scored and independently learned in every
  arm, so this does not redefine curiosity by outcome. The arms contain only
  `3.19%`, `2.36%`, and `3.23%` zero-teacher actions.
- The recurring 64-action post-drop exposure run was executed but is not
  complete. It suppresses
  1,320 raw drops and raises zero-teacher actions from 979 to 1,389 while
  keeping original ICM and the full reward mix exact. Fresh no-grace
  evaluation still fails discovery: frozen-ICM error difference is
  `+0.03397` with 95% interval `[-0.15904,0.22121]`, and support distance
  changes `-0.07130`. The rendered rollout resets at step 18 and ends fallen
  without contact. The next allowed comparison is named-joint/block teacher
  release: retain official leg/waist support while releasing both arms.
  Preserve failure causality, log every joint coefficient, keep ICM unchanged,
  and do not call this lower-squat exploration because the support block
  remains anchored.
- The named-joint blockwise release run was executed and is rejected; it is
  not a completed study. Its exact
  runtime partition retains all 15 hip/knee/ankle/waist teacher coefficients
  at one while releasing the 14 shoulder/elbow/wrist coefficients, increasing
  arm-zero-teacher exposure to `1,996/30,720` (`6.50%`, `2.04x` the scalar
  parent). Fresh update-64 resets worsen `7 -> 8`, bilateral frames remain
  `23 -> 23`, paired final height changes `-0.02083 m`, frozen-ICM
  forward-error CI crosses zero, and frozen-feature support decreases. The
  synchronized visual resets at steps 22 and 55 and ends without contact.
  This rejects advancing lower-body teacher retention as sufficient. The next
  allowed structural comparison is a failure-latched 15-D support-action
  hold with the same 14-D arm release. It is an action prior, must preserve
  the advancing mode bitwise, must not alter/gate original ICM, and still
  cannot claim lower-squat behavior.
- The failure-latched support hold is mechanically exact but scientifically
  negative. It holds the captured 15-D support action for 1,265 environment
  steps while leaving the 14-D arm release unchanged, yet fresh reset pairs
  worsen `6 -> 7`, paired final height changes `-0.02360 m`, and its frozen
  original-ICM forward-error interval crosses zero.
- The earlier supported post-drop combination does not complete Plan 09. It
  reported keeping all 15
  support coefficients at one, releases only the 14 arm joints, suppresses
  1,652 raw drops within the fixed training window, and reaches
  `3,383/30,720` (`11.01%`) zero-arm-teacher actions. All producer,
  independent, direct-TacSL, checkpoint, and source audits pass. Fresh
  no-grace resets nevertheless worsen `7 -> 9`, paired final height changes
  `-0.04458 m`, and frozen original-ICM forward-error difference is
  `-0.03063` with 95% interval `[-0.09782,0.02427]`; feature coverage alone
  rises `+0.06566`. The synchronized visual shows bilateral seed contact,
  left-only post-release contact, and no endpoint regrasp. Discovery,
  recovery, and alternative-strategy gates fail; counts remain `0/0`.
  These artifacts are withdrawn as completion evidence and must be re-audited
  and rerun from scratch. Plan 09 remains active. Do not call the old negative
  run a completed study or a successful result.
- Native official TacSL shear is channel-last
  `[environment, taxel, xy]`. Any channel-first archive must transpose the
  taxel and xy axes before reshape. The frozen generic Stage-I C4
  `raw_tactile_latest` shear archive omitted this transpose and may not be
  used as signed-shear evidence; its normal channel and normal-derived zero
  later-contact conclusion remain valid. The official policy/ICM tactile
  adapter already transposes correctly and is unaffected. Do not rewrite the
  historical runner or use a repaired visualization to imply missing
  provenance.
- The 2026-07-29 P3 whole-hand result is not an accepted CarryBox tactile
  sensor. Gate A/B and all twelve native-PhysX C2 local zone trials pass, but
  the separately predeclared C1 static CarryBox identity was not executed and
  cannot be replaced by renaming C2 algebra checks. The subsequently executed
  600-step motion-45/seed-4263 CarryBox run is both out of protocol and an
  independent numerical negative: left/right raw-to-active taxel
  median/P90 distances are `6.810/20.194 mm` and
  `7.928/23.915 mm`, per-zone 5 mm linkage is only
  approximately `0.325--0.482`, stable lifted bilateral support has zero
  accepted frames, and official termination begins at step 296. Its genuine
  float32 TacSL formula reconstruction and decoded videos do not rescue the
  spatial/load failure. Stop at human review. Do not start another sensor,
  optical gate, training, slip detector, internal reward, or policy
  integration until the user reviews the 50 FPS synchronized videos and
  explicitly continues. Whole-hand GelSight RGB/depth remains unresolved.
- The audited upstream selection is closed at the current data gate: no
  official TinyMDM, T-REX, XIRL, or RoboCLIP interface matches the complete
  selected-demo + dense causal state/direct-TacSL predictor. RoboCLIP is the
  closest selected-video semantic control, so any future `r_demo_pred` must
  state that its new contribution is the dense causal state/tactile boundary.
  It must wait for the stable alternative-strategy-positive gate; never
  inherit an upstream name to weaken this boundary.
- A future RoboCLIP-teacher/state-TacSL-student branch must classify the
  student as a new project method. Its predicted terminal compatibility is a
  potential/value estimate, not an observed instantaneous reward. Do not pay
  the raw positive prediction every step; compare sparse terminal use with a
  frozen policy-discount-matched potential difference and verify exact
  telescoping. Every label manifest and S3D demo condition must declare the
  same single teacher backend. The visual teacher is not tactile, and demo
  reward/OOD handling may never gate original-ICM novelty.
- Plan-07 Stage R0 source/claim/causal-field audit passes. The exact official
  SUGAR manifest, all 922 feature-shard hashes, admitted 200k TinyMDM/EMA,
  twelve train/validation/test motion-disjoint TacSL physics sequences, and
  four causal replay schemas are pinned. A strict 5,178-row causal
  live-prefix/offline-future candidate dataset exists but contains only failed
  goal-policy rollouts. A separate frozen-official scan records all 77 train
  motions and 74 exclusive nominal completions with distinct numeric demo
  conditions. The motion-disjoint frozen-official nominal dataset now passes
  with 74/8/10 successful train/validation/test motions and 43,341 strict
  causal-prefix/offline-future rows. Its independent audit reconstructs every
  unweighted body/box label exactly. This authorizes only the nominal label
  dataset; no reward predictor or policy result may yet be claimed.
- Nominal demo-alignment admission and tactile-confirmed recovery are separate
  gates. Exclusive official `trajectory_complete` may label nominal alignment
  without bilateral TacSL. Direct spatial pressure/shear must still be
  preserved and reported. Bilateral or otherwise task-relevant direct TacSL
  remains mandatory evidence for a failed clamp, recovery boundary,
  contact-topology switch, or tactile-conditioned alternative strategy.
- The frozen-control train scan has only two synchronous bilateral observations
  in one successful motion. Train-derived sub-millimeter mount corrections
  failed to improve bilateral validation contact and exceeded the active-taxel
  footprint bound; the global v3 mount remains unchanged. This sensor-coverage
  result must not block nominal-label construction or be relabeled as recovery
  data.
- The R1 reference condition/label audit passes: all 922 demonstrations have
  a fixed 32-window official-normalized numeric condition and 349,736
  audit-only controlled pairs preserve motion-ID split isolation. The pair
  geometry and train/validation/test direct pressure/signed-shear render passed
  manual inspection. Every such pair is predictor-ineligible; this is not a
  completed predictor dataset or a learned-reward result.
- After a tactile-confirmed failed nominal clamp, learned/exact imitation must
  not suppress valid lower-squat, asymmetric, bottom-support, or regrasp
  exploration. Any phase-dependent relaxation changes only the imitation
  ledger and must never gate original-ICM novelty.
- The paired R2 audit rejects exact selected-demo body+box loss as a strong
  global post-failure reward: another valid official SUGAR motion is scored
  worse than frozen/body-corrupted/box-corrupted/impossible-rotation behavior
  in 98.31%/98.44%/97.13%/84.73% of paired anchors. Retain exact loss only as
  an interpretable pre-failure or oracle baseline until explicit alternative
  strategies and a phase-tolerant semantic target pass.
- The admitted nominal Stage-H optimizer contract is
  `sugar_native_zero_preserving_tactile_fixed_low_lr`: upstream RSL PPO 3.0.1,
  native SUGAR action, direct spatial tactile policy, 14 additive tactile
  encoder biases frozen at exact zero, and fixed `1e-5`. ZF0/ZF1 pass; this is
  an explicit tactile-policy adaptation, not untouched official BasePPO
  hyperparameters. The separately locked five-role H2R1 direct-TacSL gate also
  passes with raw sensor provenance, noise/latency/dead-taxel transforms, and
  all five roles active from ICM initialization. Formal long-horizon discovery
  and alternative-strategy evidence remain pending.
- ICM intrinsic output must not be gated by successful lift, task progress, or
  lower slip. A novel failed experiment is still a discovery. Use the inverse
  model/action-relevant representation and test noisy-TacSL failure modes.
- The public MimicKit snapshot contains no paper Object Carry environment, HOI
  dataset, or Object Carry checkpoint. Train the official TinyMDM architecture
  on audited official SUGAR G1+box rollouts; do not substitute LaFAN weights or
  invent paper assets.
- The RGB-only SUGAR branch was stopped and downgraded to P2 historical
  diagnostic status on 2026-07-23. Do not resume its trainer, model8000 gate,
  full100 path, watcher, or GPU allocation without a new explicit user
  instruction.
- The primary sensor implementation is the official IsaacLab `v2.3.2`
  TacSL-based `isaaclab_contrib` `VisuoTactileSensor`, or a minimal auditable
  backport of that exact official sensor/assets/configs if a direct upgrade is
  incompatible. Do not replace it with a locally invented simplified sensor.
- A valid tactile observation must preserve, per hand, a taxel-resolved normal
  force/pressure field and a two-axis tangential/shear field. GelSight-style
  RGB/depth deformation streams must also be validated and recorded; policy
  inclusion is controlled by explicit ablations.
- Binary contact, contact-force thresholds, SUGAR hand-contact history,
  aggregate force/torque, hand-object distance, object pose, penetration flags,
  reward terms, and privileged-state latents are not tactile modalities. They
  may be reported only as diagnostics or explicitly labeled non-tactile proxy
  controls.
- PhysX contact reports may validate integrated sensor forces, but may not be
  substituted for the actor's tactile observation.
- The frozen physical whole-hand TacSL collider gate is negative as of
  2026-07-29. The single allowed `1024`-hull bake misses one taxel per hand
  beyond `2 mm` (`2.82965/2.44532 mm`), so the old audit that excluded those
  taxels is withdrawn. An already-run out-of-protocol `2048` diagnostic proves
  the official TacSL algebra and same-frame pose alignment but still misses
  significant raw contacts with zero active taxels and does not close box
  weight dynamics. Do not call the sensor installed/correct, train from it, or
  try another hull count, atlas density, contact-offset change, SDF margin, or
  calibration without a newly approved protocol. Its synchronized H.264
  videos are negative human-review evidence only.
- The newly approved single repair is P1 in
  `DOCS/legacy/sugar_conformal_whole_hand_tacsl_skin_protocol_20260729.md`. The exact
  official R15 reference has now been audited: its released spawner binds
  compliant stiffness/damping `10/1`, static/dynamic friction `0.5/0.5`,
  restitution zero, and neither its source USD nor spawned collision authors
  contact/rest offsets. Do not infer or copy offsets from a documentation
  example. P1 must preserve all `4426/4427` hash-locked whole-hand taxels and
  give each one exactly one finite-area regular-hexagonal convex prism on the
  existing hand rigid body, using the frozen taxel area, radius, `1 mm`
  inward thickness, and exact official-reference material. No taxel deletion,
  hull/atlas/offset/SDF/calibration sweep, new body/joint/mass/inertia,
  training, Genesis, shadow replay, or binary proxy is authorized.
- P1 is not installed or correct until its independent static, composed-stage,
  twelve-zone controlled-response, uninterrupted 600-step CarryBox
  correspondence/load, and synchronized-video gates all pass. If any gate
  fails, retain the exact negative; do not rescale forces or switch candidates
  under the same protocol. Stop after the final videos for human review before
  policy use. Whole-hand GelSight RGB/depth remains unresolved and must not be
  fabricated.
- P1 Gate A and Gate B pass as of 2026-07-29. The exact static asset has
  `4426/4427` unique finite-area collision owners. The composed official
  CarryBox stage contains no H0 path, preserves the exact robot/box hashes and
  SDF, binds `10/1`, `0.5/0.5`, zero restitution with no offset authorship,
  initializes full float32 normal/signed-XY-shear/penetration fields, and
  matches live mass/inertia/joint/actuator/29-D action fingerprints exactly to
  a separately executed untouched official-SUGAR baseline.
- P1 is rejected at Gate C as of 2026-07-29. The first mandatory `left/palm`
  trial has all `350` source steps, but the intended left direct field remains
  zero through every normal/shear phase while the non-target right hand
  responds on `350/350` frames. Its one positive raw left pair has no
  same-frame direct response. The former `4.976162 mm` direct point-to-taxel
  claim is separation-unaware and withdrawn as proof of collider/taxel
  translation: the complete tuple has `4.610698 mm` positive separation and
  maps to `0.623510 mm`, within the frozen `0.824326 mm` limit. P1 remains
  rejected because target error grows to roughly `0.4 m`, the external box
  servo saturates force/torque on `349/49` frames, and the mandatory target
  response is absent. Official normal and signed-shear arithmetic reconstructs
  with `0.0 N` maximum error, so this is not a visualization repair. The
  authoritative evidence is
  `DOCS/legacy/sugar_conformal_whole_hand_tacsl_p1_gate_c_negative_result_20260729.md`.
  Do not run P1 Gate D, finish the other zones as an acceptance path, train
  from P1, or claim whole-hand tactile solved. The synchronized videos are
  negative human-review evidence only.
- Exact-SDF whole-hand tactile P2 is rejected at Gate B. Its exact continuous
  SDF geometry/material and untouched official robot/action reconstruction
  pass, all `4426/4427` centers lie within `0.168209 mm` of the cooked
  physical surface, and no taxel is deleted. The frozen source-triangle frame
  still produces left/right `10/10` sign failures, `46/48` full-range
  monotonic failures, and `7/7` cooked-gradient alignment failures. The
  authoritative cloned-buffer result is
  `DOCS/legacy/sugar_exact_sdf_whole_hand_tacsl_p2_gate_b_negative_result_20260729.md`.
  Do not run P2 contact/behavior gates or reinterpret its diagnostic as a
  pass.
- P3 and the later native-surface `4 mm` query-skin path are both withdrawn as
  positive tactile evidence. P3's static frame checks remain diagnostic only.
  The native-surface run remains a valid 660-step nominal movement baseline
  with a `0.729171 m` lift, but its field was sampled from the same closed
  one-hull hand collider and visualized as a merged whole-hand blob. It was not
  generated by separate physical load-bearing palm/finger elastomers, and its
  TacSL vector sum did not close the box dynamics. It is therefore not
  anatomical tactile, pressure, slip, policy, reward, recovery, or strategy
  evidence.
- The only admitted successor is
  `DOCS/legacy/sugar_whole_hand_tactile_non_degradation_standard_20260729.md`:
  27 physical load-bearing patches per hand, all raw 20 x 25 normal and
  channel-last signed-XY-shear maps, symmetric geometry-fixed R15 palm
  RGB/depth, 54/54 separate controlled probes, held-out calibration,
  same-frame spatial/temporal correspondence, object gravity/dynamics closure,
  continuous nominal CarryBox, held-out 3x-mass/low-friction sensor audits,
  independent reconstruction, separately inspectable H.264 videos, and
  explicit user approval. All gates must pass together; no old tactile or slip
  checkpoint may be inherited. Recovery/alternative counts stay `0/0`.
- Until load, contact footprint, shear/slip behavior, noise/latency, and image
  response are calibrated against a physical GelSight-class sensor, results
  must be labeled `high-fidelity simulated tactile`, not physically validated
  tactile or sim-to-real.
- Do not continue old direct Isaac G1/AGILE scalar tuning, MuJoCo carrying
  paths, tactile-only paths, or non-SUGAR proxy scaffolds as active work.

## Highest Priority Cluster Safety Rules

These rules override all other project instructions.

### Login Node Hard Limit

- Never run Python experiments, data processing, validation builders, model
  loading, rendering, simulation, training, evaluation, visualization
  generation, dataset conversion, NumPy/PyTorch-heavy scripts, or any other
  compute-heavy project task on a login or management node such as
  `mgmtserver02`.
- Login nodes are only for lightweight operations: editing files, `git`
  commands, `git clone`, `git push`, small text inspection with tools such as
  `sed`/`rg`, lightweight file listing, and job/allocation submission.
- Keep login-node CPU below 300% and memory within lightweight interactive
  limits. If a command can plausibly exceed those limits, do not run it on the
  login node.
- If a project Python command is needed and it is not a trivial import-free
  syntax check, submit or run it inside a compute allocation instead.

### Compute Node Requirements

- All simulation, rendering, dataset conversion, training, evaluation, model
  loading, and visualization generation must run on compute nodes.
- GPU resources must be obtained and kept through `tmux` plus persistent
  `srun`/`salloc` allocation workflow. Do not use one-shot submission paths
  such as `sbatch` or single-use wrappers for experiments unless the user
  explicitly approves.
- Do not use `sspath` or other one-shot resource paths for this project.
- Compute nodes should only activate prebuilt local shared-filesystem
  environments. Do not perform normal dependency installation, venv creation,
  package builds, or dependency resolution on compute nodes.
- Short runs must be labeled as diagnostics or smoke tests, not as real
  training or real experiment results.

### Resource Exclusion Zone

- Do not touch, inspect, stop, reuse, attach to, or modify any `reflex`,
  `ICLR2027/Reflex`, OpenPI, Cosmos, or other non-Curiosity tmux sessions,
  allocations, processes, logs, scripts, or resources.
- If non-project sessions appear in process listings, ignore them except to
  avoid interference.

## SUGAR Fidelity Gates

- Use official SUGAR task names, official CarryBox data, official robot/object
  descriptions, official checkpoints, and the official training stage order.
- Keep SUGAR and IsaacLab local changes minimal, auditable, and limited to
  cluster/runtime compatibility unless the user explicitly asks for research
  modifications.
- **Reproduction accepted/passed by the user on 2026-07-13.** The acceptance
  criterion is that the official SUGAR CarryBox pipeline and reproduced effect
  are functionally normal; exact equality with paper-reported numbers is not
  required.
- The accepted local Refiner boundary is the official-code `model_10000.pt`.
  Do not resume Refiner training beyond iteration 10000. Its successful
  rollout, processed dataset, and visualizations are valid reproduction
  evidence.
- Tracker and Generator continuation may remain active to complete and improve
  the local artifact chain, but their unfinished or numerically non-identical
  results do not revoke the user-approved SUGAR reproduction pass. Record their
  actual status truthfully and do not claim exact paper-number reproduction.
- Any future research claim must compare against the faithful SUGAR CarryBox
  reproduction, not against archived local proxy tasks.
- The corrected causal-contact bootstrap V2 two-update boundary now passes
  independently (`13/13`, audit SHA256
  `1d17604ce4e791b2c7f8e143e722fae7c5ae8256dc3d6ff0811179fc023bf98a`).
  It restores official action 102 to both previous-action views, preserves
  real dual-hand TacSL frames 100--103, matches frozen official action 103
  within its declared fresh-PhysX tolerance, and keeps V16 slip external to
  original ICM. The fresh seed-`106781` matched no-demo/internal-reward pair
  now completes 521 updates per arm and passes its independent causal
  attribution audit (SHA256
  `4b91dab537017bbe240bb9bcfab29ab315836f184cc96faaeb3c873e484092a4`).
  It exposes 1,000,320 scheduled slots per arm and 986,633 / 986,632 valid
  transitions; the demo arm has 1,519 failure boundaries and 123,182 exact
  post-failure zero rewards. This is training evidence, not behavior. The
  complete frozen causal mass/friction/COM grid now passes every structural
  audit but is behaviorally rejected: all 432 rollouts are unsafe with zero
  success, recovery, alternative, or render candidates. Pair-audit SHA256 is
  `a6d0d73e3ff75a1b49cb4f51522a84a1bec40890de0ed60edf823347c4025c17`.
  Do not add updates to the same contract or open cross-seed/positive-render
  gates. The next allowed runtime is the no-learning exact-grid failure
  visualization declared in
  `DOCS/legacy/sugar_causal_posture_grid_v2_negative_result_20260727.md`; failed
  videos remain diagnosis only. Old V1 checkpoints remain forbidden.
