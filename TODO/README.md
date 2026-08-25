# TODO index

## Demo following

- [x] remove tactile-surface contact penalties from the goal task;
- [x] replace demo-anchor and root-up-axis false fall checks with relative physical root-height
  loss;
- [x] pass the CarryBox45 teacher-only bilateral-contact and 5 cm lift gate on 20 profiles;
- [x] train correct/unrelated selected-demo arms for 64 updates with the same CarryBox45 teacher;
- [x] freeze-evaluate 20 matched physics profiles per arm and render complete demo/actual videos;
- [x] verify that selected-demo identity changes checkpoint parameters and residual actions;
- [x] compute predictor-independent behavior adherence from existing traces using object lift,
  lifted/ground transport, robot orbit and available hand-contact topology;
- [x] record that the original seed-161581 trace cannot recover foot identity or direct kick
  contact because those evaluation fields were not archived;
- [x] predeclare three matched seed pairs for the fixed 64-update repeat;
- [x] add per-body pose, named foot-box contact and hand-box-only contact to the repeated frozen
  evaluator as evaluation labels, not actor inputs;
- [x] run seed pairs `161583/161584` and `161585/161586` serially and
  evaluate them with seeds `171583` and `171585`;
- [x] add a one-seed-at-a-time runner and a training-seed-level behavior aggregator; neither
  substitutes physics profiles for independent seeds;
- [x] run one fixed-profile serious overfit diagnostic that anneals teacher authority identically
  from `1.0` to `0.25` and adds exactly 64 updates to both seed161581 endpoints;
- [x] freeze-evaluate the diagnostic and reject a multi-seed repeat because both arms collapse,
  foot contact stays zero and the Kick-like behavior gate is `0/4`;
- [x] audit contact-role, duration and object-motion event labels over all 100 CarryBox and 99
  KickBox official references;
- [x] collect actual rollout targets with named hand/foot box contact, required-contact duration
  and ground/lifted object-motion regime;
- [x] extend the existing serious causal Transformer with multitask contact/event heads and pass
  motion-disjoint held-out plus permuted-demo checks;
- [x] freeze the seed271301 best-epoch-8 predictor and fit validation-only variance calibration;
- [x] reject seed271301 as a deployable reward after finding the 510-D actor mismatch and
  per-frame free-window phase loophole;
- [x] rebuild `10 x 121` targets with causal normalized clock phase, train/freeze seed271303 and
  pass held-out full/zero/permuted-demo plus uncertainty gates;
- [x] pass fixed Carry45/Kick21 bidirectional semantic scoring, freeze dense reward scale and the
  update-32/update-64 stopping rule;
- [x] connect the frozen phase-aware scorer to the serious SUGAR rollout boundary without changing
  SMP/original ICM, and add correct/unrelated matched dry-runs;
- [x] freeze update-32/update-64 evaluation, per-checkpoint predictor-independent behavior audits
  and final demo/actual video generation;
- [x] pass correct and unrelated formal inner-runner admission on retained H200 with zero PPO
  updates, then restore the allocation GPU hold;
- [x] remove the unintended dual-R15/TacSL scene from the explicit-zero demo controls and use the
  original SUGAR G1/CarryBox scene with no tactile sensor construction;
- [x] add a 24-step formal online-rollout smoke that checks live reward injection and executes zero
  optimizer updates;
- [x] pass a fresh-allocation Isaac Sim 5.1 H200 canary, then pass correct and unrelated 24-step
  online smokes with zero optimizer updates;
- [x] replace the stale TacSL teacher evaluator with the no-TacSL scene and pass the shared
  CarryBox45 exact-zero-residual physical prerequisite over 20 profiles and 400 steps;
- [x] bind standard-SUGAR startup material/mass/inertia/COM readback into every phase-event proof,
  restore it in frozen evaluation and re-pass both online smokes with exact matched physics;
- [x] prove without optimizer updates that selected-demo feedback changes PPO return, normalized
  advantage and the clipped actor-surrogate gradient in both correct and unrelated arms;
- [x] verify online that fixed-one teacher authority still executes the full 29-D student residual
  through ActionManager in both arms;
- [x] make formal runner probes fail closed on a machine-readable result rather than subprocess
  return code alone;
- [x] run the historical phase-event matched pair serially for exactly 64 updates;
- [x] freeze-evaluate update 32/64 on 20 matched physics profiles per checkpoint and render the two
  complete demo/actual H.264 videos;
- [x] fix evaluator-only batch restoration for 20-training-env wrapper/physics state versus the
  40-env two-checkpoint evaluation scene;
- [x] fix behavior analysis so one-dimensional trace metadata cannot be indexed as environment data;
- [x] reject semantic following at update 32 and 64: the two policies remain nearly identical Carry
  solutions and satisfy only `2/4` and `1/4` predeclared Kick-like directions;
- [x] identify that the frozen scorer rates the correct Carry rollout closer to Kick21 than Carry45
  over most middle frames;
- [x] recollect the same frozen trajectories with exact recorded 121-D prefixes and reproduce the
  deployed phase/reward/risk/uncertainty signals within float32/model tolerance;
- [x] isolate phase offset with a matched scorer-only ablation: reference-frame-197 initialization
  changes all four arm/update blocks from `0/20` to `20/20` Carry-preferring profiles;
- [x] update the scorer, formal runner and frozen evaluator to initialize the first causal phase
  clock from the restored reset reference frame instead of silently forcing zero;
- [x] re-pass corrected zero-optimizer online smokes for both arms with exact reference-frame-197
  readback, zero policy updates and unchanged parameters on retained H200 job258074;
- [x] pass corrected frozen Carry scoring for both arms at updates 32/64: every block has `20/20`
  Carry-preferring profiles and mean margin `+0.324~+0.328`;
- [x] score the motion-disjoint official Generator/Tracker Kick test motions `9/19/.../89`: all nine show
  physical Kick interaction, mean deployed-clock margin is `-0.06508`, and `8/9` profiles prefer
  Kick21; retain motion29 as a counterexample and do not call this Refiner transfer;
- [x] audit released Kick artifacts: official inference provides `generator.ckpt + tracker.pt` and
  the passing corpus executes that pair; no frozen Kick Refiner/residual checkpoint is released in
  this workspace, so retain that narrower claim instead of substituting or training a toy policy;
- [x] keep selected-demo SMP out of the phase-corrected matched pair; actual frozen Carry
  trajectories now prefer Carry45 consistently, while official TinyMDM semantic extension remains
  a separate unfinished gate;
- [x] run exactly one new from-scratch 64-update matched correct/unrelated pair under the
  reference-aware phase clock, freeze-evaluate 20 profiles at update 32/64 and complete the
  predictor-independent behavior audit;
- [x] record the single-seed result accurately: `3/4` semantic directions at both checkpoints,
  stable bilateral Carry in both arms, and no meaningful Kick foot-contact structure;
- [x] render both input-demo/actual-behavior H.264 videos from exact frozen PhysX traces after the
  H200 Vulkan camera path failed before producing a valid frame;
- [x] run the `161589/161590 -> 171589` independent-seed replication with the exact same teacher,
  phase, 64-update budget, reward weights, physics and evaluation; do not integrate SMP or tune
  weights first;
- [x] verify the replicated update-64 pattern: the same `3/4` directions and nearly identical
  lifted/ground transport and orbit deltas, but no meaningful foot-contact structure;
- [x] run one same-seed fixed 4x dense-feedback overfit pair, changing only `eta` and `reward_clip`,
  then freeze-evaluate it;
- [x] reject reward magnitude as the sole bottleneck: update-64 direction count falls `3/4 -> 1/4`,
  effects reverse and unrelated foot-contact advantage stays absent despite `-48.64` feedback;
- [x] specify one shared-checkpoint demo-conditioned actor contract using only causal frozen-predictor
  outputs and selected-demo conditioning before the actor call;
- [x] add dry-run and zero-optimizer checks that swapping Carry45/Kick21 changes the shared actor
  input and exact PPO learning direction while checkpoint, teacher, task and physics stay fixed;
- [x] train one shared actor for 64 updates with ten Carry45-conditioned and ten
  Kick21-conditioned environments under the common CarryBox45 teacher;
- [x] freeze-evaluate the exact same checkpoint and initial state under each demo condition for 20
  profiles, run the predictor-independent behavior audit and render two complete H.264 videos;
- [x] design and execute one fixed serious contact-topology diagnostic using the existing shared
  SUGAR actor, frozen predictor and official Carry45/Kick21 Tracker action directions;
- [x] train exactly 3000 actor-only overfit steps, confirm critic/tactile encoder remain frozen,
  and freeze-evaluate the same checkpoint under condition-only swap on 20 matched profiles;
- [x] record the diagnostic boundary: correct preserves Carry, unrelated produces a strong
  leg/ground-motion split but falls in `15/20`, so it proves condition use rather than valid Kick;
- [x] test a serious full-510D shared actor on both official Carry and Kick physical rollout
  distributions; retain the offline-BC and three-stage DAgger failure as evidence that low
  supervised error does not preserve closed-loop stability;
- [x] build one checkpoint containing the two parameter-exact released Tracker experts and a
  causal 798-D selected-demo router, then pass matched Carry `18/20` and Kick `20/20` with zero
  falls and exact expert actions;
- [x] run exact-initial-state condition-only swaps and record the boundary: Carry-to-Kick route
  violates the raw-action envelope after leaving distribution, while Kick-to-Carry remains
  generator-driven rather than becoming Carry;
- [x] audit the released inference chain and prove that the 510-D Tracker observation contains a
  36-D task-specific Generator command, so a valid skill route must select Generator and Tracker
  together;
- [x] run the four exact-state joint routes: SMALLBOX Carry45 is `18/20` Carry, SMALLBOX Kick21 is
  `19/20` Kick with zero falls, BIGBOX Kick21 is `20/20` Kick, and BIGBOX Carry45 is rejected at
  `8/20` Carry with raw action `68.437`;
- [x] build the profile-disjoint official transition-risk dataset with exact past `10 x 539`
  prefixes, train the 11.012M causal Transformer after a fixed serious overfit, and select the
  frozen threshold only on validation first-50 frames;
- [x] pass the held-out offline first-50 gate (`AUROC=0.7430`, balanced accuracy `0.6536`,
  probability gap `0.2655`) without future outcomes in the deployed model;
- [x] run the exact matched online BIGBOX composition and reject the frame-49 hard switch: `10/20`
  profiles latch to fallback, but the first invalid transition occurs at frame 447 in a correctly
  high-risk (`0.8885`) fallback profile, with one physical fall and action-envelope failure;
- [x] render the exact direct-versus-fallback failure trace with synchronized risk, threshold,
  route, root drift and explicit invalid-frame annotation;
- [x] audit a validation-calibrated decision at the earliest exact anchor 9 and reject the online
  hard-switch continuation: ranking passes, but test Brier `0.2773` loses to prevalence `0.2331`
  under the validation-only threshold `0.84`;
- [x] complete the first serious Carry-9 -> Kick recovery diagnostic with online physical prefix,
  released-Kick warm start/teacher, repository BCPPO and matched frozen video; retain the small
  update-64 displacement/contact/reward gain without upgrading saturated `20/20 -> 20/20` success
  to a difficult-transition claim;
- [x] sweep frozen official Kick over the predeclared Carry prefix grid; record that no strict
  upright majority-failure point exists and retain prefix41 separately as the maximum upright
  failure boundary (`14/20` safe Kick, `6/20` falls);
- [x] run unconstrained and physical-invalid-penalty update-64 recovery at prefix41; reject both
  because neither increases safe success nor reduces falls;
- [x] train official task-wide Carry/Kick TinyMDM priors on motion-ID-disjoint splits and reject
  cross-checkpoint raw-energy comparison despite `19/19` motion-level classification;
- [x] train one shared official conditional TinyMDM for 50,000 iterations and pass the `19/19`
  held-out motion gate with one checkpoint and normalizer;
- [x] connect that prior causally to online prefix41 SUGAR recovery, pass the zero-optimizer
  feature/RNG smoke, and run the matched correct-Kick versus wrong-Carry update-64 pair;
- [x] record the negative physical verdict: both arms are `16/20` safe Kick with `3/20` falls and
  wrong Carry has more contact/displacement, so absolute prior occupancy reward is not admitted;
- [ ] replace absolute occupancy reward with a matched-noise causal state-progress/transition
  objective using the same official conditional prior, then test independent seeds and geometry;
  no future labels, toy latent or weight/update sweep.

Current endpoint: the earlier three-seed result remains negative, the teacher-floor diagnostic
collapses, and the strictly matched seed161587 phase-event run is behaviorally negative because it
was trained with a wrong zero-based phase clock after restoring reference frame 197. Corrected
online and frozen Carry gates now pass, and the official Generator/Tracker Kick test split prefers
Kick21 on `8/9` profiles. The new from-scratch phase-corrected pair now shows partial `3/4`
behavioral movement at update 64 and that pattern repeats in seed161589, but both remain Carry
solutions. The fixed 4x diagnostic then degrades the paired result to `1/4`, so no scale sweep is
allowed. The shared-checkpoint actionable condition is now implemented and causally changes the
same frozen policy's actions, but update 64 still yields bilateral Carry under both demos despite a
`3/4` directional shift. The fixed 3000-step official-action diagnostic then creates a strong
same-checkpoint split: correct remains a stable Carry, while unrelated leaves Carry but falls in
`15/20` and is not a valid Kick solution. The full-510D shared MLP and three-stage DAgger follow-up
also fail closed-loop stability. The complete official-skill router is the first executable
semantic switch: in the identical SMALLBOX scene Carry45 gives `18/20` Carry while Kick21 gives
`19/20` Kick, both with zero falls. The reverse BIGBOX-to-Carry transfer remains rejected. The
causal transition-risk Transformer passes its disjoint offline check, but a frame-49 hard switch
still falls and becomes non-finite in a profile already classified risky. The earliest anchor-9
audit retains useful ranking but fails its predeclared calibration check. The first learned
Carry-9 -> Kick recovery controller is now complete and slightly improves displacement, contact and
reward while preserving `20/20` Kick; the prefix frontier and matched prefix41 recovery are now
complete and negative. The admitted shared conditional TinyMDM passes its motion-disjoint gate,
but its first absolute-occupancy online reward pair gives no correct-condition physical advantage.
The active next task is therefore a matched-noise causal state-progress/transition objective using
that same official prior, not a reward-scale, threshold or optimizer-step sweep.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It remains outside
the active execution queue until the demo-following branch completes or evidence changes priority.
