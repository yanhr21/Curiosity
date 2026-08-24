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
- [ ] run the `161589/161590 -> 171589` independent-seed replication with the exact same teacher,
  phase, 64-update budget, reward weights, physics and evaluation; do not integrate SMP or tune
  weights first.

Current endpoint: the earlier three-seed result remains negative, the teacher-floor diagnostic
collapses, and the strictly matched seed161587 phase-event run is behaviorally negative because it
was trained with a wrong zero-based phase clock after restoring reference frame 197. Corrected
online and frozen Carry gates now pass, and the official Generator/Tracker Kick test split prefers
Kick21 on `8/9` profiles. The new from-scratch phase-corrected pair now shows partial `3/4`
behavioral movement at both checkpoints but remains a Carry solution. The next work is one exact
independent-seed replication, not a reward-weight or training-budget sweep.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It remains outside
the active execution queue until the demo-following branch completes or evidence changes priority.
