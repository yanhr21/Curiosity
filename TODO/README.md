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
- [x] obtain explicit authority and run the phase-event matched pair serially for exactly 64 updates;
- [x] freeze-evaluate update 32/64 on 20 matched physics profiles per checkpoint and render the two
  complete demo/actual H.264 videos;
- [x] fix evaluator-only batch restoration for 20-training-env wrapper/physics state versus the
  40-env two-checkpoint evaluation scene;
- [x] fix behavior analysis so one-dimensional trace metadata cannot be indexed as environment data;
- [x] reject semantic following at update 32 and 64: the two policies remain nearly identical Carry
  solutions and satisfy only `2/4` and `1/4` predeclared Kick-like directions;
- [x] identify that the frozen scorer rates the correct Carry rollout closer to Kick21 than Carry45
  over most middle frames;
- [ ] rescore the same frozen trajectories with the exact recorded 121-D prefix and correctly restored
  source phase; do not train a new policy for this step;
- [ ] separate phase-offset error from Tracker-to-Refiner rollout-domain shift with matched scorer-only
  ablations;
- [ ] if phase correction is insufficient, collect a motion-disjoint Refiner-plus-residual corpus and
  re-admit the serious predictor on both Tracker and policy-rollout domains;
- [ ] keep selected-demo SMP out until its independent semantic gate passes; do not start another
  policy pair until actual frozen Carry trajectories prefer Carry45 consistently.

Current endpoint: the earlier three-seed result remains negative, the teacher-floor diagnostic
collapses, and the new strictly matched seed161587 phase-event run is also behaviorally negative.
Its frozen score exposes a transfer failure: a strong Carry rollout receives lower predicted
mismatch under Kick21 than Carry45. The next work is scorer phase/domain audit on existing frozen
trajectories, not more PPO training.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It remains outside
the active execution queue until the demo-following branch completes or evidence changes priority.
