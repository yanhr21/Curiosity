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
- [ ] obtain explicit authority before new policy training;
- [ ] run a matched policy comparison and judge
  predictor-independent frozen behavior; keep selected-demo SMP out until its semantic gate passes.

Current endpoint: the earlier three-seed policy result remains negative and the teacher-floor
diagnostic collapses. The phase-aware 11.386M event predictor now passes data, model, calibration,
runtime and fixed-demo bidirectional reward-scale gates. The missing evidence is policy-level causal
behavior under a matched rerun; predictor scores alone cannot provide that verdict.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It remains outside
the active execution queue until the demo-following branch completes or evidence changes priority.
