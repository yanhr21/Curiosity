# TODO index

## Demo following

- [x] remove tactile-surface contact penalties from the goal task;
- [x] replace demo-anchor and root-up-axis false fall checks with relative physical root-height
  loss;
- [x] pass the CarryBox45 teacher-only bilateral-contact and 5 cm lift gate on 20 profiles;
- [x] train correct/unrelated selected-demo arms for 64 updates with the same CarryBox45 teacher;
- [x] freeze-evaluate 20 matched physics profiles per arm and render complete demo/actual videos;
- [x] verify that selected-demo identity changes checkpoint parameters and residual actions;
- [ ] compute predictor-independent behavior adherence from existing traces: approach direction,
  contact topology, foot interaction, path around the box, box displacement and lift;
- [ ] predeclare at least three matched training seeds and repeat the fixed 64-update design;
- [ ] only if the multi-seed result is stable, test an identical teacher-authority schedule in both
  arms;
- [ ] redesign or replace the semantic reward gate before any selected-demo SMP integration.

Current endpoint: correct `16/20` success, unrelated `18/20`, with two physical falls in each arm.
This proves reward-signal use, not semantic demo following.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It is frozen; do
not resume training, evaluation or scale tuning without explicit user authorization.
