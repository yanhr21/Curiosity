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
- [x] record the trace limitation: foot identity and direct kick contact cannot be recovered because
  per-body pose and foot-contact force were not archived;
- [x] predeclare three matched seed pairs for the fixed 64-update repeat;
- [ ] add per-body pose, named foot-box contact and hand-box-only contact to the next frozen
  evaluator as evaluation labels, not actor inputs;
- [ ] after explicit approval, run seed pairs `161583/161584` and `161585/161586` serially and
  evaluate them with seeds `171583` and `171585`;
- [x] add a one-seed-at-a-time runner and a training-seed-level behavior aggregator; neither
  substitutes physics profiles for independent seeds;
- [ ] only if the multi-seed result is stable, test an identical teacher-authority schedule in both
  arms;
- [ ] redesign or replace the semantic reward gate before any selected-demo SMP integration.

Current endpoint: correct `16/20` success, unrelated `18/20`, with two physical falls in each arm.
The independent audit observes `0/4` declared semantic directions: both policies remain Carry-like.
This proves reward-signal use and within-Carry behavior change, not semantic demo following.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It is frozen; do
not resume training, evaluation or scale tuning without explicit user authorization.
