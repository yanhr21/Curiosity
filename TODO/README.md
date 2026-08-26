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
- [x] replace absolute occupancy reward with a matched-noise causal state-progress objective using
  the same official conditional prior and complete three matched training/evaluation seeds;
- [x] record the strict verdict: `52 vs 50` safe kicks and equal `5 vs 5` falls over 60 profiles,
  with repeatable displacement/contact shifts but no seed-robust physical advantage;
- [x] run one fixed causal contrastive transition-margin diagnostic using
  `loss(alternative) - loss(selected)` progress under matched diffusion noise; no future labels,
  toy latent, reward-weight sweep or extra update budget.
- [x] complete the independent-seed repeat: `33 vs 32` safe kicks and `3 vs 3` falls over 40
  profiles per arm; both seeds increase Kick-conditioned displacement/path/contact, but physical
  advantage is not seed-robust.
- [x] implement the next topology-level experiment: frozen exact Carry/Kick Generator+Tracker
  endpoints plus a serious causal state/demo-conditioned transition controller; first verify exact
  endpoint passthrough, then run one matched transition diagnostic without scalar-reward tuning.
- [x] complete the first separate-arm topology diagnostic: `17 vs 0` safe kicks and `2 vs 1` falls;
  retain the strong semantic split but reject a safety or same-checkpoint claim.
- [x] train one shared frozen-expert transition checkpoint with balanced Carry/Kick selected-skill
  IDs, then evaluate the identical checkpoint under condition swap and matched frozen physics.
- [x] pass the first shared-checkpoint condition-use test: Kick/Carry `19/0` safe kicks, `0/0`
  falls and elementwise-identical physical handoff; do not use wrong Carry as the safety baseline.
- [x] run the independent shared-checkpoint seed with unchanged 32/32 balance and 64-update budget;
  aggregate `36/0` safe Kick under Kick/Carry conditions with `2/0` falls.
- [x] add the exact matched learning baseline: compare `model_64` Kick with `model_pre_update` Kick
  on the same evaluation seed and initial physics. Learned/pre-update totals are `36/35` safe Kick
  and `2/0` falls, so safety improvement is false in both seeds.
- [x] run the fixed failure-rich serious overfit diagnostic. Pre/update64/128/192/256 safe/fall
  counts are `14/6`, `13/7`, `14/6`, `13/6`, `13/5`; reject learnability because no endpoint
  reduces falls without losing safe Kick.
- [x] add the causal current-rollout recovery objective and rerun the fixed overfit. Update64/128
  improve `14 safe/6 fall -> 14/5`; update192/256 regress, so select earliest passing update64.
- [x] run two formal balanced 32/32 shared-condition update64 experiments with disjoint one-to-one
  evaluation seeds. Learned and exact pre-update Kick both total `35/40` safe and `4/40` falls;
  safety improvement is false for both seeds, while same-checkpoint Kick/Carry condition use remains
  replicated at `35/3` safe kicks.
- [x] replace the single-prefix training context with online physical handoffs `41/49/57` and run
  two independent update64 seeds. Learned/pre-update totals are `97/10` versus `97/11`
  safe/fall, but only seed171642 improves; reject replicated safety benefit.
- [x] implement the causal state-dependent Carry/Kick composer over current state, both released
  commands and selected skill. Preserve both exact frozen endpoints, expose the complete `[0,1]`
  mixture range, keep future/outcome labels out of the actor, and record mixture/residual/final
  action attribution in frozen evaluation.
- [x] run the fixed seed171644 update64 IsaacLab/PhysX multi-context comparison, frozen learned/pre
  evaluation and videos, then automatically run `171645 -> 181658` after the first positive result.
  The two-seed aggregate is learned/pre `94/92` safe and `17/20` falls; both seeds pass, all six
  paired H.264 videos decode, and no human authorization gate was used.
- [x] audit the replicated causal-composition mechanism without new policy training. Prefix41 shows
  `88x/97x` endpoint-gap amplification and gains safe profiles on both seeds; prefix49 contains both
  a fall-to-safe recovery and one lost success; prefix57 is residual-dominated and loses one safe
  profile on seed171645.
- [x] freeze both learned/pre checkpoint pairs and evaluate held-out physical prefixes `33/65` with
  the same disjoint seeds and strict metrics. Across 80 profiles per endpoint, learned/pre are tied
  at `68` safe kicks and `1` fall; both seeds tie independently, so trained-context safety benefit
  does not generalize to these adjacent unseen handoffs.
- [x] run one dense-coverage causal-composer seed with training prefixes `33/41/49/57/65` and
  frozen learned/pre evaluation only on interleaved unseen `37/45/53/61`. Preserve update64,
  architecture, reward, optimizer, balanced conditions and strict metrics; automatically replicate
  as `171647 -> 181664` only if `171646 -> 181662` passes the same physical safety rule. The first
  seed is negative at learned/pre `72/73` safe and `4/3` falls, so replication was automatically
  skipped.
- [x] freeze the dense checkpoint into gate-only and residual-only output-row ablations without
  optimizer updates. Totals are full/gate-only/residual-only/exact-pre `72/72/71/73` safe and
  `4/4/4/3` falls; the two changed full outcomes are non-additive and neither path deletion is a
  validated repair.
- [x] freeze-evaluate the same dense learned/pre pair on its seen prefix schedule
  `33/41/49/57/65` with evaluation seed181662 and strict v4 metrics. Learned/pre is `92/91` safe
  and `5/5` falls; only prefix33 profile17 gains a safe outcome. Across the combined nine-prefix
  `33..65` grid, safe ties `164/164` and learned falls worsen `9/8`, closing this composer.
- [ ] audit the official MimicKit/TinyMDM code and admitted checkpoint for a documented stable
  selected-demo representation usable as causal motion deviation. Use official interfaces and
  adapter code only; record an interface blocker instead of substituting a toy or arbitrary hidden
  activation. Define the next matched topology experiment only after this audit.

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
The later matched-noise progress objective produces repeatable condition-dependent behavior across
three seeds but no seed-robust physical advantage. The two-seed contrastive follow-up likewise
produces a repeatable activity shift without robust success/safety. The scalar-prior reward family
is therefore closed for this setup. The first separate-arm frozen-expert controller creates a large
semantic split but has no safety advantage and retains a checkpoint confound. The shared conditioned
checkpoint removes that confound and replicates condition-dependent endpoint execution across two
seeds. The matched exact pre-update comparison rejects a learned safety benefit: `36/35` safe Kick
but `2/0` falls. The fixed failure-rich overfit also fails its learnability rule through update256.
The causal physical recovery objective passes fixed-context learnability only at update64/128.
Its two-seed balanced formal test is negative: learned/pre-update both give `35/40` safe Kick and
`4/40` falls. Multi-context failure-rich training is now also complete: learned/pre-update give
`97/97` safe Kick and `10/11` falls, but the one-fall aggregate difference occurs in only one of
two seeds. The active next task is state-dependent frozen-expert action composition, not another
update extension, reward-scale sweep or extra residual seed.

## Frozen tactile work

[`15_online_patch_tactile_mass_adaptation/todo.md`](15_online_patch_tactile_mass_adaptation/todo.md)
records the completed bug fixes, diagnostics and unfinished matched Z/P/PS work. It remains outside
the active execution queue until the demo-following branch completes or evidence changes priority.
