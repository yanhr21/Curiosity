# Local experiment index

`experiments/` is local-only and Git-ignored. It contains only the checkpoints, traces, videos and
small summaries needed to reproduce or inspect current conclusions. Failed launches, invalid
comparisons, superseded calibrations, duplicate renders, intermediate checkpoints and runtime logs
are under root `legacy/`.

The completed reference-aware matched policy results are
`demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/` and `seed161589/`. The
fixed 4x feedback-strength overfit and shared action-direction topology diagnostics are also
complete. The full-510D shared-MLP diagnostic is negative, while the official-Tracker router is the
current executable baseline. This README is the only experiment index; the stale package manifest
has been archived.

## 1. `demo_following/`

- `teacher_only_carry45_gate_corrected_v1/`: 20-profile zero-residual prerequisite gate;
- `matched_reward_identity_same_teacher_v1/`: historical three-seed 64-update comparison with
  checkpoints, matched frozen traces/results and videos. It proves reward use but not semantic
  demo following; it is not the active policy experiment;
- `matched_phase_event_reward_v1/seed161587/`: retained negative experiment trained with the wrong
  reset-zero phase clock after restoring CarryBox frame 197. Its policies and videos are failure
  evidence and are rejected by the active evaluator. `scorer_transfer_source_trace_v1/` contains
  the exact 121-D frozen-policy inputs, while `scorer_transfer_phase_ablation_v1/` isolates the
  phase error and restores Carry preference in all four arm/update blocks;
- `corrected_phase_runtime_gate_job258074_compute_v3/`: current pre-optimization evidence. Both
  online arms start at phase step 197 with zero policy updates; frozen Carry scores correctly in
  `20/20` profiles per block, and the motion-disjoint official Generator/Tracker Kick gate passes
  `8/9` profiles. It is scorer admission, not trained-policy behavior;
- `matched_phase_event_reward_reference_aware_v2/seed161587/`: completed reference-aware training,
  update-32/update-64 frozen traces and independent behavior audits. Both checkpoints move in
  `3/4` declared directions, but both policies remain bilateral Carry and foot contact is near
  zero. `videos_update0064_trace_exact/` contains the two H.264 demo/actual videos drawn from exact
  frozen body centers and box poses after the cluster Vulkan renderer failed before valid output;
- `matched_phase_event_reward_reference_aware_v2/seed161589/`: independent replication. Update 64
  repeats the same `3/4` direction pattern with nearly identical transport/orbit deltas; update 32
  does not. Both H.264 videos and complete frozen traces are retained;
- `matched_phase_event_reward_reference_aware_4x_overfit_v1/seed161589/`: fixed same-seed 4x
  magnitude diagnostic. Direction count degrades from baseline `3/4` to `1/4`, paired effects
  reverse and unrelated foot contact does not increase. `STRENGTH_COMPARISON.json` is the compact
  machine-readable verdict; no further scale is active;
- `shared_actionable_demo_conditioning_v1/seed161591/`: one update-64 SUGAR checkpoint conditioned
  on Carry45/Kick21. Condition-only swapping changes actions, but both outcomes remain Carry;
- `shared_topology_distillation_v1/seed161593/`: fixed 3000-step official-action direction
  diagnostic and its 20-profile same-checkpoint evaluation. Correct preserves Carry; unrelated
  increases leg/ground interaction but falls in `15/20`, so it is not valid Kick imitation. The
  two complete H.264 videos are under `videos_fixed_carry_teacher_step3000/`;
- `shared_full_tracker_v2/seed161601/`: serious exact-510D shared-MLP BC and three-stage DAgger
  negative diagnostic. Offline error becomes small, but the final student-only Carry result is
  `6/20` with `14/20` falls; do not extend it with more optimizer steps;
- `official_tracker_router_v1/seed161610/`: current executable baseline. One checkpoint contains
  the parameter-exact Carry/Kick released Tracker experts plus a causal selected-demo router.
  `frozen_eval_final/` and `videos_reference_actual_final/` retain the Tracker-only coupling
  diagnostic. The admitted result is `frozen_eval_joint_final/`: in the exact same SMALLBOX scene,
  Carry45 gives `18/20` Carry and Kick21 gives `19/20` Kick with zero falls; matched BIGBOX Kick21
  gives `20/20` Kick, while BIGBOX Carry45 is rejected at `8/20`. The four decoded H.264 videos are
  under `videos_joint_reference_actual_final/`;
- `cross_skill_recovery_v1/`: fixed online Carry-9 -> Kick recovery diagnostic. Only the finite
  `bcppo_update64_seed171629/` checkpoint pair and matched `bcppo_frozen_eval_seed181629/` traces,
  results and decoded actual-world videos are retained. Both baseline and update64 are `20/20`
  Kick with zero falls; update64 makes a small displacement/contact/reward improvement. Failed
  startup, pure-PPO and post-update64 extension directories are under root `legacy/`;
- `cross_skill_prefix_frontier_v1/seed181630/`: frozen no-training 12-length Carry-prefix sweep.
  `FRONTIER_RESULT.json` records that the strict upright majority-failure frontier is absent and
  selects prefix41 only as the maximum upright failure boundary;
- `cross_skill_recovery_prefix41_v1/`: prefix41 recovery without a termination penalty. It raises
  displacement but worsens falls from `2/20` to `3/20`; retained as the matched negative control;
- `cross_skill_recovery_prefix41_safe_v1/`: the same matched run with a physical-invalid penalty.
  It prevents the fall regression but leaves safe success and falls unchanged at `17/20` and
  `2/20`. The synchronized baseline/update64 H.264 video is under `videos/`;
- `teacher_floor_overfit_v1/`: seed161581 update-128 correct/unrelated endpoints, frozen traces,
  failure videos and the automatic negative behavior gate for common teacher floor `0.25`;
- `contact_event_reward_redesign_v1/reference_corpus_audit/`: 199-motion official Carry/Kick
  reference-event feasibility result; binary labels are reference proxies, not tactile force;
- `contact_event_reward_redesign_v1/deployable_goal_core_corpus_v1/`: 100 Carry plus 99 Kick
  IsaacLab rollouts with the exact 121-D policy core and named physical hand/foot-to-box events;
- `contact_event_reward_redesign_v1/deployable_goal_core_corpus_v1_audit/`: complete coverage,
  exact previous-action slice, force-threshold and motion-disjoint corpus verdict;
- `contact_event_reward_redesign_v1/phase_aware_goal_core_dataset_v1/`: clock-bound causal
  correct/same-task-wrong/cross-task-wrong pairs and 13-D mismatch targets;
- `contact_event_reward_redesign_v1/phase_aware_event_predictor_formal_seed271303_v1/`: frozen
  epoch-20 11.386M V3 predictor, 12/12 held-out gates and validation-only calibration;
- `contact_event_reward_redesign_v1/phase_aware_dense_feedback_scale_audit_v1/`: passing fixed
  Carry45/Kick21 bidirectional semantic gate, runtime scale and update-32/64 stopping rule;
- `predictor/`: frozen 11.9M causal future-mismatch predictor result and checkpoint;
- `smp_prior/`: generic official MimicKit TinyMDM prior used identically by both arms;
- `selected_demo_smp_v1/`: official CarryBox45/KickBox21 single-clip priors and the failed
  independent semantic-extension gate;
- `taskwide_smp_v1/`: motion-ID-disjoint official `10 x 216` Carry/Kick dataset, two task-wide
  TinyMDM priors and the retained warning that independent checkpoint energies are not directly
  comparable despite `19/19` motion-level classification;
- `conditional_taskwide_smp_v1/`: admitted shared official conditional TinyMDM, one normalizer,
  50,000-iteration checkpoint, `19/19` held-out motion gate, training-only reward calibration and
  causal recovery scoring;
- `conditional_smp_online_smoke_v1/`: zero-optimizer online prefix41 feature/RNG/reward gate;
  offline/online feature error is `2.38e-7` and all causal checks pass;
- `conditional_smp_recovery_prefix41_v1/`: matched correct-Kick versus wrong-Carry online reward
  pair at update64. Both frozen arms are `16/20` safe Kick with `3/20` falls; wrong Carry has more
  contact/displacement, so this is condition-use evidence and an absolute-reward negative result.
  `videos_single_seed181633/` contains a matched camera-enabled rollout; it proves that seed only;
- `conditional_smp_progress_recovery_prefix41_v1/` and the seed171633/171634 siblings: three
  matched-noise causal progress pairs. Aggregated correct/wrong totals are `52/50` safe kicks and
  `5/5` falls over 60 profiles per arm; all three seeds reduce foot contact and increase net
  displacement, but physical advantage is not seed-robust. Each directory contains its own
  camera-seed video and proves only that rollout;
- `conditional_smp_progress_three_seed_v1/`: machine-readable three-seed aggregate and strict claim
  boundary;
- `conditional_smp_contrastive_progress_recovery_prefix41_seed171635_v1/` and the seed171636
  sibling: two matched causal contrastive-progress pairs. Aggregated correct/wrong totals are
  `33/32` safe kicks and `3/3` falls over 40 profiles per arm. Both seeds increase net displacement,
  planar path and foot contact under the Kick condition, but only the first has a per-seed physical
  advantage. Each directory contains one independent camera-seed comparison video;
- `conditional_smp_contrastive_progress_two_seed_v1/`: machine-readable two-seed contrastive
  aggregate and the strict `condition_effect_replicated_without_seed_robust_physical_advantage`
  verdict;
- `frozen_expert_transition_prefix41_seed171637_v1/`: first topology-level matched diagnostic.
  Exact frozen selected endpoints plus separately trained serious transition residuals give
  selected Kick/Carry `17/0` safe kicks and `2/1` falls. `CHECKPOINT_AUDIT.json` proves exact expert
  preservation; `videos_seed181643_v2/` is the readable frozen comparison. This is not yet a
  same-checkpoint or safety result;
- `shared_frozen_expert_transition_prefix41_seed171638_v1/`: one shared checkpoint trained with
  exact 32/32 Carry/Kick conditions. Matched condition swap gives `19/0` safe kicks and `0/0` falls;
  initial physics is elementwise identical, exact experts remain unchanged and
  `videos_seed181645/` contains the H.264 comparison. `LEARNING_RESULT.json` compares learned Kick
  with the exact pre-update Kick endpoint and finds no safety improvement (`19/19`, `0/0` falls);
- `shared_frozen_expert_transition_prefix41_seed171639_v1/`: unchanged independent training seed.
  Condition swap gives selected Kick/Carry `17/0` safe kicks and `2/0` falls. The exact matched
  pre-update Kick baseline is `16/20` safe with `0/20` falls, so one additional safe Kick costs two
  falls and is not a safety improvement. `videos_seed181647/` contains its H.264 comparison;
- `shared_frozen_expert_transition_two_seed_v1/`: replicated condition-use aggregate: selected
  Kick/Carry totals are `36/0` safe kicks and `2/0` falls over 40 profiles per condition;
- `shared_frozen_expert_transition_learning_two_seed_v1/`: correct learning-effect aggregate:
  learned/pre-update selected Kick totals are `36/35` safe kicks and `2/0` falls. The verdict is
  `matched_kick_safety_improvement_not_replicated`;
- `frozen_expert_transition_failure_overfit_seed181630_v1/`: fixed-context learnability negative.
  Exact pre-update and update64/128/192/256 safe/fall counts are `14/6`, `13/7`, `14/6`, `13/6`,
  `13/5`. `RESULT.json` records `failure_rich_transition_not_learned_by_update256`; no endpoint is
  admitted and the directory is diagnostic-only;
- `transition_recovery_objective_overfit_seed181630_v1/`: objective-only fixed-context positive.
  Pre/update64/128/192/256 safe/fall counts are `14/6`, `14/5`, `14/5`, `13/6`, `14/6`.
  `RESULT.json` selects the earliest passing update64 and records
  `causal_recovery_objective_is_learnable`; this is not generalization evidence;
- `shared_transition_recovery_objective_prefix41_seed171640_v1/` and seed171641 sibling: balanced
  32/32 formal shared-condition runs at update64 with disjoint evaluation seeds 181648/181650.
  Each keeps learned-Kick, Carry-condition and exact-pre-update-Kick frozen results plus decoded
  world videos. Learned/pre-update are tied at `18/18 safe, 2/2 fall` and `17/17, 2/2`;
- `shared_transition_recovery_objective_two_seed_v1/`: formal aggregate. Learned and exact
  pre-update Kick both total `35/40` safe and `4/40` falls, so causal recovery benefit does not
  generalize despite replicated selected Kick/Carry condition separation (`35/3` safe kicks);
- `multi_context_transition_recovery_seed171642_v1/` and seed171643 sibling: one shared serious
  controller cycles physical Carry-prefix handoffs `41/49/57` online. Both endpoint audits pass,
  all learned/pre-update comparisons restore elementwise-identical initial physics, and each
  directory contains three learned-versus-pre-update world videos. Per-seed safe/fall totals are
  `47/3 vs 47/4` and `50/7 vs 50/7`;
- `multi_context_transition_recovery_two_seed_v1/`: strict aggregate over 120 profiles per
  endpoint. Learned/pre-update are `97/10` versus `97/11` safe/fall, but only one seed improves;
  `RESULT.json` records `multi_context_kick_safety_improvement_not_replicated`;
- `causal_action_composition_seed171644_autorun_v1/` and its seed171645 replication: exact frozen
  Carry/Kick endpoints plus state-dependent action composition. The matched two-seed trained-prefix
  result is learned/pre `94/92` safe and `17/20` falls, but the effect is more conservative and is
  restricted to `41/49/57`;
- `causal_composition_heldout_prefix33_65_v1/`: frozen no-training audit of both admitted
  checkpoints on unseen `33/65`. Learned/pre tie at `68/68` safe and `1/1` falls, rejecting
  handoff-prefix generalization; four paired world videos are retained;
- `causal_action_composition_dense_prefix_seed171646_v1/`: unchanged update64 composer trained on
  `33/41/49/57/65` and frozen only on interleaved `37/45/53/61`. Learned/pre are `72/73` safe and
  `4/3` falls; all four paired world videos decode and the automatic rule skips seed171647;
- `causal_action_composition_dense_prefix_seed171646_v1_frozen_ablation/`: no-training gate-only
  and residual-only output-row ablation of the dense checkpoint. Full/gate-only/residual-only/pre
  total `72/72/71/73` safe and `4/4/4/3` falls. The result rejects a one-path explanation and keeps
  only the two changed-profile 2x2 videos;
- `causal_action_composition_dense_prefix_seed171646_v1_seen_context_audit/`: no-training frozen
  learned/pre audit on the actual training prefixes. Learned/pre is `92/91` safe and `5/5` falls;
  only prefix33 profile17 changes safe outcome. Combined with interleaved prefixes, the complete
  nine-prefix grid ties `164/164` safe and learned worsens falls `9/8`; one paired world video is
  retained;
- `runtime_assets/`: compact frozen inputs required by current scripts.

The old 1216-update policy experiment, rejected 510-D/free-window event predictor, redundant CPU
gate, empty runtime attempts and obsolete logs are under root `legacy/`; none is active evidence.

## 2. `online_patch_tactile_mass_adaptation/`

- `corrected_rerun_20260820/`: only the model1100 tactile-only checkpoint/config and the
  model1100/model1250 frozen diagnostic summaries;
- `leakage_sweep_v1/`: three-seed, five-mass fixed-action proprioception/tactile leakage traces;
- `friction_feasibility_after_ps/`: independent frozen-Refiner `4 friction x 2 mass` sweep;
- `visualizations/official_refiner_mu1p5_6x_friction_hold_single_env/`: the one retained heavy-box
  synchronized world/bilateral-27-patch video;
- `runtime_assets/`: preconverted G1 asset required by active collectors.

No valid corrected matched Z/P/PS result exists. Historical formal endpoints, evaluations and
their videos are archived and must not be used for a tactile-benefit claim.

## 3. `isaaclab_g1_anatomical27_object_demos/`

Four retained native IsaacLab/PhysX sensor examples:

- normal CarryBox;
- free palm-area lift;
- 2 kg palm grip;
- release/failure behavior.

Each directory keeps the raw online trace, summary, world video, synchronized bilateral tactile
video and render metadata so the visualization can be independently regenerated.

## 4. `sugar_reproduction/`

- `outputs/final/official_sugar/`: frozen Refiner checkpoint, baseline summaries/videos and one
  released CarryBox inference;
- `assets/official_tacsl/`: official R15 USD and GelSight calibration.

Routine logs have been archived. Exact commands and result expectations are in
[`DOCS/reproducibility.md`](../DOCS/reproducibility.md).
