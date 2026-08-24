# Local experiment index

`experiments/` is local-only and Git-ignored. It contains only the checkpoints, traces, videos and
small summaries needed to reproduce or inspect current conclusions. Failed launches, invalid
comparisons, superseded calibrations, duplicate renders, intermediate checkpoints and runtime logs
are under root `legacy/`.

The completed reference-aware matched policy results are
`demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/` and `seed161589/`. The
next execution focus is one fixed 4x feedback-strength overfit diagnostic. This README is the only
experiment index; the stale package manifest has been archived.

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
