# Demo following and released-skill transition

The active line preserves the exact released CarryBox45 and KickBox21 Generator+Tracker endpoints
and learns only a causal state/demo-conditioned transition between them. The earlier same-teacher
`phase_event_reward_only` experiment remains documented below as historical evidence; it is not the
current execution queue. Active experiments advance through machine-checkable outcomes without a
human-authorization state.

## Current transition-recovery verdict

The causal recovery objective first failed its balanced single-prefix formal test on seeds
`171640/171641`: learned and exact pre-update Kick are tied at `35/40` safe and `4/40` falls.
The completed multi-context follow-up cycles online physical Carry handoffs `41/49/57` and uses
training/evaluation pairs `171642 -> 181652` and `171643 -> 181654`. Learned versus exact pre-update
totals are `97/120` safe for both and `10/11` falls. Only seed171642 improves, so the benefit is not
replicated. Same-checkpoint selected Kick/Carry behavior remains distinct. Do not add residual
updates, reward scales or a third seed. The later state-dependent and temporal composition
controllers documented below are also complete and do not improve the final matched physical grid.
The active work is official CHORD demonstration-contact data recovery, not another controller sweep.

The causal-composition implementation is a full `512/256/128` SUGAR-topology module over the
current `510-D` Tracker observation, current Carry/Kick `36-D` commands and selected-skill one-hot.
Its zero-initialized 30-D output gives the exact selected released endpoint at update 0, then learns
one bounded Carry/Kick mixture weight over the complete `[0,1]` convex segment plus a 29-D residual.
Both official Tracker experts remain
parameter-frozen, and no future outcome enters the actor. Run the fixed seed171644 diagnostic with:

```bash
bash scripts/sugar/demo_following/run_causal_action_composition_transition_recovery.sh \
  experiments/demo_following/causal_action_composition_seed171644_v1 cuda:0
```

This command performs update64 training, checkpoint audit, frozen learned/pre-update evaluation at
prefixes `41/49/57`, a selected-Carry control, and six world videos. It sets the cluster system
NVIDIA Vulkan ICD for both evaluation and rendering. The physical result remains open until the
generated `RESULT.json`, frozen traces and videos pass; interface tests alone are not evidence of
recovery benefit. After recording `PIPELINE_STATUS.env`, the command keeps the retained GPU above
the cluster utilization floor; switch tasks only through the recorded launcher child PGID. A
negative first seed stops scientific spending and enters the holder. A positive first seed
automatically launches the fixed independent replication `171645 -> 181658` with video seed
`181659`, then writes a two-seed aggregate before entering the holder. This decision is read from
the physical frozen-evaluation result and has no manual authorization state.

New causal-composition evaluations use the strict v4 Kick metric: 5 cm planar net motion, at least
1 cm of motion on intervals adjacent to foot contact, and at least 3 cm of path after first foot
contact. The historical any-contact-plus-net-displacement value remains visible only as a legacy
count; it cannot by itself make the first-seed decision positive.
Physical safety is likewise strict: either 35 cm root-height loss or 60-degree root tilt is a fall.
The evaluator reports height-only and tilt components separately.

The dense coverage test keeps that implementation fixed, trains only seed171646 on
`33/41/49/57/65`, and freezes it on interleaved `37/45/53/61`:

```bash
bash scripts/sugar/demo_following/run_dense_prefix_causal_composition.sh \
  experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1 cuda:0
```

It is negative: learned/exact-pre total `72/73` safe and `4/3` falls. The automatic rule skips the
second seed. Attribute its two changed outcomes without training or optimizer updates with:

```bash
bash scripts/sugar/demo_following/run_dense_prefix_composer_ablation.sh \
  experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1 \
  experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1_frozen_ablation \
  cuda:0
```

This creates exact frozen gate-only and residual-only checkpoints by zeroing only the relevant
final output rows. Full/gate-only/residual-only/exact-pre total `72/72/71/73` safe and `4/4/4/3`
falls. Both isolated paths reproduce prefix53 profile6's fall; neither isolated path reproduces
prefix61 profile14's lost safe outcome.

The fitted-context audit reuses the same checkpoints, runs no optimizer step, and evaluates the
actual training prefixes with the same unseen physical seed:

```bash
bash scripts/sugar/demo_following/run_dense_prefix_seen_context_audit.sh \
  experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1 \
  experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1_seen_context_audit \
  cuda:0
```

The completed result is learned/pre `92/91` safe and `5/5` falls. Only prefix33 profile17 gains a
safe outcome; the other four fitted prefixes tie. Combining these results with interleaved
`37/45/53/61` gives `164/164` safe and `9/8` falls over the full nine-prefix grid. The runner labels
the seen relation explicitly and renders only the one physical safe-outcome change. This closes the
current feed-forward composer; audit the official MimicKit/TinyMDM representation interface before
defining another policy topology.

Reproduce one fresh multi-context seed inside a retained GPU step with:

```bash
TRAIN_SEED_OVERRIDE=171642 EVAL_SEED_OVERRIDE=181652 VIDEO_SEED_OVERRIDE=181653 \
  bash scripts/sugar/demo_following/run_multi_context_transition_recovery.sh \
  experiments/demo_following/reproduce_multi_context_seed171642 cuda:0
```

The runner continuously trains, audits, evaluates all three contexts and renders the paired H.264
videos. It contains no manual authorization stage.

## Official CHORD one-variable causal test

The completed CHORD test keeps the serious causal temporal composer, both released Tracker experts,
the online Carry prefix schedule, seeds and frozen evaluation fixed. The ON arm adds only NVIDIA's
pinned official contact-wrench reward kernels. Its command is the reconstructed KickBox21
foot-to-box geometry; its current contact points and force directions come from the live PhysX
rollout. Neither the old binary contact label nor CHORD values enter the actor observation.

Inside the retained H200 step, run the full automatic smoke, OFF, ON, frozen evaluation and video
pipeline with:

```bash
bash scripts/sugar/demo_following/smoke_official_chord_runtime.sh \
  experiments/demo_following/official_chord_runtime_smoke_seed171648_v2 cuda:0
bash scripts/sugar/demo_following/run_official_chord_causal_matched_pair.sh \
  experiments/demo_following/official_chord_causal_matched_seed171648_v1 cuda:0
```

Training prefixes are `33/41/49/57/65`; disjoint frozen-evaluation prefixes are
`37/45/53/61`; training/evaluation/video seeds are `171648/181666/181667`; both arms stop at
update 64. The fixed CHORD term mirrors the released weights:
`10 * contact_wrench_support - 10 * unintended_contact - missed_contact`. The final claim is based
on frozen physical safe-kick/fall outcomes, not reward magnitude or action difference. Four final
H.264 comparisons place CHORD OFF and ON world rollouts side by side. The runner proceeds through
all stages automatically and retains the GPU after completion.

The matched frozen result is negative for physical benefit: OFF/ON safe Kick is `77/80` versus
`76/80`, with `3/80` falls in both arms. ON nevertheless improves mean CWS by `0.00775` and reduces
missed-contact by `0.01137`, so it changes contact representation without improving safety.

If the numerical pair was completed without cameras, render only the two learned policies (not the
internal pre-update diagnostics) with:

```bash
bash scripts/sugar/demo_following/render_official_chord_causal_matched_pair.sh \
  experiments/demo_following/official_chord_causal_matched_seed171648_v1 cuda:0
```

## Current executable baseline: causal official-Tracker router

The current baseline stores the exact released CarryBox and KickBox Tracker actors in one
checkpoint and trains only a router over the frozen 798-D causal selected-demo condition. It is a
two-skill selector, not arbitrary-demo imitation. Inside a retained GPU compute step:

```bash
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
$PYTHON scripts/sugar/demo_following/train_official_tracker_router.py
```

Freeze-evaluate the four complete official Generator+Tracker routes serially with:

```bash
OUTPUT_ROOT="$PWD/experiments/demo_following/official_tracker_router_v1/seed161610/frozen_eval_joint_final" \
  bash scripts/sugar/demo_following/run_joint_generator_tracker_router_eval.sh
```

The runner uses seed `171610` for both Carry conditions and `171611` for both Kick conditions, so
each within-domain pair has an exact common initial and post-prefix state. It automatically records
the expected rejected BIGBOX-to-Carry45 transfer and continues to the final Kick21 control; there
is no human authorization gate between arms.

Render the four admitted traces with:

```bash
OUT="$PWD/experiments/demo_following/official_tracker_router_v1/seed161610"
EVAL="$OUT/frozen_eval_joint_final"
$PYTHON scripts/sugar/demo_following/render_official_tracker_router.py \
  --carry-correct-dir "$EVAL/carry_carry45" \
  --carry-unrelated-dir "$EVAL/carry_kick21" \
  --kick-correct-dir "$EVAL/kick_carry45" \
  --kick-unrelated-dir "$EVAL/kick_kick21" \
  --output-dir "$OUT/videos_joint_reference_actual_final" \
  --source-env 0 --joint-generator-route
```

The retained result is SMALLBOX Carry45 `18/20` Carry and SMALLBOX Kick21 `19/20` Kick, both with
zero falls; matched BIGBOX Kick21 is `20/20` Kick with zero falls. The joint route fixes the old
Tracker-only Carry-to-Kick action explosion (`5.87e11 -> 5.1712`). Reverse BIGBOX-to-Carry45 is
rejected at `8/20` Carry and raw action `68.437`, which is the remaining cross-asset boundary.

Validate without simulation:

```bash
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
$PYTHON scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --dry-run
```

Inside a retained GPU allocation, validate the formal inner runner and frozen model without
creating the environment or executing PPO:

```bash
$PYTHON scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-admission-only
```

Before policy optimization, execute the real 24-step environment/reward path with zero optimizer
updates:

```bash
$PYTHON scripts/sugar/demo_following/run_matched_state_predictor.py \
  --design phase_event_reward_only --arm correct \
  --endpoint-updates 64 --stop-after-segment --runner-rollout-smoke-only
```

The demo-only arms use the original SUGAR G1/CarryBox scene and construct no TacSL sensor; exact-zero
tensors alone are not accepted as zero sensor use. On 2026-08-24, a fresh H200 minimal canary and
both correct/unrelated 24-step smokes passed. The runner uses the same system NVIDIA ICD as that
canary. These smokes prove the online reward path and zero optimizer activity, not learned behavior.

Run the two arms serially inside a retained `srun` GPU compute step. The training-only component uses
a fresh output root and checks each endpoint before starting the next arm:

```bash
OUTPUT_ROOT="$PWD/experiments/demo_following/matched_phase_event_reward_reference_aware_v2" \
  bash scripts/sugar/demo_following/run_reference_aware_phase_event_pair.sh
```

For retained allocations, launch `run_reference_aware_phase_event_pair_then_hold.sh`; after both
endpoint proofs pass it automatically evaluates updates 32/64, runs the independent behavior audit,
renders the correct/unrelated videos, and then returns the GPU to a CUDA hold after success or
failure instead of releasing the allocation. No manual approval separates these predeclared stages.

The evaluation/render stage called by the retained wrapper is:

```bash
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 phase_event_reward_only /absolute/path/to/output/seed161587
```

Set `TEACHER_ONLY_GATE=1` on the same command to run the zero-residual prerequisite gate. Existing
complete results and videos are validated and reused; incomplete directories are never silently
overwritten.

The first reference-aware pair is complete. Its independent audit observes `3/4` predeclared
directions at both update 32 and update 64, while both policies remain bilateral Carry and do not
reproduce Kick21 foot contact. This is a single-seed partial behavior shift, not semantic-following
proof. The independent replication with `161589/161590 -> 171589` is complete. Update 64 repeats
the exact `3/4` direction pattern with nearly identical lifted/ground transport and orbit deltas;
update 32 does not replicate. Both policies remain Carry and foot interaction stays near zero.
The same-seed 4x feedback-strength diagnostic is complete: update 64 degrades `3/4 -> 1/4`, effects
reverse and unrelated foot contact does not increase despite cumulative feedback reaching `-48.64`.
Do not add a second scale. The shared actionable actor is now implemented and evaluated: one
update-64 checkpoint reads a 798-D condition from the frozen serious predictor before each action.
The exact same checkpoint and initial state are evaluated once with Carry45 and once with Kick21.
Residual actions differ by mean/max absolute `0.01319/0.37943`, and behavior moves in `3/4`
predeclared directions, but both conditions remain bilateral Carry.

Reproduce the shared run inside a retained compute allocation:

```bash
$PYTHON scripts/sugar/demo_following/run_shared_actionable_demo_conditioning.py
bash scripts/sugar/demo_following/evaluate_shared_actionable_demo_pair.sh
```

The second command reuses admitted frozen results, performs the predictor-independent behavior
audit and writes two H.264 videos under
`experiments/demo_following/shared_actionable_demo_conditioning_v1/seed161591/`.

The fixed contact-topology diagnostic is also complete. It uses the existing shared actor and
official Carry45/Kick21 Tracker action directions; it is an actor-residual overfit diagnostic, not
a replacement model or final policy result. Run it serially inside a retained allocation:

```bash
$PYTHON scripts/sugar/demo_following/train_shared_topology_distillation.py
bash scripts/sugar/demo_following/evaluate_shared_topology_distillation_pair.sh
```

The first command executes exactly 3000 optimizer steps. The second freezes that one checkpoint,
evaluates 20 matched Carry initial states under correct and unrelated conditioning, runs the
predictor-independent audit and renders two exact-trace H.264 videos. Correct remains a stable
Carry (`0/20` falls); unrelated leaves Carry but falls in `15/20`. This proves a strong
condition-dependent split, not successful KickBox imitation. The next implementation must learn
from both official Carry and Kick physical rollout distributions; extending this fixed Carry
Refiner residual diagnostic is not the active route.

On the current cluster, NVIDIA Vulkan camera rendering repeatedly loses the H200 device during
scene creation before a valid frame is emitted. This does not invalidate camera-free frozen traces.
The exact-trace fallback renders recorded robot body centers and box pose without rerunning physics:

```bash
$PYTHON scripts/sugar/demo_following/render_frozen_trace_behavior.py \
  --correct-trace experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/evaluation_update0064/correct/TRACE.npz \
  --unrelated-trace experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/evaluation_update0064/unrelated/TRACE.npz \
  --output-dir experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587/videos_update0064_trace_exact
```

The output label explicitly says `no physics replay`; the pose-marker box size is illustrative,
while all body centers, object positions and object orientations are exact trace/reference values.

The admitted teacher prerequisite is `teacher_only_gate_no_tactile_v2`. It uses the original SUGAR
scene with no TacSL assets and requires a runtime scene proof in addition to exact-zero residual,
nominal PhysX readback, bilateral contact and a 5 cm lift. The job257815 run passes all 20 profiles;
the observed lift range is `0.6854--0.7224 m` with no physical robot fall.

Each active phase-event training proof also contains `no_tactile_startup_physics`, the exact
per-environment standard-SUGAR material/mass/inertia/COM readback. Frozen evaluation restores this
record before stepping. Both job257815 online smokes pass the record gate and have exactly matched
physics, actions and base rewards.

The rollout smoke additionally performs a no-optimizer reward-to-gradient admission. It computes
PPO returns and normalized advantages with the stored total reward, repeats them after subtracting
only selected-demo feedback, restores total storage, and compares exact clipped actor-surrogate
gradients. The admitted correct/unrelated gradient deltas are `0.07804/0.04430`; parameters and all
optimizer counters remain unchanged. Probe launchers require a machine-readable passing result and
do not trust process return code alone.

The admitted protocol is now `sugar_phase_event_online_rollout_gradient_authority_smoke_v3`. It also
checks `executed=teacher+residual`, coefficient/scale `1.0/1.0`, exact ActionManager raw input and the
existing `2e-6` policy-unit round-trip tolerance. Both arms pass with a nonzero residual maximum of
`3.72674` and round-trip error `4.77e-7`.

The archived historical design changed both teacher and reward demo and is not an active result.
Exact assets, commands, expected outputs and claim boundaries are in `DOCS/reproducibility.md`.

The fixed-physics teacher-floor diagnostic is reproduced inside a retained allocation with:

```bash
bash scripts/sugar/demo_following/run_teacher_floor_overfit_pair.sh
```

It resumes both seed161581 update-64 arms, executes 64 new updates while the common teacher moves
from `1.0` to `0.25`, then freezes, evaluates and renders. The recorded result is behavioral
collapse in both arms, not Carry/Kick semantic separation. The subsequent historical branch was
the contact/event reward redesign documented in `PLAN/README.md`; do not repeat this schedule over
more seeds.

The final causal temporal-composer diagnostic is reproduced inside a retained allocation with:

```bash
bash scripts/sugar/demo_following/run_dense_prefix_causal_temporal_composition.sh \
  experiments/demo_following/causal_temporal_composition_dense_prefix_seed171648_v1 cuda:0
```

It freezes the released Carry/Kick experts and trains a six-layer, 384-D, eight-head Transformer
over the causal past `10 x 584` transition record. The zero-initialized gate/residual head makes the
pre-update checkpoint exactly equal to the selected expert. The launcher automatically performs
interleaved and seen-prefix frozen evaluation, renders paired H.264 videos and combines the complete
`33..65` grid; there is no human approval state. Seed171648/181666 gives learned/exact-pre
`169/170` safe kicks and `7/7` falls over 180 profiles, so this topology is closed rather than
extended with an update, reward, history-length or model-size sweep.

Apply the released NVIDIA CHORD contact-wrench representation to the known prefix53 boundary with:

```bash
bash scripts/sugar/demo_following/run_chord_contact_geometry_collection.sh \
  experiments/demo_following/chord_contact_geometry_phase_aligned_prefix53_v1 cuda:0
```

The runner records live filtered PhysX foot-box contact points/forces for exact-pre, learned and the
released Kick21 expert, aligns them by exact SUGAR motion frame, calls the official CHORD functions
from pinned commit `5654c50e`, and enters the retained GPU holder. The fixed expert reference is
profile 0; do not aggregate asynchronous expert contacts per frame. Render the synchronized evidence
from another compute shell with:

```bash
$PYTHON scripts/sugar/demo_following/render_chord_contact_geometry.py \
  --collection-root experiments/demo_following/chord_contact_geometry_phase_aligned_prefix53_v1 \
  --output-dir experiments/demo_following/chord_contact_geometry_phase_aligned_prefix53_v1/visualizations/chord_exact_trace_v1
```

Exact-pre/learned CWS is `0.06574/0.04949`; the physical result ties at `19/20` safe and one fall.
This is a robot-expert representation diagnostic. Raw SUGAR demos lack contact positions, normals
and part IDs, so the result must not be called a human-demo CHORD reward or training result.

Recover demonstration contact geometry from the exact retargeted G1/object trajectories with:

```bash
bash scripts/sugar/demo_following/run_sugar_demo_chord_geometry_reconstruction.sh \
  experiments/demo_following/sugar_demo_chord_geometry_v2 cuda:0
```

The script uses the released CHORD `approximate_contact_with_id` body at commit `5654c50e`, its
official 1 cm threshold, official SMALLBOX/BIGBOX USD assets and G1 collision surfaces. It writes
contact position, normal and explicit single-object part ID for each left/right role. The archived
binary label is validation-only and is never an input. Render both complete references with:

```bash
bash scripts/sugar/demo_following/run_sugar_demo_chord_geometry_render.sh \
  experiments/demo_following/sugar_demo_chord_geometry_v2 cuda:0
```

The H.264 files use exact recorded 35-body centres and exact object pose; they are geometric
evidence, not a physics replay. Carry45 agrees with the independent binary timing at
`96.71%/98.99%` precision/recall. Kick21 exposes an archived-label defect rather than hiding it:
only 8 of 275 binary-positive frames overlap the 19 physically localized mesh contacts.

### Released motion-latent gates

Run the released TMR task-semantic audit inside retained compute with:

```bash
bash scripts/sugar/demo_following/run_official_tmr_motion_latent_gate.sh \
  experiments/demo_following/official_tmr_semantic_gate_v1 cuda:0
```

This maps the exact 35 G1 body centres into the explicit 22-joint HumanML3D topology, calls the
official `joints_to_guofeats` path and frozen 256-D encoder, and never fits a classifier. It passes
Carry/Kick class prototypes on held-out source and real PhysX router motions. The corresponding
causal selected-demo target is rebuilt with:

```bash
bash scripts/sugar/demo_reward/run_official_tmr_mismatch_dataset_then_hold.sh \
  experiments/demo_following/official_tmr_internal_reward_v1/motion_disjoint_predictor_dataset_suffix_v2 \
  cuda:0
```

Its machine-readable manifest fails; no predictor training follows. To reproduce the independent
reconstruction-latent check, place the official MotionGPT `t2m.pth` and exact official evaluator
`mean.npy/std.npy` under the paths checked by the launcher, then run:

```bash
bash scripts/sugar/demo_following/run_official_motiongpt_vqvae_instance_gate.sh \
  experiments/demo_following/official_motiongpt_vqvae_instance_gate_v2 cuda:0
```

The VQ-VAE reconstructs better than zero but does not reliably rank the specified source demo.
Both gates are representation evidence only; neither latent is exposed to the actor and neither
starts policy optimization.

### Official XIRL/TCC visual temporal gate

From retained Slurm GPU compute, first verify the clean Carry45/Kick21 references:

```bash
bash scripts/sugar/demo_following/prepare_official_xirl_runtime.sh

bash scripts/sugar/demo_following/run_xirl_reference_canaries_then_hold.sh \
  experiments/demo_following/official_xirl_tcc_v1/corpus_canary
```

Then run the resumable full pipeline:

```bash
bash scripts/sugar/demo_following/run_xirl_full_pipeline_then_hold.sh \
  experiments/demo_following/official_xirl_tcc_v1
```

It renders exact SUGAR root/joint/object trajectories through the official task scene and a clean
RTX world camera, with no overlays or policy output. The corpus contract is 100 CarryBox and 99
KickBox motions, 64 frames each; ID `%10==8` is validation, `%10==9` is test, all others train.
Training is the released Google Research XIRL ResNet18-linear/TCC path for 4000 iterations. The
runner tracks the one-line modern-PyTorch device patch, uses official Gym 0.17.3 and X-MAGICAL
0.0.2 dependencies, resumes incomplete checkpoints, skips training at checkpoint 4001 and skips
evaluation when its result JSON already exists.

The admitted run is negative: trained/raw test temporal MAE is `0.31675/0.29221`, Kendall tau is
`0.01282/0.10697`, and task-reference accuracy ties at `0.94737`. Its machine result is
`pretrain_runs/sugar_carry_kick_tcc_seed271402/temporal_retrieval_result.json` with `passed=false`.
This stops the pipeline before any reward predictor or policy training.
