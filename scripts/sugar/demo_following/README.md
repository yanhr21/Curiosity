# Demo following and released-skill transition

The active line preserves the exact released CarryBox45 and KickBox21 Generator+Tracker endpoints
and learns only a causal state/demo-conditioned transition between them. The earlier same-teacher
`phase_event_reward_only` experiment remains documented below as historical evidence; it is not the
current execution queue. Active experiments advance through machine-checkable outcomes without a
human-authorization state.

The latest representation-only entrypoint is:

```bash
bash scripts/sugar/demo_following/run_official_xskill_audit_then_hold.sh \
  experiments/demo_following/official_xskill_sugar_v1
```

It pins released XSkill commit `b748071`, trains the exact released skill-discovery architecture and
loss through epoch79 on two disjoint robot-only SUGAR source streams, runs the fixed
motion-disjoint temporal/task/order gate, renders four separate H.264 prototype videos and retains
the granted GPU. It never trains SAT or a policy. The result is negative: trained/initial test MAE
is `0.35887/0.33583`, tau is `0.02901/0.00698`, validation/test task accuracy is `0.45/0.68421`
and order accuracy is `0.95/0.73684`. The result and videos are under
`experiments/demo_following/official_xskill_sugar_v1/`. This is a same-embodiment compatibility
test; the next distinct audit adds the clean IsaacLab sphere embodiment required by XSkill's
released simulation contract.

The preceding frozen RoboCLIP entrypoint remains reproducible with
`run_official_roboclip_audit_then_hold.sh`; its official-dot task accuracy is `0.85/0.89474`, while
ordered-vs-reversed accuracy is only `0.35/0.31579`, so it also stops before policy training.

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

### Official Zero-WAM data preparation

Plan 17 keeps the model boundary strict: until the official Zero-WAM code and
weights are released, this repository prepares only the paired SUGAR adapter
data and never trains a local replacement world/action model.  From a retained
H200 Slurm step, run:

```bash
bash scripts/sugar/demo_following/run_zero_wam_sugar_data_pipeline.sh \
  /public/home/yanhongru/Curiosity/experiments/demo_following/zero_wam_official_v1
```

The first stage executes the released Carry/Kick Generator+Tracker pairs for all
100/99 source motions and records 700 causal control transitions per trajectory.
For every action it stores the exact pre-step 36-D Generator command, 510-D
Tracker observation, 29-D requested/executed action and pre/post physical state.
The action audit requires exact command-prefix equality, exact requested/executed
action equality, transition continuity, no reset, finite arrays and full
source-ID-disjoint coverage.

The renderer then uses those recorded states without rerunning policy or physics.
It emits 141 clean 320x320 robot frames at 10 Hz per trajectory; each adjacent
frame pair maps to exactly five of the 700 actions, including an exact terminal
post-state.  Prompt and robot streams are both re-rendered for Plan 17 with 30 m
between environment centers while the camera far clip is 20 m, so neighboring
motions cannot enter a selected-demo image.  Each trajectory is translated once
using only its first-frame robot/object midpoint; the camera never follows future
state.  The final immutable manifest joins these robot streams to the isolated
64-frame prompts and freezes split-matched wrong-task, reversed and same-task-
alternate prefixes.  Outcome/contact arrays are excluded from the deployed input
allowlist.  Passing this data gate gives 199 trajectories,
139,300 actions and 27,860 video-action intervals (22,400 train intervals) for
the bounded two-task SUGAR audit. The separate effective-diversity audit below
must also pass; neither result is Zero-WAM pre-training or an
open-ended/cross-embodiment result.

### Official Zero-WAM release and public Wan base gates

From a retained Slurm compute step, recheck the canonical repository with:

```bash
bash scripts/sugar/demo_following/run_zero_wam_official_release_audit.sh \
  experiments/demo_following/zero_wam_official_v1/official_release_status
```

The auditor records the full main commit and recursive tree, then requires real method code,
training/inference entrypoints, a data schema, world/action/IFP/MoT implementation, official
checkpoint files, strict load, checkpoint hashes, an official example and explicit official
provenance.  On 2026-08-31 canonical main is
`5a8a2da069392c1974ee98941ada13a5208b0ca5`; it has five blobs, zero Python files, zero
entrypoints/schema paths and correctly emits `release_available=false` with exit status 3.

The public Wan2.2 base is a separate preflight, not a Zero-WAM release.  The exact official source
commit is `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`; the staged Wan2.2-TI2V-5B snapshot contains
all 22 published files and `34,203,123,632` bytes.  In an isolated official-compatible runtime on
H200, run:

```bash
bash scripts/sugar/demo_following/run_wan22_base_h200_audit.sh \
  experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/Wan2.2-42bf4cfaa384bc21833865abc2f9e6c0e67233dc \
  experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/Wan2.2-TI2V-5B \
  experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/runtime_venv/bin/python \
  experiments/demo_following/zero_wam_official_v1/wan_base_h200_audit_v1
```

This gate strict-loads every published DiT shard, requires exact config/parameter identity and
BF16 H200 residency, executes all 30 official DiT blocks on a minimal valid latent, and hashes every
one of the 22 official snapshot files while rejecting any `.incomplete` fragment.  Passing it proves
only the public video backbone; it cannot open SUGAR
training without the released Zero-WAM action branch, MoT/IFP path, checkpoint and example.

The admitted public-base result has `4,999,787,712` parameters and zero
missing/unexpected/mismatched keys.  CPU strict load takes `13.6326 s`, BF16 H200 transfer takes
`5.4647 s`, and the exact 30-layer minimal-valid forward takes `0.5462 s` with
`10,263,348,736` peak allocated bytes.  The exact pinned runtime is Torch `2.7.0+cu128`, CUDA
`12.8`, Diffusers `0.33.0`, Transformers `4.51.3` and FlashAttention `2.8.3.post1`.

Recompute the explicit model/data training boundary with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/audit_zero_wam_sugar_data_diversity.py \
  --manifest experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl \
  --project-root /public/home/yanhongru/Curiosity \
  --output-dir experiments/demo_following/zero_wam_official_v1/training_data_diversity_v1

/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/audit_zero_wam_training_admission.py \
  --release-audit experiments/demo_following/zero_wam_official_v1/official_release_status/audit/OFFICIAL_RELEASE_AUDIT.json \
  --wan-base-audit experiments/demo_following/zero_wam_official_v1/wan_base_h200_audit_v1/WAN22_BASE_AUDIT.json \
  --sugar-manifest-result experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/RESULT.json \
  --sugar-data-diversity-result experiments/demo_following/zero_wam_official_v1/training_data_diversity_v1/SUGAR_TRAINING_DATA_DIVERSITY.json \
  --prompt-case-result experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_cases_v1/FROZEN_PROMPT_GATE_CASES_RESULT.json \
  --training-schedule-result experiments/demo_following/zero_wam_official_v1/bounded_posttraining_schedule_v1/BOUNDED_POSTTRAINING_SCHEDULE_RESULT.json \
  --training-schedule experiments/demo_following/zero_wam_official_v1/bounded_posttraining_schedule_v1/BOUNDED_POSTTRAINING_SCHEDULE.json \
  --output-dir experiments/demo_following/zero_wam_official_v1/training_admission_v1
```

The diversity result binds to the exact immutable manifest SHA256. It measures 11,200
non-overlapping five-action chunks per task; all 22,400 action chunks and 22,400 observation chunks
are unique, exact held-out overlap is zero, all 29 action dimensions vary, action covariance rank is
29 and each of ten phase bins contains 1,120 chunks per task. The current admission intentionally
has `training_allowed=false`: public-Wan, manifest and effective-diversity gates pass, while official
Zero-WAM release, frozen prompt dependence and official 29-DoF adapter gates remain false. It
separately emits
`sugar_foundation_pretraining.decision=forbidden_do_not_train_5b_from_scratch_on_sugar`; 22,400
intervals do not turn two tasks into foundation-scale diversity.

Freeze the official-model prompt-gate cases before any checkpoint score is observed:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/build_zero_wam_frozen_prompt_gate_cases.py \
  --source-manifest experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl \
  --output-dir experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_cases_v1
```

The passing result fixes 39 held-out source motions, ten causal phase anchors per motion and five
prompt interventions, producing 390 matched-noise groups / 1,950 score instances. The case manifest
SHA256 is `035e554a94ecd506e4e4d287f32cf90d21f78f8a5211277f34daedf4516e37f5`.
Once the strict-loaded official runner emits the required score JSONL, evaluate it with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/evaluate_zero_wam_frozen_prompt_gate.py \
  --case-manifest experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_cases_v1/FROZEN_PROMPT_GATE_CASES.jsonl \
  --score-jsonl OFFICIAL_ZERO_WAM_SUGAR_PROMPT_SCORES.jsonl \
  --expected-model-commit OFFICIAL_40_HEX_COMMIT \
  --expected-checkpoint-sha256 OFFICIAL_64_HEX_CHECKPOINT_SHA256 \
  --output-dir experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_v1
```

The evaluator first averages ten anchor margins within each source motion, then applies directional
exact sign tests and one Holm correction across task/order/identity/mask x validation/test. It also
requires separately positive Carry/Kick directions, exact matched-noise fingerprints and changed
official robot-future predictions before the action decoder. Its synthetic fixtures validate only
the decision contract and are never model evidence.

After that real prompt gate passes, audit evidence from the official 29-DoF adapter path with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/audit_zero_wam_official_29dof_adapter.py \
  --evidence-json OFFICIAL_ZERO_WAM_SUGAR_29DOF_ADAPTER_EVIDENCE.json \
  --prompt-gate-result experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_v1/FROZEN_PROMPT_GATE_RESULT.json \
  --expected-model-commit OFFICIAL_40_HEX_COMMIT \
  --expected-checkpoint-sha256 OFFICIAL_64_HEX_CHECKPOINT_SHA256 \
  --output-dir experiments/demo_following/zero_wam_official_v1/official_29dof_adapter_audit_v1
```

This audit contains no adapter or model. It requires exact official module/config/checkpoint hashes,
zero parameter change at zero updates, exactly the official trainable scope after two updates, a
bitwise-frozen VAE and finite nonzero video/action gradients. The fixed-data diagnostic uses IDs
`2/7/14/21/26/33/40/45/52/57/64/71/76/83/90/95` for each task (exactly 16 Carry/16 Kick),
selection SHA256 `949346b5e54710f49d7b8ea3683be6d96397a5a7ad22a4c459e9b4137eec645a`, at
fixed chunk anchors `21/49/77/105` for exactly 128 causal samples, and must reduce both official
video-flow and action-flow tail-median losses to at most half their initial values without worsening
IFP. Passing only opens full 160-motion / 22,400-chunk bounded post-training; it is not itself an
adapted checkpoint or closed-loop result.

Generate the model-schema-independent full-data exposure floor with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/build_zero_wam_bounded_posttraining_schedule.py \
  --source-manifest experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl \
  --output-dir experiments/demo_following/zero_wam_official_v1/bounded_posttraining_schedule_v1
```

Seed271500 freezes ten complete trajectory epochs. Each epoch covers all 160 train motions once,
interleaves Carry/Kick with prefix imbalance at most one and retains each trajectory's chronological
140 atomic intervals / 700 actions. This gives minimum totals of 224,000 interval exposures and
1,120,000 action exposures. The official loader may pack contiguous intervals to its released chunk
shape but may not omit or reorder them. Admission checks both the result and actual schedule file
against SHA256 `0f30252c0315855a1154d2c9f68d78b283cf35d2eee772a16692f5b0f1882ec1`.

After the real official run finishes, audit training sufficiency before any frozen evaluation:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/audit_zero_wam_atomic_training_consumption.py \
  --training-schedule experiments/demo_following/zero_wam_official_v1/bounded_posttraining_schedule_v1/BOUNDED_POSTTRAINING_SCHEDULE.json \
  --consumption-log OFFICIAL_ZERO_WAM_ATOMIC_CONSUMPTION.jsonl \
  --output-dir experiments/demo_following/zero_wam_official_v1/atomic_training_consumption_v1

/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/audit_zero_wam_bounded_posttraining_run.py \
  --training-admission OFFICIAL_PASSED_ZERO_WAM_TRAINING_ADMISSION.json \
  --training-schedule experiments/demo_following/zero_wam_official_v1/bounded_posttraining_schedule_v1/BOUNDED_POSTTRAINING_SCHEDULE.json \
  --run-evidence OFFICIAL_ZERO_WAM_SUGAR_POSTTRAINING_EVIDENCE.json \
  --output-dir experiments/demo_following/zero_wam_official_v1/bounded_posttraining_completion_v1
```

The official loader log has one record per consumed five-action atomic interval. It binds every
record to the immutable source row, prompt/robot sequence hashes and action trace; proves exact
frozen order and complete `10 x 160 x 140` coverage; verifies contiguous single-trajectory packing;
and records that the sample reached the official forward path with video/action/IFP targets. The
fingerprint covers exact preprocessed model inputs and must be distinct for all 22,400 intervals in
each epoch, rejecting collapsed or repeated loader outputs. The
completion evidence must include hash references to both this complete log and its passing audit.
Batch and optimizer-step indices must be monotonic and contiguous from zero. The joint optimizer
trace must contain exactly the same min/max/count step set, so no update can escape finite
video-flow, action-flow and IFP loss checks or strictly positive finite gradients on every branch.
The last complete epoch's median video/action losses must be below the first epoch and its IFP median
must not be worse; a single joint-gradient step or flat full-data trace fails.
Every counted row must also show positive effective learning rate, one applied update with no
AMP/scaler skip, consecutive update indices, positive video/action/IFP parameter-update norms and
zero non-finite trainable parameters after the update. A backward call that was skipped or produced
no branch parameter change is not credited toward the optimizer budget.

Every packed sample must stay inside one batch and optimizer step, and every batch maps to one step.
No optimizer step may aggregate more than 140 atomic intervals, one complete-trajectory equivalent.
Thus the ten-epoch 224,000-interval floor requires at least 1,600 optimizer updates by coverage.
Formal completion is stricter: it takes the maximum of that count, the paper's 4,000-step RoboTwin
post-training reference and any larger released official recipe. The full-scale contract fixture
contains 7,000 updates and explicitly rejects both a coverage-only 1,600-step trace and
one-update-per-epoch pseudo-training.

The atomic result also contains one composition row per optimizer step: unique epoch, Carry/Kick
atomic-interval counts, action exposures, forward batches and packed samples. The completion audit
requires the optimizer trace to match every composition field exactly; identical step IDs alone do
not pass. Cross-epoch accumulation and forged task composition are explicit negative tests.

Each optimizer row additionally reports official video/action/IFP losses for exactly the tasks with
nonzero consumed intervals; inactive tasks must be `null`. Carry and Kick are reduced separately by
complete epoch, and each must improve final video/action medians with non-worse IFP. Aggregate loss
improvement cannot hide a stalled task.

The evidence manifest also points to hash-verified per-trajectory epoch JSONL, optimizer JSONL,
official module before/after hashes and the complete final checkpoint file or sharded directory. The gate requires H200 Slurm execution, the
exact official entrypoint/config and checkpoint identity, all ten scheduled epochs, at least
224,000 interval and 1,120,000 action exposures, finite video/action/IFP losses, strictly positive
three-branch gradients at every optimizer step, at least 4,000 updates (or a larger released official
budget), actual non-skipped finite parameter changes at every counted step and first-to-last epoch
median loss progress,
changed official video/action trainable scopes, and a bitwise-frozen official video VAE. It rejects
undertraining, schedule drift, action-only training, collapsed inputs, frozen-scope drift, held-out leakage, local
learned modules and public-Wan-only substitutes. Passing opens frozen evaluation; it is not model
success. `--self-test` exercises only synthetic evidence-contract fixtures.

After training completion passes, audit the fixed same-checkpoint SMALLBOX rollouts with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/evaluate_zero_wam_smallbox_closed_loop.py \
  --training-completion experiments/demo_following/zero_wam_official_v1/bounded_posttraining_completion_v1/BOUNDED_POSTTRAINING_COMPLETION_AUDIT.json \
  --source-manifest experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl \
  --project-root /public/home/yanhongru/Curiosity \
  --evidence-json OFFICIAL_ZERO_WAM_SMALLBOX_CLOSED_LOOP_EVIDENCE.json \
  --output-dir experiments/demo_following/zero_wam_official_v1/smallbox_closed_loop_v1
```

Seed281500 fixes 20 profiles, Carry45/Kick21 prompt-only swaps, exact released endpoint baselines and
650 frames. The audit hashes 80 traces, restores identical physics/history, verifies a single adapted
checkpoint and one cached encoding per prompt, and recomputes Carry/Kick/fall outcomes from root,
object and filtered hand/foot contact traces. It requires at least 16/20 safe matched outcomes per
prompt, topology-specific condition advantages and no fall regression. Every adapted frame must
prove official predicted future -> action-decoder input -> finite executed 29-D action, with no
teacher future, future target, outcome label, router or demo reward. Passing is only a two-prompt
SMALLBOX result; motion-disjoint and cross-asset gates remain separate.

Freeze the motion-disjoint test rollouts independently of model outcomes with:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/build_zero_wam_motion_disjoint_closed_loop_cases.py \
  --source-manifest experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/ICL_MANIFEST.jsonl \
  --output-dir experiments/demo_following/zero_wam_official_v1/motion_disjoint_closed_loop_cases_v2 \
  --self-test
```

The immutable result contains every test motion (10 Carry, 9 Kick), ten paired physics profiles and
matched/reversed/same-task-alternate/wrong-task prompts: 190 four-condition groups and 760 adapted
rollouts. Matched/wrong-task cases add 380 exact endpoint baselines from identical initial states,
for 1,140 traces / 741,000 fixed closed-loop frames. All prompt sources are test-only and share exact per-group physics
and scoring-noise seeds. Its case-manifest SHA256 is
`4be15b98fc8b0b71792dee053e657e39bdeaf0f8dd68840514c5b2d08f1d05e5`. Matched and wrong-task
prompts must produce `8/10` safe prompted-task outcomes per source without endpoint fall regression;
matched-vs-reversed order and matched-vs-same-task identity official-flow margins are evaluated
separately with source-motion statistics and one Holm family. All three gates must pass together.

After SMALLBOX passes and the complete grid is collected, run:

```bash
/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python \
  scripts/sugar/demo_following/evaluate_zero_wam_motion_disjoint_closed_loop.py \
  --smallbox-audit experiments/demo_following/zero_wam_official_v1/smallbox_closed_loop_v1/SMALLBOX_CLOSED_LOOP_AUDIT.json \
  --training-completion experiments/demo_following/zero_wam_official_v1/bounded_posttraining_completion_v1/BOUNDED_POSTTRAINING_COMPLETION_AUDIT.json \
  --case-result experiments/demo_following/zero_wam_official_v1/motion_disjoint_closed_loop_cases_v2/MOTION_DISJOINT_CLOSED_LOOP_CASES_RESULT.json \
  --case-manifest experiments/demo_following/zero_wam_official_v1/motion_disjoint_closed_loop_cases_v2/MOTION_DISJOINT_CLOSED_LOOP_CASES.jsonl \
  --project-root /public/home/yanhongru/Curiosity \
  --evidence-json OFFICIAL_ZERO_WAM_MOTION_DISJOINT_EVIDENCE.json \
  --flow-scores OFFICIAL_ZERO_WAM_MOTION_DISJOINT_FLOW_SCORES.jsonl \
  --output-dir experiments/demo_following/zero_wam_official_v1/motion_disjoint_closed_loop_v1
```

This evaluator re-audits every full trace, exact endpoint file and prompt cache rather than accepting
self-reported outcomes. Every one of the 190 matched causal traces is scored at fixed full-horizon
steps `49/99/.../649`, producing 2,470 matched-noise rows and 7,410 official flow comparisons under
matched/reversed/alternate prompts. Margins reduce over 13 anchors and then ten profiles before the
source-motion sign tests and Holm correction. Physical safety, order and identity must all pass.

Keep the release and admission decisions live inside the retained H200 allocation with:

```bash
bash scripts/sugar/demo_following/run_zero_wam_release_training_monitor.sh \
  experiments/demo_following/zero_wam_official_v1/release_training_monitor_v1 600 30
```

Each poll archives the canonical commit/tree, recomputes training admission and writes one compact
`MONITOR_STATE.json`.  It exits automatically if the official repository changes to a real release
so the next autonomous stage can inspect and strict-load the unknown official schema; it has no
approval flag, sentinel or manual checkpoint gate.
