# Plan 17: Official Zero-WAM for Executable Selected-Video Following

**Status (2026-08-31): official-method audit complete; official code/model/data not released.**
The paper and project repository are public, but the repository at main commit `5a8a2da` contains
only the README, license and project assets.  The authors state that code, model and data are
expected before 2026-09-15.  Until those artifacts exist, no local implementation may be called a
Zero-WAM reproduction and no homemade video/action world model may substitute for them.

This plan is the active demo-following direction after the Newton receding-knot Refiner gate failed
at `0.01005 m` lift and `0/1` strict completion.  It changes the causal controller topology rather
than extending a Newton residual, scalar reward or frozen-embedding router.

Official sources:

- paper v2: <https://arxiv.org/abs/2608.26103>;
- project/release repository: <https://github.com/robbyant-research/Zero-WAM>;
- project page: <https://robbyant-research.github.io/Zero-WAM/>.

## 1. Why this is the right idea

The current SUGAR evidence separates three increasingly strong statements:

1. frozen video/motion representations can recognize Carry versus Kick, but have not reliably
   encoded selected-demo order or identity;
2. selected-demo rewards and causal conditions can change PPO gradients and behavior, but the
   closed-loop policy usually remains the same Carry solution;
3. routing complete released Generator+Tracker pairs can switch Carry/Kick in the compatible
   SMALLBOX scene, but that is discrete endpoint selection rather than arbitrary-video following.

Zero-WAM attacks the missing link directly.  It does not reduce the demonstration to a scalar
reward or ask an action MLP to extrapolate across endpoint policies.  Its deployed causal chain is:

```text
in-context demonstration video
        + causal robot video/action history
        -> predicted next robot-domain video chunk
        -> inverse-dynamics action chunk
        -> closed-loop execution
```

The central hypothesis for SUGAR is therefore:

> A shared, serious video-action policy can use the selected SUGAR demonstration as prefix memory
> to predict a task- and sequence-consistent future in the robot domain, then decode an executable
> 29-DoF action chunk from that predicted future.  Explicit future-chunk supervision should prevent
> the main world branch from ignoring the demo in favor of the recent robot-state shortcut.

This is a better match to the requested endpoint than another demo-score reward: task intent must
pass through an executable predicted future before it reaches action.

## 2. What the official method actually is

### 2.1 Data

Zero-WAM uses two task-balanced families rather than raw trajectory-frequency sampling:

- Task-diverse VA: more than 6,000 manipulation tasks and roughly 400K robot trajectories sampled
  per pre-training epoch from five public robot datasets;
- HumanGen: 74.2K semantically paired human-video/robot-trajectory examples over 8.6K tasks and
  more than 45 robot embodiments.

HumanGen starts with a robot trajectory that retains executable actions.  A VLM extracts the task,
initial state, object-state changes and final state; an image editor converts the first robot frame
to a human scene; a video model generates the human manipulation; a VLM filters semantic and
physical failures.  The resulting human video is paired back to the original executable robot
trajectory.  The pairing is semantic, not pose-by-pose or embodiment-identical.

### 2.2 Model and causal factorization

The released paper instantiates the policy from `Wan-2.2-TI2V-5B`.  The video branch has hidden
dimension 3072 and 30 Transformer layers.  A separate 3072-D action Transformer is initialized from
the video branch.  The Mixture-of-Transformers design gives the two modalities separate QKV, FFN
and output parameters while allowing interaction through shared attention.

For chunk index `i`, the model factorizes

```text
p(next video, next action | history, condition)
  = p_video(next video | robot video/action history, demo or language)
  * p_action(next action | robot history, predicted next robot video, language).
```

The human/demo video is prepended as prefix memory to the video branch.  The action branch does not
directly attend to the demo: task information must first change the predicted robot future, and the
action branch acts as inverse dynamics from that future.  During training the action branch sees
the ground-truth next robot video under teacher forcing; at inference it sees the generated future.
Human and robot videos share the official Wan VAE, with a height-axis RoPE offset (`32` in the
paper) to distinguish the prompt tokens from robot-observation tokens.

### 2.3 In-context future prediction (IFP)

Immediate next-chunk prediction can ignore the demonstration and extrapolate recent robot history.
IFP adds four training-only future-video heads with stride two and fixed loss weights
`0.5/0.25/0.15/0.15`.  Each head has one video-Transformer-layer architecture initialized from the
last video layer.  The heads receive a fused multi-layer representation from the main robot-video
branch, but do not receive the demo directly.  Therefore future supervision must force the deployed
main representation to carry demo information.  The IFP heads are removed at inference.

### 2.4 Scale and evidence

The paper reports a `1:5` Task-diverse-VA/HumanGen pre-training sampling ratio, random video chunks
of one to four, 15,360 GPU-hours of pre-training, and a 64-GPU/4,000-step RoboTwin post-training run.
Inference uses two-frame chunks and caches the video prompt once.  On seven task-held-out RoboTwin
tasks, full Zero-WAM reports `46.95%` mean success versus `17.45%` for LingBot-VA.  Human-video ICL
without the large pre-train raises the Wan baseline from `10.98%` to `36.36%`; adding IFP raises the
reported average from `28.55%` to `46.95%`.  The hardest three-block stack remains only `9%`.

The claim is meaningful but bounded: this is mostly stationary tabletop manipulation, the average
success is not close to solved, and the main result is zero-shot task/configuration transfer—not
exact trajectory imitation.

## 3. SUGAR scope and claim boundary

The historical XIRL prompt corpus remains valid evidence for its closed negative audit, but visual
inspection during Plan 17 found that its 2.5 m tiled-scene spacing lets neighboring environments
enter the camera frustum.  It must not be reused as a selected-demo training stream.  Plan 17
therefore re-renders the same exact source motions with 30 m environment spacing, a 20 m camera far
clip and one fixed first-frame robot/object XY centering transform per trajectory.  No future or
per-frame camera tracking is used.  The isolated corpus keeps the same bounded scale:

- 100 CarryBox and 99 KickBox source motions;
- exactly 64 clean `320 x 320` RGB frames per motion;
- no text, plots, borders, metrics or policy output in the frame;
- fixed source-ID split: IDs ending in 8 are validation, IDs ending in 9 are test, all others train.

The paired executable target must come from the released SUGAR inference chain and include its
actual action annotation:

- Carry: released Carry Generator + Carry Tracker;
- Kick: released Kick Generator + Kick Tracker;
- the current 36-D Generator command and 510-D Tracker observation remain part of the causal action
  record; no Tracker-only counterfactual is a valid target.

Before model work, an adapter audit must prove which of the 199 source motions has a complete,
finite, temporally aligned `(prompt RGB, robot RGB, Generator command, Tracker observation,
29-D action)` record.  Missing pairs are reported, never synthesized with a local expert.

That data audit is complete as of 2026-08-31. All 199 motions have one complete isolated prompt and
robot stream plus 700 exact released Generator+Tracker transitions, for 139,300 transitions total.
The immutable v2 manifest contains 27,860 synchronized video/action intervals, including 22,400
train intervals. It passes every predeclared split, finiteness, continuity, executed-action,
command/observation equality, camera-isolation and pixel-nonreuse check. Across 12,736 normalized
prompt/target frame comparisons and 199 complete streams there are zero identical pairs. The data
are sufficient for the fixed two-task SUGAR audit once the official model exists; they do not make
the corpus equivalent to HumanGen or authorize a local substitute.

SUGAR currently supplies only two task families.  Consequently:

- a Carry/Kick result is a two-task, same-embodiment selected-video audit;
- motion-ID-disjoint evaluation can establish transfer to held-out motions inside those tasks;
- it cannot establish the paper's 8.6K-task, cross-embodiment or open-ended zero-shot claim;
- a generated human-video prompt can be used only if the released official HumanGen pipeline and
  its filters can be run faithfully.  The existing sphere-agent videos remain an explicitly
  separate diagnostic and are not HumanGen.

## 4. Fixed experimental stages

All stages run automatically from machine-readable gates.  An external artifact being absent is a
compatibility status, not a request for human authorization.

### Stage A — official release and strict-load audit

1. Resolve the first official commit that contains training/inference code, model weights and data
   schema; record the full commit and released checkpoint identity.
2. Confirm that the files are actual method code and weights, not another project-page-only commit.
3. Install only compatibility glue needed by the official environment; do not change architecture,
   attention masks, factorization, IFP targets or sampling.
4. Strict-load the released checkpoint and reproduce one official inference example plus the
   documented tensor shapes, parameter counts and prompt-cache behavior.
5. Audit H200 memory and wall time for official two-frame inference.  An OOM is recorded as a
   compute incompatibility; it is not repaired by shrinking the model.

Stage A passes only when `OFFICIAL_RELEASE_AUDIT.json` proves strict loading, exact official model
class/config, a complete official example and no local model substitution.

The live release gate is implemented and fail-closed.  On 2026-08-31 it resolves canonical main to
full commit `5a8a2da069392c1974ee98941ada13a5208b0ca5`: the recursive tree contains only five blobs,
zero Python files, zero training/inference entrypoints and zero data-schema paths, so
`release_available=false`.  Separately, the exact public Wan2.2 source at commit
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc` and all 22 published Wan2.2-TI2V-5B files
(`34,203,123,632` bytes) are staged for an H200 base-runtime audit.  A passing Wan base audit is
preflight evidence only: it cannot satisfy Stage A or open SUGAR training because it contains no
released Zero-WAM action branch, MoT/IFP implementation, checkpoint or official example.

That public-base preflight now passes on H200.  The exact three-shard DiT strict-loads with zero
missing, unexpected or mismatched keys and has `4,999,787,712` parameters.  BF16 residency uses
`10,001,017,344` allocated bytes; the full minimal-valid 30-layer forward peaks at
`10,263,348,736` bytes and finishes in `0.5462 s`.  CPU load and H200 transfer take `13.6326 s` and
`5.4647 s`.  The runtime is pinned to Torch `2.7.0+cu128`, Diffusers `0.33.0`, Transformers
`4.51.3` and official FlashAttention `2.8.3.post1`; the final snapshot gate additionally requires
the exact 22 paths/bytes, no incomplete fragments and SHA256 for every file.

The model/data training-admission gate is also executable and distinguishes two claims that must
never be merged.  The current immutable SUGAR corpus passes every fixed requirement for the bounded
two-task post-training audit (`160/20/19` train/validation/test motions and `22,400` train
video-action intervals). A second audit binds this conclusion to the exact immutable manifest and
checks effective rather than nominal sample count: Carry/Kick contribute `11,200/11,200`
non-overlapping five-action chunks, all `22,400` action chunks and all `22,400` 510-D observation
chunks are byte-distinct, exact held-out overlap is zero, every one of the 29 action dimensions
varies and action covariance has numerical rank 29. Each task also has exactly 1,120 chunks in each
of ten causal phase bins, while one trajectory contributes at most `0.625%` of training. Foundation
pre-training is nevertheless forbidden: two tasks and 199 trajectories
are orders of magnitude below the reported `>6,000` robot tasks / about 400K trajectories per epoch
plus 74.2K HumanGen pairs over 8.6K tasks.  Formal SUGAR training remains false until one official
Zero-WAM commit/checkpoint/example, the frozen task/order/identity prompt gate and a documented
official 29-DoF adapter all pass.  Public Wan compatibility alone can never satisfy those checks.

### Stage B — immutable SUGAR ICL manifest

Build a manifest without rendering presentation/composite videos.  Every row records task, source ID,
split, prompt frames, target robot frames, Generator commands, 510-D Tracker observations, 29-D
actions, frame timestamps and the exact released expert identities.  Enforce:

- source IDs are disjoint across train/validation/test;
- prompt and target streams are synchronized by physical time, not free-window nearest matching;
- all prompt/target environments are beyond the camera far clip of every neighboring environment,
  and a strict non-constant-frame gate passes after fixed first-frame causal centering;
- all arrays are finite and every action is the action actually executed by the paired expert;
- no future target, outcome label, task-success label or selected-demo ID enters deployed history;
- counterfactual prompts change only prefix memory; robot history, target, noise and language state
  are held fixed;
- language is disabled for the primary ICL audit, matching the paper's ICL inference setting.

The manifest gate fails closed if executable action pairing is incomplete or if prompt and target
derive from the same rendered frames rather than separately generated prompt/robot streams.

### Stage C — frozen official prompt-dependence gate

Before post-training an action head, test the released checkpoint with teacher-forced SUGAR robot
history and matched diffusion noise. The frozen executable case manifest uses all 39 held-out source
motions, ten fixed causal phase anchors per motion (`7/21/.../133`) and five conditions per anchor,
for 390 matched-noise groups and 1,950 official-model score calls. For every held-out target, score:

1. matching selected prompt;
2. wrong-task prompt;
3. identical matching frames in reverse temporal order;
4. a different same-task source motion.
5. the matching prompt with cached prompt latents masked after official preprocessing.

Use the official next-video flow loss and, if the released API exposes it, the official IFP loss.
No locally invented embedding distance is an admission metric.  Validation fixes all normalization
and evaluation settings; test is read once.  Each of task, temporal order and selected-motion
identity and prompt presence pass only when both validation and test have positive mean paired
margin and paired win rate above 0.5, separately positive Carry and Kick means/win rates, and a
Holm-corrected predeclared directional exact sign `p < 0.05`. The correction family contains all
eight comparisons (four interventions x two splits). Ten anchor losses are averaged inside each
source motion before inference, so frames are never treated as independent samples. Identical robot
histories with swapped prompts must change the predicted future before the action decoder at every
anchor.

The case builder is complete and binds to immutable ICL manifest SHA256
`24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8`; its frozen case
manifest SHA256 is `035e554a94ecd506e4e4d287f32cf90d21f78f8a5211277f34daedf4516e37f5`.
The model-independent evaluator requires one complete 1,950-row score matrix, exact official commit
and checkpoint hashes, official next-video flow loss, identical noise tensor fingerprints across
conditions and pre-action-decoder predicted-future hashes. Synthetic contract tests pass a fully
positive fixture and reject a wrong-task-better fixture. These fixtures test only the gate and are
not model evidence.

Failure at task, order or identity closes SUGAR policy adaptation from that checkpoint.  It may be
reported as task semantics, ordering or instance information respectively, but may not be promoted
to selected-demo following or repaired by threshold/loss/model-size sweeps.

### Stage D — official G1 action-interface adaptation

This stage opens only after Stage C passes all three levels and the released code exposes a
documented embodiment/action adaptation path.  Reuse the official Wan VAE, video Transformer,
action Transformer, MoT attention, flow-matching losses and IFP modules exactly.  Adapt only the
officially supported action/state interfaces to the paired 29-DoF SUGAR record.  Do not add the
frozen 11.386M scorer, TinyMDM energy, PPO demo reward, endpoint router or a local latent model.

First run a zero-update audit proving prompt-only swaps, causal masks, target-only future labels and
zero parameter change.  Then run a fixed two-update optimizer smoke proving finite nonzero changes
only in the modules selected by the official configuration.  A single predeclared fixed-data
overfit follows; its target is both official video-flow prediction and executed 29-D action chunks,
not action MSE alone.  The stopping point and optimizer are copied from the released SUGAR/embodiment
recipe if one exists.  If none exists, this stage is blocked as an unsupported embodiment rather
than replaced with a hand-written head.

The adapter evidence auditor is frozen before release and contains no model implementation. It
requires an official repository-relative adapter/config document and exact commit/checkpoint hashes;
29-D continuous executed-action chunks; the released VAE, video Transformer, action Transformer,
MoT and IFP; no architecture diff or local learned module; and future/outcome labels excluded from
deployed inputs. Zero-update hashes must be identical across every official module scope. At two
updates the changed scope must equal the official trainable-module set exactly, the VAE must remain
bitwise unchanged, and both video/action branch gradient norms must be finite and positive.

The fixed-data diagnostic uses exactly 32 distinct train motions (16 Carry, 16 Kick): source IDs
`2/7/14/21/26/33/40/45/52/57/64/71/76/83/90/95` for each task, selected evenly from the sorted
80-motion train set before release. The canonical selection SHA256 is
`949346b5e54710f49d7b8ea3683be6d96397a5a7ad22a4c459e9b4137eec645a`. Each motion uses fixed
chunk anchors `21/49/77/105`, for exactly 128 causal samples. Its optimizer, schedule, stopping
point and module selection come from a config
frozen before the run. The tail-median official video-flow and action-flow losses must each be at
most `0.5x` their initial value, while IFP may not worsen. This diagnostic tests interface
learnability only; a pass does not reduce the formal post-training corpus below all 160 train motions
and 22,400 non-overlapping chunks. Synthetic evidence tests pass the complete contract and reject
action-only improvement, frozen-VAE drift and an undocumented adapter; they are not model results.

Formal bounded post-training has a separate full-data exposure floor. Seed `271500` creates ten
distinct deterministic epochs; every epoch contains all 160 train trajectories exactly once, with
Carry/Kick strictly interleaved so every prefix differs by at most one trajectory. Each trajectory
retains all 140 chronological atomic video/action intervals and 700 actions. The official loader may
pack contiguous intervals to its released action-chunk schema, but may not drop, cross-trajectory or
reorder them. The minimum therefore exposes 224,000 atomic intervals and 1,120,000 actions. If the
official recipe requires more, extend only with complete epochs under the same deterministic rule;
validation/test may never choose an early stop below ten. The exact schedule SHA256 is
`0f30252c0315855a1154d2c9f68d78b283cf35d2eee772a16692f5b0f1882ec1`. This is a training
coverage floor, not evidence that ten epochs guarantee physical success.

Formal training completion is a separate fail-closed evidence gate. The official run must bind to
the exact passed admission file, official commit/checkpoint and frozen schedule; execute on an H200
Slurm compute node; record every one of the 160 trajectory exposures in every complete epoch; and
retain hash-verified optimizer, module-scope and final-checkpoint artifacts. At least ten complete
epochs, 224,000 atomic intervals and 1,120,000 actions must be observed. Official video-flow,
action-flow and IFP losses must remain finite, and all three gradient norms must be finite and
strictly positive at every recorded optimizer step. Per-epoch medians must show lower final
video/action flow loss than the first complete epoch and non-worse final IFP; one positive joint
step or a positive run-wide gradient sum is not sufficient. The official
video-world-model and action-decoder scopes must change, and the released video VAE and every other
official frozen scope must remain bitwise exact. Validation/test exposure, validation-selected
early stopping, action-only training, a local learned module or a public-Wan-only substitute rejects
the run. Passing proves adequate official post-training execution only and automatically opens the
frozen Stage E evaluation; it is not a selected-demo or physical-success claim.

Trajectory-level counters are not sufficient evidence of data use. The official loader must also
emit one ordered record for every consumed atomic interval. The fixed ten-epoch floor therefore
contains exactly 224,000 records, each bound to its immutable source-manifest row, prompt/robot
sequence hashes, action trace and exact five-action range. Packed samples may contain only a
contiguous interval partition from one trajectory, and every record must reach the official forward
path with video-flow, action-flow and IFP targets present. The forward fingerprint includes the
exact preprocessed model inputs and must remain distinct for all 22,400 intervals within each epoch,
so a collapsed/repeated loader cannot satisfy counters alone. Global consumption order, batch and
optimizer indices are monotonic; every epoch matches the frozen trajectory order and contains each
of 160 x 140 intervals exactly once. Future/outcome inputs, held-out rows, duplicated/dropped
intervals, collapsed inputs, trajectory reordering and cross-trajectory packing reject the run. The formal completion
gate must hash and require this passing atomic-consumption audit and its complete log.

The consumption log's batch and optimizer-step indices must be monotonic and contiguous from zero.
It reports the exact optimizer-step range and count; the formal optimizer trace must contain exactly
that full step set with no missing or additional record. Every recorded step must carry finite
official video-flow, action-flow and IFP losses and strictly positive finite gradients for all three
branches. First/last complete-epoch medians must show video/action improvement and non-worse IFP.
This closes the gap between “data reached some forward calls” and “every optimizer update in the
formal run was jointly active and the full-data objective made measurable progress.”

Packed-sample membership is also audited: one packed sample cannot straddle batches or optimizer
steps, and one batch cannot map to multiple steps. A single optimizer step may consume at most one
trajectory equivalent (`140` atomic intervals). The fixed 224,000-interval floor therefore requires
at least 1,600 contiguous optimizer updates; the current full-scale contract fixture contains 7,000.
This is a coverage floor, not a locally invented optimizer or a substitute for the released recipe.

### Stage E — same-checkpoint closed-loop frozen evaluation

The first physical gate uses the compatible SMALLBOX scene where exact released endpoints already
provide stable Carry and Kick baselines.  Load one adapted checkpoint twice from elementwise
identical robot, object, joint and causal-history tensors; change only the cached prompt video.

The fixed selected prompts are Carry45 and Kick21.  Over 20 matched profiles per condition:

- Carry prompt: at least `16/20` strict Carry outcomes with at least 5 cm lift;
- Kick prompt: at least `16/20` strict-v4 Kick outcomes;
- neither condition may exceed its corresponding exact released endpoint's fall count;
- prompt swap must change predicted future chunks before changing action chunks;
- every deployed action must decode from the model's predicted robot future, never a teacher-forced
  future or future target;
- all traces must be finite and the prompt is encoded once then cached.

Passing proves one same-checkpoint two-task video-conditioned policy in a compatible scene.  It does
not yet prove arbitrary selected-motion or cross-asset following.

Only that pass opens motion-disjoint task-wide evaluation.  Use all fixed test IDs and preserve each
selected prompt/target pairing.  Report task success, falls, selected-motion flow-loss ranking,
ordered-versus-reversed preference and same-task alternate-prompt preference separately.  A final
claim requires the physical task outcome and selected-motion/order gates to pass together; task
classification or generated-video quality alone is insufficient.

The motion-disjoint closed-loop set is also frozen before any adapted result. All 19 immutable test
motions (Carry `9/19/.../99`, Kick `9/19/.../89`) expand over ten deterministic paired physics
profiles and four prompt conditions: matched, identical reversed, fixed same-task alternate and
fixed wrong-task. This yields 190 paired groups and 760 adapted rollouts. Matched and wrong-task
cases additionally require 380 exact released-endpoint rollouts from identical initial physics, so
the full no-regression evidence is 1,140 rollouts / 741,000 required 650-frame closed-loop steps. Every condition in a group shares physics seed and matched scoring-noise seed; every
prompt source remains test-only, and no evaluation target enters the deployed model. The exact case
manifest SHA256 is `4be15b98fc8b0b71792dee053e657e39bdeaf0f8dd68840514c5b2d08f1d05e5`.

For every source motion, matched and wrong-task prompts must each produce at least `8/10` safe
outcomes for the prompted task with no fall regression against its exact released endpoint. Order
and selected-motion identity remain distinct gates: using matched official diffusion noise, matched
must beat reversed and same-task-alternate official flow loss at source-motion level, with positive
per-task means, win rate above `0.5`, and one-sided exact sign tests passing one Holm family. Physical
task switching, temporal order and identity must pass together; any single failure closes the broad
test-motion claim without a guidance/reward/update sweep.

The result evaluator reopens and hashes all 1,140 traces rather than trusting rollout summaries.
Every unique prompt fingerprint is encoded exactly once and bound to every adapted trace that reuses
its cache. Exact endpoint checkpoint files are hashed for all baseline routes. Every one of the 190
matched rollout histories is rescored at fixed full-horizon causal steps `49/99/.../649` under
matched/reversed/same-task prompts with identical diffusion noise. This requires 2,470 matched-noise
rows and 7,410 official flow comparisons; every row binds to the matched causal-trace hash, fixed
score-noise seed and same formal checkpoint. Margins reduce over the 13 anchors before the ten
profiles, then over source motions for inference. The aggregate passes only when all 19 per-source
physical gates and the joint Holm-corrected order/identity family pass together.

The SMALLBOX decision is frozen as an executable trace audit before the model exists. Evaluation
seed `281500` runs exactly 20 matched profiles x Carry45/Kick21 x adapted/released-endpoint routes x
650 frames. All four routes for a profile must restore identical full-state/history hashes and
identical observed initial physics; the adapted arms load the same formally completed checkpoint and
change only the cached prompt. Each prompt is encoded exactly once and reused for all 13,000 frames.
The evaluator hashes every one of 80 unique traces, independently recomputes safe Carry, strict-v4
safe Kick and root-height/root-tilt falls, and hashes the actual released Generator/Tracker files
used for the two no-regression baselines. Carry45 and Kick21 each require at least 16/20 matched safe
successes, each must outperform the other prompt on its own physical topology, and neither fall
count may exceed its exact endpoint baseline.

Every adapted frame must use the official predicted robot future as the action decoder input, emit
a finite 29-D decoded action equal to the executed action within `1e-6`, and record zero
teacher-forced future, future target, outcome label, endpoint router and demo reward use. For all 20
profiles, prompt swapping must change the predicted future before or at the first changed action.
Any checkpoint mismatch, prompt re-encoding, initial-state mismatch, action-without-future change,
decoded/executed mismatch, endpoint identity drift, fall regression or `15/20` task result rejects
the checkpoint without a threshold/reward/update sweep. Passing remains a bounded two-prompt
SMALLBOX result, not arbitrary-demo or cross-asset following.

### Stage F — cross-embodiment HumanGen follow-up

This stage is outside the first SUGAR claim and opens only if the official HumanGen generator,
filter and released data schema are available and Stages A–E pass.  Generate human prompts from
the same action-grounded SUGAR robot trajectories with the exact official pipeline.  Preserve the
robot target/actions, vary only the human visual domain, rerun semantic and physical gates, and call
the result cross-embodiment only if the human and G1 streams are genuinely distinct.

## 5. Automatic decision rules

- Official code/weights absent: finish manifest/data audit, record `official_release_available=false`,
  and do not train a substitute.
- Current SUGAR data may be admitted only for bounded post-training from an official pretrained
  Zero-WAM checkpoint; it is permanently rejected as 5B foundation-pretraining data.
- Official strict load or official example fails: stop Zero-WAM execution as a compatibility issue;
  do not patch model semantics.
- Frozen prompt gate fails: do not post-train a SUGAR action policy from the failed signal.
- Frozen prompt gate passes but no official G1/action adaptation path exists: record an embodiment
  blocker; do not hand-roll an action Transformer.
- Small fixed overfit fails jointly on future-video and action targets: close the adapter before a
  formal training budget.
- SMALLBOX physical gate fails: do not run task-wide, cross-asset or HumanGen stages and do not tune
  prompt guidance, IFP weights or action chunk size.
- SMALLBOX passes: automatically run the predeclared held-out motion evaluation, then the
  cross-embodiment stage only if its official prerequisites also exist.

## 6. Explicit non-goals

- no toy VAE, small Transformer, local world model or simplified Zero-WAM;
- no rebranding the existing 11.386M event predictor as a world model;
- no use of XIRL, RoboCLIP, TMR, MotionGPT or TinyMDM scores as surrogate Zero-WAM labels;
- no composite presentation video as training input;
- no free-window temporal matching, future actor input or evaluation outcome leakage;
- no claim of 8.6K-task/open-ended/cross-embodiment generalization from Carry/Kick;
- no reward-scale, guidance-scale, IFP-weight, chunk-length or model-size sweep after a fixed gate
  fails;
- no human authorization state between already declared stages.

## 7. Deliverables

- exact official release/strict-load audit;
- immutable SUGAR video-action ICL manifest and split audit;
- frozen matched/wrong/reversed/same-task prompt report;
- official G1 interface and zero/two-update optimizer audits when supported;
- same-checkpoint SMALLBOX physical result and, only after it passes, held-out motion result;
- synchronized prompt/predicted-future/actual-world/action evidence for admitted rollouts;
- source/docs commit only.  Checkpoints, datasets, traces and videos stay ignored under
  `experiments/`.
