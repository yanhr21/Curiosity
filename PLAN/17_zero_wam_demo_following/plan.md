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

The existing immutable prompt corpus is suitable for an audit but not for reproducing the paper's
scale:

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

### Stage B — immutable SUGAR ICL manifest

Build a manifest without rendering new presentation videos.  Every row records task, source ID,
split, prompt frames, target robot frames, Generator commands, 510-D Tracker observations, 29-D
actions, frame timestamps and the exact released expert identities.  Enforce:

- source IDs are disjoint across train/validation/test;
- prompt and target streams are synchronized by physical time, not free-window nearest matching;
- all arrays are finite and every action is the action actually executed by the paired expert;
- no future target, outcome label, task-success label or selected-demo ID enters deployed history;
- counterfactual prompts change only prefix memory; robot history, target, noise and language state
  are held fixed;
- language is disabled for the primary ICL audit, matching the paper's ICL inference setting.

The manifest gate fails closed if executable action pairing is incomplete or if prompt and target
derive from the same rendered frames rather than separately generated prompt/robot streams.

### Stage C — frozen official prompt-dependence gate

Before post-training an action head, test the released checkpoint with teacher-forced SUGAR robot
history and matched diffusion noise.  For every held-out target, score four prefix conditions:

1. matching selected prompt;
2. wrong-task prompt;
3. identical matching frames in reverse temporal order;
4. a different same-task source motion.

Use the official next-video flow loss and, if the released API exposes it, the official IFP loss.
No locally invented embedding distance is an admission metric.  Validation fixes all normalization
and evaluation settings; test is read once.  Each of task, temporal order and selected-motion
identity passes only when both validation and test have positive mean paired margin, paired win rate
above 0.5 and Holm-corrected exact paired-sign `p < 0.05`.  Prompt masking must also be worse than
the matching prompt.  Identical robot histories with swapped prompts must change the predicted
future before the action decoder.

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

### Stage F — cross-embodiment HumanGen follow-up

This stage is outside the first SUGAR claim and opens only if the official HumanGen generator,
filter and released data schema are available and Stages A–E pass.  Generate human prompts from
the same action-grounded SUGAR robot trajectories with the exact official pipeline.  Preserve the
robot target/actions, vary only the human visual domain, rerun semantic and physical gates, and call
the result cross-embodiment only if the human and G1 streams are genuinely distinct.

## 5. Automatic decision rules

- Official code/weights absent: finish manifest/data audit, record `official_release_available=false`,
  and do not train a substitute.
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
