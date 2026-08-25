# Same-teacher demo-following

The active experiment fixes the CarryBox45 official Refiner teacher in both arms and changes only
the selected reward demo: CarryBox45 (`correct`) or KickBox21 (`unrelated`). The active
`phase_event_reward_only` design uses the frozen phase-aware contact/event scorer and stops at
updates 32 and 64. Active experiments execute without artificial human-authorization gates.

## Current transition-recovery verdict

The causal recovery objective first failed its balanced single-prefix formal test on seeds
`171640/171641`: learned and exact pre-update Kick are tied at `35/40` safe and `4/40` falls.
The completed multi-context follow-up cycles online physical Carry handoffs `41/49/57` and uses
training/evaluation pairs `171642 -> 181652` and `171643 -> 181654`. Learned versus exact pre-update
totals are `97/120` safe for both and `10/11` falls. Only seed171642 improves, so the benefit is not
replicated. Same-checkpoint selected Kick/Carry behavior remains distinct. Do not add residual
updates, reward scales or a third seed; the next controller must learn a causal state-dependent
composition of both exact released actions under the same frozen comparison.

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

Reproduce one fresh multi-context seed inside a retained GPU step with:

```bash
TRAIN_SEED_OVERRIDE=171642 EVAL_SEED_OVERRIDE=181652 VIDEO_SEED_OVERRIDE=181653 \
  bash scripts/sugar/demo_following/run_multi_context_transition_recovery.sh \
  experiments/demo_following/reproduce_multi_context_seed171642 cuda:0
```

The runner continuously trains, audits, evaluates all three contexts and renders the paired H.264
videos. It contains no manual authorization stage.

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
collapse in both arms, not Carry/Kick semantic separation. The automatic next branch is therefore
the contact/event reward redesign documented in `PLAN/README.md`; do not repeat this schedule over
more seeds.
