# Same-teacher demo-following

The active experiment fixes the CarryBox45 official Refiner teacher in both arms and changes only
the selected reward demo: CarryBox45 (`correct`) or KickBox21 (`unrelated`). The active
`phase_event_reward_only` design uses the frozen phase-aware contact/event scorer and stops at
updates 32 and 64. Active experiments execute without artificial human-authorization gates.

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
