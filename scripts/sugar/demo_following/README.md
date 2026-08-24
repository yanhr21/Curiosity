# Same-teacher demo-following

The active experiment fixes the CarryBox45 official Refiner teacher in both arms and changes only
the selected reward demo: CarryBox45 (`correct`) or KickBox21 (`unrelated`). The active
`phase_event_reward_only` design uses the frozen phase-aware contact/event scorer and stops at
updates 32 and 64. No policy training may start without explicit user approval.

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

After explicit approval, run the two arms serially inside a retained GPU allocation with
`--policy-training-authorized`. Use `scripts/sugar/native_tactile/launch_retained_child.sh` and a
fresh `--output-root`. After both endpoints pass, evaluate updates 32/64, run the independent
behavior audit for each checkpoint, and render update 64 with:

```bash
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 phase_event_reward_only /absolute/path/to/output/seed161587
```

Set `TEACHER_ONLY_GATE=1` on the same command to run the zero-residual prerequisite gate. Existing
complete results and videos are validated and reused; incomplete directories are never silently
overwritten.

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
