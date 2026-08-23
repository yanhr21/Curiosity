# Same-teacher demo-following

The active experiment fixes the CarryBox45 official Refiner teacher in both arms and changes only
the selected reward demo: CarryBox45 (`correct`) or KickBox21 (`unrelated`). Defaults are the
current `same_teacher_reward_only` design and 64 updates.

Validate without simulation:

```bash
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
$PYTHON scripts/sugar/demo_following/run_matched_state_predictor.py \
  --arm correct --stop-after-segment --dry-run
```

Run the two arms serially inside a retained GPU allocation. Use
`scripts/sugar/native_tactile/launch_retained_child.sh` and pass a fresh `--output-root`. After both
endpoints pass, evaluate and render with:

```bash
bash scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh \
  64 same_teacher_reward_only /absolute/path/to/output/seed161581
```

Set `TEACHER_ONLY_GATE=1` on the same command to run the zero-residual prerequisite gate. Existing
complete results and videos are validated and reused; incomplete directories are never silently
overwritten.

The archived historical design changed both teacher and reward demo and is not an active result.
Exact assets, commands, expected outputs and claim boundaries are in `DOCS/reproducibility.md`.
