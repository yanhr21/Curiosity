# Phase 03 Curiosity Reward Replay V1

## Scope

This is the first Phase 03 curiosity reward evaluation on already validated
Phase 02 Newton cup lift-hold baseline rollouts. It evaluates reward components
and ablations before any policy adaptation.

This run is diagnostic replay evaluation only:

- no model training;
- no policy update;
- no hand-written placeholder T-Rex/VQ-VAE/world model;
- no exact T-Rex schema promotion;
- tactile source is `newton.contact_proxy_only`.

## Files

- Spec: `docs/curiosity_reward_spec_v1.md`
- Config: `experiments/configs/curiosity_reward_baseline_replay_v1.json`
- Evaluator: `experiments/configs/evaluate_curiosity_reward_baseline_replay.py`
- Launcher: `experiments/configs/launch_curiosity_reward_baseline_replay_tmux.sh`
- Allocation runner: `experiments/configs/run_curiosity_reward_baseline_replay_in_alloc.sh`
- Output JSON: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`
- Output CSV: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv`
- Log: `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`

## Command

Launched from login node as a lightweight tmux/srun dispatch into the existing
Curiosity allocation:

```bash
JOB_ID=154023 TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
  bash experiments/configs/launch_curiosity_reward_baseline_replay_tmux.sh
```

The compute-side runner executed on `server56`, activated the prebuilt local
venv at `envs/newton/.venv`, and reread `AGENTS.md` inside the compute job.

## Gate Checks

For every rollout, the evaluator required:

- fresh official Newton sanity JSON;
- camera export summary JSON;
- automated visual validation JSON;
- manual visual inspection JSON with pass status;
- Phase 02 lift-hold metrics JSON;
- rollout NPZ.

If any gate failed, the evaluator would stop before computing curiosity reward.

## Result

Status: pass.

Rollouts evaluated: 9.

Held-out cells included:

- `full_low`;
- `empty_high`.

Logged ablations:

- no curiosity;
- random intrinsic;
- object-motion-only;
- contact-only;
- tactile-only via contact proxy;
- vision+tactile via object plus contact proxy;
- shuffled tactile;
- delayed tactile.

Summary values:

| Mass | Friction | Held out | Intrinsic reward mean | Object error mean | Contact error mean | Learning progress proxy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| empty | medium | false | 0.0438140703 | 0.00004223899 | 1.2618384401 | 0.0085609402 |
| half | medium | false | 0.0397711353 | 0.00004301188 | 1.1559888579 | 0.0072078045 |
| full | medium | false | 0.0382908298 | 0.00004536419 | 1.2284122563 | 0.0090711557 |
| empty | low | false | 0.0433660021 | 0.00004222839 | 1.1030640669 | 0.0081110990 |
| half | low | false | 0.0392662636 | 0.00004284005 | 1.0807799443 | 0.0067484565 |
| full | low | true | 0.0384965341 | 0.00004588505 | 1.2924791086 | 0.0090151045 |
| half | high | false | 0.0414414675 | 0.00004086782 | 1.2256267409 | 0.0077095647 |
| full | high | false | 0.0394635577 | 0.00004351452 | 1.1754874652 | 0.0097291875 |
| empty | high | true | 0.0446768550 | 0.00004207343 | 1.0919220056 | 0.0095668467 |

## Interpretation

The replay reward is wired correctly over the validated mass/friction grid and
can now be used as the Phase 03 baseline-rollout reward-shape gate.

The next required step is not to call this a learned world model. The next step
is to add the learned forward-model target path or a faithful official model
adapter while preserving these logged components and the same sanity/visual
gate requirements.
