# Phase 03: Curiosity Reward

## Goal

Define and implement intrinsic objectives around physical prediction, not raw
pixel prediction.

## Reward Components

- object-motion prediction error;
- contact prediction error;
- unexpected object acceleration or slip proxy;
- lift-response mismatch under expected mass;
- bounded impact/useful-change reward;
- safety penalty for excessive force or unstable motion;
- no-op penalty;
- learning progress over time.

## Required Design

Curiosity must target physical learning progress, not raw RGB novelty. The
intrinsic reward should use bounded and controllable prediction improvement:

```text
intrinsic_reward =
  learning_progress(object/contact/tactile prediction)
+ controllable_disagreement
+ bounded_useful_change
- safety_penalty
- no_op_penalty
- excessive_force_penalty
```

The first forward models should predict object pose delta, object velocity,
contact count/proxy, slip or contact-loss risk, tactile-marker response when
available, and success/failure risk. Raw prediction error can be logged, but it
must not be the only reward because chaotic collisions and unstable drops can
produce large errors without useful learning.

## Required Ablations

- no curiosity;
- random intrinsic reward;
- object-motion-only curiosity;
- contact-only curiosity;
- tactile-only curiosity;
- vision+tactile curiosity;
- shuffled tactile;
- delayed tactile.

## Completion Criteria

- Reward spec exists.
- Components are logged separately.
- Reward can be evaluated on baseline rollouts before it is used for policy
  adaptation.
- Held-out mass/friction results show whether curiosity improves adaptation
  without hiding safety failures.

## Completed Replay Evaluation V1

2026-06-27: implemented and ran the first Phase 03 replay evaluator on the
validated Phase 02 cup mass/friction grid.

- Spec: `docs/curiosity_reward_spec_v1.md`.
- Config: `experiments/configs/curiosity_reward_baseline_replay_v1.json`.
- Evaluator: `experiments/configs/evaluate_curiosity_reward_baseline_replay.py`.
- Launcher: `experiments/configs/launch_curiosity_reward_baseline_replay_tmux.sh`.
- Compute runner: `experiments/configs/run_curiosity_reward_baseline_replay_in_alloc.sh`.
- Output JSON: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- Output CSV: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv`.
- Log: `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`.

Execution used existing tmux-held allocation `154023` on `server56`, activated
the prebuilt local shared-filesystem Newton venv at `envs/newton/.venv`, and
reread `AGENTS.md` inside the compute job.

V1 is a diagnostic replay reward evaluator only. It trains no model, creates no
placeholder VQ-VAE/world model, and does not claim exact T-Rex schema
compatibility. It uses Newton contact count as `newton.contact_proxy_only`
because tactile marker fields are not present in the Phase 02 rollouts.

Replay gate result:

- status: pass;
- evaluated rollouts: 9;
- held-out cells included: `full_low`, `empty_high`;
- required gates checked per rollout: fresh official Newton sanity, camera
  summary, automated visual validation, manual visual inspection, and Phase 02
  lift-hold metrics;
- ablations logged: no curiosity, random intrinsic, object-motion-only,
  contact-only, tactile-only via contact proxy, vision+tactile via object plus
  contact proxy, shuffled tactile, delayed tactile.

This clears the baseline-rollout reward-shape gate before policy adaptation.
It does not clear the later learned forward-model or true tactile-marker
representation requirement.

## Forward-Model Target Contract V1

2026-06-27: added the first learned forward-model target contract without
implementing or claiming a learned world model.

- Spec: `docs/residual_adapter_forward_model_contract_v1.md`.
- Config: `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- Status: `target_contract_ready_training_not_started`.

Defined targets:

- object pose delta;
- object velocity;
- contact proxy next step;
- slip risk;
- contact-loss risk;
- lift-response residual;
- success/failure risk;
- tactile-marker response, blocked until real `taccel.marker.*` evidence is
  added.

This satisfies the target-definition gate for learned curiosity, but not the
training gate. No placeholder MLP, VQ-VAE, Transformer, or world model has been
created.
