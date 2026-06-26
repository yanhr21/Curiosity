# Phase 03 TODO: Curiosity Reward

- [x] Write post-pivot curiosity reward spec.
      Evidence: `docs/curiosity_reward_spec_v1.md`.
- [x] Add object-motion prediction error.
      Evidence: replay evaluator logs constant-velocity diagnostic object
      prediction error.
- [x] Add contact prediction error.
      Evidence: replay evaluator logs contact-persistence diagnostic error from
      `newton.panda.rigid_contact_count`.
- [x] Add bounded impact/useful-change reward.
      Evidence: replay evaluator logs clipped useful lift progress while
      contact is present.
- [x] Add safety/excessive-force penalty.
      Evidence: replay evaluator logs acceleration, drop, and contact-force
      proxy penalties.
- [x] Add no-op penalty.
      Evidence: replay evaluator logs active-command frames with no contact and
      no object motion.
- [x] Add learning-progress metric.
      Evidence: replay evaluator logs a replay-window learning-progress proxy;
      this is diagnostic only, not policy learning.
- [x] Prefer learning progress and bounded useful change over raw prediction
      error before using curiosity for policy updates.
      Evidence: `docs/curiosity_reward_spec_v1.md` and
      `experiments/configs/curiosity_reward_baseline_replay_v1.json`.
- [ ] Add learned forward-model targets for object pose delta, object velocity,
      contact proxy, slip/contact-loss risk, and tactile-marker response when
      tactile evidence exists. Current V1 only adds diagnostic replay predictors
      and must not be treated as a learned world model.
- [x] Run curiosity ablations: no curiosity, random intrinsic reward,
      object-motion-only, contact-only, tactile-only, vision+tactile, shuffled
      tactile, and delayed tactile.
      Evidence: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- [x] Validate reward on baseline rollouts before policy adaptation.
      Evidence: compute run in `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`
      passed all 9 Phase 02 rollout gates before evaluation.
