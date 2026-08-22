# Newton-native RL for CarryBox

Retrain (or fine-tune) SUGAR's tracker against Newton's contact model, because
`validation/g1_carrybox_policy.py` showed the official `tracker.pt` only partially
transfers: it lifts the box about a third of the reference height, and its wrists sit at
their effort limit 37% of the carry against Isaac's 1.9%.

    # smoke test
    python -m sugar_newton.rl.train --num-envs 16 --iterations 4 --clips data_000

    # the intended use: fine-tune from the official checkpoint
    python -m sugar_newton.rl.train --num-envs 512 --iterations 2000 \
        --warm-start SUGAR/demo_ckpts/CarryBox/tracker.pt --out runs/finetune

Run inside the Newton container; `renders/render_carrybox_policy.sh` in the `third_party/newton`
submodule shows the srun incantation.

## What is faithful to SUGAR, and what is not

Faithful, transcribed rather than re-derived:

- the 510-D observation, validated offline against Isaac's own recorded actions to
  RMSE 0.088 by `validation/verify_tracker_obs.py`
- actuator gains, armature, effort limits and the `0.25 * effort / stiffness` action scale
- reward weights and stds (`BaseRewardsCfg`) and the `exp(-error/std^2)` term shapes
- termination thresholds (`BaseTerminationsCfg`)
- PPO hyperparameters (`BasePPORunnerCfg`)

Deliberate gaps, all recorded in the code:

- **Four reward terms are missing**: `feet_slide`, `feet_air_time`, `undesired_contacts`,
  `hoi_contact`. All four need per-body contact forces, which the env does not surface
  yet. See `rewards.OMITTED`.
- **The critic is not privileged.** SUGAR's critic takes an 890-D observation built from
  future reference frames and teacher terms; this one takes the same 510-D actor
  observation and trains from scratch, so `--warm-start` loads the actor only.
- **Timeout bootstrapping.** GAE cuts the value target at every `done`, timeouts included,
  which biases truncated episodes low.

## Cost of accurate contact

Worlds are replicated with `spacing=(0,0,0)` on Newton's own advice -- separated worlds
are numerically worse, and worlds do not collide. Note that Newton's `replicate` docstring
recommends `approximate_meshes()` before replication so one simplified mesh is shared;
that is convex approximation of interacting geometry, which this project does not do, so
model construction scales with the real 45k-triangle hand and 50k-triangle box. Expect
build time, not step time, to dominate at high `--num-envs`.
