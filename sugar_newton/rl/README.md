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

## STATUS: the environment runs; the trainer does not train yet

Measured, 16 worlds, from scratch, 3 iterations:

    it 1  return    -15.8  ep_len 4.6  diverged  47   4 env-steps/s
    it 2  return  -1242.9  ep_len 2.8  diverged 138   4 env-steps/s
    it 3  return -48947.7  ep_len 1.8  diverged 234   3 env-steps/s

Two open problems, neither solved:

1. **Throughput.** 4 env-steps/s across 16 worlds is ~20x *worse* per world than the
   single-world validation scene (5.2 steps/s). `njmax`/`nconmax` are per world
   (`solver_mujoco.py:3183`) and an earlier version scaled `nconmax` by `num_envs`,
   allocating 128k contacts per world; that was wrong and is fixed, but fixing it changed
   the number not at all, so the cause is still unknown. Note episodes are terminating
   after ~2 steps here, so `reset()` runs almost every step -- that is the next thing to
   rule out, not a diagnosis.
2. **Reward explodes from scratch.** Diverged envs are detected and zeroed, so this comes
   from envs that are finite but violent.

   The `joint_acc` term was the suspect and has been **ruled out by measurement**. Taking
   Isaac's own recorded `joint_vel` from `isaac/rollouts_isaac` and differencing it exactly
   the way this env does gives `joint_acc_l2` of mean 25.9k and worst-step 744k, i.e. a
   reward contribution of -0.0065 mean and -0.19 at worst. SUGAR's -2.5e-7 weight is
   correctly calibrated for this quantity. Reaching a return of -4.9e4 needs
   `joint_acc_l2` around 2e11, about 8e6x Isaac's mean -- roughly 3000x Isaac's joint
   velocities. That is a flailing policy, not a mis-specified reward term.

   Which points at the missing BC curriculum: stages 1-2 exist precisely so the policy is
   never allowed to flail (below).

## The algorithm is NOT SUGAR's

SUGAR's tracker uses **BCPPO** (`BCPPORunnerCfg` -> `utils/rsl_rl_bcppo.py`), a three-stage
curriculum around the frozen refiner as teacher:

    stage 1   step < 500        loss = distill                       (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value     (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

Supervising signals in SUGAR's tracker training:

| signal | where it enters |
|---|---|
| RL reward, the 17 terms below | surrogate loss, stage 3 only |
| teacher action distribution, KL(teacher \|\| student) over the 29-D Gaussians | stages 1-2 dominant, fades in stage 3 |
| value targets (returns) | critic, warmed up alone in stage 2 |
| entropy bonus 0.005 | stage 3, scaled by alpha |
| adaptive-KL LR control, desired_kl 0.01 | fixed during stage 1 |

and three observation groups, not one: policy 510-D, critic 890-D privileged, teacher
890-D. Symmetry and RND exist in the class but neither config enables them for CarryBox.

**What is implemented here is stage 3 without the distillation term** -- plain PPO. The
teacher checkpoint is recovered (TODO 16) but the 890-D teacher group is not built, so
BCPPO is the next piece of work, and is likely a precondition for training stability
rather than an enhancement.

## What is faithful to SUGAR, and what is not

Faithful, transcribed rather than re-derived:

- the 510-D observation, validated offline against Isaac's own recorded actions to
  RMSE 0.088 by `validation/verify_tracker_obs.py`
- actuator gains, armature, effort limits and the `0.25 * effort / stiffness` action scale
- reward weights and stds (`BaseRewardsCfg`) and the `exp(-error/std^2)` term shapes
- termination thresholds (`BaseTerminationsCfg`)
- PPO hyperparameters, identical between `BasePPORunnerCfg` and `BCPPORunnerCfg` except
  `init_noise_std`, which is 0.5 for the tracker (an earlier version used the inference
  task's 1.0)

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
