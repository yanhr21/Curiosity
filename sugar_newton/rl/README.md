# Newton-native RL for CarryBox

Retrain (or fine-tune) SUGAR's tracker against Newton's contact model, because
`validation/g1_carrybox_policy.py` showed the official `tracker.pt` only partially
transfers: it lifts the box about a third of the reference height, and its wrists sit at
their effort limit 37% of the carry against Isaac's 1.9%.

    # smoke test
    python -m sugar_newton.rl.train_bcppo --num-envs 16 --max-iterations 3 \
        --clips data_000 --logger tensorboard

    # training, logging to wandb
    python -m sugar_newton.rl.train_bcppo --num-envs 512 --max-iterations 30001 \
        --wandb-project sugar_newton --run-name carrybox_bcppo

Run inside the Newton container; `renders/render_carrybox_policy.sh` in the `third_party/newton`
submodule shows the srun incantation.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: BCPPO trains on Newton and logs to wandb

First working run, 8 worlds, 3 iterations, `data_000`
(https://wandb.ai/nvr-amri/sugar_newton/runs/acs79r73)::

    iter 0   reward  9.48   ep_len 10.75   noise std 0.51   diverged 0
    iter 1   reward  7.54   ep_len 11.19                    diverged 0
    iter 2   reward  6.23   ep_len 11.21                    diverged 0

Per-term at iteration 0: anchor_pos 0.869, anchor_ori 0.795, body_pos 0.920,
obj_pos 0.853, obj_ori 0.946, obj_ang_vel 0.934, joint_pos 0.100.
Three iterations is far too short to read a trend from; what it establishes is that the
port runs, stays stable and logs.

### The BC curriculum is the stability mechanism

Against the hand-written PPO that used to live here (deleted), same environment, from
scratch::

                        hand-written PPO        SUGAR's BCPPO
    mean reward         -15.8 -> -48948         +9.48
    divergences         47 -> 138 -> 234        0
    joint_acc term      ~2e11                   883929

That last row settles an earlier false alarm. 883929 sits inside the range measured from
Isaac's own rollouts (mean 25.9k, worst step 744k), so `joint_acc` was never
mis-specified -- the flailing policy was the entire problem, and stages 1-2 remove it.

### Contact limits were wrong, and that invalidated everything measured before

`njmax`/`nconmax` are per world, and they must be sized for the WORST case. Leaving them
`None` lets Newton size from the initial near-static pose, which gives `nconmax=1024`;
real motion generates up to **6524 contacts per world**, and MJWarp silently drops
everything above the limit (489 `exceeded MJWarp limit` messages in one short benchmark,
144 `nefc overflow`). So the physics was wrong wherever contact matters most -- exactly
during the grip -- and every number measured before this was measured on it. Now 8192 /
2048 per world, with headroom over the measured peaks. Overflow count: 0.

### Speed, measured against correct physics

    envs   step_ms   obs+rew   contacts   env-steps/s
      1     133.8m     10.6m       1677      7.5
      2     231.9m     10.7m       2437      8.6
      4     258.8m     10.8m       3011     15.5
      8     686.8m     10.7m       4422     11.6
     16    1654.3m     10.9m       6082      9.7

Trustworthy:

* `obs+rew` is ~11 ms and flat in world count -- about 1.6% of a step at 8 worlds. The
  observation and reward code is not the bottleneck.
* ~8-15 env-steps/s overall, and it does not improve past 4 worlds. Step spreads are
  91-619 ms, so treat these as approximate.

NOT trustworthy, and not to be quoted: the `collide`/`solve` split. The consistency check
`(collide + solve + obs) / step` comes out at 1.13x-3.16x, and standalone `collide`
exceeds the whole step that contains it -- most likely GPU work that overlaps inside
`step` but serialises when timed in a tight loop. The decomposition is wrong; only the
end-to-end `step_ms` and the pure-torch `obs+rew` column mean anything.

At ~12 env-steps/s, SUGAR's 30k iterations x 24 steps x 8 envs is about 5.8M env-steps,
i.e. **~5.5 days**. SUGAR trains in Isaac with thousands of environments. One to two
orders of magnitude are missing and more worlds alone will not supply them.

Part of that gap is not a defect: Isaac is fast partly *because* its URDF importer hulls
every collider, and this project does not hull interacting geometry. Accurate contact
costs more. An untested idea worth trying: keep exact meshes for the hands and the box,
simplify colliders on links that never touch the box.

## Evaluation videos

`--video-interval N` (default 100) renders a deterministic rollout and logs it to wandb as
`video/rollout`, alongside `video/box_lift` and `video/box_lift_reference` so the clip has
a number attached. `--video-interval 0` disables it.

A separate one-world environment is used: the training worlds are replicated at zero
spacing and sit on top of each other, so rendering the training model shows every robot
superimposed. Actions are the policy mean rather than a sample, and the clip and start
frame are fixed, so successive videos differ because the policy changed. Rendering needs
`renders/render_env_egl.sh` sourced -- without it the viewer silently falls back to
software rasterisation and each frame costs seconds; the recorder warns once if it detects
this. A render failure is caught and logged, never allowed to end a training run.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: BCPPO trains on Newton and logs to wandb

First working run, 8 worlds, 3 iterations, `data_000`
(https://wandb.ai/nvr-amri/sugar_newton/runs/acs79r73)::

    iter 0   reward  9.48   ep_len 10.75   noise std 0.51   diverged 0
    iter 1   reward  7.54   ep_len 11.19                    diverged 0
    iter 2   reward  6.23   ep_len 11.21                    diverged 0

Per-term at iteration 0: anchor_pos 0.869, anchor_ori 0.795, body_pos 0.920,
obj_pos 0.853, obj_ori 0.946, obj_ang_vel 0.934, joint_pos 0.100.
Three iterations is far too short to read a trend from; what it establishes is that the
port runs, stays stable and logs.

### The BC curriculum is the stability mechanism

Against the hand-written PPO that used to live here (deleted), same environment, from
scratch::

                        hand-written PPO        SUGAR's BCPPO
    mean reward         -15.8 -> -48948         +9.48
    divergences         47 -> 138 -> 234        0
    joint_acc term      ~2e11                   883929

That last row settles an earlier false alarm. 883929 sits inside the range measured from
Isaac's own rollouts (mean 25.9k, worst step 744k), so `joint_acc` was never
mis-specified -- the flailing policy was the entire problem, and stages 1-2 remove it.

### Open: throughput

~3.6 env-steps/s at 8 worlds (collection 43-53 s for 192 timesteps). Algorithm-independent
-- the same figure appeared under the hand-written PPO -- so it is a property of the
environment. `njmax`/`nconmax` were being scaled by `num_envs` when they are per world;
that was wrong and is fixed, and fixing it changed nothing. A per-component breakdown
across 1/2/4/8/16 worlds is the next measurement; the hypothesis to test is that the
broad phase is N-by-N across all shapes in all worlds rather than per world, which would
make cost quadratic in world count.

## The algorithm is SUGAR's, imported not reimplemented

`train_bcppo.py` imports `BCPPO` from `SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`
and runs it inside `rsl_rl`'s own `OnPolicyRunner`, with the hyperparameters transcribed
from `BCPPORunnerCfg`. `BCPPO` is registered by the same mechanism SUGAR uses
(`setattr(builtins, "BCPPO", ...)`, `scripts/sugar_rl/train.py:147-150`), because the
runner resolves the algorithm with `eval(alg_cfg["class_name"])`.

The only local code in the training loop is `vec_env.py`, which presents the Newton
environment as an `rsl_rl.env.VecEnv` with the three observation groups the config asks
for:

    policy   510-D   validated against Isaac's recorded actions to RMSE 0.088
    critic   890-D   obs_890.py
    teacher  890-D   obs_890.py -- what the frozen refiner is asked to imitate

BCPPO's curriculum, for reference:

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

The teacher checkpoint is required, not optional: `BCPPO.__init__` asserts on a missing
one, and stages 1-2 have no loss without it. Default path is the recovered
`refiner_model10000.pt` (see TODO 16).

An earlier version of this directory carried a hand-written PPO (`ppo.py`, `train.py`).
It has been deleted. It was stage 3 with the distillation dropped, which is not the
algorithm SUGAR trains the tracker with.

## Logging

`rsl_rl` has native wandb support, so nothing here writes to wandb directly: the runner
config sets `logger: wandb` and `wandb_project`, and the run name is the log directory's
basename. Credentials follow the convention used elsewhere in this workspace --
`WANDB_API_KEY` from the environment, else `~/.netrc` for `api.wandb.ai` -- and
`train_bcppo.py` checks for one before building the environment, so a run cannot get
several minutes in with logging silently off. Pass `--logger tensorboard` to opt out.

## STATUS: environment runs; BCPPO port is UNTESTED

The numbers below are from the deleted hand-written PPO, from scratch with no teacher.
They are kept because the throughput figure is a property of the environment, not the
algorithm, and it is still unexplained. The reward collapse should not recur under BCPPO,
whose stages 1-2 exist precisely to stop the policy flailing -- but that is a prediction,
not a measurement.

Measured, 16 worlds, from scratch, 3 iterations, hand-written PPO (deleted):

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
