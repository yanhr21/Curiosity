# Newton-native RL for CarryBox

Retrain (or fine-tune) SUGAR's tracker against Newton's contact model, because
`validation/g1_carrybox_policy.py` showed the official `tracker.pt` only partially
transfers: it lifts the box about a third of the reference height, and its wrists sit at
their effort limit 37% of the carry against Isaac's 1.9%.

    # Run only inside an H200 Slurm compute shell entered from tmux.
    export NEWTON_PY=/public/home/yanhongru/envs/isaac_arena_py312/bin/python
    export PYTHONPATH=/public/home/yanhongru/envs/newton_warp_114:$PWD/third_party/newton:$PWD
    export SUGAR_RSL_RL_ROOT=/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/rsl_rl

    # execution smoke (raw motions; never formal evidence)
    $NEWTON_PY -m sugar_newton.rl.train_bcppo --num-envs 1 --max-iterations 1 \
        --motion-root SUGAR/data/CarryBox --clips data_000 --logger tensorboard

    # formal command shape (official processed Refiner rollout as student data)
    $NEWTON_PY -m sugar_newton.rl.train_bcppo --num-envs 8 --max-iterations 30001 \
        --motion-root experiments/sugar_reproduction/outputs/newton_refiner_dataset_20260827/rollout_datasets/refiner/rl_dataset \
        --teacher-motion-root SUGAR/data/CarryBox \
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

The motion inputs follow the official `SUGAR/train.sh` contract. `--motion-root` is the
Refiner rollout `rl_dataset` seen by the Tracker student and critic;
`--teacher-motion-root` defaults to raw `SUGAR/data/CarryBox` for the frozen Refiner.
Folders named `data_{motion_id}_{env_id}` are aligned with raw `data_{motion_id}` exactly
as `MotionCommand` does. Missing IDs or length differences above two frames stop before
training. Pointing both roots at raw data is allowed only for the three-iteration execution
smoke and is not faithful Tracker training.

Formal training fails closed if `rewards.OMITTED` becomes non-empty. The official
contact-dependent `feet_slide`, `feet_air_time`, `undesired_contacts` and `hoi_contact`
terms are reduced from Newton's resolved contact forces with the official three-frame
history. `validation/contact_rewards.py` passes on H200 with live body and box-filtered
forces, independent bilateral-contact comparison, excluded-body audit and a controlled
short-air event (`-0.6`, weighted `-3.0`). The validated default is eight worlds; the old
512-world default was never benchmarked and is not retained.

The official H200 Refiner rollout produced 912 endpoint-complete student clips from 1000
worlds. They cover 95 source motion IDs; `18/40/48/60/66` had no successful Refiner
trajectory and remain an explicit coverage limitation. Processed clips contain the exact
14 configured Tracker bodies, while raw teacher clips contain 35 URDF bodies. The loader
keeps separate indices for those layouts and rejects every other body count, non-finite
array or student/teacher length difference above two frames.

The formal seed-0 run started from scratch on Slurm H200 job `262332` with eight worlds and
all 912 admitted clips. Iteration 0 completed 192 transitions with mean reward `32.08`,
distillation weight `1.0`, zero divergences and all four contact terms present. Logs and
checkpoints are under
`experiments/sugar_reproduction/outputs/newton_bcppo_h200_20260827/logs/carrybox_bcppo_seed0`.
This is a training/runtime result, not yet a physical transfer result.

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

### Where the time goes, by elimination

Measured at 8 worlds. This is physics only -- there is no viewer anywhere in the training
loop or the benchmark; video rendering happens once per `--video-interval` and is separate.

* `obs+rew`: ~11 ms of a ~690 ms step -> **1.6%**. Not the bottleneck.
* solver iterations: dropping `iterations`/`ls_iterations` from 100/50 to 10/5 takes the
  step from 687.5 ms to 602.2 ms -> **~12%** for a tenfold reduction. Not the bottleneck.
* remainder -> **~85% is collision detection**, over a 45 748-triangle hand and a
  100 000-triangle box, generating 1 677-6 082 contacts per world.

### The lever: the asset's own collision approximation

`descriptions/objects/small_box/Props/instanceable_meshes.usd` authors
`physics:approximation = 'convexDecomposition'` on the box mesh, and Isaac honours it.
This environment collides the raw 100 000-triangle mesh instead.

Note what the box actually is, because it matters: it is an **open carton**, not a solid
block. Mesh volume 3 889.9 cm^3 against a convex hull of 35 502.8 cm^3 -- a single hull
would fill it in and add 813% material, which would be a real change to the physics and is
exactly what the no-hulling rule is about. Convex *decomposition* is a different thing: it
preserves the concavity with a set of convex parts, and it is the setting the asset itself
declares. Honouring it is arguably more faithful than what is done here now.

The hand is concave too (hull adds 135%), so the same distinction applies there.

Part of the gap is not a defect:  Isaac is fast partly *because* its URDF importer hulls
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
