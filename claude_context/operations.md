# Operating notes — Curiosity

How to run this repo on this cluster. Read before launching, submitting or
monitoring anything.

<!-- BEGIN GENERAL — generated, do not hand-edit
     source: context-site/references/operating-notes.md + clusters.md @ 451c10c
     cluster: OCI ORD
     content: sha256:2a3373245dc5
     refresh: python3 /home/shengzew/.claude/skills/context-site/make_operations.py <site-dir> -->

## General — portable rules

True on any cluster of this shape. The concrete values for the machine this repo
is on are in the section immediately below this one; where that section is
missing, these rules still hold and the values need recording.

**This block is generated — do not hand-edit it.** Corrections go to
`~/.claude/skills/context-site/references/operating-notes.md` (rules) or
`clusters.md` (values), then re-run
`python3 ~/.claude/skills/context-site/make_operations.py <site-dir>`. Anything
true only of *this repo* goes in the hand-written section at the end, which the
generator never touches.

### Filesystem

**`/tmp` is node-local; nothing a job reads may live there.** The login node's
`/tmp` is not the compute node's, and neither is mounted inside a container. A
wrapper written there and launched under `srun` fails with exit 127, and the
usual symptom is a job that appears to start and does nothing. Put every script a
job reads on shared storage.

**Learn which mounts are aliases before copying.** Two paths can be one
filesystem; copying makes a second copy that drifts and a "backup" that is not
one. Check before rsyncing between roots.

**Outputs are append-only.** Never delete an old result directory to make room;
pick a fresh name. Results get compared long after the run.

**Never write into shared dataset directories.** Read from them; write to
user-owned paths.

### Environment

**Activate last.** Sourcing a shell rc file after `conda activate` re-initialises
conda and silently drops you to `base`, so the job runs on the wrong interpreter
and dies on the first import. Set the variables you need with individual
`export`s instead of sourcing the whole file.

**One repo often needs several envs** — training, preprocessing and paper/figure
tooling are frequently three. Record which entrypoint needs which.

### Allocation and launch

**Do not reserve a whole node for one GPU.** `--exclusive` with a single GPU gets
reclaimed by the scheduler on shared partitions — jobs died within seconds. Use
it only when the job genuinely wants every GPU on the node.

**Pass container images the way the launcher expects.** Wrapping the flag in a
generic passthrough double-specifies it and the job exits 127 before running. If
autoresume then resubmits elsewhere, the failure will not look like the same job.

**Partition pool size matters more than the per-user job cap.** A partition that
lets you hold several nodes but contains very few will simply never schedule you.
Check how many nodes a partition actually has, not just what it permits.

**`srun --overlap` changes what a step can see.** Driving an existing allocation
it exposed half the GPUs with no warning, and training continued quietly at half
batch. Omit it for real work; fine for a read-only probe.

**Interactive allocations expire, and the shell does not say so.** On expiry the
screen window falls back to the login node silently, so work fails with "no
NVIDIA driver" or half-runs on CPU. Confirm a RUNNING job, then probe
`nvidia-smi`, before sending work in.

### Running and watching

**Long GPU work lives in a screen session**, never a shell backgrounded inside a
tool call — the disconnect orphans the `srun` step silently.

**Stuff a script, not a command.** `screen -X stuff` passes text through two
shells, so inline quoting, parens and pipes get double-interpreted and die on a
syntax error. Write it to a file on shared storage and stuff `bash /path/run.sh`.
The trailing `\n` is what submits it. `cd` to the repo first.

**Use the channel the user is watching.** If they have a named session, send
there — they cannot see `srun` output. Ask for the name rather than hunting.

**Do not block on long sleeps.** Check immediately, or hand over the monitoring
command. If a wait is unavoidable keep it under a minute.

### Diagnosis

**When a job retries, read the first log.** Autoresume repeats the same broken
code; the newest log shows the newest symptom, while the first attempt has the
clean traceback and proves the fault was there from the start.

**Liveness is two samples at least 90 s apart.** One stale mtime is not evidence.
Never cancel or kill on a single reading — work that was merely quiet is not
recoverable.

**Version control on a large tree.** `git status`/`commit` can wedge in D-state
scanning big untracked output trees on a networked filesystem; use
`git -c status.showUntrackedFiles=no …`, and `-prune` heavy directories in `find`
rather than filtering after the walk. Do not commit one-time operational edits
(a resume path, a temporary config) to tracked configs.

### Reporting

**Validation is not test.** In-training val is a handful of noisy samples and is
not comparable to a test-split evaluation. Label which a number is; describe
val-only findings qualitatively rather than tabling them.

**Paths quoted to a human are absolute.** Sibling checkouts across several roots
make relative paths ambiguous.

**wandb stays on** unless explicitly disabled.

## This cluster — OCI ORD

*From `clusters.md` §cs-oci-ord; detected as `cs-oci-ord` via slurm ClusterName.*

### Filesystem

| root | shared with compute | container-mounted | notes |
|---|---|---|---|
| `/lustre/fs12` | yes | yes | same files as `/lustre/fsw` |
| `/lustre/fsw` | yes | yes | same files as `/lustre/fs12` |
| `$HOME` | yes | yes | |
| `/tmp` | **no** | **no** | node-local; a job cannot read what you wrote on the login node |

`/lustre/fs12` and `/lustre/fsw` are **the same filesystem under two names** —
never rsync between them, and do not treat a path under one as a backup of the
other. Anything a job must read goes on Lustre or `$HOME`; a wrapper written to
`/tmp` and launched with `srun --container-image=… bash /tmp/foo.sh` fails with
exit 127.

### Allocation

| partition | nodes | time cap | per-user |
|---|---|---|---|
| `interactive` | 10 | 4 h | 3 nodes |
| `interactive_singlenode` | 1243 | 4 h | 1 node |
| `batch_singlenode` | 10 | 4 h | |
| `cpu_interactive` | 94 | 24 h | |

The pool size matters far more than the job cap: `interactive` lets you hold 3
nodes but has 10 in total, while `interactive_singlenode` has 1243 and is what
actually gets you a GPU. Both expire hard at 4 h, and on expiry a screen window
falls back to the login node silently — work then fails with "no NVIDIA driver"
or half-runs on CPU.

*Verified 2026-08-13 with `sinfo -o '%P %D %l'` and `scontrol show config`.*

### Launch

Account `nvr_nxp_visionconferencing`, login node `oci-ord-cs`.

`submit_job` takes `--image` / `--mounts` for containers. Passing the same thing
through `--more_srun_args="--container-image=…"` double-specifies it: `pyxis:
--container-image specified multiple times`, exit 127 — and autoresume then
resubmits to a *different* partition, so the failure does not look like the same
job.

---

<!-- END GENERAL -->

## This repo — Curiosity

*Hand-written. This script never touches anything outside the block above.*

> **Plan 15 does not run on OCI ORD.** Everything in the cluster block above
> describes the machine this checkout sits on — not the machine this repo
> executes on. IsaacLab/PhysX training, evaluation and video all run on a
> **separate host** rooted at `/public/home/yanhongru/Curiosity`, which does not
> exist here (`ls /public/home/yanhongru` → *No such file or directory*). This
> checkout is for reading, editing, testing and committing source. It cannot run
> a single Plan-15 command.
>
> The consequence people trip on: `experiments/` is gitignored and holds only
> `README.md` and `ACTIVE_PACKAGE_MANIFEST.json` here. Every checkpoint, trace,
> frozen-evaluation JSON, `patch_channel_scales.json` and video the README cites
> lives **only** on that host. A path in the README that starts with
> `experiments/` will not resolve on this filesystem, and that is expected — not
> a broken link.

### Environments

| task | env | pinned in |
|---|---|---|
| all IsaacLab/PhysX work | `/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python`, exported as `PYTHON_BIN` | `README.md` § 运行环境 — there is no env yml in the repo |
| unit tests `tests/native_tactile/` | any Python with `torch` + `numpy`, no IsaacLab import | the tests themselves — the only part of this repo runnable off the GPU host |

There is no conda env. `PYTHON_BIN` is an absolute interpreter path that every
shell script reads from the environment; forget to export it and the scripts fall
through to whatever `python` resolves to and die on the first `isaaclab` import.

Three variables must be set in **every** shell before an IsaacLab command:

```bash
export PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
export DISPLAY=                 # empty, not unset — headless Kit still probes it
export OMNI_KIT_ACCEPT_EULA=Y
```

### Filesystem

All paths below are on the **runtime host**, relative to
`/public/home/yanhongru/Curiosity`.

| what | where |
|---|---|
| formal endpoints (`model_2999.pt` + train metadata only) | `experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/<branch>_seed<N>/` |
| frozen evaluation, 5 masses × 20 profiles | `experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff/` |
| **frozen channel scales — required by every train and eval run** | `experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json` |
| official frozen Refiner teacher | `experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt` |
| TacSL R15 calibration | `experiments/sugar_reproduction/assets/official_tacsl/calibration` |
| synchronized world + 54-patch H.264 | `experiments/online_patch_tactile_mass_adaptation/visualizations/` |
| retained-child PID/PGID/status/log records | `experiments/online_patch_tactile_mass_adaptation/runtime/` |
| failed / superseded / historical runs — **not** a reproduction entry point | root `legacy/` (gitignored) |

`experiments/` and `legacy/` are both gitignored on purpose. Checkpoints, traces,
videos and logs must never be committed or pushed.

### Launching

There is **no SLURM in this repo's workflow**. Work runs inside a reserved GPU
compute-node shell, and long jobs are wrapped so their process group is
recorded — not backgrounded with `&`.

```bash
# one formal training seed — endpoint is fixed at 3000 updates
BRANCH=PS                       # Z, P, or PS
SEED=151016
SCALE="$PWD/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
OUT="$PWD/experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025/${BRANCH,,}_seed${SEED}"

"$PYTHON_BIN" -u SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py \
  --task "Sugar-G129dof-CarryBox-OnlineMass-Patch-${BRANCH}-BCPPO" \
  --patch-scale-file "$SCALE" --seed "$SEED" --log_dir "$OUT" \
  --headless --device cuda:0
```

`--patch-scale-file` is **not optional** — the launcher refuses to start without
it. Teacher and warm-start checkpoints are injected by
`_inject_official_training_contract`; passing anything other than the official
Refiner / official Tracker raises rather than silently substituting.

```bash
# frozen evaluation of one endpoint (PS additionally requires the explicit gate)
PLAN15_ALLOW_PS_ENDPOINT_EVALUATION=1 \
bash scripts/sugar/native_tactile/run_plan15_frozen_seed.sh \
  PS <endpoint>/model_2999.pt 151016 152016 <fresh-output-dir> cuda:0

# long-running sensing jobs: record the child process group
bash scripts/sugar/native_tactile/launch_retained_child.sh \
  --record <runtime>/leakage.process --status <runtime>/leakage.status \
  --log <runtime>/leakage.log --tag plan15-leakage --foreground -- \
  "$PYTHON_BIN" scripts/sugar/native_tactile/run_online_mass_leakage_sweep.py \
    --output-root <out> --device cuda:0
```

Off the runtime host, the only thing that runs is the test suite:

```bash
python3 -m pytest tests/native_tactile -q      # needs torch + numpy, not IsaacLab
```

### Watching a run

Follow the `--log` file the retained-child wrapper writes; `--status` holds the
exit state and `--record` the PID/PGID. Do not exit the allocation when a task
finishes — kill only the **recorded child process group** and start the next one
in the same shell. A fresh allocation means re-exporting all three environment
variables and re-importing the G1 USD.

A formal seed is 3000 updates and does not finish in one sitting; check back on
the log rather than blocking on a long sleep.

### Resume and silent failure

`--resume_checkpoint_path <OUT>/model_<N>.pt` resumes from a numbered checkpoint.
It is **mutually exclusive with `--warm_start_checkpoint_path`** and raises if
both are given. The endpoint stays 3000 updates regardless of where the resume
started.

The things that do *not* raise:

- **A broken tactile sensor looks like "Z wins."** The Z branch calls
  `exact_zero_online_patch_tactile_actor_history`, which never touches
  `env.scene.sensors` at all. Any TacSL misconfiguration degrades P and PS while
  leaving Z running perfectly — and the experiment's whole output is a
  P/PS-versus-Z comparison. Check the `_runtime_diagnostics` contact counters
  before believing a Z-favourable result.
- **`friction_utilization` cannot see the object's friction.** It divides by the
  *sensor config's* `friction_coefficient` (pinned to `0.5` by
  `CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT`), and TacSL already caps the
  shear numerator with that same constant. A friction sweep changes the box's
  PhysX material and leaves this channel's definition untouched, so the run
  completes normally and produces a tactile stream that is blind to the variable
  under test.
- **The teacher prefix stores the wrong action.**
  `OnlineTeacherHandoffVecEnvWrapper.step` executes the *teacher's* action but
  the rollout buffer keeps the *student's* sampled action and log-prob. That is
  only safe because `training_mask_obs_group` drops those transitions from the
  surrogate, value and entropy terms. Unset or mis-name that group and the
  prefix silently becomes ordinary on-policy PPO data with mismatched actions.
  Note the distillation loss is *not* masked (`rsl_rl_bcppo.py:77-82`).
- **The channel scales are read at import time.**
  `patch_channel_scales: list[float] = _patch_channel_scales()` is a class-body
  default in `rsl_rl_online_patch_mass_bcppo_cfg.py:44`, evaluated when the
  module is imported. The launcher works only because `_consume_scale_file` sets
  `SUGAR_ONLINE_PATCH_CHANNEL_SCALES` before the import. Today an unset variable
  yields nine `NaN`s and the actor raises — loud. Anyone who "fixes" that
  sentinel to a plausible default turns it silent, and the run trains on
  arbitrary normalization.
- **The configured learning rate is not the one that runs.** `train.py`'s warm-start
  path overwrites every optimizer `param_group["lr"]` with the LR stored in the Tracker
  checkpoint's own optimizer state, and BCPPO holds `schedule="fixed"` for updates 0–499. From update 500 the
  schedule flips to `adaptive` and the KL controller moves the LR ×1.5/÷1.5 **per
  mini-batch** (20× per update) within `[1e-5, 1e-2]` — starting from the Tracker's LR,
  never from the configured one. `BCPPOCfg.learning_rate = 1.0e-3` is used only to build
  Adam; read the LR out of the run log, not the config.
- **A logged `surrogate` of exactly 0.0 means "no post-handoff transitions in that
  update", not convergence.** All three PPO terms are masked by the handoff mask and
  reduced with a `clamp_min(1.0)` denominator, so an all-prefix update logs `surrogate =
  value_function = entropy = 0.0` and degenerates to pure behaviour cloning. With 4 envs
  starting in lockstep this happens for the first 1–15 rollouts of every episode. The
  report that would distinguish the two, `alg.last_training_mask_report`, is written every
  update but only *read* on `-Preflight-` runs — formal training keeps no record of it.
- **`configure_tactile_actor_finetune` does not gate anything.** It overrides the parent
  class's freeze-and-mask implementation without calling `super()`, and since nothing in
  this path ever set `requires_grad=False`, it is a no-op plus a report dict. No
  `mask_base_columns` hook is installed, so the 504 warm-started Tracker columns train
  from update 0 — the "≡ the released Tracker" property holds only for the single forward
  pass inside the equivalence audit. `train.py` still calls it "the tactile finetune gate"
  and writes that description into `tactile_finetune_resume.json`.
- **`warm_start_tactile_gain = 0.01` is a one-shot init trick, not a standing
  constraint.** Adam's per-step displacement is ≈ lr regardless of a parameter's current
  magnitude, and the schedule runs 20 optimizer steps per update, so at the configured LR
  the 100× down-scaling of the actor's patch columns is undone inside a single BCPPO
  update.
- **Nothing binds a `patch_channel_scales.json` to the channel definitions that produced
  it.** The scales are baked into the encoder's persistent buffer and therefore into every
  checkpoint, so a checkpoint and a scale file can be recombined incorrectly with no error
  at all. This matters directly for any sensing fix, since every one of them changes those
  definitions.
- **`contact = 0` means "not touching the box", not "not touching anything".** The TacSL
  sensors observe only `{ENV_REGEX_NS}/Obj`, and only its first SDF child mesh. Contact
  with the ground, the other hand or any other body produces exactly zero tactile output.
- **`torch.nan_to_num` silently converts a NaN force into a zero-load reading**, and the
  downstream `torch.isfinite(output).all()` check is vacuous because sanitisation already
  happened. A `+inf` penetration survives as an *active* taxel.
- **The reward pays the policy for not touching the box.** `hoi_contact` (+1.0) reads
  ContactSensors on `left_rubber_hand`/`right_rubber_hand`, whose collision subtrees the
  robot spawner deactivates, so `is_contact` is permanently `False` and the term rewards
  agreement with a *no-contact* reference. `undesired_contacts` (−1.0) matches every body
  except the ankles and those two hands — which includes all 54 elastomer patches. Nothing
  raises; the run trains normally against an objective that is anti-correlated with
  grasping.
- **Training and evaluation use different motions.** Training assigns
  `motion_id = env_id % num_motion` (motions 0–3 at the default `num_envs = 4`); the
  evaluator pins motion 45 by monkey-patching `_sample_init_state`. Every reported number
  is out-of-distribution and nothing in the pipeline flags it.
- **`episode_length_s` differs between the training and Play configs.** The training
  configs inherit `30.0`; the `*PlayEnvCfg` classes the evaluator uses set `1.0e9`. Reading
  the `1.0e9` lines and assuming they apply to training is an easy and consequential
  mistake.
- **The patch history is stale on the first frame of every evaluation batch after the
  first.** `_patch_history` caches on `common_step_counter`, which the evaluator's
  `env.reset()` does not advance, and the reset refill sits inside the recompute branch.
  Affects P and PS, not Z.
- **The acting teacher's SHA-256 pin is disabled.** `OnlineTeacherHandoffVecEnvWrapper`
  constructs `FrozenOfficialRefinerTeacher` with `expected_sha256=None`, so only path
  equality and `iter == 10000` are checked.
- `evaluate_online_patch_mass_bcppo.py` raises `FileExistsError` when
  `--output-root` already exists. Loud, but it costs you the job — always point a
  rerun at a fresh directory.

### Standing rules

- **Never commit or push `experiments/` or `legacy/`.** Source, tests and docs
  only.
- **Never merge the high-friction 6×/10× feasibility sweep into the Z/P/PS
  statistics.** Changing friction changes pickup dynamics and moves the jump
  frame (325–328), so those rollouts are not matched to the formal comparison.
- **Never auto-start the next formal seed.** Each endpoint stops exactly at
  `model_2999.pt` and passes the eight-item review in `TODO` § D before its
  frozen evaluation is opened, and before the next seed begins.
- **Never substitute a toy model** for the serious SUGAR policy, official Tracker
  warm start, frozen Refiner teacher or repository BCPPO (`AGENTS.md` § 2).
- **Never let privileged signals into the actor or the slip callable** — measured
  object state, mass factor, jump flag, relative contact velocity, RGB or future
  frames are evaluation labels only (`AGENTS.md` § 3). *This rule is currently
  violated in substance: TacSL derives shear from `relative_velocity_world`, so
  `friction_utilization` carries contact relative velocity into
  `PatchSlipDetector`. See finding F-0008.*
- **Never claim "only tactile can sense mass."** Mass leaks into proprioception
  through joint sag and tracking error; the only defensible claim is an
  incremental benefit over the 504-D deployable base.
- A camera-enabled rollout describes **its own** rollout. It is never a
  frame-by-frame replay of a camera-free formal trace.
