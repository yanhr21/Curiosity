# Carrying Visualization Diagnostic Report

Date: 2026-07-02.

## Scope

This report records the first visual diagnostic toward:

```text
robot approaches a box -> probes unknown load -> adjusts carrying posture
-> lifts and carries with visualization
```

The implemented artifact is a browser-only visualization under:

- `src/carrying_visualization/index.html`
- `src/carrying_visualization/app.js`
- `src/carrying_visualization/styles.css`
- `src/carrying_visualization/diagnostic_storyboard.svg`
- `src/carrying_visualization/simulation_manifest.json`
- `src/carrying_visualization/README.md`

This is a small browser kinematic simulation platform. It is not a
physics-accurate simulator, not RL, and not a success claim. It is a
diagnostic visual scaffold for the behavior sequence and posture-selection
logic.

## Behavior Implemented

The visualization shows a simplified humanoid that:

- approaches a box on a table;
- probes the box with a push-pull / micro-lift visual cue;
- selects a carrying strategy from box mass, width, COM offset, friction,
  robot height, arm reach, and torque capacity;
- adjusts stance width, torso lean, hold height, and gait speed;
- lifts and carries the box;
- displays phase, strategy, effort, balance, and slip-risk metrics.
- starts the browser animation automatically.
- includes a static five-panel SVG storyboard for approach, probe, repose,
  lift, and carry.

The available strategies are:

- front carry;
- low carry;
- chest support;
- asymmetric carry;
- abort posture.

## Verification Performed On Login Node

Allowed lightweight checks:

```bash
node --check src/carrying_visualization/app.js
rg -n "app.js|styles.css|scene|toggleRun|robotHeight|boxMass" src/carrying_visualization
wc -l src/carrying_visualization/*
```

Result:

- JavaScript syntax check passed.
- HTML references to `styles.css`, `app.js`, canvas, controls, and key IDs are
  present.
- `diagnostic_storyboard.svg` exists as a static visual artifact. `xmllint`
  was not available on the login node, so XML validation was not performed.

## Compute-Node Visualization Verification Attempt

Rendering and screenshot generation were not run on `mgmtserver02`.

Attempted Curiosity-owned tmux + `srun` allocation flow:

```bash
tmux new-session -d -s curiosity_carry_viz \
  "cd /public/home/yanhongru/Curiosity && \
   srun -p gpu --gres=gpu:1 -N1 -n1 --time=00:20:00 --pty bash"
```

Result:

- Job `161801` queued on `gpu`.
- It did not allocate promptly and was later absent from `squeue`; no render
  command was run.

Second attempt:

```bash
tmux new-session -d -s curiosity_carry_viz_cpu \
  "cd /public/home/yanhongru/Curiosity && \
   srun -p cpu -N1 -n1 --time=00:15:00 --pty bash"
```

Result:

- Job `161803` queued on `cpu`.
- It did not allocate promptly and was cancelled.

Other partition checks:

```bash
srun -p test -N1 -n1 --time=00:10:00 --pty bash
srun -p gaosh -N1 -n1 --time=00:10:00 --pty bash
srun -p engram -N1 -n1 --time=00:10:00 --pty bash
```

Result:

- `test`, `gaosh`, and `engram` returned invalid account/partition
  combinations for this account.

No non-Curiosity tmux sessions, jobs, or resources were inspected or modified.

Follow-up attempt:

```bash
tmux new-session -d -s curiosity_carry_viz_verify \
  "cd /public/home/yanhongru/Curiosity && \
   srun -p cpu -N1 -n1 --cpus-per-task=1 --mem=2G \
     --time=00:20:00 --pty bash"
```

Result:

- Job `161805` queued on `cpu`.
- `squeue` reported:
  `Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions`.
- The job was cancelled.

GPU retry:

```bash
tmux new-session -d -s curiosity_carry_viz_verify_gpu \
  "cd /public/home/yanhongru/Curiosity && \
   srun -p gpu -N1 -n1 --cpus-per-task=1 --mem=2G \
     --time=00:20:00 --pty bash"
```

Result:

- Job `161806` queued, then briefly began running on `server10` at the same
  moment the cancellation command was sent. It was cleaned up; no render
  command was run.

Final GPU retry:

```bash
tmux new-session -d -s curiosity_carry_viz_verify_gpu2 \
  "cd /public/home/yanhongru/Curiosity && \
   srun -p gpu -N1 -n1 --cpus-per-task=1 --mem=2G \
     --time=00:20:00 --pty bash"
```

Result:

- Job `161807` queued on `gpu` with reason `Priority`.
- It did not allocate within the waiting window and was cancelled.

## Current Status

Completed:

- A concrete browser visualization artifact exists.
- A static SVG storyboard visualization exists.
- It visually models carrying posture adjustment logic once opened in a
  browser.
- Static syntax/reference checks passed.

Not completed for later, higher-fidelity work:

- No compute-node screenshot or MP4 was generated because no usable allocation
  was obtained in this turn.
- No Isaac Lab, MuJoCo, or physics simulator was run.
- No RL policy, real physics validation, or dataset/model loading occurred.

## Next Required Step

When a `cpu` or `gpu` allocation is available, run a headless browser screenshot
inside the allocation, for example:

```bash
mkdir -p experiments/visuals/carrying_visualization
firefox --headless \
  --window-size=1400,900 \
  --screenshot experiments/visuals/carrying_visualization/diagnostic.png \
  file:///public/home/yanhongru/Curiosity/src/carrying_visualization/index.html
```

The screenshot path is intentionally under `experiments/visuals/`, which is
ignored by git except for directory README files.
