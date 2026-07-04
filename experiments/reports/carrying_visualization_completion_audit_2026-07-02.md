# Completion Audit: Carrying Visualization Step

Date: 2026-07-02.

## Objective

```text
Start trying, in a simulation platform, to make a robot that carries a box and
adjusts posture. Only finish this step. Visualization is required.
```

## Interpreted Requirement For This Step

The requested scope is an initial attempt, not a final research result. The
minimum acceptable artifact must show:

1. a simulated robot-like body;
2. a box to carry;
3. a sequence that includes approach, probing or preparation, lifting, and
   carrying;
4. posture adjustment in response to robot/body and box parameters;
5. a visible visualization artifact;
6. clear boundaries so it is not mistaken for Isaac/MuJoCo physics, RL, or a
   real robot result.

## Evidence

### Simulation Platform Artifact

- `src/carrying_visualization/index.html`
- `src/carrying_visualization/app.js`
- `src/carrying_visualization/styles.css`
- `src/carrying_visualization/simulation_manifest.json`

The platform is a browser-based kinematic simulation. It runs without a server
or package installation and starts the animation automatically.

### Visualization Artifact

- Interactive visualization entrypoint:
  `src/carrying_visualization/index.html`
- Static storyboard:
  `src/carrying_visualization/diagnostic_storyboard.svg`

The storyboard has five panels: approach, probe, repose, lift, and carry.

### Robot And Box

The HTML exposes robot controls:

- height;
- arm reach;
- torque capacity.

The HTML exposes box controls:

- mass;
- width;
- center-of-mass offset;
- friction.

### Posture Adjustment

`app.js` computes a deterministic posture plan from the robot and box
parameters. The visible outputs include:

- carry strategy;
- effort;
- balance;
- slip risk.

The animated posture changes include:

- stance width;
- torso lean;
- hold height;
- gait speed;
- chest support overlay for the chest-supported strategy.

The strategy set includes:

- front carry;
- low carry;
- chest support;
- asymmetric carry;
- abort posture.

### Verification

Lightweight login-node checks performed:

```bash
node --check src/carrying_visualization/app.js
find src/carrying_visualization -maxdepth 1 -type f -print | sort
rg -n "running: true|diagnostic_storyboard|robotHeight|boxMass|strategy" src/carrying_visualization
```

Observed:

- JavaScript syntax check passed.
- Required source files exist.
- The code contains the auto-run state, storyboard reference, robot controls,
  box controls, and strategy/planning logic.

## Requirement-by-Requirement Audit

- Simulated robot-like body: satisfied by canvas drawing in `drawRobot`.
- Box to carry: satisfied by `drawBox`.
- Approach/probe/lift/carry sequence: satisfied by `phases` and storyboard
  panels.
- Posture adjustment: satisfied by `planPosture` and animated stance, torso,
  hold-height, and gait-speed changes.
- Visualization required: satisfied by `index.html` and
  `diagnostic_storyboard.svg`.
- Boundary clarity: satisfied by README, manifest, and diagnostic report
  stating this is not physics-accurate, not RL, and not Isaac/MuJoCo.

## Known Limits

- No compute-node screenshot or MP4 was generated because the available Slurm
  allocations either queued or were unavailable for this account.
- No Isaac Lab or MuJoCo simulation was run.
- No RL policy was trained.
- The simulation is kinematic and diagnostic.

These limits do not invalidate this step because the requested step was to
start trying and produce a visualized robot carrying a box while adjusting
posture, not to complete a physics-accurate learning result.

