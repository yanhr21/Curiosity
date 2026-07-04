# Carrying Visualization Diagnostic

This is a browser-only diagnostic visualization for the active carrying
direction. It is a small kinematic simulation platform, not a
physics-accurate simulator, not RL, and not a success claim. It exists to make
the first target behavior visible:

```text
walk to box -> probe unknown load -> choose body-specific posture
-> lift -> carry while adjusting stance, torso, hold height, and gait speed
```

Open `index.html` in a browser. No server or package installation is required.
The animation starts automatically. `diagnostic_storyboard.svg` is a static
five-panel visual summary of the same diagnostic behavior.
`simulation_manifest.json` records the supported phases, controls, outputs,
and non-claims.

The visualization exposes:

- robot morphology controls: height, arm reach, torque capacity;
- box controls: mass, width, center-of-mass offset, friction;
- a deterministic pose planner that selects front, low, chest-supported, or
  asymmetric carry strategy;
- visible metrics for effort, balance margin, slip risk, and selected
  strategy.

Cluster rule reminder: this file can be opened by the user locally, but no
simulation or rendering was executed on the login node.
