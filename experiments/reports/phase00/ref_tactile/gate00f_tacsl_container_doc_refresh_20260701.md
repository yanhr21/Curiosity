# Gate 00F TacSL Container Documentation Refresh

- Date: `2026-07-01`
- Classification: `official_doc_refresh_not_runtime_not_gate_completion`

## Sources Checked

- NVIDIA NGC Isaac Lab container catalog:
  `https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab`
- Isaac Lab visuo-tactile sensor documentation:
  `https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html`
- Isaac Lab Docker guide:
  `https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html`
- Isaac Lab Docker example guide:
  `https://isaac-sim.github.io/IsaacLab/main/source/deployment/run_docker_example.html`
- IsaacLab TacSL RGB asset issue:
  `https://github.com/isaac-sim/IsaacLab/issues/4528`

## Findings

- The official Isaac Lab docs describe TacSL visuo-tactile sensing as providing
  camera-based tactile data and force-field tactile data.
- The official Isaac Lab Docker docs and NGC catalog support the container
  route for Isaac Lab runtime acquisition.
- The local source-compatibility evidence remains useful for the candidate
  `nvcr.io/nvidia/isaac-lab:2.3.2` route, because local IsaacLab source
  exposes TacSL fields and demo flags.
- A public IsaacLab issue reports that `tacsl_sensor.py` with
  `--use_tactile_rgb` can fail because a GelSight R15 `bg.jpg` background
  image is missing.
- Local static asset check found no `bg.jpg` or obvious GelSight R15 asset
  under `external/IsaacLab_official`.

## Gate Effect

This strengthens the IsaacLab TacSL route but also records a runtime risk:
even with a dependency-complete IsaacLab container, the `--use_tactile_rgb`
official sanity may fail unless the required TacSL/GelSight background asset
is present or the official upstream issue is resolved. Do not weaken Gate 00F
by dropping tactile RGB silently; record the asset/runtime failure if it
occurs.

This is documentation and static source evidence only. No container was
pulled, no image was run, no module was imported, no simulation ran, and Gate
00F remains open.
