# Latest Reference Recheck V3

Date: 2026-07-01

Classification: source acquisition update only. This is not official sanity,
not dense tactile export, not Gate completion, and not curiosity training.

## Source State

- Newton latest source worktree:
  `external/newton_8c501` at
  `8c501b47847569fecdda97a9f7f01205c6f7964f`. This remains source-only until
  H200 runtime sanity, dense tactile export, reference-video comparison,
  channel audit, and Gate review pass.
- TacEx:
  `external/TacEx` at `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`.
- IsaacLabTactile:
  `external/IsaacLabTactile` at
  `21bcb476b27ceedccccd63afef6bbd822adc2b2b`.

## IsaacLabTactile Acquisition

The previous v2 recheck recorded a fetch-pack disconnect. The official source
was acquired with:

```bash
GIT_LFS_SKIP_SMUDGE=1 timeout 1200 git clone --filter=blob:none --single-branch \
  https://github.com/UM-ARM-Lab/IsaacLabTactile.git external/IsaacLabTactile
```

The clone is about `60M`. `git-lfs` is not installed on the current PATH, so
LFS asset completeness is not verified. Treat the repository as source
evidence, not as a complete official runtime environment.

## Official URL Corrections

- Taccel: `https://github.com/Taccel-Simulator/Taccel`
- TaCauchy: `https://github.com/figsama/TaCauchy`
- HydroShear: `https://github.com/MMintLab/hydroshear`
- TacEx: `https://github.com/DH-Ng/TacEx`
- IsaacLabTactile: `https://github.com/UM-ARM-Lab/IsaacLabTactile`

Earlier failed probes of other namespaces are not official source evidence.

## Gate Effect

This update removes the IsaacLabTactile local source-acquisition blocker. It
does not remove the official dependency/sanity blocker. Gate 00F still depends
on official UniVTAC/TaCauchy sanity or a faithful accepted blocker.
