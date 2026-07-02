# Gate 00F Static Source Audit

Date: 2026-07-01

Classification: static source audit only. This is not compute execution, not
official sanity, not training, and not Gate 00F completion.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/gate00f_static_source_audit_20260701_v1.json`

## Purpose

This audit refines the current Gate 00F blocker from official source files
without running heavy work on the login node. The goal is to make the next
faithful step explicit: official UniVTAC and TaCauchy sanity must validate the
candidate Newton tactile channels before curiosity training can restart.

## UniVTAC Findings

UniVTAC is the visuo-tactile manipulation benchmark reference. Its README says
it is built on NVIDIA Isaac Lab and TacEx/UIPC, lists contact-rich tasks such as
Lift Bottle, Lift Can, insertion tasks, Pull Out Key, Put Bottle in Shelf, and
Grasp & Classify, and records ACT / ACT ablation / ViTAL baselines. It also
states that current data collection and evaluation are only supported on
GelSight Mini.

Relevant source paths:

- `external/UniVTAC/README.md`
- `external/UniVTAC/scripts/install.sh`
- `external/UniVTAC/encoder/dataloader.py`
- `external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py`
- `external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_taxim_fem.py`

The dataloader gives the schema we need to match for left/right tactile data:
`rgb_marker`, `marker`, `depth`, `rgb`, and `pose` under
`tactile/left_tactile/*` and `tactile/right_tactile/*`. The source normalizes
the first 63 marker positions by `[320, 240]` and normalizes depth with a
`24..34` window. This is useful for deciding the exact candidate-to-reference
mapping later, but it does not validate our candidate data by itself.

The install script is a heavy all-in-one path: conda env creation, Isaac Sim,
Isaac Lab, TacEx, cuRobo, and TacEx example execution. That cannot be run as an
untracked login-node shortcut or moved to compute-node dependency installation.

Gate effect: UniVTAC remains required for official left/right visual tactile
schema sanity, but Gate 00F is still blocked until a dependency-complete
approved environment exists and official sanity passes in a Curiosity
tmux-held Slurm allocation.

## TaCauchy Findings

TaCauchy is the mechanics-semantic reference. Its README describes direct
Cauchy stress extraction, contact traction recovery, adaptive mesh refinement,
and multi-sensor support. It explicitly targets normal pressure and tangential
traction, which are the missing high-value semantics for the Newton candidate
channels.

Relevant source paths:

- `external/TaCauchy/README.md`
- `external/TaCauchy/REPRODUCTION.md`
- `external/TaCauchy/ASSETS.md`
- `external/TaCauchy/scripts/demos/shape_touch/simple_tactile_demo.py`
- `external/TaCauchy/scripts/demos/shape_touch/contact_force_visualization_demo_modular.py`
- `external/TaCauchy/scripts/demos/shape_touch/benchmark_tactile_performance.py`
- `external/TaCauchy/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py`
- `external/TaCauchy/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_taxim_fem.py`

The reproduction guide requires Python 3.11, Isaac Sim 5.0.0, Isaac Lab
v2.2.1, GCC 11.4, CMake >= 3.26, vcpkg, and UIPC/libuipc build steps. The
asset guide records that large USD/calibration/texture assets are separate
from the source repository. After the approved local asset reuse, file presence
is better, but official sanity still has not run.

The GelSight Mini configs expose `tactile_rgb`, `marker_motion`, `height_map`,
`camera_depth`, and `camera_rgb` style outputs, with 320x240 tactile image
resolution in the optical simulator path. These are the reference semantics we
need before calling Newton marker/deformation panels more than candidate
force-derived visualizations.

Gate effect: TaCauchy remains the strongest official check for stress,
pressure, and tangential traction semantics. It is still blocked by dependency
readiness and official demo sanity, not by source availability alone.

## IsaacLabTactile Finding

`external/IsaacLabTactile` is cloned at
`21bcb476b27ceedccccd63afef6bbd822adc2b2b`, but the local clone currently
looks like generic Isaac Lab source. Static search found generic contact sensor
coverage, including net force readout paths, but no obvious local TacSL,
GelSight, Taxim, FOTS, or TacEx tactile demo entrypoint. The clone was also
acquired with LFS smudge skipped, and large USD/OBJ/MP4/PT/HDF5-style assets
are not verified while `git-lfs` is unavailable.

Gate effect: this source is not an adequate replacement for UniVTAC/TaCauchy
semantic validation. It remains a source gap/reference candidate until an
asset-complete tactile entrypoint and official sanity are identified.

## Current Gate 00F Conclusion

Gate 00F is still blocked by official semantic validation, not by lack of
candidate Newton visualization. The current Newton d58 chain is the strongest
candidate because it passed the runtime target at 82.7 FPS and has candidate
Fn/Ft/normal/area/marker-style evidence. The newer Newton 8c501 chain ran on
H200 and measured 80.1 and 80.8 FPS, which is acceptable for dense tactile
export under the current gate.

Remaining effective blockers:

- UniVTAC official dependency-complete environment is not proven.
- UniVTAC official sanity has not passed.
- TaCauchy official dependency-complete environment is not proven.
- TaCauchy official sanity has not passed.
- Heavy Isaac/TacEx/UIPC dependency work cannot run on the login node and
  cannot be moved to compute-node dependency installation under current rules.
- `git-lfs`, `cmake`, `nvcc`, and `nvidia-smi` are absent from the current PATH
  lookup; only project-local `envs/taccel/cuda-toolkit/bin/nvcc` was found.
- IsaacLabTactile is not yet an asset-complete tactile semantic replacement.

## Next Faithful Action

Keep d58 as the current strongest Newton candidate, then locate or prepare
approved dependency-complete UniVTAC and TaCauchy environments on shared
storage. Only after that should official reference sanity run inside a
Curiosity tmux-held Slurm allocation. Curiosity training remains disallowed.
