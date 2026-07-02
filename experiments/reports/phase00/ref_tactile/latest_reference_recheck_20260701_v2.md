# Latest Reference Recheck 20260701 V2

Date: 2026-07-01

Scope: web search, official source-page review, `git ls-remote`, and
lightweight source acquisition only. No simulation, rendering, training,
dependency installation, model loading, dataset conversion, official sanity, or
Slurm work was run.

## Source Updates

- Newton official repo: `https://github.com/newton-physics/newton`
  - remote `main`: `8c501b47847569fecdda97a9f7f01205c6f7964f`
  - new local worktree: `external/newton_8c501`
  - status: latest code path prepared, not compute-sanity checked
- Taccel official remote: `https://github.com/Taccel-Simulator/Taccel.git`
  - remote/local `main`: `cb23bc251b531ba6908a3788c2f91423cd543149`
- T-Rex official remote: `https://github.com/ZhuoyangLiu2005/T-Rex.git`
  - remote `main`: `43ff632259d76f08373c085c53111825060d029b`
  - local `external/T-Rex`: `db7a02992504ad9be53a7e764f7b05d81d86c767`
  - status: local dirty/behind; do not overwrite silently
- HydroShear official/local remote: `https://github.com/MMintLab/hydroshear.git`
  - remote/local `master`: `a53a51cb74f0608ca53839415d7f1964a99f1db0`
- UniVTAC official remote: `https://github.com/univtac/UniVTAC.git`
  - remote/local `main`: `05bcd3edb92237107efa40105292a24f1a9fd761`
- TaCauchy official remote: `https://github.com/figsama/TaCauchy.git`
  - remote/local `main`: `c228cfe9050904cd5d71d64f6eb5104768d4cbda`
- TacEx official remote: `https://github.com/DH-Ng/TacEx.git`
  - remote/local `main`: `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`
  - new local clone: `external/TacEx`
  - status: cloned, not installed, not sanity checked
- IsaacLabTactile official remote: `https://github.com/UM-ARM-Lab/IsaacLabTactile.git`
  - remote `main`: `21bcb476b27ceedccccd63afef6bbd822adc2b2b`
  - acquisition status: clone failed with
    `fetch-pack: unexpected disconnect while reading sideband packet`; partial
    directory was removed/left absent from active source use

## Web Findings

- Newton remains the primary physics path: the official README describes it as
  GPU-accelerated, built on NVIDIA Warp, with robot, contact, sensor, hydro,
  and USD-oriented examples.
- UniVTAC remains the mandatory visuo-tactile manipulation benchmark path: its
  README describes NVIDIA Isaac Lab plus TacEx/UIPC-based tactile simulation,
  expert-demonstration collection, policy training, and contact-rich task
  evaluation with GelSight Mini/GF225/XenseWS style sensors.
- T-Rex remains the model/reference architecture path: the project page
  describes a 100-hour tactile-synchronized dataset, temporal tactile VQ-VAE,
  variable-rate Mixture-of-Transformer, and high-frequency tactile residual
  refinement.
- Isaac Lab TacSL remains a comparison/reference path: the official docs list
  tactile RGB, tactile depth, normal force, and shear force observations, plus
  stiffness/friction configuration.
- TacEx remains the official tactile extension path: its README states GelSight
  Mini support, Taxim/FOTS/UIPC components, Isaac Sim 4.5 and IsaacLab 2.1.1
  compatibility, and git-lfs requirements.

## Current Decision

Do not replace the current d58 evidence with `external/newton_8c501` until it
passes the same tmux-held Slurm benchmark, dense tactile export, reference
comparison, channel audit, and Gate review sequence. Do not claim Gate 00F
completion from source acquisition alone.

Remaining effective Gate 00F failures:

- `univtac_official_reference_sanity`
- `tacauchy_official_reference_sanity`
