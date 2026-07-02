# Gate 00F Runtime Locator Probe

- Date: `2026-07-01`
- Classification: `lightweight_runtime_locator_probe_not_training_not_gate_completion`

This was a lightweight shell/path probe. It did not run training, simulation,
rendering, model loading, dataset conversion, package installation, package
builds, or recursive full-home searches.

## Result

No dependency-complete UniVTAC, TaCauchy, or IsaacLab TacSL runtime was found.
This does not clear Gate 00F.

## Command Availability

- `module`: missing
- `ml`: missing
- `singularity`: missing
- `apptainer`: missing
- `enroot`: missing
- `docker`: `/usr/bin/docker`
- `git-lfs`: missing
- `cmake`: missing
- `nvcc`: missing
- `nvidia-smi`: missing

## Project Env Prefixes Observed

- `envs/curiosity/conda/bin/python`: Python `3.12.6`
- `envs/newton/conda/bin/python`: Python `3.10.12`
- `envs/residual_adapter/conda/bin/python`: Python `3.10.12`
- `envs/tacauchy/conda/bin/python`: Python `3.11.15`
- `envs/taccel/conda/bin/python`: Python `3.10.12`
- `envs/trex/conda/bin/python`: Python `3.10.12`
- `envs/trex_dataset/conda/bin/python`: Python `3.10.12`
- `envs/univtac/conda/bin/python`: Python `3.10.20`

Only the UniVTAC and TaCauchy base Python prefixes match the intended
reference targets. They are still base envs, not dependency-complete official
reference runtimes.

## Gate Effect

Gate 00F remains blocked until an allowed resolution path provides one of:

- dependency-complete executable envs for UniVTAC, TaCauchy, and IsaacLab
  TacSL;
- already-built, approved shared-filesystem containers; or
- a compliant non-login/non-experiment env-prep workflow whose outputs are
  registered and then validated by the Gate 00F bundle plus strict acceptance
  checker.
