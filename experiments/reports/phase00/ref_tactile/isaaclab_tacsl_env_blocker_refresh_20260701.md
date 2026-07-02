# IsaacLab TacSL Env Blocker Refresh

- created_at: `2026-07-01 17:21:41 CST`
- classification: `blocker_refresh_not_training_not_sanity_success`
- target: `official_isaaclab_tacsl_gate00f_sanity`
- curiosity_training_allowed: `false`

## Findings

- Running Slurm job `160860` is not usable for Curiosity. `scontrol show job
  160860` reports `WorkDir=/public/home/yanhongru/ICLR2027/Reflex`, so the
  project resource-exclusion rule forbids reusing or attaching to it.
- No executable IsaacLab/TacSL env prefix was found at
  `envs/isaaclab_tacsl`, `envs/isaaclab_tacsl/conda`, or
  `envs/isaaclab_tacsl/.venv`.
- Existing reference env hits are only `envs/univtac`, `envs/tacauchy`, and
  package-cache directories under `envs/conda_pkgs/`.
- A limited Curiosity-local maxdepth-4 search found no TacSL/Isaac/TacEx/
  TaCauchy/UniVTAC `.sif`, `.sqsh`, `.tar`, or `.tar.gz` prebuilt container
  archive.

## Blocker

Official IsaacLab TacSL sanity cannot be run faithfully right now. There is no
usable Curiosity-owned compute allocation for this work, and no
dependency-complete IsaacLab/TacSL env or project-local prebuilt container was
found. The currently running Slurm job is Reflex-owned and must not be reused.

## Next Valid Actions

- Provide or locate a Curiosity-owned dependency-complete IsaacLab/TacSL env.
- Provide or locate a Curiosity-owned prebuilt container archive.
- Start a new Curiosity tmux-held Slurm allocation only after an env or
  container can make the sanity meaningful.
- Then run `launch_isaaclab_tacsl_sanity_tmux.sh` and feed the resulting
  `ISAACLAB_TACSL_SANITY_SUMMARY` into Gate 00F review.
