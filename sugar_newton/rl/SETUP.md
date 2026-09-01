# Environment setup

One conda env, `robotbaby`, described by `env/environment.yml` at the repo root.

    bash env/setup_env.sh          # create or update; run on the LOGIN node (needs network)
    source env/activate.sh         # use it

`activate.sh` also puts `sugar_newton` and the in-repo `third_party/newton` submodule on
`PYTHONPATH`. Newton is deliberately **not** pip-installed: it is imported from the
submodule, so a `git clone --recurse-submodules` is runnable on its own and an edit to
Newton is live with no reinstall. That is what makes this branch self-contained -- nothing
resolves through a sibling checkout on the filesystem.

Everything is pinned to the versions the measurements in `README.md` were taken with, so
this is a port of a known-good environment rather than a fresh resolve.

## Two things the env cannot cover

**GL and Xvfb** are OS packages inside an ephemeral container root, so they are reinstalled
per container by `slurm/setup_container.sh` (which the dev-node holder job runs for you).
The .debs cache on Lustre, so only the first container pays for the download.

**The conda env must live on Lustre**, which it does by default, because the container root
is wiped on every `srun` while `/lustre` is mounted through.

## The two pinning traps, if you ever re-resolve

1. **`warp-lang` is only on NVIDIA's index** (`https://pypi.nvidia.com/`) and the pin is a
   nightly, `1.15.0.dev20260612`. If that build has been pruned, relax to
   `warp-lang>=1.15.0.dev0` and record what you actually get.
2. **`torch` must come from the cu128 index.** The default PyPI wheel is a different CUDA
   build and will not share a context with warp. After any change, confirm with
   `python -c "import torch; print(torch.__version__)"` that it still reads
   `2.11.0+cu128` and not a bare `2.11.0`.

## Running on a GPU

    sbatch slurm/devnode.sbatch                       # from the repo root
    bash slurm/devrun.sh "source env/activate.sh && python -m sugar_newton.validation.g1_carrybox_policy"

`slurm/devnode8.sbatch` is the 8-GPU variant for `sugar_newton/par_policy.sh`; target it
with `RB_STATE=slurm/.devnode8 bash slurm/devrun.sh ...`. If `interactive_singlenode`
refuses the job with `QOSMaxJobsPerUserLimit`, submit to `interactive` instead:
`sbatch -p interactive slurm/devnode.sbatch`.

For viewers that need a real GLX context rather than headless EGL, source
`slurm/render_env.sh` after `env/activate.sh`.

## History

This used to be a uv `.venv` inside a sibling `Curiosity_newton` checkout, and installing
the RL stack into it meant a `pip install --target` cross-install of cp312 wheels from the
login node, because that venv's interpreter lived inside the container and could not be run
from outside. The conda env removes the whole problem: `rsl-rl-lib`, `tensordict` and
`wandb` are ordinary pip entries in `environment.yml`.
