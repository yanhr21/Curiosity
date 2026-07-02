# Gate 00F Module/Env Probe

Date: 2026-07-01

Classification: lightweight login-node probe only. This did not install
dependencies, run Python imports, run official sanity, render, simulate, or
train.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/gate00f_module_env_probe_20260701_v1.json`

## Result

The current login shell does not expose `module` or `ml`, so module-based lookup
for `cmake`, `git-lfs`, CUDA, or Isaac cannot be used from this shell.

The shallow target-env file-name probe over `envs/univtac/conda` and
`envs/tacauchy/conda` found no Isaac, TacEx, UIPC, cuRobo, or Torch component
names at max depth 4.

## Gate Effect

This supports the current Gate 00F blocker: the base env prefixes exist, but
dependency-complete official UniVTAC/TaCauchy readiness is not proven and
official sanity has not passed.

This probe does not prove that no cluster-level module exists under every
possible shell initialization path. If an admin-provided module path or
different shell setup is supplied later, rerun a lightweight tool lookup before
any dependency work.
