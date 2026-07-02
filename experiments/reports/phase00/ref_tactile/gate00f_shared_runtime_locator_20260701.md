# Gate 00F Shared Runtime Locator

- Date: `2026-07-01`
- Classification: `lightweight_shared_runtime_locator_not_training_not_gate_completion`

This was a lightweight locator for already-existing shared runtimes or
containers. It did not start containers, install dependencies, build packages,
run simulation, render, train, evaluate, load models, or inspect excluded
non-Curiosity resources.

## Scope

Checked common shared/container paths at top level or max depth 2:

- `/public/software`
- `/public/apps`
- `/public/share`
- `/public/container`
- `/public/containers`
- `/public/home/yanhongru/software`
- `/public/home/yanhongru/Software`
- `/public/home/yanhongru/containers`
- `/public/home/yanhongru/Images`
- `/opt`
- `/usr/local`

Existing checked paths were `/public/share`, `/opt`, and `/usr/local`.

## Result

No Isaac, Omniverse, TacEx, TaCauchy, UniVTAC, TacSL, UIPC, SIF, SQSH, or
container name hit was found in the checked shared paths.

The refreshed shell exposes `/usr/bin/docker`, but the read-only Docker image
name query found no Isaac/TacEx/TaCauchy/UniVTAC/TacSL/UIPC-related image.

A project-local maxdepth-5 artifact search for SIF/SQSH/TAR and Isaac/TacEx
YAML files was interrupted after it exceeded the intended lightweight window,
with no hits observed before interruption.

## Gate Effect

This does not clear Gate 00F. The next valid step remains one of the allowed
dependency resolution paths, followed by runtime preflight, Gate 00F reference
bundle execution, and strict bundle acceptance.
