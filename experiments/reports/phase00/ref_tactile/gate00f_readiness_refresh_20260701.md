# Gate 00F Readiness Refresh

Date: 2026-07-01

Classification: lightweight readiness refresh only. This is not official
UniVTAC sanity, not official TaCauchy sanity, not Gate completion, and not
training.

## Commands

```bash
bash experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh
bash experiments/configs/phase00/ref_tactile/envprep/check_gate00f_readiness.sh
```

## Result

Gate 00F remains open:

- `gate00f_ready`: `false`
- reason: `blocked_official_sanity_or_gate_review_not_passed`
- effective failed checks:
  `univtac_official_reference_sanity`,
  `tacauchy_official_reference_sanity`
- curiosity training allowed: `false`

Candidate base env pythons are present:

- `envs/univtac/conda/bin/python`
- `envs/tacauchy/conda/bin/python`

Assets are present after approved reuse:

- TaCauchy asset tree: `412M`
- TaCauchy GelSight Mini sensor USD: present
- TaCauchy tactile test shape USD count: `21`
- UniVTAC bundled TacEx asset tree: `410M`
- UniVTAC tactile test shape USD count: `21`

## Toolchain Gap

The checked PATH still lacks `git-lfs`, executable `cmake`, `nvcc`, and
`nvidia-smi`. Project-local conda and project-local CUDA `nvcc` exist, but
official dependency readiness and official sanity have not passed.

## IsaacLabTactile Note

`external/IsaacLabTactile` is cloned at
`21bcb476b27ceedccccd63afef6bbd822adc2b2b`, but `.gitattributes` marks assets
such as `*.usd`, `*.obj`, `*.mp4`, `*.pt`, and `*.hdf5` as LFS files. Since
`git-lfs` is not available on the current PATH and the clone used
`GIT_LFS_SKIP_SMUDGE=1`, treat this as source evidence only.

## Current Decision

Do not start curiosity training. Do not block `8c501...` dense tactile export
on FPS because its H200 runtime benchmark is acceptable around 80 FPS. Continue
from d58 as the strongest complete runtime/candidate tactile evidence chain
until 8c501 downstream tactile evidence exists, while resolving or faithfully
documenting the official UniVTAC/TaCauchy sanity blocker.
