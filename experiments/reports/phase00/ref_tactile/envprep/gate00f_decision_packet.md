# Gate 00F Decision Packet

Status: asset decision executed; base reference env prefixes created; official
dependency installation and sanity remain blocked. This
packet records the decision boundary for the current official tactile
semantic-validation blocker.

Current best positive evidence:
- d58 Gate review:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_d58_marker_v1_20260701_071843/phase00_gate_review_summary.json`
- result: `open_not_curiosity_ready`
- passed checks: 12
- failed checks: `reference_env_availability`,
  `reference_asset_availability`, `univtac_official_reference_sanity`,
  `tacauchy_official_reference_sanity`
- note: this failed-check list is from the pre-copy Gate review. Post-copy,
  TaCauchy asset file presence is cleared by approved local reuse, but a fresh
  Gate review has not consumed that evidence yet.

Toolchain recheck:
- `envs/taccel/cuda-toolkit/bin/nvcc` exists, CUDA `12.8`
- `envs/taccel/miniforge/bin/conda` exists, `conda 26.3.2`
- `git-lfs` is still missing
- executable `cmake` is still missing
- `envs/univtac/conda/bin/python` is present, `Python 3.10.20`
- `envs/tacauchy/conda/bin/python` is present, `Python 3.11.15`
- non-Curiosity OmniWorld/ICLR2027 env/resource hits were observed but not
  inspected or reused

Decision 1: TaCauchy assets
- resolved by approved local reuse after the user said `全都允许继续`
- execution report:
  `experiments/reports/phase00/ref_tactile/envprep/approved_asset_reuse_execution.md`
- caveat: this is not the official TaCauchy Git LFS asset setup; it is accepted
  local reuse from the official local UniVTAC bundled TacEx tree

Decision 2: official reference environments
- approved action completed: controlled local base env creation for
  `envs/univtac/conda` and `envs/tacauchy/conda`
- result: UniVTAC base env is `Python 3.10.20`; TaCauchy base env is
  `Python 3.11.15`
- note: the first UniVTAC attempts failed with `LockError`; retrying with
  `--no-lock --solver classic` succeeded
- evidence:
  `experiments/reports/phase00/ref_tactile/envprep/reference_env_create_execution.md`
- next: stage official dependency installation or document blockers, then run
  official sanity inside Curiosity tmux-held Slurm allocation

Still not allowed as a silent shortcut:
- heavy Isaac/TacEx/UIPC dependency installation without staged logs
- dependency installation on compute nodes
- using non-Curiosity OmniWorld/ICLR2027 resources
- claiming curiosity readiness from d58 candidate evidence alone

Success still requires: official dependency readiness, UniVTAC official sanity,
TaCauchy official sanity, and a fresh Gate review consuming the post-copy asset
and env evidence.
