# Reference Env Create Execution

Date: 2026-07-01

After the user said `全都允许继续`, I created only the base local conda env
prefixes needed for Gate 00F availability. This did not install UniVTAC,
TaCauchy, Isaac Sim/Lab, TacEx, UIPC, or project dependencies, and it did not
run official sanity.

## Results

- UniVTAC base env:
  `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`
  - version: `Python 3.10.20`
  - size: `140M`
  - successful command: `conda create --no-lock --solver classic ... python=3.10`
  - log:
    `logs/newton/phase00/ref_tactile/envprep/univtac/create_env_execute_retry3_no_lock_classic_20260701.log`
- TaCauchy base env:
  `/public/home/yanhongru/Curiosity/envs/tacauchy/conda/bin/python`
  - version: `Python 3.11.15`
  - size: `166M`
  - successful command: `conda create --no-lock --solver classic ... python=3.11`
  - log:
    `logs/newton/phase00/ref_tactile/envprep/tacauchy/create_env_execute_no_lock_classic_20260701.log`

## Gate Effect

`check_reference_env_availability.sh` now reports
`gate_00f_ready=candidate_envs_present_pending_compute_sanity`.

`check_gate00f_readiness.sh` still reports `gate00f_ready=false`, now with
`reason=blocked_official_sanity_or_gate_review_not_passed`. The effective
remaining failed checks are `univtac_official_reference_sanity` and
`tacauchy_official_reference_sanity`.

This is env availability progress only, not official reference sanity and not
curiosity readiness.
