# UniVTAC Env Create Attempts

Date: 2026-07-01

After the user said `全都允许继续`, I attempted only local shared-filesystem
conda env creation for the UniVTAC reference env. No simulation, rendering,
training, model loading, dataset conversion, package import, or compute-node
dependency setup was run.

## Result

- target Python:
  `/public/home/yanhongru/Curiosity/envs/univtac/conda/bin/python`
- status: `present`
- version: `Python 3.10.20`
- env size: `140M`
- Gate effect: `gate00f_ready=false`,
  `reason=blocked_official_sanity_or_gate_review_not_passed`

## Attempts

- attempt 01: default solver/default cache
  - log:
    `logs/newton/phase00/ref_tactile/envprep/univtac/create_env_execute_20260701.log`
  - result: failed, no target Python
  - observed issue: `LockError: Failed to acquire lock`
- attempt 02: default solver/independent cache
  - log:
    `logs/newton/phase00/ref_tactile/envprep/univtac/create_env_execute_retry1_20260701.log`
  - result: failed, no target Python
  - observed issue: `LockError: Failed to acquire lock`
- attempt 03: classic solver/independent cache
  - log:
    `logs/newton/phase00/ref_tactile/envprep/univtac/create_env_execute_retry2_classic_20260701.log`
  - result: failed, no target Python
  - observed issue: `LockError: Failed to acquire lock`
- attempt 04: classic solver/independent cache with `--no-lock`
  - log:
    `logs/newton/phase00/ref_tactile/envprep/univtac/create_env_execute_retry3_no_lock_classic_20260701.log`
  - result: success, base Python env prefix present
  - observed issue: previous lock blocker bypassed with `--no-lock`

## Partial Artifacts

- `envs/conda_pkgs/univtac`: `137M`
- `envs/conda_pkgs/univtac_classic`: `170M`
- `envs/conda_pkgs/univtac_no_lock`: `194M`

The first two are partial package caches from failed attempts. The `no_lock`
cache supported the successful base env creation.

## Interpretation

Do not treat this as official UniVTAC sanity. The useful positive result is
narrower: the base Python prefix exists and the lock blocker has a concrete
workaround. The remaining Gate 00F blockers are official UniVTAC/TaCauchy
dependency readiness and official reference sanity.
