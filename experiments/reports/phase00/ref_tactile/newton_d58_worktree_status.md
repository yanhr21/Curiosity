# Newton d58 Worktree Status

Date: 2026-07-01

This started as code preparation and now has compute-side evidence. It is still
not training, not curiosity success, and not Gate completion.

Machine-readable status:
`experiments/configs/phase00/ref_tactile/newton_d58_worktree_status_v1.json`

## Result

A separate latest-upstream Newton worktree now exists:

- path: `external/newton_d58`
- commit: `d58e70266be0db803261f3e46a2f7d923a43db37`
- upstream message: `Add multi-world soft contact filtering (#3118)`
- compute evidence: runtime benchmark, candidate tactile export, reference
  comparison, channel audit, and Gate review have run.

The existing active evidence worktree was preserved:

- path: `external/newton_main`
- commit: `a217e55fab3d373a08fba374cc5cafc1826cf27f`

The stable tag worktree remains:

- path: `external/newton_v1.3`
- commit: `ce11136b3a28390944f7fe5a32801b31d8aa5670`

## Commands Used

```bash
git -C external/newton fetch origin main
git -C external/newton worktree add --detach ../newton_d58 d58e70266be0db803261f3e46a2f7d923a43db37
```

## Compute Evidence

- benchmark: `p00_bench_d58_hot_v1_20260701_070611`, `82.7 FPS`
- tactile export: `p00_mjw_d58_marker_v1_20260701_071248`
- reference compare: `p00_refcmp_d58_marker_v1_20260701_071521`
- channel audit: `p00_chan_d58_marker_v1_20260701_071757`
- Gate review: `p00_gate_d58_marker_v1_20260701_071843`,
  `open_not_curiosity_ready`

The next faithful action is not more d58 runtime proof. It is resolving or
faithfully documenting the official UniVTAC/TaCauchy environment and asset
blockers required by Gate 00F.
