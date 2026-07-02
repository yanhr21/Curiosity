# Reference Asset Reuse Plan

Date: 2026-07-01

This was a candidate plan. It has now been executed after explicit user
approval on 2026-07-01. No official Git LFS download was run.

Machine-readable plan:
`experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json`

## Observation

The local UniVTAC bundled TacEx asset tree is much larger and more complete
than the current TaCauchy asset tree:

- UniVTAC bundled TacEx data:
  `external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data`
  - observed size: `410M`
- TaCauchy data:
  `external/TaCauchy/source/tacex_assets/tacex_assets/data`
  - observed size: `1.8M`

UniVTAC bundled TacEx includes GelSight Mini, GF225, Franka/GF225,
Franka/GelSight_Mini, and tactile test shape assets. TaCauchy currently has
partial placeholder assets only.

## Executed Reuse Path

The approved local asset stage copied missing assets from the UniVTAC bundled
TacEx tree into TaCauchy with `--ignore-existing`:

```bash
rsync -a --ignore-existing --info=stats2 \
  external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data/ \
  external/TaCauchy/source/tacex_assets/tacex_assets/data/
```

Execution log:
`logs/newton/phase00/ref_tactile/envprep/assets/tacauchy/approved_univtac_asset_reuse_copy_20260701.log`

Observed result:
- created files: `273`
- regular files transferred: `234`
- transferred bytes: `429244198`
- target size after copy: `412M`
- TaCauchy `Sensors/GelSight_Mini/Sensor.usd`: present
- TaCauchy tactile test shape USD count: `21`

## Risks

- UniVTAC uses a modified bundled TacEx; assets may not exactly match the
  TaCauchy upstream asset expectation.
- TaCauchy references DIGIT and 9DTact assets that are not fully covered by
  the observed UniVTAC bundled set.
- Asset reuse would not solve the missing TaCauchy environment, UIPC/libuipc
  build, or official compute-side sanity.

## Remaining Evidence Required

- target UniVTAC/TaCauchy environments must exist;
- official UniVTAC/TaCauchy sanity must pass inside Curiosity tmux-held Slurm;
- fresh Gate review must consume the post-copy asset availability;
- asset reuse must continue to be labeled as approved local reuse, not official
  TaCauchy Git LFS asset setup.
