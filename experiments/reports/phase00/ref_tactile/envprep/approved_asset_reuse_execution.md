# Approved Asset Reuse Execution

Status: executed after user approval on 2026-07-01. This is not official
TaCauchy sanity, not training, and not Gate 00F completion.

Source:
`external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data`

Target:
`external/TaCauchy/source/tacex_assets/tacex_assets/data`

Command:

```bash
rsync -a --ignore-existing --info=stats2 external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data/ external/TaCauchy/source/tacex_assets/tacex_assets/data/
```

Log:
`logs/newton/phase00/ref_tactile/envprep/assets/tacauchy/approved_univtac_asset_reuse_copy_20260701.log`

Observed result:
- created files: `273`
- regular files transferred: `234`
- transferred bytes: `429244198`
- target size after copy: `412M`
- TaCauchy `Sensors/GelSight_Mini/Sensor.usd`: present
- TaCauchy tactile test shape USD count: `21`

Remaining blockers:
target UniVTAC/TaCauchy reference environments are still missing, official
reference sanity has not passed, and a fresh Gate review has not yet consumed
the post-copy asset availability.
